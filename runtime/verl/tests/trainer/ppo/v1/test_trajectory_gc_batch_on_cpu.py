# Copyright 2026 Project B contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only storage garbage-collection tests using the real file stores."""

import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import pytest


def load_trajectory_state():
    """Load only the pure storage package, avoiding the full VeRL import."""
    package_root = Path(__file__).parents[4] / "verl/trainer/ppo/v1/trajectory_state"
    name = "project_b_trajectory_gc_test_isolated"
    spec = importlib.util.spec_from_file_location(
        name, package_root / "__init__.py", submodule_search_locations=[str(package_root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trajectory_state = load_trajectory_state()
CanonicalPolicyKVIdentity = trajectory_state.CanonicalPolicyKVIdentity
DurableTrajectoryCoordinator = trajectory_state.DurableTrajectoryCoordinator
HandoffManifest = trajectory_state.HandoffManifest
FileObjectStore = trajectory_state.storage.FileObjectStore


IDENTITY = asdict(CanonicalPolicyKVIdentity("1", "weights-1", "template-1", "kv-abi-1"))


class RecordingObjectStore(FileObjectStore):
    def __init__(self, root, *, fail_delete_key=None):
        super().__init__(root)
        self.deleted_keys = []
        self.fail_delete_key = fail_delete_key

    def delete(self, key):
        self.deleted_keys.append(key)
        if key == self.fail_delete_key:
            self.fail_delete_key = None
            raise OSError("injected object delete failure")
        super().delete(key)


class MetadataRecorder:
    def __init__(self):
        self.stages = []
        self.fail_before_file_fsync = False

    def __call__(self, stage):
        self.stages.append(stage)
        if stage == "before_file_fsync" and self.fail_before_file_fsync:
            self.fail_before_file_fsync = False
            raise OSError("injected metadata save failure")

    def save_count(self):
        return self.stages.count("before_file_fsync")


def make_coordinator(root, *, object_store=None):
    recorder = MetadataRecorder()
    store = object_store if object_store is not None else RecordingObjectStore(root / "objects")
    coordinator = DurableTrajectoryCoordinator(str(root), "run-1", object_store=store, fault_hook=recorder)
    recorder.stages.clear()
    return coordinator, store, recorder


def put_object(coordinator, payload):
    return coordinator._transaction(lambda draft: coordinator._put_object(draft, "inflight", payload))


def queue_for_cleanup(coordinator, references):
    coordinator._transaction(lambda draft: draft["gc_pending"].extend(references))


def test_batch_cleanup_uses_one_metadata_save_and_counts_distinct_references(tmp_path):
    coordinator, store, recorder = make_coordinator(tmp_path)
    references = [put_object(coordinator, bytes([index])) for index in range(3)]
    queue_for_cleanup(coordinator, [*references, references[0]])
    recorder.stages.clear()

    assert coordinator.collect_garbage() == 3
    assert recorder.save_count() == 1
    assert len(store.deleted_keys) == 3 and len(set(store.deleted_keys)) == 3
    assert coordinator._state["objects"] == {}
    assert coordinator._state["gc_pending"] == []


def test_fully_queued_shared_key_is_deleted_once(tmp_path):
    coordinator, store, _ = make_coordinator(tmp_path)
    first = put_object(coordinator, b"shared")
    second = put_object(coordinator, b"shared")
    queue_for_cleanup(coordinator, [first, second])
    storage_key = coordinator._state["objects"][first]["storage_key"]

    assert coordinator.collect_garbage() == 2
    assert store.deleted_keys == [storage_key]
    assert not store._path(storage_key).exists()
    assert coordinator._state["objects"] == {}


def test_partially_queued_shared_key_and_unqueued_object_are_not_deleted(tmp_path):
    coordinator, store, _ = make_coordinator(tmp_path)
    queued = put_object(coordinator, b"shared")
    retained_alias = put_object(coordinator, b"shared")
    untouched = put_object(coordinator, b"untouched")
    queue_for_cleanup(coordinator, [queued])
    storage_key = coordinator._state["objects"][queued]["storage_key"]

    assert coordinator.collect_garbage() == 1
    assert store.deleted_keys == []
    assert store._path(storage_key).exists()
    assert coordinator._state["objects"] == {
        retained_alias: {
            "sha256": coordinator._state["objects"][retained_alias]["sha256"],
            "bytes": 6,
            "storage_key": storage_key,
        },
        untouched: coordinator._state["objects"][untouched],
    }


def test_pinned_alias_keeps_physical_object_but_removes_queued_alias(tmp_path):
    coordinator, store, recorder = make_coordinator(tmp_path)
    pinned = put_object(coordinator, b"pinned-shared")
    queued = put_object(coordinator, b"pinned-shared")
    manifest = HandoffManifest(
        handoff_id="pin-1",
        source_rollouter="source",
        source_generation=0,
        policy_identity=IDENTITY,
        environment_version="env-1",
        tool_version="tool-1",
        completed_artifacts=(),
        checkpointed_trajectories=(
            {
                "trajectory_id": "group-1#0",
                "group_id": "group-1",
                "member_index": 0,
                "group_size": 2,
                "expected_generation": 0,
                "snapshot_ref": pinned,
                "input_ref": pinned,
            },
        ),
        completed_members=(),
        pending_samples=(
            {
                "trajectory_id": "group-1#1",
                "group_id": "group-1",
                "member_index": 1,
                "group_size": 2,
                "record_ref": pinned,
            },
        ),
    )
    # This private fixture models only the unfinished handoff that pins the
    # first alias; it intentionally does not fabricate queue publication.
    coordinator._state["handoffs"]["pin-1"] = {
        "manifest": asdict(manifest),
        "state": "COMMITTED",
        "prepared_at": 0,
        "committed_at": 0,
        "source_released_at": None,
        "restoring_started_at": None,
        "finished_at": None,
        "restored_member_refs": {},
        "restored_member_metrics": {},
        "published_artifact_id": None,
        "publication": None,
    }
    queue_for_cleanup(coordinator, [queued])
    recorder.stages.clear()
    storage_key = coordinator._state["objects"][pinned]["storage_key"]

    assert coordinator.collect_garbage() == 1
    assert store.deleted_keys == []
    assert store._path(storage_key).exists()
    assert pinned in coordinator._state["objects"]
    assert queued not in coordinator._state["objects"]
    assert coordinator._state["gc_pending"] == []
    assert recorder.save_count() == 1


def test_object_delete_failure_preserves_retry_list_and_succeeds_after_retry(tmp_path):
    coordinator, store, _ = make_coordinator(tmp_path)
    references = [put_object(coordinator, b"one"), put_object(coordinator, b"two")]
    queue_for_cleanup(coordinator, references)
    store.fail_delete_key = coordinator._state["objects"][references[0]]["storage_key"]

    with pytest.raises(OSError, match="injected object delete failure"):
        coordinator.collect_garbage()
    assert coordinator._state["gc_pending"] == references
    assert set(coordinator._state["objects"]) == set(references)

    assert coordinator.collect_garbage() == 2
    assert coordinator._state["gc_pending"] == []
    assert coordinator._state["objects"] == {}


def test_metadata_failure_after_deletion_survives_restart_without_reading_deleted_data(tmp_path):
    root = tmp_path / "run-1"
    object_store = RecordingObjectStore(root / "objects")
    coordinator, _, recorder = make_coordinator(tmp_path, object_store=object_store)
    references = [put_object(coordinator, b"one"), put_object(coordinator, b"two")]
    queue_for_cleanup(coordinator, references)
    recorder.fail_before_file_fsync = True
    storage_keys = [coordinator._state["objects"][reference]["storage_key"] for reference in references]

    with pytest.raises(OSError, match="injected metadata save failure"):
        coordinator.collect_garbage()
    assert coordinator._state["gc_pending"] == references
    assert all(not object_store._path(storage_key).exists() for storage_key in storage_keys)
    coordinator.close()

    reopened, reopened_store, _ = make_coordinator(tmp_path, object_store=FileObjectStore(root / "objects"))
    assert reopened.collect_garbage() == 2
    assert reopened._state["gc_pending"] == []
    assert reopened._state["objects"] == {}
    assert all(not reopened_store._path(descriptor["storage_key"]).exists() for descriptor in ())
