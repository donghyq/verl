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

"""CPU-only coordinator fault boundaries; no model or optimizer execution."""

import ast
import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

from verl.trainer.ppo.v1 import trajectory_state
from verl.trainer.ppo.v1.trajectory_state import (
    CanonicalPolicyKVIdentity,
    CoordinationStorageError,
    DurableTrajectoryCoordinator,
    HandoffManifest,
    PublicationBlocked,
    TrajectorySnapshot,
    decode_handoff_member_record,
    encode_handoff_member_record,
)
from verl.trainer.ppo.v1.trajectory_state.snapshot import canonical_bytes


IDENTITY = asdict(CanonicalPolicyKVIdentity("1", "weights-1", "template-1", "kv-abi-1"))


class Clock:
    def __init__(self, monotonic=100.0, wall=1_700_000_000.0):
        self.monotonic, self.wall = monotonic, wall

    def now(self):
        return self.monotonic

    def unix(self):
        return self.wall


def snapshot(trajectory_id, member_index, *, generation=0):
    return TrajectorySnapshot(
        trajectory_id=trajectory_id, group_id="group-1", member_index=member_index, group_size=8,
        execution_generation=generation,
        agent_state={
            "messages": [{"role": "user", "content": "search dinner"},
                         {"role": "assistant", "content": "search"},
                         {"role": "tool", "content": "found dinner"}],
            "prompt_ids": [1, 2, 3, 4, 5], "response_ids": [3],
            "response_mask": [1, 0, 0], "response_logprobs": [-0.7, 0.0, 0.0],
            "turn_scores": [], "tool_rewards": [], "user_turns": 1, "assistant_turns": 1,
            "extra_fields": {"trajectory_segments": [{
                "start": 0, "end": 1, "policy_version": "1", "execution_generation": generation,
                "sampling_config_fingerprint": "sample-1",
            }]},
            "tools_kwargs": {}, "metrics": {}, "request_id": trajectory_id, "mm_processor_kwargs": {},
        },
        sampling_params={"temperature": 0.8}, limits={"response_length": 128, "max_turns": 4},
        environment_version="env-1", tool_version="tool-1", policy_identity=IDENTITY,
        tool_selection=["search"],
        completed_tool_batches=[{"calls": [{"name": "search", "arguments": {}}], "results": ["found dinner"]}],
    )


@dataclass
class Stack:
    coordinator: DurableTrajectoryCoordinator
    root: Path
    clock: Clock
    leases: dict
    manifest: HandoffManifest
    input_refs: dict


@pytest.fixture
def make_stack(tmp_path):
    opened = []

    def factory(*, committed=True, released=True):
        root = tmp_path / ("state-" + str(len(opened)))
        clock = Clock()
        coordinator = DurableTrajectoryCoordinator(str(root), "run-1", clock=clock.now, wall_clock=clock.unix)
        opened.append(coordinator)
        generation = coordinator.register_producer("source", "incarnation-a")
        inputs, completed, leases, checkpointed = {}, {}, {}, []
        for i in range(8):
            inputs[i] = coordinator.put_handoff_member_record(canonical_bytes({
                "uid": "group-1", "session_id": i, "original_prompt": [1, 2], "seed": i,
            }))
        for i in (0, 1):
            completed[i] = coordinator.put_handoff_member_record(canonical_bytes({"member_index": i, "tokens": [9]}))
        for i in (2, 3, 4, 5):
            trajectory_id = f"group-1#{i}"
            lease = coordinator.open_trajectory(trajectory_id, "source", 600.0)
            leases[i] = lease
            coordinator.register_running_trajectory(trajectory_id)
            coordinator.request_trajectory_preemption(trajectory_id)
            coordinator.mark_trajectory_safe_point(trajectory_id)
            reference = coordinator.commit_trajectory_snapshot(lease, snapshot(trajectory_id, i), 3600.0)
            checkpointed.append({
                "trajectory_id": trajectory_id, "group_id": "group-1", "member_index": i, "group_size": 8,
                "expected_generation": lease.generation, "snapshot_ref": reference, "input_ref": inputs[i],
            })
        manifest = HandoffManifest(
            handoff_id="handoff-1", source_rollouter="source", source_generation=generation,
            policy_identity=IDENTITY, environment_version="env-1", tool_version="tool-1",
            completed_artifacts=(), checkpointed_trajectories=tuple(checkpointed),
            completed_members=tuple({
                "trajectory_id": f"group-1#{i}", "group_id": "group-1", "member_index": i,
                "group_size": 8, "record_ref": completed[i],
            } for i in (0, 1)),
            pending_samples=tuple({
                "trajectory_id": f"group-1#{i}", "group_id": "group-1", "member_index": i,
                "group_size": 8, "record_ref": inputs[i],
            } for i in (6, 7)),
        )
        coordinator.prepare_handoff_manifest(manifest)
        if committed:
            coordinator.commit_handoff_manifest(manifest.handoff_id)
            if released:
                coordinator.release_handoff_source(manifest.handoff_id)
        return Stack(coordinator, root, clock, leases, manifest, inputs)

    yield factory
    for coordinator in opened:
        coordinator.close()


def reopen(stack, *, monotonic=1.0):
    stack.coordinator.close()
    stack.clock.monotonic = monotonic  # different monotonic epoch on the replacement host
    stack.coordinator = DurableTrajectoryCoordinator(
        str(stack.root), "run-1", clock=stack.clock.now, wall_clock=stack.clock.unix
    )
    return stack.coordinator


def complete_target_members(stack):
    coordinator = stack.coordinator
    coordinator.begin_handoff_restore("handoff-1")
    saved = coordinator.get_handoff_restored_members("handoff-1")
    for member in stack.manifest.checkpointed_trajectories + stack.manifest.pending_samples:
        i = member["member_index"]
        if i in saved:
            continue
        acquire = (coordinator.acquire_handoff_trajectory if i in (2, 3, 4, 5)
                   else coordinator.acquire_handoff_pending)
        owner = acquire("handoff-1", member["trajectory_id"], "target", 600.0)
        coordinator.record_handoff_member_result(
            "handoff-1", i, owner, canonical_bytes({"member_index": i, "tokens": [3, 7]}),
            {"tool_replay_count": 0, "resume_input_tokens": 5},
        )


def binding():
    return {"partition_id": "train", "keys": [f"group-1_{i}_0" for i in range(8)],
            "content_digest": hashlib.sha256(b"frozen actual training fields").hexdigest()}


def receipt(publication):
    # An adapter's readback receipt fixture. Actual TQ field verification has
    # separate integration tests; this does not pretend to be a queue GET.
    return {"publication_id": publication["publication_id"], "binding": publication["binding"],
            "backend_reference": {"partition_id": "train", "actual_indices": list(range(8))}}


def test_migrated_protocol_symbols_match_pinned_source_hashes():
    package = Path(trajectory_state.__file__).parent
    provenance = json.loads((package / "source_provenance.json").read_text())
    assert provenance["source_commit"] == "f91c201d52665b835a92e119b013f36ec9a49c58"
    for symbol in provenance["symbols"]:
        source = (package / symbol["destination_file"]).read_text()
        tree, lines = ast.parse(source), source.splitlines(keepends=True)
        parts = symbol["destination_symbol"].split(".")
        container = tree
        for name in parts:
            node = next(n for n in container.body if getattr(n, "name", None) == name or
                        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
            container = node
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        actual = "".join(lines[first - 1:node.end_lineno])
        assert hashlib.sha256(actual.encode()).hexdigest() == symbol["destination_sha256"], parts


def test_snapshot_canonical_bytes_and_optional_kv_field_remain_unchanged():
    original = snapshot("group-1#2", 2)
    expected = json.dumps(asdict(original), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert original.encode() == expected
    assert TrajectorySnapshot.decode(expected) == original
    assert json.loads(expected)["kv_ref"] is None
    with pytest.raises(ValueError, match="KV reference"):
        replace(original, kv_ref={"export_complete": True}).encode()


def test_private_member_codec_preserves_original_full_output():
    value = {"tokens": [1, 2], "logprobs": [-0.4, -0.8], "tools": [{"result": "already committed"}]}
    assert decode_handoff_member_record(encode_handoff_member_record(value)) == value
    assert decode_handoff_member_record(canonical_bytes(value)) == value


def test_restart_restores_n8_partition_and_original_bytes_but_not_old_owner(make_stack):
    stack = make_stack()
    before = stack.manifest.encode()
    coordinator = reopen(stack, monotonic=-10_000.0)
    try:
        assert coordinator.get_handoff_manifest("handoff-1").encode() == before
        assert coordinator.get_current_producer_generation("source") == 0
        with pytest.raises(ValueError, match="stale or expired"):
            coordinator.check_trajectory_owner(stack.leases[2])
        coordinator.begin_handoff_restore("handoff-1")
        target = coordinator.acquire_handoff_trajectory("handoff-1", "group-1#2", "target", 600.0)
        assert target.generation == 1
        restored = coordinator.load_trajectory_snapshot(target)
        assert restored.encode() == snapshot("group-1#2", 2).encode()
        assert restored.completed_tool_batches[0]["results"] == ["found dinner"]
        assert coordinator.summary()["active_execution_leases"] == 1
    finally:
        coordinator.close()


def test_target_generation_keeps_increasing_after_repeated_coordinator_failure(make_stack):
    stack = make_stack()
    coordinator = stack.coordinator
    coordinator.begin_handoff_restore("handoff-1")
    first = coordinator.acquire_handoff_trajectory("handoff-1", "group-1#2", "target-a", 600.0)
    second_coordinator = reopen(stack)
    try:
        with pytest.raises(ValueError, match="stale or expired"):
            second_coordinator.renew_trajectory_owner(first, 600.0)
        second = second_coordinator.acquire_handoff_trajectory("handoff-1", "group-1#2", "target-b", 600.0)
        assert (stack.leases[2].generation, first.generation, second.generation) == (0, 1, 2)
        with pytest.raises(ValueError, match="stale or expired"):
            second_coordinator.commit_trajectory_snapshot(first, snapshot("group-1#2", 2, generation=1), 600.0)
    finally:
        second_coordinator.close()


def test_source_release_is_required_before_live_target_takeover(make_stack):
    stack = make_stack(released=False)
    coordinator = stack.coordinator
    with pytest.raises(ValueError, match="leases must be released"):
        coordinator.record_handoff_source_release("handoff-1")
    coordinator.begin_handoff_restore("handoff-1")
    with pytest.raises(ValueError, match="takeover requires"):
        coordinator.acquire_handoff_trajectory("handoff-1", "group-1#2", "target", 600.0)


def test_manifest_cannot_commit_missing_snapshot_or_changed_source_generation(make_stack):
    stack = make_stack(committed=False)
    stack.coordinator.register_producer("source", "incarnation-b")
    with pytest.raises(ValueError, match="generation changed"):
        stack.coordinator.commit_handoff_manifest("handoff-1")


def test_snapshot_committed_marker_is_idempotent_but_cannot_invent_a_pointer(make_stack):
    stack = make_stack(committed=False)
    ref = stack.manifest.checkpointed_trajectories[0]["snapshot_ref"]
    assert stack.coordinator.mark_trajectory_snapshot_committed("group-1#2", ref) == "snapshot_committed"
    with pytest.raises(ValueError, match="durable snapshot"):
        stack.coordinator.mark_trajectory_snapshot_committed("group-1#2", "inflight/invented")


def test_completed_target_members_survive_restart_without_reexecution(make_stack):
    stack = make_stack()
    coordinator = stack.coordinator
    coordinator.begin_handoff_restore("handoff-1")
    lease = coordinator.acquire_handoff_trajectory("handoff-1", "group-1#2", "target", 600.0)
    payload = canonical_bytes({"tokens": [3, 6], "tool_results": ["already committed"]})
    ref = coordinator.record_handoff_member_result("handoff-1", 2, lease, payload, {"tool_replay_count": 0})
    coordinator = reopen(stack)
    try:
        assert coordinator.get_handoff_restored_members("handoff-1")[2]["record_ref"] == ref
        assert coordinator.get_handoff_member_record(ref) == payload
        with pytest.raises(ValueError, match="already restored"):
            coordinator.acquire_handoff_trajectory("handoff-1", "group-1#2", "replacement", 600.0)
        complete_target_members(stack)
        assert set(coordinator.get_handoff_restored_members("handoff-1")) == {2, 3, 4, 5, 6, 7}
    finally:
        coordinator.close()


def test_live_member_result_cannot_be_rewritten(make_stack):
    stack = make_stack()
    c = stack.coordinator
    c.begin_handoff_restore("handoff-1")
    owner = c.acquire_handoff_trajectory("handoff-1", "group-1#2", "target", 600.0)
    payload = canonical_bytes({"tokens": [3, 7]})
    original = c.record_handoff_member_result("handoff-1", 2, owner, payload, {})
    assert c.record_handoff_member_result("handoff-1", 2, owner, payload, {}) == original
    with pytest.raises(ValueError, match="cannot be rewritten"):
        c.record_handoff_member_result("handoff-1", 2, owner, canonical_bytes({"tokens": [99]}), {})


def test_pending_member_retry_uses_original_input_and_next_generation(make_stack):
    stack = make_stack()
    c = stack.coordinator
    c.begin_handoff_restore("handoff-1")
    original_input = c.get_handoff_member_record(stack.input_refs[6])
    first = c.acquire_handoff_pending("handoff-1", "group-1#6", "target", 600.0)
    c = reopen(stack)
    try:
        second = c.acquire_handoff_pending("handoff-1", "group-1#6", "replacement", 600.0)
        assert (first.generation, second.generation) == (0, 1)
        assert c.get_handoff_member_record(stack.input_refs[6]) == original_input
    finally:
        c.close()


@pytest.mark.parametrize("mutation", ["weights", "tools", "missing_input"])
def test_manifest_identity_or_original_input_mismatch_is_rejected(make_stack, mutation):
    stack = make_stack(committed=False)
    kwargs = asdict(stack.manifest)
    kwargs["handoff_id"] = "new-handoff"
    if mutation == "weights":
        kwargs["policy_identity"] = {**IDENTITY, "model_weights_fingerprint": "different-weights"}
    elif mutation == "tools":
        kwargs["tool_version"] = "different-tools"
    else:
        kwargs["checkpointed_trajectories"][0].pop("input_ref")
    with pytest.raises(ValueError, match="identity|original checkpointed"):
        stack.coordinator.prepare_handoff_manifest(HandoffManifest(**kwargs))


def test_retention_deadline_is_not_rebased_by_coordinator_restart(make_stack):
    stack = make_stack()
    stack.clock.wall += 3601.0
    c = reopen(stack, monotonic=-100.0)
    try:
        c.begin_handoff_restore("handoff-1")
        with pytest.raises(ValueError, match="snapshot expired"):
            c.acquire_handoff_trajectory("handoff-1", "group-1#2", "target", 600.0)
    finally:
        c.close()


@pytest.mark.parametrize("stage,expected_state", [("before_replace", "preparing"), ("after_replace", "committed")])
def test_process_kill_at_metadata_boundary_preserves_only_installed_state(make_stack, stage, expected_state):
    stack = make_stack(committed=False)
    stack.coordinator.close()
    script = """
import os, sys
from verl.trainer.ppo.v1.trajectory_state import DurableTrajectoryCoordinator
c = DurableTrajectoryCoordinator(sys.argv[1], 'run-1', wall_clock=lambda: 1700000000.0)
c._metadata.fault_hook = lambda stage: os._exit(37) if stage == sys.argv[2] else None
c.commit_handoff_manifest('handoff-1')
raise SystemExit('fault was not reached')
"""
    result = subprocess.run([sys.executable, "-c", script, str(stack.root), stage], capture_output=True, timeout=40)
    assert result.returncode == 37, result.stderr.decode()
    c = reopen(stack)
    try:
        assert c.summary()["handoffs"]["handoff-1"] == expected_state
        assert c.get_handoff_manifest("handoff-1").encode() == stack.manifest.encode()
        for ref in stack.input_refs.values():
            assert c.get_handoff_member_record(ref)
    finally:
        c.close()


def test_uncertain_metadata_write_stops_serving_until_restart(make_stack):
    stack = make_stack(committed=False)
    class InjectedCrash(Exception):
        pass
    def fault(stage):
        if stage == "after_replace":
            raise InjectedCrash
    stack.coordinator._metadata.fault_hook = fault
    with pytest.raises(InjectedCrash):
        stack.coordinator.commit_handoff_manifest("handoff-1")
    with pytest.raises(CoordinationStorageError, match="uncertain durable write"):
        stack.coordinator.summary()
    c = reopen(stack)
    try:
        assert c.summary()["handoffs"]["handoff-1"] == "committed"
    finally:
        c.close()


def test_second_writer_is_rejected_by_actual_cross_process_lock(make_stack):
    stack = make_stack()
    script = """
import sys
from verl.trainer.ppo.v1.trajectory_state import DurableTrajectoryCoordinator, CoordinationStorageError
try:
    DurableTrajectoryCoordinator(sys.argv[1], 'run-1')
except CoordinationStorageError:
    raise SystemExit(23)
raise SystemExit('second writer acquired the lock')
"""
    result = subprocess.run([sys.executable, "-c", script, str(stack.root)], capture_output=True, timeout=40)
    assert result.returncode == 23, result.stderr.decode()


def test_corrupt_state_fails_closed_and_releases_lock(make_stack):
    stack = make_stack()
    stack.coordinator.close()
    path = stack.root / "run-1" / "coordinator-state.json"
    envelope = json.loads(path.read_text())
    envelope["state"]["sequence"] += 1  # old digest no longer matches
    path.write_text(json.dumps(envelope))
    for _ in range(2):
        with pytest.raises(CoordinationStorageError, match="decode durable"):
            DurableTrajectoryCoordinator(str(stack.root), "run-1")


def test_corrupt_snapshot_bytes_prevent_restore(make_stack):
    stack = make_stack()
    stack.coordinator.close()
    state = json.loads((stack.root / "run-1" / "coordinator-state.json").read_text())["state"]
    ref = stack.manifest.checkpointed_trajectories[0]["snapshot_ref"]
    key = state["objects"][ref]["storage_key"]
    (stack.root / "run-1" / "objects" / key).write_bytes(b"corrupt actual object")
    with pytest.raises(CoordinationStorageError, match="object is unavailable"):
        DurableTrajectoryCoordinator(str(stack.root), "run-1")


def test_publication_requires_all_members_and_exact_group_keys(make_stack):
    stack = make_stack()
    stack.coordinator.begin_handoff_restore("handoff-1")
    with pytest.raises(ValueError, match="all handoff members"):
        stack.coordinator.prepare_publication("handoff-1", binding())
    complete_target_members(stack)
    incomplete = binding()
    incomplete["keys"].pop()
    with pytest.raises(ValueError, match="one verified final output"):
        stack.coordinator.prepare_publication("handoff-1", incomplete)


def test_dispatch_success_then_coordinator_crash_does_not_resend(make_stack):
    stack = make_stack()
    complete_target_members(stack)
    intent = stack.coordinator.prepare_publication("handoff-1", binding())
    stack.coordinator.mark_publication_started("handoff-1")
    # The real queue has already stored the data; the coordinator did not get
    # its receipt. Only the immutable intent survives this crash boundary.
    c = reopen(stack)
    try:
        assert c.get_publication("handoff-1")["state"] == "DISPATCHED"
        with pytest.raises(PublicationBlocked, match="query the real queue"):
            c.mark_publication_started("handoff-1")
        verified = c.reconcile_publication("handoff-1", receipt(intent))
        assert verified["state"] == "PUBLISHED"
    finally:
        c.close()


def test_absent_or_unreadable_queue_write_remains_blocked_not_retried(make_stack):
    stack = make_stack()
    complete_target_members(stack)
    stack.coordinator.prepare_publication("handoff-1", binding())
    stack.coordinator.mark_publication_started("handoff-1")
    assert stack.coordinator.reconcile_publication("handoff-1", None)["state"] == "BLOCKED"
    with pytest.raises(PublicationBlocked):
        stack.coordinator.mark_publication_started("handoff-1")
    with pytest.raises(PublicationBlocked, match="verified complete-group"):
        stack.coordinator.finish_handoff_manifest("handoff-1", "unverified")


def test_receipt_wrong_content_or_identity_is_rejected(make_stack):
    stack = make_stack()
    complete_target_members(stack)
    intent = stack.coordinator.prepare_publication("handoff-1", binding())
    stack.coordinator.mark_publication_started("handoff-1")
    wrong = receipt(intent)
    wrong["binding"] = {**binding(), "content_digest": "0" * 64}
    with pytest.raises(ValueError, match="does not match"):
        stack.coordinator.confirm_publication("handoff-1", wrong)
    wrong = receipt(intent)
    wrong["publication_id"] = "wrong-id"
    with pytest.raises(ValueError, match="does not match"):
        stack.coordinator.confirm_publication("handoff-1", wrong)


def test_publication_identity_is_immutable(make_stack):
    stack = make_stack()
    complete_target_members(stack)
    original = stack.coordinator.prepare_publication("handoff-1", binding())
    assert stack.coordinator.prepare_publication("handoff-1", binding()) == original
    changed = {**binding(), "partition_id": "val"}
    with pytest.raises(ValueError, match="cannot be rewritten"):
        stack.coordinator.prepare_publication("handoff-1", changed)


def test_verified_queue_reference_is_persistent_but_must_be_revalidated_after_restart(make_stack):
    stack = make_stack()
    complete_target_members(stack)
    intent = stack.coordinator.prepare_publication("handoff-1", binding())
    stack.coordinator.mark_publication_started("handoff-1")
    actual = receipt(intent)
    stack.coordinator.confirm_publication("handoff-1", actual)
    c = reopen(stack)
    try:
        assert c.get_publication("handoff-1")["state"] == "NEEDS_VERIFICATION"
        with pytest.raises(PublicationBlocked):
            c.finish_handoff_manifest("handoff-1", intent["publication_id"])
        assert c.reconcile_publication("handoff-1", actual)["state"] == "PUBLISHED"
        assert c.finish_handoff_manifest("handoff-1", intent["publication_id"]) == "finished"
    finally:
        c.close()


def test_finished_is_durable_before_gc_and_cleanup_survives_another_restart(make_stack):
    stack = make_stack()
    complete_target_members(stack)
    extra = stack.coordinator.put_handoff_member_record(
        stack.coordinator.get_handoff_member_record(stack.input_refs[6])
    )  # identical bytes, separately owned logical reference
    intent = stack.coordinator.prepare_publication("handoff-1", binding())
    stack.coordinator.mark_publication_started("handoff-1")
    stack.coordinator.confirm_publication("handoff-1", receipt(intent))
    stack.coordinator.finish_handoff_manifest("handoff-1", intent["publication_id"])
    assert stack.coordinator.summary()["gc_pending"] > 0
    # No cleanup was done yet; every retained snapshot remains physically present.
    ref = stack.manifest.checkpointed_trajectories[0]["snapshot_ref"]
    assert stack.coordinator.get_handoff_member_record(ref)
    c = reopen(stack)
    try:
        assert c.summary()["handoffs"]["handoff-1"] == "finished"
        assert c.collect_garbage() > 0
        assert c.summary()["gc_pending"] == 0
        assert c.collect_garbage() == 0
        assert c.get_handoff_member_record(extra)
        with pytest.raises(CoordinationStorageError, match="unknown durable"):
            c.get_handoff_member_record(ref)
    finally:
        c.close()


@pytest.mark.parametrize("run_id", ["", ".", "..", "other/run", "../escape"])
def test_run_namespace_is_path_safe(tmp_path, run_id):
    with pytest.raises(ValueError, match="path-safe"):
        DurableTrajectoryCoordinator(str(tmp_path), run_id)


@pytest.mark.skipif(sys.platform != "linux", reason="cluster CPU/Ray restart regression")
def test_real_ray_actor_restart_invalidates_leases_and_keeps_saved_results(make_stack):
    ray = pytest.importorskip("ray")
    if ray.is_initialized():
        pytest.skip("this regression requires its own CPU-only Ray runtime")
    stack = make_stack()
    stack.coordinator.close()
    from verl.trainer.ppo.v1.trajectory_state import (
        RayTrajectoryCoordinatorClient,
        create_ray_trajectory_coordinator_actor,
    )

    with tempfile.TemporaryDirectory(prefix="ts-ray-", dir="/tmp") as ray_tmp:
        ray.init(address="local", num_cpus=2, num_gpus=0, include_dashboard=False, _temp_dir=ray_tmp)
        actor = None
        try:
            Actor = create_ray_trajectory_coordinator_actor(max_restarts=2)
            actor = Actor.remote(str(stack.root), "run-1", wall_clock=lambda: 1_700_000_000.0)
            client = RayTrajectoryCoordinatorClient(actor)
            # Callbacks serialize this lightweight client without coordinator locks.
            client = ray.cloudpickle.loads(ray.cloudpickle.dumps(client))

            async def wait_ready():
                # ray.kill is asynchronous. A summary from the old process can
                # still complete before it exits; this actor only becomes ready
                # after reconstructing with no restored execution leases.
                for _ in range(200):
                    try:
                        state = await client.summary()
                        if state["active_execution_leases"] == 0:
                            return state
                    except ray.exceptions.RayActorError:
                        pass
                    await asyncio.sleep(0.2)
                raise AssertionError("coordinator did not restart")

            async def run():
                await wait_ready()
                await client.begin_handoff_restore("handoff-1")
                target = await client.acquire_handoff_trajectory("handoff-1", "group-1#2", "target", 600.0)
                assert target.generation == 1
                restored = await client.load_trajectory_snapshot(target)
                assert restored.encode() == snapshot("group-1#2", 2).encode()
                ray.kill(actor, no_restart=False)
                await wait_ready()
                with pytest.raises(ray.exceptions.RayTaskError, match="stale or expired"):
                    await client.check_trajectory_owner(target)
                replacement = await client.acquire_handoff_trajectory(
                    "handoff-1", "group-1#2", "replacement", 600.0
                )
                assert replacement.generation == 2
                actual = canonical_bytes({"tokens": [3, 7], "tool_results": ["already committed"]})
                reference = await client.record_handoff_member_result(
                    "handoff-1", 2, replacement, actual, {"tool_replay_count": 0}
                )
                ray.kill(actor, no_restart=False)
                await wait_ready()
                assert (await client.get_handoff_restored_members("handoff-1"))[2]["record_ref"] == reference
                assert await client.get_handoff_member_record(reference) == actual
                with pytest.raises(ray.exceptions.RayTaskError, match="already restored"):
                    await client.acquire_handoff_trajectory("handoff-1", "group-1#2", "third", 600.0)
                assert await client.get_current_producer_generation("source") == 0
                metrics = await client.get_handoff_metrics()
                assert metrics["handoffs"][0]["prepare_to_commit_ms"] is None

            asyncio.run(run())
        finally:
            if actor is not None:
                ray.kill(actor, no_restart=True)
            ray.shutdown()
