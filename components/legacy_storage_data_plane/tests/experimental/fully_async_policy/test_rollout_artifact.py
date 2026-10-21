from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import ray
import torch


_MODULE_PATH = Path(__file__).parents[3] / "verl/experimental/fully_async_policy/rollout_artifact.py"
_SPEC = importlib.util.spec_from_file_location("rollout_artifact_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

from verl.experimental.fully_async_policy import rollout_artifact as ray_artifact

InMemoryRolloutArtifactStore = _MODULE.InMemoryRolloutArtifactStore
MooncakePayloadStore = _MODULE.MooncakePayloadStore
MooncakeRolloutArtifactStore = _MODULE.MooncakeRolloutArtifactStore
CanonicalPolicyKVIdentity = _MODULE.CanonicalPolicyKVIdentity
RolloutArtifactManifest = _MODULE.RolloutArtifactManifest
RolloutArtifactStatus = _MODULE.RolloutArtifactStatus
RolloutTrainingSemantics = _MODULE.RolloutTrainingSemantics
TrajectorySegment = _MODULE.TrajectorySegment
rollout_training_semantics_from_batch = _MODULE.rollout_training_semantics_from_batch
validate_rollout_training_payload = _MODULE.validate_rollout_training_payload


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeMooncakeStore:
    def __init__(self, setup_return_code=0):
        self.setup_return_code = setup_return_code
        self.remove_return_code = 0
        self.force_supported = True
        self.setup_args = None
        self.objects = {}

    def setup(self, *args):
        self.setup_args = args
        return self.setup_return_code

    def put(self, key, value):
        self.objects[key] = value
        return 0

    def get(self, key):
        return self.objects.get(key, b"")

    def is_exist(self, key):
        return key in self.objects

    def remove(self, key, force=False):
        if force and not self.force_supported:
            raise TypeError("remove() got an unexpected keyword argument 'force'")
        if force:
            self.objects.pop(key, None)
            return 0
        return_code = self.remove_return_code
        if return_code != -706:
            self.objects.pop(key, None)
        return return_code


def identity(version: str = "42") -> CanonicalPolicyKVIdentity:
    return CanonicalPolicyKVIdentity(version, "weights-42", "template-a", "kv-schema-a")


def manifest(artifact_id: str, index: int, *, version: str = "42", group_id: str = "group-a", size: int = 2):
    return RolloutArtifactManifest(
        artifact_id=artifact_id,
        producer_id="rollouter-0",
        identity=identity(version),
        rollout_group_id=group_id,
        rollout_group_size=size,
        rollout_sample_index=index,
    )


def test_canonical_identity_digest_and_cache_salt_are_stable():
    first = identity()
    second = CanonicalPolicyKVIdentity("42", "weights-42", "template-a", "kv-schema-a")

    assert first.digest == second.digest
    assert first.cache_salt == first.digest


def training_semantics(**overrides):
    values = dict(
        trajectory_policy_version='42', environment_version='env-1', tool_version='tools-1',
        producer_generation=0, generated_token_ids=(10, 11), behavior_logprobs=(-0.1, -0.2),
        trajectory_token_counts=(2,),
        segments=(TrajectorySegment(0, 0, 2, '42', 'a' * 64, (-0.1, -0.2)),),
    )
    values.update(overrides)
    return RolloutTrainingSemantics(**values)


@pytest.mark.parametrize("store_kind", ["memory", "mooncake"])
@pytest.mark.parametrize("claim_by_notification", [False, True])
def test_claim_filters_stale_segment_with_accepted_final_version(store_kind, claim_by_notification):
    fake = FakeMooncakeStore()
    store = InMemoryRolloutArtifactStore() if store_kind == "memory" else MooncakeRolloutArtifactStore(
        payload_store=MooncakePayloadStore(store=fake)
    )
    store.register_producer("rollouter-0", "worker-a")
    segments = (
        TrajectorySegment(0, 0, 1, "41", "a" * 64, (-0.1,)),
        TrajectorySegment(0, 1, 2, "42", "a" * 64, (-0.2,)),
    )
    store.publish(_MODULE.replace(manifest("stale-segment", 0, size=1),
                                  training_semantics=training_semantics(segments=segments)), b"stale")
    store.publish(_MODULE.replace(manifest("fresh", 0, group_id="fresh", size=1),
                                  training_semantics=training_semantics()), b"fresh")
    kwargs = dict(accepted_policy_versions={"42"}, lease_timeout_seconds=10)
    if claim_by_notification:
        assert store.claim_group_for_artifact("stale-segment", **kwargs) is None
        lease = store.claim_group_for_artifact("fresh", **kwargs)
    else:
        lease = store.claim_complete_group(**kwargs)
    assert store.get_status("stale-segment") is RolloutArtifactStatus.REJECTED
    assert lease is not None and lease.artifact_ids == ("fresh",)
    assert lease.payloads == (b"fresh",)
    if store_kind == "mooncake":
        assert list(fake.objects.values()) == [b"fresh"]


def test_stale_segment_member_does_not_make_partial_group_trainable():
    store = InMemoryRolloutArtifactStore()
    store.register_producer("rollouter-0", "worker-a")
    segments = (
        TrajectorySegment(0, 0, 1, "41", "a" * 64, (-0.1,)),
        TrajectorySegment(0, 1, 2, "42", "a" * 64, (-0.2,)),
    )
    store.publish(_MODULE.replace(manifest("a0", 0),
                                  training_semantics=training_semantics(segments=segments)), b"stale")
    store.publish(_MODULE.replace(manifest("a1", 1), training_semantics=training_semantics()), b"fresh")
    assert store.claim_group_for_artifact("a1", accepted_policy_versions={"42"}, lease_timeout_seconds=10) is None
    assert store.get_status("a0") is RolloutArtifactStatus.REJECTED
    assert store.get_status("a1") is RolloutArtifactStatus.READY


@pytest.mark.parametrize("claim_by_notification", [True])
def test_claim_rejects_stale_segment_published_after_sweep(claim_by_notification):
    """A stale-segment artifact that reaches READY after the sweep must still be
    rejected when its group is assembled for a lease (concurrent Ray actor: the
    publisher runs as a separate task from the consumer's claim)."""
    store = InMemoryRolloutArtifactStore()
    store.register_producer("rollouter-0", "worker-a")
    # A fresh, acceptable artifact exists before any sweep runs.
    store.publish(_MODULE.replace(manifest("fresh", 0, group_id="fresh", size=1),
                                  training_semantics=training_semantics()), b"fresh")
    # Drive one claim to consume the notification and run the sweep against the
    # store state so far (the stale artifact does not exist yet).
    kwargs = dict(accepted_policy_versions={"42"}, lease_timeout_seconds=10)
    lease = store.claim_group_for_artifact("fresh", **kwargs)
    assert lease is not None and lease.artifact_ids == ("fresh",)
    store.release(lease)
    # Now the publisher flips a retract-recovered mixed-version artifact to
    # READY after the sweep: the only remaining path into training is a claim,
    # and that claim must reject it instead of leasing it.
    segments = (
        TrajectorySegment(0, 0, 1, "41", "a" * 64, (-0.1,)),
        TrajectorySegment(0, 1, 2, "42", "a" * 64, (-0.2,)),
    )
    store.publish(_MODULE.replace(manifest("late-stale", 0, group_id="late-stale", size=1),
                                  training_semantics=training_semantics(segments=segments)), b"late-stale")
    assert store.claim_group_for_artifact("late-stale", **kwargs) is None
    assert store.get_status("late-stale") is RolloutArtifactStatus.REJECTED
    rejected_reason = store._artifacts["late-stale"].rejected_reason
    assert rejected_reason == "policy_segment_not_accepted"


def test_segment_filter_preserves_live_lease_until_release():
    store = InMemoryRolloutArtifactStore()
    store.register_producer("rollouter-0", "worker-a")
    segments = (
        TrajectorySegment(0, 0, 1, "41", "a" * 64, (-0.1,)),
        TrajectorySegment(0, 1, 2, "42", "a" * 64, (-0.2,)),
    )
    store.publish(_MODULE.replace(manifest("a0", 0, size=1),
                                  training_semantics=training_semantics(segments=segments)), b"payload")
    lease = store.claim_complete_group(accepted_policy_versions={"41", "42"}, lease_timeout_seconds=10)
    assert lease is not None
    assert store.reject_unaccepted_policy_versions({"42"}) == 0
    assert store.get_status("a0") is RolloutArtifactStatus.LEASED
    store.release(lease)
    assert store.claim_group_for_artifact("a0", accepted_policy_versions={"42"}, lease_timeout_seconds=10) is None
    assert store.get_status("a0") is RolloutArtifactStatus.REJECTED


def test_training_semantics_are_derived_from_masked_payload_tokens():
    class Batch:
        batch = {
            "responses": torch.tensor([[10, 0, 11]]),
            "response_mask": torch.tensor([[1, 0, 1]]),
            "rollout_log_probs": torch.tensor([[-0.1, 0.0, -0.2]]),
        }
        non_tensor_batch = {'trajectory_segments': [[
            {'start': 0, 'end': 2, 'policy_version': '42', 'sampling_config_fingerprint': 'a' * 64, 'turn_index': 0}
        ]]}

    semantics = rollout_training_semantics_from_batch(
        Batch(), policy_version="42", environment_version="env-1", tool_version="tools-1", producer_generation=0
    )
    assert semantics.generated_token_ids == (10, 11)
    assert semantics.behavior_logprobs == pytest.approx((-0.1, -0.2))
    Batch.batch["response_mask"] = torch.tensor([[1, 2, 0]])
    with pytest.raises(ValueError, match="response_mask"):
        rollout_training_semantics_from_batch(
            Batch(), policy_version="42", environment_version="env-1", tool_version="tools-1", producer_generation=0
        )


def test_training_semantics_detects_payload_token_or_logprob_change():
    class Batch:
        batch = {
            "responses": torch.tensor([[10, 11]]),
            "response_mask": torch.tensor([[1, 1]]),
            "rollout_log_probs": torch.tensor([[-0.1, -0.2]]),
        }
        non_tensor_batch = {'trajectory_segments': [[
            {'start': 0, 'end': 2, 'policy_version': '42', 'sampling_config_fingerprint': 'a' * 64, 'turn_index': 0}
        ]]}

    expected = rollout_training_semantics_from_batch(
        Batch(), policy_version="42", environment_version="env-1", tool_version="tools-1", producer_generation=0
    )
    Batch.batch["responses"][0, 1] = 12
    assert rollout_training_semantics_from_batch(
        Batch(), policy_version="42", environment_version="env-1", tool_version="tools-1", producer_generation=0
    ) != expected
    Batch.batch["responses"][0, 1] = 11
    Batch.batch["rollout_log_probs"][0, 1] = -0.5
    assert rollout_training_semantics_from_batch(
        Batch(), policy_version="42", environment_version="env-1", tool_version="tools-1", producer_generation=0
    ) != expected


def test_segment_provenance_accepts_cross_policy_continuation_and_rejects_stale_segment():
    class Batch:
        batch = {
            "responses": torch.tensor([[10, 11, 12, 13]]),
            "response_mask": torch.ones((1, 4), dtype=torch.long),
            "rollout_log_probs": torch.tensor([[-0.1, -0.2, -0.3, -0.4]]),
        }
        non_tensor_batch = {
            "min_global_steps": [41],
            "max_global_steps": [42],
            "trajectory_segments": [[
                {'start': 0, 'end': 2, 'policy_version': '41', 'sampling_config_fingerprint': 'a' * 64, 'turn_index': 0},
                {'start': 2, 'end': 4, 'policy_version': '42', 'sampling_config_fingerprint': 'b' * 64, 'turn_index': 0},
            ]],
            "rollout_group_id": ["group-a"],
            "rollout_group_size": [1],
            "rollout_sample_index": [0],
            "rollout_group_role": ["sample"],
            "producer_generation": [0],
        }

    class Sample:
        sample_id = "group-a"
        full_batch = Batch()

    semantics = rollout_training_semantics_from_batch(
        Batch(), policy_version="42", environment_version="env-1", tool_version="tools-1", producer_generation=0
    )
    assert [(segment.start, segment.end, segment.policy_version) for segment in semantics.segments] == [
        (0, 2, "41"), (2, 4, "42")
    ]
    assert semantics.trainable_token_mask({"42"}) == (False, False, True, True)
    artifact = _MODULE.replace(manifest("a0", 0, size=1), training_semantics=semantics)
    payload = ray.cloudpickle.dumps(Sample())
    validate_rollout_training_payload(
        payload, artifact, environment_version="env-1", tool_version="tools-1",
        accepted_policy_versions={"41", "42"},
    )
    with pytest.raises(ValueError, match="stale policy segment"):
        validate_rollout_training_payload(
            payload, artifact, environment_version="env-1", tool_version="tools-1",
            accepted_policy_versions={"42"},
        )

    # The artifact's KV identity is anchored at 42; it cannot make the older
    # behavior-policy segment eligible for training.
    assert artifact.identity.policy_version == "42"
    original_segments = Batch.non_tensor_batch["trajectory_segments"][0]
    for segments in (
        [{**original_segments[0], "end": 1}, original_segments[1]],  # gap
        [original_segments[0], {**original_segments[1], "start": 1}],  # overlap
        [original_segments[0], {**original_segments[1], "end": 5}],  # out of bounds
        [{**original_segments[0], "sampling_config_fingerprint": "bad"}, original_segments[1]],
    ):
        Batch.non_tensor_batch["trajectory_segments"] = [segments]
        with pytest.raises(ValueError, match="trajectory segment"):
            rollout_training_semantics_from_batch(
                Batch(), policy_version="42", environment_version="env-1", tool_version="tools-1", producer_generation=0
            )
    Batch.non_tensor_batch["trajectory_segments"] = [[{**original_segments[0], "policy_version": "40"}, original_segments[1]]]
    with pytest.raises(ValueError, match="differ from payload"):
        validate_rollout_training_payload(
            ray.cloudpickle.dumps(Sample()), artifact, environment_version="env-1", tool_version="tools-1"
        )
    Batch.non_tensor_batch["trajectory_segments"] = [original_segments]
    Batch.batch["rollout_log_probs"][0, 1] = -0.9
    with pytest.raises(ValueError, match="differ from payload"):
        validate_rollout_training_payload(
            ray.cloudpickle.dumps(Sample()), artifact, environment_version="env-1", tool_version="tools-1"
        )
def test_training_payload_validation_rejects_tampered_or_stale_serialized_samples():
    class Sample:
        pass

    class Batch:
        pass

    sample = Sample()
    sample.sample_id = "group-a"
    sample.full_batch = Batch()
    sample.full_batch.batch = {
        "responses": torch.tensor([[10, 11]]),
        "response_mask": torch.tensor([[1, 1]]),
        "rollout_log_probs": torch.tensor([[-0.1, -0.2]]),
    }
    sample.full_batch.non_tensor_batch = {
        "min_global_steps": [42],
        "max_global_steps": [42],
        "trajectory_segments": [[{'start': 0, 'end': 2, 'policy_version': '42', 'sampling_config_fingerprint': 'a' * 64, 'turn_index': 0}]],
        "rollout_group_id": ["group-a"],
        "rollout_group_size": [1],
        "rollout_sample_index": [0],
        "rollout_group_role": ["sample"],
        "producer_generation": [0],
    }
    semantics = rollout_training_semantics_from_batch(
        sample.full_batch, policy_version="42", environment_version="env-1", tool_version="tools-1",
        producer_generation=0
    )
    artifact = _MODULE.replace(manifest("a0", 0), training_semantics=semantics)
    payload = ray.cloudpickle.dumps(sample)
    validate_rollout_training_payload(payload, artifact, environment_version="env-1", tool_version="tools-1")

    sample.full_batch.non_tensor_batch["min_global_steps"] = [41]
    with pytest.raises(ValueError, match="policy version differs"):
        validate_rollout_training_payload(
            ray.cloudpickle.dumps(sample), artifact, environment_version="env-1", tool_version="tools-1"
        )
    sample.full_batch.non_tensor_batch["min_global_steps"] = [42]

    with pytest.raises(ValueError, match="differ from payload"):
        validate_rollout_training_payload(payload, artifact, environment_version="env-2", tool_version="tools-1")
    sample.full_batch.batch["responses"][0, 1] = 12
    with pytest.raises(ValueError, match="differ from payload"):
        validate_rollout_training_payload(
            ray.cloudpickle.dumps(sample), artifact, environment_version="env-1", tool_version="tools-1"
        )
    sample.full_batch.batch["responses"][0, 1] = 11
    sample.full_batch.batch["rollout_log_probs"][0, 1] = -0.5
    with pytest.raises(ValueError, match="differ from payload"):
        validate_rollout_training_payload(
            ray.cloudpickle.dumps(sample), artifact, environment_version="env-1", tool_version="tools-1"
        )


def test_training_payload_validation_rejects_mixed_or_incomplete_response_group():
    class Sample:
        pass

    class Batch:
        pass

    sample = Sample()
    sample.sample_id = "group-a"
    sample.full_batch = Batch()
    sample.full_batch.batch = {
        "responses": torch.tensor([[10, 11], [12, 13]]),
        "response_mask": torch.ones((2, 2), dtype=torch.long),
        "rollout_log_probs": torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
    }
    metadata = {
        "min_global_steps": np.array([42, 42], dtype=np.int64),
        "max_global_steps": np.array([42, 42], dtype=np.int64),
        "trajectory_segments": [[{'start': 0, 'end': 2, 'policy_version': '42', 'sampling_config_fingerprint': 'a' * 64, 'turn_index': 0}] for _ in range(2)],
        "rollout_group_id": np.array(["group-a", "group-a"], dtype=object),
        "rollout_group_size": np.array([2, 2], dtype=np.int64),
        "rollout_sample_index": np.array([0, 1], dtype=np.int64),
        "rollout_group_role": np.array(["sample", "sample"], dtype=object),
        "producer_generation": np.array([0, 0], dtype=np.int64),
    }
    sample.full_batch.non_tensor_batch = metadata
    semantics = rollout_training_semantics_from_batch(
        sample.full_batch, policy_version="42", environment_version="env-1", tool_version="tools-1",
        producer_generation=0
    )
    artifact = _MODULE.replace(manifest("a0", 0, size=1), trajectory_count=2, training_semantics=semantics)

    def validate():
        validate_rollout_training_payload(
            ray.cloudpickle.dumps(sample), artifact, environment_version="env-1", tool_version="tools-1"
        )

    validate()
    sample.sample_id = "old-group"
    with pytest.raises(ValueError, match="payload group differs"):
        validate()
    sample.sample_id = "group-a"
    with pytest.raises(ValueError, match="payload group differs"):
        validate_rollout_training_payload(
            ray.cloudpickle.dumps(sample), _MODULE.replace(artifact, trajectory_count=1),
            environment_version="env-1", tool_version="tools-1",
        )

    for field, invalid in (
        ("rollout_group_id", ["group-a", "old-group"]),
        ("rollout_group_size", [2, 1]),
        ("rollout_sample_index", [0, 0]),
        ("rollout_sample_index", [False, 1]),
        ("rollout_group_role", ["sample", "manifest"]),
    ):
        sample.full_batch.non_tensor_batch = {**metadata, field: invalid}
        with pytest.raises(ValueError, match="group metadata"):
            validate()
    sample.full_batch.non_tensor_batch = {
        key: value for key, value in metadata.items() if key != "rollout_sample_index"
    }
    with pytest.raises(ValueError, match="group metadata is missing"):
        validate()
    sample.full_batch.non_tensor_batch = {**metadata, "min_global_steps": [42]}
    with pytest.raises(ValueError, match="policy version differs"):
        validate()


def test_payload_metadata_rejects_mixed_producer_generation():
    class Sample:
        pass

    class Batch:
        pass

    sample = Sample()
    sample.sample_id = "group-a"
    sample.full_batch = Batch()
    sample.full_batch.batch = {
        "responses": torch.tensor([[10, 11], [12, 13]]),
        "response_mask": torch.ones((2, 2), dtype=torch.long),
        "rollout_log_probs": torch.tensor([[-0.1, -0.2], [-0.3, -0.4]]),
    }
    sample.full_batch.non_tensor_batch = {
        "min_global_steps": np.array([42, 42], dtype=np.int64),
        "max_global_steps": np.array([42, 42], dtype=np.int64),
        "trajectory_segments": [[{'start': 0, 'end': 2, 'policy_version': '42', 'sampling_config_fingerprint': 'a' * 64, 'turn_index': 0}] for _ in range(2)],
        "rollout_group_id": np.array(["group-a", "group-a"], dtype=object),
        "rollout_group_size": np.array([2, 2], dtype=np.int64),
        "rollout_sample_index": np.array([0, 1], dtype=np.int64),
        "rollout_group_role": np.array(["sample", "sample"], dtype=object),
        "producer_generation": np.array([0, 0], dtype=np.int64),
    }
    semantics = rollout_training_semantics_from_batch(
        sample.full_batch,
        policy_version="42",
        environment_version="env-1",
        tool_version="tools-1",
        producer_generation=0,
    )
    artifact = _MODULE.replace(
        manifest("a0", 0, size=1), trajectory_count=2, training_semantics=semantics
    )
    validate_rollout_training_payload(
        ray.cloudpickle.dumps(sample), artifact, environment_version="env-1", tool_version="tools-1"
    )

    sample.full_batch.non_tensor_batch = {
        **sample.full_batch.non_tensor_batch,
        "producer_generation": [0, 1],
    }
    with pytest.raises(ValueError, match="producer generation differs"):
        validate_rollout_training_payload(
            ray.cloudpickle.dumps(sample), artifact, environment_version="env-1", tool_version="tools-1"
        )


def test_producer_registry_rejects_stale_generation_publish_and_claim():
    store = InMemoryRolloutArtifactStore()
    assert store.register_producer("rollouter-0", "worker-a") == 0
    assert store.register_producer("rollouter-0", "worker-a") == 0
    assert store.register_producer("rollouter-0", "worker-b") == 1
    with pytest.raises(ValueError, match="retired producer incarnation is not current"):
        store.register_producer("rollouter-0", "worker-a")
    assert store.get_current_producer_generation("rollouter-0") == 1

    old_artifact = _MODULE.replace(
        manifest("a0", 0, group_id="group-a", size=1),
        trajectory_count=2,
        training_semantics=training_semantics(producer_generation=0),
    )
    with pytest.raises(ValueError, match="stale producer generation"):
        store.publish(old_artifact, b"old-payload")
    assert store.get_metrics()["total"] == 0

    assert store.register_producer("rollouter-0", "worker-c") == 2
    current_artifact = _MODULE.replace(
        manifest("a1", 0, group_id="group-b", size=1),
        trajectory_count=2,
        training_semantics=training_semantics(producer_generation=2),
    )
    store.publish(current_artifact, b"current-payload")
    assert store.claim_group_for_artifact(
        "a1",
        accepted_policy_versions={"42"},
        lease_timeout_seconds=10,
        accepted_environment_versions={"env-1"},
        accepted_tool_versions={"tools-1"},
        require_current_producer_generation=True,
    ) is not None


def test_optional_claim_fence_rejects_old_ready_artifact_after_generation_bump():
    store = InMemoryRolloutArtifactStore()
    assert store.register_producer("rollouter-0", "worker-a") == 0
    old_artifact = _MODULE.replace(
        manifest("a0", 0, group_id="group-a", size=1),
        trajectory_count=2,
        training_semantics=training_semantics(producer_generation=0),
    )
    store.publish(old_artifact, b"old-payload")
    assert store.register_producer("rollouter-0", "worker-b") == 1

    assert store.claim_group_for_artifact(
        "a0",
        accepted_policy_versions={"42"},
        lease_timeout_seconds=10,
        accepted_environment_versions={"env-1"},
        accepted_tool_versions={"tools-1"},
        require_current_producer_generation=True,
    ) is None
    assert store.get_status("a0") is RolloutArtifactStatus.REJECTED


def test_committed_artifact_survives_generation_bump_and_late_commit_is_fenced():
    store = InMemoryRolloutArtifactStore()
    assert store.register_producer("rollouter-0", "worker-a") == 0
    old_artifact = _MODULE.replace(
        manifest("a0", 0, group_id="group-a", size=1),
        trajectory_count=2,
        training_semantics=training_semantics(producer_generation=0),
    )
    pending = _MODULE.replace(old_artifact, artifact_id="pending", rollout_group_id="group-pending")
    store.publish(old_artifact, b"committed-before-handoff")
    store.reserve(pending, b"not-yet-committed")
    assert store.register_producer("rollouter-0", "worker-b") == 1

    with pytest.raises(ValueError, match="stale producer generation"):
        store.commit("pending", b"not-yet-committed")
    with pytest.raises(ValueError, match="stale producer generation"):
        store.publish(_MODULE.replace(old_artifact, artifact_id="late", rollout_group_id="group-late"), b"late")

    lease = store.claim_group_for_artifact(
        "a0",
        accepted_policy_versions={"42"},
        lease_timeout_seconds=10,
        accepted_environment_versions={"env-1"},
        accepted_tool_versions={"tools-1"},
    )
    assert lease is not None
    assert lease.payloads == (b"committed-before-handoff",)
    assert lease.manifests[0].training_semantics.producer_generation == 0
    store.acknowledge(lease)
    assert store.get_status("a0") is RolloutArtifactStatus.CONSUMED


def test_mooncake_rejects_stale_producer_before_payload_write():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(payload_store=MooncakePayloadStore(store=fake_store))
    assert store.register_producer("rollouter-0", "worker-a") == 0
    stale_artifact = _MODULE.replace(
        manifest("a0", 0, group_id="group-a", size=1),
        trajectory_count=2,
        training_semantics=training_semantics(producer_generation=0),
    )
    assert store.register_producer("rollouter-0", "worker-b") == 1

    with pytest.raises(ValueError, match="stale producer generation"):
        store.publish(stale_artifact, b"payload")
    assert fake_store.objects == {}


def test_default_artifact_path_does_not_require_producer_generation():
    store = InMemoryRolloutArtifactStore()
    store.publish(manifest("a0", 0, size=1), b"payload")
    assert store.claim_group_for_artifact(
        "a0", accepted_policy_versions={"42"}, lease_timeout_seconds=10
    ) is not None


@pytest.mark.parametrize(
    ("producer_id", "incarnation_id", "error"),
    [
        ("", "worker-a", "producer_id must be a non-empty string"),
        ("rollouter-0", "", "incarnation_id must be a non-empty string"),
    ],
)
def test_producer_registry_rejects_empty_identifiers(producer_id, incarnation_id, error):
    store = InMemoryRolloutArtifactStore()
    with pytest.raises(ValueError, match=error):
        store.register_producer(producer_id, incarnation_id)


def test_metadata_snapshot_json_roundtrip_preserves_state_and_retired_incarnations():
    store = InMemoryRolloutArtifactStore()
    assert store.register_producer("rollouter-0", "worker-a") == 0
    assert store.register_producer("rollouter-0", "worker-b") == 1
    artifact = _MODULE.replace(
        manifest("a0", 0, group_id="group-a", size=1),
        trajectory_count=2,
        training_semantics=training_semantics(producer_generation=1),
    )
    store.publish(artifact, b"payload")
    snapshot = json.loads(json.dumps(store.export_metadata_snapshot()))

    restored = InMemoryRolloutArtifactStore()
    restored.publish(manifest("old", 0), b"old")
    restored.restore_metadata_snapshot(snapshot, payload_resolver=lambda manifest: b"payload")

    assert restored.get_status("a0") is RolloutArtifactStatus.READY
    assert restored.get_metrics()["total"] == 1
    assert restored.register_producer("rollouter-0", "worker-b") == 1
    with pytest.raises(ValueError, match="retired producer incarnation is not current"):
        restored.register_producer("rollouter-0", "worker-a")
    lease = restored.claim_complete_group(
        accepted_policy_versions={"42"},
        lease_timeout_seconds=10,
        accepted_environment_versions={"env-1"},
        accepted_tool_versions={"tools-1"},
        require_current_producer_generation=True,
    )
    assert lease is not None
    assert lease.payloads == (b"payload",)


def test_metadata_snapshot_restores_leased_artifact_as_ready_and_preserves_writing():
    store = InMemoryRolloutArtifactStore()
    store.publish(manifest("a0", 0), b"payload")
    store.publish(manifest("a1", 1), b"payload")
    lease = store.claim_complete_group(
        accepted_policy_versions={"42"},
        lease_timeout_seconds=10,
    )
    assert lease is not None
    leased_snapshot = store.export_metadata_snapshot()
    assert all(item["status"] == "leased" for item in leased_snapshot["artifacts"])

    restored = InMemoryRolloutArtifactStore()
    restored.restore_metadata_snapshot(leased_snapshot, payload_resolver=lambda manifest: b"payload")
    assert restored.get_status("a0") is RolloutArtifactStatus.READY
    assert restored.get_status("a1") is RolloutArtifactStatus.READY
    assert restored.claim_complete_group(
        accepted_policy_versions={"42"}, lease_timeout_seconds=10
    ) is not None

    writing_snapshot = json.loads(json.dumps(leased_snapshot))
    writing_snapshot["artifacts"][0]["status"] = "writing"
    writing_snapshot["artifacts"][1]["status"] = "writing"
    writing_restored = InMemoryRolloutArtifactStore()
    writing_restored.restore_metadata_snapshot(writing_snapshot)
    assert writing_restored.get_status("a0") is RolloutArtifactStatus.WRITING
    assert writing_restored.get_status("a1") is RolloutArtifactStatus.WRITING


def test_metadata_snapshot_rejects_unknown_or_malformed_schema():
    store = InMemoryRolloutArtifactStore()
    snapshot = store.export_metadata_snapshot()
    unknown_schema = {**snapshot, "schema_version": 3}
    with pytest.raises(ValueError, match="unsupported metadata snapshot schema"):
        InMemoryRolloutArtifactStore().restore_metadata_snapshot(unknown_schema)

    old_schema = {**snapshot, "schema_version": 1}
    with pytest.raises(ValueError, match="unsupported metadata snapshot schema"):
        InMemoryRolloutArtifactStore().restore_metadata_snapshot(old_schema)

    malformed = {**snapshot, "artifacts": [{"manifest": {}, "status": "ready", "rejected_reason": None}]}
    with pytest.raises(ValueError, match="invalid schema"):
        InMemoryRolloutArtifactStore().restore_metadata_snapshot(malformed)


@pytest.mark.parametrize('override, error', [
    ({'trajectory_policy_version': '41'}, 'trajectory policy version'),
    ({'environment_version': ''}, 'environment and tool versions'),
    ({'tool_version': ''}, 'environment and tool versions'),
    ({'generated_token_ids': ()}, 'generated_token_ids'),
    ({'generated_token_ids': (True, 11)}, 'generated_token_ids'),
    ({'behavior_logprobs': (-0.1,)}, 'behavior_logprobs'),
    ({'behavior_logprobs': (float('nan'), -0.2)}, 'behavior_logprobs'),
])
def test_training_semantics_rejects_malformed_artifacts(override, error):
    store = InMemoryRolloutArtifactStore()
    with pytest.raises(ValueError, match=error):
        store.publish(_MODULE.replace(manifest('a0', 0), training_semantics=training_semantics(**override)), b'data')
    assert store.get_metrics()['total'] == 0


def test_training_semantics_keeps_kv_identity_and_checks_group_freshness():
    first = _MODULE.replace(manifest('a0', 0), training_semantics=training_semantics())
    assert first.identity.digest == manifest('a0', 0).identity.digest
    store = InMemoryRolloutArtifactStore()
    assert store.register_producer('rollouter-0', 'worker-a') == 0
    assert store.register_producer('rollouter-0', 'worker-a') == 0
    store.publish(first, b'first')
    with pytest.raises(ValueError, match='environment and tool versions'):
        store.publish(_MODULE.replace(manifest('a1', 1), training_semantics=training_semantics(tool_version='tools-2')), b'second')
    with pytest.raises(ValueError, match='environment and tool versions'):
        store.publish(manifest('a1', 1), b'second')
    store.publish(_MODULE.replace(manifest('a1', 1), training_semantics=training_semantics()), b'second')
    assert store.claim_complete_group(accepted_policy_versions={'42'}, lease_timeout_seconds=10) is not None


@pytest.mark.parametrize('store_kind', ['memory', 'mooncake'])
def test_training_freshness_rejects_missing_or_stale_versions(store_kind):
    store = InMemoryRolloutArtifactStore() if store_kind == 'memory' else MooncakeRolloutArtifactStore(
        payload_store=MooncakePayloadStore(store=FakeMooncakeStore())
    )
    assert store.register_producer('rollouter-0', 'worker-a') == 0
    store.publish(manifest('legacy', 0, group_id='legacy', size=1), b'legacy')
    store.publish(_MODULE.replace(manifest('fresh', 0, group_id='fresh', size=1),
                                  training_semantics=training_semantics()), b'fresh')
    store.publish(_MODULE.replace(manifest('stale', 0, group_id='stale', size=1),
                                  training_semantics=training_semantics(environment_version='env-0')), b'stale')
    lease = store.claim_complete_group(accepted_policy_versions={'42'}, lease_timeout_seconds=10,
                                       accepted_environment_versions={'env-1'}, accepted_tool_versions={'tools-1'})
    assert lease is not None and lease.artifact_ids == ('fresh',)
    assert store.get_status('legacy') is RolloutArtifactStatus.REJECTED
    assert store.get_status('stale') is RolloutArtifactStatus.REJECTED


def test_notification_claim_rejects_stale_tool_version():
    store = InMemoryRolloutArtifactStore()
    store.register_producer('rollouter-0', 'worker-a')
    store.publish(_MODULE.replace(manifest('old', 0, size=1), training_semantics=training_semantics(
        tool_version='tools-0'
    )), b'old')
    assert store.claim_group_for_artifact(
        'old', accepted_policy_versions={'42'}, lease_timeout_seconds=10,
        accepted_environment_versions={'env-1'}, accepted_tool_versions={'tools-1'},
    ) is None
    assert store.get_status('old') is RolloutArtifactStatus.REJECTED


def test_fully_async_identity_factory_is_shared_by_artifact_and_rollout_paths():
    v10_first = CanonicalPolicyKVIdentity.for_fully_async_policy_version(10)
    v10_second = CanonicalPolicyKVIdentity.for_fully_async_policy_version("10")
    v11 = CanonicalPolicyKVIdentity.for_fully_async_policy_version(11)

    assert v10_first.cache_salt == v10_first.digest == v10_second.cache_salt
    assert v11.cache_salt == v11.digest
    assert v10_first.cache_salt != v11.cache_salt


def test_complete_group_is_atomically_claimed_and_ack_is_idempotent():
    store = InMemoryRolloutArtifactStore()
    store.publish(manifest("a0", 0), {"sample": 0})
    assert store.claim_complete_group(accepted_policy_versions={"42"}, lease_timeout_seconds=10) is None

    store.publish(manifest("a1", 1), {"sample": 1})
    lease = store.claim_complete_group(accepted_policy_versions={"42"}, lease_timeout_seconds=10)
    assert lease is not None
    assert [payload["sample"] for payload in lease.payloads] == [0, 1]
    assert store.claim_complete_group(accepted_policy_versions={"42"}, lease_timeout_seconds=10) is None

    store.acknowledge(lease)
    store.acknowledge(lease)
    assert store.get_status("a0") is RolloutArtifactStatus.CONSUMED
    assert store.get_status("a1") is RolloutArtifactStatus.CONSUMED


def test_rejects_duplicate_indexes_and_incompatible_group_identity():
    store = InMemoryRolloutArtifactStore()
    store.publish(manifest("a0", 0), "payload")
    with pytest.raises(ValueError, match="duplicate"):
        store.publish(manifest("duplicate", 0), "payload")
    with pytest.raises(ValueError, match="same identity"):
        store.publish(manifest("wrong-version", 1, version="43"), "payload")


def test_rejects_stale_versions_before_consume():
    store = InMemoryRolloutArtifactStore()
    store.publish(manifest("a0", 0), 0)
    store.publish(manifest("a1", 1), 1)

    assert store.claim_complete_group(accepted_policy_versions={"43"}, lease_timeout_seconds=10) is None
    assert store.get_status("a0") is RolloutArtifactStatus.REJECTED
    assert store.get_metrics()["rejected"] == 2


def test_expired_lease_can_be_claimed_again_but_wrong_lease_cannot_ack():
    clock = FakeClock()
    store = InMemoryRolloutArtifactStore(clock=clock)
    store.publish(manifest("a0", 0), 0)
    store.publish(manifest("a1", 1), 1)
    first = store.claim_complete_group(accepted_policy_versions={"42"}, lease_timeout_seconds=5)
    assert first is not None

    clock.now = 6
    second = store.claim_complete_group(accepted_policy_versions={"42"}, lease_timeout_seconds=5)
    assert second is not None and second.lease_token != first.lease_token
    with pytest.raises(ValueError, match="lease does not own"):
        store.acknowledge(first)
    store.acknowledge(second)


def test_notification_claims_only_its_announced_complete_group():
    store = InMemoryRolloutArtifactStore()
    store.publish(manifest("a0", 0, group_id="group-a"), "a0")
    store.publish(manifest("a1", 1, group_id="group-a"), "a1")
    store.publish(manifest("b0", 0, group_id="group-b"), "b0")
    store.publish(manifest("b1", 1, group_id="group-b"), "b1")

    lease = store.claim_group_for_artifact(
        "b1", accepted_policy_versions={"42"}, lease_timeout_seconds=10
    )

    assert lease is not None
    assert lease.artifact_ids == ("b0", "b1")
    assert store.get_status("a0") is RolloutArtifactStatus.READY
    assert store.get_status("a1") is RolloutArtifactStatus.READY


def test_notification_for_incomplete_or_already_leased_group_claims_nothing_else():
    store = InMemoryRolloutArtifactStore()
    store.publish(manifest("incomplete", 0, group_id="group-a"), "a0")
    store.publish(manifest("b0", 0, group_id="group-b"), "b0")
    store.publish(manifest("b1", 1, group_id="group-b"), "b1")

    assert (
        store.claim_group_for_artifact("incomplete", accepted_policy_versions={"42"}, lease_timeout_seconds=10)
        is None
    )
    assert store.get_status("b0") is RolloutArtifactStatus.READY

    lease = store.claim_group_for_artifact("b0", accepted_policy_versions={"42"}, lease_timeout_seconds=10)
    assert lease is not None
    assert (
        store.claim_group_for_artifact("b1", accepted_policy_versions={"42"}, lease_timeout_seconds=10) is None
    )


def test_v1_fails_closed_for_kv_ref_and_non_sample_role():
    store = InMemoryRolloutArtifactStore()
    bad = RolloutArtifactManifest(
        artifact_id="baseline",
        producer_id="rollouter-0",
        identity=identity(),
        rollout_group_id="group-a",
        rollout_group_size=2,
        rollout_sample_index=-1,
        rollout_group_role="remax_baseline",
        kv_ref="unsafe-kv-handle",  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="kv_ref"):
        store.publish(bad, None)


def test_rejects_an_empty_atomic_trajectory_group():
    store = InMemoryRolloutArtifactStore()
    bad = RolloutArtifactManifest(
        artifact_id="empty-group",
        producer_id="rollouter-0",
        identity=identity(),
        rollout_group_id="group-a",
        rollout_group_size=1,
        rollout_sample_index=0,
        trajectory_count=0,
    )
    with pytest.raises(ValueError, match="trajectory_count"):
        store.publish(bad, None)


def test_mooncake_backend_offloads_payloads_and_removes_them_after_ack():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(
        namespace="test-artifacts",
        payload_store=MooncakePayloadStore(
            local_hostname="worker-0",
            metadata_server="http://master:8080/metadata",
            master_server_address="master:50051",
            store=fake_store,
        ),
    )
    store.publish(manifest("a0", 0), b"payload-0")
    store.publish(manifest("a1", 1), b"payload-1")

    assert len(fake_store.objects) == 2
    assert all(key.startswith("test-artifacts/") for key in fake_store.objects)
    lease = store.claim_complete_group(accepted_policy_versions={"42"}, lease_timeout_seconds=10)
    assert lease is not None
    assert lease.payloads == (b"payload-0", b"payload-1")
    assert lease.manifests[0].payload_digest == hashlib.sha256(b"payload-0").hexdigest()

    store.acknowledge(lease)
    assert fake_store.objects == {}
    assert store.get_status("a0") is RolloutArtifactStatus.CONSUMED


def test_mooncake_backend_release_keeps_payload_for_retry():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(
        payload_store=MooncakePayloadStore(local_hostname="worker-0", store=fake_store)
    )
    store.publish(manifest("a0", 0), b"payload-0")
    store.publish(manifest("a1", 1), b"payload-1")

    first = store.claim_complete_group(accepted_policy_versions={"42"}, lease_timeout_seconds=10)
    assert first is not None
    store.release(first)
    assert len(fake_store.objects) == 2

    second = store.claim_complete_group(accepted_policy_versions={"42"}, lease_timeout_seconds=10)
    assert second is not None
    assert second.payloads == (b"payload-0", b"payload-1")


def test_mooncake_expired_lease_reclaims_once_and_rejects_late_ack():
    clock = FakeClock()
    payload_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(
        clock=clock, payload_store=MooncakePayloadStore(store=payload_store)
    )
    store.publish(manifest("a0", 0), b"first")
    store.publish(manifest("a1", 1), b"second")
    first = store.claim_group_for_artifact(
        "a0", accepted_policy_versions={"42"}, lease_timeout_seconds=5
    )
    assert first is not None
    assert store.claim_group_for_artifact(
        "a1", accepted_policy_versions={"42"}, lease_timeout_seconds=5
    ) is None

    clock.now = 6
    second = store.claim_group_for_artifact(
        "a1", accepted_policy_versions={"42"}, lease_timeout_seconds=5
    )
    assert second is not None and second.lease_token != first.lease_token
    with pytest.raises(ValueError, match="lease does not own"):
        store.acknowledge(first)
    assert second.payloads == (b"first", b"second")
    store.acknowledge(second)
    store.acknowledge(second)
    assert payload_store.objects == {}


def test_mooncake_retry_does_not_overwrite_or_delete_owned_payload():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(payload_store=MooncakePayloadStore(store=fake_store))
    store.publish(manifest("a0", 0), b"original")
    store.publish(manifest("a0", 0), b"original")

    with pytest.raises(ValueError, match="already exists"):
        store.publish(manifest("a0", 0), b"replacement")

    assert len(fake_store.objects) == 1
    assert next(iter(fake_store.objects.values())) == b"original"
    store.publish(manifest("a1", 1), b"second")
    lease = store.claim_complete_group(accepted_policy_versions={"42"}, lease_timeout_seconds=10)
    assert lease is not None
    assert lease.payloads == (b"original", b"second")


def test_mooncake_retry_adopts_matching_orphan_after_interrupted_metadata_publish():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(payload_store=MooncakePayloadStore(store=fake_store))
    payload = b"original"
    key = store._payload_handle(
        _MODULE.replace(manifest("orphan", 0), payload_digest=hashlib.sha256(payload).hexdigest())
    ).key
    fake_store.objects[key] = payload

    store.publish(manifest("orphan", 0), payload)

    assert fake_store.objects[key] == payload
    assert store.get_status("orphan") is RolloutArtifactStatus.READY
    assert store.get_metrics()["total"] == 1


def test_mooncake_retry_rejects_mismatched_orphan_payload():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(payload_store=MooncakePayloadStore(store=fake_store))
    payload = b"original"
    key = store._payload_handle(
        _MODULE.replace(manifest("orphan", 0), payload_digest=hashlib.sha256(payload).hexdigest())
    ).key
    fake_store.objects[key] = b"different"

    with pytest.raises(ValueError, match="different payload"):
        store.publish(manifest("orphan", 0), payload)

    assert fake_store.objects[key] == b"different"
    assert store.get_metrics()["total"] == 0


def test_mooncake_retry_refuses_missing_owned_object():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(payload_store=MooncakePayloadStore(store=fake_store))
    store.publish(manifest("a0", 0), b"original")
    fake_store.objects.clear()

    with pytest.raises(ValueError, match="already exists"):
        store.publish(manifest("a0", 0), b"original")

    assert fake_store.objects == {}
    assert store.get_status("a0") is RolloutArtifactStatus.READY


def test_mooncake_rejected_publish_only_removes_new_unowned_payload():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(payload_store=MooncakePayloadStore(store=fake_store))
    store.publish(manifest("a0", 0), b"original")
    original_key = next(iter(fake_store.objects))

    with pytest.raises(ValueError, match="duplicate"):
        store.publish(manifest("a1", 0), b"duplicate index")

    assert fake_store.objects == {original_key: b"original"}
    assert store.get_metrics()["total"] == 1


def test_mooncake_rejected_artifact_cleanup_is_idempotent_and_tolerates_missing_object():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(payload_store=MooncakePayloadStore(store=fake_store))
    store.publish(manifest("stale", 0, group_id="stale-group", size=1), b"payload")

    assert store.reject_unaccepted_policy_versions({"43"}) == 1
    assert store.get_status("stale") is RolloutArtifactStatus.REJECTED
    assert fake_store.objects == {}

    assert store.cleanup_rejected_payloads() == 1
    assert store.cleanup_rejected_payloads() == 1
    assert fake_store.objects == {}


def test_mooncake_backend_fails_closed_and_releases_lease_on_payload_corruption():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(
        payload_store=MooncakePayloadStore(local_hostname="worker-0", store=fake_store)
    )
    store.publish(manifest("a0", 0), b"payload-0")
    store.publish(manifest("a1", 1), b"payload-1")
    fake_store.objects[next(iter(fake_store.objects))] = b"corrupted"

    with pytest.raises(RuntimeError, match="digest mismatch"):
        store.claim_complete_group(accepted_policy_versions={"42"}, lease_timeout_seconds=10)

    assert store.get_status("a0") is RolloutArtifactStatus.READY
    assert store.get_status("a1") is RolloutArtifactStatus.READY


def test_mooncake_backend_rejects_unserialized_payloads_and_setup_failures():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(
        payload_store=MooncakePayloadStore(local_hostname="worker-0", store=fake_store)
    )
    with pytest.raises(TypeError, match="serialized bytes"):
        store.publish(manifest("a0", 0), {"not": "bytes"})

    with pytest.raises(RuntimeError, match="return_code=17"):
        MooncakePayloadStore(store=FakeMooncakeStore(setup_return_code=17))


def test_mooncake_backend_defers_removal_while_object_lease_is_active():
    fake_store = FakeMooncakeStore()
    fake_store.force_supported = False
    store = MooncakeRolloutArtifactStore(
        payload_store=MooncakePayloadStore(local_hostname="worker-0", store=fake_store)
    )
    store.publish(manifest("a0", 0), b"payload-0")
    store.publish(manifest("a1", 1), b"payload-1")
    lease = store.claim_complete_group(accepted_policy_versions={"42"}, lease_timeout_seconds=10)
    assert lease is not None

    fake_store.remove_return_code = -706
    store.acknowledge(lease)
    assert store.get_status("a0") is RolloutArtifactStatus.CONSUMED
    assert store.get_metrics()["mooncake_pending_removals"] == 2
    assert len(fake_store.objects) == 2

    fake_store.force_supported = True
    metrics = store.get_metrics()
    assert metrics["mooncake_pending_removals"] == 0
    assert fake_store.objects == {}


def test_ray_actor_restart_bumps_producer_generation_and_fences_old_ready_artifact():
    def actor_manifest(artifact_id: str, group_id: str, generation: int):
        return ray_artifact.RolloutArtifactManifest(
            artifact_id=artifact_id,
            producer_id="rollouter-0",
            identity=ray_artifact.CanonicalPolicyKVIdentity(
                "42", "weights-42", "template-a", "kv-schema-a"
            ),
            rollout_group_id=group_id,
            rollout_group_size=1,
            rollout_sample_index=0,
            trajectory_count=2,
            training_semantics=ray_artifact.RolloutTrainingSemantics(
                trajectory_policy_version="42",
                environment_version="env-1",
                tool_version="tools-1",
                producer_generation=generation,
                generated_token_ids=(10, 11),
                behavior_logprobs=(-0.1, -0.2),
                trajectory_token_counts=(2,),
                segments=(ray_artifact.TrajectorySegment(0, 0, 2, '42', 'a' * 64, (-0.1, -0.2)),),
            ),
        )

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=3, include_dashboard=False, log_to_driver=False)

    try:
        StoreActor = ray_artifact.create_ray_rollout_artifact_store_actor()
        store = StoreActor.remote()

        @ray.remote(num_cpus=1)
        class Producer:
            def register(self, store_actor):
                incarnation_id = str(ray.get_runtime_context().get_worker_id())
                generation = ray.get(
                    store_actor.register_producer.remote(
                        producer_id="rollouter-0",
                        incarnation_id=incarnation_id,
                    )
                )
                return incarnation_id, generation

        first = Producer.remote()
        first_incarnation, first_generation = ray.get(first.register.remote(store))
        assert first_incarnation
        assert first_generation == 0

        old_artifact = actor_manifest("old", "old-group", first_generation)
        ray.get(store.publish.remote(old_artifact, b"old-payload"))

        ray.kill(first)
        second = Producer.remote()
        second_incarnation, second_generation = ray.get(second.register.remote(store))
        assert second_incarnation != first_incarnation
        assert second_generation > first_generation

        assert ray.get(
            store.claim_group_for_artifact.remote(
                "old",
                {"42"},
                10,
                {"env-1"},
                {"tools-1"},
                True,
            )
        ) is None

        fresh_artifact = actor_manifest("fresh", "fresh-group", second_generation)
        ray.get(store.publish.remote(fresh_artifact, b"fresh-payload"))
        lease = ray.get(
            store.claim_group_for_artifact.remote(
                "fresh",
                {"42"},
                10,
                {"env-1"},
                {"tools-1"},
                True,
            )
        )
        assert lease is not None
        assert lease.payloads == (b"fresh-payload",)
    finally:
        if started_here and ray.is_initialized():
            ray.shutdown()


def test_mooncake_snapshot_recovery_reconciles_writing_missing_and_corrupt_payloads():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(payload_store=MooncakePayloadStore(store=fake_store))

    ready_payload = b"ready"
    writing_payload = b"writing"
    missing_payload = b"missing"
    corrupt_payload = b"corrupt"

    manifests = {}
    for artifact_id, payload in (
        ("ready", ready_payload),
        ("writing", writing_payload),
        ("missing", missing_payload),
        ("corrupt", corrupt_payload),
    ):
        m = _MODULE.replace(
            manifest(artifact_id, 0, group_id=artifact_id, size=1),
            payload_digest=hashlib.sha256(payload).hexdigest(),
        )
        manifests[artifact_id] = m

    snapshot = {
        "schema_version": 2,
        "producers": {},
        "artifacts": [
            {"manifest": _MODULE._encode_manifest(manifests["ready"]), "status": "ready", "rejected_reason": None},
            {"manifest": _MODULE._encode_manifest(manifests["writing"]), "status": "writing", "rejected_reason": None},
            {"manifest": _MODULE._encode_manifest(manifests["missing"]), "status": "ready", "rejected_reason": None},
            {"manifest": _MODULE._encode_manifest(manifests["corrupt"]), "status": "leased", "rejected_reason": None},
        ],
    }

    fake_store.objects[store._payload_handle(manifests["ready"]).key] = ready_payload
    fake_store.objects[store._payload_handle(manifests["writing"]).key] = writing_payload
    fake_store.objects[store._payload_handle(manifests["corrupt"]).key] = b"damaged"

    store.restore_metadata_snapshot(snapshot)

    assert store.get_status("ready") is RolloutArtifactStatus.READY
    assert store.get_status("writing") is RolloutArtifactStatus.READY
    assert store.get_status("missing") is RolloutArtifactStatus.REJECTED
    assert store.get_status("corrupt") is RolloutArtifactStatus.REJECTED

    recovered = store.export_metadata_snapshot()
    by_id = {item["manifest"]["artifact_id"]: item for item in recovered["artifacts"]}
    assert by_id["missing"]["rejected_reason"] == "payload_missing_after_recovery"
    assert by_id["corrupt"]["rejected_reason"] == "payload_corrupt_after_recovery"


def test_mooncake_reserve_write_and_commit_lifecycle_is_fail_closed():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(payload_store=MooncakePayloadStore(store=fake_store))
    artifact = manifest("lifecycle-a0", 0, group_id="lifecycle-group", size=1)

    store.reserve(artifact, b"payload")
    assert store.get_status("lifecycle-a0") is RolloutArtifactStatus.WRITING
    assert fake_store.objects == {}
    with pytest.raises(ValueError, match="payload is missing before commit"):
        store.commit(artifact, b"payload")
    assert store.get_status("lifecycle-a0") is RolloutArtifactStatus.WRITING

    handle = store._payload_handle(_MODULE.replace(artifact, payload_digest=hashlib.sha256(b"payload").hexdigest()))
    fake_store.objects[handle.key] = b"payload"
    assert store.get_status("lifecycle-a0") is RolloutArtifactStatus.WRITING

    store.commit(artifact, b"payload")
    assert store.get_status("lifecycle-a0") is RolloutArtifactStatus.READY


def test_mooncake_recovery_rejects_reserve_without_payload_and_promotes_written_payload():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(payload_store=MooncakePayloadStore(store=fake_store))
    artifact = manifest("crash-a0", 0, group_id="crash-group", size=1)

    store.reserve(artifact, b"payload")
    reserved_snapshot = store.export_metadata_snapshot()
    store.restore_metadata_snapshot(reserved_snapshot)
    assert store.get_status("crash-a0") is RolloutArtifactStatus.REJECTED
    recovered = store.export_metadata_snapshot()
    assert recovered["artifacts"][0]["rejected_reason"] == "payload_missing_after_recovery"

    fake_store.objects.clear()
    store = MooncakeRolloutArtifactStore(payload_store=MooncakePayloadStore(store=fake_store))
    store.reserve(artifact, b"payload")
    handle = store._payload_handle(_MODULE.replace(artifact, payload_digest=hashlib.sha256(b"payload").hexdigest()))
    fake_store.objects[handle.key] = b"payload"
    written_snapshot = store.export_metadata_snapshot()
    store.restore_metadata_snapshot(written_snapshot)
    assert store.get_status("crash-a0") is RolloutArtifactStatus.READY


def test_mooncake_recovery_cleans_consumed_payload_residue():
    fake_store = FakeMooncakeStore()
    store = MooncakeRolloutArtifactStore(payload_store=MooncakePayloadStore(store=fake_store))
    store.publish(manifest("consumed-a0", 0, group_id="consumed-group", size=1), b"payload")
    lease = store.claim_group_for_artifact(
        "consumed-a0", accepted_policy_versions={"42"}, lease_timeout_seconds=10
    )
    assert lease is not None

    fake_store.force_supported = False
    fake_store.remove_return_code = -706
    store.acknowledge(lease)
    assert store.get_status("consumed-a0") is RolloutArtifactStatus.CONSUMED
    assert len(fake_store.objects) == 1

    fake_store.force_supported = True
    fake_store.remove_return_code = 0
    store.restore_metadata_snapshot(store.export_metadata_snapshot())
    assert fake_store.objects == {}


def test_mooncake_handoff_restores_metadata_payloads_and_generation_history():
    fake_store = FakeMooncakeStore()
    namespace = "verl/test-handoff"

    old_owner = MooncakeRolloutArtifactStore(
        namespace=namespace,
        payload_store=MooncakePayloadStore(store=fake_store),
    )
    generation0 = old_owner.register_producer("rollouter-0", "worker-old")
    assert generation0 == 0

    old_manifest = _MODULE.replace(
        manifest("handoff-old", 0, group_id="handoff-old", size=1),
        training_semantics=training_semantics(producer_generation=generation0),
    )
    old_owner.publish(old_manifest, b"old-generation-payload")
    lease = old_owner.claim_group_for_artifact(
        "handoff-old",
        accepted_policy_versions={"42"},
        lease_timeout_seconds=30,
        accepted_environment_versions={"env-1"},
        accepted_tool_versions={"tools-1"},
        require_current_producer_generation=True,
    )
    assert lease is not None
    assert old_owner.get_status("handoff-old") is RolloutArtifactStatus.LEASED

    key = old_owner.persist_handoff_snapshot("switch-0")
    assert key == f"{namespace}/_handoff/switch-0"
    # Persist is idempotent while the source snapshot is unchanged.
    assert old_owner.persist_handoff_snapshot("switch-0") == key

    new_owner = MooncakeRolloutArtifactStore(
        namespace=namespace,
        payload_store=MooncakePayloadStore(store=fake_store),
    )
    new_owner.restore_handoff_snapshot("switch-0")

    # Leases are process-local ownership. Recovery makes the complete group
    # claimable again while retaining the immutable Mooncake payload.
    assert new_owner.get_status("handoff-old") is RolloutArtifactStatus.READY
    assert new_owner.get_current_producer_generation("rollouter-0") == generation0

    generation1 = new_owner.register_producer("rollouter-0", "worker-new")
    assert generation1 == generation0 + 1

    # The old generation is fenced before a new trainer can consume it.
    assert new_owner.claim_group_for_artifact(
        "handoff-old",
        accepted_policy_versions={"42"},
        lease_timeout_seconds=30,
        accepted_environment_versions={"env-1"},
        accepted_tool_versions={"tools-1"},
        require_current_producer_generation=True,
    ) is None
    assert new_owner.get_status("handoff-old") is RolloutArtifactStatus.REJECTED

    fresh_manifest = _MODULE.replace(
        manifest("handoff-new", 0, group_id="handoff-new", size=1),
        training_semantics=training_semantics(producer_generation=generation1),
    )
    new_owner.publish(fresh_manifest, b"new-generation-payload")
    fresh = new_owner.claim_group_for_artifact(
        "handoff-new",
        accepted_policy_versions={"42"},
        lease_timeout_seconds=30,
        accepted_environment_versions={"env-1"},
        accepted_tool_versions={"tools-1"},
        require_current_producer_generation=True,
    )
    assert fresh is not None
    assert fresh.payloads == (b"new-generation-payload",)

    new_owner.delete_handoff_snapshot("switch-0")
    assert key not in fake_store.objects


class _IdentityConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, name, default=None):
        return getattr(self, name, default)


class _IdentityTokenizer:
    def __init__(self, *, chat_template="template-a"):
        self.name_or_path = "tokenizer-a"
        self.chat_template = chat_template
        self.vocab_size = 1000
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0


def _runtime_identity(
    version="7",
    *,
    template="template-a",
    tp=2,
    quantization=None,
    replica_generation=None,
    disaggregation=None,
):
    model = _IdentityConfig(
        path="/models/qwen",
        tokenizer_path="/models/qwen",
        custom_chat_template=None,
        trust_remote_code=True,
        model_type="language_model",
        override_config={"rope_scaling": None},
    )
    rollout = _IdentityConfig(
        name="sglang",
        dtype="bfloat16",
        tensor_model_parallel_size=tp,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_parallel_size=1,
        moe_tensor_parallel_size=1,
        quantization=quantization,
        quantization_config_file=None,
        sglang_engine_mode="local",
        disaggregation=disaggregation
        or _IdentityConfig(
            enabled=False,
            prefill_replicas=1,
            decode_replicas=1,
            decode_tensor_model_parallel_size=None,
            transfer_backend="nixl",
            bootstrap_port=None,
            ib_device=None,
        ),
        replica_generation=replica_generation,
    )
    return _MODULE.build_fully_async_policy_kv_identity(
        version,
        model_config=model,
        rollout_config=rollout,
        tokenizer=_IdentityTokenizer(chat_template=template),
    )


def test_runtime_policy_kv_identity_is_stable_and_sensitive_to_compatibility_inputs():
    first = _runtime_identity()
    same = _runtime_identity()
    assert first == same
    assert first.digest == same.digest
    assert first.cache_salt == first.digest

    assert _runtime_identity("8").digest != first.digest
    assert _runtime_identity(template="template-b").digest != first.digest
    assert _runtime_identity(tp=4).digest != first.digest
    assert _runtime_identity(quantization="fp8").digest != first.digest


def test_runtime_policy_kv_identity_excludes_replica_generation():
    assert _runtime_identity(replica_generation=0) == _runtime_identity(replica_generation=99)


def test_runtime_policy_kv_identity_excludes_disaggregation_execution_fields():
    mapping = {
        "enabled": True,
        "prefill_replicas": 2,
        "decode_replicas": 3,
        "decode_tensor_model_parallel_size": 4,
        "transfer_backend": "mooncake",
        "bootstrap_port": 23456,
        "ib_device": "mlx5_0",
    }
    object_config = _IdentityConfig(**mapping)
    assert _runtime_identity(disaggregation=mapping) == _runtime_identity(disaggregation=object_config)
    assert _runtime_identity(disaggregation=mapping) == _runtime_identity()


def test_runtime_policy_kv_identity_uses_model_config_content_not_local_path(tmp_path):
    first_model_dir = tmp_path / "cache-a" / "qwen"
    second_model_dir = tmp_path / "cache-b" / "qwen"
    for model_dir in (first_model_dir, second_model_dir):
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text('{"model_type":"qwen"}', encoding="utf-8")

    def identity(model_dir):
        model = _IdentityConfig(
            path=str(model_dir),
            tokenizer_path=str(model_dir),
            custom_chat_template=None,
            trust_remote_code=True,
            model_type="language_model",
            override_config={"rope_scaling": None},
        )
        rollout = _IdentityConfig(
            name="sglang",
            dtype="bfloat16",
            tensor_model_parallel_size=2,
            sglang_engine_mode="local",
        )
        return _MODULE.build_fully_async_policy_kv_identity(
            "7", model_config=model, rollout_config=rollout, tokenizer=_IdentityTokenizer()
        )

    assert identity(first_model_dir) == identity(second_model_dir)
    assert identity(first_model_dir).cache_salt == identity(second_model_dir).cache_salt




def test_ray_store_shares_weight_and_artifact_shadow_admission_view():
    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)

    try:
        StoreActor = ray_artifact.create_ray_rollout_artifact_store_actor(
            transfer_admission_config={
                "shadow": True,
                "max_inflight_transfers": 1,
                "max_inflight_bytes": 1024,
                "max_background_inflight": 1,
                "max_records": 100,
            }
        )
        store = StoreActor.remote()
        weight_request = ray_artifact.TransferRequest(
            transfer_id="weight-sync:7",
            intent=ray_artifact.TransferIntent.WEIGHT_SYNC,
            bytes=256,
        )
        assert ray.get(store.begin_weight_sync_admission.remote(weight_request))["action"] == "admit"

        # A duplicate begin exercises the fail-open telemetry boundary without
        # blocking the real weight sync.
        assert ray.get(store.begin_weight_sync_admission.remote(weight_request)) is None

        artifact_manifest = ray_artifact.RolloutArtifactManifest(
            artifact_id="artifact",
            producer_id="rollouter-0",
            identity=ray_artifact.CanonicalPolicyKVIdentity(
                "42", "weights-42", "template-a", "kv-schema-a"
            ),
            rollout_group_id="artifact-group",
            rollout_group_size=1,
            rollout_sample_index=0,
            trajectory_count=1,
        )
        ray.get(store.publish.remote(artifact_manifest, b"payload"))
        records = ray.get(store.get_transfer_admission_records.remote())
        by_id = {record["transfer_id"]: record for record in records}
        assert by_id["weight-sync:7"]["action"] == "admit"
        assert by_id["rollout-artifact:artifact"]["action"] == "queue"
        assert all(record["affects_execution"] is False for record in records)

        ray.get(store.complete_transfer_admission.remote(weight_request.transfer_id))
        summary = ray.get(store.get_transfer_admission_summary.remote())
        assert summary["transfer_admission/queue"] == 1.0
        assert summary["transfer_admission/active"] == 0.0
    finally:
        if started_here and ray.is_initialized():
            ray.shutdown()


def test_shadow_admission_wrapper_fails_open_on_controller_exception():
    class BrokenController:
        def begin(self, request):
            raise RuntimeError("telemetry failure")

        def complete(self, transfer_id):
            raise RuntimeError("telemetry failure")

    request = ray_artifact.TransferRequest(
        transfer_id="weight-sync:7",
        intent=ray_artifact.TransferIntent.WEIGHT_SYNC,
        bytes=1,
    )

    assert ray_artifact.begin_transfer_admission_safely(BrokenController(), request) is None
    ray_artifact.complete_transfer_admission_safely(BrokenController(), request.transfer_id)


def test_ray_store_covers_artifact_claim_read_in_shared_shadow_admission():
    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)

    try:
        StoreActor = ray_artifact.create_ray_rollout_artifact_store_actor(
            transfer_admission_config={
                "shadow": True,
                "max_inflight_transfers": 1,
                "max_inflight_bytes": 1024,
                "max_background_inflight": 1,
                "max_records": 100,
            }
        )
        store = StoreActor.remote()
        manifest = ray_artifact.RolloutArtifactManifest(
            artifact_id="read-artifact",
            producer_id="rollouter-0",
            identity=ray_artifact.CanonicalPolicyKVIdentity(
                "42", "weights-42", "template-a", "kv-schema-a"
            ),
            rollout_group_id="read-group",
            rollout_group_size=1,
            rollout_sample_index=0,
            trajectory_count=1,
        )
        ray.get(store.publish.remote(manifest, b"payload"))

        weight_request = ray_artifact.TransferRequest(
            transfer_id="weight-sync:7",
            intent=ray_artifact.TransferIntent.WEIGHT_SYNC,
            bytes=256,
        )
        assert ray.get(store.begin_weight_sync_admission.remote(weight_request))["action"] == "admit"

        lease = ray.get(
            store.claim_group_for_artifact.remote(
                "read-artifact",
                {"42"},
                10.0,
            )
        )
        assert lease is not None
        assert lease.payloads == (b"payload",)

        ray.get(store.complete_transfer_admission.remote(weight_request.transfer_id))
        records = ray.get(store.get_transfer_admission_records.remote())
        read_records = [
            record
            for record in records
            if record["transfer_id"].startswith("rollout-artifact-read:")
        ]
        assert len(read_records) == 1
        assert read_records[0]["action"] == "queue"
        assert read_records[0]["affects_execution"] is False
        assert ray.get(store.get_transfer_admission_summary.remote())[
            "transfer_admission/active"
        ] == 0.0
    finally:
        if started_here and ray.is_initialized():
            ray.shutdown()
