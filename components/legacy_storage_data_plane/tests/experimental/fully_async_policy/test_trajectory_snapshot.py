from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from verl.experimental.fully_async_policy.rollout_artifact import (
    CanonicalPolicyKVIdentity, InMemoryRolloutArtifactStore,
)
from verl.experimental.fully_async_policy.trajectory_snapshot import (
    TrajectorySnapshot, TrajectorySnapshotRegistry, token_digest,
)


def snapshot(generation=0):
    return TrajectorySnapshot(
        trajectory_id="task-1", group_id="group-1", member_index=1, group_size=4,
        execution_generation=generation, environment_version="env-1", tool_version="tools-1",
        policy_identity=asdict(CanonicalPolicyKVIdentity("0", "weights", "template", "kv")),
        sampling_params={"temperature": 0.7}, limits={"response_length": 1024},
        tool_selection=["read"], completed_tool_batches=[{
            "turn_index": 1, "calls": [{"name": "read", "arguments": {"id": 7}}],
            "results": [{"role": "tool", "content": "verified observation"}],
        }], agent_state={
            "messages": [{"role": "user", "content": "look up 7"},
                         {"role": "tool", "content": "verified observation"}],
            "prompt_ids": [100, 101, 10, 11, 202, 203], "response_ids": [10, 11],
            "response_mask": [1, 1, 0, 0], "response_logprobs": [-0.1, -0.2, 0.0, 0.0],
            "turn_scores": [], "tool_rewards": [1.0], "user_turns": 1, "assistant_turns": 1,
            "extra_fields": {"trajectory_segments": [{"start": 0, "end": 2,
                "policy_version": "0", "sampling_config_fingerprint": "a" * 64}]},
            "tools_kwargs": {}, "metrics": {"tool_calls": 0.05}, "request_id": "request-1",
            "mm_processor_kwargs": {},
        },
    )


def registry():
    clock = [0.0]
    store = InMemoryRolloutArtifactStore(clock=lambda: clock[0])
    return TrajectorySnapshotRegistry(store, clock=lambda: clock[0]), clock


def test_snapshot_preserves_actual_history_and_sampling_provenance():
    original = snapshot()
    assert TrajectorySnapshot.decode(original.encode()) == original
    ref = {"export_complete": True, "prefix_tokens": 4,
        "token_digest": token_digest(original.agent_state["prompt_ids"][:4]),
        "policy_identity": original.policy_identity, "kv_bytes": 8, "storage_reference": {"backend": "mooncake", "master_address": "store:50051",
        "source_completed_tokens": 4, "objects": [{"key": "page-kv", "bytes": 8, "sha256": "a" * 64}]}}
    with_kv = replace(original, kv_ref=ref)
    assert TrajectorySnapshot.decode(with_kv.encode()) == with_kv


@pytest.mark.parametrize("field", ["policy_version", "sampling_config_fingerprint", "execution_generation", "tool_result"])
def test_continuation_cannot_relabel_historical_sampling_or_tool_result(field):
    reg, clock = registry()
    a = reg.open("task-1", "a", 10)
    reg.commit(a, snapshot(), 100)
    clock[0] = 11
    b = reg.takeover("task-1", 0, "b", 10)
    candidate = snapshot(b.generation)
    if field == "tool_result":
        candidate.completed_tool_batches[0]["results"][0]["content"] = "changed result"
    else:
        candidate.agent_state["extra_fields"]["trajectory_segments"][0][field] = {
            "policy_version": "1", "sampling_config_fingerprint": "b" * 64,
            "execution_generation": 1}[field]
    with pytest.raises(ValueError, match="cannot be rewritten"):
        reg.commit(b, candidate, 100)
    assert reg.load(b) == snapshot()


@pytest.mark.parametrize("change", ["logprob", "mask", "segment", "media", "opaque", "kv"])
def test_invalid_snapshot_is_rejected(change):
    original = snapshot()
    state = deepcopy(original.agent_state)
    if change == "logprob":
        state["response_logprobs"][-1] = -0.1
    elif change == "mask":
        state["response_mask"][0] = 2
    elif change == "segment":
        state["extra_fields"]["trajectory_segments"][0]["end"] = 1
    elif change == "media":
        state["messages"][0]["content"] = [{"type": "image"}]
    elif change == "opaque":
        state["tools_kwargs"]["handle"] = object()
    else:
        original = replace(original, kv_ref={"export_complete": False})
    with pytest.raises((ValueError, TypeError)):
        replace(original, agent_state=state).encode()


def test_takeover_fences_old_owner_and_preserves_committed_generation():
    reg, clock = registry()
    a = reg.open("task-1", "worker-a", 10)
    key = reg.commit(a, snapshot(a.generation), 100)
    with pytest.raises(ValueError, match="expired"):
        reg.takeover("task-1", a.generation, "worker-b", 10)
    clock[0] = 11
    b = reg.takeover("task-1", a.generation, "worker-b", 10)
    assert b.generation == a.generation + 1
    assert reg.load(b) == snapshot(a.generation)
    assert reg.latest["task-1"] == key
    with pytest.raises(ValueError, match="stale"):
        reg.commit(a, snapshot(a.generation), 100)
    with pytest.raises(ValueError):
        reg.takeover("task-1", a.generation, "worker-c", 10)
    reg.commit(b, snapshot(b.generation), 100)
    assert reg.load(b).execution_generation == b.generation
    reg.finish(b)
    assert not reg.latest


def test_snapshots_are_separate_from_trainer_and_no_uncommitted_takeover():
    trainer_store = InMemoryRolloutArtifactStore()
    reg, clock = registry()
    a = reg.open("task-1", "a", 10)
    clock[0] = 11
    with pytest.raises(ValueError):
        reg.takeover("task-1", a.generation, "b", 10)
    a = reg.owners["task-1"]
    clock[0] = 1
    reg.commit(a, snapshot(a.generation), 100)
    assert trainer_store.claim_complete_group(accepted_policy_versions={"0"},
        lease_timeout_seconds=30) is None


def test_ttl_gc_respects_live_execution_owner():
    reg, clock = registry()
    a = reg.open("task-1", "a", 30)
    reg.commit(a, snapshot(a.generation), 5)
    clock[0] = 6
    assert reg.collect_expired() == 0
    with pytest.raises(ValueError, match="expired"):
        reg.load(a)
    clock[0] = 31
    assert reg.collect_expired() == 1
    assert not reg.latest


def test_digest_failure_never_returns_partially_decoded_snapshot():
    reg, _ = registry()
    a = reg.open("task-1", "a", 30)
    key = reg.commit(a, snapshot(), 100)
    # Existing payload digest boundary must reject corrupted committed bytes.
    reg.store._artifacts[key].payload = b"corrupted"
    with pytest.raises((ValueError, TypeError)):
        reg.load(a)


def test_lease_expiring_during_put_does_not_publish_snapshot_pointer():
    reg, clock = registry()
    a = reg.open("task-1", "a", 10)
    publish = reg.store.publish
    def delayed_publish(manifest, payload):
        publish(manifest, payload)
        clock[0] = 11
    reg.store.publish = delayed_publish
    with pytest.raises(ValueError, match="expired"):
        reg.commit(a, snapshot(), 100)
    assert not reg.latest and not reg.versions
