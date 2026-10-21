"""restore_handoff orchestration tests on the real snapshot/fencing stack.

Scenario (requirement 7): one n=8 GRPO group with 2 completed members,
4 preempted (checkpointed) members and 2 pending members.  The target must
restore the 6 finished records with tool_replay_count == 0, re-prefill every
checkpointed token (KV Restore = OFF), reject stale source writes, and the
final group publish must cover members 0..7 exactly once.
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from verl.experimental.fully_async_policy.handoff_executor import (
    publish_restored_group,
    restore_handoff,
)
from verl.experimental.fully_async_policy.preemptible_handoff import (
    HandoffManifest,
    HandoffManifestRegistry,
    HandoffMemberRecordStore,
    HandoffRestoreCoordinator,
    TrajectoryPreemptionTracker,
)
from verl.experimental.fully_async_policy.rollout_artifact import (
    CanonicalPolicyKVIdentity,
    InMemoryRolloutArtifactStore,
    RolloutArtifactManifest,
)
from verl.experimental.fully_async_policy.trajectory_snapshot import (
    TrajectorySnapshot,
    TrajectorySnapshotRegistry,
)

PRODUCER = "fully_async_rollouter"
ENV, TOOLS = "env-1", "tools-1"


def make_snapshot(trajectory_id, group_id, member_index, group_size, generation):
    return TrajectorySnapshot(
        trajectory_id=trajectory_id, group_id=group_id,
        member_index=member_index, group_size=group_size,
        execution_generation=generation,
        environment_version=ENV, tool_version=TOOLS,
        policy_identity=asdict(CanonicalPolicyKVIdentity("0", "weights", "template", "kv")),
        sampling_params={"temperature": 0.7}, limits={"response_length": 1024},
        tool_selection=["read"],
        completed_tool_batches=[{
            "turn_index": 1,
            "calls": [{"name": "read", "arguments": {"id": 7}}],
            "results": [{"role": "tool", "content": "verified observation"}],
        }],
        agent_state={
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


class FakeHandoffClient:
    """Async façade over the real in-process store/registry objects."""

    def __init__(self):
        self.store = InMemoryRolloutArtifactStore()
        self.snapshots = TrajectorySnapshotRegistry(InMemoryRolloutArtifactStore())
        self.tracker = TrajectoryPreemptionTracker()
        self.handoffs = HandoffManifestRegistry(
            self.tracker, current_producer_generation=self.store.get_current_producer_generation)
        self.records = HandoffMemberRecordStore()
        self.restore = HandoffRestoreCoordinator(self.handoffs, self.snapshots, self.records, self.store)
        self.generation = self.store.register_producer(PRODUCER, "incarnation-source")

    # source-side helpers used by the test rig
    async def commit_source_snapshot(self, trajectory_id, member_index,
                                      snapshot_member_index=None):
        lease = self.snapshots.open(trajectory_id, f"source-{member_index}", 600)
        stored_member = snapshot_member_index if snapshot_member_index is not None else member_index
        snapshot = make_snapshot(trajectory_id, "group-1", stored_member, 8, lease.generation)
        key = self.snapshots.commit(lease, snapshot, 3600)
        self.tracker.register_running(trajectory_id)
        self.tracker.request_preemption(trajectory_id)
        self.tracker.mark_safe_point(trajectory_id)
        self.tracker.mark_snapshot_committed(trajectory_id, key)
        self.snapshots.release_owner(lease)  # explicit source release
        return lease, key

    async def get_handoff_manifest(self, handoff_id):
        return self.handoffs.get(handoff_id).manifest

    async def begin_handoff_restore(self, handoff_id):
        return self.handoffs.begin_restore(handoff_id).state.value

    async def finish_handoff_manifest(self, handoff_id, artifact_id):
        return self.restore.finish(handoff_id, artifact_id).state.value

    async def acquire_handoff_trajectory(self, handoff_id, trajectory_id, owner_id, seconds):
        return self.restore.acquire(handoff_id, trajectory_id, owner_id, seconds)

    async def acquire_handoff_pending(self, handoff_id, trajectory_id, owner_id, seconds):
        return self.restore.acquire_pending(handoff_id, trajectory_id, owner_id, seconds)

    async def record_handoff_member_result(self, handoff_id, index, lease, payload, metrics):
        return self.restore.record_result(handoff_id, index, lease, payload, metrics)

    async def get_handoff_restored_members(self, handoff_id):
        return self.restore.restored(handoff_id)

    async def publish_fixture_group(self, merged, artifact_id="artifact-id-1"):
        manifest = RolloutArtifactManifest(
            artifact_id=artifact_id, producer_id=PRODUCER,
            identity=CanonicalPolicyKVIdentity("0", "weights", "template", "kv"),
            rollout_group_id="group-1", rollout_group_size=1, rollout_sample_index=0,
            trajectory_count=len(merged))
        self.store.publish(manifest, json.dumps(sorted(merged)).encode())
        return artifact_id

    async def takeover_trajectory(self, trajectory_id, expected_generation, owner_id, seconds):
        return self.snapshots.takeover(trajectory_id, expected_generation, owner_id, seconds)

    async def load_trajectory_snapshot(self, lease):
        return self.snapshots.load(lease)

    async def finish_trajectory(self, lease):
        self.snapshots.finish(lease)

    async def mark_trajectory_handed_off(self, trajectory_id):
        return self.tracker.mark_handed_off(trajectory_id).state.value

    async def get_handoff_member_record(self, reference):
        return self.records.get(reference)

    async def discard_trajectory_preemption(self, trajectory_id):
        self.tracker.discard(trajectory_id)


async def fake_resume(member, snapshot):
    """Token re-prefill continuation; committed tools are never re-executed."""
    record = {"member_index": member["member_index"],
              "response_ids": list(snapshot.agent_state["response_ids"]) + [30, 31],
              "tool_calls_executed": 0}
    return record, {"tool_replay_count": 0,
                    "recomputed_prefill_tokens": len(snapshot.agent_state["prompt_ids"])}


async def build_n8_handoff(client: FakeHandoffClient, handoff_id="handoff-n8",
                            tamper_snapshot_member_index=None):
    refs = {}
    for index in (2, 3, 4, 5):
        trajectory_id = f"group-1#{index}"
        stored = tamper_snapshot_member_index if (index == 2 and tamper_snapshot_member_index) else None
        _, key = await client.commit_source_snapshot(
            trajectory_id, index, snapshot_member_index=stored)
        refs[index] = key
    completed_refs = {}
    for index in (0, 1):
        completed_refs[index] = client.records.put(json.dumps({
            "member_index": index, "response_ids": [9, 9], "tool_calls_executed": 1}).encode())

    checkpointed = tuple({
        "trajectory_id": f"group-1#{index}", "group_id": "group-1",
        "member_index": index, "group_size": 8,
        "expected_generation": 0, "snapshot_ref": refs[index],
    } for index in (2, 3, 4, 5))
    completed = tuple({
        "trajectory_id": f"group-1#{index}", "group_id": "group-1",
        "member_index": index, "group_size": 8,
        "record_ref": completed_refs[index],
    } for index in (0, 1))
    pending = tuple({
        "trajectory_id": f"group-1#{index}", "group_id": "group-1",
        "member_index": index, "group_size": 8,
    } for index in (6, 7))
    manifest = HandoffManifest(
        handoff_id=handoff_id, source_rollouter=PRODUCER,
        source_generation=client.generation,
        policy_identity=asdict(CanonicalPolicyKVIdentity("0", "weights", "template", "kv")),
        environment_version=ENV, tool_version=TOOLS,
        completed_artifacts=(), checkpointed_trajectories=checkpointed,
        completed_members=completed, pending_samples=pending,
    )
    client.handoffs.prepare(manifest)
    client.handoffs.commit(handoff_id)
    client.handoffs.record_source_release(handoff_id)
    return refs


@pytest.mark.asyncio
async def test_restore_handoff_n8_two_four_two():
    client = FakeHandoffClient()
    refs = await build_n8_handoff(client)

    outcome, records = await restore_handoff(
        client, "handoff-n8", resume_trajectory=fake_resume)

    assert outcome.completed_members == [0, 1]
    assert outcome.resumed_members == [2, 3, 4, 5]
    assert outcome.pending_members == [6, 7]
    assert sorted(records) == [0, 1, 2, 3, 4, 5]
    # KV Restore = OFF: every checkpointed prompt is re-prefilled token-by-token.
    assert outcome.recomputed_prefill_tokens == 4 * 6
    assert outcome.tool_replay_count == 0
    assert outcome.group_completion_latency_ms is not None
    assert client.handoffs.get("handoff-n8").state.value == "restoring"
    assert len(client.snapshots.latest) == 4  # only actual group publication permits GC
    # Completed members keep their historical execution counts (no replay).
    assert records[0]["tool_calls_executed"] == 1
    assert records[2]["response_ids"] == [10, 11, 30, 31]
    assert records[2]["tool_calls_executed"] == 0


@pytest.mark.asyncio
async def test_restore_rejects_handoff_that_is_not_committed():
    client = FakeHandoffClient()
    # Build an identical-shape draft on a fresh client, but never commit it.
    await build_n8_handoff(client, handoff_id="handoff-draft")
    client.handoffs.records["handoff-draft"].state = client.handoffs.records[
        "handoff-draft"].state.__class__("preparing")
    with pytest.raises(ValueError, match="not one of"):
        client.handoffs.begin_restore("handoff-draft")
    # restore_handoff propagates the same fail-closed rejection.
    with pytest.raises(ValueError, match="not one of"):
        await restore_handoff(client, "handoff-draft", resume_trajectory=fake_resume)


@pytest.mark.asyncio
async def test_late_source_write_fails_closed_after_takeover():
    client = FakeHandoffClient()
    refs = await build_n8_handoff(client)
    # Keep a source-side lease alive by committing one more trajectory outside
    # the manifest, then simulate its takeover by the target.
    trajectory_id = "group-1#2"
    target_lease = client.snapshots.takeover(trajectory_id, 0, "target-rollouter", 600)
    # The source's original lease object is now stale: any late commit raises.
    source_lease = client.snapshots.open("late-write#0", "source-late", 600)
    snapshot = make_snapshot("late-write#0", "group-1", 7, 8, source_lease.generation)
    key = client.snapshots.commit(source_lease, snapshot, 3600)
    client.snapshots.release_owner(source_lease)
    client.snapshots.takeover("late-write#0", source_lease.generation, "target-late", 600)
    stale_snapshot = replace(snapshot, execution_generation=source_lease.generation)
    # A source retry with its revoked lease must fail closed.
    with pytest.raises(ValueError, match="stale or expired"):
        client.snapshots._check(source_lease)
    with pytest.raises(ValueError, match="stale or expired"):
        client.snapshots.commit(source_lease, stale_snapshot, 3600)


@pytest.mark.asyncio
async def test_restore_fails_closed_on_identity_mismatch():
    client = FakeHandoffClient()
    # The manifest says member 2, but the committed snapshot stored member 5.
    await build_n8_handoff(client, handoff_id="handoff-bad",
                           tamper_snapshot_member_index=5)
    with pytest.raises(ValueError, match="identity does not match"):
        await restore_handoff(client, "handoff-bad", resume_trajectory=fake_resume)
    # The failed restore leaves the handoff in RESTORING (never silently FINISHED).
    assert client.handoffs.get("handoff-bad").state.value == "restoring"


@pytest.mark.asyncio
async def test_publish_restored_group_requires_all_eight_members():
    client = FakeHandoffClient()
    await build_n8_handoff(client)
    outcome, records = await restore_handoff(
        client, "handoff-n8", resume_trajectory=fake_resume)
    published = []

    async def publish_group(merged):
        published.append(sorted(merged))
        return await client.publish_fixture_group(merged)

    pending_records = {index: {"member_index": index} for index in (6, 7)}
    await publish_restored_group(client, outcome, records, pending_records,
                                 publish_group=publish_group)
    assert published == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert client.handoffs.get("handoff-n8").state.value == "finished"
    assert not client.snapshots.latest

    with pytest.raises(ValueError, match="does not cover"):
        await publish_restored_group(client, outcome, records,
                                     {6: {"member_index": 6}}, publish_group=publish_group)
    with pytest.raises(ValueError, match="duplicate member"):
        await publish_restored_group(client, outcome, records,
                                     {2: {"member_index": 2}}, publish_group=publish_group)


def main():
    import asyncio

    async def run_all():
        await test_restore_handoff_n8_two_four_two()
        await test_restore_rejects_handoff_that_is_not_committed()
        await test_late_source_write_fails_closed_after_takeover()
        await test_publish_restored_group_requires_all_eight_members()
        print("HANDOFF_EXECUTOR_OK")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
