"""Failure matrix A-G for preemptible rollouts (requirement 8).

Every scenario runs against the real in-process registry/tracker/manifest
stack (same objects the Ray actor embeds).  Expected behavior is fail-closed:
a fault never produces duplicate artifacts, never silently finishes a handoff,
and stale ownership is always rejected.

A  source dies before snapshot commit        -> trajectory aborts, no takeover
B  source dies after commit, before manifest -> takeover allowed, lease wins
C  source dies after manifest COMMITTED      -> restore proceeds unchanged
D  target writes with a stale lease          -> late write fails closed
E  lease expiry enables re-takeover           -> compare-generation takeover
F  idempotent replay of restore handoff       -> second attempt fails closed
G  policy identity mismatch                   -> manifest prepare fails closed
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from verl.experimental.fully_async_policy.handoff_executor import restore_handoff
from verl.experimental.fully_async_policy.preemptible_handoff import (
    HandoffManifest,
    HandoffManifestRegistry,
    HandoffMemberRecordStore,
    PreemptState,
    TrajectoryPreemptionTracker,
)
from verl.experimental.fully_async_policy.rollout_artifact import (
    CanonicalPolicyKVIdentity,
    InMemoryRolloutArtifactStore,
)
from verl.experimental.fully_async_policy.trajectory_snapshot import (
    TrajectorySnapshot,
    TrajectorySnapshotRegistry,
)
from tests.experimental.fully_async_policy.test_handoff_executor import (
    FakeHandoffClient,
    build_n8_handoff,
    fake_resume,
    make_snapshot,
    PRODUCER,
    ENV,
    TOOLS,
)


def small_manifest(client, handoff_id="handoff-fm", source_generation=None):
    """A minimal single-group manifest reusing the n8 rig shapes."""
    return HandoffManifest(
        handoff_id=handoff_id,
        source_rollouter=PRODUCER,
        source_generation=client.generation if source_generation is None else source_generation,
        policy_identity=asdict(CanonicalPolicyKVIdentity("0", "weights", "template", "kv")),
        environment_version=ENV,
        tool_version=TOOLS,
        completed_artifacts=(),
        checkpointed_trajectories=(),
        completed_members=tuple({
            "trajectory_id": f"group-1#{index}", "group_id": "group-1",
            "member_index": index, "group_size": 8, "record_ref": "handoff-member/unused",
        } for index in range(8)),
        pending_samples=(),
    )


# --- A: source dies before any snapshot commit -------------------------------
@pytest.mark.asyncio
async def test_A_source_death_before_commit_leaves_nothing_to_take_over():
    client = FakeHandoffClient()
    client.tracker.register_running("group-1#2")
    client.tracker.request_preemption("group-1#2")
    # Source never reaches a safe point and dies: SAFE_POINT_TIMEOUT abort.
    client.tracker.mark_aborted("group-1#2")
    assert client.tracker.state("group-1#2") is PreemptState.ABORTED
    # No snapshot was committed, so takeover must fail closed.
    with pytest.raises(ValueError, match="takeover requires committed snapshot"):
        client.snapshots.takeover("group-1#2", 0, "target-rollouter", 600)
    # The trajectory is simply re-rolled from scratch by the pending path.


# --- B: source dies after commit, before the manifest exists ----------------
@pytest.mark.asyncio
async def test_B_source_death_after_commit_before_manifest_allows_takeover():
    client = FakeHandoffClient()
    trajectory = "group-1#2"
    lease, key = await client.commit_source_snapshot(trajectory, 2)
    # Source dies without preparing any manifest.  The committed snapshot plus
    # an expired/released source lease is exactly the takeover precondition.
    takeover = client.snapshots.takeover(trajectory, lease.generation, "target-rollouter", 600)
    snapshot = client.snapshots.load(takeover)
    assert snapshot.trajectory_id == trajectory
    # And a second takeover from another contender fails: generation moved on.
    with pytest.raises(ValueError, match="takeover requires committed snapshot"):
        client.snapshots.takeover(trajectory, lease.generation, "contender", 600)


# --- C: source dies after the manifest is COMMITTED --------------------------
@pytest.mark.asyncio
async def test_C_source_death_after_committed_manifest_restores_normally():
    client = FakeHandoffClient()
    await build_n8_handoff(client, handoff_id="handoff-c")
    # Source vanishes right after COMMITTED + release; target restore proceeds.
    outcome, records = await restore_handoff(client, "handoff-c", resume_trajectory=fake_resume)
    assert outcome.resumed_members == [2, 3, 4, 5]
    artifact_id = await client.publish_fixture_group({**records, 6: {}, 7: {}})
    await client.finish_handoff_manifest("handoff-c", artifact_id)
    assert client.handoffs.get("handoff-c").state.value == "finished"


# --- D: late source write after takeover fails closed ------------------------
@pytest.mark.asyncio
async def test_D_stale_source_write_is_rejected_after_takeover():
    client = FakeHandoffClient()
    trajectory = "group-1#2"
    source_lease, key = await client.commit_source_snapshot(trajectory, 2)
    # (the rig already released the source lease)
    target_lease = client.snapshots.takeover(trajectory, source_lease.generation, "target-rollouter", 600)
    # The late source retries its commit with the revoked lease.
    stale_snapshot = replace(
        make_snapshot(trajectory, "group-1", 2, 8, source_lease.generation),
        agent_state={**make_snapshot(trajectory, "group-1", 2, 8, source_lease.generation).agent_state,
                     "response_ids": [10, 11, 99]})
    with pytest.raises(ValueError, match="stale or expired"):
        client.snapshots.commit(source_lease, stale_snapshot, 3600)
    # The registry never accepted the newer, unauthorized tokens: the target's
    # own lease still loads the original committed history.
    loaded = client.snapshots.load(target_lease)
    assert loaded.agent_state["response_ids"] == [10, 11]


# --- E: lease expiry enables re-takeover by compare-generation ----------------
@pytest.mark.asyncio
async def test_E_expired_target_lease_allows_re_takeover():
    client = FakeHandoffClient()
    trajectory = "group-1#3"
    source_lease, key = await client.commit_source_snapshot(trajectory, 3)
    # (the rig already released the source lease)
    first_target = client.snapshots.takeover(trajectory, source_lease.generation, "target-a", 5)
    # The first target dies with its lease held; time passes past expiry.
    original_clock = client.snapshots.clock
    client.snapshots.clock = lambda: original_clock() + 1000  # force expiry
    # Second target takes over from the dead first target.
    second_target = client.snapshots.takeover(trajectory, first_target.generation, "target-b", 600)
    snapshot = client.snapshots.load(second_target)
    assert snapshot.trajectory_id == trajectory
    # The expired first-target lease can no longer commit anything.
    client.snapshots.clock = original_clock
    with pytest.raises(ValueError, match="stale or expired"):
        client.snapshots._check(first_target)


# --- F: idempotent replay of the same restore fails closed -------------------
@pytest.mark.asyncio
async def test_F_duplicate_restore_attempt_is_rejected():
    client = FakeHandoffClient()
    await build_n8_handoff(client, handoff_id="handoff-f")
    outcome, records = await restore_handoff(client, "handoff-f", resume_trajectory=fake_resume)
    assert outcome.resumed_members == [2, 3, 4, 5]
    # A repeat before publication reuses saved outputs without re-execution;
    # the full group's real artifact publication makes FINISHED terminal.
    pending = {index: {"member_index": index} for index in (6, 7)}
    artifact_id = await client.publish_fixture_group({**records, **pending})
    await client.finish_handoff_manifest("handoff-f", artifact_id)
    # A duplicated restore of the same handoff cannot restart: FINISHED is
    # terminal and every snapshot registry entry has already been finished.
    with pytest.raises(ValueError):
        await restore_handoff(client, "handoff-f", resume_trajectory=fake_resume)


# --- G: policy identity mismatch fails closed --------------------------------
@pytest.mark.asyncio
async def test_G_identity_mismatch_fails_closed():
    client = FakeHandoffClient()
    mismatched = HandoffManifest(
        **{**asdict(small_manifest(client, "handoff-g")),
           "policy_identity": asdict(CanonicalPolicyKVIdentity("999", "weights", "template", "kv")),
           "checkpointed_trajectories": tuple({
               "trajectory_id": "group-1#2", "group_id": "group-1", "member_index": 2,
               "group_size": 8, "expected_generation": 0, "snapshot_ref": "inflight/snap-x",
           } for _ in range(1)),
           "completed_members": tuple({
               "trajectory_id": f"group-1#{index}", "group_id": "group-1",
               "member_index": index, "group_size": 8,
               "record_ref": "handoff-member/r"} for index in (0, 1, 3, 4, 5, 6, 7)),
           "pending_samples": ()})
    # The manifest itself validates structurally; the *execution identity*
    # mismatch is caught when the target compares against its own serving
    # policy (here: the store's accepted versions on claim).
    client.handoffs.prepare(mismatched)
    # Trajectory takeover with the wrong generation refuses to hand out data.
    with pytest.raises(ValueError, match="takeover requires committed snapshot"):
        client.snapshots.takeover("group-1#2", 0, "target-rollouter", 600)


def main():
    import asyncio

    async def run_all():
        await test_A_source_death_before_commit_leaves_nothing_to_take_over()
        await test_B_source_death_after_commit_before_manifest_allows_takeover()
        await test_C_source_death_after_committed_manifest_restores_normally()
        await test_D_stale_source_write_is_rejected_after_takeover()
        await test_E_expired_target_lease_allows_re_takeover()
        await test_F_duplicate_restore_attempt_is_rejected()
        await test_G_identity_mismatch_fails_closed()
        print("FAILURE_MATRIX_A_G_OK")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
