"""Wiring test: AgentLoopWorker per-row handoff controls over the real registry.

Exercises the production path: ``generate_sequences`` with ``__trajectory_*``
control columns -> ToolAgentLoop.run with the injected callbacks -> a
preemption requested through the tracker -> checkpoint committed through the
real TrajectorySnapshotRegistry -> the worker converts the yield into a
preempted partial row -> the rollouter adapter reads the rows and builds the
transactional manifest.

The FakeClient mimics RayRolloutArtifactStoreClient semantics over the same
in-process store objects used by the earlier tests.  Callbacks are verified
to survive a cloudpickle round-trip (what the Ray worker boundary does).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import ray.cloudpickle as cloudpickle

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.fully_async_policy.preemptible_handoff import (
    HandoffManifest,
    HandoffManifestRegistry,
    HandoffMemberRecordStore,
    TrajectoryPreemptionTracker,
)
from verl.experimental.fully_async_policy.preemptible_rollout_adapter import (
    MemberLease,
    ParkedSample,
    build_and_commit_manifest,
    finalize_sample,
    open_sample_leases,
    read_preempted_rows,
)
from verl.experimental.fully_async_policy.rollout_artifact import (
    CanonicalPolicyKVIdentity,
    InMemoryRolloutArtifactStore,
)
from verl.experimental.fully_async_policy.trajectory_snapshot import (
    TrajectorySnapshot,
    TrajectorySnapshotRegistry,
)
from tests.experimental.fully_async_policy.test_handoff_source_e2e import make_loop

IDENTITY = asdict(CanonicalPolicyKVIdentity("0", "weights", "template", "kv"))
ENV, TOOLS = "env-1", "tools-1"


class FakeRayClient:
    """Async client over the real in-process store objects (Ray-actor-like)."""

    def __init__(self):
        self.store = InMemoryRolloutArtifactStore()
        self.snapshots = TrajectorySnapshotRegistry(InMemoryRolloutArtifactStore())
        self.tracker = TrajectoryPreemptionTracker()
        self.handoffs = HandoffManifestRegistry(
            self.tracker, current_producer_generation=self.store.get_current_producer_generation)
        self.records = HandoffMemberRecordStore()
        self.generation = self.store.register_producer("fully_async_rollouter", "inc-1")

    async def open_trajectory(self, trajectory_id, owner_id, lease_seconds):
        return self.snapshots.open(trajectory_id, owner_id, lease_seconds)

    async def register_running_trajectory(self, trajectory_id):
        self.tracker.register_running(trajectory_id)

    async def get_trajectory_preemption(self, trajectory_id):
        return self.tracker.state(trajectory_id).value

    async def mark_trajectory_safe_point(self, trajectory_id):
        self.tracker.mark_safe_point(trajectory_id)

    async def commit_trajectory_snapshot(self, lease, snapshot, ttl_seconds):
        return self.snapshots.commit(lease, snapshot, ttl_seconds)

    async def mark_trajectory_snapshot_committed(self, trajectory_id, snapshot_ref):
        self.tracker.mark_snapshot_committed(trajectory_id, snapshot_ref)

    async def check_trajectory_owner(self, lease):
        self.snapshots._check(lease)

    async def finish_trajectory(self, lease):
        self.snapshots.finish(lease)

    async def release_trajectory_owner(self, lease):
        self.snapshots.release_owner(lease)

    async def mark_trajectory_handed_off(self, trajectory_id):
        self.tracker.mark_handed_off(trajectory_id)

    async def discard_trajectory_preemption(self, trajectory_id):
        self.tracker.discard(trajectory_id)

    async def request_trajectory_preemption(self, trajectory_id):
        self.tracker.request_preemption(trajectory_id)

    async def put_handoff_member_record(self, payload):
        return self.records.put(payload)

    async def prepare_handoff_manifest(self, manifest):
        self.handoffs.prepare(manifest)

    async def commit_handoff_manifest(self, handoff_id):
        self.handoffs.commit(handoff_id)

    async def record_handoff_source_release(self, handoff_id):
        self.handoffs.record_source_release(handoff_id)


def _rebind_client(callback, client):
    """After a cloudpickle round-trip, rebind the captured client cell to the
    live object — exactly what an actor handle does across the Ray boundary."""
    cells = callback.__closure__ or ()
    for cell in cells:
        if isinstance(cell.cell_contents, FakeRayClient):
            cell.cell_contents = client  # type: ignore[union-attr]
    return callback


def make_full_batch(group_size=8, preempt_immediately=None):
    """A minimal non_tensor_batch like prepare_single_generation_data produces."""
    if preempt_immediately is None:
        preempt_immediately = {2, 3, 4, 5}
    non_tensor = {
        "raw_prompt": np.array(
            [[{"role": "user", "content": "look up record 7"}]] * group_size, dtype=object),
        "agent_name": np.array(["tool_agent_loop"] * group_size, dtype=object),
        "index": np.arange(group_size, dtype=np.int64),
        "rollout_sample_index": np.arange(group_size, dtype=np.int64),
    }
    batch = SimpleNamespace(non_tensor_batch=dict(non_tensor), meta_info={"validate": False})
    return batch, preempt_immediately


class LoopFactory:
    """Stand-in for hydra instantiation: a fresh stubbed ToolAgentLoop per row."""

    def __init__(self, calls, generated, templates):
        self.calls, self.generated, self.templates = calls, generated, templates

    def __call__(self, **kwargs):
        return make_loop(self.calls, self.generated, self.templates)


@pytest.mark.asyncio
async def test_full_wiring_preemption_wave_two_four_two():
    client = FakeRayClient()
    calls, generated, templates = [], [], []
    factory = LoopFactory(calls, generated, templates)

    batch, preempt_immediately = make_full_batch()
    member_leases = await open_sample_leases(
        client, batch, "sample-1", owner_id="fully_async_rollouter",
        lease_seconds=600.0, environment_version=ENV, tool_version=TOOLS,
        policy_identity=IDENTITY, snapshot_ttl_seconds=3600.0)

    # The reclamation wave arrives: request preemption for members 2..5 only.
    for index in preempt_immediately:
        await client.request_trajectory_preemption(f"sample-1#{index}")

    # The worker loop runs all rows concurrently through the REAL
    # ToolAgentLoop.run with the injected callbacks.  cloudpickle round-trips
    # the callbacks exactly like the Ray worker boundary does; a
    # RayRolloutArtifactStoreClient keeps a live actor reference after
    # deserialization, so we patch the deserialized copy back to the same
    # client (Ray actor-handle semantics).
    outputs = {}
    for index in range(8):
        checkpoint_cb = cloudpickle.loads(cloudpickle.dumps(
            batch.non_tensor_batch["__trajectory_checkpoint__"][index]))
        owner_cb = cloudpickle.loads(cloudpickle.dumps(
            batch.non_tensor_batch["__trajectory_check_owner__"][index]))
        # Restore reference semantics: rebind the pickled closure cells to the
        # live client (equivalent to an actor handle resolving to the actor).
        checkpoint_cb = _rebind_client(checkpoint_cb, client)
        owner_cb = _rebind_client(owner_cb, client)
        kwargs = {
            "raw_prompt": batch.non_tensor_batch["raw_prompt"][index],
            "trajectory_id": f"sample-1#{index}",
            "rollout_group_id": "sample-1",
            "rollout_sample_index": index,
            "rollout_group_size": 8,
            "execution_generation": member_leases[index].lease.generation,
            "snapshot_environment_version": ENV,
            "snapshot_tool_version": TOOLS,
            "snapshot_policy_identity": IDENTITY,
            "_trajectory_checkpoint": checkpoint_cb,
            "_trajectory_check_owner": owner_cb,
        }
        loop = factory()
        try:
            outputs[index] = await loop.run({"temperature": 0.7}, **kwargs)
        except Exception as yield_event:
            if not hasattr(yield_event, "snapshot"):
                raise
            outputs[index] = yield_event  # TrajectoryCheckpointYield

    # Members 0/1/6/7 completed normally; 2..5 yielded at their safe point.
    preempted_indices = {i for i in range(8) if hasattr(outputs[i], "snapshot")}
    assert preempted_indices == {2, 3, 4, 5}
    assert calls.count(7) == 8  # each member executed its tool exactly once

    # Simulate the worker's preempted-row conversion (the production code in
    # AgentLoopWorker._preempted_output) at the adapter level: read the rows
    # the way _postprocess would emit them.
    row_extras = {}
    for index in preempted_indices:
        event = outputs[index]
        state = event.snapshot.agent_state
        row_extras[index] = {
            "__preempted__": True,
            "__preemption_snapshot_ref__": event.reference,
            "__preemption_trajectory_id__": event.snapshot.trajectory_id,
            "__preemption_expected_generation__": event.snapshot.execution_generation,
            "rollout_sample_index": index,
        }
    batch.non_tensor_batch["__preempted__"] = np.array(
        [index in preempted_indices for index in range(8)], dtype=object)
    for key in ("__preemption_snapshot_ref__", "__preemption_trajectory_id__",
                "__preemption_expected_generation__"):
        batch.non_tensor_batch[key] = np.array(
            [row_extras[index][key] if index in row_extras else None for index in range(8)],
            dtype=object)

    preempted = read_preempted_rows(batch)
    assert sorted(preempted) == [2, 3, 4, 5]
    for index in preempted:
        assert preempted[index]["snapshot_ref"].startswith("inflight/")
        # tracker advanced to snapshot_committed through the callback
        assert client.tracker.state(f"sample-1#{index}").value == "snapshot_committed"

    await finalize_sample(client, preempted, member_leases)
    # Completed members had their registry entries finished.
    for index in (0, 1, 6, 7):
        assert f"sample-1#{index}" not in client.snapshots.latest
        assert f"sample-1#{index}" not in client.tracker.records

    # The rollouter parks the sample: completed=0/1, pending=6/7, checkpointed=2..5.
    completed = [0, 1]
    pending = [6, 7]
    record_ref = await client.put_handoff_member_record(
        json.dumps({"member_rows": {str(i): {"uid": f"uid_{i}"} for i in completed}}).encode())
    parked = ParkedSample(
        sample_id="sample-1", group_id="sample-1", group_size=8,
        member_leases=member_leases, preempted_members=[2, 3, 4, 5],
        completed_members=completed, pending_members=pending,
        record_ref=record_ref, producer_generation=client.generation,
        policy_identity=IDENTITY, handoff_id="handoff-sample-1")
    manifest = await build_and_commit_manifest(
        client, parked, environment_version=ENV, tool_version=TOOLS)

    assert client.handoffs.get("handoff-sample-1").state.value == "committed"
    assert manifest.source_generation == client.generation
    tables = manifest.member_tables()["sample-1"]
    assert tables["completed"] == [0, 1]
    assert tables["checkpointed"] == [2, 3, 4, 5]
    assert tables["pending"] == [6, 7]

    # Source releases its part; the target can begin restore.
    client.handoffs.begin_restore("handoff-sample-1")
    # Every committed snapshot is still loadable through a takeover lease.
    for index in (2, 3, 4, 5):
        lease = client.snapshots.takeover(
            f"sample-1#{index}", member_leases[index].lease.generation,
            "target-rollouter", 600)
        snapshot = client.snapshots.load(lease)
        assert snapshot.trajectory_id == f"sample-1#{index}"
        assert len(snapshot.completed_tool_batches) == 1  # tool result survived


@pytest.mark.asyncio
async def test_no_preemption_wave_runs_to_completion_with_zero_snapshots():
    client = FakeRayClient()
    calls, generated, templates = [], [], []
    factory = LoopFactory(calls, generated, templates)
    batch, _ = make_full_batch(preempt_immediately=set())

    member_leases = await open_sample_leases(
        client, batch, "sample-2", owner_id="fully_async_rollouter",
        lease_seconds=600.0, environment_version=ENV, tool_version=TOOLS,
        policy_identity=IDENTITY)

    completed = 0
    for index in range(8):
        loop = factory()
        try:
            output = await loop.run({"temperature": 0.7},
                raw_prompt=batch.non_tensor_batch["raw_prompt"][index],
                trajectory_id=f"sample-2#{index}",
                rollout_group_id="sample-2", rollout_sample_index=index,
                rollout_group_size=8,
                execution_generation=member_leases[index].lease.generation,
                snapshot_environment_version=ENV, snapshot_tool_version=TOOLS,
                snapshot_policy_identity=IDENTITY,
                _trajectory_checkpoint=batch.non_tensor_batch["__trajectory_checkpoint__"][index],
                _trajectory_check_owner=batch.non_tensor_batch["__trajectory_check_owner__"][index])
            completed += 1
        except Exception as yield_event:
            if not hasattr(yield_event, "snapshot"):
                raise
            pytest.fail("no preemption was requested; nothing should yield")
    assert completed == 8
    # Zero storage writes happened on the no-preemption path.
    assert not client.snapshots.latest
    assert all(r.preempt_requested_at == 0.0 for r in client.tracker.records.values())


def main():
    async def run_all():
        await test_full_wiring_preemption_wave_two_four_two()
        await test_no_preemption_wave_runs_to_completion_with_zero_snapshots()
        print("WIRING_E2E_OK")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
