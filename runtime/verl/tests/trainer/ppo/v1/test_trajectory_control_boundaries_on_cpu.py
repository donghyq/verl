"""Failed verification and capacity pressure cannot report successful recovery."""

import asyncio
from types import SimpleNamespace

import pytest

from verl.trainer.ppo.v1.trajectory_entry import V1TrajectoryEntry
from verl.trainer.ppo.v1.trajectory_handoff_controller import TrajectoryControlError, TrajectoryHandoffController
from verl.trainer.ppo.v1.trajectory_state.coordinator import PublicationBlocked


@pytest.mark.asyncio
async def test_blocked_reconciliation_is_not_successful_takeover(monkeypatch):
    async def summary():
        return {"handoffs": {"handoff": "finished"}}

    async def reconcile(**kwargs):
        return {"state": "BLOCKED", "prompt_ready_verified": False, "readback_verified": False}

    monkeypatch.setattr("verl.trainer.ppo.v1.trajectory_publication.reconcile_trajectory_group_publication", reconcile)
    runtime = SimpleNamespace(client=SimpleNamespace(summary=summary))
    entry = V1TrajectoryEntry(SimpleNamespace(), runtime)
    with pytest.raises(PublicationBlocked, match="did not verify"):
        await entry.resume_handoff("handoff")


@pytest.mark.asyncio
async def test_busy_controller_rejects_before_dispatch_and_retains_unknown_operations():
    requests = []
    release = asyncio.Event()

    async def preempt(uid):
        requests.append(uid)
        return 0

    async def status(uid):
        return {0: {"pending_members": [0], "completed_members": [], "handoff_id": "handoff"}}

    async def handoffs():
        return [{"group_uid": "first", "handoff_id": "handoff", "source_rollouter": "source"}]

    async def resume(handoff_id, *, target_worker_index):
        await release.wait()
        return {"state": "FINISHED", "prompt_ready_verified": True, "readback_verified": True}

    manager = SimpleNamespace(
        request_trajectory_preemption=preempt, trajectory_group_status=status,
        trajectory_coordinator=SimpleNamespace(list_recoverable_handoffs=handoffs),
        agent_loop_workers=[object(), object()], _draining_worker_indices=set(),
        trajectory_worker_owner_ids={"source": 0}, trajectory_status_poll_interval_s=.001,
        resume_trajectory_handoff=resume,
    )
    controller = TrajectoryHandoffController(manager, max_operations=1)
    first = asyncio.create_task(controller.run("first", target_worker_index=None, timeout_s=2))
    try:
        for _ in range(1000):
            if controller.status("first") and controller.status("first")["state"] == "dispatched":
                break
            await asyncio.sleep(.001)
        else:
            raise AssertionError("first operation never dispatched")
        with pytest.raises(TrajectoryControlError) as busy:
            await controller.run("second", target_worker_index=None, timeout_s=1)
        assert busy.value.status == "controller_busy" and requests == ["first"]
        first.cancel()
        with pytest.raises(TrajectoryControlError) as cancelled:
            await first
        assert cancelled.value.status == "cancelled_with_dispatch"
        assert controller.status("first")["state"] == "dispatched"
        release.set()
        result = await controller.run("first", target_worker_index=None, timeout_s=1)
        assert result["status"] == "resumed" and requests == ["first"]
    finally:
        release.set()
        if not first.done():
            first.cancel()
        await asyncio.gather(first, return_exceptions=True)
