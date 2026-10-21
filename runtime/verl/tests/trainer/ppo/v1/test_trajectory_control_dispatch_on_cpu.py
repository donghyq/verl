"""Actual sync/async API dispatch preserves one background control operation."""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest


def load_module(filename, name):
    root = Path(__file__).parents[4] / "verl/trainer/ppo/v1"
    spec = importlib.util.spec_from_file_location(name, root / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


control = load_module("trajectory_handoff_controller.py", "project_b_control_dispatch_test_controller")
dispatch = load_module("trajectory_control_dispatch.py", "project_b_control_dispatch_test_dispatch")


class Manager:
    def __init__(self):
        self.calls = 0
        self.agent_loop_workers = [object(), object()]
        self._draining_worker_indices = set()
        self.trajectory_worker_owner_ids = {"source": 0}
        self.trajectory_status_poll_interval_s = 0.001
        self.trajectory_coordinator = self
        self.controller = control.TrajectoryHandoffController(self)

    async def request_trajectory_preemption(self, uid):
        return 0

    async def trajectory_group_status(self, uid):
        return {0: {"pending_members": [0], "completed_members": []}}

    async def list_recoverable_handoffs(self):
        return [{"group_uid": "group", "handoff_id": "handoff", "source_rollouter": "source"}]

    async def resume_trajectory_handoff(self, handoff_id, *, target_worker_index):
        self.calls += 1
        await asyncio.sleep(0.08)
        return {"unit_test_result": "finished"}

    @dispatch.trajectory_control_api
    async def run(self, timeout):
        return await self.controller.run("group", target_worker_index=None, timeout_s=timeout)


def test_sync_timeout_does_not_destroy_the_background_loop():
    manager = Manager()
    try:
        with pytest.raises(control.TrajectoryControlError) as unknown:
            manager.run(0.01)
        assert unknown.value.status == "unknown"
        assert manager.run(1)["status"] == "resumed"
        assert manager.calls == 1
    finally:
        manager._trajectory_control_dispatcher.close()


@pytest.mark.asyncio
async def test_async_callers_share_the_same_dispatcher_across_results():
    manager = Manager()
    try:
        with pytest.raises(control.TrajectoryControlError) as unknown:
            await manager.run(0.01)
        assert unknown.value.status == "unknown"
        assert (await manager.run(1))["status"] == "resumed"
        assert manager.calls == 1
    finally:
        manager._trajectory_control_dispatcher.close()
