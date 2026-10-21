# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Control-plane CPU tests with the real coordinator and file object store."""

import asyncio
from dataclasses import asdict

import pytest

from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.trainer.ppo.v1.trajectory_handoff_controller import (
    TrajectoryControlError,
    TrajectoryHandoffController,
)
from verl.trainer.ppo.v1.trajectory_state import (
    CanonicalPolicyKVIdentity,
    DurableTrajectoryCoordinator,
)
from verl.trainer.ppo.v1.trajectory_state.preemptible_tool_agent_loop import TrajectoryCheckpointYield
from verl.trainer.ppo.v1.trajectory_state.runtime import PromptGroupHandoffRuntime

IDENTITY = asdict(CanonicalPolicyKVIdentity("1", "fixture-weight", "fixture-tokenizer", "fixture-kv"))


class AsyncClient:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.loop = asyncio.get_running_loop()

    def __getattr__(self, name):
        async def invoke(*args, **kwargs):
            async def on_owner_loop():
                return getattr(self.coordinator, name)(*args, **kwargs)

            if asyncio.get_running_loop() is self.loop:
                return await on_owner_loop()
            future = asyncio.run_coroutine_threadsafe(on_owner_loop(), self.loop)
            return await asyncio.wrap_future(future)

        return invoke


class RemoteMethod:
    def __init__(self, handler):
        self.handler = handler
        self.loop = asyncio.get_running_loop()

    def remote(self, *args, **kwargs):
        async def invoke():
            result = self.handler(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result

        async def dispatch():
            if asyncio.get_running_loop() is self.loop:
                return await invoke()
            future = asyncio.run_coroutine_threadsafe(invoke(), self.loop)
            return await asyncio.wrap_future(future)

        return dispatch()


class FakeWorker:
    def __init__(self, *, request, status, resume):
        self.request_trajectory_preemption = RemoteMethod(request)
        self.trajectory_group_status = RemoteMethod(status)
        self.resume_trajectory_handoff = RemoteMethod(resume)


def make_runtime(coordinator, *, owner="run/worker-0"):
    return PromptGroupHandoffRuntime(
        AsyncClient(coordinator),
        owner_id=owner,
        incarnation_id=owner + "-incarnation",
        policy_identity=IDENTITY,
        environment_version="env-v1",
        tool_version="tool-v1",
    )


def make_manager(coordinator, workers):
    manager = object.__new__(AgentLoopManagerTQ)
    manager.agent_loop_workers = workers
    manager.trajectory_coordinator = AsyncClient(coordinator)
    manager._draining_worker_indices = set()
    manager.trajectory_worker_owner_ids = {"run/worker-0": 0, "run/worker-1": 1}
    manager.trajectory_control_timeout_s = 2.0
    manager.trajectory_status_poll_interval_s = 0.001
    manager._trajectory_controller = TrajectoryHandoffController(manager)
    return manager


def input_kwargs(n):
    return [
        {
            "sampling_params": {"temperature": 0.7, "logprobs": True},
            "trajectory": {"original_index": i},
            "agent_name": "durable_tool_agent",
            "raw_prompt": [{"role": "user", "content": f"member {i}"}],
        }
        for i in range(n)
    ]


def snapshot(uid, index, n, lease, kwargs):
    # Reuse the canonical supported text/tool safe-point fixture already used
    # by runtime tests; do not construct a weaker incomplete snapshot.
    from test_trajectory_state_runtime_on_cpu import snapshot as runtime_snapshot

    return runtime_snapshot(uid, index, n, lease, kwargs)


async def wait_until(predicate):
    for _ in range(2000):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition did not arrive")


async def start_mixed_source(runtime, *, uid="group"):
    release = asyncio.Event()

    async def run_member(index, kwargs, lease, commit_snapshot, check_owner):
        await check_owner()
        if index < 2:
            return {"member_index": index}
        await release.wait()
        value = snapshot(uid, index, 8, lease, kwargs)
        reference = await commit_snapshot(value)
        assert reference is not None
        raise TrajectoryCheckpointYield(value, reference)

    task = asyncio.create_task(runtime.run_source(uid, input_kwargs(8), run_member, start_members=list(range(6))))
    await wait_until(
        lambda: runtime.group_status(uid) is not None and runtime.group_status(uid)["completed_members"] == [0, 1]
    )
    return task, release


async def complete_normal_source(runtime, *, uid="normal"):
    async def complete_member(index, kwargs, lease, commit, check):
        await check()
        return {"member_index": index}

    members = [{**value, "uid": uid, "session_id": index} for index, value in enumerate(input_kwargs(8))]
    return await runtime.run_source(uid, members, complete_member, start_members=[0])


@pytest.mark.asyncio
async def test_preempt_resume_waits_for_committed_released_metadata_and_auto_selects_target(tmp_path):
    async def scenario():
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            source = make_runtime(coordinator)
            source_task, release = await start_mixed_source(source)
            resume_calls = []

            async def request(uid):
                return bool(await source.request_preemption(uid))

            async def resume(handoff_id):
                resume_calls.append(handoff_id)
                return {"queue_publication": handoff_id}

            workers = [
                FakeWorker(request=request, status=source.group_status, resume=resume),
                FakeWorker(request=lambda uid: False, status=lambda uid: None, resume=resume),
            ]
            manager = make_manager(coordinator, workers)

            with pytest.raises(TrajectoryControlError) as timeout:
                await manager.preempt_and_resume_trajectory("group", timeout_s=0.03)
            assert timeout.value.status == "timeout" and resume_calls == []
            assert coordinator.list_recoverable_handoffs() == []

            release.set()
            await source_task
            handoff_id = coordinator.list_recoverable_handoffs()[0]["handoff_id"]
            status = await manager.get_handoff_status(handoff_id)
            assert status["state"] == "committed" and status["source_released"] is True
            assert status["group_uid"] == "group" and "owner_lease" not in status

            result = await manager.preempt_and_resume_trajectory("group", timeout_s=2)
            assert result["status"] == "resumed" and result["source_worker_index"] == 0
            assert result["target_worker_index"] == 1 and result["receipt"] == {
                "queue_publication": result["handoff_id"]
            }
            assert resume_calls == [result["handoff_id"]]
        finally:
            coordinator.close()

    await scenario()


@pytest.mark.asyncio
async def test_unknown_timeout_is_not_retried_and_repeated_call_shares_result(tmp_path):
    async def scenario():
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            source = make_runtime(coordinator)
            source_task, release = await start_mixed_source(source)
            resume_calls = []
            target_release = asyncio.Event()

            async def request(uid):
                return bool(await source.request_preemption(uid))

            async def resume(handoff_id):
                resume_calls.append(handoff_id)
                await target_release.wait()
                return {"queue_publication": handoff_id}

            async def release_source():
                await wait_until(lambda: source.group_status("group")["preemption_requested"])
                release.set()
                await source_task

            workers = [
                FakeWorker(request=request, status=source.group_status, resume=resume),
                FakeWorker(request=lambda uid: False, status=lambda uid: None, resume=resume),
            ]
            manager = make_manager(coordinator, workers)
            release_task = asyncio.create_task(release_source())
            with pytest.raises(TrajectoryControlError) as first_unknown:
                await manager.preempt_and_resume_trajectory("group", timeout_s=2)
            assert first_unknown.value.status == "unknown"
            await release_task

            for _ in range(2):
                with pytest.raises(TrajectoryControlError) as unknown:
                    await manager.preempt_and_resume_trajectory("group", timeout_s=0.02)
                assert unknown.value.status == "unknown"
            assert len(resume_calls) == 1
            assert (await manager.trajectory_operation_status("group"))["state"] == "dispatched"

            target_release.set()
            result = await manager.preempt_and_resume_trajectory("group", timeout_s=2)
            assert result["receipt"] == {"queue_publication": result["handoff_id"]}
            assert len(resume_calls) == 1
            assert (await manager.trajectory_operation_status("group"))["state"] == "completed"
        finally:
            coordinator.close()

    await scenario()


@pytest.mark.asyncio
async def test_terminal_control_outcomes_are_explicit(tmp_path):
    async def scenario():
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            source = make_runtime(coordinator)
            source_task, release = await start_mixed_source(source)
            resume_calls = []

            async def request(uid):
                return bool(await source.request_preemption(uid))

            async def resume(handoff_id):
                resume_calls.append(handoff_id)
                return {"queue_publication": handoff_id}

            workers = [
                FakeWorker(request=request, status=source.group_status, resume=resume),
                FakeWorker(request=lambda uid: False, status=lambda uid: None, resume=resume),
            ]
            manager = make_manager(coordinator, workers)
            await source.request_preemption("group")
            release.set()
            await source_task
            manager._draining_worker_indices.add(1)
            with pytest.raises(TrajectoryControlError) as no_target:
                await manager.preempt_and_resume_trajectory("group", timeout_s=2)
            assert no_target.value.status == "no_target" and resume_calls == []

            normal = make_runtime(coordinator, owner="run/worker-1")

            async def complete_member(index, kwargs, lease, commit, check):
                await check()
                return {"member_index": index}

            normal_result = await complete_normal_source(normal)
            workers[0].trajectory_group_status = RemoteMethod(
                lambda uid: normal.group_status(uid) if uid == "normal" else None
            )
            workers[0].request_trajectory_preemption = RemoteMethod(lambda uid: False)
            manager._draining_worker_indices.clear()
            assert normal.group_status("normal")["publication_finished"] is False
            assert TrajectoryHandoffController._normal_completion(normal.group_status("normal")) is False
            # This is the minimal stand-in for a successful original-UID
            # finished PUT; the test does not fabricate a TQ/GPU receipt.
            assert await normal.confirm_normal_publication(normal_result) is True
            result = await manager.preempt_and_resume_trajectory("normal", timeout_s=2)
            assert result["status"] == "already_completed" and result["handoff_id"] is None

            with pytest.raises(TrajectoryControlError) as missing:
                await manager.preempt_and_resume_trajectory("absent", timeout_s=2)
            assert missing.value.status == "uid_not_found"
        finally:
            coordinator.close()

    await scenario()


@pytest.mark.asyncio
async def test_normal_cleanup_waits_for_publication_and_removes_only_verified_source(tmp_path):
    async def scenario():
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            runtime = make_runtime(coordinator)
            result = await complete_normal_source(runtime)
            source_refs = set(result.input_record_refs.values()) | set(result.output_record_refs.values())
            unrelated = coordinator.put_handoff_member_record(b'{"unrelated":"retained object"}')
            assert len(source_refs) == 16 and coordinator._state["gc_pending"] == []
            assert runtime.group_status("normal")["publication_finished"] is False

            assert (
                coordinator.queue_completed_group_cleanup(
                    "normal",
                    runtime.owner_id,
                    result.source_generation,
                    result.input_record_refs,
                    result.output_record_refs,
                )
                == 16
            )
            assert set(coordinator._state["gc_pending"]) == source_refs
            assert all(reference in coordinator._state["objects"] for reference in source_refs)

            assert await runtime.confirm_normal_publication(result) is True
            assert runtime.group_status("normal")["publication_finished"] is True
            assert coordinator._state["gc_pending"] == []
            assert all(reference not in coordinator._state["objects"] for reference in source_refs)
            assert unrelated in coordinator._state["objects"]
        finally:
            coordinator.close()

    await scenario()


@pytest.mark.asyncio
async def test_normal_cleanup_rejects_wrong_identity_stale_generation_unreleased_owner_and_pin(tmp_path, monkeypatch):
    async def scenario():
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            runtime = make_runtime(coordinator)
            result = await complete_normal_source(runtime)
            args = (runtime.owner_id, result.source_generation, result.input_record_refs, result.output_record_refs)

            with pytest.raises(ValueError, match="not released by this source"):
                coordinator.queue_completed_group_cleanup("other", *args)
            with pytest.raises(ValueError, match="no longer current"):
                coordinator.queue_completed_group_cleanup(
                    "normal",
                    runtime.owner_id,
                    result.source_generation + 1,
                    result.input_record_refs,
                    result.output_record_refs,
                )

            coordinator._transaction(lambda draft: draft["trajectories"]["normal/member-0"].pop("released", None))
            with pytest.raises(ValueError, match="not released"):
                coordinator.queue_completed_group_cleanup("normal", *args)
            coordinator._transaction(lambda draft: draft["trajectories"]["normal/member-0"].update({"released": True}))

            # Inject the storage pin guard. A fully completed normal group
            # cannot itself form a valid preemptible manifest, so do not
            # fabricate one as proof of a real handoff.
            pinned_reference = result.output_record_refs[0]
            monkeypatch.setattr(coordinator, "_reference_is_pinned", lambda reference: reference == pinned_reference)
            with pytest.raises(ValueError, match="pinned"):
                coordinator.queue_completed_group_cleanup("normal", *args)
            assert coordinator._state["gc_pending"] == []
        finally:
            coordinator.close()

    await scenario()


@pytest.mark.asyncio
async def test_startup_recovery_requires_published_engine_identity(tmp_path):
    async def scenario():
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            source = make_runtime(coordinator)
            source_task, release = await start_mixed_source(source)
            await source.request_preemption("group")
            release.set()
            await source_task
            resume_calls = []

            async def resume(handoff_id):
                resume_calls.append(handoff_id)
                return {"queue_publication": handoff_id}

            workers = [
                FakeWorker(request=lambda uid: False, status=lambda uid: None, resume=resume),
                FakeWorker(request=lambda uid: False, status=lambda uid: None, resume=resume),
            ]
            manager = make_manager(coordinator, workers)
            with pytest.raises(TrajectoryControlError) as blocked:
                await manager.recover_trajectory_handoffs(timeout_s=2)
            assert blocked.value.status == "policy_not_published" and resume_calls == []

            async def published_policy():
                return {"policy_version": "1", "engine_observation": "fixture-published"}

            manager.trajectory_coordinator.get_active_published_policy = published_policy
            results = await manager.recover_trajectory_handoffs(timeout_s=2)
            assert len(results) == 1 and results[0]["status"] == "resumed"
            assert results[0]["source_worker_index"] is None and results[0]["target_worker_index"] == 0
            assert len(resume_calls) == 1
        finally:
            coordinator.close()

    await scenario()
