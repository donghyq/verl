# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Native hook ordering and ActorHandle-shaped RPC regressions; no GPUs."""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from test_trajectory_checkpoint_storage_on_cpu import publish, saved, state

from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trajectory_handoff_controller import TrajectoryControlError, TrajectoryHandoffController
from verl.trainer.ppo.v1.trajectory_state.client import RayTrajectoryCoordinatorClient


class StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class RemoteMethod:
    def __init__(self, method):
        self.method = method

    async def remote(self, *args, **kwargs):
        return self.method(*args, **kwargs)


class ActorHandleShape:
    """Methods must use .remote(), just as the production ActorHandle does."""

    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        return RemoteMethod(getattr(self.target, name))


def manager(*, busy_reads=0, errors=None):
    events, reads = [], []

    def status():
        reads.append(True)
        return {"background_tasks": int(len(reads) <= busy_reads), "active_groups": 0,
                "cleanup_pending": 0, "errors": errors or {}}

    worker = SimpleNamespace(
        set_trajectory_checkpoint_paused=lambda paused: events.append(("paused", paused)),
        trajectory_checkpoint_status=status,
        request_trajectory_preemption=lambda uid: False,
        trajectory_group_status=lambda uid: {"completed_members": list(range(8)), "publication_finished": True},
    )
    coordinator = SimpleNamespace(
        summary=lambda: {"active_execution_leases": 0, "unsettled_publications": [], "handoffs": {},
                         "policy_publication_blocked": False, "handoff_policy_barrier": False},
        list_recoverable_handoffs=lambda: [],
        get_handoff_status=lambda handoff: {"handoff_id": handoff, "source_released": True},
        export_checkpoint=lambda destination, step: events.append(("export", step)) or {"checkpoint_step": step},
    )
    value = AgentLoopManagerTQ.__new__(AgentLoopManagerTQ)
    value.agent_loop_workers = [ActorHandleShape(worker)]
    value.trajectory_coordinator = ActorHandleShape(coordinator)
    value._trajectory_coordinator_client = RayTrajectoryCoordinatorClient(value.trajectory_coordinator)
    value.trajectory_control_timeout_s = 1
    value.trajectory_status_poll_interval_s = .001
    value._draining_worker_indices = set()
    value.trajectory_worker_owner_ids = {}
    value._trajectory_controller = TrajectoryHandoffController(value)
    return value, events, reads


def test_control_calls_actor_facade_and_preserves_raw_handle():
    value, _, _ = manager()
    assert value.preempt_and_resume_trajectory("done")["status"] == "already_completed"
    assert value.get_handoff_status("saved")["source_released"] is True
    assert value.list_recoverable_handoffs() == []
    assert value.trajectory_coordinator_client.actor is value.trajectory_coordinator


def test_checkpoint_waits_for_accepted_work_before_export():
    value, events, reads = manager(busy_reads=2)
    value.prepare_trajectory_checkpoint()
    assert len(reads) == 3 and events == [("paused", True)]
    with pytest.raises(RuntimeError, match="stopped new admission"):
        value.generate_sequences(None)
    with pytest.raises(TrajectoryControlError, match="checkpoint_paused"):
        value.preempt_and_resume_trajectory("new")
    assert value.export_trajectory_checkpoint("fixture", 1) == {"checkpoint_step": 1}
    value.resume_trajectory_checkpoint_admission()
    assert events == [("paused", True), ("export", 1), ("paused", False)]


@pytest.mark.parametrize("fault", ["timeout", "worker"])
def test_checkpoint_cannot_report_success_for_unfinished_or_failed_work(fault):
    value, events, _ = manager(busy_reads=100000 if fault == "timeout" else 0,
                               errors={"uid": "publication_failed"} if fault == "worker" else {})
    with pytest.raises(TrajectoryControlError, match="checkpoint_timeout" if fault == "timeout"
                       else "checkpoint_worker_failed"):
        value.prepare_trajectory_checkpoint(timeout_s=.02)
    assert events == [("paused", True)]
    value.resume_trajectory_checkpoint_admission()
    assert events[-1] == ("paused", False)


def trainer(tmp_path, *, enabled=True):
    value = StubTrainer.__new__(StubTrainer)
    value.trainer_mode, value.parameter_sync_step, value.global_steps = "separate_async", 1, 1
    value.config = OmegaConf.create({
        "trainer": {"resume_mode": "disable", "default_local_dir": str(tmp_path / "native")},
        "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}, "rollout": {
            "trajectory_state": {"enabled": enabled, "continuous_policy_publication": True,
                                 "root_dir": str(tmp_path / "live"), "run_id": "run",
                                 "model_configuration_fingerprint": "fixture-config"}}},
    })
    return value


@pytest.mark.parametrize("save_fails", [False, True])
def test_native_checkpoint_is_after_sidecar_and_admission_is_resumed_on_failure(tmp_path, save_fails):
    value = trainer(tmp_path)
    control, events, _ = manager()
    value.agent_loop_manager = control

    def native():
        assert events == [("paused", True), ("export", 1)]
        events.append("native-model-optimizer-dataloader-tq")
        if save_fails:
            raise OSError("native save failed")

    value._save_checkpoint_native = native
    if save_fails:
        with pytest.raises(OSError, match="native save failed"):
            value._save_checkpoint()
    else:
        value._save_checkpoint()
    assert events == [("paused", True), ("export", 1), "native-model-optimizer-dataloader-tq", ("paused", False)]


def test_sidecar_restores_before_model_boot_into_independent_root(tmp_path):
    coordinator, _, source, _ = saved(tmp_path)
    try:
        value = trainer(tmp_path)
        value.config.trainer.resume_mode = "resume_path"
        value.config.trainer.resume_from_path = str(source.parent)
        events = []
        value._setup = lambda: events.append(("boot", value.config.actor_rollout_ref.rollout.trajectory_state.root_dir))
        value.on_init_end = lambda: events.append("publish-loaded-model")
        value.init()
        restored = value.config.actor_rollout_ref.rollout.trajectory_state.root_dir
        assert restored != str(tmp_path / "live")
        assert value._resolved_checkpoint_folder == str(source.parent)
        assert events == [("boot", restored), "publish-loaded-model"]
    finally:
        coordinator.close()


def test_disabled_checkpoint_path_does_not_pause_or_export(tmp_path):
    value = trainer(tmp_path, enabled=False)
    calls = []
    value._save_checkpoint_native = lambda: calls.append("native")
    value._save_checkpoint()
    assert calls == ["native"]


@pytest.mark.parametrize("fault", ["parameter_sync", "async_save", "missing_sidecar"])
def test_unsupported_or_missing_sidecar_fails_before_model_boot(tmp_path, fault):
    value = trainer(tmp_path)
    calls = []
    value._setup = lambda: calls.append("boot")
    if fault == "parameter_sync":
        value.parameter_sync_step = 2
    elif fault == "async_save":
        value.config.actor_rollout_ref.actor.checkpoint.async_save = True
    else:
        value.config.trainer.resume_mode = "resume_path"
        value.config.trainer.resume_from_path = str(tmp_path / "global_step_1")
    with pytest.raises((ValueError, FileNotFoundError)):
        value.init()
    assert calls == []


def test_completed_eight_member_runtime_is_quiesced_for_export(tmp_path):
    from verl.trainer.ppo.v1.trajectory_state import runtime as runtime_module

    coordinator = state.DurableTrajectoryCoordinator(str(tmp_path / "live"), "run")
    record = publish(coordinator, 0)

    class Client:
        def __getattr__(self, name):
            async def called(*args, **kwargs):
                return getattr(coordinator, name)(*args, **kwargs)
            return called

    async def run():
        client = Client()
        runtime = runtime_module.PromptGroupHandoffRuntime(
            client, owner_id="source", incarnation_id="first", policy_identity=record["identity"],
            environment_version="fixture-env", tool_version="fixture-tool", published_policy_client=client,
        )

        async def member(index, kwargs, lease, commit_snapshot, check_owner):
            await check_owner()
            return {"member": index}

        members = [{"uid": "uid", "session_id": i, "__trajectory_group_size__": 8} for i in range(8)]
        result = await runtime.run_source("uid", members, member)
        # This CPU fixture acknowledges a prompt PUT; it does not stand in for
        # the separate live TransferQueue end-to-end validation.
        await runtime.confirm_normal_publication(result)
        assert result.status == "completed" and coordinator.summary()["active_execution_leases"] == 0
        assert coordinator._state["objects"] == {}
        receipt = coordinator.export_checkpoint(str(tmp_path / "quiesced"), 1)
        assert receipt["quiesced"] is True and receipt["object_keys"] == []

    try:
        asyncio.run(run())
    finally:
        coordinator.close()

