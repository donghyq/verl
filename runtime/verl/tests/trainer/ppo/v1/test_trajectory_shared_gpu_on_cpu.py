# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Reclamation must settle data publication before physical memory release."""

import asyncio
from types import SimpleNamespace

import pytest
from test_separate_async_step_switch_on_cpu import _trainer
from test_trajectory_checkpoint_integration_on_cpu import ActorHandleShape, manager

from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode
from verl.trainer.ppo.v1.trajectory_handoff_controller import TrajectoryControlError


def reclaim_manager(*, busy=False, failed=False):
    value, events, _ = manager()
    value._trajectory_reserved_target = 1
    source = value.agent_loop_workers[0].target
    source.trajectory_checkpoint_status = lambda: {
        "background_tasks": int(busy), "active_groups": int(busy), "cleanup_pending": 0,
        "group_uids": [], "errors": {"group": "unknown_publication"} if failed else {},
    }
    source.resume_after_trajectory_reclaim = lambda: events.append("source-reopened")
    value.agent_loop_workers.append(ActorHandleShape(SimpleNamespace(
        set_trajectory_checkpoint_paused=lambda paused: events.append(("target-paused", paused)),
        trajectory_checkpoint_status=lambda: {"background_tasks": 0, "active_groups": 0,
                                             "group_uids": [], "cleanup_pending": 0, "errors": {}},
    )))
    return value, events


def test_reclaim_receipt_does_not_reopen_admission_until_physical_reclaim_finishes():
    value, events = reclaim_manager()
    receipt = value.prepare_trajectory_resource_reclaim()
    assert receipt["status"] == "quiesced"
    with pytest.raises(RuntimeError, match="stopped new admission"):
        value.generate_sequences(None)
    with pytest.raises(TrajectoryControlError, match="reclaim_not_quiesced"):
        value.finish_trajectory_resource_reclaim({**receipt, "operation_id": "foreign"})
    value.finish_trajectory_resource_reclaim(receipt)
    assert events == [("paused", True), ("target-paused", True), ("paused", False), ("target-paused", False)]
    with pytest.raises(TrajectoryControlError, match="reclaim_not_quiesced"):
        value.finish_trajectory_resource_reclaim(receipt)


@pytest.mark.parametrize("failed", [False, True])
def test_reclaim_timeout_or_unknown_publication_never_authorizes_memory_release(failed):
    value, _ = reclaim_manager(busy=not failed, failed=failed)
    with pytest.raises(TrajectoryControlError, match="reclaim_worker_failed" if failed else "reclaim_timeout"):
        value.prepare_trajectory_resource_reclaim(timeout_s=.02)
    assert value._trajectory_checkpoint_paused
    with pytest.raises(TrajectoryControlError, match="reclaim_not_quiesced"):
        value.finish_trajectory_resource_reclaim({"status": "quiesced"})


def shared_trainer(*, fault=None):
    value = _trainer(parameter_sync_step=1)
    value.config.actor_rollout_ref.rollout.trajectory_state = {"shared_gpu_handoff": True}
    value.checkpoint_manager.stop_generation_routing = lambda: value.events.append("native-routing-removed")

    def prepare():
        value.events.append("trajectory-settled")
        if fault == "timeout":
            raise TrajectoryControlError("reclaim_timeout")
        return {"status": "unknown" if fault == "unknown" else "quiesced", "receipts": []}

    value.agent_loop_manager = SimpleNamespace(
        prepare_trajectory_resource_reclaim=prepare,
        finish_trajectory_resource_reclaim=lambda receipt: value.events.append("admission-reopened"),
    )
    return value


def test_trainer_sleeps_only_after_confirmed_original_uid_publication():
    value = shared_trainer()
    value.switch_to_trainer()
    assert value.events == ["remove", "native-routing-removed", "trajectory-settled", "hybrid_abort",
                            "hybrid_sleep", "admission-reopened"]
    assert value.current_mode is HybridEngineMode.TRAINER
    assert len(value.trajectory_resource_reclaims) == 1


@pytest.mark.parametrize("fault", ["timeout", "unknown"])
def test_trainer_never_aborts_or_sleeps_after_unconfirmed_handoff(fault):
    value = shared_trainer(fault=fault)
    with pytest.raises((TrajectoryControlError, RuntimeError)):
        value.switch_to_trainer()
    assert value.checkpoint_manager.sleep_calls == value.checkpoint_manager.abort_calls == 0
    assert value.current_mode is HybridEngineMode.ROLLOUT


def test_reserved_takeover_worker_is_excluded_from_normal_admission(monkeypatch):
    import verl.trainer.ppo.v1.agent_loop_tq as module

    value, events = reclaim_manager()
    value.agent_loop_workers[0].target.generate_sequences = lambda batch: events.append("source-admitted")
    value.agent_loop_workers[1].target.generate_sequences = lambda batch: pytest.fail("reserved target admitted")
    monkeypatch.setattr(module.ray, "get", lambda futures: [asyncio.run(future) for future in futures])
    # Record the selected actors without launching Ray or a model.
    class Batch:
        def chunk(self, count):
            assert count == 1
            return [self]

    value.generate_sequences(Batch())
    assert events == ["source-admitted"]


def test_reborrow_installs_current_policy_without_another_optimizer_publication():
    value = shared_trainer()
    value.llm_server_manager = SimpleNamespace(_continuous_publication_client=lambda: "coordinator")
    copies = []
    value.checkpoint_manager.install_same_version_weights = lambda step, client: copies.append((step, client))
    value.switch_to_rollout()
    assert copies == [(value.global_steps, "coordinator")]
    assert value.checkpoint_manager.update_calls == []
    assert value.checkpoint_manager.resume_calls == 1
