from types import SimpleNamespace

import pytest
import torch
from native_training_audit import learning_signal_summary, make_audited_trainer, verify_publication, verify_rank_update
from training_update_audit import begin_worker_update_audit_step, install_worker_update_audit, read_worker_update_audit


def row(rank=0, calls=0, verified=0, step=None, pid=1):
    return dict(
        rank=rank,
        pid=pid,
        trainer_step=step,
        optimizer_calls_since_attach=calls,
        verified_counter_advancements_since_attach=verified,
    )


def test_missing_rank_cannot_be_counted_as_a_distributed_update():
    with pytest.raises(ValueError, match="Every original"):
        verify_rank_update([row()], [row(calls=1, verified=1, step=1)], rank_count=8, step=1)


def test_optimizer_call_without_state_advancement_is_not_verified():
    with pytest.raises(ValueError, match="counter advancement"):
        verify_rank_update([row()], [row(calls=1, step=1)], rank_count=1, step=1)


def test_process_restart_between_observations_is_rejected():
    with pytest.raises(ValueError, match="process or step"):
        verify_rank_update([row()], [row(calls=1, verified=1, step=1, pid=2)], rank_count=1, step=1)


def test_stale_actual_engine_version_is_rejected():
    with pytest.raises(ValueError, match="Native engines"):
        verify_publication({"actual_parameter_version": 0}, 1, "model")


def build_trainer(tmp_path, *, skipped=False, fail_publication=False, read_signal=None):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    worker = SimpleNamespace(rank=0, engine=SimpleNamespace(optimizer=optimizer))
    events = []

    class Original:
        def _compute_advantage(self, batch, metrics):
            self.original_advantage_calls = getattr(self, "original_advantage_calls", 0) + 1
            return batch

        def _update_actor(self, batch, metrics):
            del batch
            if not skipped:
                (parameter**2).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            metrics["actor/grad_norm"] = 2.0 if not skipped else 0.0
            metrics["actor/pg_loss"] = 0.0
            return "original-result"

        def on_step_end(self):
            self.original_publications += 1
            if fail_publication:
                raise RuntimeError("native publication failed")

    def publication(trainer):
        return {
            "actual_parameter_version": trainer.global_steps,
            "active_policy_record": {
                "identity": {"policy_version": str(trainer.global_steps)},
                "model_configuration_fingerprint": "model",
            },
        }

    cls = make_audited_trainer(
        Original,
        rpc=lambda trainer, func, *args: [func(worker, *args)],
        event=lambda name, **fields: events.append((name, fields)),
        read_publication=publication,
        rank_count=1,
        model_configuration_fingerprint="model",
        begin_audit=begin_worker_update_audit_step,
        read_audit=read_worker_update_audit,
        read_learning_signal=read_signal,
    )
    trainer = cls()
    trainer.global_steps = 1
    trainer.audit_update_steps, trainer.audit_verified_steps, trainer.audit_publication_steps = [], [], []
    trainer.original_publications = 0
    install_worker_update_audit(worker, tmp_path)
    return trainer, worker, parameter, events


def test_original_torch_update_and_publication_are_observed_once(tmp_path):
    trainer, worker, parameter, events = build_trainer(tmp_path)
    assert trainer._update_actor(None, {}) == "original-result"
    trainer.on_step_end()
    assert parameter.item() < 1.0
    assert trainer.audit_verified_steps == trainer.audit_publication_steps == [1]
    assert trainer.original_publications == 1
    assert read_worker_update_audit(worker)["optimizer_calls_since_attach"] == 1
    assert [name for name, fields in events][-1] == "native_policy_publication_verified"


def test_no_gradient_cannot_pass_and_must_not_repeat_update(tmp_path):
    trainer, worker, parameter, events = build_trainer(tmp_path, skipped=True)
    with pytest.raises(ValueError, match="counter advancement"):
        trainer._update_actor(None, {})
    with pytest.raises(RuntimeError, match="Never repeat"):
        trainer._update_actor(None, {})
    assert read_worker_update_audit(worker)["optimizer_calls_since_attach"] == 1
    assert parameter.item() == 1.0
    assert trainer.audit_verified_steps == []


def test_publication_failure_preserves_completed_update(tmp_path):
    trainer, worker, parameter, events = build_trainer(tmp_path, fail_publication=True)
    trainer._update_actor(None, {})
    with pytest.raises(RuntimeError, match="native publication"):
        trainer.on_step_end()
    assert trainer.audit_verified_steps == [1] and trainer.audit_publication_steps == []
    assert read_worker_update_audit(worker)["verified_counter_advancements_since_attach"] == 1


def signal_fixture():
    keys = ["prompt42_session0_0", "prompt42_session1_0", "prompt73_session0_0", "prompt73_session1_0"]
    data = {"uid": torch.tensor([42, 42, 73, 73]), "response_mask": torch.ones((4, 2)),
            "advantages": torch.tensor([[-1., -1.], [1., 1.], [0., 0.], [0., 0.]]),
            "rm_scores": torch.tensor([[0., 0.], [0., 1.], [0., 2.], [0., 2.]]),
            "responses": torch.tensor([[11, 12], [11, 13], [15, 16], [15, 16]])}
    return data, keys


def test_original_group_rewards_and_effective_advantages_are_observed_without_modification():
    data, keys = signal_fixture()
    before = {name: value.clone() for name, value in data.items()}
    report = learning_signal_summary(data, keys, group_size=2, group_count=2)
    assert report["nonzero_training_advantage"] is True and report["nonzero_advantage_fraction"] == 0.5
    assert report["groups"] == [
        {"members": 2, "final_response_unique_count": 2, "reward_std": 0.5},
        {"members": 2, "final_response_unique_count": 1, "reward_std": 0.0}]
    assert all(torch.equal(before[name], value) for name, value in data.items())


def test_zero_advantage_is_not_a_learning_signal_even_when_adamw_changes_weights():
    data, keys = signal_fixture()
    data["advantages"].zero_()
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.1)
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert parameter.item() < 1. and optimizer.state[parameter]["step"] == 1
    report = learning_signal_summary(data, keys, group_size=2, group_count=2)
    assert report["nonzero_training_advantage"] is False and report["advantage_l2"] == 0


def test_masked_advantages_do_not_count_as_effective_training_signal():
    data, keys = signal_fixture()
    data["response_mask"][:, 0] = 0
    data["advantages"][:, 0] = float("nan")
    report = learning_signal_summary(data, keys, group_size=2, group_count=2)
    assert report["effective_tokens"] == 4 and report["nonzero_advantage_tokens"] == 2


def test_incomplete_prompt_group_is_rejected_before_an_update_can_be_dispatched():
    data, keys = signal_fixture()
    data = {name: value[:3] for name, value in data.items()}
    with pytest.raises(ValueError, match="complete"):
        learning_signal_summary(data, keys[:3], group_size=2, group_count=2)


def test_nonfinite_effective_advantage_is_rejected():
    data, keys = signal_fixture()
    data["advantages"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        learning_signal_summary(data, keys, group_size=2, group_count=2)


def test_reward_variance_uses_final_session_and_does_not_double_count_partial_outputs():
    data, keys = signal_fixture()
    data = {name: torch.cat((value[:1], value)) for name, value in data.items()}
    data["rm_scores"][0, -1] = 99
    keys = ["prompt42_session0_-1", *keys]
    report = learning_signal_summary(data, keys, group_size=2, group_count=2)
    assert report["complete_prompt_groups"] == 2 and report["groups"][0]["reward_std"] == 0.5


@pytest.mark.parametrize("nonzero", [False, True])
def test_native_advantage_and_update_are_called_once_and_learning_is_counted_separately(tmp_path, nonzero):
    data, keys = signal_fixture()
    if not nonzero:
        data["advantages"].zero_()
    trainer, worker, parameter, events = build_trainer(
        tmp_path, read_signal=lambda trainer, batch: learning_signal_summary(
            data, batch.keys, group_size=2, group_count=2
        )
    )
    batch = SimpleNamespace(keys=keys)
    assert trainer._compute_advantage(batch, {}) is batch
    assert trainer._update_actor(batch, {}) == "original-result"
    assert trainer.original_advantage_calls == 1
    assert read_worker_update_audit(worker)["optimizer_calls_since_attach"] == 1
    proof = next(fields for name, fields in events if name == "native_actor_update_verified")
    assert proof["learning_signal_verified"] is nonzero
    assert proof["policy_loss"] == 0.0
    assert getattr(trainer, "audit_learning_update_steps", []) == ([1] if nonzero else [])


def test_changed_batch_after_observation_is_rejected_before_optimizer_dispatch(tmp_path):
    data, keys = signal_fixture()
    trainer, worker, parameter, events = build_trainer(
        tmp_path, read_signal=lambda trainer, batch: learning_signal_summary(
            data, batch.keys, group_size=2, group_count=2
        )
    )
    trainer._compute_advantage(SimpleNamespace(keys=keys), {})
    with pytest.raises(ValueError, match="batch differs"):
        trainer._update_actor(SimpleNamespace(keys=[*keys, "other_session0_0"]), {})
    assert read_worker_update_audit(worker)["optimizer_calls_since_attach"] == 0
