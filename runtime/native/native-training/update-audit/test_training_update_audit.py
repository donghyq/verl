import json
from types import SimpleNamespace

import pytest
import torch
from training_update_audit import (
    OptimizerUpdateAudit,
    begin_worker_update_audit_step,
    install_worker_update_audit,
    read_worker_update_audit,
)


def test_actual_adam_update_is_unchanged_and_skipped_call_is_not_counted(tmp_path):
    original = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    observed = torch.nn.Parameter(original.detach().clone())
    plain = torch.optim.AdamW([original], lr=0.01)
    traced = torch.optim.AdamW([observed], lr=0.01)
    audit = OptimizerUpdateAudit(traced, rank=0, output=tmp_path / "actual.jsonl")
    for step in range(1, 4):
        audit.begin_trainer_step(step)
        # Native finite-gradient gate skips step 2; the observer invents no call.
        if step == 2:
            continue
        original.grad = torch.ones_like(original)
        observed.grad = torch.ones_like(observed)
        plain.step()
        traced.step()
        assert torch.equal(original, observed)
    assert audit.snapshot()["optimizer_calls_since_attach"] == 2
    assert audit.snapshot()["verified_counter_advancements_since_attach"] == 2
    assert [json.loads(line)["trainer_step"] for line in audit.output.read_text().splitlines()] == [1, 3]
    assert audit.snapshot()["parameter_change_verified"] is None


def test_adam_call_without_gradients_does_not_claim_counter_advance(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    audit = OptimizerUpdateAudit(optimizer, rank=0, output=tmp_path / "no-grad.jsonl")
    audit.begin_trainer_step(1)
    optimizer.step()
    row = json.loads(audit.output.read_text())
    assert row["optimizer_calls_since_attach"] == 1
    assert row["optimizer_counter_advanced"] is None
    assert row["verified_counter_advancements_since_attach"] == 0
    assert torch.equal(parameter, torch.tensor([1.0]))


def test_new_process_observer_reads_restored_adam_frontier(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    for _ in range(3):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = torch.optim.AdamW([restored_parameter], lr=0.1)
    restored.load_state_dict(optimizer.state_dict())
    audit = OptimizerUpdateAudit(restored, rank=1, output=tmp_path / "resumed.jsonl")
    assert audit.snapshot()["initial_optimizer_step_frontier"] == 3.0
    audit.begin_trainer_step(4)
    restored_parameter.grad = torch.ones_like(restored_parameter)
    restored.step()
    assert audit.snapshot()["optimizer_calls_since_attach"] == 1
    assert audit.snapshot()["last_optimizer_step_frontier"] == 4.0


def test_receipt_failure_preserves_completed_update(tmp_path, monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    audit = OptimizerUpdateAudit(optimizer, rank=0, output=tmp_path / "io-error.jsonl")
    audit.begin_trainer_step(1)
    parameter.grad = torch.ones_like(parameter)
    monkeypatch.setattr("training_update_audit.os.fsync", lambda _: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError, match="injected"):
        optimizer.step()
    assert not torch.equal(parameter, torch.tensor([1.0]))
    assert audit.snapshot()["optimizer_calls_since_attach"] == 1
    assert audit.snapshot()["verified_counter_advancements_since_attach"] == 1


def test_actor_worker_selection_and_duplicate_installation(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    actor = SimpleNamespace(engine=SimpleNamespace(optimizer=optimizer))
    worker = SimpleNamespace(rank=2, worker_dict={"actor": actor, "other": object()})
    assert install_worker_update_audit(worker, tmp_path)["rank"] == 2
    begin_worker_update_audit_step(worker, 1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert read_worker_update_audit(worker)["optimizer_calls_since_attach"] == 1
    with pytest.raises(RuntimeError, match="already"):
        install_worker_update_audit(worker, tmp_path)


def test_existing_receipt_and_non_monotonic_steps_are_rejected(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    path = tmp_path / "existing.jsonl"
    path.write_text("preserved")
    with pytest.raises(FileExistsError):
        OptimizerUpdateAudit(optimizer, rank=0, output=path)
    audit = OptimizerUpdateAudit(optimizer, rank=0, output=tmp_path / "fresh.jsonl")
    audit.begin_trainer_step(2)
    for step in [1, 2, True]:
        with pytest.raises(ValueError):
            audit.begin_trainer_step(step)
