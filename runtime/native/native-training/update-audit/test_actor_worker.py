import sys
from types import SimpleNamespace

import pytest
import torch
from actor_worker import observe_original_parameters, resolve_training_worker
from training_update_audit import begin_worker_update_audit_step, install_worker_update_audit, read_worker_update_audit


@pytest.mark.parametrize("role", ["actor", "actor_rollout", "actor_rollout_ref"])
def test_actual_optimizer_observer_works_for_all_native_training_roles(tmp_path, role):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    selected = SimpleNamespace(rank=3, actor=SimpleNamespace(engine=SimpleNamespace(optimizer=optimizer)))
    worker = SimpleNamespace(rank=3, worker_dict={role: selected, "ref": object(), "rollout": object()})
    assert resolve_training_worker(worker) is selected
    assert install_worker_update_audit(worker, tmp_path)["rank"] == 3
    begin_worker_update_audit_step(worker, 1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert read_worker_update_audit(worker)["verified_counter_advancements_since_attach"] == 1


@pytest.mark.parametrize("roles", [{}, {"ref": object()}, {"actor": object(), "actor_rollout": object()},
                                  {"actor_rollout": object(), "actor_rollout_ref": object()}])
def test_missing_and_ambiguous_roles_are_rejected(roles):
    with pytest.raises(ValueError, match="exactly one"):
        resolve_training_worker(SimpleNamespace(worker_dict=roles))


def test_rank_mismatch_and_role_without_optimizer_are_rejected():
    with pytest.raises(ValueError, match="ranks differ"):
        resolve_training_worker(SimpleNamespace(rank=1, worker_dict={"actor_rollout": SimpleNamespace(rank=0)}))
    with pytest.raises(ValueError, match="optimizer"):
        resolve_training_worker(SimpleNamespace(worker_dict={"actor_rollout": SimpleNamespace(actor=object())}))


def test_parameter_probe_receives_original_worker_and_result_is_unchanged(monkeypatch):
    selected = SimpleNamespace(rank=2, actor=SimpleNamespace(engine=SimpleNamespace(optimizer=object())))
    result = {"rank": 2, "sample": [1.0, 2.0]}
    calls = []

    def probe(worker):
        calls.append(worker)
        return result

    monkeypatch.setitem(sys.modules, "run_v1_functional_gate", SimpleNamespace(_parameter_probe=probe))
    actual = observe_original_parameters(SimpleNamespace(rank=2, worker_dict={"actor_rollout_ref": selected}))
    assert actual is result and calls == [selected]
