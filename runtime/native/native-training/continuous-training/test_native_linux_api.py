"""Run only in the pinned Linux runtime; imports and schema checks, no models."""

import importlib.metadata
import os

import pytest


@pytest.mark.skipif(os.environ.get("NATIVE_LINUX_API_GATE") != "1", reason="Pinned Linux CPU gate only")
def test_original_native_api_schema_and_audit_serialization(tmp_path):
    import ray.cloudpickle
    from native_training_audit import make_audited_trainer
    from omegaconf import OmegaConf
    from prepare_native_training import prepare_config
    from test_prepare_native_training import original_config
    from training_update_audit import install_worker_update_audit

    from verl.trainer.ppo.v1 import get_trainer_cls
    from verl.utils.config import omega_conf_to_dataclass

    assert importlib.metadata.version("torch") == "2.13.0+cu130"
    assert importlib.metadata.version("tensordict") == "0.10.0"
    assert importlib.metadata.version("transferqueue") == "0.1.10"
    assert importlib.metadata.version("sglang") == "0.5.20"
    config = prepare_config(
        original_config.__wrapped__(tmp_path),
        target=10,
        checkpoint_root=tmp_path / "checkpoint",
        durable_root=tmp_path / "durable",
        run_id="linux-import-only",
        entry_dir=tmp_path / "entry",
        audit_dir=tmp_path / "audit",
    )
    actor = omega_conf_to_dataclass(OmegaConf.create(config["actor_rollout_ref"]["actor"]))
    rollout = omega_conf_to_dataclass(OmegaConf.create(config["actor_rollout_ref"]["rollout"]))
    assert actor.ppo_epochs == 1 and rollout.trajectory_state.shared_gpu_handoff
    base = get_trainer_cls("separate_async")
    cls = make_audited_trainer(
        base,
        rpc=None,
        event=None,
        read_publication=None,
        rank_count=8,
        model_configuration_fingerprint="schema-only",
        begin_audit=None,
        read_audit=None,
    )
    assert cls.fit is base.fit and cls.step is base.step
    restored_cls, restored_function = ray.cloudpickle.loads(ray.cloudpickle.dumps((cls, install_worker_update_audit)))
    assert restored_cls.fit is base.fit and restored_function.__name__ == "install_worker_update_audit"


@pytest.mark.skipif(os.environ.get("NATIVE_LINUX_API_GATE") != "1", reason="Pinned Linux CPU gate only")
@pytest.mark.parametrize("role_name", ["Actor", "ActorRollout", "ActorRolloutRef"])
def test_optimizer_observer_on_real_native_worker_dict_class(tmp_path, role_name):
    from types import SimpleNamespace

    import ray
    import torch
    from actor_worker import resolve_training_worker
    from training_update_audit import (
        begin_worker_update_audit_step,
        install_worker_update_audit,
        read_worker_update_audit,
    )

    from verl.single_controller.ray.base import RayClassWithInitArgs, create_colocated_worker_cls
    from verl.trainer.ppo.utils import Role
    from verl.workers.engine_workers import ActorRolloutRefWorker

    role = str(getattr(Role, role_name))
    worker_type = ActorRolloutRefWorker
    descriptor = create_colocated_worker_cls({role: RayClassWithInitArgs(cls=ray.remote(worker_type))})
    # Instantiate the real outer/inner types without model or distributed setup.
    outer = object.__new__(descriptor.cls.__ray_actor_class__)
    inner = object.__new__(worker_type)
    outer._rank = inner._rank = 3
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    engine = SimpleNamespace(optimizer=optimizer)
    inner.actor = SimpleNamespace(engine=engine)
    outer.worker_dict = {role: inner}
    assert resolve_training_worker(outer) is inner
    assert outer.execute_with_func_generator(install_worker_update_audit, tmp_path)["rank"] == 3
    outer.execute_with_func_generator(begin_worker_update_audit_step, 1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    observed = outer.execute_with_func_generator(read_worker_update_audit)
    assert observed["verified_counter_advancements_since_attach"] == 1
