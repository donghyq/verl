# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Private shared-pool ownership and admission guards without GPU execution."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from test_sglang_gateway_lifecycle_on_cpu import Registry, _gateway_factory_config

from verl.workers.rollout.sglang_gateway_lifecycle import NativeGatewayLifecycle, make_gateway_client


@pytest.mark.asyncio
async def test_shared_gateway_removes_only_its_owned_replica_set():
    standalone, hybrid = "http://standalone:30000", "http://hybrid:30000"
    client = Registry([standalone])
    client.config = SimpleNamespace(expected_worker_urls=(standalone, hybrid), allow_partial_worker_set=True)

    async def observe(urls):
        assert tuple(urls) == (hybrid,) and hybrid not in client.workers
        return {hybrid: client.version}

    client.observe_unregistered_engine_versions = observe
    lifecycle = NativeGatewayLifecycle(client, [hybrid])
    await lifecycle.start_admission(7)
    await lifecycle.stop_admission()
    assert set(client.workers) == {standalone}
    assert [event for event in client.events if event[0] == "remove"] == [("remove", hybrid)]


def shared_config():
    config = _gateway_factory_config(trainer_mode="separate_async")
    config.trainer.v1.separate_async.hybrid_rollout.enable_switch = True
    config.trainer.v1.separate_async.parameter_sync_step = 1
    config.actor_rollout_ref.rollout.agent = {"num_workers": 2}
    config.actor_rollout_ref.rollout.trajectory_state = {
        "enabled": True, "continuous_policy_publication": True,
        "shared_gpu_handoff": True, "handoff_target_worker_index": 1,
    }
    return config


def test_shared_gpu_factory_allows_dynamic_membership_only_with_guards():
    client = make_gateway_client(shared_config(), ["standalone:30000", "hybrid:30000"])
    assert client.config.allow_partial_worker_set is True
    assert set(client.config.expected_worker_urls) == {"http://standalone:30000", "http://hybrid:30000"}


@pytest.mark.parametrize("field,value", [
    ("actor_rollout_ref.rollout.trajectory_state.enabled", False),
    ("actor_rollout_ref.rollout.trajectory_state.continuous_policy_publication", False),
    ("actor_rollout_ref.rollout.trajectory_state.handoff_target_worker_index", 2),
    ("trainer.v1.separate_async.parameter_sync_step", 2),
    ("trainer.v1.separate_async.hybrid_rollout.enable_switch", False),
    ("actor_rollout_ref.hybrid_engine", False),
])
def test_shared_gpu_factory_rejects_incomplete_control_configuration(field, value):
    config = shared_config()
    OmegaConf.update(config, field, value)
    with pytest.raises(ValueError, match="guarded separate_async"):
        make_gateway_client(config, ["hybrid:30000"])
