"""CPU-only integration-style tests for trajectory publication wiring.

These tests use explicit fixtures for observed engine versions and cache salts.
They do not start Ray services, models, GPU workers, or native SGLang.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from test_trajectory_state_runtime_on_cpu import AsyncClient

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.trainer.ppo.v1.trajectory_policy_publication import (
    TrajectoryPolicyPublicationGatewayHooks,
    canonical_identity_from_runtime,
)
from verl.trainer.ppo.v1.trajectory_state import (
    CanonicalPolicyKVIdentity,
    DurableTrajectoryCoordinator,
    PolicyPublicationBlocked,
)
from verl.workers.rollout.sglang_gateway_client import GatewayProtocolError, SGLangGatewayClient
from verl.workers.rollout.sglang_gateway_lifecycle import TrajectoryAwareGatewayLifecycle

ENGINES = ["http://owned-a:30000", "http://owned-b:30000"]


def runtime_identity(version: int = 0) -> dict[str, str]:
    return {
        "policy_version": str(version),
        "model_weights_fingerprint": "actual-runtime-weight-lineage",
        "tokenizer_template_fingerprint": "actual-tokenizer-template",
        "kv_cache_schema_fingerprint": "actual-kv-schema",
    }


def observe(version: int, *, urls=ENGINES):
    return [{"engine_url": url, "weight_version": str(version), "source": "model_info"} for url in urls]


class NativeOnlyLifecycle:
    def __init__(self, events):
        self.events = events

    async def stop_admission(self):
        self.events.append("stop")

    async def start_admission(self, step):
        self.events.append(("start", step))


class FailingFailLifecycle(NativeOnlyLifecycle):
    def __init__(self, events, hooks):
        super().__init__(events)
        self.hooks = hooks

    async def prepare_policy_publication(self, global_steps):
        return await self.hooks.prepare_policy_publication(global_steps)

    async def mark_policy_update_started(self, ticket):
        await self.hooks.mark_policy_update_started(ticket)

    async def confirm_policy_publication_before_admission(self, ticket):
        return await self.hooks.confirm_policy_publication_before_admission(ticket)

    async def fail_policy_publication(self, ticket, reason):
        await self.hooks.fail_policy_publication(ticket, reason)
        raise RuntimeError("fail record path also failed")


@pytest.mark.asyncio
async def test_checkpoint_manager_calls_prepare_confirm_around_single_weight_update(tmp_path):
    coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run-a")
    try:
        events = []
        hooks = TrajectoryPolicyPublicationGatewayHooks(
            AsyncClient(coordinator),
            run_id="run-a",
            runtime_policy_identity=runtime_identity(),
            model_configuration_fingerprint="actual-model-config",
            publisher_id="checkpoint-owner",
            engine_urls=ENGINES,
            engine_observer=lambda: _async_return(observe(7)),
        )
        lifecycle = TrajectoryAwareGatewayLifecycle(NativeOnlyLifecycle(events), hooks)

        manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
        manager.gateway_control = lifecycle

        async def update(self, global_steps):
            events.append(("update", global_steps))
            with pytest.raises(PolicyPublicationBlocked):
                coordinator.get_published_policy(global_steps)
            return {"bytes": 1}

        manager._update_weights = update.__get__(manager, CheckpointEngineManager)
        result = await manager.update_weights(7)
        payload = {key: value for key, value in result.items() if not key.startswith("timing_s/checkpoint/")}
        assert payload == {"bytes": 1}
        assert all(math.isfinite(value) and value >= 0 for key, value in result.items()
                   if key.startswith("timing_s/checkpoint/"))
        assert events == ["stop", ("update", 7), ("start", 7)]
        record = coordinator.get_published_policy(7)
        assert record["identity"]["policy_version"] == "7"
        assert list(record["engine_observations"]) == observe(7)
    finally:
        coordinator.close()


@pytest.mark.asyncio
async def test_partial_observation_failure_keeps_gateway_closed_and_blocks_unknown_version(tmp_path):
    coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run-b")
    try:
        events = []
        hooks = TrajectoryPolicyPublicationGatewayHooks(
            AsyncClient(coordinator),
            run_id="run-b",
            runtime_policy_identity=runtime_identity(),
            model_configuration_fingerprint="actual-model-config",
            publisher_id="checkpoint-owner",
            engine_urls=ENGINES,
            engine_observer=lambda: _async_return(observe(8, urls=ENGINES[:1])),
        )
        lifecycle = TrajectoryAwareGatewayLifecycle(NativeOnlyLifecycle(events), hooks)
        manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
        manager.gateway_control = lifecycle

        async def update(self, global_steps):
            events.append(("update", global_steps))
            return {}

        manager._update_weights = update.__get__(manager, CheckpointEngineManager)
        with pytest.raises(ValueError):
            await manager.update_weights(8)
        assert events == ["stop", ("update", 8)]
        with pytest.raises(PolicyPublicationBlocked):
            coordinator.get_published_policy(8)
        with pytest.raises(PolicyPublicationBlocked):
            coordinator.get_active_published_policy()
    finally:
        coordinator.close()


@pytest.mark.asyncio
async def test_fail_recording_exception_does_not_hide_original_update_failure(tmp_path):
    coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run-c")
    try:
        events = []
        hooks = TrajectoryPolicyPublicationGatewayHooks(
            AsyncClient(coordinator),
            run_id="run-c",
            runtime_policy_identity=runtime_identity(),
            model_configuration_fingerprint="actual-model-config",
            publisher_id="checkpoint-owner",
            engine_urls=ENGINES,
            engine_observer=lambda: _async_return(observe(9)),
        )
        manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
        manager.gateway_control = FailingFailLifecycle(events, hooks)

        async def update(self, global_steps):
            events.append(("update", global_steps))
            raise RuntimeError("weight transfer failed directly")

        manager._update_weights = update.__get__(manager, CheckpointEngineManager)
        with pytest.raises(RuntimeError, match="weight transfer failed directly"):
            await manager.update_weights(9)
        assert events == ["stop", ("update", 9)]
        state = coordinator.get_policy_coordination_state()
        assert state["publication"]["failure"] == "RuntimeError"
    finally:
        coordinator.close()


def test_policy_cache_salt_is_stable_per_run_and_identity():
    identity1 = CanonicalPolicyKVIdentity("1", "w1", "tok", "kv")
    identity2 = CanonicalPolicyKVIdentity("2", "w2", "tok", "kv")
    same = SGLangGatewayClient.build_policy_cache_salt(
        run_namespace="run-x", policy_version=1, policy_identity_salt=identity1.cache_salt
    )
    same_again = SGLangGatewayClient.build_policy_cache_salt(
        run_namespace="run-x", policy_version="1", policy_identity_salt=identity1.cache_salt
    )
    other_version = SGLangGatewayClient.build_policy_cache_salt(
        run_namespace="run-x", policy_version=2, policy_identity_salt=identity2.cache_salt
    )
    other_run = SGLangGatewayClient.build_policy_cache_salt(
        run_namespace="run-y", policy_version=1, policy_identity_salt=identity1.cache_salt
    )
    assert same == same_again
    assert same != other_version
    assert same != other_run


def test_legacy_numeric_cache_salt_remains_allowed_but_wrong_namespace_is_rejected():
    assert SGLangGatewayClient._compose_cache_salt(4, "4") == "4"
    namespaced = SGLangGatewayClient.build_policy_cache_salt(
        run_namespace="run-z", policy_version=4, policy_identity_salt="digest-4"
    )
    assert SGLangGatewayClient._compose_cache_salt(4, namespaced) == namespaced
    with pytest.raises(GatewayProtocolError, match="cache_salt"):
        SGLangGatewayClient._compose_cache_salt(4, "run-z:5:digest-4")


def test_cross_run_namespaced_salts_differ_even_for_same_version_and_identity():
    digest = CanonicalPolicyKVIdentity("4", "weights-4", "tok", "kv").cache_salt
    salt_a = SGLangGatewayClient.build_policy_cache_salt(
        run_namespace="run-a", policy_version=4, policy_identity_salt=digest
    )
    salt_b = SGLangGatewayClient.build_policy_cache_salt(
        run_namespace="run-b", policy_version=4, policy_identity_salt=digest
    )
    assert salt_a != salt_b


def test_resume_member_uses_snapshot_identity_instead_of_runtime_default():
    from verl.trainer.ppo.v1.trajectory_entry import V1TrajectoryEntry

    calls = []

    class Loop:
        async def resume_from_snapshot(self, snapshot, *, check_owner, execution_generation,
                                       environment_version, tool_version, **kwargs):
            calls.append(kwargs["snapshot_policy_identity"])
            await check_owner()
            return "ok"

    worker = SimpleNamespace(
        _prepare_agent_loop_outputs=lambda *args, **kwargs: _async_return(
            (["prompt_0_0"], SimpleNamespace(batch_size=[1], unbind=lambda dim: []), [{"status": "success"}])
        )
    )
    runtime = SimpleNamespace(policy_identity=runtime_identity(99), environment_version="env", tool_version="tool")
    entry = V1TrajectoryEntry(worker, runtime)
    entry._new_loop = lambda original: Loop()
    entry._prepare_member = lambda i, output, original: _async_return(output)
    snapshot = SimpleNamespace(policy_identity=runtime_identity(3))

    async def owner():
        return None

    import asyncio

    asyncio.run(entry._resume_member(
        0,
        snapshot,
        {
            "uid": "prompt",
            "raw_prompt": [{"role": "user", "content": "x"}],
            "sampling_params": {},
            "trajectory": {"validate": False},
            "__trajectory_group_size__": 8,
            "session_id": 0,
        },
        SimpleNamespace(trajectory_id="prompt/member-0", generation=1),
        owner,
    ))
    assert calls == [runtime_identity(3)]


def test_canonical_identity_builder_keeps_runtime_fingerprints_and_only_replaces_version():
    built = canonical_identity_from_runtime(policy_version=11, runtime_identity=runtime_identity(0))
    assert built == runtime_identity(11)


def test_coordinator_summary_exposes_root_dir_for_foreign_root_rejection(tmp_path):
    coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run-root")
    try:
        summary = coordinator.summary()
        assert summary["run_id"] == "run-root"
        assert summary["root_dir"] == str(tmp_path.resolve())
    finally:
        coordinator.close()


async def _async_return(value):
    return value
