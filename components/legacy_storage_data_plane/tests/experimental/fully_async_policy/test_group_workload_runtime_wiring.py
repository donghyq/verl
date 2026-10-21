"""Actual routing/worker source contracts, using CPU-only RPC test doubles."""

import asyncio
import json
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop import agent_loop as worker_module
from verl.experimental.fully_async_policy.group_workload_scheduler import (
    GroupRequestContext,
    bind_group_request_context,
    group_request_context,
)
from verl.protocol import DataProto
from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer, LLMServerClient, LLMServerManager
from verl.workers.rollout.replica import TokenOutput


LoadBalancer = GlobalRequestLoadBalancer.__ray_metadata__.modified_class


def request(index=0, size=4, group="g", workload="short"):
    return GroupRequestContext(f"{group}#{index}", group, index, size, workload, 64, 128)


def observations():
    return [
        dict(workload_key=key, replica_id=replica, prompt_tokens=64, max_new_tokens=128,
             generated_tokens=8, elapsed_s=duration, first_token_s=None, tool_time_s=0.0,
             source_id=f"CPU_fixture/{key}/{replica}/{index}")
        for key, duration in (("short", 1.0), ("long", 9.0))
        for replica in ("a", "b") for index in range(3)
    ]


class Immediate:
    def __init__(self, value):
        self.value = value

    def __await__(self):
        async def resolve():
            return self.value
        return resolve().__await__()


class RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return Immediate(self.function(*args, **kwargs))


class Handle:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        return RemoteMethod(getattr(self.target, name))


class Engine:
    def __init__(self):
        self.requests = []

    def generate(self, **kwargs):
        self.requests.append(kwargs)
        return TokenOutput(token_ids=[7, 8], log_probs=[-0.1, -0.2])


def test_default_and_unregistered_context_use_original_routing():
    lb = LoadBalancer(servers={"a": None, "b": None})
    assert not lb.register_group([asdict(request(0, size=1))])
    assert lb.acquire_server("first", group_context=asdict(request()))[0] == "a"
    assert lb.acquire_server("second")[0] == "b"
    assert lb.acquire_server("first")[0] == "a"
    lb.remove_servers(["a"])
    assert lb.acquire_server("first")[0] == "b"
    handles = lb.get_server_handles()
    handles.clear()
    assert lb.get_all_servers() == ["b"]
    assert not lb.complete_trajectory("unregistered", 3)


def test_complete_registration_is_idempotent_but_conflicting_identity_rejected():
    lb = LoadBalancer(servers={"a": None, "b": None}, group_workload_enabled=True)
    contexts = [request(index) for index in range(4)]
    assert lb.register_group([asdict(context) for context in contexts])
    assert lb.register_group(contexts)
    changed = [asdict(context) for context in contexts]
    changed[0]["workload_key"] = "changed"
    with pytest.raises(ValueError, match="conflicting"):
        lb.register_group(changed)
    unknown = request(0, size=1, group="not-registered")
    assert lb.acquire_server("unknown", group_context=asdict(unknown))[0] == "a"


def test_tool_turn_release_retains_forecast_and_owner_removal_fails_safely():
    lb = LoadBalancer(servers={"a": None, "b": None}, group_workload_enabled=True,
                      calibration_observations=observations())
    contexts = [request(index, workload="long" if index == 3 else "short") for index in range(4)]
    lb.register_group(contexts)
    owner = lb.acquire_server("turn-1", group_context=asdict(contexts[0]))[0]
    lb.release_server(owner)
    assert lb.get_inflight_count(owner) == 0
    assert lb._group_scheduler.request_context(contexts[0].trajectory_id) is not None
    assert not lb._group_scheduler.is_settled(contexts[0].trajectory_id)
    assert lb.acquire_server("turn-2", group_context=asdict(contexts[0]))[0] == owner
    lb.release_server(owner)
    lb.remove_servers([owner])
    with pytest.raises(RuntimeError, match="handoff path"):
        lb.acquire_server("turn-3", group_context=asdict(contexts[0]))
    assert lb.cancel_trajectory(contexts[0].trajectory_id)
    assert not lb.cancel_trajectory(contexts[0].trajectory_id)
    cancelled = lb.get_routing_decisions()[0]
    assert cancelled["actual_completion_at"] is None
    assert cancelled["actual_generated_tokens"] is None


def test_native_client_only_sends_group_context_to_router():
    async def run():
        engines = {replica: Engine() for replica in ("a", "b")}
        lb = LoadBalancer(servers={replica: Handle(engine) for replica, engine in engines.items()},
                          group_workload_enabled=True, calibration_observations=observations())
        context = request(0, size=1)
        lb.register_group([context])
        client = LLMServerClient(OmegaConf.create({}), Handle(lb))
        with bind_group_request_context(context):
            for _ in range(2):
                output = await client.generate("logical-session", prompt_ids=[1, 2], sampling_params={"temperature": 0},
                                               allow_remote_cache_restore=False)
                assert output.token_ids == [7, 8]
                assert lb.get_status()["total_inflight"] == 0
                await asyncio.sleep(0)  # real client is not holding a request during tool wait
            assert await client.complete_trajectory(context, generated_tokens=4, tool_time_s=0.01)
            assert not await client.complete_trajectory(context, generated_tokens=4, tool_time_s=0.01)
        assert group_request_context.get() is None
        requests = [item for engine in engines.values() for item in engine.requests]
        assert len(requests) == 2
        assert len([engine for engine in engines.values() if engine.requests]) == 1
        for payload in requests:
            assert not any("group" in key or "context" in key for key in payload)
            assert payload["allow_remote_cache_restore"] is False
        traces = lb.get_routing_decisions()
        assert len(traces) == 2
        assert all(trace["actual_generated_tokens"] == 4 for trace in traces)
        assert all(trace["actual_tool_time_s"] == 0.01 for trace in traces)
    asyncio.run(run())


def test_manager_loads_only_opt_in_independent_calibration(tmp_path):
    manager = object.__new__(LLMServerManager)
    manager.config = OmegaConf.create({})
    assert manager._group_workload_router_options() == {}
    source = tmp_path / "calibration.json"
    source.write_text(json.dumps({"role": "independent_calibration", "observations": observations()}))
    manager.config = OmegaConf.create({"async_training": {"group_workload_scheduling": {
        "enabled": True, "calibration_path": str(source), "min_samples": 3,
    }}})
    assert manager._group_workload_router_options()["calibration_observations"] == observations()
    source.write_text(json.dumps({"role": "acceptance", "observations": observations()}))
    with pytest.raises(ValueError, match="independent_calibration"):
        manager._group_workload_router_options()


def make_worker(monkeypatch, client, *, interruption=None):
    worker = object.__new__(worker_module.AgentLoopWorker)
    worker.config = OmegaConf.create({"data": {}})
    worker.llm_client = client
    worker.tokenizer = worker.processor = worker.dataset_cls = None
    worker.tools = []
    seen = {}

    class Loop:
        async def run(self, sampling, **kwargs):
            seen["context"] = group_request_context.get()
            seen["kwargs"] = kwargs
            if interruption is not None:
                raise interruption
            return worker_module.AgentLoopOutput(prompt_ids=[1], response_ids=[2, 3, 4], response_mask=[1, 0, 1],
                                                 response_logprobs=[-0.1, 0.0, -0.2], metrics={"tool_calls": 0.03})

    monkeypatch.setattr(worker_module, "rollout_trace_attr", lambda **kwargs: nullcontext())
    monkeypatch.setattr(worker_module, "_agent_loop_registry", {"fixture": {"_target_": "fixture"}})
    monkeypatch.setattr(worker_module.hydra.utils, "instantiate", lambda **kwargs: Loop())

    async def postprocess(output, validate, **kwargs):
        seen["postprocess_kwargs"] = kwargs
        return output

    worker._agent_loop_postprocess = postprocess
    return worker, seen


def test_worker_binds_context_and_uses_actual_output_mask_and_tool_timer(monkeypatch):
    callbacks = []

    class Client:
        async def complete_trajectory(self, context, **kwargs):
            callbacks.append((context, kwargs))
        async def cancel_trajectory(self, context):
            pytest.fail("successful trajectory should not cancel")

    worker, seen = make_worker(monkeypatch, Client())
    context = request(0, size=1)
    asyncio.run(worker._run_agent_loop({}, dict(step=0, sample_index=0, rollout_n=0, validate=False),
                                      agent_name="fixture", group_route_context=asdict(context), raw_prompt=[]))
    assert seen["context"] == context
    assert "group_route_context" not in seen["kwargs"]
    assert "group_route_context" not in seen["postprocess_kwargs"]
    assert callbacks == [(context, {"generated_tokens": 2, "tool_time_s": 0.03})]
    assert group_request_context.get() is None


def test_worker_preemption_releases_forecast_and_preserves_exception(monkeypatch):
    callbacks = []

    class Client:
        async def complete_trajectory(self, context, **kwargs):
            pytest.fail("interrupted trajectory is not completed")
        async def cancel_trajectory(self, context):
            callbacks.append(context.trajectory_id)

    error = RuntimeError("CPU_fixture_preemption")
    worker, _ = make_worker(monkeypatch, Client(), interruption=error)
    with pytest.raises(RuntimeError, match="CPU_fixture_preemption"):
        asyncio.run(worker._run_agent_loop({}, dict(step=0, sample_index=0, rollout_n=0, validate=False),
                                          agent_name="fixture", group_route_context=asdict(request(0, size=1)), raw_prompt=[]))
    assert callbacks == ["g#0"]
    assert group_request_context.get() is None


def test_worker_strips_internal_column_before_agent_input_and_training_output(monkeypatch):
    worker = object.__new__(worker_module.AgentLoopWorker)
    worker.rollout_config = SimpleNamespace(temperature=0.7, top_p=1.0, top_k=-1, calculate_log_probs=True,
                                           agent=SimpleNamespace(default_agent_loop="fixture"))
    monkeypatch.setattr(worker_module.RolloutTraceConfig, "get_instance", lambda: SimpleNamespace(max_samples_per_step_per_worker=None))

    async def trajectories(*args):
        return [dict(step=0, sample_index=0, rollout_n=0, validate=False)]
    monkeypatch.setattr(worker_module, "get_trajectory_info", trajectories)
    recorded = []

    async def run(sampling, trajectory, **kwargs):
        recorded.append(kwargs)
        return "CPU_fixture_only"
    worker._run_agent_loop = run
    worker._postprocess = lambda outputs, input_non_tensor_batch, validate: input_non_tensor_batch
    controls = np.empty(1, dtype=object)
    controls[0] = asdict(request(0, size=1))
    prompts = np.empty(1, dtype=object)
    prompts[0] = [{"role": "user", "content": "fixture"}]
    batch = DataProto.from_dict(tensors={"input_ids": torch.tensor([[1, 2]])},
                                non_tensors={"raw_prompt": prompts, "__group_route_context__": controls})
    result = asyncio.run(worker.generate_sequences(batch))
    assert "__group_route_context__" not in result
    assert "__group_route_context__" not in recorded[0]
    assert recorded[0]["group_route_context"] == controls[0]


def test_known_handoff_integer_controls_normalize_numpy_scalars_only():
    worker = object.__new__(worker_module.AgentLoopWorker)
    controls = {
        "__trajectory_sample_index__": [np.int64(2)],
        "__trajectory_group_size__": [np.int32(8)],
        "__trajectory_execution_generation__": [np.uint64(3)],
        "__trajectory_checkpoint_after_batches__": [np.int64(1)],
        "__trajectory_id__": [np.int64(99)],  # unrelated field stays unmodified
    }
    kwargs = {}
    worker._apply_trajectory_handoff_controls(kwargs, controls, 0)
    assert all(type(kwargs[field]) is int for field in (
        "rollout_sample_index", "rollout_group_size", "execution_generation", "_trajectory_checkpoint_after_batches",
    ))
    assert type(kwargs["trajectory_id"]) is np.int64
    controls["__trajectory_execution_generation__"] = [np.float64(3.0)]
    worker._apply_trajectory_handoff_controls(kwargs, controls, 0)
    assert type(kwargs["execution_generation"]) is np.float64
