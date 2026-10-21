# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGLang request preservation and the shared checkpoint-engine contract.

These are CPU tests of orchestration, not proof of native GPU cache reuse.
"""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("sglang")

import torch

from verl.checkpoint_engine import base
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.sglang_rollout import async_sglang_server
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer, SGLangReplica


@pytest.fixture(autouse=True)
def no_cuda_initialization():
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def _replica():
    server = object.__new__(SGLangHttpServer)
    server.node_rank = 0
    server.tokenizer_manager = SimpleNamespace(pause_generation=AsyncMock(), continue_generation=AsyncMock())
    handle = SimpleNamespace(
        pause_generation=SimpleNamespace(remote=server.pause_generation),
        resume_generation=SimpleNamespace(remote=server.resume_generation),
    )
    replica = object.__new__(SGLangReplica)
    replica.config = SimpleNamespace(
        name="sglang",
        preserve_kv_cache=True,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        disaggregation=SimpleNamespace(enabled=False),
        quantization=SimpleNamespace(enabled=False),
        checkpoint_engine=SimpleNamespace(backend="nccl"),
        engine_kwargs={},
        mtp=None,
    )
    replica.model_config = SimpleNamespace(lora_rank=0, lora={}, hf_config=SimpleNamespace())
    replica.is_multimodal = False
    replica.rollout_mode = RolloutMode.STANDALONE
    replica.workers = [0]
    replica.servers = [handle]
    replica.abort_all_requests = AsyncMock()
    replica.release_kv_cache = AsyncMock()
    replica.resume_kv_cache = AsyncMock()
    return replica, server


@pytest.mark.parametrize(
    "unsupported",
    [
        None,
        "default",
        "hybrid",
        "multimodal",
        "dp",
        "pp",
        "pd",
        "quantized",
        "hf_quantized",
        "engine_quantized",
        "lora",
        "delta",
        "speculative",
    ],
)
def test_shared_weight_sync_lifecycle(monkeypatch, unsupported):
    replica, server = _replica()
    if unsupported == "default":
        replica.config.preserve_kv_cache = False
    elif unsupported == "hybrid":
        replica.rollout_mode = RolloutMode.HYBRID
    elif unsupported == "multimodal":
        replica.is_multimodal = True
    elif unsupported in ("dp", "pp"):
        setattr(replica.config, "data_parallel_size" if unsupported == "dp" else "pipeline_model_parallel_size", 2)
    elif unsupported == "pd":
        replica.config.disaggregation.enabled = True
    elif unsupported == "quantized":
        replica.config.quantization.enabled = True
    elif unsupported == "hf_quantized":
        replica.model_config.hf_config.quantization_config = {"quant_method": "fp8"}
    elif unsupported == "engine_quantized":
        replica.config.engine_kwargs = {"sglang": {"quantization": "fp8"}}
    elif unsupported == "lora":
        replica.model_config.lora = {"rank": 8, "merge": True}
    elif unsupported == "delta":
        replica.config.checkpoint_engine.backend = "nccl_delta"
    elif unsupported == "speculative":
        replica.config.engine_kwargs = {"sglang": {"speculative_algorithm": "EAGLE"}}
    events = []

    class WorkerGroup:
        world_size = 1

        def __init__(self, **kwargs):
            pass

        def update_weights(self, **kwargs):
            events.append(("transfer", kwargs))
            return [{}]

        def execute_checkpoint_engine(self, methods):
            events.append(("finalize", methods))
            return [None]

    monkeypatch.setattr(base, "RayWorkerGroup", WorkerGroup)
    monkeypatch.setattr(base.ray, "get", lambda refs: refs)
    manager = object.__new__(base.CheckpointEngineManager)
    manager.backend = "nccl"
    manager.replicas = [replica]
    manager.actor_wg = WorkerGroup()
    manager.build_process_group = lambda workers: events.append(("prepare", None))
    manager.abort_replicas = AsyncMock(side_effect=replica.abort_all_requests)
    manager.release_kv_cache_replicas = AsyncMock(side_effect=replica.release_kv_cache)
    manager.resume_kv_cache_replicas = AsyncMock(side_effect=replica.resume_kv_cache)
    manager.resume_generation_replicas = replica.resume_generation
    asyncio.run(inspect.unwrap(base.CheckpointEngineManager.update_weights)(manager, global_steps=8))

    receiver_options = next(options for name, options in events if name == "transfer" and "mode" not in options)
    assert receiver_options == (
        {"global_steps": 8, "preserve_kv_cache": True} if unsupported is None else {"global_steps": 8}
    )
    assert [name for name, _ in events] == ["prepare", "transfer", "transfer", "finalize", "finalize"]
    if unsupported is None:
        manager.abort_replicas.assert_not_awaited()
        manager.release_kv_cache_replicas.assert_not_awaited()
        manager.resume_kv_cache_replicas.assert_not_awaited()
        assert server.tokenizer_manager.pause_generation.await_args.args[0].mode == "in_place"
    else:
        manager.abort_replicas.assert_awaited_once()
        manager.release_kv_cache_replicas.assert_awaited_once()
        manager.resume_kv_cache_replicas.assert_awaited_once()
        server.tokenizer_manager.pause_generation.assert_not_awaited()
    server.tokenizer_manager.continue_generation.assert_awaited_once()


def test_non_primary_server_does_not_control_scheduler():
    _, server = _replica()
    server.node_rank = 1
    asyncio.run(server.pause_generation())
    asyncio.run(server.resume_generation())
    server.tokenizer_manager.pause_generation.assert_not_awaited()
    server.tokenizer_manager.continue_generation.assert_not_awaited()


@pytest.mark.parametrize(
    "preserve,mode,prefix,salt,expected",
    [
        (False, "versioned", True, None, None),
        (True, "versioned", True, None, "8"),
        (True, "versioned", True, "", "8"),
        (True, "versioned", True, "task-2", "task-2"),
        (True, "shared", True, "task-2", None),
        (True, "versioned", False, "task-2", None),
    ],
)
def test_request_namespace_and_preserved_output(monkeypatch, preserve, mode, prefix, salt, expected):
    _, server = _replica()
    server._disaggregation_role = "null"
    server._pd_decode_peers = []
    server.global_steps = 8
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    monkeypatch.setattr(
        async_sglang_server.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_actor_name=lambda: "unit-test-server"),
    )
    server.config = SimpleNamespace(
        preserve_kv_cache=preserve,
        prefix_cache_mode=mode,
        enable_prefix_caching=prefix,
        max_model_len=1024,
        prompt_length=32,
        response_length=128,
        enable_rollout_routing_replay=False,
        mtp=None,
    )
    requests = []

    async def generate(request, _):
        requests.append(request)
        server.global_steps = 9
        yield {
            "output_ids": [11, 12, 13],
            "meta_info": {
                "output_token_logprobs": [(-0.1, 11, None), (-0.2, 12, None), (-0.3, 13, None)],
                "finish_reason": {"type": "length"},
            },
        }

    server.tokenizer_manager.generate_request = generate
    output = asyncio.run(server.generate([1, 2], {"max_new_tokens": 3, "logprobs": True}, "rid-1", cache_salt=salt))
    assert getattr(requests[0], "cache_salt", None) == expected
    assert output.token_ids == [11, 12, 13]
    assert output.log_probs == [-0.1, -0.2, -0.3]
    assert output.stop_reason == "length"
    if preserve:
        assert output.extra_fields["min_global_steps"] == 8
        assert output.extra_fields["max_global_steps"] == 9
    else:
        assert output.extra_fields == {"global_steps": 9}
    assert output.extra_fields.get("cache_salt") == expected


@pytest.mark.parametrize("failure", ["prepare", "transfer", "finalize"])
def test_failed_weight_sync_keeps_requests_paused(monkeypatch, failure):
    replica, server = _replica()

    def fail():
        raise RuntimeError("injected weight-sync failure")

    class WorkerGroup:
        world_size = 1

        def __init__(self, **kwargs):
            pass

        def update_weights(self, **kwargs):
            if failure == "transfer":
                fail()
            return [None]

        def execute_checkpoint_engine(self, methods):
            if failure == "finalize":
                fail()
            return [None]

    monkeypatch.setattr(base, "RayWorkerGroup", WorkerGroup)
    monkeypatch.setattr(base.ray, "get", lambda refs: refs)
    manager = object.__new__(base.CheckpointEngineManager)
    manager.backend = "nccl"
    manager.replicas = [replica]
    manager.actor_wg = WorkerGroup()
    manager.build_process_group = lambda workers: fail() if failure == "prepare" else None
    manager.resume_generation_replicas = replica.resume_generation
    with pytest.raises(RuntimeError, match="injected weight-sync failure"):
        asyncio.run(inspect.unwrap(base.CheckpointEngineManager.update_weights)(manager, global_steps=9))
    server.tokenizer_manager.pause_generation.assert_awaited_once()
    server.tokenizer_manager.continue_generation.assert_not_awaited()
    replica.release_kv_cache.assert_not_awaited()
