from __future__ import annotations

import asyncio
from types import SimpleNamespace


def test_partial_rollout_records_policy_boundary_at_each_generation_result(monkeypatch):
    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerClient
    from verl.workers.rollout.llm_server import LLMServerClient
    from verl.workers.rollout.replica import TokenOutput

    outputs = iter([
        TokenOutput(token_ids=[10, 11], log_probs=[-0.1, -0.2], stop_reason="aborted", extra_fields={"global_steps": 41}),
        TokenOutput(token_ids=[12], log_probs=[-0.3], stop_reason="completed", extra_fields={"global_steps": 42}),
    ])
    prompts = []

    async def fake_generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        prompts.append(list(prompt_ids))
        return next(outputs)

    async def no_sleep(_):
        return None

    monkeypatch.setattr(LLMServerClient, "generate", fake_generate)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    client = object.__new__(FullyAsyncLLMServerClient)
    client.config = SimpleNamespace(async_training=SimpleNamespace(partial_rollout=True))
    result = asyncio.run(client.generate(
        "request-1", prompt_ids=[1, 2], sampling_params={"max_new_tokens": 3, "temperature": 0.7}
    ))
    assert prompts == [[1, 2], [1, 2, 10, 11]]
    assert result.token_ids == [10, 11, 12]
    assert result.log_probs == [-0.1, -0.2, -0.3]
    assert [(item["start"], item["end"], item["policy_version"]) for item in result.extra_fields["trajectory_segments"]] == [
        (0, 2, "41"), (2, 3, "42")
    ]
    assert result.extra_fields["trajectory_segments"][0]["sampling_config_fingerprint"] != result.extra_fields["trajectory_segments"][1]["sampling_config_fingerprint"]


def test_sglang_same_owner_pause_uses_backend_in_place_mode(monkeypatch):
    server_module = _import_async_sglang_server_module(monkeypatch)
    seen = []

    class PauseRequest:
        def __init__(self, mode):
            self.mode = mode

    async def pause(request):
        seen.append(request.mode)

    monkeypatch.setattr(server_module, "PauseGenerationReqInput", PauseRequest)
    server = object.__new__(server_module.SGLangHttpServer)
    server.tokenizer_manager = SimpleNamespace(pause_generation=pause)
    asyncio.run(server._pause_generation_mode("preserve_kv"))
    asyncio.run(server._pause_generation_mode("retract"))
    assert seen == ["in_place", "retract"]


def test_rollouter_and_sglang_server_runtime_identity_use_the_same_builder(monkeypatch):
    from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter as FullyAsyncRollouterActor

    tokenizer = SimpleNamespace(chat_template="template-a")
    model_config = SimpleNamespace(
        path="/models/qwen", tokenizer_path="/models/qwen", tokenizer=tokenizer
    )
    rollout_config = SimpleNamespace(name="sglang", tensor_model_parallel_size=2)
    sentinel = object()
    calls = []

    def fake_builder(policy_version, *, model_config, rollout_config, tokenizer):
        calls.append(
            {
                "policy_version": policy_version,
                "model_config": model_config,
                "rollout_config": rollout_config,
                "tokenizer": tokenizer,
            }
        )
        return sentinel

    rollouter = object.__new__(FullyAsyncRollouterActor.__ray_actor_class__)
    rollouter.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(model=model_config, rollout=rollout_config)
    )
    rollouter.tokenizer = tokenizer

    import verl.experimental.fully_async_policy.fully_async_rollouter as rollouter_module

    monkeypatch.setattr(
        rollouter_module, "build_fully_async_policy_kv_identity", fake_builder
    )
    assert rollouter._policy_kv_identity(7) is sentinel

    server_module = _import_async_sglang_server_module(monkeypatch)
    monkeypatch.setattr(
        server_module, "build_fully_async_policy_kv_identity", fake_builder
    )
    server = object.__new__(server_module.SGLangHttpServer)
    server.config = rollout_config
    server.model_config = model_config
    server.global_steps = 0
    asyncio.run(server.set_global_steps(7))

    assert server.policy_kv_identity is sentinel
    assert len(calls) == 2
    assert all(call["policy_version"] == 7 for call in calls)
    assert all(call["model_config"] is model_config for call in calls)
    assert all(call["rollout_config"] is rollout_config for call in calls)
    assert all(call["tokenizer"] is tokenizer for call in calls)


def _import_async_sglang_server_module(monkeypatch):
    import sys
    import types

    def module(name):
        value = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    sglang = module("sglang")
    sglang.__version__ = "0.0.0-test"
    srt = module("sglang.srt")
    sglang.srt = srt
    entrypoints = module("sglang.srt.entrypoints")
    srt.entrypoints = entrypoints
    engine = module("sglang.srt.entrypoints.engine")
    entrypoints.engine = engine
    http_server = module("sglang.srt.entrypoints.http_server")
    entrypoints.http_server = http_server
    http_server.ServerArgs = type("ServerArgs", (), {})
    http_server._GlobalState = type("_GlobalState", (), {})
    http_server.app = object()
    http_server.set_global_state = lambda *args, **kwargs: None

    managers = module("sglang.srt.managers")
    srt.managers = managers
    io_struct = module("sglang.srt.managers.io_struct")
    managers.io_struct = io_struct
    for class_name in (
        "ContinueGenerationReqInput",
        "GenerateReqInput",
        "PauseGenerationReqInput",
        "ReleaseMemoryOccupationReqInput",
        "ResumeMemoryOccupationReqInput",
    ):
        setattr(io_struct, class_name, type(class_name, (), {}))
    tokenizer_manager = module("sglang.srt.managers.tokenizer_manager")
    managers.tokenizer_manager = tokenizer_manager
    tokenizer_manager.ServerStatus = type("ServerStatus", (), {})

    sglang_rollout = module("verl.workers.rollout.sglang_rollout.sglang_rollout")
    sglang_rollout._set_envs_and_config = lambda *args, **kwargs: None

    import verl.workers.rollout.sglang_rollout.async_sglang_server as server_module

    return server_module
