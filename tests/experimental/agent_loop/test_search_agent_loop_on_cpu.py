from __future__ import annotations

import json
from typing import Any, Optional

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.search_agent_loop import SearchAgentLoop
from verl.experimental.agent_loop.search_environment import InMemorySearchAdapter, RecordingSearchAdapter, load_snapshot_store
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class _FakeTokenizer:
    padding_side = "right"
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools=None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        del tools, add_generation_prompt, tokenize, kwargs
        content = " ".join(str(message.get("content", "")) for message in messages)
        return self.encode(content, add_special_tokens=False)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(i) for i in ids)


class _FakeServerManager:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompt_snapshots: list[list[int]] = []

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        del request_id, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs
        self.prompt_snapshots.append(list(prompt_ids))
        text = self.responses.pop(0)
        token_ids = [ord(ch) for ch in text]
        return TokenOutput(token_ids=token_ids, log_probs=[-0.1] * len(token_ids), extra_fields={})


def _config(**search_agent_overrides):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 256,
                    "response_length": 256,
                    "multi_turn": {"tool_config_path": None},
                    "custom": {
                        "search_agent": {
                            "max_turns": 4,
                            "max_searches": 2,
                            "max_total_tokens": 512,
                            "max_observation_tokens": 64,
                            "max_query_length": 128,
                            "min_top_k": 1,
                            "max_top_k": 5,
                            "allowed_recall_profiles": ["mock"],
                            "search_timeout_s": 0.01,
                            **search_agent_overrides,
                        }
                    },
                },
                "model": {},
            },
            "data": {
                "tool_config_path": None,
                "apply_chat_template_kwargs": {},
            },
        }
    )


def _build_loop(*, responses: list[str], adapter: Optional[InMemorySearchAdapter] = None, **overrides) -> SearchAgentLoop:
    config = _config(**overrides)
    tokenizer = _FakeTokenizer()
    server_manager = _FakeServerManager(responses)
    loop = SearchAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
        search_adapter=adapter,
    )
    loop._fake_server = server_manager
    return loop


def _adapter(**kwargs) -> InMemorySearchAdapter:
    docs = [
        {"doc_id": "doc-1", "content": "VeRL supports search agents.", "metadata": {"topic": "verl"}},
        {"doc_id": "doc-2", "content": "response_mask uses 0 for environment tokens.", "metadata": {"topic": "mask"}},
    ]
    return InMemorySearchAdapter(docs, **kwargs)


@pytest.mark.asyncio
async def test_single_turn_direct_answer() -> None:
    loop = _build_loop(responses=["<answer>done</answer>"])
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], trajectory_id="traj-1")
    assert output.extra_fields["stop_reason"] == "answer"
    assert all(mask == 1 for mask in output.response_mask)


@pytest.mark.asyncio
async def test_search_then_answer() -> None:
    loop = _build_loop(
        responses=[
            '<search>{"query":"mask","top_k":1,"recall_profile":"mock"}</search>',
            "<answer>doc-2 explains it</answer>",
        ],
        adapter=_adapter(),
    )
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "what is response_mask"}], trajectory_id="traj-2")
    assert output.extra_fields["stop_reason"] == "answer"
    assert output.extra_fields["num_searches"] == 1
    assert 0 in output.response_mask
    assert 1 in output.response_mask
    assert len(output.response_ids) == len(output.response_mask)
    assert len(output.extra_fields["search_trace_summary"]) == 1
    event = output.extra_fields["search_trace_summary"][0]
    assert event["doc_ids"] == ["doc-2"]


@pytest.mark.asyncio
async def test_multiple_searches_then_answer() -> None:
    loop = _build_loop(
        responses=[
            '<search>{"query":"verl","top_k":1,"recall_profile":"mock"}</search>',
            '<search>{"query":"mask","top_k":1,"recall_profile":"mock"}</search>',
            "<answer>doc-1 and doc-2</answer>",
        ],
        adapter=_adapter(),
    )
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "use search twice"}], trajectory_id="traj-3")
    assert output.extra_fields["num_searches"] == 2
    assert output.extra_fields["stop_reason"] == "answer"


@pytest.mark.asyncio
async def test_generated_tokens_not_decode_reencode_for_actions() -> None:
    loop = _build_loop(responses=["<answer>token-identity</answer>"])
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], trajectory_id="traj-4")
    expected_ids = [ord(ch) for ch in "<answer>token-identity</answer>"]
    assert output.response_ids[: len(expected_ids)] == expected_ids


@pytest.mark.asyncio
async def test_environment_token_mask_zero_and_ranges() -> None:
    loop = _build_loop(
        responses=['<search>{"query":"mask","top_k":1,"recall_profile":"mock"}</search>', '<answer>done</answer>'],
        adapter=_adapter(),
    )
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "test"}], trajectory_id="traj-5")
    env_ranges = output.extra_fields["environment_token_ranges"]
    assert env_ranges
    env_range = env_ranges[0]
    assert all(output.response_mask[i] == 0 for i in range(env_range["start"], env_range["end"]))


@pytest.mark.asyncio
async def test_llm_token_mask_one() -> None:
    loop = _build_loop(responses=["<answer>done</answer>"])
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], trajectory_id="traj-6")
    action_range = output.extra_fields["action_token_ranges"][0]
    assert all(output.response_mask[i] == 1 for i in range(action_range["start"], action_range["end"]))


@pytest.mark.asyncio
async def test_malformed_search_action_fallback() -> None:
    loop = _build_loop(responses=["<search>{bad-json}</search>", "<answer>done</answer>"], adapter=_adapter())
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], trajectory_id="traj-7")
    assert output.extra_fields["invalid_action_count"] == 1
    summary = output.extra_fields["search_trace_summary"][0]
    assert summary["error_type"] == "malformed_payload"


@pytest.mark.asyncio
async def test_search_timeout_fallback() -> None:
    adapter = _adapter(sleep_queries_ms={"slow": 100})
    loop = _build_loop(
        responses=['<search>{"query":"slow","top_k":1,"recall_profile":"mock"}</search>', '<answer>done</answer>'],
        adapter=adapter,
        search_timeout_s=0.001,
    )
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], trajectory_id="traj-8")
    summary = output.extra_fields["search_trace_summary"][0]
    assert summary["error_type"] == "timeout"


@pytest.mark.asyncio
async def test_max_turns() -> None:
    loop = _build_loop(
        responses=['<search>{"query":"verl","top_k":1,"recall_profile":"mock"}</search>'] * 4,
        adapter=_adapter(),
        max_turns=1,
    )
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], trajectory_id="traj-9")
    assert output.extra_fields["stop_reason"] in {"max_turns", "response_length", "max_total_tokens"}


@pytest.mark.asyncio
async def test_max_searches() -> None:
    loop = _build_loop(
        responses=[
            '<search>{"query":"verl","top_k":1,"recall_profile":"mock"}</search>',
            '<search>{"query":"mask","top_k":1,"recall_profile":"mock"}</search>',
        ],
        adapter=_adapter(),
        max_searches=1,
    )
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], trajectory_id="traj-10")
    assert output.extra_fields["num_searches"] == 1
    assert output.extra_fields["stop_reason"] == "max_searches"


@pytest.mark.asyncio
async def test_observation_token_truncation() -> None:
    adapter = InMemorySearchAdapter(
        [{"doc_id": "doc-1", "content": "x" * 1000, "metadata": {}}],
        explicit_results={"big": ["doc-1"]},
    )
    loop = _build_loop(
        responses=['<search>{"query":"big","top_k":1,"recall_profile":"mock"}</search>', '<answer>done</answer>'],
        adapter=adapter,
        max_observation_tokens=16,
    )
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], trajectory_id="traj-11")
    env_range = output.extra_fields["environment_token_ranges"][0]
    assert env_range["truncated"] is True
    assert env_range["token_count"] <= 16


@pytest.mark.asyncio
async def test_stop_reason_recorded() -> None:
    loop = _build_loop(responses=["<answer>done</answer>"])
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], trajectory_id="traj-12")
    assert output.extra_fields["stop_reason"] == "answer"


@pytest.mark.asyncio
async def test_trajectory_metadata_complete() -> None:
    loop = _build_loop(
        responses=['<search>{"query":"mask","top_k":1,"recall_profile":"mock"}</search>', '<answer>done</answer>'],
        adapter=_adapter(),
    )
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], trajectory_id="traj-13")
    required_keys = {
        "trajectory_id",
        "request_id",
        "search_traces",
        "search_trace_summary",
        "num_searches",
        "stop_reason",
        "renderer_version",
        "environment_token_ranges",
        "action_token_ranges",
    }
    assert required_keys.issubset(output.extra_fields)
    event = output.extra_fields["search_trace_summary"][0]
    assert {"query", "top_k", "recall_profile", "doc_ids", "content_hashes", "index_version", "response_hash"}.issubset(event)


@pytest.mark.asyncio
async def test_config_driven_snapshot_record_mode(tmp_path) -> None:
    snapshot_path = tmp_path / "search-record.json"
    loop = _build_loop(
        responses=['<search>{"query":"mask","top_k":1,"recall_profile":"mock"}</search>', '<answer>done</answer>'],
        adapter=_adapter(),
        record_mode=True,
        record_snapshot_id="cfg-snapshot",
        record_snapshot_path=str(snapshot_path),
    )
    assert isinstance(loop.search_adapter, RecordingSearchAdapter)
    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], trajectory_id="traj-14")
    metadata = output.extra_fields["snapshot_recording"]
    assert metadata["enabled"] is True
    assert metadata["snapshot_id"] == "cfg-snapshot"
    assert metadata["record_count"] == 1
    store = load_snapshot_store(snapshot_path)
    assert store.snapshot_id == "cfg-snapshot"
    assert len(store.records) == 1
