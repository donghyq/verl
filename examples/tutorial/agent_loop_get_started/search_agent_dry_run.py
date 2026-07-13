#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.search_agent_loop import SearchAgentLoop
from verl.experimental.agent_loop.search_environment import InMemorySearchAdapter, RecordingSearchAdapter
from verl.experimental.agent_loop.search_reward import (
    evaluate_search_trajectory_reward,
    summarize_search_reward_batch,
)
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class FakeTokenizer:
    padding_side = "right"
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=True, tokenize=True, **kwargs):
        del tools, add_generation_prompt, tokenize, kwargs
        return self.encode(" ".join(str(message.get("content", "")) for message in messages), add_special_tokens=False)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(i) for i in ids)


class FakeServerManager:
    def __init__(self) -> None:
        self.responses = [
            '<search>{"query":"response mask","top_k":1,"recall_profile":"mock"}</search>',
            '<answer>doc-mask says environment tokens use mask 0.</answer>',
        ]

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
        del request_id, prompt_ids, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs
        text = self.responses.pop(0)
        ids = [ord(ch) for ch in text]
        return TokenOutput(token_ids=ids, log_probs=[-0.1] * len(ids), extra_fields={})


async def main() -> None:
    config = OmegaConf.create(
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
                            "search_timeout_s": 0.05,
                        }
                    },
                },
                "model": {},
            },
            "data": {"tool_config_path": None, "apply_chat_template_kwargs": {}},
        }
    )

    adapter = RecordingSearchAdapter(
        InMemorySearchAdapter(
        [
            {
                "doc_id": "doc-mask",
                "content": "response_mask uses 0 for environment tokens and 1 for policy action tokens.",
                "metadata": {"topic": "mask"},
            }
        ],
        explicit_results={"response mask": ["doc-mask"]},
        index_version="mock-search.v1",
        ),
        snapshot_id="demo-snapshot",
    )
    tokenizer = FakeTokenizer()
    server_manager = FakeServerManager()
    loop = SearchAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
        search_adapter=adapter,
    )

    output = await loop.run(
        {},
        raw_prompt=[{"role": "user", "content": "What does response_mask mean?"}],
        trajectory_id="demo-trajectory",
    )
    reward = evaluate_search_trajectory_reward(
        trajectory_metadata=output.extra_fields,
        answer_text="doc-mask says environment tokens use mask 0.",
        answer_correct=True,
        gold_doc_ids=["doc-mask"],
    )
    batch_summary = summarize_search_reward_batch(
        [
            {
                "trajectory_metadata": output.extra_fields,
                "answer_text": "doc-mask says environment tokens use mask 0.",
                "answer_correct": True,
                "gold_doc_ids": ["doc-mask"],
            }
        ]
    )

    result = {
        "trajectory_id": output.extra_fields["trajectory_id"],
        "stop_reason": output.extra_fields["stop_reason"],
        "num_searches": output.extra_fields["num_searches"],
        "response_mask_prefix": output.response_mask[:80],
        "search_trace_summary": output.extra_fields["search_trace_summary"],
        "snapshot_recording": output.extra_fields.get("snapshot_recording"),
        "snapshot_record_count": len(adapter.snapshot_store.records),
        "reward": reward.model_dump(mode="json"),
        "reward_batch_summary": batch_summary.model_dump(mode="json"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
