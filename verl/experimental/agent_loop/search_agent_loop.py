"""Minimal multi-turn search agent loop for the environment prototype stage."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.experimental.agent_loop.search_environment import (
    DEFAULT_ALLOWED_RECALL_PROFILES,
    SEARCH_RENDERER_VERSION,
    InMemorySearchAdapter,
    RecordingSearchAdapter,
    SearchAction,
    SearchAdapter,
    SearchObservation,
    SearchObservationRenderer,
    SearchTrace,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class ParsedAction:
    kind: str
    raw_text: str
    answer_text: Optional[str] = None
    action: Optional[SearchAction] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def _find_tag_spans(text: str, tag: str) -> list[tuple[int, int, str]]:
    start_token = f"<{tag}>"
    end_token = f"</{tag}>"
    spans: list[tuple[int, int, str]] = []
    start = 0
    while True:
        s = text.find(start_token, start)
        if s < 0:
            break
        e = text.find(end_token, s + len(start_token))
        if e < 0:
            break
        content_start = s + len(start_token)
        spans.append((s, e + len(end_token), text[content_start:e]))
        start = e + len(end_token)
    return spans


def _build_request_id(kwargs: dict[str, Any]) -> str:
    request_id = kwargs.get("trajectory_id") or kwargs.get("request_id")
    if request_id is not None:
        value = request_id.item() if hasattr(request_id, "item") else request_id
        return str(value)

    uid = kwargs.get("uid")
    session_id = kwargs.get("session_id")
    if uid is not None or session_id is not None:
        uid_value = uid.item() if hasattr(uid, "item") else uid
        session_value = session_id.item() if hasattr(session_id, "item") else session_id
        return f"uid:{uid_value}|session:{session_value}"
    return uuid4().hex


@register("search_agent")
class SearchAgentLoop(AgentLoopBase):
    """Minimal search-answer loop built on VeRL's token-in/token-out agent API."""

    def __init__(
        self,
        *args,
        search_adapter: Optional[SearchAdapter] = None,
        observation_renderer: Optional[SearchObservationRenderer] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        custom_cfg = (self.rollout_config.custom or {}).get("search_agent", {}) if self.rollout_config.custom else {}
        self.max_turns = int(custom_cfg.get("max_turns", 4))
        self.max_searches = int(custom_cfg.get("max_searches", 2))
        self.max_total_tokens = int(custom_cfg.get("max_total_tokens", self.prompt_length + self.response_length))
        self.max_observation_tokens = int(custom_cfg.get("max_observation_tokens", 192))
        self.max_query_length = int(custom_cfg.get("max_query_length", 256))
        self.min_top_k = int(custom_cfg.get("min_top_k", 1))
        self.max_top_k = int(custom_cfg.get("max_top_k", 10))
        self.allowed_recall_profiles = tuple(
            custom_cfg.get("allowed_recall_profiles", list(DEFAULT_ALLOWED_RECALL_PROFILES))
        )
        self.search_timeout_s = float(custom_cfg.get("search_timeout_s", 2.0))
        self.record_mode_enabled = bool(custom_cfg.get("record_mode", False))
        self.record_snapshot_id = str(custom_cfg.get("record_snapshot_id", "search-agent-snapshot"))
        self.record_snapshot_path = custom_cfg.get("record_snapshot_path")

        default_documents = custom_cfg.get(
            "mock_documents",
            [
                {
                    "doc_id": "doc-search-1",
                    "content": "VeRL agent loops support token-in token-out multi-turn rollout.",
                    "metadata": {"topic": "verl"},
                },
                {
                    "doc_id": "doc-search-2",
                    "content": "response_mask marks LLM action tokens with 1 and environment tokens with 0.",
                    "metadata": {"topic": "mask"},
                },
            ],
        )
        base_adapter = search_adapter or InMemorySearchAdapter(
            default_documents,
            index_version=str(custom_cfg.get("index_version", "mock-search.v1")),
            base_latency_ms=float(custom_cfg.get("base_latency_ms", 5.0)),
        )
        if self.record_mode_enabled and not isinstance(base_adapter, RecordingSearchAdapter):
            base_adapter = RecordingSearchAdapter(
                base_adapter,
                snapshot_id=self.record_snapshot_id,
                deduplicate_by_action_identity=bool(custom_cfg.get("record_deduplicate", True)),
            )
        self.search_adapter = base_adapter
        self.observation_renderer = observation_renderer or SearchObservationRenderer(
            renderer_version=SEARCH_RENDERER_VERSION,
            max_documents=int(custom_cfg.get("max_documents", 3)),
            max_document_tokens=int(custom_cfg.get("max_document_tokens", 96)),
        )

    async def _encode_text(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            return await self.loop.run_in_executor(None, lambda: self.tokenizer.encode(text, add_special_tokens=False))

        def _tokenize_fallback() -> list[int]:
            tokenized = self.tokenizer(text, add_special_tokens=False)
            if isinstance(tokenized, dict):
                return list(tokenized["input_ids"])
            return list(tokenized.input_ids)

        return await self.loop.run_in_executor(None, _tokenize_fallback)

    async def _decode_tokens(self, token_ids: list[int]) -> str:
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(token_ids, skip_special_tokens=False),
        )

    async def _parse_action(self, token_ids: list[int]) -> ParsedAction:
        text = await self._decode_tokens(token_ids)
        search_spans = _find_tag_spans(text, "search")
        answer_spans = _find_tag_spans(text, "answer")

        if len(search_spans) > 1 or len(answer_spans) > 1:
            return ParsedAction(
                kind="invalid",
                raw_text=text,
                error_type="multiple_actions",
                error_message="Multiple <search> or <answer> blocks found in a single model turn.",
            )

        if search_spans and answer_spans:
            return ParsedAction(
                kind="invalid",
                raw_text=text,
                error_type="simultaneous_actions",
                error_message="Model emitted both <search> and <answer> in the same turn.",
            )

        if answer_spans:
            return ParsedAction(kind="answer", raw_text=text, answer_text=answer_spans[0][2])

        if search_spans:
            payload = search_spans[0][2]
            validation = SearchAction.validate_payload(
                payload,
                max_query_length=self.max_query_length,
                min_top_k=self.min_top_k,
                max_top_k=self.max_top_k,
                allowed_recall_profiles=self.allowed_recall_profiles,
            )
            if validation.ok:
                return ParsedAction(kind="search", raw_text=text, action=validation.action)
            return ParsedAction(
                kind="invalid",
                raw_text=text,
                error_type=validation.error_type,
                error_message=validation.error_message,
            )

        return ParsedAction(
            kind="invalid",
            raw_text=text,
            error_type="no_action",
            error_message="Model turn did not contain a <search> or <answer> action.",
        )

    def _maybe_trim_to_total_limit(
        self,
        *,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: list[float],
        metadata: dict[str, Any],
    ) -> tuple[list[int], list[int], list[int], list[float], Optional[str]]:
        if self.max_total_tokens <= 0:
            return prompt_ids, response_ids, response_mask, response_logprobs, None

        total_length = len(prompt_ids) + len(response_ids)
        if total_length <= self.max_total_tokens:
            return prompt_ids, response_ids, response_mask, response_logprobs, None

        overflow = total_length - self.max_total_tokens
        if overflow >= len(response_ids):
            trimmed_response_ids = []
            trimmed_response_mask = []
            trimmed_logprobs = []
        else:
            trimmed_response_ids = response_ids[:-overflow]
            trimmed_response_mask = response_mask[:-overflow]
            trimmed_logprobs = response_logprobs[:-overflow] if response_logprobs else []

        metadata.setdefault("token_limit_events", []).append(
            {
                "type": "max_total_tokens",
                "max_total_tokens": self.max_total_tokens,
                "total_length_before_trim": total_length,
                "overflow_tokens": overflow,
            }
        )
        return prompt_ids, trimmed_response_ids, trimmed_response_mask, trimmed_logprobs, "max_total_tokens"

    def _make_invalid_observation(
        self,
        *,
        trajectory_id: str,
        parsed: ParsedAction,
        logical_step: int,
    ) -> SearchObservation:
        trace_id = f"{trajectory_id}-invalid-{logical_step}"
        return SearchObservation.error(
            trace_id=trace_id,
            index_version=getattr(self.search_adapter, "index_version", "mock-search.v1"),
            recall_profile="invalid",
            error_type=parsed.error_type or "invalid_action",
            error_message=parsed.error_message or "Invalid action.",
            latency_ms=0.0,
            degraded=True,
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        trajectory_id = _build_request_id(kwargs)

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        metrics = AgentLoopMetrics()
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        search_traces: list[dict[str, Any]] = []
        search_trace_summary: list[dict[str, Any]] = []
        action_token_ranges: list[dict[str, Any]] = []
        environment_token_ranges: list[dict[str, Any]] = []
        token_limit_events: list[dict[str, Any]] = []
        invalid_action_count = 0
        assistant_turns = 0
        environment_turns = 0
        num_searches = 0
        stop_reason = "unknown"

        while True:
            if assistant_turns >= self.max_turns:
                stop_reason = "max_turns"
                break
            if num_searches >= self.max_searches and assistant_turns > 0:
                stop_reason = "max_searches"
                break
            if len(prompt_ids) + len(response_ids) >= self.max_total_tokens:
                stop_reason = "max_total_tokens"
                break
            if len(response_ids) >= self.response_length:
                stop_reason = "response_length"
                break

            current_prompt_ids = prompt_ids + response_ids

            generation_timing: dict[str, Any] = {}
            with simple_timer("generate_sequences", generation_timing):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=trajectory_id,
                    prompt_ids=current_prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                )

            metrics.generate_sequences += float(generation_timing.get("generate_sequences", 0.0))
            if output.num_preempted is not None:
                metrics.num_preempted = (
                    output.num_preempted
                    if metrics.num_preempted < 0
                    else metrics.num_preempted + int(output.num_preempted)
                )

            assistant_turns += 1
            action_start = len(response_ids)
            response_ids.extend(output.token_ids)
            response_mask.extend([1] * len(output.token_ids))
            if output.log_probs:
                response_logprobs.extend(output.log_probs)
            elif response_logprobs:
                response_logprobs.extend([0.0] * len(output.token_ids))

            action_token_ranges.append(
                {
                    "turn": assistant_turns,
                    "start": action_start,
                    "end": len(response_ids),
                    "token_count": len(output.token_ids),
                }
            )

            prompt_ids, response_ids, response_mask, response_logprobs, trimmed_reason = self._maybe_trim_to_total_limit(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                metadata={"token_limit_events": token_limit_events},
            )
            if trimmed_reason is not None:
                stop_reason = trimmed_reason
                break
            if len(response_ids) >= self.response_length:
                stop_reason = "response_length"
                break

            parsed = await self._parse_action(output.token_ids)
            if parsed.kind == "answer":
                stop_reason = "answer"
                break

            if parsed.kind != "search":
                invalid_action_count += 1
                observation = self._make_invalid_observation(
                    trajectory_id=trajectory_id,
                    parsed=parsed,
                    logical_step=assistant_turns,
                )
            else:
                if num_searches >= self.max_searches:
                    stop_reason = "max_searches"
                    break

                search_timing: dict[str, Any] = {}
                with simple_timer("tool_calls", search_timing):
                    try:
                        observation = await asyncio.wait_for(
                            self.search_adapter.search(parsed.action),
                            timeout=self.search_timeout_s,
                        )
                    except asyncio.TimeoutError:
                        observation = SearchObservation.error(
                            trace_id=f"{trajectory_id}-timeout-{assistant_turns}",
                            index_version=getattr(self.search_adapter, "index_version", "mock-search.v1"),
                            recall_profile=parsed.action.recall_profile,
                            error_type="timeout",
                            error_message=f"Search timed out after {self.search_timeout_s:.2f}s",
                            latency_ms=self.search_timeout_s * 1000.0,
                        )
                    except Exception as exc:  # pragma: no cover - defensive fallback
                        observation = SearchObservation.error(
                            trace_id=f"{trajectory_id}-failure-{assistant_turns}",
                            index_version=getattr(self.search_adapter, "index_version", "mock-search.v1"),
                            recall_profile=parsed.action.recall_profile,
                            error_type="adapter_failure",
                            error_message=str(exc),
                            latency_ms=0.0,
                        )

                metrics.tool_calls += float(search_timing.get("tool_calls", 0.0))
                num_searches += 1

            rendered = await self.observation_renderer.render(
                observation,
                encode_text=self._encode_text,
                decode_tokens=self._decode_tokens,
                max_total_tokens=self.max_observation_tokens,
            )

            env_start = len(response_ids)
            response_ids.extend(rendered.token_ids)
            response_mask.extend([0] * len(rendered.token_ids))
            if response_logprobs:
                response_logprobs.extend([0.0] * len(rendered.token_ids))

            environment_turns += 1
            environment_token_ranges.append(
                {
                    "turn": assistant_turns,
                    "start": env_start,
                    "end": len(response_ids),
                    "token_count": len(rendered.token_ids),
                    "renderer_version": rendered.renderer_version,
                    "truncated": rendered.truncated,
                }
            )

            action_payload = (
                parsed.action.identity_payload() if parsed.action is not None else {"error_type": parsed.error_type}
            )
            search_trace = SearchTrace(
                logical_step=assistant_turns,
                action=action_payload,
                observation=observation,
                timestamp_ms=int(time.time() * 1000),
                index_version=observation.index_version,
                renderer_version=rendered.renderer_version,
            )
            search_traces.append(search_trace.model_dump(mode="json"))
            search_trace_summary.append(
                {
                    "logical_step": assistant_turns,
                    "query": parsed.action.query if parsed.action is not None else None,
                    "top_k": parsed.action.top_k if parsed.action is not None else None,
                    "recall_profile": parsed.action.recall_profile if parsed.action is not None else None,
                    "doc_ids": [doc.doc_id for doc in observation.documents],
                    "content_hashes": [doc.content_hash for doc in observation.documents],
                    "index_version": observation.index_version,
                    "response_hash": observation.response_hash,
                    "search_latency_ms": observation.latency_ms,
                    "cost": observation.cost.model_dump(mode="json"),
                    "degraded": observation.degraded,
                    "error_type": observation.error_type,
                    "error_message": observation.error_message,
                }
            )

            prompt_ids, response_ids, response_mask, response_logprobs, trimmed_reason = self._maybe_trim_to_total_limit(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                metadata={"token_limit_events": token_limit_events},
            )
            if trimmed_reason is not None:
                stop_reason = trimmed_reason
                break
            if len(response_ids) >= self.response_length:
                stop_reason = "response_length"
                break

            if observation.error_type is not None and parsed.kind != "search":
                continue

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=(response_logprobs[: self.response_length] if response_logprobs else None),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=1 + assistant_turns + environment_turns,
            metrics=metrics,
            extra_fields={
                "trajectory_id": trajectory_id,
                "request_id": trajectory_id,
                "search_traces": search_traces,
                "search_trace_summary": search_trace_summary,
                "num_searches": num_searches,
                "invalid_action_count": invalid_action_count,
                "stop_reason": stop_reason,
                "renderer_version": self.observation_renderer.renderer_version,
                "environment_token_ranges": environment_token_ranges,
                "action_token_ranges": action_token_ranges,
                "token_limit_events": token_limit_events,
                "search_reward_hint": {
                    "search_timeout_s": self.search_timeout_s,
                    "max_turns": self.max_turns,
                    "max_searches": self.max_searches,
                },
                "snapshot_recording": self._snapshot_recording_metadata(),
                "turn_scores": [],
                "tool_rewards": [],
            },
        )
        await self._maybe_dump_snapshot_records()
        return output

    def _snapshot_recording_metadata(self) -> dict[str, Any]:
        if not isinstance(self.search_adapter, RecordingSearchAdapter):
            return {
                "enabled": False,
                "snapshot_id": None,
                "record_count": 0,
                "snapshot_path": None,
            }
        return {
            "enabled": True,
            "snapshot_id": self.search_adapter.snapshot_id,
            "record_count": len(self.search_adapter.snapshot_store.records),
            "snapshot_path": self.record_snapshot_path,
        }

    async def _maybe_dump_snapshot_records(self) -> None:
        if not self.record_snapshot_path:
            return
        if not isinstance(self.search_adapter, RecordingSearchAdapter):
            return
        await self.loop.run_in_executor(None, lambda: self.search_adapter.dump(self.record_snapshot_path))
