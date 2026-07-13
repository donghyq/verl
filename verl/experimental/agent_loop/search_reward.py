"""Deterministic baseline reward components for search-agent trajectories.

This module intentionally does not implement PPO/GRPO logic. It only defines a
pure metadata-driven evaluator so M1 can validate that collected trajectory
metadata is sufficient for downstream reward infrastructure in M2.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

SEARCH_REWARD_SCHEMA_VERSION = "search_reward.v1"


class SearchRewardComponents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_correctness: float = 0.0
    groundedness: float = 0.0
    retrieval_utility: float = 0.0
    search_latency_cost: float = 0.0
    search_resource_cost: float = 0.0
    invalid_action_penalty: float = 0.0
    failure_penalty: float = 0.0
    format_penalty: float = 0.0


class SearchRewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward_schema_version: str = SEARCH_REWARD_SCHEMA_VERSION
    components: SearchRewardComponents
    weights: dict[str, float]
    total_reward: float


class SearchRewardBatchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward_schema_version: str = SEARCH_REWARD_SCHEMA_VERSION
    count: int
    mean_total_reward: float
    totals_by_component: dict[str, float]
    means_by_component: dict[str, float]
    stop_reason_histogram: dict[str, int]


class SearchRewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward_schema_version: str = SEARCH_REWARD_SCHEMA_VERSION
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "answer_correctness": 1.0,
            "groundedness": 0.4,
            "retrieval_utility": 0.4,
            "search_latency_cost": 0.01,
            "search_resource_cost": 0.05,
            "invalid_action_penalty": 1.0,
            "failure_penalty": 1.0,
            "format_penalty": 0.5,
        }
    )
    latency_normalizer_ms: float = 1000.0
    candidate_count_normalizer: float = 10.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_groundedness(answer_text: str, trace_summary: list[dict[str, Any]]) -> float:
    if not answer_text or not trace_summary:
        return 0.0
    answer_lower = answer_text.lower()
    for event in reversed(trace_summary):
        for doc_id in event.get("doc_ids", []):
            if str(doc_id).lower() in answer_lower:
                return 1.0
    return 0.0


def _infer_retrieval_utility(
    trace_summary: list[dict[str, Any]],
    *,
    gold_doc_ids: Optional[list[str]] = None,
    gold_content_hashes: Optional[list[str]] = None,
) -> float:
    if not trace_summary:
        return 0.0

    observed_doc_ids = []
    observed_hashes = []
    for event in trace_summary:
        observed_doc_ids.extend(event.get("doc_ids", []))
        observed_hashes.extend(event.get("content_hashes", []))

    gold_doc_ids = gold_doc_ids or []
    gold_content_hashes = gold_content_hashes or []
    if gold_doc_ids:
        return 1.0 if any(doc_id in observed_doc_ids for doc_id in gold_doc_ids) else 0.0
    if gold_content_hashes:
        return 1.0 if any(content_hash in observed_hashes for content_hash in gold_content_hashes) else 0.0
    return 1.0 if observed_doc_ids else 0.0


def evaluate_search_trajectory_reward(
    *,
    trajectory_metadata: dict[str, Any],
    answer_text: str,
    answer_correct: bool,
    config: Optional[SearchRewardConfig] = None,
    gold_doc_ids: Optional[list[str]] = None,
    gold_content_hashes: Optional[list[str]] = None,
) -> SearchRewardBreakdown:
    """Pure, deterministic reward evaluator over trajectory metadata."""

    config = config or SearchRewardConfig()
    trace_summary = list(trajectory_metadata.get("search_trace_summary", []))
    invalid_action_count = int(trajectory_metadata.get("invalid_action_count", 0) or 0)
    stop_reason = str(trajectory_metadata.get("stop_reason", ""))

    total_latency_ms = 0.0
    total_candidate_count = 0.0
    failure_count = 0
    format_penalty = 0.0
    if stop_reason in {"max_turns", "max_total_tokens", "response_length", "unknown"}:
        format_penalty = 1.0

    for event in trace_summary:
        total_latency_ms += _safe_float(event.get("search_latency_ms", 0.0))
        cost = event.get("cost", {}) or {}
        total_candidate_count += _safe_float(cost.get("candidate_count", 0.0))
        if event.get("error_type") is not None:
            failure_count += 1

    components = SearchRewardComponents(
        answer_correctness=1.0 if answer_correct else 0.0,
        groundedness=_infer_groundedness(answer_text, trace_summary),
        retrieval_utility=_infer_retrieval_utility(
            trace_summary,
            gold_doc_ids=gold_doc_ids,
            gold_content_hashes=gold_content_hashes,
        ),
        search_latency_cost=(total_latency_ms / config.latency_normalizer_ms) if total_latency_ms > 0 else 0.0,
        search_resource_cost=(total_candidate_count / config.candidate_count_normalizer)
        if total_candidate_count > 0
        else 0.0,
        invalid_action_penalty=float(invalid_action_count),
        failure_penalty=float(failure_count),
        format_penalty=format_penalty,
    )

    total = 0.0
    component_dict = components.model_dump(mode="json")
    for name, value in component_dict.items():
        weight = float(config.weights.get(name, 0.0))
        if name.endswith("_cost") or name.endswith("_penalty"):
            total -= weight * float(value)
        else:
            total += weight * float(value)

    return SearchRewardBreakdown(
        reward_schema_version=config.reward_schema_version,
        components=components,
        weights={k: float(v) for k, v in config.weights.items()},
        total_reward=float(total),
    )


def evaluate_search_reward_batch(
    items: list[dict[str, Any]],
    *,
    config: Optional[SearchRewardConfig] = None,
) -> list[SearchRewardBreakdown]:
    """Evaluate a batch of trajectory metadata dictionaries deterministically.

    Each item may contain:
    - trajectory_metadata
    - answer_text
    - answer_correct
    - gold_doc_ids
    - gold_content_hashes
    """

    config = config or SearchRewardConfig()
    outputs: list[SearchRewardBreakdown] = []
    for item in items:
        outputs.append(
            evaluate_search_trajectory_reward(
                trajectory_metadata=item["trajectory_metadata"],
                answer_text=item.get("answer_text", ""),
                answer_correct=bool(item.get("answer_correct", False)),
                config=config,
                gold_doc_ids=item.get("gold_doc_ids"),
                gold_content_hashes=item.get("gold_content_hashes"),
            )
        )
    return outputs


def summarize_search_reward_batch(
    items: list[dict[str, Any]],
    *,
    config: Optional[SearchRewardConfig] = None,
) -> SearchRewardBatchSummary:
    config = config or SearchRewardConfig()
    outputs = evaluate_search_reward_batch(items, config=config)
    count = len(outputs)
    if count == 0:
        zero_components = SearchRewardComponents().model_dump(mode="json")
        return SearchRewardBatchSummary(
            reward_schema_version=config.reward_schema_version,
            count=0,
            mean_total_reward=0.0,
            totals_by_component={k: 0.0 for k in zero_components},
            means_by_component={k: 0.0 for k in zero_components},
            stop_reason_histogram={},
        )

    totals_by_component = {k: 0.0 for k in outputs[0].components.model_dump(mode="json")}
    total_reward = 0.0
    stop_reason_histogram: dict[str, int] = {}

    for item, output in zip(items, outputs, strict=True):
        total_reward += float(output.total_reward)
        for name, value in output.components.model_dump(mode="json").items():
            totals_by_component[name] += float(value)
        stop_reason = str(item.get("trajectory_metadata", {}).get("stop_reason", "unknown"))
        stop_reason_histogram[stop_reason] = stop_reason_histogram.get(stop_reason, 0) + 1

    means_by_component = {name: value / count for name, value in totals_by_component.items()}
    return SearchRewardBatchSummary(
        reward_schema_version=config.reward_schema_version,
        count=count,
        mean_total_reward=total_reward / count,
        totals_by_component=totals_by_component,
        means_by_component=means_by_component,
        stop_reason_histogram=stop_reason_histogram,
    )
