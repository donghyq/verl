from __future__ import annotations

from verl.experimental.agent_loop.search_reward import (
    SearchRewardConfig,
    evaluate_search_reward_batch,
    evaluate_search_trajectory_reward,
    summarize_search_reward_batch,
)


def _metadata() -> dict:
    return {
        "search_trace_summary": [
            {
                "query": "mask",
                "top_k": 1,
                "recall_profile": "mock",
                "doc_ids": ["doc-2"],
                "content_hashes": ["hash-2"],
                "index_version": "idx.v1",
                "response_hash": "resp-1",
                "search_latency_ms": 250.0,
                "cost": {"candidate_count": 4},
                "error_type": None,
            }
        ],
        "invalid_action_count": 0,
        "stop_reason": "answer",
    }


def test_reward_component_breakdown() -> None:
    reward = evaluate_search_trajectory_reward(
        trajectory_metadata=_metadata(),
        answer_text="doc-2 explains the answer",
        answer_correct=True,
        gold_doc_ids=["doc-2"],
    )
    components = reward.components
    assert components.answer_correctness == 1.0
    assert components.groundedness == 1.0
    assert components.retrieval_utility == 1.0
    assert components.search_latency_cost > 0
    assert components.search_resource_cost > 0


def test_reward_deterministic() -> None:
    kwargs = dict(
        trajectory_metadata=_metadata(),
        answer_text="doc-2 explains the answer",
        answer_correct=True,
        gold_doc_ids=["doc-2"],
    )
    a = evaluate_search_trajectory_reward(**kwargs)
    b = evaluate_search_trajectory_reward(**kwargs)
    assert a.model_dump() == b.model_dump()


def test_invalid_action_penalty() -> None:
    metadata = _metadata()
    metadata["invalid_action_count"] = 2
    reward = evaluate_search_trajectory_reward(
        trajectory_metadata=metadata,
        answer_text="wrong",
        answer_correct=False,
    )
    assert reward.components.invalid_action_penalty == 2.0


def test_timeout_penalty() -> None:
    metadata = _metadata()
    metadata["search_trace_summary"][0]["error_type"] = "timeout"
    reward = evaluate_search_trajectory_reward(
        trajectory_metadata=metadata,
        answer_text="wrong",
        answer_correct=False,
    )
    assert reward.components.failure_penalty == 1.0


def test_measured_latency_cost() -> None:
    reward = evaluate_search_trajectory_reward(
        trajectory_metadata=_metadata(),
        answer_text="doc-2 explains the answer",
        answer_correct=True,
    )
    assert reward.components.search_latency_cost == 0.25


def test_gold_evidence_utility() -> None:
    reward = evaluate_search_trajectory_reward(
        trajectory_metadata=_metadata(),
        answer_text="answer",
        answer_correct=False,
        gold_content_hashes=["hash-2"],
    )
    assert reward.components.retrieval_utility == 1.0


def test_reward_schema_version() -> None:
    reward = evaluate_search_trajectory_reward(
        trajectory_metadata=_metadata(),
        answer_text="answer",
        answer_correct=False,
    )
    assert reward.reward_schema_version == "search_reward.v1"


def test_weight_change_only_affects_total() -> None:
    base = evaluate_search_trajectory_reward(
        trajectory_metadata=_metadata(),
        answer_text="doc-2 explains the answer",
        answer_correct=True,
        gold_doc_ids=["doc-2"],
    )
    changed = evaluate_search_trajectory_reward(
        trajectory_metadata=_metadata(),
        answer_text="doc-2 explains the answer",
        answer_correct=True,
        gold_doc_ids=["doc-2"],
        config=SearchRewardConfig(weights={**SearchRewardConfig().weights, "answer_correctness": 2.0}),
    )
    assert base.components.model_dump() == changed.components.model_dump()
    assert base.total_reward != changed.total_reward


def test_reward_batch_evaluator() -> None:
    items = [
        {
            "trajectory_metadata": _metadata(),
            "answer_text": "doc-2 explains the answer",
            "answer_correct": True,
            "gold_doc_ids": ["doc-2"],
        },
        {
            "trajectory_metadata": {**_metadata(), "stop_reason": "max_turns", "invalid_action_count": 1},
            "answer_text": "wrong",
            "answer_correct": False,
            "gold_doc_ids": ["missing"],
        },
    ]
    outputs = evaluate_search_reward_batch(items)
    assert len(outputs) == 2
    assert outputs[0].components.answer_correctness == 1.0
    assert outputs[1].components.invalid_action_penalty == 1.0


def test_reward_batch_summary() -> None:
    items = [
        {
            "trajectory_metadata": _metadata(),
            "answer_text": "doc-2 explains the answer",
            "answer_correct": True,
            "gold_doc_ids": ["doc-2"],
        },
        {
            "trajectory_metadata": {**_metadata(), "stop_reason": "max_turns", "invalid_action_count": 1},
            "answer_text": "wrong",
            "answer_correct": False,
            "gold_doc_ids": ["missing"],
        },
    ]
    summary = summarize_search_reward_batch(items)
    assert summary.count == 2
    assert "answer_correctness" in summary.totals_by_component
    assert summary.stop_reason_histogram["answer"] == 1
    assert summary.stop_reason_histogram["max_turns"] == 1


def test_empty_reward_batch_summary() -> None:
    summary = summarize_search_reward_batch([])
    assert summary.count == 0
    assert summary.mean_total_reward == 0.0
