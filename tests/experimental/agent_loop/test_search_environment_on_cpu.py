from __future__ import annotations

import json

import pytest

from verl.experimental.agent_loop.search_environment import (
    SEARCH_ENV_SCHEMA_VERSION,
    SEARCH_SNAPSHOT_SCHEMA_VERSION,
    InMemorySearchAdapter,
    RecordingSearchAdapter,
    SearchAction,
    SearchDocument,
    SearchObservation,
    SearchSnapshotRecord,
    SearchSnapshotStore,
    SnapshotReplaySearchAdapter,
    dump_snapshot_records,
    load_snapshot_store,
    stable_content_hash,
)


def _docs() -> list[dict[str, object]]:
    return [
        {
            "doc_id": "doc-1",
            "content": "verl supports token in token out multi-turn generation.",
            "metadata": {"source": "kb"},
        },
        {
            "doc_id": "doc-2",
            "content": "response_mask uses 1 for model action tokens and 0 for environment tokens.",
            "metadata": {"source": "kb"},
        },
        {
            "doc_id": "doc-3",
            "content": "Search agent loops can record trace ids and index versions.",
            "metadata": {"source": "kb"},
        },
    ]


def test_search_action_valid() -> None:
    result = SearchAction.validate_payload(
        {"query": "  response mask semantics  ", "top_k": 2, "recall_profile": "mock"}
    )
    assert result.ok is True
    assert result.action is not None
    assert result.action.query == "response mask semantics"


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        ({"query": "", "top_k": 2, "recall_profile": "mock"}, "empty_query"),
        ({"query": "x" * 300, "top_k": 2, "recall_profile": "mock"}, "query_too_long"),
        ({"query": "ok", "top_k": 0, "recall_profile": "mock"}, "top_k_out_of_range"),
        ({"query": "ok", "top_k": 2, "recall_profile": "bad"}, "invalid_recall_profile"),
        ('{"query":', "malformed_payload"),
        ({"query": "ok", "top_k": 2, "recall_profile": "mock", "unknown": 1}, "unknown_fields"),
    ],
)
def test_search_action_invalid_cases(payload, error_type: str) -> None:
    result = SearchAction.validate_payload(payload)
    assert result.ok is False
    assert result.error_type == error_type


@pytest.mark.asyncio
async def test_mock_exact_result_and_order_stable() -> None:
    adapter = InMemorySearchAdapter(_docs(), explicit_results={"mask": ["doc-2", "doc-1"]})
    action = SearchAction(query="mask", top_k=2, recall_profile="mock")
    observation = await adapter.search(action)
    assert [doc.doc_id for doc in observation.documents] == ["doc-2", "doc-1"]
    observation2 = await adapter.search(action)
    assert [doc.doc_id for doc in observation2.documents] == ["doc-2", "doc-1"]
    assert observation.response_hash == observation2.response_hash


def test_content_hash_stable() -> None:
    content = "stable hashing content"
    assert stable_content_hash(content) == stable_content_hash(content)
    assert stable_content_hash(content) != stable_content_hash(content + "!")


@pytest.mark.asyncio
async def test_response_hash_stable() -> None:
    adapter = InMemorySearchAdapter(_docs())
    action = SearchAction(query="token out", top_k=2, recall_profile="mock")
    observation = await adapter.search(action)
    observation2 = await adapter.search(action)
    assert observation.response_hash == observation2.response_hash


@pytest.mark.asyncio
async def test_timeout_observation() -> None:
    adapter = InMemorySearchAdapter(_docs(), timeout_queries={"timeout"}, query_latency_ms={"timeout": 1})
    observation = await adapter.search(SearchAction(query="timeout", top_k=2, recall_profile="mock"))
    assert observation.error_type == "timeout"


@pytest.mark.asyncio
async def test_empty_result() -> None:
    adapter = InMemorySearchAdapter(_docs(), empty_queries={"empty"})
    observation = await adapter.search(SearchAction(query="empty", top_k=2, recall_profile="mock"))
    assert observation.documents == ()
    assert observation.error_type is None


@pytest.mark.asyncio
async def test_degraded_result() -> None:
    adapter = InMemorySearchAdapter(_docs(), degraded_queries={"mask"})
    observation = await adapter.search(SearchAction(query="mask", top_k=2, recall_profile="mock"))
    assert observation.degraded is True
    assert observation.cost.degraded is True


@pytest.mark.asyncio
async def test_injected_failure() -> None:
    adapter = InMemorySearchAdapter(_docs(), failure_queries={"failure": "boom"})
    observation = await adapter.search(SearchAction(query="failure", top_k=2, recall_profile="mock"))
    assert observation.error_type == "adapter_failure"
    assert "boom" in observation.error_message


@pytest.mark.asyncio
async def test_snapshot_hit(tmp_path) -> None:
    adapter = InMemorySearchAdapter(_docs())
    action = SearchAction(query="token out", top_k=2, recall_profile="mock")
    observation = await adapter.search(action)
    record = SearchSnapshotRecord.from_action(snapshot_id="snap-1", action=action, observation=observation)
    path = tmp_path / "snapshot.json"
    dump_snapshot_records([record], path)
    replay = SnapshotReplaySearchAdapter(path, expected_index_version=observation.index_version)
    replay_observation = await replay.search(action)
    assert replay_observation.response_hash == observation.response_hash


@pytest.mark.asyncio
async def test_snapshot_miss(tmp_path) -> None:
    adapter = InMemorySearchAdapter(_docs())
    action = SearchAction(query="token out", top_k=2, recall_profile="mock")
    observation = await adapter.search(action)
    record = SearchSnapshotRecord.from_action(snapshot_id="snap-1", action=action, observation=observation)
    path = tmp_path / "snapshot.json"
    dump_snapshot_records([record], path)
    replay = SnapshotReplaySearchAdapter(path)
    miss = await replay.search(SearchAction(query="other", top_k=2, recall_profile="mock"))
    assert miss.error_type == "snapshot_miss"


@pytest.mark.asyncio
async def test_snapshot_schema_version_mismatch(tmp_path) -> None:
    adapter = InMemorySearchAdapter(_docs())
    action = SearchAction(query="token out", top_k=2, recall_profile="mock")
    observation = await adapter.search(action)
    record = SearchSnapshotRecord.from_action(snapshot_id="snap-1", action=action, observation=observation)
    payload = {"schema_version": SEARCH_SNAPSHOT_SCHEMA_VERSION, "snapshot_id": "snap-1", "entries": [record.model_dump(mode="json")]}
    payload["entries"][0]["schema_version"] = "wrong.version"
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    replay = SnapshotReplaySearchAdapter(path)
    result = await replay.search(action)
    assert result.error_type == "snapshot_schema_version_mismatch"


@pytest.mark.asyncio
async def test_snapshot_index_version_mismatch(tmp_path) -> None:
    adapter = InMemorySearchAdapter(_docs(), index_version="idx-a")
    action = SearchAction(query="token out", top_k=2, recall_profile="mock")
    observation = await adapter.search(action)
    record = SearchSnapshotRecord.from_action(snapshot_id="snap-1", action=action, observation=observation)
    path = tmp_path / "snapshot.json"
    dump_snapshot_records([record], path)
    replay = SnapshotReplaySearchAdapter(path, expected_index_version="idx-b")
    result = await replay.search(action)
    assert result.error_type == "snapshot_index_version_mismatch"


@pytest.mark.asyncio
async def test_snapshot_corrupted_hash(tmp_path) -> None:
    adapter = InMemorySearchAdapter(_docs(), index_version="idx-a")
    action = SearchAction(query="token out", top_k=2, recall_profile="mock")
    observation = await adapter.search(action)
    record = SearchSnapshotRecord.from_action(snapshot_id="snap-1", action=action, observation=observation)
    payload = {"schema_version": SEARCH_SNAPSHOT_SCHEMA_VERSION, "snapshot_id": "snap-1", "entries": [record.model_dump(mode="json")]}
    payload["entries"][0]["response_hash"] = "corrupted"
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    replay = SnapshotReplaySearchAdapter(path, expected_index_version="idx-a")
    result = await replay.search(action)
    assert result.error_type == "snapshot_hash_mismatch"


def test_search_observation_schema_version_defaults() -> None:
    doc = SearchDocument.from_content(
        doc_id="doc-1",
        content="hello",
        score=1.0,
        source="test",
        index_version="idx",
    )
    observation = SearchObservation(documents=(doc,), trace_id="trace", index_version="idx")
    assert observation.schema_version == SEARCH_ENV_SCHEMA_VERSION
    assert observation.response_hash


@pytest.mark.asyncio
async def test_recording_search_adapter_records_snapshot(tmp_path) -> None:
    base = InMemorySearchAdapter(_docs(), index_version="idx-record")
    recorder = RecordingSearchAdapter(base, snapshot_id="snap-record")
    action = SearchAction(query="token out", top_k=2, recall_profile="mock")
    observation = await recorder.search(action)
    assert observation.index_version == "idx-record"
    assert len(recorder.snapshot_store.records) == 1
    record = recorder.snapshot_store.records[0]
    assert record.action_identity == action.identity()
    path = tmp_path / "recorded.json"
    recorder.dump(path)
    loaded = load_snapshot_store(path)
    assert loaded.snapshot_id == "snap-record"
    assert len(loaded.records) == 1


@pytest.mark.asyncio
async def test_recording_search_adapter_deduplicates_by_identity() -> None:
    base = InMemorySearchAdapter(_docs(), index_version="idx-record")
    recorder = RecordingSearchAdapter(base, snapshot_id="snap-record", deduplicate_by_action_identity=True)
    action = SearchAction(query="token out", top_k=2, recall_profile="mock")
    await recorder.search(action)
    await recorder.search(action)
    assert len(recorder.snapshot_store.records) == 1


def test_snapshot_store_round_trip_jsonl(tmp_path) -> None:
    doc = SearchDocument.from_content(
        doc_id="doc-1",
        content="round trip",
        score=1.0,
        source="test",
        index_version="idx",
    )
    observation = SearchObservation(documents=(doc,), trace_id="trace-1", index_version="idx")
    action = SearchAction(query="round trip", top_k=1, recall_profile="mock")
    record = SearchSnapshotRecord.from_action(snapshot_id="snap-jsonl", action=action, observation=observation)
    path = tmp_path / "snap.jsonl"
    path.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    store = load_snapshot_store(path)
    assert store.snapshot_id == "snap-jsonl"
    assert len(store.records) == 1
