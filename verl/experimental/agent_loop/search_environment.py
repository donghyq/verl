"""Versioned search environment contracts and lightweight adapters.

This module intentionally targets the M0/M1 prototype stage:

- versioned action / observation / trace / snapshot contracts
- deterministic in-memory mock adapter
- snapshot replay adapter without real network dependencies
- stable hashing helpers for environment identity and replay

The contracts are designed to be consumed by :mod:`verl.experimental.agent_loop`
without modifying core PPO / GRPO training logic.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SEARCH_ENV_SCHEMA_VERSION = "search_env.v1"
SEARCH_SNAPSHOT_SCHEMA_VERSION = "search_snapshot.v1"
SEARCH_RENDERER_VERSION = "search_renderer.v1"

DEFAULT_ALLOWED_RECALL_PROFILES = ("mock", "snapshot", "hybrid", "lexical")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _tokenize_lexical(text: str) -> list[str]:
    base_tokens = re.findall(r"[\w\-]+", text.lower())
    expanded_tokens: list[str] = []
    for token in base_tokens:
        expanded_tokens.append(token)
        expanded_tokens.extend(part for part in re.split(r"[_\-]", token) if part and part != token)
    return expanded_tokens


class SearchAction(BaseModel):
    """Versioned search action contract for the M1 environment prototype."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    top_k: int = 10
    recall_profile: str = "mock"
    filters: Optional[dict[str, Any]] = None
    geo: Optional[dict[str, Any]] = None
    rerank: Optional[bool] = None

    def identity_payload(self) -> dict[str, Any]:
        return {
            "query": _normalize_query(self.query),
            "top_k": int(self.top_k),
            "recall_profile": str(self.recall_profile),
            "filters": self.filters,
            "geo": self.geo,
            "rerank": self.rerank,
        }

    def identity(self) -> str:
        payload = {
            "schema_version": SEARCH_ENV_SCHEMA_VERSION,
            "action": self.identity_payload(),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @classmethod
    def validate_payload(
        cls,
        payload: Any,
        *,
        max_query_length: int = 256,
        min_top_k: int = 1,
        max_top_k: int = 20,
        allowed_recall_profiles: tuple[str, ...] = DEFAULT_ALLOWED_RECALL_PROFILES,
    ) -> "SearchActionValidationResult":
        """Validate arbitrary payload without throwing to the caller.

        Unknown fields, malformed payloads, or value-range errors are returned as
        structured results instead of exceptions so an agent loop can convert them
        into environment observations.
        """

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                return SearchActionValidationResult(
                    ok=False,
                    error_type="malformed_payload",
                    error_message=f"Malformed JSON payload: {exc}",
                )

        if not isinstance(payload, dict):
            return SearchActionValidationResult(
                ok=False,
                error_type="malformed_payload",
                error_message=f"Search action payload must be a JSON object, got {type(payload).__name__}",
            )

        unknown_fields = sorted(set(payload) - set(cls.model_fields))
        if unknown_fields:
            return SearchActionValidationResult(
                ok=False,
                error_type="unknown_fields",
                error_message=f"Unknown search action fields: {unknown_fields}",
                unknown_fields=unknown_fields,
            )

        try:
            action = cls.model_validate(payload)
        except ValidationError as exc:
            return SearchActionValidationResult(
                ok=False,
                error_type="validation_error",
                error_message=exc.json(),
            )

        query = _normalize_query(action.query)
        if not query:
            return SearchActionValidationResult(
                ok=False,
                error_type="empty_query",
                error_message="Search query must be non-empty.",
            )
        if len(query) > max_query_length:
            return SearchActionValidationResult(
                ok=False,
                error_type="query_too_long",
                error_message=(
                    f"Search query length {len(query)} exceeds max_query_length={max_query_length}."
                ),
            )
        if action.top_k < min_top_k or action.top_k > max_top_k:
            return SearchActionValidationResult(
                ok=False,
                error_type="top_k_out_of_range",
                error_message=f"top_k={action.top_k} must be in [{min_top_k}, {max_top_k}]",
            )
        if action.recall_profile not in allowed_recall_profiles:
            return SearchActionValidationResult(
                ok=False,
                error_type="invalid_recall_profile",
                error_message=(
                    f"recall_profile={action.recall_profile!r} not in allowlist {list(allowed_recall_profiles)}"
                ),
            )

        if query != action.query:
            action = action.model_copy(update={"query": query})

        return SearchActionValidationResult(ok=True, action=action)


class SearchActionValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    action: Optional[SearchAction] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    unknown_fields: list[str] = Field(default_factory=list)


class SearchDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    content: str
    content_hash: str
    score: float
    source: str
    index_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_content(
        cls,
        *,
        doc_id: str,
        content: str,
        score: float,
        source: str,
        index_version: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "SearchDocument":
        return cls(
            doc_id=doc_id,
            content=content,
            content_hash=stable_content_hash(content),
            score=float(score),
            source=source,
            index_version=index_version,
            metadata=metadata or {},
        )


class SearchCost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_latency_ms: float = 0.0
    queue_latency_ms: float = 0.0
    cache_hit: bool = False
    candidate_count: int = 0
    returned_count: int = 0
    recall_profile: str = ""
    rerank_used: bool = False
    degraded: bool = False


def compute_observation_response_hash(
    *,
    schema_version: str,
    index_version: str,
    documents: tuple[SearchDocument, ...],
    degraded: bool,
    error_type: Optional[str],
    error_message: Optional[str],
) -> str:
    payload = {
        "schema_version": schema_version,
        "index_version": index_version,
        "documents": [
            {
                "doc_id": doc.doc_id,
                "content_hash": doc.content_hash,
                "score": float(doc.score),
                "source": doc.source,
                "index_version": doc.index_version,
            }
            for doc in documents
        ],
        "degraded": bool(degraded),
        "error_type": error_type,
        "error_message": error_message,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class SearchObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SEARCH_ENV_SCHEMA_VERSION
    documents: tuple[SearchDocument, ...] = ()
    trace_id: str
    response_hash: str = ""
    latency_ms: float = 0.0
    cost: SearchCost = Field(default_factory=SearchCost)
    degraded: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    index_version: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.index_version and self.documents:
            object.__setattr__(self, "index_version", self.documents[0].index_version)
        if not self.response_hash:
            object.__setattr__(
                self,
                "response_hash",
                compute_observation_response_hash(
                    schema_version=self.schema_version,
                    index_version=self.index_version,
                    documents=self.documents,
                    degraded=self.degraded,
                    error_type=self.error_type,
                    error_message=self.error_message,
                ),
            )

    @classmethod
    def error(
        cls,
        *,
        trace_id: str,
        index_version: str,
        recall_profile: str,
        error_type: str,
        error_message: str,
        latency_ms: float = 0.0,
        degraded: bool = True,
        candidate_count: int = 0,
        returned_count: int = 0,
    ) -> "SearchObservation":
        return cls(
            documents=(),
            trace_id=trace_id,
            latency_ms=float(latency_ms),
            cost=SearchCost(
                total_latency_ms=float(latency_ms),
                queue_latency_ms=0.0,
                cache_hit=False,
                candidate_count=int(candidate_count),
                returned_count=int(returned_count),
                recall_profile=recall_profile,
                rerank_used=False,
                degraded=degraded,
            ),
            degraded=degraded,
            error_type=error_type,
            error_message=error_message,
            index_version=index_version,
        )


class SearchTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SEARCH_ENV_SCHEMA_VERSION
    logical_step: int
    action: dict[str, Any]
    observation: SearchObservation
    timestamp_ms: int
    index_version: str
    renderer_version: str = SEARCH_RENDERER_VERSION


class SearchSnapshotRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SEARCH_SNAPSHOT_SCHEMA_VERSION
    snapshot_id: str
    index_version: str
    action_identity: str
    observation: SearchObservation
    response_hash: str

    @classmethod
    def from_action(
        cls,
        *,
        snapshot_id: str,
        action: SearchAction,
        observation: SearchObservation,
    ) -> "SearchSnapshotRecord":
        return cls(
            snapshot_id=snapshot_id,
            index_version=observation.index_version,
            action_identity=action.identity(),
            observation=observation,
            response_hash=observation.response_hash,
        )


class SearchSnapshotStore(BaseModel):
    """In-memory snapshot store used by record/replay workflows."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SEARCH_SNAPSHOT_SCHEMA_VERSION
    snapshot_id: str
    records: list[SearchSnapshotRecord] = Field(default_factory=list)

    def add(self, record: SearchSnapshotRecord) -> None:
        self.records.append(record)

    def get_by_action_identity(self, action_identity: str) -> Optional[SearchSnapshotRecord]:
        for record in self.records:
            if record.action_identity == action_identity:
                return record
        return None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "entries": [record.model_dump(mode="json") for record in self.records],
        }


class SearchAdapter(ABC):
    """Unified async search adapter interface."""

    @abstractmethod
    async def search(self, action: SearchAction) -> SearchObservation:
        raise NotImplementedError


class RecordingSearchAdapter(SearchAdapter):
    """Adapter wrapper that records observed action->observation pairs as snapshots."""

    def __init__(
        self,
        base_adapter: SearchAdapter,
        *,
        snapshot_id: str,
        snapshot_store: Optional[SearchSnapshotStore] = None,
        deduplicate_by_action_identity: bool = True,
    ) -> None:
        self.base_adapter = base_adapter
        self.snapshot_id = snapshot_id
        self.snapshot_store = snapshot_store or SearchSnapshotStore(snapshot_id=snapshot_id)
        self.deduplicate_by_action_identity = deduplicate_by_action_identity

    async def search(self, action: SearchAction) -> SearchObservation:
        observation = await self.base_adapter.search(action)
        record = SearchSnapshotRecord.from_action(
            snapshot_id=self.snapshot_id,
            action=action,
            observation=observation,
        )
        if self.deduplicate_by_action_identity:
            existing = self.snapshot_store.get_by_action_identity(record.action_identity)
            if existing is None:
                self.snapshot_store.add(record)
        else:
            self.snapshot_store.add(record)
        return observation

    def dump(self, path: str | Path) -> None:
        dump_snapshot_records(self.snapshot_store.records, path, snapshot_id=self.snapshot_id)


class InMemorySearchAdapter(SearchAdapter):
    """Deterministic lexical/mock search adapter for CPU tests and dry runs."""

    def __init__(
        self,
        documents: list[SearchDocument | dict[str, Any]],
        *,
        index_version: str = "memory.v1",
        source: str = "in_memory",
        base_latency_ms: float = 5.0,
        query_latency_ms: Optional[dict[str, float]] = None,
        cache_hit_queries: Optional[set[str]] = None,
        empty_queries: Optional[set[str]] = None,
        degraded_queries: Optional[set[str]] = None,
        timeout_queries: Optional[set[str]] = None,
        failure_queries: Optional[dict[str, str]] = None,
        sleep_queries_ms: Optional[dict[str, float]] = None,
        explicit_results: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self.index_version = index_version
        self.source = source
        self.base_latency_ms = float(base_latency_ms)
        self.query_latency_ms = query_latency_ms or {}
        self.cache_hit_queries = cache_hit_queries or set()
        self.empty_queries = empty_queries or set()
        self.degraded_queries = degraded_queries or set()
        self.timeout_queries = timeout_queries or set()
        self.failure_queries = failure_queries or {}
        self.sleep_queries_ms = sleep_queries_ms or {}
        self.explicit_results = explicit_results or {}

        normalized_documents: list[SearchDocument] = []
        for document in documents:
            if isinstance(document, SearchDocument):
                normalized_documents.append(document)
            else:
                normalized_documents.append(
                    SearchDocument.from_content(
                        doc_id=document["doc_id"],
                        content=document["content"],
                        score=float(document.get("score", 0.0)),
                        source=document.get("source", source),
                        index_version=document.get("index_version", index_version),
                        metadata=document.get("metadata", {}),
                    )
                )
        self.documents = normalized_documents
        self._doc_by_id = {doc.doc_id: doc for doc in normalized_documents}

    def _trace_id(self, action: SearchAction) -> str:
        return f"trace-{action.identity()[:12]}"

    def _score_documents(self, action: SearchAction) -> list[SearchDocument]:
        query = _normalize_query(action.query)
        if query in self.explicit_results:
            scored = []
            for rank, doc_id in enumerate(self.explicit_results[query]):
                if doc_id in self._doc_by_id:
                    doc = self._doc_by_id[doc_id]
                    scored.append(doc.model_copy(update={"score": float(len(self.explicit_results[query]) - rank)}))
            return scored

        query_terms = _tokenize_lexical(query)
        scored_documents: list[SearchDocument] = []
        for document in self.documents:
            content_terms = _tokenize_lexical(document.content)
            overlap = sum(content_terms.count(term) for term in query_terms)
            if overlap <= 0:
                continue
            score = float(overlap) + (1.0 / (1 + len(document.doc_id)))
            scored_documents.append(document.model_copy(update={"score": score}))
        scored_documents.sort(key=lambda doc: (-float(doc.score), doc.doc_id))
        return scored_documents

    async def search(self, action: SearchAction) -> SearchObservation:
        trace_id = self._trace_id(action)
        query = _normalize_query(action.query)
        sleep_ms = float(self.sleep_queries_ms.get(query, self.query_latency_ms.get(query, self.base_latency_ms)))

        try:
            if sleep_ms > 0:
                await asyncio.sleep(sleep_ms / 1000.0)

            if query in self.failure_queries:
                return SearchObservation.error(
                    trace_id=trace_id,
                    index_version=self.index_version,
                    recall_profile=action.recall_profile,
                    error_type="adapter_failure",
                    error_message=self.failure_queries[query],
                    latency_ms=sleep_ms,
                )

            if query in self.timeout_queries:
                return SearchObservation.error(
                    trace_id=trace_id,
                    index_version=self.index_version,
                    recall_profile=action.recall_profile,
                    error_type="timeout",
                    error_message="Injected search timeout.",
                    latency_ms=sleep_ms,
                )

            if query in self.empty_queries:
                documents: tuple[SearchDocument, ...] = ()
                degraded = query in self.degraded_queries
                return SearchObservation(
                    documents=documents,
                    trace_id=trace_id,
                    latency_ms=sleep_ms,
                    cost=SearchCost(
                        total_latency_ms=sleep_ms,
                        queue_latency_ms=0.0,
                        cache_hit=query in self.cache_hit_queries,
                        candidate_count=0,
                        returned_count=0,
                        recall_profile=action.recall_profile,
                        rerank_used=bool(action.rerank),
                        degraded=degraded,
                    ),
                    degraded=degraded,
                    index_version=self.index_version,
                )

            candidates = self._score_documents(action)
            documents = tuple(candidates[: action.top_k])
            degraded = query in self.degraded_queries
            return SearchObservation(
                documents=documents,
                trace_id=trace_id,
                latency_ms=sleep_ms,
                cost=SearchCost(
                    total_latency_ms=sleep_ms,
                    queue_latency_ms=0.0,
                    cache_hit=query in self.cache_hit_queries,
                    candidate_count=len(candidates),
                    returned_count=len(documents),
                    recall_profile=action.recall_profile,
                    rerank_used=bool(action.rerank),
                    degraded=degraded,
                ),
                degraded=degraded,
                index_version=self.index_version,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            return SearchObservation.error(
                trace_id=trace_id,
                index_version=self.index_version,
                recall_profile=action.recall_profile,
                error_type="adapter_failure",
                error_message=str(exc),
                latency_ms=sleep_ms,
            )


class SnapshotReplaySearchAdapter(SearchAdapter):
    """Replay adapter backed by JSON / JSONL snapshot files."""

    def __init__(
        self,
        snapshot_path: str | Path,
        *,
        expected_schema_version: str = SEARCH_SNAPSHOT_SCHEMA_VERSION,
        expected_index_version: Optional[str] = None,
        fallback_adapter: Optional[SearchAdapter] = None,
        fallback_on_miss: bool = False,
    ) -> None:
        self.snapshot_path = Path(snapshot_path)
        self.expected_schema_version = expected_schema_version
        self.expected_index_version = expected_index_version
        self.fallback_adapter = fallback_adapter
        self.fallback_on_miss = fallback_on_miss
        self.records = self._load_records(self.snapshot_path)
        self._record_by_identity = {record.action_identity: record for record in self.records}

    def _load_records(self, path: Path) -> list[SearchSnapshotRecord]:
        raw_text = path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return []

        records: list[SearchSnapshotRecord] = []
        if path.suffix == ".jsonl":
            for line in raw_text.splitlines():
                if line.strip():
                    records.append(SearchSnapshotRecord.model_validate_json(line))
            return records

        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and "entries" in parsed:
            for entry in parsed["entries"]:
                records.append(SearchSnapshotRecord.model_validate(entry))
            return records
        if isinstance(parsed, list):
            for entry in parsed:
                records.append(SearchSnapshotRecord.model_validate(entry))
            return records
        records.append(SearchSnapshotRecord.model_validate(parsed))
        return records

    async def search(self, action: SearchAction) -> SearchObservation:
        identity = action.identity()
        record = self._record_by_identity.get(identity)
        if record is None:
            if self.fallback_adapter is not None and self.fallback_on_miss:
                return await self.fallback_adapter.search(action)
            return SearchObservation.error(
                trace_id=f"snapshot-miss-{identity[:12]}",
                index_version=self.expected_index_version or "snapshot.unknown",
                recall_profile=action.recall_profile,
                error_type="snapshot_miss",
                error_message=f"No snapshot record for action_identity={identity}",
            )

        if record.schema_version != self.expected_schema_version:
            return SearchObservation.error(
                trace_id=record.observation.trace_id,
                index_version=record.index_version,
                recall_profile=action.recall_profile,
                error_type="snapshot_schema_version_mismatch",
                error_message=(
                    f"Expected schema_version={self.expected_schema_version}, got {record.schema_version}"
                ),
                latency_ms=record.observation.latency_ms,
            )

        if self.expected_index_version is not None and record.index_version != self.expected_index_version:
            return SearchObservation.error(
                trace_id=record.observation.trace_id,
                index_version=record.index_version,
                recall_profile=action.recall_profile,
                error_type="snapshot_index_version_mismatch",
                error_message=(
                    f"Expected index_version={self.expected_index_version}, got {record.index_version}"
                ),
                latency_ms=record.observation.latency_ms,
            )

        recomputed_hash = compute_observation_response_hash(
            schema_version=record.observation.schema_version,
            index_version=record.observation.index_version,
            documents=record.observation.documents,
            degraded=record.observation.degraded,
            error_type=record.observation.error_type,
            error_message=record.observation.error_message,
        )
        if recomputed_hash != record.response_hash or record.observation.response_hash != record.response_hash:
            return SearchObservation.error(
                trace_id=record.observation.trace_id,
                index_version=record.index_version,
                recall_profile=action.recall_profile,
                error_type="snapshot_hash_mismatch",
                error_message=(
                    "Snapshot response_hash does not match serialized observation content."
                ),
                latency_ms=record.observation.latency_ms,
            )

        return record.observation


class RenderedSearchObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    token_ids: list[int]
    truncated: bool = False
    renderer_version: str = SEARCH_RENDERER_VERSION
    rendered_document_count: int = 0


class SearchObservationRenderer:
    """Stable text renderer for search observations."""

    def __init__(
        self,
        *,
        renderer_version: str = SEARCH_RENDERER_VERSION,
        max_documents: int = 3,
        max_document_tokens: int = 96,
    ) -> None:
        self.renderer_version = renderer_version
        self.max_documents = max_documents
        self.max_document_tokens = max_document_tokens

    async def render(
        self,
        observation: SearchObservation,
        *,
        encode_text,
        decode_tokens,
        max_total_tokens: int,
    ) -> RenderedSearchObservation:
        lines = [
            (
                f"[search_observation schema={observation.schema_version} renderer={self.renderer_version} "
                f"trace_id={observation.trace_id}]"
            ),
            f"response_hash={observation.response_hash}",
            f"index_version={observation.index_version}",
            f"latency_ms={observation.latency_ms:.2f}",
            f"degraded={str(observation.degraded).lower()}",
        ]

        if observation.error_type is not None:
            lines.append(f"error_type={observation.error_type}")
            lines.append(f"error_message={observation.error_message or ''}")
        elif not observation.documents:
            lines.append("documents=0")
            lines.append("status=empty_result")
        else:
            lines.append(f"documents={min(len(observation.documents), self.max_documents)}")
            for index, document in enumerate(observation.documents[: self.max_documents], start=1):
                content_ids = await encode_text(document.content)
                truncated = len(content_ids) > self.max_document_tokens
                if truncated:
                    content_text = await decode_tokens(content_ids[: self.max_document_tokens])
                    content_text = content_text + " …[truncated]"
                else:
                    content_text = document.content
                lines.append(
                    (
                        f"[doc {index}] doc_id={document.doc_id} score={float(document.score):.4f} "
                        f"source={document.source} content_hash={document.content_hash} "
                        f"index_version={document.index_version}"
                    )
                )
                lines.append(f"content={content_text}")

        text = "\n".join(lines)
        token_ids = await encode_text(text)
        truncated = False
        if max_total_tokens > 0 and len(token_ids) > max_total_tokens:
            token_ids = token_ids[:max_total_tokens]
            text = await decode_tokens(token_ids)
            truncated = True

        return RenderedSearchObservation(
            text=text,
            token_ids=token_ids,
            truncated=truncated,
            renderer_version=self.renderer_version,
            rendered_document_count=min(len(observation.documents), self.max_documents),
        )


def dump_snapshot_records(
    records: list[SearchSnapshotRecord],
    path: str | Path,
    *,
    snapshot_id: Optional[str] = None,
) -> None:
    path = Path(path)
    entries = [record.model_dump(mode="json") for record in records]
    payload = {
        "schema_version": SEARCH_SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id or (records[0].snapshot_id if records else "snapshot"),
        "entries": entries,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_snapshot_store(path: str | Path) -> SearchSnapshotStore:
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return SearchSnapshotStore(snapshot_id=path.stem)

    if path.suffix == ".jsonl":
        records = [SearchSnapshotRecord.model_validate_json(line) for line in raw_text.splitlines() if line.strip()]
        snapshot_id = records[0].snapshot_id if records else path.stem
        return SearchSnapshotStore(snapshot_id=snapshot_id, records=records)

    parsed = json.loads(raw_text)
    if isinstance(parsed, dict) and "entries" in parsed:
        records = [SearchSnapshotRecord.model_validate(entry) for entry in parsed.get("entries", [])]
        return SearchSnapshotStore(
            schema_version=parsed.get("schema_version", SEARCH_SNAPSHOT_SCHEMA_VERSION),
            snapshot_id=parsed.get("snapshot_id", path.stem),
            records=records,
        )
    if isinstance(parsed, list):
        records = [SearchSnapshotRecord.model_validate(entry) for entry in parsed]
        snapshot_id = records[0].snapshot_id if records else path.stem
        return SearchSnapshotStore(snapshot_id=snapshot_id, records=records)

    record = SearchSnapshotRecord.model_validate(parsed)
    return SearchSnapshotStore(snapshot_id=record.snapshot_id, records=[record])
