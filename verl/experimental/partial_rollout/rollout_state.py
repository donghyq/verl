"""Rollout state, lifecycle and snapshot primitives for partial rollout.

Phase A keeps this module dependency-free (pure stdlib) so the pause/save/
resume/reclaim closed loop can be exercised in isolation without GPU, Ray or
torch. A ``kv_handle`` is modelled as an opaque string token; in a real SGLang
deployment it would correspond to the server-side KV cache block id list that
``pause_generation`` / ``release_memory_occupation`` / ``resume_memory_occupation``
operate on.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class InvalidResumeError(RuntimeError):
    """Raised when a snapshot cannot be resumed.

    Reasons include model weight / tokenizer version drift, resuming an already
    running / completed / cancelled rollout, or attempting a duplicate resume.
    """


class RolloutLifecycle(str, Enum):
    """Lifecycle of a single rollout request.

    - ``RUNNING``: actively generating (or freshly created).
    - ``PAUSED``: generation paused, state captured; KV cache fate undecided.
    - ``RESIDENT``: paused but KV cache still resident in accelerator memory.
    - ``OFFLOADED``: paused and KV cache released / offloaded to host.
    - ``RESUMING``: snapshot accepted, rollout is being restored.
    - ``COMPLETED``: finished normally, resources reclaimed.
    - ``EXPIRED``: reserved for TTL-based eviction (not used in Phase A).
    - ``CANCELLED``: cancelled by the caller, resources reclaimed.
    - ``INVALIDATED``: a resume attempt failed (e.g. version drift), making the
      paused state unusable.
    """

    RUNNING = "running"
    PAUSED = "paused"
    RESIDENT = "resident"
    OFFLOADED = "offloaded"
    RESUMING = "resuming"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"

    @property
    def is_terminal(self) -> bool:
        return self in (
            RolloutLifecycle.COMPLETED,
            RolloutLifecycle.CANCELLED,
            RolloutLifecycle.EXPIRED,
            RolloutLifecycle.INVALIDATED,
        )

    @property
    def is_resumable(self) -> bool:
        return self in (
            RolloutLifecycle.PAUSED,
            RolloutLifecycle.RESIDENT,
            RolloutLifecycle.OFFLOADED,
        )


@dataclass
class RolloutState:
    """Live, mutable rollout state held by :class:`PartialRolloutManager`."""

    request_id: str
    prompt_ids: list[int]
    generated_ids: list[int]
    sampling_params: dict[str, Any]
    sequence_position: int
    stop_reason: Optional[str] = None
    kv_handle: Optional[str] = None
    model_weight_version: str = "v0"
    tokenizer_fingerprint: str = "tok-v0"
    lifecycle: RolloutLifecycle = RolloutLifecycle.RUNNING
    created_at: float = field(default_factory=time.time)
    paused_at: Optional[float] = None
    resumed_at: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sequence_position < 0:
            raise ValueError("sequence_position must be >= 0")
        if self.sequence_position == 0 and self.prompt_ids:
            self.sequence_position = len(self.prompt_ids)

    def total_length(self) -> int:
        return len(self.prompt_ids) + len(self.generated_ids)

    def validate_resume(
        self,
        model_weight_version: str,
        tokenizer_fingerprint: str,
    ) -> None:
        """Verify that this state can be resumed under the given versions.

        Raises :class:`InvalidResumeError` on any mismatch. A mismatch on model
        weights or tokenizer makes the captured token ids / KV cache unsafe to
        reuse, so the caller (the manager) should invalidate the paused state.
        """
        if self.model_weight_version != model_weight_version:
            raise InvalidResumeError(
                f"model_weight_version mismatch: snapshot has "
                f"{self.model_weight_version!r} but manager is "
                f"{model_weight_version!r}"
            )
        if self.tokenizer_fingerprint != tokenizer_fingerprint:
            raise InvalidResumeError(
                f"tokenizer_fingerprint mismatch: snapshot has "
                f"{self.tokenizer_fingerprint!r} but manager is "
                f"{tokenizer_fingerprint!r}"
            )

    def to_snapshot(self) -> "RolloutStateSnapshot":
        return RolloutStateSnapshot(
            request_id=self.request_id,
            prompt_ids=list(self.prompt_ids),
            generated_ids=list(self.generated_ids),
            sampling_params=dict(self.sampling_params),
            sequence_position=self.sequence_position,
            stop_reason=self.stop_reason,
            kv_handle=self.kv_handle,
            model_weight_version=self.model_weight_version,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            created_at=self.created_at,
            paused_at=self.paused_at if self.paused_at is not None else time.time(),
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: "RolloutStateSnapshot",
        lifecycle: RolloutLifecycle = RolloutLifecycle.RESUMING,
    ) -> "RolloutState":
        return cls(
            request_id=snapshot.request_id,
            prompt_ids=list(snapshot.prompt_ids),
            generated_ids=list(snapshot.generated_ids),
            sampling_params=dict(snapshot.sampling_params),
            sequence_position=snapshot.sequence_position,
            stop_reason=snapshot.stop_reason,
            kv_handle=snapshot.kv_handle,
            model_weight_version=snapshot.model_weight_version,
            tokenizer_fingerprint=snapshot.tokenizer_fingerprint,
            lifecycle=lifecycle,
            created_at=snapshot.created_at,
            paused_at=snapshot.paused_at,
            resumed_at=time.time(),
        )


@dataclass
class RolloutStateSnapshot:
    """Serializable, side-effect-free view of a paused rollout.

    This is the unit handed across save/load boundaries. It intentionally
    carries no live lifecycle: a snapshot only ever represents a paused point in
    time. ``to_dict`` / ``from_dict`` round-trip through plain JSON-friendly
    types so it can be persisted (e.g. to disk or a queue) between processes.
    """

    request_id: str
    prompt_ids: list[int]
    generated_ids: list[int]
    sampling_params: dict[str, Any]
    sequence_position: int
    stop_reason: Optional[str] = None
    kv_handle: Optional[str] = None
    model_weight_version: str = "v0"
    tokenizer_fingerprint: str = "tok-v0"
    created_at: float = 0.0
    paused_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RolloutStateSnapshot":
        known = {
            "request_id",
            "prompt_ids",
            "generated_ids",
            "sampling_params",
            "sequence_position",
            "stop_reason",
            "kv_handle",
            "model_weight_version",
            "tokenizer_fingerprint",
            "created_at",
            "paused_at",
        }
        extra = set(data) - known
        if extra:
            raise ValueError(f"unknown snapshot keys: {sorted(extra)}")
        return cls(**{k: data.get(k) for k in known})

    @property
    def generated_length(self) -> int:
        return len(self.generated_ids)

    def with_kv_handle(self, kv_handle: Optional[str]) -> "RolloutStateSnapshot":
        """Return a copy with the KV handle replaced.

        Used to simulate a token-only resume by clearing the handle (the KV
        cache was released / offloaded and can no longer be reused).
        """
        return RolloutStateSnapshot(
            request_id=self.request_id,
            prompt_ids=list(self.prompt_ids),
            generated_ids=list(self.generated_ids),
            sampling_params=dict(self.sampling_params),
            sequence_position=self.sequence_position,
            stop_reason=self.stop_reason,
            kv_handle=kv_handle,
            model_weight_version=self.model_weight_version,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            created_at=self.created_at,
            paused_at=self.paused_at,
        )


def new_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:12]}"


def new_kv_handle() -> str:
    """Mint an opaque KV cache handle.

    In production this would be the list of KV block ids returned by SGLang's
    ``pause_generation``; here it is an opaque token the manager can compare for
    reuse vs. release.
    """
    return f"kv-{uuid.uuid4().hex[:16]}"
