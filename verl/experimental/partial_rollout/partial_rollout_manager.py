"""Partial rollout manager: pause / save / resume / complete / cancel.

Phase A is a pure-Python closed loop. The manager owns live :class:`RolloutState`
objects and produces :class:`RolloutStateSnapshot` objects on pause. It tracks
the metrics needed to compare token-only resume against KV-aware resume:

- ``recomputed_tokens``: tokens that must be regenerated when KV cache is not
  reused. KV-aware resume -> 0; token-only resume -> len(generated_ids).
- ``save_latency`` / ``resume_latency``: wall-clock time spent in pause/resume.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from verl.experimental.partial_rollout.rollout_state import (
    InvalidResumeError,
    RolloutLifecycle,
    RolloutState,
    RolloutStateSnapshot,
    new_request_id,
)


@dataclass
class RolloutMetrics:
    """Aggregate metrics tracked by the manager."""

    recomputed_tokens: int = 0
    save_latency: float = 0.0
    resume_latency: float = 0.0
    pause_count: int = 0
    resume_count: int = 0
    complete_count: int = 0
    cancel_count: int = 0
    kv_reused_count: int = 0
    invalidated_count: int = 0

    def as_dict(self) -> dict:
        return {
            "recomputed_tokens": self.recomputed_tokens,
            "save_latency": self.save_latency,
            "resume_latency": self.resume_latency,
            "pause_count": self.pause_count,
            "resume_count": self.resume_count,
            "complete_count": self.complete_count,
            "cancel_count": self.cancel_count,
            "kv_reused_count": self.kv_reused_count,
            "invalidated_count": self.invalidated_count,
        }


class PartialRolloutManager:
    """Manage the lifecycle of partial rollouts.

    The manager models a single serving engine's view of in-flight rollouts.
    ``model_weight_version`` and ``tokenizer_fingerprint`` represent the
    currently loaded model; they are checked on every resume so a stale snapshot
    (captured before a weight update) cannot be silently reused.

    KV handle semantics:
    - On ``pause_rollout`` the KV handle (if any) is preserved in the snapshot.
    - On ``resume_rollout``:
      * If the snapshot carries a ``kv_handle`` AND it matches a still-resident
        handle in the manager, the KV cache is reused -> ``recomputed_tokens = 0``.
      * Otherwise the KV cache must be recomputed from scratch ->
        ``recomputed_tokens = len(generated_ids)``.
    """

    def __init__(
        self,
        model_weight_version: str = "v0",
        tokenizer_fingerprint: str = "tok-v0",
    ) -> None:
        self.model_weight_version = model_weight_version
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self._states: dict[str, RolloutState] = {}
        self._resident_kv: dict[str, str] = {}
        self.metrics = RolloutMetrics()

    def create_rollout(
        self,
        prompt_ids: list[int],
        sampling_params: Optional[dict] = None,
        generated_ids: Optional[list[int]] = None,
        request_id: Optional[str] = None,
        kv_handle: Optional[str] = None,
    ) -> RolloutState:
        """Register a new in-flight rollout.

        In Phase A this is a test helper: a real deployment would create the
        rollout implicitly when generation starts. ``generated_ids`` may be
        non-empty to simulate a rollout that already produced some tokens before
        being paused.
        """
        rid = request_id or new_request_id()
        if rid in self._states:
            raise ValueError(f"request_id {rid!r} already exists")
        state = RolloutState(
            request_id=rid,
            prompt_ids=list(prompt_ids),
            generated_ids=list(generated_ids or []),
            sampling_params=dict(sampling_params or {}),
            sequence_position=len(prompt_ids),
            kv_handle=kv_handle,
            model_weight_version=self.model_weight_version,
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            lifecycle=RolloutLifecycle.RUNNING,
        )
        self._states[rid] = state
        if state.kv_handle is not None:
            self._resident_kv[rid] = state.kv_handle
        return state

    def pause_rollout(self, request_id: str) -> RolloutStateSnapshot:
        """Pause a running rollout and return a serializable snapshot.

        The snapshot retains the KV handle so the caller can decide later
        whether to resume KV-aware (keep handle) or token-only (clear handle).
        The live state transitions to ``PAUSED`` then ``RESIDENT`` or
        ``OFFLOADED`` depending on whether a KV handle exists.
        """
        state = self._get_state(request_id)
        if state.lifecycle != RolloutLifecycle.RUNNING:
            raise InvalidResumeError(
                f"cannot pause rollout {request_id!r} in lifecycle "
                f"{state.lifecycle.value}"
            )
        t0 = time.perf_counter()
        state.lifecycle = RolloutLifecycle.PAUSED
        state.paused_at = time.time()
        snapshot = state.to_snapshot()
        elapsed = time.perf_counter() - t0
        self.metrics.save_latency += elapsed
        self.metrics.pause_count += 1
        if state.kv_handle is not None:
            state.lifecycle = RolloutLifecycle.RESIDENT
            self._resident_kv[request_id] = state.kv_handle
        else:
            state.lifecycle = RolloutLifecycle.OFFLOADED
        return snapshot

    def release_kv(self, request_id: str) -> None:
        """Release (offload) the KV cache for a paused rollout.

        After this the rollout can only be resumed token-only. Mirrors SGLang's
        ``release_memory_occupation``.
        """
        state = self._get_state(request_id)
        if not state.lifecycle.is_resumable:
            raise InvalidResumeError(
                f"cannot release KV for rollout {request_id!r} in lifecycle "
                f"{state.lifecycle.value}"
            )
        self._resident_kv.pop(request_id, None)
        state.kv_handle = None
        state.lifecycle = RolloutLifecycle.OFFLOADED

    def resume_rollout(self, snapshot: RolloutStateSnapshot) -> RolloutState:
        """Resume a paused rollout from a snapshot.

        If the snapshot's ``kv_handle`` is still resident in the manager, the KV
        cache is reused (``recomputed_tokens = 0``). Otherwise all generated
        tokens must be recomputed (``recomputed_tokens = len(generated_ids)``).

        Raises :class:`InvalidResumeError` if:
        - model weight / tokenizer version does not match the manager.
        - the request_id is already live (running / resuming).
        - the request_id is in a terminal lifecycle.
        """
        t0 = time.perf_counter()
        rid = snapshot.request_id

        existing = self._states.get(rid)
        if existing is not None:
            if existing.lifecycle == RolloutLifecycle.RUNNING:
                self.metrics.invalidated_count += 1
                raise InvalidResumeError(
                    f"rollout {rid!r} is already RUNNING, cannot resume"
                )
            if existing.lifecycle == RolloutLifecycle.RESUMING:
                self.metrics.invalidated_count += 1
                raise InvalidResumeError(
                    f"rollout {rid!r} is already RESUMING, duplicate resume"
                )
            if existing.lifecycle.is_terminal:
                self.metrics.invalidated_count += 1
                raise InvalidResumeError(
                    f"cannot resume rollout {rid!r} in terminal lifecycle "
                    f"{existing.lifecycle.value}"
                )

        state = RolloutState.from_snapshot(
            snapshot, lifecycle=RolloutLifecycle.RESUMING
        )
        try:
            state.validate_resume(
                self.model_weight_version, self.tokenizer_fingerprint
            )
        except InvalidResumeError:
            if existing is not None:
                existing.lifecycle = RolloutLifecycle.INVALIDATED
                self._resident_kv.pop(rid, None)
            self.metrics.invalidated_count += 1
            raise

        kv_reused = (
            snapshot.kv_handle is not None
            and self._resident_kv.get(rid) == snapshot.kv_handle
        )
        if kv_reused:
            self.metrics.kv_reused_count += 1
            state.kv_handle = snapshot.kv_handle
        else:
            self.metrics.recomputed_tokens += snapshot.generated_length
            state.kv_handle = None

        state.lifecycle = RolloutLifecycle.RUNNING
        state.resumed_at = time.time()
        self._states[rid] = state
        if state.kv_handle is not None:
            self._resident_kv[rid] = state.kv_handle
        else:
            self._resident_kv.pop(rid, None)

        elapsed = time.perf_counter() - t0
        self.metrics.resume_latency += elapsed
        self.metrics.resume_count += 1
        return state

    def complete_rollout(self, request_id: str) -> RolloutState:
        """Mark a rollout as completed and reclaim resources."""
        state = self._get_state(request_id)
        if state.lifecycle.is_terminal:
            raise InvalidResumeError(
                f"cannot complete rollout {request_id!r} in terminal lifecycle "
                f"{state.lifecycle.value}"
            )
        state.lifecycle = RolloutLifecycle.COMPLETED
        state.stop_reason = state.stop_reason or "completed"
        self._reclaim(request_id)
        self.metrics.complete_count += 1
        return state

    def cancel_rollout(self, request_id: str) -> RolloutState:
        """Mark a rollout as cancelled and reclaim resources."""
        state = self._get_state(request_id)
        if state.lifecycle.is_terminal:
            raise InvalidResumeError(
                f"cannot cancel rollout {request_id!r} in terminal lifecycle "
                f"{state.lifecycle.value}"
            )
        state.lifecycle = RolloutLifecycle.CANCELLED
        state.stop_reason = state.stop_reason or "cancelled"
        self._reclaim(request_id)
        self.metrics.cancel_count += 1
        return state

    def get_state(self, request_id: str) -> Optional[RolloutState]:
        return self._states.get(request_id)

    def has_resident_kv(self, request_id: str) -> bool:
        return request_id in self._resident_kv

    @property
    def active_count(self) -> int:
        return sum(
            1 for s in self._states.values() if not s.lifecycle.is_terminal
        )

    def update_model_version(self, model_weight_version: str) -> None:
        """Simulate a weight update.

        Any paused snapshot captured before this update will fail
        ``validate_resume`` with a version mismatch.
        """
        self.model_weight_version = model_weight_version

    def update_tokenizer(self, tokenizer_fingerprint: str) -> None:
        self.tokenizer_fingerprint = tokenizer_fingerprint

    def _get_state(self, request_id: str) -> RolloutState:
        state = self._states.get(request_id)
        if state is None:
            raise KeyError(f"unknown request_id {request_id!r}")
        return state

    def _reclaim(self, request_id: str) -> None:
        self._resident_kv.pop(request_id, None)
        state = self._states.get(request_id)
        if state is not None:
            state.kv_handle = None
