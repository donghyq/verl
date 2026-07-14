"""SGLang-style integration harness for partial rollout (Phase B).

This module wires Phase A's :class:`PartialRolloutManager` to SGLang-style
server APIs — ``pause_generation``, ``release_memory_occupation``,
``resume_memory_occupation`` — so the pause/save/resume/reclaim loop can be
exercised against a realistic call sequence without a real GPU server.

The :class:`FakeSGLangServer` records every server-side call so tests can assert
on the exact API sequence. The :class:`SGLangPartialRolloutCoordinator` is the
thin bridge between the server and the manager, mirroring how the real
``verl/workers/rollout/sglang_rollout/sglang_rollout.py`` adapter wraps
``release_memory_occupation`` / ``resume_memory_occupation`` with ``tags``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from verl.experimental.partial_rollout.partial_rollout_manager import (
    PartialRolloutManager,
)
from verl.experimental.partial_rollout.rollout_state import (
    InvalidResumeError,
    RolloutLifecycle,
    RolloutState,
    RolloutStateSnapshot,
    new_kv_handle,
)


@dataclass
class FakeSGLangServer:
    """In-process stand-in for an SGLang serving engine.

    Records the call log so tests can assert on the API sequence. State that
    matters for resume correctness:
    - ``resident_kv``: request_id -> kv_handle, still resident in accelerator.
    - ``released``: request_ids whose KV cache was released via
      ``release_memory_occupation``.
    - ``weights_loaded``: whether model weights are currently on-device.
    """

    weights_loaded: bool = True
    resident_kv: dict[str, str] = field(default_factory=dict)
    released: set[str] = field(default_factory=set)
    paused: set[str] = field(default_factory=set)
    call_log: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # server-side API surface
    # ------------------------------------------------------------------ #

    def pause_generation(self, request_id: str) -> None:
        self.call_log.append(f"pause_generation({request_id})")
        self.paused.add(request_id)

    def release_memory_occupation(self, request_id: str, tags: list[str]) -> None:
        self.call_log.append(
            f"release_memory_occupation({request_id}, tags={tags})"
        )
        if "kv_cache" in tags:
            self.resident_kv.pop(request_id, None)
            self.released.add(request_id)
        if "weights" in tags:
            self.weights_loaded = False

    def resume_memory_occupation(self, request_id: str, tags: list[str]) -> None:
        self.call_log.append(
            f"resume_memory_occupation({request_id}, tags={tags})"
        )
        if "weights" in tags:
            self.weights_loaded = True

    def allocate_kv(self, request_id: str) -> str:
        handle = new_kv_handle()
        self.resident_kv[request_id] = handle
        self.call_log.append(f"allocate_kv({request_id})")
        return handle

    def is_kv_resident(self, request_id: str) -> bool:
        return request_id in self.resident_kv

    @property
    def last_call(self) -> str:
        return self.call_log[-1] if self.call_log else ""


class SGLangPartialRolloutCoordinator:
    """Bridge between a :class:`FakeSGLangServer` and :class:`PartialRolloutManager`.

    Mirrors the real adapter's pattern: ``sleep_level`` controls what
    ``release_memory_occupation`` releases.
    - ``sleep_level=1``: release kv_cache only (keep base weights alive).
    - ``sleep_level=2``: release kv_cache + weights (full sleep).
    """

    def __init__(
        self,
        server: FakeSGLangServer,
        model_weight_version: str = "v0",
        tokenizer_fingerprint: str = "tok-v0",
        sleep_level: int = 2,
    ) -> None:
        self.server = server
        self.manager = PartialRolloutManager(
            model_weight_version=model_weight_version,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        self.sleep_level = sleep_level

    # ------------------------------------------------------------------ #
    # high-level operations
    # ------------------------------------------------------------------ #

    def start_rollout(
        self,
        prompt_ids: list[int],
        sampling_params: Optional[dict] = None,
        generated_ids: Optional[list[int]] = None,
        request_id: Optional[str] = None,
    ) -> RolloutState:
        """Create a rollout and allocate server-side KV cache."""
        rid = request_id or f"req-{id(prompt_ids)}"
        kv_handle = self.server.allocate_kv(rid)
        return self.manager.create_rollout(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            generated_ids=generated_ids,
            request_id=rid,
            kv_handle=kv_handle,
        )

    def pause_and_save(self, request_id: str) -> RolloutStateSnapshot:
        """Pause generation on the server, then capture manager state.

        Call sequence: ``pause_generation`` -> ``manager.pause_rollout``.
        The snapshot retains the KV handle for optional KV-aware resume.
        """
        self.server.pause_generation(request_id)
        return self.manager.pause_rollout(request_id)

    def release_memory(self, request_id: str) -> None:
        """Release GPU memory for a paused rollout.

        Call sequence: ``release_memory_occupation`` -> ``manager.release_kv``.
        After this the rollout can only be resumed token-only.
        """
        if self.sleep_level == 1:
            tags = ["kv_cache"]
        else:
            tags = ["kv_cache", "weights"]
        self.server.release_memory_occupation(request_id, tags=tags)
        self.manager.release_kv(request_id)

    def resume_from_snapshot(self, snapshot: RolloutStateSnapshot) -> RolloutState:
        """Resume a paused rollout, optionally reusing KV cache.

        Call sequence: ``resume_memory_occupation`` -> ``manager.resume_rollout``.
        KV-aware resume: snapshot keeps kv_handle, server still resident ->
        recomputed_tokens = 0.
        Token-only resume: snapshot cleared kv_handle or server released ->
        recomputed_tokens = len(generated_ids).
        """
        resume_tags = ["kv_cache"]
        if not self.server.weights_loaded:
            resume_tags.append("weights")
        self.server.resume_memory_occupation(
            snapshot.request_id, tags=resume_tags
        )
        return self.manager.resume_rollout(snapshot)

    def complete(self, request_id: str) -> RolloutState:
        self._release_server_memory(request_id)
        return self.manager.complete_rollout(request_id)

    def cancel(self, request_id: str) -> RolloutState:
        self._release_server_memory(request_id)
        return self.manager.cancel_rollout(request_id)

    def _release_server_memory(self, request_id: str) -> None:
        """Release any server-side KV still resident for this request."""
        if self.server.is_kv_resident(request_id):
            self.server.release_memory_occupation(request_id, tags=["kv_cache"])

    @property
    def metrics(self):
        return self.manager.metrics

    @property
    def call_log(self) -> list[str]:
        return self.server.call_log

    def update_weights(self, new_version: str) -> None:
        """Simulate a weight update on the server side."""
        self.server.weights_loaded = False
        self.manager.update_model_version(new_version)
