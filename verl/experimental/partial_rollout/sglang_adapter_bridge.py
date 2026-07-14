"""Phase C: async adapter bridge mirroring the real SGLang rollout adapter.

The real ``verl/workers/rollout/sglang_rollout/sglang_rollout.py`` adapter
exposes three async methods that control memory lifecycle:

    async def release(self)            # release_memory_occupation(tags=...)
    async def resume(self, tags)       # resume_memory_occupation(tags=tags)
    async def update_weights(weights, global_steps, **kwargs)

``sleep_level`` controls what ``release`` frees: 1 = kv_cache only (LoRA path),
2 = kv_cache + weights (merge path).

:class:`SGLangAdapterBridge` implements the same async contract but backed by
the pure-Python :class:`FakeSGLangServer` + :class:`PartialRolloutManager`,
so test code that expects the adapter interface can run without GPU/Ray/torch.
"""

from __future__ import annotations

import asyncio
from typing import Any, Generator, Optional

from verl.experimental.partial_rollout.rollout_state import (
    InvalidResumeError,
    RolloutState,
    RolloutStateSnapshot,
)
from verl.experimental.partial_rollout.sglang_integration import (
    FakeSGLangServer,
    SGLangPartialRolloutCoordinator,
)


class SGLangAdapterBridge:
    """Async bridge with the same method contract as the real SGLang adapter.

    Unlike :class:`SGLangPartialRolloutCoordinator` (sync), this bridge exposes
    ``async def`` methods matching the adapter signatures so it can be dropped
    into async test harnesses or mock-based integration tests.
    """

    def __init__(
        self,
        server: Optional[FakeSGLangServer] = None,
        model_weight_version: str = "v0",
        tokenizer_fingerprint: str = "tok-v0",
        sleep_level: int = 2,
    ) -> None:
        self.server = server or FakeSGLangServer()
        self._coord = SGLangPartialRolloutCoordinator(
            server=self.server,
            model_weight_version=model_weight_version,
            tokenizer_fingerprint=tokenizer_fingerprint,
            sleep_level=sleep_level,
        )
        self.sleep_level = sleep_level

    @property
    def manager(self):
        return self._coord.manager

    @property
    def metrics(self):
        return self._coord.metrics

    @property
    def call_log(self) -> list[str]:
        return self.server.call_log

    # ------------------------------------------------------------------ #
    # sync helpers (for non-async tests)
    # ------------------------------------------------------------------ #

    def start_rollout(
        self,
        prompt_ids: list[int],
        sampling_params: Optional[dict] = None,
        generated_ids: Optional[list[int]] = None,
        request_id: Optional[str] = None,
    ) -> RolloutState:
        return self._coord.start_rollout(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            generated_ids=generated_ids,
            request_id=request_id,
        )

    def pause_and_save(self, request_id: str) -> RolloutStateSnapshot:
        return self._coord.pause_and_save(request_id)

    # ------------------------------------------------------------------ #
    # async contract — mirrors real adapter
    # ------------------------------------------------------------------ #

    async def release(self) -> list[str]:
        """Release GPU memory, mirroring ``adapter.release()``.

        Returns the tags that were sent to ``release_memory_occupation``.
        """
        tags = ["kv_cache"] if self.sleep_level == 1 else ["kv_cache", "weights"]
        active_paused = [
            rid
            for rid, s in self.manager._states.items()
            if s.lifecycle.is_resumable
        ]
        for rid in active_paused:
            self.server.release_memory_occupation(rid, tags=tags)
            self.manager.release_kv(rid)
        return tags

    async def release_one(self, request_id: str) -> list[str]:
        """Release memory for a single request (used when pausing selectively)."""
        tags = ["kv_cache"] if self.sleep_level == 1 else ["kv_cache", "weights"]
        self.server.release_memory_occupation(request_id, tags=tags)
        self.manager.release_kv(request_id)
        return tags

    async def resume(self, tags: list[str]) -> None:
        """Resume memory occupation, mirroring ``adapter.resume(tags)``.

        This only resumes server-side memory; actual rollout resume is done via
        ``resume_from_snapshot``.
        """
        for rid in list(self.server.resident_kv.keys()):
            self.server.resume_memory_occupation(rid, tags=tags)
        if "weights" in tags:
            self.server.weights_loaded = True

    async def resume_from_snapshot(self, snapshot: RolloutStateSnapshot) -> RolloutState:
        """Resume a rollout from a snapshot, mirroring the adapter pattern."""
        resume_tags = ["kv_cache"]
        if not self.server.weights_loaded:
            resume_tags.append("weights")
        self.server.resume_memory_occupation(snapshot.request_id, tags=resume_tags)
        return self.manager.resume_rollout(snapshot)

    async def update_weights(
        self,
        weights: Any = None,
        global_steps: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Update model weights, mirroring ``adapter.update_weights()``.

        In the real adapter this streams weight tensors; here we just bump the
        version string so any snapshot captured before this call will fail
        ``validate_resume``.

        Returns the new weight version.
        """
        new_version = f"v{global_steps}" if global_steps is not None else f"v{int(self.manager.model_weight_version[1:]) + 1}" if self.manager.model_weight_version.startswith("v") else "v1"
        self.server.weights_loaded = False
        self.manager.update_model_version(new_version)
        self._coord.server.weights_loaded = False
        return new_version

    async def complete(self, request_id: str) -> RolloutState:
        self._coord._release_server_memory(request_id)
        return self.manager.complete_rollout(request_id)

    async def cancel(self, request_id: str) -> RolloutState:
        self._coord._release_server_memory(request_id)
        return self.manager.cancel_rollout(request_id)

    def run_async(self, coro):
        """Run a coroutine in a sync context (for non-async tests)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.ensure_future(coro)
        except RuntimeError:
            pass
        return asyncio.run(coro)
