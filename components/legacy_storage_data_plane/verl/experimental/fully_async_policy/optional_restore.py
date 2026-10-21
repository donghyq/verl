"""Bounded admission at the optional I/O owner, never on the metadata Actor.

The owner is the long-lived inference service. A disconnected trajectory caller
does not cancel its internal operation or release bytes prematurely. If that
service itself disappears, permits remain held until an operator confirms the
transport has stopped; no timeout pretends to prove backend cancellation.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from verl.experimental.fully_async_policy.transfer_admission import TransferIntent, TransferRequest


@dataclass(frozen=True)
class OptionalRestoreResult:
    used_optional_io: bool
    value: Any = None
    reason: str = "completed"
    queue_seconds: float = 0.0


class SettledRestoreFailure(RuntimeError):
    """The backend confirmed failure/cancellation and has no remaining I/O."""


class OptionalRestoreRunner:
    def __init__(self, begin: Callable, complete: Callable, *, resource_domain: str,
                 queue_timeout_seconds: float = 1.0, poll_seconds: float = 0.02):
        if (not resource_domain or queue_timeout_seconds < 0 or poll_seconds <= 0):
            raise ValueError("optional restore requires a resource domain and bounded queue")
        self.begin, self.complete = begin, complete
        self.resource_domain = resource_domain
        self.queue_timeout_seconds, self.poll_seconds = queue_timeout_seconds, poll_seconds
        self.operations: set[asyncio.Task] = set()
        self.enabled = True
        self.unsettled_transfers: set[str] = set()

    async def run(self, request: TransferRequest, operation: Callable[[], Awaitable]) -> OptionalRestoreResult:
        if request.intent not in {TransferIntent.REMOTE_KV_RESTORE, TransferIntent.PREFETCH}:
            raise ValueError("execution admission applies only to optional KV I/O")
        if request.metadata.get("resource_domain") != self.resource_domain:
            raise ValueError("restore is charged to the wrong physical resource domain")
        if not self.enabled:
            return OptionalRestoreResult(False, reason="disabled")
        # Shield acquisition as well as I/O: otherwise caller cancellation could
        # land after ADMIT was recorded but before the operation owns its permit.
        task = asyncio.create_task(self._run_owned(request, operation))
        self.operations.add(task)
        task.add_done_callback(self._finished)
        return await asyncio.shield(task)

    def _finished(self, task):
        self.operations.discard(task)
        if not task.cancelled():
            task.exception()  # Consume errors when the original caller exited.

    async def _run_owned(self, request, operation):
        started = time.monotonic()
        while True:
            if not self.enabled:
                return OptionalRestoreResult(False, reason="disabled", queue_seconds=time.monotonic() - started)
            try:
                decision = await self.begin(request)
            except Exception:
                # Acquisition may have succeeded remotely. Do not invent a
                # completion or start optional transport after an uncertain RPC.
                self.unsettled_transfers.add(request.transfer_id)
                return OptionalRestoreResult(False, reason="admission_unavailable",
                    queue_seconds=time.monotonic() - started)
            if decision is None or decision["shadow"] or decision["action"] == "admit":
                break
            if decision["action"] == "reject":
                return OptionalRestoreResult(False, reason=decision["reason"],
                    queue_seconds=time.monotonic() - started)
            elapsed = time.monotonic() - started
            if elapsed >= self.queue_timeout_seconds:
                return OptionalRestoreResult(False, reason="queue_timeout", queue_seconds=elapsed)
            await asyncio.sleep(min(self.poll_seconds, self.queue_timeout_seconds - elapsed))
        queue_seconds = time.monotonic() - started
        if not self.enabled:
            # An ADMIT arrived after disabling but no transport was launched.
            if decision is not None:
                try:
                    await self.complete(request.transfer_id)
                except Exception:
                    self.unsettled_transfers.add(request.transfer_id)
                    raise
            return OptionalRestoreResult(False, reason="disabled", queue_seconds=queue_seconds)
        settled = False
        try:
            value = await operation()
            settled = True
            return OptionalRestoreResult(True, value=value, queue_seconds=queue_seconds)
        except SettledRestoreFailure:
            settled = True
            raise
        finally:
            # The supplied operation must report backend completion/failure,
            # not HTTP submission. Its own cancellation contract must settle
            # actual I/O before returning or raising.
            if decision is not None:
                if settled:
                    try:
                        await self.complete(request.transfer_id)
                        self.unsettled_transfers.discard(request.transfer_id)
                    except Exception:
                        self.unsettled_transfers.add(request.transfer_id)
                        raise
                else:
                    self.unsettled_transfers.add(request.transfer_id)

    async def disable(self, wait_timeout_seconds: float = 1.0) -> dict:
        """Stop new optional I/O; drain actual operations without cancelling them.

        A nonempty result is a visible incomplete shutdown, not proof of
        cancellation. Operators must confirm backend termination before
        clearing unresolved permits at the existing metadata authority.
        """
        if wait_timeout_seconds < 0:
            raise ValueError("disable timeout must be nonnegative")
        self.enabled = False
        if self.operations:
            await asyncio.wait(tuple(self.operations), timeout=wait_timeout_seconds)
        return {"enabled": False, "pending_operations": len(self.operations),
            "unsettled_transfer_ids": sorted(self.unsettled_transfers)}
