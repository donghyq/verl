# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Production control flow from cooperative preemption to one target resume."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


class TrajectoryControlError(RuntimeError):
    """A bounded control request did not produce a verified resume result."""

    def __init__(
        self,
        status: str,
        *,
        uid: str | None = None,
        operation_id: str | None = None,
        detail: str | None = None,
        source_worker_index: int | None = None,
    ):
        self.status, self.uid, self.operation_id, self.detail = status, uid, operation_id, detail
        self.source_worker_index = source_worker_index
        message = f"trajectory control request {status}"
        if uid is not None:
            message += f" for uid={uid}"
        if operation_id is not None:
            message += f" operation_id={operation_id}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


@dataclass
class _TrajectoryOperation:
    uid: str
    operation_id: str
    task: asyncio.Task[Any]
    source_worker_index: int | None = None
    handoff_id: str | None = None
    target_worker_index: int | None = None
    requested_target_worker_index: int | None = None
    preemption_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    dispatched: bool = False
    terminal_status: str | None = None
    error: BaseException | None = field(default=None, repr=False)


class TrajectoryHandoffController:
    """Serialize control per prompt UID without duplicating uncertain PUTs."""

    def __init__(self, manager, *, max_operations: int = 128):
        if type(max_operations) is not int or max_operations <= 0:
            raise ValueError("max_operations must be a positive integer")
        self.manager = manager
        self.max_operations = max_operations
        self._operations: OrderedDict[str, _TrajectoryOperation] = OrderedDict()

    @staticmethod
    def _validate_timeout(timeout_s: float | None) -> float:
        if timeout_s is None:
            raise TrajectoryControlError("timeout_required")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise TrajectoryControlError("invalid_timeout", detail="timeout_s must be finite and positive")
        return float(timeout_s)

    @staticmethod
    def _normal_completion(status: dict | None) -> bool:
        if status is None or status.get("handoff_id") is not None or status.get("state") == "blocked":
            return False
        active = any(
            status.get(category)
            for category in (
                "pending_members",
                "starting_members",
                "running_members",
                "checkpointed_members",
                "failed_members",
            )
        )
        return status.get("publication_finished") is True and bool(status.get("completed_members")) and not active

    def _select_target(self, source_worker_index: int | None, target_worker_index: int | None) -> int:
        if target_worker_index is not None:
            valid = (
                type(target_worker_index) is int
                and 0 <= target_worker_index < len(self.manager.agent_loop_workers)
                and target_worker_index not in self.manager._draining_worker_indices
                and target_worker_index != source_worker_index
            )
            if not valid:
                raise TrajectoryControlError(
                    "no_target",
                    source_worker_index=source_worker_index,
                    detail="requested target is unavailable or is the source",
                )
            return target_worker_index
        for index in range(len(self.manager.agent_loop_workers)):
            if index not in self.manager._draining_worker_indices and index != source_worker_index:
                return index
        raise TrajectoryControlError(
            "no_target", source_worker_index=source_worker_index, detail="no worker other than the source accepts work"
        )

    async def _wait_for_released_handoff(self, uid: str) -> dict | None:
        while True:
            client = getattr(self.manager, "trajectory_coordinator_client", self.manager.trajectory_coordinator)
            rows = await client.list_recoverable_handoffs()
            matches = [row for row in rows if row.get("group_uid") == uid]
            if len(matches) > 1:
                raise TrajectoryControlError(
                    "ambiguous_handoff", uid=uid, detail="multiple released handoffs exist for this UID"
                )
            if matches:
                return matches[0]
            statuses = await self.manager.trajectory_group_status(uid)
            if any(self._normal_completion(value) for value in statuses.values()):
                return None
            await asyncio.sleep(self.manager.trajectory_status_poll_interval_s)

    async def _run_operation(
        self, operation: _TrajectoryOperation, *, target_worker_index: int | None, require_active: bool
    ) -> dict:
        try:
            source_index = operation.source_worker_index if require_active else None
            if source_index is None and require_active:
                source_index = await self._ensure_preempted(operation)

            if source_index is None:
                statuses = await self.manager.trajectory_group_status(operation.uid)
                if require_active and not statuses:
                    raise TrajectoryControlError("uid_not_found", uid=operation.uid)
                if any(self._normal_completion(value) for value in statuses.values()):
                    operation.terminal_status = "completed"
                    return {
                        "status": "already_completed",
                        "uid": operation.uid,
                        "source_worker_index": None,
                        "handoff_id": None,
                        "receipt": None,
                    }
                if require_active and not any(value is not None for value in statuses.values()):
                    raise TrajectoryControlError("uid_not_found", uid=operation.uid)
            else:
                statuses = await self.manager.trajectory_group_status(operation.uid)
                if any(self._normal_completion(value) for value in statuses.values()):
                    operation.terminal_status = "completed"
                    return {
                        "status": "already_completed",
                        "uid": operation.uid,
                        "source_worker_index": source_index,
                        "handoff_id": None,
                        "receipt": None,
                    }

            handoff = await self._wait_for_released_handoff(operation.uid)
            if handoff is None:
                operation.terminal_status = "completed"
                return {
                    "status": "already_completed",
                    "uid": operation.uid,
                    "source_worker_index": source_index,
                    "handoff_id": None,
                    "receipt": None,
                }
            operation.handoff_id = handoff["handoff_id"]
            source_index = self.manager.trajectory_worker_owner_ids.get(
                handoff.get("source_rollouter"), source_index
            )
            operation.source_worker_index = source_index
            if source_index is not None:
                # A worker-side token-KV interruption may not have passed
                # through the manager's preemption RPC. Exclude that source
                # from subsequent normal admissions while its handoff is open.
                self.manager._draining_worker_indices.add(source_index)
            requested_target = operation.requested_target_worker_index
            target_index = self._select_target(source_index, requested_target if requested_target is not None else target_worker_index)
            operation.target_worker_index = target_index

            # This is the only dispatch point. Once set, timeout/cancel cannot
            # cancel the remote resume or issue another PUT.
            before_resume = getattr(self.manager, "_before_trajectory_handoff_resume", None)
            if before_resume is not None:
                await before_resume(operation.uid, operation.handoff_id, source_index, target_index)
            operation.dispatched = True
            receipt = await self.manager.resume_trajectory_handoff(
                operation.handoff_id, target_worker_index=target_index
            )
            operation.terminal_status = "completed"
            return {
                "status": "resumed",
                "uid": operation.uid,
                "handoff_id": operation.handoff_id,
                "source_worker_index": source_index,
                "target_worker_index": target_index,
                "receipt": receipt,
            }
        except asyncio.CancelledError:
            operation.terminal_status = "cancelled"
            raise
        except BaseException as error:
            operation.error = error
            operation.terminal_status = "failed"
            raise

    def _start(
        self, uid: str, *, target_worker_index: int | None, source_worker_index: int | None, require_active: bool
    ) -> _TrajectoryOperation:
        existing = self._operations.get(uid)
        if existing is not None:
            return existing
        while len(self._operations) >= self.max_operations:
            completed_uid = next((key for key, value in self._operations.items() if value.task.done()), None)
            if completed_uid is None:
                raise TrajectoryControlError("controller_busy", uid=uid,
                                             detail="all control slots have unfinished operations")
            del self._operations[completed_uid]
        operation = _TrajectoryOperation(uid, uuid4().hex, None, source_worker_index=source_worker_index,
                                         requested_target_worker_index=target_worker_index)
        operation.task = asyncio.create_task(
            self._run_operation(operation, target_worker_index=target_worker_index, require_active=require_active)
        )
        # A dispatched operation outlives its caller. Consume a later exception
        # while retaining the task/result for status queries and safe retries.
        operation.task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        self._operations[uid] = operation
        return operation

    async def _ensure_preempted(self, operation):
        if operation.source_worker_index is not None:
            return operation.source_worker_index
        if operation.preemption_task is None:
            operation.preemption_task = asyncio.create_task(
                self.manager.request_trajectory_preemption(operation.uid)
            )
            operation.preemption_task.add_done_callback(
                lambda task: None if task.cancelled() else task.exception()
            )
        source = await asyncio.shield(operation.preemption_task)
        operation.source_worker_index = source
        return source

    async def run(
        self,
        uid: str,
        *,
        target_worker_index: int | None,
        timeout_s: float | None,
        source_worker_index: int | None = None,
        require_active: bool = True,
    ) -> dict:
        if not isinstance(uid, str) or not uid:
            raise TrajectoryControlError("invalid_uid")
        timeout = self._validate_timeout(timeout_s)
        operation = self._start(
            uid,
            target_worker_index=target_worker_index,
            source_worker_index=source_worker_index,
            require_active=require_active,
        )
        if target_worker_index is not None:
            existing_target = operation.target_worker_index if operation.dispatched else operation.requested_target_worker_index
            if existing_target is not None and existing_target != target_worker_index:
                raise TrajectoryControlError("target_conflict", uid=uid, operation_id=operation.operation_id)
            if not operation.dispatched:
                operation.requested_target_worker_index = target_worker_index

        async def wait_for_result():
            # A passive completion observer may have created this operation.
            # An explicit reclaim request must still initiate preemption once.
            if require_active and not operation.task.done() and not operation.dispatched:
                await self._ensure_preempted(operation)
            return await asyncio.shield(operation.task)
        try:
            return await asyncio.wait_for(wait_for_result(), timeout)
        except asyncio.TimeoutError:
            if operation.dispatched:
                raise TrajectoryControlError(
                    "unknown",
                    uid=uid,
                    operation_id=operation.operation_id,
                    detail="target resume was dispatched; it was not retried or cancelled",
                ) from None
            operation.task.cancel()
            try:
                await operation.task
            except (asyncio.CancelledError, BaseException):
                pass
            if self._operations.get(uid) is operation:
                self._operations.pop(uid, None)
            raise TrajectoryControlError("timeout", uid=uid, operation_id=operation.operation_id) from None
        except asyncio.CancelledError:
            if operation.dispatched:
                raise TrajectoryControlError(
                    "cancelled_with_dispatch",
                    uid=uid,
                    operation_id=operation.operation_id,
                    detail="target resume continues; its result can be queried later",
                ) from None
            operation.task.cancel()
            try:
                await operation.task
            except (asyncio.CancelledError, BaseException):
                pass
            if self._operations.get(uid) is operation:
                self._operations.pop(uid, None)
            raise TrajectoryControlError("cancelled", uid=uid, operation_id=operation.operation_id) from None
        except Exception:
            if not operation.dispatched and self._operations.get(uid) is operation:
                if not operation.task.done():
                    operation.task.cancel()
                    try:
                        await operation.task
                    except BaseException:
                        pass
                self._operations.pop(uid, None)
            raise

    def status(self, uid: str) -> dict | None:
        operation = self._operations.get(uid)
        if operation is None:
            return None
        if operation.task.done():
            try:
                task_error = operation.task.exception()
            except asyncio.CancelledError:
                task_error = asyncio.CancelledError()
            state = "failed" if task_error is not None else "completed"
        else:
            state = "dispatched" if operation.dispatched else "running"
        return {
            "uid": uid,
            "operation_id": operation.operation_id,
            "state": state,
            "source_worker_index": operation.source_worker_index,
            "handoff_id": operation.handoff_id,
            "target_worker_index": operation.target_worker_index,
            "terminal_status": operation.terminal_status,
            "error_status": operation.error.status if isinstance(operation.error, TrajectoryControlError) else None,
        }
