# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Intent-aware transfer admission for Fully Async rollout data movement.

This is a VeRL control-plane policy, not a replacement for Mooncake/TENT.
The default mode is shadow-only: decisions are recorded while the underlying
transfer proceeds unchanged.  An execution-enforcing mode exists for CPU
simulation and future integration, but Project B must not enable it until the
real transport API and contention behavior are validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class TransferIntent(StrEnum):
    WEIGHT_SYNC = "weight_sync"
    ROLLOUT_ARTIFACT = "rollout_artifact"
    REMOTE_KV_RESTORE = "remote_kv_restore"
    CHECKPOINT = "checkpoint"
    PREFETCH = "prefetch"


class AdmissionAction(StrEnum):
    ADMIT = "admit"
    QUEUE = "queue"
    REJECT = "reject"


class AdmissionReason(StrEnum):
    CAPACITY_AVAILABLE = "capacity_available"
    TRANSFER_SLOT_PRESSURE = "transfer_slot_pressure"
    BYTE_PRESSURE = "byte_pressure"
    BACKGROUND_BUDGET_EXHAUSTED = "background_budget_exhausted"
    REQUEST_TOO_LARGE = "request_too_large"
    RESTORE_COST_GATE_DENIED = "restore_cost_gate_denied"


_PRIORITY = {
    TransferIntent.WEIGHT_SYNC: 100,
    TransferIntent.ROLLOUT_ARTIFACT: 80,
    TransferIntent.REMOTE_KV_RESTORE: 60,
    TransferIntent.CHECKPOINT: 40,
    TransferIntent.PREFETCH: 20,
}
_BACKGROUND = {TransferIntent.CHECKPOINT, TransferIntent.PREFETCH}


@dataclass(frozen=True)
class TransferAdmissionConfig:
    shadow: bool = True
    max_inflight_transfers: int = 4
    max_inflight_bytes: int = 512 * 1024 * 1024
    max_background_inflight: int = 1
    max_records: int = 10_000

    def __post_init__(self) -> None:
        if type(self.shadow) is not bool:
            raise ValueError("shadow must be a bool")
        for name in ("max_inflight_transfers", "max_inflight_bytes", "max_records"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_background_inflight) is not int or self.max_background_inflight < 0:
            raise ValueError("max_background_inflight must be a non-negative integer")
        if self.max_background_inflight > self.max_inflight_transfers:
            raise ValueError("max_background_inflight cannot exceed max_inflight_transfers")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "TransferAdmissionConfig":
        values = dict(values or {})
        allowed = {
            "shadow",
            "max_inflight_transfers",
            "max_inflight_bytes",
            "max_background_inflight",
            "max_records",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown transfer admission config fields: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class TransferRequest:
    transfer_id: str
    intent: TransferIntent
    bytes: int
    policy_version: str | None = None
    producer_generation: int | None = None
    cost_gate_approved: bool | None = None
    deadline_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.transfer_id, str) or not self.transfer_id:
            raise ValueError("transfer_id must be a non-empty string")
        if not isinstance(self.intent, TransferIntent):
            raise ValueError("intent must be a TransferIntent")
        if type(self.bytes) is not int or self.bytes < 0:
            raise ValueError("bytes must be a non-negative integer")
        if self.policy_version is not None and (
            not isinstance(self.policy_version, str) or not self.policy_version
        ):
            raise ValueError("policy_version must be a non-empty string when provided")
        if self.producer_generation is not None and (
            type(self.producer_generation) is not int or self.producer_generation < 0
        ):
            raise ValueError("producer_generation must be a non-negative integer when provided")
        if self.cost_gate_approved is not None and type(self.cost_gate_approved) is not bool:
            raise ValueError("cost_gate_approved must be a bool when provided")
        if self.deadline_s is not None and (
            isinstance(self.deadline_s, bool)
            or not isinstance(self.deadline_s, (int, float))
            or self.deadline_s < 0
        ):
            raise ValueError("deadline_s must be a non-negative number when provided")

    @property
    def priority(self) -> int:
        return _PRIORITY[self.intent]

    @property
    def background(self) -> bool:
        return self.intent in _BACKGROUND


@dataclass(frozen=True)
class TransferAdmissionDecision:
    transfer_id: str
    intent: TransferIntent
    action: AdmissionAction
    reason: AdmissionReason
    priority: int
    active_transfers: int
    active_bytes: int
    background_active: int
    shadow: bool
    affects_execution: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "intent": str(self.intent),
            "action": str(self.action),
            "reason": str(self.reason),
            "priority": self.priority,
            "active_transfers": self.active_transfers,
            "active_bytes": self.active_bytes,
            "background_active": self.background_active,
            "shadow": self.shadow,
            "affects_execution": self.affects_execution,
        }


class TransferAdmissionController:
    """Small stateful admission model with explicit shadow semantics."""

    def __init__(self, config: TransferAdmissionConfig | None = None):
        self.config = config or TransferAdmissionConfig()
        self._active: dict[str, TransferRequest] = {}
        self._records: list[TransferAdmissionDecision] = []
        self._decision_counts = {action: 0 for action in AdmissionAction}
        self._max_observed_inflight_transfers = 0
        self._max_observed_inflight_bytes = 0

    def _active_stats(self) -> tuple[int, int, int]:
        active_transfers = len(self._active)
        active_bytes = sum(request.bytes for request in self._active.values())
        background_active = sum(request.background for request in self._active.values())
        return active_transfers, active_bytes, background_active

    def evaluate(self, request: TransferRequest) -> TransferAdmissionDecision:
        active_transfers, active_bytes, background_active = self._active_stats()

        if request.intent is TransferIntent.REMOTE_KV_RESTORE and request.cost_gate_approved is not True:
            action = AdmissionAction.REJECT
            reason = AdmissionReason.RESTORE_COST_GATE_DENIED
        elif request.bytes > self.config.max_inflight_bytes:
            action = AdmissionAction.REJECT
            reason = AdmissionReason.REQUEST_TOO_LARGE
        elif request.background and background_active >= self.config.max_background_inflight:
            action = AdmissionAction.QUEUE
            reason = AdmissionReason.BACKGROUND_BUDGET_EXHAUSTED
        elif active_transfers >= self.config.max_inflight_transfers:
            action = AdmissionAction.QUEUE
            reason = AdmissionReason.TRANSFER_SLOT_PRESSURE
        elif active_bytes + request.bytes > self.config.max_inflight_bytes:
            action = AdmissionAction.QUEUE
            reason = AdmissionReason.BYTE_PRESSURE
        else:
            action = AdmissionAction.ADMIT
            reason = AdmissionReason.CAPACITY_AVAILABLE

        decision = TransferAdmissionDecision(
            transfer_id=request.transfer_id,
            intent=request.intent,
            action=action,
            reason=reason,
            priority=request.priority,
            active_transfers=active_transfers,
            active_bytes=active_bytes,
            background_active=background_active,
            shadow=self.config.shadow,
            affects_execution=not self.config.shadow,
        )
        self._decision_counts[action] += 1
        if len(self._records) < self.config.max_records:
            self._records.append(decision)
        return decision

    def begin(self, request: TransferRequest) -> TransferAdmissionDecision:
        if request.transfer_id in self._active:
            raise ValueError(f"transfer is already active: {request.transfer_id}")
        decision = self.evaluate(request)
        if self.config.shadow or decision.action is AdmissionAction.ADMIT:
            # In shadow mode the real transfer proceeds, so keep its actual
            # occupancy visible to subsequent hypothetical decisions.
            self._active[request.transfer_id] = request
            active_transfers, active_bytes, _ = self._active_stats()
            self._max_observed_inflight_transfers = max(
                self._max_observed_inflight_transfers, active_transfers
            )
            self._max_observed_inflight_bytes = max(
                self._max_observed_inflight_bytes, active_bytes
            )
        return decision

    def complete(self, transfer_id: str) -> None:
        self._active.pop(transfer_id, None)

    @property
    def records(self) -> tuple[TransferAdmissionDecision, ...]:
        return tuple(self._records)

    def summary(self) -> dict[str, float]:
        return {
            "transfer_admission/decisions": float(sum(self._decision_counts.values())),
            "transfer_admission/admit": float(self._decision_counts[AdmissionAction.ADMIT]),
            "transfer_admission/queue": float(self._decision_counts[AdmissionAction.QUEUE]),
            "transfer_admission/reject": float(self._decision_counts[AdmissionAction.REJECT]),
            "transfer_admission/active": float(len(self._active)),
            "transfer_admission/max_observed_inflight_transfers": float(
                self._max_observed_inflight_transfers
            ),
            "transfer_admission/max_observed_inflight_bytes": float(
                self._max_observed_inflight_bytes
            ),
            "transfer_admission/shadow": float(self.config.shadow),
        }
