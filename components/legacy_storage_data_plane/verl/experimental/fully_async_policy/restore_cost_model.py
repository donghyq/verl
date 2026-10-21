# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Restore-versus-recompute profiling for same-version rollout KV.

This module answers one question per request: is it cheaper to fetch a
same-version prefix from remote KV storage, or to recompute that prefix with a
local prefill?  It is deliberately side-effect free and runs in **shadow
mode**: it produces decisions, traces and machine-readable benchmark records,
but it never changes which replica actually serves a request.

Measured inputs follow the Project B profiling contract:
``missing_prefix_tokens``, ``kv_bytes``, ``lookup``, ``queue``, ``allocation``,
``remote_transfer``, ``import/H2D`` and ``actual prefill`` latency.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from verl.experimental.fully_async_policy.transfer_admission import (
    TransferAdmissionController,
    TransferIntent,
    TransferRequest,
)


class CostGatePath(StrEnum):
    """The mutually exclusive outcomes recorded by the cost gate."""

    REMOTE_RESTORE = "remote_restore"
    TOKEN_ONLY_REPREFILL = "token_only_reprefill"


class CostGateReason(StrEnum):
    """Why a path was chosen. Every decision must carry one of these."""

    RESTORE_CHEAPER_THAN_RECOMPUTE = "restore_cheaper_than_recompute"
    RESTORE_COST_EXCEEDS_RECOMPUTE = "restore_cost_exceeds_recompute"
    SAVINGS_BELOW_THRESHOLD = "savings_below_threshold"
    NO_REUSABLE_PREFIX = "no_reusable_prefix"


def _require_non_negative(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return float(value)


@dataclass(frozen=True)
class RestoreTimingBreakdown:
    """The measured latency components of one remote KV restore attempt.

    ``transfer_s`` is pure wire time. ``queue_s`` stays observable but is
    deliberately excluded from the cost-gate execution estimate: T6 adds the
    chosen replica's queue once when it computes estimated finish time.
    Keeping it separate prevents a remote candidate from paying queueing twice
    while still retaining the full end-to-end evidence for a profile sample.
    """

    lookup_s: float = 0.0
    queue_s: float = 0.0
    allocation_s: float = 0.0
    transfer_s: float = 0.0
    import_s: float = 0.0
    runtime_overhead_s: float = 0.0
    overlap_s: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "lookup_s",
            "queue_s",
            "allocation_s",
            "transfer_s",
            "import_s",
            "runtime_overhead_s",
            "overlap_s",
        ):
            _require_non_negative(name, getattr(self, name))

    @property
    def total_s(self) -> float:
        """End-to-end restore time, including queueing."""
        additive_total = (
            self.lookup_s
            + self.queue_s
            + self.allocation_s
            + self.transfer_s
            + self.import_s
            + self.runtime_overhead_s
        )
        return max(additive_total - self.overlap_s, 0.0)

    @property
    def execution_s(self) -> float:
        """Restore execution cost after a request has been admitted."""
        additive_execution = (
            self.lookup_s
            + self.allocation_s
            + self.transfer_s
            + self.import_s
            + self.runtime_overhead_s
        )
        return max(additive_execution - self.overlap_s, 0.0)

    @property
    def non_transfer_s(self) -> float:
        """Execution cost excluding just the storage wire transfer."""
        return self.lookup_s + self.allocation_s + self.import_s + self.runtime_overhead_s

    def to_dict(self) -> dict[str, float]:
        return {
            "lookup_s": self.lookup_s,
            "queue_s": self.queue_s,
            "allocation_s": self.allocation_s,
            "transfer_s": self.transfer_s,
            "import_s": self.import_s,
            "runtime_overhead_s": self.runtime_overhead_s,
            "overlap_s": self.overlap_s,
            "execution_s": self.execution_s,
            "total_s": self.total_s,
        }


@dataclass(frozen=True)
class RestoreProfileSample:
    """One observed request used to calibrate and validate the cost model."""

    sample_id: str
    policy_kv_identity_digest: str
    missing_prefix_tokens: int
    kv_bytes: int
    timing: RestoreTimingBreakdown
    recompute_latency_s: float
    restored_tokens: int = 0
    residual_prefill_tokens: int = 0
    observed_restore_execution_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if not self.policy_kv_identity_digest:
            raise ValueError("policy_kv_identity_digest must be non-empty")
        if not isinstance(self.missing_prefix_tokens, int) or self.missing_prefix_tokens < 0:
            raise ValueError("missing_prefix_tokens must be a non-negative integer")
        if not isinstance(self.kv_bytes, int) or self.kv_bytes < 0:
            raise ValueError("kv_bytes must be a non-negative integer")
        if not isinstance(self.restored_tokens, int) or self.restored_tokens < 0:
            raise ValueError("restored_tokens must be a non-negative integer")
        if self.restored_tokens > self.missing_prefix_tokens:
            raise ValueError("restored_tokens cannot exceed missing_prefix_tokens")
        if not isinstance(self.residual_prefill_tokens, int) or self.residual_prefill_tokens < 0:
            raise ValueError("residual_prefill_tokens must be a non-negative integer")
        if self.residual_prefill_tokens > self.missing_prefix_tokens:
            raise ValueError("residual_prefill_tokens cannot exceed missing_prefix_tokens")
        if not isinstance(self.recompute_latency_s, (int, float)) or isinstance(self.recompute_latency_s, bool):
            raise ValueError("recompute_latency_s must be a number")
        if self.recompute_latency_s <= 0:
            raise ValueError("recompute_latency_s must be positive to model recompute cost")
        if self.observed_restore_execution_s is not None:
            _require_non_negative("observed_restore_execution_s", self.observed_restore_execution_s)

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, Any]) -> "RestoreProfileSample":
        """Build a sample from the flat Project B profiling metric names.

        ``remote_transfer_latency`` and ``effective_bandwidth`` are
        alternatives: pass the measured latency when available, otherwise the
        observed bandwidth is converted into a transfer latency.
        """

        kv_bytes = int(metrics["kv_bytes"])
        transfer_s = metrics.get("remote_transfer_latency_s")
        if transfer_s is None:
            bandwidth = metrics.get("effective_bandwidth_bytes_per_s")
            if bandwidth is None or bandwidth <= 0:
                raise ValueError("either remote_transfer_latency_s or a positive effective_bandwidth is required")
            transfer_s = kv_bytes / float(bandwidth)

        return cls(
            sample_id=str(metrics["sample_id"]),
            policy_kv_identity_digest=str(metrics["policy_kv_identity_digest"]),
            missing_prefix_tokens=int(metrics["missing_prefix_tokens"]),
            kv_bytes=kv_bytes,
            timing=RestoreTimingBreakdown(
                lookup_s=float(metrics.get("lookup_latency_s", 0.0)),
                queue_s=float(metrics.get("queue_latency_s", 0.0)),
                allocation_s=float(metrics.get("allocation_latency_s", 0.0)),
                transfer_s=float(transfer_s),
                import_s=float(metrics.get("import_latency_s", 0.0)),
                runtime_overhead_s=float(metrics.get("runtime_overhead_latency_s", 0.0)),
                overlap_s=float(metrics.get("overlap_latency_s", 0.0)),
            ),
            recompute_latency_s=float(metrics["actual_prefill_latency_s"]),
            restored_tokens=int(metrics.get("restored_tokens", 0)),
            residual_prefill_tokens=int(metrics.get("residual_prefill_tokens", 0)),
            observed_restore_execution_s=(
                float(metrics["observed_restore_execution_latency_s"])
                if metrics.get("observed_restore_execution_latency_s") is not None
                else None
            ),
            metadata=dict(metrics.get("metadata", {})),
        )

    @property
    def t_restore_s(self) -> float:
        """Execution cost used by the restore-versus-recompute gate."""
        if self.observed_restore_execution_s is not None:
            return self.observed_restore_execution_s
        return self.timing.execution_s

    @property
    def t_restore_e2e_s(self) -> float:
        """Observed end-to-end restore time, retained for diagnosis only."""
        return self.timing.total_s

    @property
    def t_recompute_s(self) -> float:
        return float(self.recompute_latency_s)

    @property
    def net_savings_s(self) -> float:
        return self.t_recompute_s - self.t_restore_s

    @property
    def effective_bandwidth_bytes_per_s(self) -> float:
        if self.timing.transfer_s <= 0:
            return math.inf if self.kv_bytes > 0 else 0.0
        return self.kv_bytes / self.timing.transfer_s

    @property
    def measured_prefill_tokens_per_second(self) -> float:
        return self.missing_prefix_tokens / self.recompute_latency_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "policy_kv_identity_digest": self.policy_kv_identity_digest,
            "missing_prefix_tokens": self.missing_prefix_tokens,
            "kv_bytes": self.kv_bytes,
            "restored_tokens": self.restored_tokens,
            "residual_prefill_tokens": self.residual_prefill_tokens,
            "t_restore_s": self.t_restore_s,
            "t_restore_e2e_s": self.t_restore_e2e_s,
            "observed_restore_execution_s": self.observed_restore_execution_s,
            "t_recompute_s": self.t_recompute_s,
            "net_savings_s": self.net_savings_s,
            "effective_bandwidth_bytes_per_s": self.effective_bandwidth_bytes_per_s,
            "measured_prefill_tokens_per_second": self.measured_prefill_tokens_per_second,
            **self.timing.to_dict(),
        }


@dataclass(frozen=True)
class RestoreCostModel:
    """Empirical remote-restore cost with separately reported wire bandwidth.

    Storage I/O and host-to-device import may overlap. ``execution_seconds_per_byte``
    is therefore fitted from observed request execution time rather than assumed
    to equal ``1 / wire_bandwidth_bytes_per_s``. The latter remains a diagnostic
    metric for the data plane.
    """

    fixed_overhead_s: float
    bandwidth_bytes_per_s: float
    execution_seconds_per_byte: float = 0.0
    residual_prefill_seconds_per_token: float = 0.0
    sample_count: int = 0

    def __post_init__(self) -> None:
        _require_non_negative("fixed_overhead_s", self.fixed_overhead_s)
        _require_non_negative("bandwidth_bytes_per_s", self.bandwidth_bytes_per_s)
        _require_non_negative("execution_seconds_per_byte", self.execution_seconds_per_byte)
        _require_non_negative(
            "residual_prefill_seconds_per_token", self.residual_prefill_seconds_per_token
        )

    @classmethod
    def calibrate(
        cls,
        samples: Sequence[RestoreProfileSample],
        *,
        residual_prefill_seconds_per_token: float | None = None,
    ) -> "RestoreCostModel":
        """Fit the model from measured restore attempts.

        Wire bandwidth is the aggregate bytes-over-transfer-time ratio. The
        request-time fit is separate because prefetch, import and residual
        prefill can overlap. Queueing is intentionally excluded because T6
        adds it once at the selected replica.
        """

        if not samples:
            raise ValueError("at least one sample is required to calibrate a restore cost model")

        total_bytes = sum(sample.kv_bytes for sample in samples)
        total_transfer_s = sum(sample.timing.transfer_s for sample in samples)
        if total_transfer_s > 0:
            wire_bandwidth = total_bytes / total_transfer_s
        else:
            wire_bandwidth = math.inf if total_bytes > 0 else 0.0

        if residual_prefill_seconds_per_token is None:
            total_recompute_tokens = sum(sample.missing_prefix_tokens for sample in samples)
            total_recompute_latency_s = sum(sample.t_recompute_s for sample in samples)
            if total_recompute_tokens <= 0 or total_recompute_latency_s <= 0:
                raise ValueError("restore calibration needs positive recompute token counts and latencies")
            residual_prefill_seconds_per_token = total_recompute_latency_s / total_recompute_tokens
        _require_non_negative(
            "residual_prefill_seconds_per_token", residual_prefill_seconds_per_token
        )
        byte_counts = [sample.kv_bytes for sample in samples]
        adjusted_execution = [
            sample.t_restore_s - sample.residual_prefill_tokens * residual_prefill_seconds_per_token
            for sample in samples
        ]
        if len(samples) == 1:
            # One point cannot identify both intercept and slope. Preserve the
            # observed execution time as a fixed cost instead of inventing a
            # bandwidth-derived slope.
            execution_seconds_per_byte = 0.0
            fixed_overhead = max(0.0, adjusted_execution[0])
        else:
            mean_bytes = sum(byte_counts) / len(byte_counts)
            mean_execution = sum(adjusted_execution) / len(adjusted_execution)
            variance = sum((value - mean_bytes) ** 2 for value in byte_counts)
            covariance = sum(
                (byte_count - mean_bytes) * (execution - mean_execution)
                for byte_count, execution in zip(byte_counts, adjusted_execution)
            )
            execution_seconds_per_byte = max(covariance / variance, 0.0) if variance else 0.0
            fixed_overhead = max(0.0, mean_execution - execution_seconds_per_byte * mean_bytes)

        return cls(
            fixed_overhead_s=fixed_overhead,
            bandwidth_bytes_per_s=wire_bandwidth,
            execution_seconds_per_byte=execution_seconds_per_byte,
            residual_prefill_seconds_per_token=residual_prefill_seconds_per_token,
            sample_count=len(samples),
        )

    def t_restore_s(self, *, kv_bytes: int, residual_prefill_tokens: int = 0) -> float:
        _require_non_negative("kv_bytes", kv_bytes)
        _require_non_negative("residual_prefill_tokens", residual_prefill_tokens)
        if self.execution_seconds_per_byte > 0 or self.sample_count > 0:
            transfer_execution_s = kv_bytes * self.execution_seconds_per_byte
        elif kv_bytes > 0 and math.isinf(self.bandwidth_bytes_per_s):
            transfer_execution_s = 0.0
        elif self.bandwidth_bytes_per_s > 0:
            transfer_execution_s = kv_bytes / self.bandwidth_bytes_per_s
        else:
            transfer_execution_s = math.inf if kv_bytes > 0 else 0.0
        return (
            self.fixed_overhead_s
            + transfer_execution_s
            + self.residual_prefill_seconds_per_token * residual_prefill_tokens
        )

    def to_dict(self) -> dict[str, Any]:
        wire_bandwidth = self.bandwidth_bytes_per_s
        return {
            "fixed_overhead_s": self.fixed_overhead_s,
            # Retained name for compatibility. It is the storage I/O wire
            # bandwidth, not an assumption about the overlapped request path.
            "bandwidth_bytes_per_s": None if math.isinf(wire_bandwidth) else wire_bandwidth,
            "wire_bandwidth_bytes_per_s": None if math.isinf(wire_bandwidth) else wire_bandwidth,
            "execution_seconds_per_byte": self.execution_seconds_per_byte,
            "residual_prefill_seconds_per_token": self.residual_prefill_seconds_per_token,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class RecomputeCostModel:
    """Linear request execution cost: ``fixed + tokens / throughput``."""

    prefill_tokens_per_second: float
    fixed_overhead_s: float = 0.0
    sample_count: int = 0

    def __post_init__(self) -> None:
        if self.prefill_tokens_per_second <= 0:
            raise ValueError("prefill_tokens_per_second must be positive")
        _require_non_negative("fixed_overhead_s", self.fixed_overhead_s)

    @classmethod
    def calibrate(cls, samples: Sequence[RestoreProfileSample]) -> "RecomputeCostModel":
        if not samples:
            raise ValueError("at least one sample is required to calibrate a recompute cost model")
        token_counts = [sample.missing_prefix_tokens for sample in samples]
        latencies = [sample.t_recompute_s for sample in samples]
        total_tokens = sum(token_counts)
        total_latency = sum(latencies)
        if total_tokens <= 0 or total_latency <= 0:
            raise ValueError("recompute calibration needs positive token counts and latencies")

        mean_tokens = total_tokens / len(samples)
        mean_latency = total_latency / len(samples)
        variance = sum((tokens - mean_tokens) ** 2 for tokens in token_counts)
        covariance = sum(
            (tokens - mean_tokens) * (latency - mean_latency)
            for tokens, latency in zip(token_counts, latencies)
        )
        if variance > 0 and covariance > 0:
            seconds_per_token = covariance / variance
            fixed_overhead_s = max(0.0, mean_latency - seconds_per_token * mean_tokens)
        else:
            seconds_per_token = total_latency / total_tokens
            fixed_overhead_s = 0.0

        return cls(
            prefill_tokens_per_second=1.0 / seconds_per_token,
            fixed_overhead_s=fixed_overhead_s,
            sample_count=len(samples),
        )

    @property
    def incremental_seconds_per_token(self) -> float:
        return 1.0 / self.prefill_tokens_per_second

    def t_recompute_s(self, missing_prefix_tokens: int) -> float:
        _require_non_negative("missing_prefix_tokens", missing_prefix_tokens)
        return self.fixed_overhead_s + missing_prefix_tokens / self.prefill_tokens_per_second

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "fixed_overhead_s": self.fixed_overhead_s,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class CostGateDecision:
    """The recorded outcome of one shadow cost-gate evaluation.

    ``shadow`` and ``affects_routing`` are always ``True``/``False`` in the
    current implementation. They exist so that any exported record states
    explicitly that no request was rerouted by this decision.
    """

    path: CostGatePath
    t_restore_s: float
    t_recompute_s: float
    missing_prefix_tokens: int
    kv_bytes: int
    safety_margin_s: float
    min_savings_s: float
    reason: CostGateReason
    identity_digest: str | None = None
    should_restore: bool = False
    shadow: bool = True
    affects_routing: bool = False

    @property
    def net_savings_s(self) -> float:
        return self.t_recompute_s - self.t_restore_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_digest": self.identity_digest,
            "path": str(self.path),
            "should_restore": self.should_restore,
            "reason": str(self.reason),
            "t_restore_s": self.t_restore_s,
            "t_recompute_s": self.t_recompute_s,
            "net_savings_s": self.net_savings_s,
            "missing_prefix_tokens": self.missing_prefix_tokens,
            "kv_bytes": self.kv_bytes,
            "safety_margin_s": self.safety_margin_s,
            "min_savings_s": self.min_savings_s,
            "shadow": self.shadow,
            "affects_routing": self.affects_routing,
        }


def evaluate_restore_cost_gate(
    *,
    restore_model: RestoreCostModel,
    recompute_model: RecomputeCostModel,
    missing_prefix_tokens: int,
    kv_bytes: int,
    residual_prefill_tokens: int = 0,
    safety_margin_s: float = 0.0,
    min_savings_s: float = 0.0,
    identity_digest: str | None = None,
) -> CostGateDecision:
    """Decide restores only when the measured saving survives a safety margin.

    The margin is subtracted from the saving before the comparison because a
    restore that is only marginally cheaper still loses once allocator
    pressure, page remap and TP synchronisation are taken into account.
    """

    _require_non_negative("safety_margin_s", safety_margin_s)
    _require_non_negative("min_savings_s", min_savings_s)

    if missing_prefix_tokens <= 0:
        return CostGateDecision(
            path=CostGatePath.TOKEN_ONLY_REPREFILL,
            t_restore_s=0.0,
            t_recompute_s=0.0,
            missing_prefix_tokens=missing_prefix_tokens,
            kv_bytes=kv_bytes,
            safety_margin_s=safety_margin_s,
            min_savings_s=min_savings_s,
            reason=CostGateReason.NO_REUSABLE_PREFIX,
            identity_digest=identity_digest,
        )

    t_restore = restore_model.t_restore_s(
        kv_bytes=kv_bytes, residual_prefill_tokens=residual_prefill_tokens
    )
    t_recompute = recompute_model.t_recompute_s(missing_prefix_tokens)
    net_savings = t_recompute - t_restore

    if t_restore + safety_margin_s >= t_recompute:
        path = CostGatePath.TOKEN_ONLY_REPREFILL
        reason = CostGateReason.RESTORE_COST_EXCEEDS_RECOMPUTE
        should_restore = False
    elif net_savings - safety_margin_s < min_savings_s:
        path = CostGatePath.TOKEN_ONLY_REPREFILL
        reason = CostGateReason.SAVINGS_BELOW_THRESHOLD
        should_restore = False
    else:
        path = CostGatePath.REMOTE_RESTORE
        reason = CostGateReason.RESTORE_CHEAPER_THAN_RECOMPUTE
        should_restore = True

    return CostGateDecision(
        path=path,
        t_restore_s=t_restore,
        t_recompute_s=t_recompute,
        missing_prefix_tokens=missing_prefix_tokens,
        kv_bytes=kv_bytes,
        safety_margin_s=safety_margin_s,
        min_savings_s=min_savings_s,
        reason=reason,
        identity_digest=identity_digest,
        should_restore=should_restore,
    )


DEFAULT_TOKEN_GRID: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def build_crossover_curve(
    *,
    restore_model: RestoreCostModel,
    recompute_model: RecomputeCostModel,
    bytes_per_token: float,
    token_grid: Iterable[int] | None = None,
    safety_margin_s: float = 0.0,
    min_savings_s: float = 0.0,
    restored_token_fraction: float = 1.0,
    residual_prefill_tokens: int = 0,
) -> list[CostGateDecision]:
    """Return the restore/recompute decision over a grid of prefix lengths.

    Restore has a fixed overhead and therefore only wins above a crossover
    length; the curve makes that threshold explicit instead of reporting a
    single favourable measurement.
    """

    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")
    if not 0.0 <= restored_token_fraction <= 1.0:
        raise ValueError("restored_token_fraction must be within [0, 1]")
    if residual_prefill_tokens < 0:
        raise ValueError("residual_prefill_tokens must be non-negative")
    grid = list(token_grid) if token_grid is not None else list(DEFAULT_TOKEN_GRID)
    if not grid:
        raise ValueError("token_grid must not be empty")
    if any(token <= 0 for token in grid):
        raise ValueError("token_grid entries must be positive")
    if grid != sorted(grid):
        raise ValueError("token_grid must be sorted ascending")

    decisions = []
    for token in grid:
        remaining_tokens = min(
            token, max(residual_prefill_tokens, int(round(token * (1.0 - restored_token_fraction))))
        )
        restored_tokens = token - remaining_tokens
        decisions.append(
            evaluate_restore_cost_gate(
                restore_model=restore_model,
                recompute_model=recompute_model,
                missing_prefix_tokens=token,
                kv_bytes=int(round(restored_tokens * bytes_per_token)),
                residual_prefill_tokens=remaining_tokens,
                safety_margin_s=safety_margin_s,
                min_savings_s=min_savings_s,
            )
        )
    return decisions


def find_crossover_tokens(
    *,
    restore_model: RestoreCostModel,
    recompute_model: RecomputeCostModel,
    bytes_per_token: float,
    safety_margin_s: float = 0.0,
    min_savings_s: float = 0.0,
    restored_token_fraction: float = 1.0,
    residual_prefill_tokens: int = 0,
    max_tokens: int = 1 << 20,
) -> int | None:
    """Return the smallest prefix length where restore wins, or ``None``.

    ``None`` means restore never wins inside ``max_tokens``: the fixed
    overhead is larger than any achievable saving, so the gate must stay off
    for this workload.
    """

    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")
    if not 0.0 <= restored_token_fraction <= 1.0:
        raise ValueError("restored_token_fraction must be within [0, 1]")
    if residual_prefill_tokens < 0:
        raise ValueError("residual_prefill_tokens must be non-negative")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    def wins(token: int) -> bool:
        remaining_tokens = min(
            token, max(residual_prefill_tokens, int(round(token * (1.0 - restored_token_fraction))))
        )
        decision = evaluate_restore_cost_gate(
            restore_model=restore_model,
            recompute_model=recompute_model,
            missing_prefix_tokens=token,
            kv_bytes=int(round((token - remaining_tokens) * bytes_per_token)),
            residual_prefill_tokens=remaining_tokens,
            safety_margin_s=safety_margin_s,
            min_savings_s=min_savings_s,
        )
        return decision.path is CostGatePath.REMOTE_RESTORE

    if not wins(max_tokens):
        return None
    low, high = 1, max_tokens
    while low < high:
        mid = (low + high) // 2
        if wins(mid):
            high = mid
        else:
            low = mid + 1
    return low


@dataclass(frozen=True)
class RestoreRecomputeBenchmark:
    """Machine-readable collection of profiling samples and shadow decisions."""

    samples: tuple[RestoreProfileSample, ...] = ()
    decisions: tuple[CostGateDecision, ...] = ()
    label: str = "project-b-restore-vs-recompute"

    def __post_init__(self) -> None:
        if not self.samples and not self.decisions:
            raise ValueError("a benchmark must contain at least one sample or decision")

    def to_rows(self) -> list[dict[str, Any]]:
        return [sample.to_dict() for sample in self.samples]

    def decision_rows(self) -> list[dict[str, Any]]:
        return [decision.to_dict() for decision in self.decisions]

    def to_json_dict(
        self,
        *,
        restore_model: RestoreCostModel | None = None,
        recompute_model: RecomputeCostModel | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        restore_wins = sum(decision.path is CostGatePath.REMOTE_RESTORE for decision in self.decisions)
        payload: dict[str, Any] = {
            "label": self.label,
            "shadow_mode": True,
            "affects_routing": False,
            "decision_count": len(self.decisions),
            "restore_decision_count": restore_wins,
            "recompute_decision_count": len(self.decisions) - restore_wins,
            "models": {
                "restore": restore_model.to_dict() if restore_model is not None else None,
                "recompute": recompute_model.to_dict() if recompute_model is not None else None,
            },
            "samples": self.to_rows(),
            "decisions": self.decision_rows(),
        }
        if extra:
            payload.update(dict(extra))
        return payload

    def write_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_rows()
        if not rows:
            raise ValueError("cannot write a sample CSV without samples")
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return target

    def write_decisions_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.decision_rows()
        if not rows:
            raise ValueError("cannot write a decision CSV without decisions")
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return target

    def write_json(
        self,
        path: str | Path,
        *,
        restore_model: RestoreCostModel | None = None,
        recompute_model: RecomputeCostModel | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_json_dict(restore_model=restore_model, recompute_model=recompute_model, extra=extra)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target


@dataclass
class ShadowCostGateRecorder:
    """Collect shadow decisions without influencing the real request path."""

    label: str = "project-b-restore-vs-recompute"
    max_decisions: int = 10_000
    transfer_admission: TransferAdmissionController | None = None
    _decisions: list[CostGateDecision] = field(default_factory=list)

    def record(
        self,
        *,
        restore_model: RestoreCostModel,
        recompute_model: RecomputeCostModel,
        missing_prefix_tokens: int,
        kv_bytes: int,
        residual_prefill_tokens: int = 0,
        safety_margin_s: float = 0.0,
        min_savings_s: float = 0.0,
        identity_digest: str | None = None,
    ) -> CostGateDecision:
        decision = evaluate_restore_cost_gate(
            restore_model=restore_model,
            recompute_model=recompute_model,
            missing_prefix_tokens=missing_prefix_tokens,
            kv_bytes=kv_bytes,
            residual_prefill_tokens=residual_prefill_tokens,
            safety_margin_s=safety_margin_s,
            min_savings_s=min_savings_s,
            identity_digest=identity_digest,
        )
        if len(self._decisions) < self.max_decisions:
            self._decisions.append(decision)
        if self.transfer_admission is not None:
            try:
                self.transfer_admission.evaluate(
                    TransferRequest(
                        transfer_id=f"remote-kv-restore-shadow:{len(self._decisions)}",
                        intent=TransferIntent.REMOTE_KV_RESTORE,
                        bytes=kv_bytes,
                        policy_version=None,
                        cost_gate_approved=decision.should_restore,
                        metadata={
                            "identity_digest": identity_digest,
                            "cost_gate_reason": str(decision.reason),
                            "shadow": True,
                        },
                    )
                )
            except Exception:
                # The cost decision remains valid even if shadow admission
                # telemetry cannot be recorded.
                pass
        return decision

    @property
    def decisions(self) -> tuple[CostGateDecision, ...]:
        return tuple(self._decisions)

    def summary(self) -> dict[str, float]:
        total = len(self._decisions)
        if total == 0:
            return {}
        restore = sum(decision.path is CostGatePath.REMOTE_RESTORE for decision in self._decisions)
        return {
            "rollout_restore_cost_gate/decisions": float(total),
            "rollout_restore_cost_gate/restore_decisions": float(restore),
            "rollout_restore_cost_gate/recompute_decisions": float(total - restore),
            "rollout_restore_cost_gate/restore_decision_ratio": float(restore / total),
            "rollout_restore_cost_gate/net_savings_s_total": float(
                sum(decision.net_savings_s for decision in self._decisions if decision.should_restore)
            ),
        }

    def export(
        self,
        *,
        samples: Sequence[RestoreProfileSample] = (),
        restore_model: RestoreCostModel | None = None,
        recompute_model: RecomputeCostModel | None = None,
    ) -> RestoreRecomputeBenchmark:
        return RestoreRecomputeBenchmark(
            samples=tuple(samples),
            decisions=self.decisions,
            label=self.label,
        )
