# Copyright 2026 Individual Contributor: Hanyu Dong
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
"""Opt-in initial placement for the completion time of a whole rollout group.

This module never reads KV, changes ownership, moves an admitted trajectory,
or declares a replica compatible. The caller supplies already eligible
replicas after the existing policy/readiness checks. It can run in the
serialized load-balancer actor without importing Ray or an inference engine.

The cost model is deliberately modest: independent completed-trajectory
calibration, a serial service approximation for each replica, and the maximum
predicted completion time of all group members. It is a prediction, not a GPU
queue observation. The actual engine remains free to batch/interleave calls;
the benchmark must not serialize it to make this model look accurate.
"""

from __future__ import annotations

import itertools
import math
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping, Sequence


def _finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def prompt_bucket(tokens: int) -> str:
    if tokens < 0:
        raise ValueError("prompt tokens must be non-negative")
    for boundary in (256, 1024, 4096, 16384):
        if tokens < boundary:
            return f"lt_{boundary}"
    return "ge_16384"


@dataclass(frozen=True)
class GroupRequestContext:
    """Known input metadata; never accept actual future output as a hint."""

    trajectory_id: str
    group_id: str
    member_index: int
    group_size: int
    workload_key: str
    prompt_tokens: int
    max_new_tokens: int
    agent_turn: int | None = None

    def validate(self) -> None:
        if not self.trajectory_id or not self.group_id or not self.workload_key:
            raise ValueError("trajectory, group and workload keys must be non-empty")
        if self.group_size < 1 or not 0 <= self.member_index < self.group_size:
            raise ValueError("member_index must belong to the group")
        if self.prompt_tokens < 0 or self.max_new_tokens < 1:
            raise ValueError("invalid prompt length or generation budget")
        if self.agent_turn is not None and self.agent_turn < 0:
            raise ValueError("agent_turn must be non-negative")


group_request_context: ContextVar[GroupRequestContext | None] = ContextVar("group_request_context", default=None)


@contextmanager
def bind_group_request_context(context: GroupRequestContext | None):
    """Per-trajectory async-task scope; the client consumes it, not the engine."""
    token = group_request_context.set(context)
    try:
        yield
    finally:
        group_request_context.reset(token)


@dataclass(frozen=True)
class CalibrationObservation:
    workload_key: str
    replica_id: str
    prompt_tokens: int
    max_new_tokens: int
    generated_tokens: int
    elapsed_s: float
    first_token_s: float | None
    tool_time_s: float | None
    source_id: str

    def validate(self) -> None:
        if not self.workload_key or not self.replica_id or not self.source_id:
            raise ValueError("calibration must identify workload, replica and evidence source")
        if self.prompt_tokens < 0 or self.generated_tokens < 0 or self.max_new_tokens < 1:
            raise ValueError("invalid calibration token counts")
        _finite_nonnegative(self.elapsed_s, "elapsed_s")
        for name in ("first_token_s", "tool_time_s"):
            value = getattr(self, name)
            if value is not None:
                _finite_nonnegative(value, name)
                if value > self.elapsed_s:
                    raise ValueError(f"{name} cannot exceed total elapsed time")
        if self.first_token_s is not None and self.tool_time_s is not None:
            if self.first_token_s + self.tool_time_s > self.elapsed_s:
                raise ValueError("first-token and tool intervals cannot exceed elapsed time")


@dataclass(frozen=True)
class WorkloadEstimate:
    service_s: float
    predicted_generated_tokens: float
    remaining_generation_cost_s: float | None
    prefix_state_acquire_cost_s: float | None
    tool_wait_cost_s: float | None
    unseparated_service_cost_s: float | None
    sample_count: int
    source_ids: tuple[str, ...]


class WorkloadCalibration:
    """Freeze a calibration set before acceptance traffic starts.

    Samples must be completed independent requests, not replay rows from the
    acceptance window. A missing replica/class/budget/bin causes fallback.
    Unknown timing components remain null instead of labelling total latency
    as isolated prefill/decode time.
    """

    def __init__(self, observations: Sequence[CalibrationObservation] = (), min_samples: int = 3):
        if min_samples < 1:
            raise ValueError("min_samples must be positive")
        self.min_samples = min_samples
        self._observations: dict[tuple, list[CalibrationObservation]] = defaultdict(list)
        source_ids = set()
        for observation in observations:
            observation.validate()
            if observation.source_id in source_ids:
                raise ValueError("duplicate calibration evidence source")
            source_ids.add(observation.source_id)
            self._observations[self._key(observation)].append(observation)

    @staticmethod
    def _key(observation: CalibrationObservation) -> tuple:
        return (
            observation.workload_key,
            observation.replica_id,
            prompt_bucket(observation.prompt_tokens),
            observation.max_new_tokens,
        )

    def estimate(self, request: GroupRequestContext, replica_id: str) -> WorkloadEstimate | None:
        rows = self._observations.get(
            (request.workload_key, replica_id, prompt_bucket(request.prompt_tokens), request.max_new_tokens), []
        )
        if len(rows) < self.min_samples:
            return None
        total = statistics.median(row.elapsed_s for row in rows)
        has_first = all(row.first_token_s is not None for row in rows)
        has_tool = all(row.tool_time_s is not None for row in rows)
        first = statistics.median(row.first_token_s for row in rows) if has_first else None
        tool = statistics.median(row.tool_time_s for row in rows) if has_tool else None
        generation = (
            statistics.median(row.elapsed_s - row.first_token_s - row.tool_time_s for row in rows)
            if has_first and has_tool
            else None
        )
        return WorkloadEstimate(
            service_s=total,
            predicted_generated_tokens=statistics.median(row.generated_tokens for row in rows),
            remaining_generation_cost_s=generation,
            prefix_state_acquire_cost_s=first,
            tool_wait_cost_s=tool,
            unseparated_service_cost_s=total - (tool or 0.0) if generation is None else None,
            sample_count=len(rows),
            source_ids=tuple(row.source_id for row in rows),
        )


@dataclass(frozen=True)
class GroupRoutingDecision:
    trajectory_id: str
    group_id: str
    member_index: int
    chosen_replica: str
    reason: str
    estimated_completion_at: float | None
    estimated_group_completion_at: float | None
    candidates: tuple[dict, ...]
    group_state: dict
    prediction_model: str = "independent_calibration_serial_service_group_max_v1"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Reservation:
    context: GroupRequestContext
    replica_id: str
    start_at: float
    finish_at: float
    estimate: WorkloadEstimate


@dataclass
class _Group:
    contexts: tuple[GroupRequestContext, ...]
    registered_at: float
    replica_ids: tuple[str, ...]
    plans: dict[str, _Reservation] = field(default_factory=dict)
    alternatives: dict[str, tuple[dict, ...]] = field(default_factory=dict)
    started: set[str] = field(default_factory=set)
    completed: dict[str, float] = field(default_factory=dict)
    failed: set[str] = field(default_factory=set)
    fallback_reason: str | None = None


class GroupWorkloadScheduler:
    """Plan initial group placement, preserving each trajectory's first owner.

    ``enabled=False`` is the default. The normal acquire/release counters stay
    in the existing router; ``running_requests`` is its actual observation.
    Call ``complete`` once when the whole trajectory finishes, not after every
    LLM turn. No completed sample or optimizer outcome feeds this predictor.
    """

    def __init__(
        self,
        calibration: WorkloadCalibration | None = None,
        *,
        enabled: bool = False,
        sticky: bool = True,
        max_assignments: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_assignments < 1:
            raise ValueError("max_assignments must be positive")
        if enabled and not sticky:
            raise ValueError("group-aware placement requires trajectory sticky; live routing is out of scope")
        self.enabled = enabled
        self.sticky = sticky
        self.calibration = calibration or WorkloadCalibration()
        self.max_assignments = max_assignments
        self.clock = clock
        self._groups: dict[str, _Group] = {}
        self._requests: dict[str, _Group] = {}
        self._owners: dict[str, str] = {}
        self._reservations: dict[str, _Reservation] = {}
        self._traces: list[dict] = []

    def register_group(self, requests: Sequence[GroupRequestContext], eligible_replica_ids: Sequence[str]) -> None:
        """Register all known members before concurrent first generation calls.

        Membership metadata must come from the existing rollout group, not
        duplicated trajectories. The caller performs identity/readiness
        filtering before this method. A failed prediction keeps the group
        on the unchanged least-inflight first-acquire path.
        """
        if not requests or not eligible_replica_ids or len(set(eligible_replica_ids)) != len(eligible_replica_ids):
            raise ValueError("non-empty group and unique eligible replicas required")
        contexts = tuple(sorted(requests, key=lambda request: request.member_index))
        for request in contexts:
            request.validate()
        head = contexts[0]
        if any(request.group_id != head.group_id or request.group_size != head.group_size for request in contexts):
            raise ValueError("members must describe the same complete group")
        if len(contexts) != head.group_size or {request.member_index for request in contexts} != set(range(head.group_size)):
            raise ValueError("group registration requires every member exactly once")
        if len({request.trajectory_id for request in contexts}) != len(contexts):
            raise ValueError("trajectory IDs must be unique")
        if head.group_id in self._groups or any(request.trajectory_id in self._requests for request in contexts):
            raise ValueError("group or trajectory already registered")
        group = _Group(contexts, self.clock(), tuple(eligible_replica_ids))
        if not self.enabled:
            group.fallback_reason = "disabled"
        elif len(eligible_replica_ids) ** len(contexts) > self.max_assignments:
            group.fallback_reason = "planning_budget_exceeded"
        else:
            self._plan_group(group)
        self._groups[head.group_id] = group
        for request in contexts:
            self._requests[request.trajectory_id] = group
        self._reservations.update(group.plans)

    def _plan_group(self, group: _Group) -> None:
        now = group.registered_at
        estimates = {
            (request.trajectory_id, replica): self.calibration.estimate(request, replica)
            for request in group.contexts
            for replica in group.replica_ids
        }
        if any(estimate is None for estimate in estimates.values()):
            group.fallback_reason = "missing_independent_calibration"
            return
        backlog = {
            replica: sum(
                max(0.0, reservation.finish_at - max(now, reservation.start_at))
                for reservation in self._reservations.values()
                if reservation.replica_id == replica
            )
            for replica in group.replica_ids
        }
        best = None
        best_forced: dict[tuple[str, str], tuple] = {}
        for assignment in itertools.product(group.replica_ids, repeat=len(group.contexts)):
            clocks = {replica: now + backlog[replica] for replica in group.replica_ids}
            plans = {}
            for request, replica in zip(group.contexts, assignment, strict=True):
                estimate = estimates[(request.trajectory_id, replica)]
                start = clocks[replica]
                finish = start + estimate.service_s
                clocks[replica] = finish
                plans[request.trajectory_id] = _Reservation(request, replica, start, finish, estimate)
            group_finish = max(plan.finish_at for plan in plans.values())
            # Secondary terms only break equal predicted group maxima. They
            # do not replace the group objective with a weighted router score.
            rank = (group_finish, sum(plan.finish_at for plan in plans.values()), assignment)
            candidate = (rank, plans, group_finish)
            if best is None or rank < best[0]:
                best = candidate
            for request, replica in zip(group.contexts, assignment, strict=True):
                key = (request.trajectory_id, replica)
                if key not in best_forced or rank < best_forced[key][0]:
                    best_forced[key] = candidate
        group.plans = best[1]
        for request in group.contexts:
            alternatives = []
            for replica in group.replica_ids:
                _, plans, group_finish = best_forced[(request.trajectory_id, replica)]
                plan = plans[request.trajectory_id]
                alternatives.append(
                    {
                        "replica_id": replica,
                        "queue_cost_s": plan.start_at - now,
                        "remaining_generation_cost_s": plan.estimate.remaining_generation_cost_s,
                        "prefix_state_acquire_cost_s": plan.estimate.prefix_state_acquire_cost_s,
                        "tool_wait_cost_s": plan.estimate.tool_wait_cost_s,
                        "unseparated_service_cost_s": plan.estimate.unseparated_service_cost_s,
                        "resource_pressure_penalty_s": None,
                        "group_tail_penalty_s": group_finish - plan.finish_at,
                        "reclaim_risk_penalty_s": None,
                        "estimated_completion_at": plan.finish_at,
                        "estimated_group_completion_at": group_finish,
                        "predicted_generated_tokens": plan.estimate.predicted_generated_tokens,
                        "calibration_sample_count": plan.estimate.sample_count,
                        "calibration_source_ids": list(plan.estimate.source_ids),
                        "queue_cost_source": "predicted_serial_service_reservations",
                        "disabled_signals": ["kv_pressure", "prefix_cache_hit", "reclaim_risk"],
                    }
                )
            group.alternatives[request.trajectory_id] = tuple(alternatives)

    def acquire(self, trajectory_id: str, running_requests: Mapping[str, int]) -> GroupRoutingDecision:
        """Use a planned first placement or preserve a trajectory's owner.

        This does not increment router counters. The surrounding serialized
        router does that atomically with returning the server handle.
        """
        if trajectory_id not in self._requests:
            raise KeyError("unknown trajectory; register its complete group first")
        if not running_requests or any(count < 0 for count in running_requests.values()):
            raise ValueError("actual non-negative running-request counts required")
        group = self._requests[trajectory_id]
        request = next(request for request in group.contexts if request.trajectory_id == trajectory_id)
        if trajectory_id in group.completed or trajectory_id in group.failed:
            raise ValueError("completed/failed trajectory cannot acquire again")
        if self.sticky and trajectory_id in self._owners:
            owner = self._owners[trajectory_id]
            if owner not in running_requests:
                raise RuntimeError("admitted trajectory owner unavailable; use the handoff path")
            reason = "trajectory_sticky"
        elif group.plans:
            owner = group.plans[trajectory_id].replica_id
            if owner not in running_requests:
                # Never quietly move members after admission. Re-registration
                # before admission is an integration-owner operation.
                raise RuntimeError("planned replica unavailable; invalidate pending placement first")
            reason = "minimum_predicted_group_completion"
        else:
            owner = min(running_requests, key=running_requests.get)
            reason = "least_inflight_" + (group.fallback_reason or "fallback")
        self._owners[trajectory_id] = owner
        group.started.add(trajectory_id)
        plan = group.plans.get(trajectory_id)
        candidates = []
        for alternative in group.alternatives.get(trajectory_id, ()):
            candidate = dict(alternative)
            candidate["running_requests"] = running_requests.get(candidate["replica_id"])
            candidate["queued_tokens"] = None  # no engine queue observation provided
            candidates.append(candidate)
        if not candidates:
            candidates = [
                {
                    "replica_id": replica,
                    "running_requests": count,
                    "queued_tokens": None,
                    "queue_cost_s": None,
                    "remaining_generation_cost_s": None,
                    "prefix_state_acquire_cost_s": None,
                    "resource_pressure_penalty_s": None,
                    "group_tail_penalty_s": None,
                    "reclaim_risk_penalty_s": None,
                }
                for replica, count in running_requests.items()
            ]
        decision = GroupRoutingDecision(
            trajectory_id, request.group_id, request.member_index, owner, reason,
            plan.finish_at if plan else None,
            max(reservation.finish_at for reservation in group.plans.values()) if plan else None,
            tuple(candidates), self.group_state(request.group_id),
        )
        trace = decision.to_dict()
        trace.update(
            {
                "decision_at": self.clock(),
                "registered_at": group.registered_at,
                "workload_key": request.workload_key,
                "prompt_tokens": request.prompt_tokens,
                "max_new_tokens": request.max_new_tokens,
                "agent_turn": request.agent_turn,
                "actual_completion_at": None,
                "prediction_error_s": None,
                "actual_generated_tokens": None,
                "actual_tool_time_s": None,
            }
        )
        self._traces.append(trace)
        return decision

    def complete(
        self,
        trajectory_id: str,
        *,
        generated_tokens: int | None,
        tool_time_s: float | None = None,
        success: bool = True,
    ) -> dict:
        """Record a real final callback; never train on these acceptance rows."""
        group = self._requests[trajectory_id]
        if trajectory_id not in group.started or trajectory_id in group.completed or trajectory_id in group.failed:
            raise ValueError("completion requires an active, not previously completed trajectory")
        if generated_tokens is None and success:
            raise ValueError("successful completion requires actual generated token count")
        if generated_tokens is not None and generated_tokens < 0:
            raise ValueError("generated_tokens must be non-negative")
        if tool_time_s is not None:
            _finite_nonnegative(tool_time_s, "tool_time_s")
        now = self.clock()
        if success:
            group.completed[trajectory_id] = now
        else:
            group.failed.add(trajectory_id)
        self._reservations.pop(trajectory_id, None)
        for trace in self._traces:
            if trace["trajectory_id"] == trajectory_id:
                trace["actual_completion_at"] = now
                estimate = trace["estimated_completion_at"]
                trace["prediction_error_s"] = now - estimate if estimate is not None else None
                trace["actual_generated_tokens"] = generated_tokens
                trace["actual_tool_time_s"] = tool_time_s
                trace["success"] = success
        return self.group_state(group.contexts[0].group_id)

    def group_state(self, group_id: str) -> dict:
        group = self._groups[group_id]
        return {
            "completed_members": len(group.completed),
            "running_members": len(group.started - group.completed.keys() - group.failed),
            "pending_members": len(group.contexts) - len(group.started) - len(group.failed - group.started),
            "failed_members": len(group.failed),
            "current_group_tail_at": max(group.completed.values()) if group.completed else None,
            "predicted_group_tail_at": max((plan.finish_at for plan in group.plans.values()), default=None),
            "group_size": len(group.contexts),
        }

    def decision_traces(self) -> list[dict]:
        """Return detached JSON-ready traces for evidence collection."""
        import copy

        return copy.deepcopy(self._traces)

    def is_started(self, trajectory_id: str) -> bool:
        group = self._requests.get(trajectory_id)
        return bool(group is not None and trajectory_id in group.started)

    def request_context(self, trajectory_id: str) -> GroupRequestContext | None:
        group = self._requests.get(trajectory_id)
        if group is None:
            return None
        return next(request for request in group.contexts if request.trajectory_id == trajectory_id)

    def registered_contexts(self, group_id: str) -> tuple[GroupRequestContext, ...] | None:
        group = self._groups.get(group_id)
        return group.contexts if group is not None else None

    def is_settled(self, trajectory_id: str) -> bool:
        group = self._requests.get(trajectory_id)
        return bool(group is not None and (trajectory_id in group.completed or trajectory_id in group.failed))

    def cancel(self, trajectory_id: str) -> dict:
        """Release predicted work on cancellation/preemption, including pending members.

        Cancellation is not a successful trajectory completion and does not
        claim an observed generated-token count or first-token latency.
        """
        group = self._requests[trajectory_id]
        if not self.is_settled(trajectory_id):
            group.failed.add(trajectory_id)
            self._reservations.pop(trajectory_id, None)
            for trace in self._traces:
                if trace["trajectory_id"] == trajectory_id:
                    trace["cancelled_at"] = self.clock()
                    trace["success"] = False
                    trace["actual_completion_at"] = None
                    trace["prediction_error_s"] = None
                    trace["actual_generated_tokens"] = None
                    trace["actual_tool_time_s"] = None
        return self.group_state(group.contexts[0].group_id)

    def forget_completed_group(self, group_id: str) -> None:
        """Bound state after delivery; export this group's traces before calling."""
        group = self._groups[group_id]
        if len(group.completed) + len(group.failed) != len(group.contexts):
            raise ValueError("cannot forget an incomplete group")
        for request in group.contexts:
            self._requests.pop(request.trajectory_id)
            self._owners.pop(request.trajectory_id, None)
        self._traces = [trace for trace in self._traces if trace["group_id"] != group_id]
        del self._groups[group_id]
