"""Observation hooks for the existing business/Trainer path; no routing controller.

Call at the original coordinator's first dispatch and final accepted group sites.
All group times must come from that coordinator clock, including retry completion.
GPU intervals come from the MLP allocation receipt, not configured GPU counts or
GPU utilization. A missing observation stays None. These hooks never accept data
on behalf of the framework and never change tokens, logprobs, tools or rewards.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class WindowJournal:
    """One journal per window, written by the existing coordinator only."""

    def __init__(self, path, spec, *, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.clock = clock
        self.clock_id = f"coordinator:{os.getpid()}:{time.monotonic_ns()}"
        self.starts = {}
        self.attempts = {}
        self.finished_groups = set()
        self.event("window_started", spec=spec, clock_id=self.clock_id)

    def event(self, event, **fields):
        row = {"event": event, "observed_wall_s": self.clock(), **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        return row

    def first_dispatch(self, group_id, member_id, attempt_id):
        """Use original stable group/member IDs, including after partial retry."""
        start = self.starts.setdefault(group_id, self.clock())
        self.event("member_dispatched", group_id=group_id, member_id=member_id,
                   attempt_id=attempt_id, first_dispatch_wall_s=start,
                   clock_id=self.clock_id)

    def attempt_finished(self, group_id, member_id, attempt_id, *, legal,
                         business_success=None, reward=None, generated_tokens=None,
                         effective_training_tokens=None, failure=None,
                         policy_source=None, history_before=None, history_after=None):
        """legal is the framework's verdict; history digests cover committed data.

        Supply digests of the same committed token/mask/logprob/tool/source slice
        before and after continuation. New generation is allowed to differ.
        """
        if group_id not in self.starts:
            raise ValueError("First dispatch must be observed before completion")
        if type(legal) is not bool:
            raise ValueError("The original framework must provide a legal verdict")
        for key, value in (("reward", reward), ("generated_tokens", generated_tokens),
                           ("effective_training_tokens", effective_training_tokens)):
            if value is not None and (not isinstance(value, (int, float)) or
                                      isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"Invalid observed {key}")
        if history_before is not None and history_before != history_after:
            raise ValueError("Previously committed history changed during continuation")
        row = self.event("attempt_finished", group_id=group_id, member_id=member_id,
                         attempt_id=attempt_id, legal=legal, business_success=business_success,
                         reward=reward, generated_tokens=generated_tokens,
                         effective_training_tokens=effective_training_tokens,
                         failure=failure, policy_source=policy_source,
                         committed_history_verified=(history_before is not None and
                                                     history_before == history_after),
                         clock_id=self.clock_id)
        self.attempts[(group_id, member_id, attempt_id)] = row
        return row

    def group_accepted(self, group_id, selected_attempts, *, framework_proof):
        """Call only after original complete-group/source validation succeeds.

        selected_attempts is [(original_member_id, original_attempt_id), ...].
        A negative reward does not by itself invalidate a legal training sample.
        """
        if group_id in self.finished_groups:
            raise ValueError("Do not count one complete group twice")
        if len(selected_attempts) != self.spec["group_size"]:
            raise ValueError("Incomplete GRPO group cannot be counted")
        if len({member for member, _ in selected_attempts}) != self.spec["group_size"]:
            raise ValueError("Duplicate group member cannot be counted")
        if framework_proof.get("complete_group") is not True or framework_proof.get("policy_sources_accepted") is not True:
            raise ValueError("Original framework acceptance proof is required")
        rows = [self.attempts[(group_id, member, attempt)] for member, attempt in selected_attempts]
        if any(not row["legal"] for row in rows):
            raise ValueError("Selected terminal attempts must be legal")
        final = max(row["observed_wall_s"] for row in rows)
        if final < self.starts[group_id]:
            raise ValueError("Coordinator clock boundary cannot produce a negative group duration")
        self.finished_groups.add(group_id)
        return self.event("group_accepted", group_id=group_id, selected_attempts=selected_attempts,
                          framework_proof=framework_proof, clock_id=self.clock_id,
                          first_dispatch_wall_s=self.starts[group_id],
                          final_legal_attempt_completion_wall_s=final,
                          group_latency_s=final - self.starts[group_id])

    def cost(self, operation, *, elapsed_s, receipt):
        """flush, kernel_warmup, cache_prewarm, restart and weight_publish separately."""
        if operation not in {"flush", "kernel_warmup", "cache_prewarm", "restart", "weight_publish"}:
            raise ValueError("Unknown cost operation")
        if not isinstance(elapsed_s, (int, float)) or not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("A measured operation duration is required")
        self.event("operation_cost", operation=operation, elapsed_s=elapsed_s, receipt=receipt)


def observe_gateway_class(base, emit):
    """Wrap the original client in the worker process, preserving all arguments.

    Emit to a worker-local journal. Join by the original request ID. This hook
    does not provide group acceptance, first-dispatch time or allocation proof.
    """
    class ObservedGatewayClient(base):
        async def generate(self, request_id, **kwargs):
            started = time.monotonic()
            try:
                output = await super().generate(request_id, **kwargs)
            except BaseException as error:
                emit("native_request_failed", request_id=request_id,
                     elapsed_s=time.monotonic() - started, failure=type(error).__name__,
                     generated_tokens=None)
                raise
            fields = output.extra_fields
            emit("native_request_returned", request_id=request_id,
                 elapsed_s=time.monotonic() - started,
                 prompt_tokens=len(kwargs["prompt_ids"]), generated_tokens=len(output.token_ids),
                 cached_tokens=fields.get("num_cached_tokens"),
                 worker_id=fields.get("gateway_worker_id"),
                 engine_weight_version=fields.get("engine_weight_version"),
                 salt_match=fields.get("salt_match"), stop_reason=output.stop_reason)
            return output
    return ObservedGatewayClient


def gpu_hours(intervals):
    """Union measured exclusive allocation intervals per actual GPU UUID."""
    if not intervals:
        return None, "MLP actual allocation intervals unavailable"
    per_gpu = {}
    for row in intervals:
        if (row.get("exclusive") is not True or not row.get("source_receipt") or
                not row.get("gpu_uuid") or row.get("end_wall_s") is None):
            return None, "Actual UUID, exclusive allocation and final receipt are required"
        start, end = row.get("start_wall_s"), row["end_wall_s"]
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
               for v in (start, end)) or end <= start:
            return None, "Invalid allocation duration"
        per_gpu.setdefault(row["gpu_uuid"], []).append((start, end))
    seconds = 0.0
    for spans in per_gpu.values():
        spans.sort()
        start, end = spans[0]
        for next_start, next_end in spans[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                seconds += end - start
                start, end = next_start, next_end
        seconds += end - start
    return seconds / 3600.0, None


def summarize_window(spec, events, allocation_intervals, *, accounting_complete,
                     initial_conditions, runtime_contract, min_tail_groups=400):
    """Compute measurements; never infer missing observations from configurations."""
    groups = [row for row in events if row["event"] == "group_accepted"]
    unique = {row["group_id"] for row in groups}
    if len(unique) != len(groups):
        raise ValueError("Duplicate accepted group observation")
    hours, allocation_reason = gpu_hours(allocation_intervals)
    valid_count = len(groups) if accounting_complete else None
    durations = sorted(row["group_latency_s"] for row in groups)
    p95 = durations[math.ceil(.95 * len(durations)) - 1] if durations else None
    attempts = [row for row in events if row["event"] == "attempt_finished"]
    by_attempt = {(row["group_id"], row["member_id"], row["attempt_id"]): row for row in attempts}
    selected = [by_attempt[(group["group_id"], member, attempt)]
                for group in groups for member, attempt in group["selected_attempts"]]
    def observed(values, operation):
        return operation(values) if values and all(v is not None for v in values) else None
    rewards = [row.get("reward") for row in selected]
    successes = [row.get("business_success") for row in attempts]
    total_generated = observed([row.get("generated_tokens") for row in attempts], sum)
    effective = observed([row.get("effective_training_tokens") for row in selected], sum)
    rates = observed(successes, lambda xs: sum(xs) / len(xs))
    result = {"spec": spec, "initial_conditions": initial_conditions,
              "runtime_contract": runtime_contract, "accounting_complete": accounting_complete,
              "allocation_intervals": allocation_intervals, "gpu_hours": hours,
              "legal_complete_groups": valid_count,
              "legal_complete_groups_per_gpu_hour": valid_count / hours if hours and valid_count is not None else None,
              "group_p95_s": p95, "group_p95_count": len(groups),
              "p95_sample_size_sufficient_for_screening": len(groups) >= min_tail_groups,
              "p95_stability_claim_supported": False,
              "mean_reward": observed(rewards, lambda xs: sum(xs) / len(xs)),
              "business_success_rate": rates,
              "generated_tokens_including_failed_attempts": total_generated,
              "mean_generated_tokens_per_attempt": total_generated / len(attempts) if total_generated is not None and attempts else None,
              "effective_training_tokens": effective,
              "effective_training_tokens_per_gpu_hour": effective / hours if effective is not None and hours else None,
              "failure_distribution": {}, "operation_costs": [row for row in events if row["event"] == "operation_cost"],
              "cached_tokens": None, "missing": {}}
    for row in attempts:
        if row.get("failure"):
            key = row["failure"]
            result["failure_distribution"][key] = result["failure_distribution"].get(key, 0) + 1
    native = [row for row in events if row["event"] == "native_request_returned"]
    result["cached_tokens"] = observed([row.get("cached_tokens") for row in native], sum)
    for key in ("mean_reward", "business_success_rate", "generated_tokens_including_failed_attempts",
                "effective_training_tokens", "cached_tokens"):
        if result[key] is None:
            result["missing"][key] = "Required original runtime observation was not supplied"
    if hours is None:
        result["missing"]["gpu_hours"] = allocation_reason
    if valid_count is None:
        result["missing"]["legal_complete_groups"] = "Original group accounting did not reach a confirmed terminal state"
    if p95 is None:
        result["missing"]["group_p95_s"] = "No complete group had first-dispatch and final legal-attempt observations"
    return result
