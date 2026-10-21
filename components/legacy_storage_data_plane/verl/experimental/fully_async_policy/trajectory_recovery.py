"""Use the existing replica selector only when resuming a committed trajectory."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from verl.experimental.fully_async_policy.rollout_artifact import CanonicalPolicyKVIdentity
from verl.experimental.fully_async_policy.trajectory_snapshot import TrajectorySnapshot, token_digest
from verl.workers.rollout.utils import (
    GroupPlacementRequest, ReplicaCostSignal, ReplicaRoutingState, select_group_affinity_replica,
)


@dataclass(frozen=True)
class TrajectoryRecoveryPlan:
    replica_id: str
    prefix_source: str
    reason: str
    allow_remote_cache_restore: bool
    confirmed_prefix_tokens: int = 0
    estimates: tuple = ()


def plan_trajectory_recovery(
    snapshot: TrajectorySnapshot, *, target_identity: CanonicalPolicyKVIdentity,
    replicas: list[ReplicaRoutingState], load_observed: Mapping[str, bool],
    measured_cost_signals: Mapping[str, ReplicaCostSignal] | None = None,
    cost_evidence: Mapping[str, str] | None = None,
) -> TrajectoryRecoveryPlan:
    """Never invent idle queues, prefix availability or a measured cost.

    `replicas` must contain the current, observed model identity and readiness.
    Cost evidence points to the frozen measured profile for this workload,
    target and cache conditions. Its collection belongs to the adapter; this
    function does not fabricate unavailable observations or route normal work.
    The serving adapter must recheck ownership and actual weights before I/O.
    """
    snapshot.validate()
    eligible = [r for r in replicas if r.policy_kv_identity == target_identity
                and r.lifecycle == "ready" and r.accepts_requests
                and r.runtime_validity.is_valid() and r.kv_cache_pressure < 0.95]
    if not eligible:
        raise ValueError("no ready replica with the target weight and KV layout")
    if not all(load_observed.get(r.replica_id, False) for r in eligible):
        return TrajectoryRecoveryPlan(sorted(eligible, key=lambda r: r.replica_id)[0].replica_id,
            "token_only_reprefill", "missing_load_observation", False)
    ref = snapshot.kv_ref
    confirmed = bool(ref and ref.get("export_complete") is True
                     and ref.get("policy_identity") == asdict(target_identity))
    prefix = ref["prefix_tokens"] if confirmed else 0
    signals = dict(measured_cost_signals or {})
    evidence = cost_evidence or {}
    have_costs = all(r.replica_id in signals and evidence.get(r.replica_id) for r in eligible)
    if have_costs:
        # Remote approval is valid only for the actual externally confirmed
        # prefix. The cost model's approval alone is not an export receipt.
        from dataclasses import replace
        signals = {r.replica_id: replace(signals[r.replica_id],
            remote_restore_approved=bool(confirmed and signals[r.replica_id].remote_restore_approved
                and signals[r.replica_id].remote_prefix_tokens <= prefix),
            remote_prefix_tokens=min(signals[r.replica_id].remote_prefix_tokens, prefix))
            for r in eligible}
    request = GroupPlacementRequest(snapshot.group_id, snapshot.member_index, snapshot.group_size,
        token_digest(snapshot.agent_state["prompt_ids"]), target_identity,
        prompt_token_count=len(snapshot.agent_state["prompt_ids"]))
    decision = select_group_affinity_replica(request, eligible, cost_signals=signals if have_costs else None)
    if decision.replica_id is None:
        raise ValueError("no valid recovery target")
    estimates = decision.finish_time_estimates
    chosen = next((x for x in estimates if x.replica_id == decision.replica_id), None)
    source = chosen.chosen_prefix_source if chosen else "token_only_reprefill"
    remote = source == "remote_restore"
    return TrajectoryRecoveryPlan(decision.replica_id, source,
        decision.reason if have_costs else "missing_cost_observation", remote,
        prefix if remote else 0, tuple(x.to_dict() for x in estimates))
