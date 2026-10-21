from dataclasses import asdict, replace

import pytest

from verl.experimental.fully_async_policy.trajectory_recovery import plan_trajectory_recovery
from verl.experimental.fully_async_policy.rollout_artifact import CanonicalPolicyKVIdentity
from verl.experimental.fully_async_policy.trajectory_snapshot import token_digest
from verl.workers.rollout.utils import ReplicaRoutingState, ReplicaCostSignal
from .test_trajectory_snapshot import snapshot


def setup():
    s = snapshot()
    i = CanonicalPolicyKVIdentity(**s.policy_identity)
    return s, i, [ReplicaRoutingState("b", i)]


def test_missing_load_or_cost_does_not_approve_remote_io():
    s, i, replicas = setup()
    plan = plan_trajectory_recovery(s, target_identity=i, replicas=replicas, load_observed={})
    assert plan.reason == "missing_load_observation" and not plan.allow_remote_cache_restore
    plan = plan_trajectory_recovery(s, target_identity=i, replicas=replicas, load_observed={"b": True})
    assert plan.reason == "missing_cost_observation" and not plan.allow_remote_cache_restore


def test_completed_prefix_and_measured_cost_are_both_required():
    s, i, replicas = setup()
    ref = {"export_complete": True, "prefix_tokens": 4, "token_digest": token_digest(s.agent_state["prompt_ids"][:4]),
        "policy_identity": asdict(i), "kv_bytes": 8,
        "storage_reference": {"backend": "mooncake", "master_address": "store:50051",
            "source_completed_tokens": 4, "objects": [{"key": "page", "bytes": 8, "sha256": "a" * 64}]}}
    s = replace(s, kv_ref=ref)
    costs = {"b": ReplicaCostSignal("b", 100, 100, remote_prefix_tokens=4,
        predicted_restore_latency_s=0.01, predicted_recompute_latency_s=0.1, remote_restore_approved=True)}
    kwargs = dict(target_identity=i, replicas=replicas, load_observed={"b": True}, measured_cost_signals=costs)
    assert not plan_trajectory_recovery(s, **kwargs).allow_remote_cache_restore
    plan = plan_trajectory_recovery(s, **kwargs, cost_evidence={"b": "test-fixture-profile"})
    assert plan.allow_remote_cache_restore and plan.confirmed_prefix_tokens == 4
    new_i = replace(i, policy_version="1", model_weights_fingerprint="new-weights")
    plan = plan_trajectory_recovery(s, target_identity=new_i, replicas=[ReplicaRoutingState("b", new_i)],
        load_observed={"b": True}, measured_cost_signals=costs, cost_evidence={"b": "test-fixture-profile"})
    assert not plan.allow_remote_cache_restore


def test_unavailable_or_incompatible_target_is_not_selected():
    s, i, _ = setup()
    with pytest.raises(ValueError, match="ready replica"):
        plan_trajectory_recovery(s, target_identity=i, replicas=[ReplicaRoutingState("a", i, lifecycle="draining")],
            load_observed={"a": True})
