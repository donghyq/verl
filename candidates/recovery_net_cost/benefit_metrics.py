"""Finite benchmark accounting; no inference, reward or recovery implementation."""
import math
import statistics

ORDER = ("token_recompute", "token_recompute", "token_recompute", "physical_restore", "physical_restore", "token_recompute")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def window_metrics(row):
    """Count only native completed groups with all eight scored outputs."""
    groups = row.get("groups", [])
    if row.get("status") != "COMPLETE" or not groups:
        raise ValueError("Original native group did not complete")
    for group in groups:
        members = group.get("members", [])
        if (group.get("native_runtime_status") != "completed" or len(members) != 8
                or {x.get("member_index") for x in members} != set(range(8))
                or set(group.get("native_output_record_indices", [])) != set(range(8))
                or group.get("original_grpo_gate",{}).get("original_native_helpers_used") is not True
                or group.get("original_grpo_gate",{}).get("complete_prompt_groups") != 1
                or any(x.get("complete") is not True or x.get("history_preserved") is not True
                       or x.get("uid") != group.get("uid") or not finite(x.get("reward")) for x in members)
                or not finite(group.get("completion_seconds")) or group["completion_seconds"] <= 0):
            raise ValueError("Original eight-member/owner/history/reward coverage is incomplete")
    seconds = row.get("total_allocated_seconds")
    business_seconds = row.get("request_business_seconds")
    devices = row.get("actual_gpu_uuids", [])
    tokens = row.get("actual_new_model_tokens")
    if (not finite(seconds) or seconds <= 0 or not finite(business_seconds) or not 0 < business_seconds <= seconds
            or not devices or len(set(devices)) != len(devices) or type(tokens) is not int or tokens <= 0
            or row.get("source_and_target_cleanup_verified") is not True):
        raise ValueError("Actual device interval, token counts or final cleanup is unavailable")
    tails = sorted(x["completion_seconds"] for x in groups)
    gpu_hours = len(devices) * seconds / 3600
    return {"complete_groups_per_gpu_hour": len(groups) / gpu_hours,
            "group_p95_s": tails[math.ceil(.95 * len(tails)) - 1],
            "group_completion_distribution_s": tails, "group_count": len(groups),
            "total_gpu_hours": gpu_hours, "request_business_seconds": business_seconds,
            "request_business_gpu_hours": len(devices) * business_seconds / 3600,
            "generated_tokens_per_gpu_hour": tokens / gpu_hours,
            "seconds_per_1000_actual_new_model_tokens": seconds * 1000 / tokens,
            "actual_new_model_tokens": tokens,
            "effective_training_tokens":sum(m["effective_training_tokens"] for g in groups for m in g["members"]),
            "nonzero_advantage_group_count":sum(g["original_grpo_gate"].get("nonzero_training_advantage") is True for g in groups),
            "mean_original_reward": statistics.fmean(m["reward"] for g in groups for m in g["members"]),
            "p95_is_descriptive_only": True, "p95_stability_claim": False}


def _noise(values):
    return abs(values[0] - values[1]) / statistics.fmean(values)


def assess(windows, *, max_reward_drop=0.0):
    result = {"status": "NOT_ACCEPTED", "additional_windows_allowed": False,
              "maximum_windows": 6, "token_equality_prewaived": True,
              "performance_comparison_performed":False,"performance_direction":None,
              "p95_stability_claim": False, "statistical_significance_claim": False,
              "windows": windows}
    if len(windows) != 6 or tuple(x.get("arm") for x in windows) != ORDER:
        return dict(result, reason="EXACT_AA_PLUS_ABBA_BUDGET_NOT_COMPLETED")
    try:
        metrics = [window_metrics(row) for row in windows]
    except ValueError as error:
        return dict(result, reason=str(error))
    frozen = ("fixture_sha256", "sampling_sha256", "policy_identity_sha256", "original_reward_implementation_sha256")
    if any(not windows[0].get(key) or any(x.get(key) != windows[0][key] for x in windows) for key in frozen):
        return dict(result, reason="FIXED_ORIGINAL_WORKLOAD_POLICY_OR_REWARD_CHANGED")
    if any(row["actual_gpu_uuids"] != windows[0]["actual_gpu_uuids"] for row in windows):
        return dict(result, reason="ACTUAL_ALLOCATION_CHANGED")
    before_noise = {key:_noise([x[key] for x in metrics[:2]]) for key in
                   ("complete_groups_per_gpu_hour","group_p95_s","seconds_per_1000_actual_new_model_tokens")}
    a = [metrics[2], metrics[5]]
    b = [metrics[3], metrics[4]]
    comparisons = {}
    for key, higher in (("complete_groups_per_gpu_hour", True), ("group_p95_s", False),
                        ("seconds_per_1000_actual_new_model_tokens", False)):
        baseline = [x[key] for x in a]; treatment = [x[key] for x in b]
        ma, mb = statistics.fmean(baseline), statistics.fmean(treatment)
        effect = (mb-ma)/ma if higher else (ma-mb)/ma
        noise = max(_noise(baseline), _noise(treatment), before_noise[key])
        comparisons[key] = {"baseline_mean": ma, "physical_restore_mean": mb,
                            "relative_improvement": effect, "observed_noise_fraction": noise,
                            "above_noise": effect > noise}
    quality = statistics.fmean(x["mean_original_reward"] for x in b) >= statistics.fmean(x["mean_original_reward"] for x in a) - max_reward_drop
    useful = all(x["nonzero_advantage_group_count"]==x["group_count"] for x in metrics)
    passed = quality and useful and all(x["above_noise"] for x in comparisons.values())
    return dict(result, status="OBSERVED_FINITE_BUSINESS_BENEFIT" if passed else "NOT_ACCEPTED",
                performance_comparison_performed=True,
                reason=None if passed else "ORIGINAL_REWARD_OR_LEARNING_SIGNAL_FAILED_OR_EFFECT_NOT_ABOVE_NOISE",
                prior_same_arm_noise_fraction=before_noise, comparisons=comparisons,
                original_reward_preserved=quality, per_window_metrics=metrics,
                original_nonzero_learning_signal_preserved=useful,
                total_gpu_hours=sum(x["total_gpu_hours"] for x in metrics),
                metric_scope="Whole assigned visible devices including both cold starts, handoff, reward and cleanup; request interval separate; shared allocation is not GPU utilization",
                formal_generalized_benefit_claim=False)
