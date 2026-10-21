"""Four fixed windows; output equality is waived, native work is reported."""
import math
import statistics


def summarize(windows):
    result = {"status": "NOT_ACCEPTED", "window_budget": "baseline,baseline,cache_aware,cache_aware",
              "additional_windows_allowed": False, "token_equality_required": False,
              "statistical_significance_claim": False, "windows": windows}
    if len(windows) != 4 or [x.get("arm") for x in windows] != ["baseline"] * 2 + ["cache_aware"] * 2:
        return {**result, "reason": "EXACT_FOUR_WINDOW_BUDGET_NOT_COMPLETED"}
    versions = {str(x.get("policy_version")) for x in windows}
    gpus = {x.get("allocated_gpu_count") for x in windows}
    groups = {x.get("expected_groups") for x in windows}
    if len(versions) != 1 or "None" in versions or len(gpus) != 1 or len(groups) != 1:
        return {**result, "reason": "VERSION_ALLOCATION_OR_WORKLOAD_CHANGED"}
    metrics = []
    for row in windows:
        durations = sorted(row.get("group_completion_seconds", []))
        wall, gpu, tokens = (row.get(k) for k in ("wall_s", "allocated_gpu_count", "generated_tokens"))
        if (row.get("status") != "COMPLETE" or row.get("expected_groups", 0) <= 0
                or row.get("legal_complete_groups") != row["expected_groups"]
                or len(durations) != row["expected_groups"]
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) or x <= 0 for x in [wall, gpu, tokens, *durations])):
            return {**result, "reason": "INCOMPLETE_OR_INVALID_ORIGINAL_BUSINESS_WINDOW"}
        hours = gpu * wall / 3600
        metrics.append({"complete_groups_per_gpu_hour": len(durations) / hours,
                        "group_p95_s": durations[math.ceil(.95 * len(durations)) - 1],
                        "generated_tokens_per_gpu_hour": tokens / hours,
                        "seconds_per_1000_generated_tokens": wall * 1000 / tokens})
    comparisons = {}
    for key, higher in (("complete_groups_per_gpu_hour", True), ("group_p95_s", False),
                        ("seconds_per_1000_generated_tokens", False)):
        base = [x[key] for x in metrics[:2]]
        treatment = [x[key] for x in metrics[2:]]
        a, b = statistics.fmean(base), statistics.fmean(treatment)
        noise_a = abs(base[0] - base[1]) / a
        noise_b = abs(treatment[0] - treatment[1]) / b
        effect = (b - a) / a if higher else (a - b) / a
        comparisons[key] = {"baseline_mean": a, "cache_aware_mean": b,
                            "baseline_two_window_noise": noise_a,
                            "cache_aware_two_window_noise": noise_b,
                            "relative_improvement": effect,
                            "above_observed_noise": effect > max(noise_a, noise_b)}
    passed = all(x["above_observed_noise"] for x in comparisons.values())
    return {**result, "status": "OBSERVED_ABOVE_TWO_WINDOW_NOISE" if passed else "NOT_ACCEPTED",
            "reason": None if passed else "BENEFIT_NOT_ABOVE_FIXED_WINDOW_NOISE",
            "per_window_metrics": metrics, "comparisons": comparisons,
            "primary_metrics": ["complete_groups_per_gpu_hour", "group_p95_s"],
            "work_normalization": "seconds_per_1000_actual_returned_model_tokens",
            "cache_hit_is_process_metric_only": True}
