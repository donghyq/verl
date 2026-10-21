"""Bounded native-routing comparison for the owned Qwen3.5 MLP job.

prepare reads actual mounted tokenizer and business Parquet, without Ray/CUDA.
run delegates each window to the existing native-entry executor supplied by the
run owner. It never implements a router, admission controller, tools or reward.
The executor must return original events/receipts; absent interfaces produce
NOT_ACCEPTED rather than invented measurements. This optional post-training
experiment is not a prerequisite for the first formal-model submission.

Executor API: run_window(spec, freeze, output_dir) -> metadata dictionary with
events_file, terminal_result_file, allocation_intervals, initial_conditions and
runtime_contract. All files must come from the owned native run. Allocation
intervals cover setup, flush/prewarm, failures/retries and final shutdown, with
actual UUIDs and MLP receipt references. See native_observations for observations.
"""

import argparse
import asyncio
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
from urllib.parse import urlsplit

from routing_experiment_hooks import digest, gpu_hours

HERE = Path(__file__).resolve().parent
R2_RELATIVE = "project-b-final-validation-20260928/production-entry-20261008/evidence/production-benefit-20261009/run_native_routing_compare_r2.py"
ARMS = ("baseline", "coordinated", "coordinated", "baseline")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def shared_prefix(prompts):
    if len(prompts) < 2:
        raise ValueError("At least two distinct original business records are required")
    prefix = list(prompts[0])
    for prompt in prompts[1:]:
        common = 0
        for left, right in zip(prefix, prompt):
            if left != right:
                break
            common += 1
        prefix = prefix[:common]
    return prefix


def representative_prompt(tokens):
    """The original requirement is 6–7K total first-turn tokens, with a shared prefix."""
    return 6000 <= len(tokens) <= 7000


def owned_urls(manifest):
    if manifest.get("scope") != "owned_mlp" or not manifest.get("run_id") or not manifest.get("source_receipt"):
        raise ValueError("Owned MLP run identity and original receipt reference are required")
    workers = manifest.get("worker_urls", [])
    if len(workers) < 2 or len(workers) != len(set(workers)):
        raise ValueError("Two distinct actual rollout workers are required")
    for url in [manifest["gateway_url"], *workers]:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
            raise ValueError("Use exact owned MLP URLs without credentials or query strings")
    return workers


class NativeR2Primitives:
    """Reuse r2's native HTTP/flush protocol, restricted to the owned MLP URLs.

    Construction and gateway_argv are read-only. flush performs an explicit
    owned-engine operation when the existing executor invokes it. r2.window's
    128-token first-turn workload and output-equality gate are not reused.
    """

    def __init__(self, freeze):
        self.freeze = freeze
        self.workers = owned_urls(freeze["owned_runtime"])
        self.urls = {freeze["owned_runtime"]["gateway_url"], *self.workers}
        path = freeze["r2_driver"]["path"]
        if file_sha(path) != freeze["r2_driver"]["sha256"]:
            raise ValueError("The reused routing-abba-r2 source changed")
        self.r2 = load_module(path, "routing_experiment_reused_r2")

    def flush(self, worker_url):
        if worker_url not in self.workers:
            raise ValueError("Refuse an engine outside this owned MLP run")
        reply = self.r2.http(worker_url.rstrip("/") + "/flush_cache", {})
        if reply.get("http_status") != 200 or not reply.get("text", "").startswith("Cache flushed."):
            raise RuntimeError("The actual native engine did not confirm cache flush")
        return reply

    def gateway_argv(self, spec):
        url = urlsplit(self.freeze["owned_runtime"]["gateway_url"])
        if url.port is None:
            raise ValueError("The owner must bind the actual Gateway listen port")
        if spec["native_policy"] != self.freeze["arm_policies"][spec["arm"]]:
            raise ValueError("Frozen native policy changed")
        return ["smg", "launch", "--host", "0.0.0.0", "--port", str(url.port),
                "--policy", spec["native_policy"], "--enable-rl", "--disable-retries",
                "--worker-urls", *self.workers]


def prepare(args):
    from omegaconf import OmegaConf

    if Path(args.output).exists():
        raise ValueError("Refuse to overwrite an existing comparison freeze")
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    source = Path(args.runtime_source_root).resolve(strict=True)
    # Reuse the verified native object. Re-verifying would re-read 72GB, while
    # rebuilding a plain loop with tools=None renders a different business prompt.
    original_freeze = getattr(args, "verified_native_freeze", None)
    fixture = getattr(args, "live_rendered_fixture", None)
    if not isinstance(original_freeze, dict) or not isinstance(fixture, dict):
        raise ValueError("Use the live before-teardown hook with the original verified freeze and Worker factory")
    if (read_json(args.native_freeze) != original_freeze
            or original_freeze.get("config_sha256") != file_sha(args.config)):
        raise ValueError("The already verified native config/freeze changed")

    rollout = config["actor_rollout_ref"]["rollout"]
    if args.scenario == "cold_worker" and args.native_policy != "cache_aware":
        raise ValueError("Cold/load comparison requires actual least_load versus cache_aware policies")
    if rollout["n"] != 8 or not rollout["multi_turn"]["enable"]:
        raise ValueError("Use the original eight-member multi-turn business configuration")
    if config["trainer"].get("use_v1") is not True or config["algorithm"]["adv_estimator"] != "grpo":
        raise ValueError("The existing native V1/GRPO entry is required")
    rows = fixture["original_records"]
    inputs = [row["prompt_ids"] for row in rows]
    actual_lengths = [len(tokens) for tokens in inputs]
    input_files = [{"path": row["path"], "sha256": row["sha256"]}
                   for row in original_freeze["files"] if row["role"] == "train_data"]
    if not input_files or any(file_sha(row["path"]) != row["sha256"] for row in input_files):
        raise ValueError("Original business data changed after native preparation")
    length_distribution = {"original_records": len(actual_lengths),
        "minimum": min(actual_lengths) if actual_lengths else None,
        "maximum": max(actual_lengths) if actual_lengths else None,
        "p50": statistics.median(actual_lengths) if actual_lengths else None,
        "p95": sorted(actual_lengths)[math.ceil(.95 * len(actual_lengths)) - 1] if actual_lengths else None,
        "records_between_6000_and_7000_tokens": sum(6000 <= length <= 7000 for length in actual_lengths),
        "representative_original_records_selected": len(rows),
        "multi_turn_history_growth_tokens": None,
        "history_growth_reason": "Requires actual later business turns; no model was called during prepare"}
    if (len(rows) < max(2, args.groups_per_window)
            or len({row["original_record_sha256"] for row in rows}) < 2
            or len({digest(tokens) for tokens in inputs}) < 2
            or any(not representative_prompt(tokens) or len(tokens) > rollout["prompt_length"] for tokens in inputs)):
        result = {"status": "REAL_PROMPT_WINDOW_NOT_AVAILABLE", "original_prompt_length_distribution": length_distribution,
                  "reason": "Insufficient distinct original 6–7K first-turn business prompts; do not pad or rewrite them",
                  "gpu_operations": 0, "formal_model_calls": 0}
        write_json(args.output, result)
        return result
    prefix = shared_prefix(inputs)
    if not prefix:
        raise ValueError("The selected original business prompts have no actual shared token prefix")
    owner = read_json(args.owned_runtime)
    workers = owned_urls(owner)
    r2 = (source / R2_RELATIVE).resolve(strict=True)
    executor = Path(args.executor).resolve(strict=True)
    limits = read_json(args.quality_limits)
    for key in ("max_reward_drop", "max_success_rate_drop", "min_generation_length_ratio", "max_generation_length_ratio"):
        if key not in limits or not isinstance(limits[key], (int, float)) or not math.isfinite(limits[key]):
            raise ValueError("Explicit business quality limit is required: " + key)
    if args.gpu_budget_hours <= 0 or args.max_window_s <= 0 or args.allocated_gpus <= 0:
        raise ValueError("Freeze a positive full-cost GPU budget and bounded window duration")
    if args.allocated_gpus * args.max_window_s / 3600 * 6 > args.gpu_budget_hours:
        raise ValueError("Budget cannot cover the two same-arm noise windows plus bounded ABBA comparison")
    freeze = {"schema": 1, "status": "PREPARED_NOT_EXECUTED", "scenario": args.scenario,
              "native_policy": args.native_policy,
              "arm_policies": {"baseline": "least_load" if args.scenario == "cold_worker" else args.native_policy,
                               "coordinated": args.native_policy}, "group_size": 8,
              "groups_per_window": args.groups_per_window, "group_p95_min_count": args.min_p95_groups,
              "native_config": {"path": str(Path(args.config).resolve()), "sha256": file_sha(args.config)},
              "native_freeze": {"path": str(Path(args.native_freeze).resolve()), "sha256": file_sha(args.native_freeze)},
              "runtime_source_root": str(source), "model_source_identity": {
                  "model_path": original_freeze["model_path"], "policy_identity": original_freeze["policy_identity"],
                  "already_verified_provenance": original_freeze.get("model_identity_verification")},
              "tokenizer_files": [{"path": row["path"], "sha256": row["sha256"]}
                  for row in original_freeze["files"] if "tokenizer" in row["role"] or "chat_template" in row["role"]],
              "original_business_input_files": input_files,
              "ordered_original_rows": [{key: value for key, value in row.items() if key != "prompt_ids"} for row in rows],
              "window_input_order_sha256": fixture["window_input_order_sha256"],
              "common_prefix_tokens": len(prefix), "common_prefix_sha256": digest(prefix),
              "common_prefix_fraction_by_selected_prompt": [len(prefix) / len(tokens) for tokens in inputs],
              "original_prompt_length_distribution": length_distribution,
              "continuous_token_builder": "worker.trajectory_entry._new_loop(actual_original_record)",
              "actual_business_loop_classes": sorted({row["actual_loop_class"] for row in rows}),
              "builder_source_sha256": sorted({row["actual_loop_source_sha256"] for row in rows}),
              "original_sampling_params": {key: rollout.get(key) for key in ("temperature", "top_p", "top_k", "do_sample")},
              "sampling_and_rewards_must_remain_original": True,
              "partial_rollout_and_session_affinity_required": True,
              "r2_driver": {"path": str(r2), "sha256": file_sha(r2)},
              "executor": {"path": str(executor), "sha256": file_sha(executor)},
              "owned_runtime": owner, "hot_worker_rotation": [workers[0], workers[1], workers[0]],
              "gpu_budget_hours": args.gpu_budget_hours, "max_window_s": args.max_window_s,
              "allocated_gpus_for_budget_reservation_only": args.allocated_gpus,
              "quality_limits": limits, "max_abba_blocks": 1, "campaign_max_windows": 6,
              "only_treatment_difference": "native least_load versus cache_aware with identical controlled cold-worker initialization"
                  if args.scenario == "cold_worker" else "same native policy; version-switch coordination of router prediction records",
              "cache_failure_interpretation": "engine flush with stale router records means position-prediction mismatch or load skew; wrong KV reuse is not inferred",
              "shared_cache_present": None,
              "shared_cache_reason": "No shared-cache capability is assumed; if observed, exact model-version rejection needs its own proof",
              "new_output_token_equality_required": False,
              "previously_committed_history_must_be_unchanged": True}
    write_json(args.output, freeze)
    return {"status": freeze["status"], "common_prefix_tokens": len(prefix), "groups": len(rows),
            "original_prompt_length_distribution": length_distribution,
            "gpu_operations": 0, "formal_model_calls": 0}


def specifications(freeze, repetition):
    result = []
    for index, arm in enumerate(ARMS):
        rotation = repetition + (index // 2 if freeze["scenario"] == "cold_worker" else 0)
        hot = freeze["hot_worker_rotation"][rotation % len(freeze["hot_worker_rotation"])]
        result.append({"repetition": repetition, "window_index": repetition * 4 + index, "arm": arm,
             "scenario": freeze["scenario"], "native_policy": freeze["arm_policies"][arm],
             "hot_worker_url": hot, "cold_worker_urls": [url for url in freeze["owned_runtime"]["worker_urls"] if url != hot],
             "group_size": freeze["group_size"], "groups_per_window": freeze["groups_per_window"],
             "input_order_sha256": digest(freeze["ordered_original_rows"]),
             "max_window_s": freeze["max_window_s"]})
    return result


def native_observations(metadata, spec, freeze):
    """Read original events, retaining null when legal member/timing fields are absent.

    NativeEventNames are the run owner's protocol. No alternate framework
    acceptance decision is made here. Existing learning-signal/update counts are
    retained as diagnostic evidence and never substituted for legal group tails.
    """
    events = [json.loads(line) for line in Path(metadata["events_file"]).read_text().splitlines() if line.strip()]
    terminal = read_json(metadata["terminal_result_file"])
    dispatch = [row for row in events if row.get("event") == "group_dispatch_observed"]
    admissions = [row for row in events if row.get("event") == "training_group_admission_observed"]
    metrics = metadata.get("original_metrics", {})
    hours, reason = gpu_hours(metadata.get("allocation_intervals"))
    legal_count = metrics.get("legal_complete_groups")
    if metadata.get("original_group_accounting_complete") is not True:
        legal_count = None
    # Values must be copied from the actual original acceptance/timing observer.
    worker_latencies = metrics.get("first_dispatch_to_final_legal_attempt_s")
    latencies = metrics.get("first_dispatch_to_driver_observed_group_available_s", worker_latencies)
    latencies = sorted(latencies) if isinstance(latencies, list) and latencies else None
    if latencies is not None and any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in latencies):
        raise ValueError("Invalid original group timing observation")
    if legal_count is not None and (type(legal_count) is not int or legal_count < 0):
        raise ValueError("Invalid original complete-group count")
    if latencies is not None and legal_count != len(latencies):
        latencies = None  # A partially observed tail cannot represent all legal groups.
    p95 = latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None
    result = {"spec": spec, "terminal_status": terminal.get("status"),
              "initial_conditions": metadata.get("initial_conditions", {}),
              "runtime_contract": metadata.get("runtime_contract", {}),
              "independent_run_id": metadata.get("independent_run_id"),
              "gpu_hours": hours, "allocation_intervals": metadata.get("allocation_intervals"),
              "legal_complete_groups": legal_count,
              "legal_complete_groups_per_gpu_hour": legal_count / hours if hours and legal_count is not None else None,
              "group_p95_s": p95, "group_p95_count": len(latencies) if latencies else 0,
              "group_completion_distribution_s": latencies,
              "group_timing_scope": metrics.get("group_timing_scope", "Original worker completion when exact clock domains are comparable"),
              "worker_completion_distribution_s": worker_latencies,
              "worker_completion_p95_s": sorted(worker_latencies)[math.ceil(.95 * len(worker_latencies)) - 1] if worker_latencies else None,
              "p95_sample_size_sufficient_for_screening": bool(latencies and len(latencies) >= freeze["group_p95_min_count"]),
              "p95_stability_claim_supported": False,
              "mean_reward": metrics.get("mean_reward"), "business_success_rate": metrics.get("business_success_rate"),
              "mean_generated_tokens_per_attempt": metrics.get("mean_generated_tokens_per_attempt"),
              "generated_tokens_including_failed_attempts": metrics.get("generated_tokens_including_failed_attempts"),
              "effective_training_tokens": metrics.get("effective_training_tokens"),
              "failure_distribution": metrics.get("failure_distribution"),
              "cached_tokens": metrics.get("cached_tokens"),
              "multi_turn_history_growth_tokens": metrics.get("multi_turn_history_growth_tokens"),
              "operation_costs": metrics.get("operation_costs"),
              "native_events": {"dispatch_events": len(dispatch), "admission_events": len(admissions),
                                "engine_version_observations": [], "learning_signal_observations": []}, "missing": {}}
    for row in events:
        event = row.get("event")
        if event == "original_trainer_initialized":
            result["native_events"]["engine_version_observations"].append(row.get("publication", {}))
        elif event == "native_policy_publication_verified":
            result["native_events"]["engine_version_observations"].append({key: row.get(key) for key in
                ("trainer_step", "actual_parameter_version", "active_policy_record")})
        elif event == "native_learning_signal_observed":
            result["native_events"]["learning_signal_observations"].append({key: row.get(key) for key in
                ("trainer_step", "complete_prompt_groups", "effective_tokens", "nonzero_advantage_tokens")})
    result["native_events"]["dispatch_payloads"] = dispatch
    result["native_events"]["admission_payloads"] = admissions
    generated = result["generated_tokens_including_failed_attempts"]
    result["gpu_seconds_per_1000_generated_tokens_auxiliary"] = (
        hours * 3600 * 1000 / generated if hours is not None and isinstance(generated, (int, float)) and generated > 0 else None)
    result["generated_token_normalization_is_auxiliary"] = True
    for key in ("legal_complete_groups", "group_p95_s", "mean_reward", "business_success_rate",
                "mean_generated_tokens_per_attempt", "generated_tokens_including_failed_attempts",
                "effective_training_tokens", "failure_distribution", "cached_tokens", "operation_costs"):
        if result[key] is None:
            result["missing"][key] = "Original event lacks complete member/attempt acceptance, timing or metric payload"
    if hours is None:
        result["missing"]["gpu_hours"] = metadata.get("allocation_reason") or reason
    return result


def quality_ok(base, treatment, limits):
    required = ("mean_reward", "business_success_rate", "mean_generated_tokens_per_attempt",
                "effective_training_tokens", "generated_tokens_including_failed_attempts", "failure_distribution")
    if any(row.get(key) is None for row in (base, treatment) for key in required):
        return False, "Original quality/workload/failure observations are incomplete"
    if treatment["mean_reward"] < base["mean_reward"] - limits["max_reward_drop"]:
        return False, "Reward exceeds the frozen degradation limit"
    if treatment["business_success_rate"] < base["business_success_rate"] - limits["max_success_rate_drop"]:
        return False, "Business success exceeds the frozen degradation limit"
    length = base["mean_generated_tokens_per_attempt"]
    if length <= 0:
        return False, "Baseline generated length is not positive"
    ratio = treatment["mean_generated_tokens_per_attempt"] / length
    if not limits["min_generation_length_ratio"] <= ratio <= limits["max_generation_length_ratio"]:
        return False, "Generated workload lies outside the frozen comparison range"
    return True, None


def assess(freeze, windows):
    """Report bounded ABBA samples/noise; no extra repetition requirement."""
    result = {"status": "NOT_ACCEPTED", "reason": None, "gpu_hours": None,
              "independent_repetitions": 0, "repetitions": [], "extend_windows_allowed": False,
              "p95_stability_claim_supported": False}
    if not windows or len(windows) % 4:
        result["reason"] = "A complete ABBA block with two windows per arm is required"
        return result
    hours = [window.get("gpu_hours") for window in windows]
    if any(value is None for value in hours):
        result["reason"] = "Actual allocation, failure/retry and setup costs are incomplete"
        return result
    result["gpu_hours"] = sum(hours)
    if result["gpu_hours"] > freeze["gpu_budget_hours"]:
        result["reason"] = "Frozen full-cost GPU budget exceeded"
        return result
    for offset in range(0, len(windows), 4):
        block = windows[offset:offset + 4]
        if tuple(row["spec"]["arm"] for row in block) != ARMS:
            result["reason"] = "Original ABBA order changed"
            return result
        conditions = [row.get("initial_conditions", {}) for row in block]
        keys = ("checkpoint_sha256", "input_order_sha256", "sampling_sha256", "router_initial_state_sha256")
        if any(not conditions[0].get(key) or any(row.get(key) != conditions[0][key] for row in conditions[1:]) for key in keys):
            result["reason"] = "The two arms do not prove identical original starting conditions"
            return result
        for a, b in ((conditions[0], conditions[1]), (conditions[3], conditions[2])):
            if any(not a.get(key) or a[key] != b.get(key) for key in ("engine_cache_initial_state_sha256", "hot_worker_url")):
                result["reason"] = "Paired arms do not prove identical controlled cache initialization"
                return result
        if freeze["scenario"] == "cold_worker" and conditions[0]["hot_worker_url"] == conditions[3]["hot_worker_url"]:
            result["reason"] = "Bounded ABBA did not exchange actual hot/cold worker roles"
            return result
        for row in block:
            contract = row.get("runtime_contract", {})
            if (contract.get("native_policy") != row["spec"]["native_policy"] or
                    contract.get("partial_rollout_preserved") is not True or contract.get("session_affinity_preserved") is not True or
                    contract.get("committed_history_preserved") is not True or contract.get("old_policy_worker_admission_blocked") is not True):
                result["reason"] = "Original native policy, strong baseline or policy/history safety is unproven"
                return result
            if contract.get("shared_cache_present") is True and contract.get("shared_cache_model_version_rejection_verified") is not True:
                result["reason"] = "Observed shared cache does not prove exact model-version rejection"
                return result
            if row.get("operation_costs") is None:
                result["reason"] = "Flush and prewarm costs were not separately observed"
                return result
            if freeze["scenario"] == "cold_worker" and contract.get("controlled_restart_or_eviction_confirmed") is not True:
                result["reason"] = "Controlled cold-worker restart or eviction is unproven"
                return result
        if freeze["scenario"] == "version_switch":
            states = [row["runtime_contract"].get("router_prediction_invalidated_at_switch") for row in block]
            if any(state is None for state in states) or states[0] == states[1] or states[2] == states[3]:
                result["reason"] = "No measured difference in version-switch coordination; this is functional evidence only"
                return result
            if any(row["runtime_contract"].get("engine_cache_invalidated_at_switch") is not True for row in block):
                result["reason"] = "Actual engine cache invalidation is unproven"
                return result
        if any(row.get("legal_complete_groups_per_gpu_hour") is None or row.get("group_p95_s") is None for row in block):
            result["reason"] = "Legal group throughput or first-dispatch-to-final-legal-attempt timing is unavailable"
            return result
        base, treatment = [block[0], block[3]], [block[1], block[2]]
        rates_a = [row["legal_complete_groups_per_gpu_hour"] for row in base]
        rates_b = [row["legal_complete_groups_per_gpu_hour"] for row in treatment]
        if min(*rates_a, *rates_b) <= 0:
            result["reason"] = "No positive legal group throughput in every window"
            return result
        noise = max(abs(rates_a[0] - rates_a[1]) / (sum(rates_a) / 2),
                    abs(rates_b[0] - rates_b[1]) / (sum(rates_b) / 2), freeze.get("prior_same_arm_noise_fraction", 0))
        effects = [b / a - 1 for a, b in zip(rates_a, rates_b)]
        for a, b in zip(base, treatment):
            okay, reason = quality_ok(a, b, freeze["quality_limits"])
            if not okay:
                result["reason"] = reason
                return result
        tails = [1 - b["group_p95_s"] / a["group_p95_s"] if a["group_p95_s"] > 0 else None
                 for a, b in zip(base, treatment)]
        result["repetitions"].append({"noise_fraction_from_same_arm_repeats": noise,
                                      "paired_group_throughput_gain": effects, "paired_group_p95_fraction_saved": tails})
        if min(effects) <= noise:
            result["reason"] = "Effect does not exceed same-arm noise; do not expand windows"
            return result
        if any(value is None or value < 0 for value in tails):
            result["reason"] = "Group tail regressed or is unavailable"
            return result
    blocks = [windows[index:index + 4] for index in range(0, len(windows), 4)]
    enough_tail = all(row.get("p95_sample_size_sufficient_for_screening") for row in windows)
    result.update(status="ACCEPTED_CONTROLLED_COMPARISON" if enough_tail else "LIMITED_CONTROLLED_COMPARISON_OBSERVED",
                  reason="Observed limited full-cost ABBA comparison exceeds same-arm noise and preserves quality",
                  abba_blocks=len(blocks), windows_per_arm=len(windows) // 2,
                  observed_legal_groups=sum(row.get("legal_complete_groups", 0) or 0 for row in windows),
                  independent_repetitions=None,
                  independence_reason="Separate jobs/repetitions are not required or inferred from window labels",
                  hot_cold_worker_roles_exchanged=len({row["initial_conditions"]["hot_worker_url"] for row in windows}) >= 2,
                  scope="Only the observed policy, workload, sample counts and budget; no P95 stability claim",
                  group_p95_is_descriptive_only=not enough_tail, extend_windows_allowed=False)
    result["strong_benefit_claim_supported"] = False
    return result


async def run_campaign(freeze, output, executor):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    windows, noise_windows = [], []
    reserve = freeze["allocated_gpus_for_budget_reservation_only"] * freeze["max_window_s"] / 3600
    preparation_hours, preparation_reason = gpu_hours(freeze.get("preparation_allocation_intervals"))
    if preparation_hours is None:
        summary = {"status": "NOT_ACCEPTED", "reason": "Actual GPU allocation during live preparation is unavailable",
            "preparation_allocation_reason": freeze.get("preparation_allocation_reason") or preparation_reason,
            "total_windows": 0, "routing_benefit": None, "extend_windows_allowed": False}
        write_json(out / "routing_experiment_summary.json", summary)
        return summary
    if 6 * reserve + preparation_hours > freeze["gpu_budget_hours"]:
        raise ValueError("Frozen budget does not cover bounded AA + ABBA windows")

    async def execute(spec):
            window_dir = out / f"routing_experiment_window_{spec['window_index']:02d}_{spec['arm']}"
            window_dir.mkdir()
            try:
                metadata = executor.run_window(spec, freeze, str(window_dir))
                if inspect.isawaitable(metadata):
                    metadata = await metadata
                result = native_observations(metadata, spec, freeze)
            except BaseException as error:
                summary = {"status": "NOT_ACCEPTED", "reason": f"Original executor failed: {type(error).__name__}",
                           "extend_windows_allowed": False, "gpu_hours": None,
                           "allocation_and_failed_run_receipts_required": True}
                write_json(out / "routing_experiment_summary.json", summary)
                raise
            write_json(window_dir / "routing_experiment_window.json", result)
            return result

    original_specs = specifications(freeze, 0)
    for index in range(2):
        spec = dict(original_specs[0], window_index=index, phase="prior_same_arm_noise")
        noise_windows.append(await execute(spec))
    rates = [row.get("legal_complete_groups_per_gpu_hour") for row in noise_windows]
    hours = [row.get("gpu_hours") for row in noise_windows]
    if any(value is None or value <= 0 for value in rates) or any(value is None for value in hours):
        summary = {"status": "NOT_ACCEPTED", "reason": "Prior same-arm actual allocation/group throughput is unavailable; do not add comparison windows",
                   "noise_probe_windows": noise_windows, "extend_windows_allowed": False, "strong_benefit_claim_supported": False}
        write_json(out / "routing_experiment_summary.json", summary)
        return summary
    prior_noise = abs(rates[0] - rates[1]) / (sum(rates) / 2)
    spent = sum(hours) + preparation_hours
    if spent + 4 * reserve > freeze["gpu_budget_hours"]:
        summary = {"status": "NOT_ACCEPTED", "reason": "Remaining full-cost budget cannot cover ABBA", "extend_windows_allowed": False,
                   "gpu_hours": spent, "noise_probe_windows": noise_windows}
        write_json(out / "routing_experiment_summary.json", summary)
        return summary
    comparison_freeze = dict(freeze, prior_same_arm_noise_fraction=prior_noise, gpu_budget_hours=freeze["gpu_budget_hours"]-spent)
    for spec in original_specs:
        spec = dict(spec, window_index=spec["window_index"]+2, phase="controlled_comparison")
        windows.append(await execute(spec))
    summary = assess(comparison_freeze, windows)
    summary.update(prior_same_arm_noise_fraction=prior_noise, noise_probe_windows=noise_windows,
                   total_windows=len(noise_windows)+len(windows), campaign_max_windows=6,
                   common_live_preparation_gpu_hours=preparation_hours,
                   preparation_allocation_intervals=freeze["preparation_allocation_intervals"],
                   evaluation_kind="cache_aware_load_routing_vs_original_least_load",
                   recovery_benefit_is_not_routing_benefit=True)
    if summary.get("gpu_hours") is not None:
        summary["gpu_hours"] += spent  # Includes all noise/setup/flush/prewarm/CPU restart costs.
    summary["campaign_legal_complete_groups"] = sum(row.get("legal_complete_groups") or 0 for row in noise_windows + windows)
    summary["campaign_legal_complete_groups_per_gpu_hour"] = (
        summary["campaign_legal_complete_groups"] / summary["gpu_hours"] if summary.get("gpu_hours") else None)
    write_json(out / "routing_experiment_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("config", "native-freeze", "runtime-source-root", "owned-runtime", "executor", "quality-limits", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--repo", help="Compatibility only; actual helper root is the verified --runtime-source-root")
    p.add_argument("--scenario", choices=("version_switch", "cold_worker"), default="version_switch")
    p.add_argument("--native-policy", choices=("least_load", "cache_aware"), default="cache_aware")
    p.add_argument("--groups-per-window", type=int, default=32)
    p.add_argument("--min-p95-groups", type=int, default=400)
    p.add_argument("--gpu-budget-hours", type=float, required=True)
    p.add_argument("--max-window-s", type=float, required=True)
    p.add_argument("--allocated-gpus", type=int, required=True)
    r = commands.add_parser("run")
    r.add_argument("--freeze", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--execute-owned-native", action="store_true")
    a = commands.add_parser("assess")
    a.add_argument("--freeze", required=True)
    a.add_argument("--window", action="append", required=True)
    a.add_argument("--output", required=True)
    o = commands.add_parser("observe")
    o.add_argument("--events", required=True)
    o.add_argument("--terminal-result", required=True)
    o.add_argument("--metadata", help="Original observer metadata, including measured allocations/metrics when available")
    o.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "observe":
        metadata = read_json(args.metadata) if args.metadata else {}
        metadata.update(events_file=args.events, terminal_result_file=args.terminal_result)
        result = native_observations(metadata, {"scope": "original_run_observation_only"}, {"group_p95_min_count": 400})
        result["status"] = "OBSERVATION_ONLY_NOT_ROUTING_BENEFIT"
        write_json(args.output, result)
    else:
        frozen = read_json(args.freeze)
        if args.command == "assess":
            result = assess(frozen, [read_json(path) for path in args.window])
            write_json(args.output, result)
        else:
            if not args.execute_owned_native or sys.platform != "linux":
                raise RuntimeError("Explicit existing owned Linux native execution is required")
            if frozen.get("status") != "PREPARED_NOT_EXECUTED":
                raise ValueError("Actual first-turn business prompt window was not prepared")
            owned_urls(frozen["owned_runtime"])
            for key in ("executor", "r2_driver", "native_config", "native_freeze"):
                if file_sha(frozen[key]["path"]) != frozen[key]["sha256"]:
                    raise ValueError("Frozen original input changed: " + key)
            executor = load_module(frozen["executor"]["path"], "routing_experiment_owned_executor")
            if not callable(getattr(executor, "run_window", None)):
                raise ValueError("Existing entry executor must expose run_window(spec, freeze, output_dir)")
            result = asyncio.run(run_campaign(frozen, args.output, executor))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
