"""Protocol fixtures only; no model, tokenizer asset, VeRL or GPU is loaded."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "project-b-final-validation-20260928" / "production-entry-20261008"


def module(filename):
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), TOOLS / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


freeze = module("freeze_workload_manifest.py")
receipts = module("collect_native_receipts.py")


def plan():
    return {
        "responses_per_group": 8,
        "phases": ["initial_cold", "natural_reuse", "weight_update", "new_version_cold", "new_version_reuse"],
        "arms": {
            "A": {"entry": "verl_default", "routing_policy": "existing_default"},
            "B": {"entry": "native_smg", "routing_policy": "least_inflight", "concurrency": 8},
            "C": {"entry": "native_smg", "routing_policy": "cache_aware", "concurrency": 8},
        },
        "budget": {
            "formal_windows": 3, "max_repair_rounds": 2, "max_window_seconds": 100,
            "max_requests_per_window": 64, "max_groups_per_window": 4,
        },
    }


def manifest(*, formal=False, with_plan=False):
    result = {
        "schema_version": 1, "formal_measurement_preconditions_met": formal,
        "tokenization": {"rows": [{"row_index": 0, "ordered_token_sha256": receipts.digest_json([1, 2, 3, 4, 5, 6])}]},
    }
    if with_plan:
        result["comparison"] = freeze.validate_plan(plan())
    result["manifest_sha256"] = receipts.digest_json(result)
    return result


def native_response(version="v0"):
    return {
        "output_ids": [11, 12],
        "meta_info": {
            "finish_reason": {"type": "stop"}, "prompt_tokens": 6, "completion_tokens": 2,
            "cached_tokens": 2, "weight_version": version,
            "weight_versions": [{"version": version, "start": 0, "end": 2}],
            "output_token_logprobs": [[-0.1, 11, None], [-0.2, 12, None]],
            "queue_time": 0.03, "first_token_latency": 0.2,
        },
    }


def events(frozen, *, arm="B", phases=None):
    phases = phases or ["initial_cold"]
    common = {"arm": arm, "window_id": f"{arm}-0", "manifest_sha256": frozen["manifest_sha256"]}
    records = []
    for phase_index, phase in enumerate(phases):
        group_id = f"g{phase_index}"
        version = "v1" if phase.startswith("new_version") else "v0"
        members = []
        for index in range(8):
            request_id = f"{group_id}-r{index}"
            records.append({
                **common, "event": "request", "request_id": request_id, "group_id": group_id,
                "member_index": index, "turn_index": 0, "replica_id": "fixture-replica", "phase": phase,
                "requested_weight_version": version, "accepted_weight_versions": [version],
                "final_response": True, "native_response": native_response(version),
                "salt_match": True,
                "input_token_sha256": receipts.digest_json([1, 2, 3, 4, 5, 6]),
                "clock_domain": "fixture-driver",
                "started_monotonic_s": 1.0 + phase_index * 3,
                "completed_monotonic_s": 2.0 + phase_index * 3,
            })
            members.append({
                "member_index": index, "request_ids": [request_id], "complete": True,
                "training_semantics_valid": True, "reward": 0.0, "business_success": False,
            })
        records.append({
            **common, "event": "group", "group_id": group_id, "n": 8, "members": members,
            "phase": phase, "input_row_index": 0, "dispatch_index": phase_index,
        })
    if len(phases) == 4:
        records.append({
            **common, "event": "weight_update", "version_before": "v0", "version_after": "v1",
            "native_update_completed": True, "cache_clear_completed": True, "active_requests_at_update": 0,
            "clock_domain": "fixture-driver", "started_monotonic_s": 5.5, "completion_monotonic_s": 6.5,
        })
    records.append({
        **common, "event": "window", "closed": True, "active_requests_at_close": 0,
        "window_seconds": 20.0, "allocated_gpu_seconds": 40.0, "gpu_uuids": ["fixture-GPU-0", "fixture-GPU-1"],
        "allocation_observation_source": "runtime_gpu_allocation_interval",
        "clock_domain": "fixture-driver", "initial_cache_clear_completed": True,
        "comparison_id": "fixture-comparison",
        "phase_sequence": plan()["phases"], "arm_config_sha256": receipts.digest_json(plan()["arms"][arm]),
    })
    return records


def update(frozen):
    return {
        "event": "trainer_update", "arm": "B", "window_id": "B-0", "manifest_sha256": frozen["manifest_sha256"],
        "group_ids": ["g0"], "consumed_group_ids": ["g0"], "optimizer_step_completed": True,
        "optimizer_step_before": 7, "optimizer_step_after": 8, "gradient_norm": 0.4, "learning_rate": 1e-6,
        "positive_advantages": True, "negative_advantages": True, "changed_parameter_elements": None,
        "update_end_monotonic_s": 3.0, "ack_monotonic_s": 4.0,
    }


def test_only_native_policy_can_differ_between_b_and_c():
    expected = plan()
    assert freeze.validate_plan(expected)["C_vs_B_difference"] == "native_routing_policy_only"
    expected["arms"]["C"]["concurrency"] = 16
    with pytest.raises(ValueError, match="fields other"):
        freeze.validate_plan(expected)


@pytest.mark.parametrize("field,value", [("max_repair_rounds", 3), ("formal_windows", 10), ("max_requests_per_window", 0)])
def test_budgets_are_finite_before_execution(field, value):
    expected = plan()
    expected["budget"][field] = value
    with pytest.raises(ValueError):
        freeze.validate_plan(expected)


def test_existing_system_prompt_cannot_be_replaced_or_prepended():
    row = {"prompt": [{"role": "system", "content": "original"}, {"role": "user", "content": "query"}]}
    with pytest.raises(ValueError, match="differs"):
        freeze.prompt_messages(row, "changed", "row")
    with pytest.raises(ValueError, match="no existing"):
        freeze.prompt_messages(row, "original", "prepend_once")
    assert row["prompt"][0]["content"] == "original"


def test_missing_data_preserves_null_without_estimated_tokens(tmp_path):
    args = SimpleNamespace(
        dataset=[str(tmp_path / "missing.parquet")], system_prompt=None, tool_schema=None, tool_config=None,
        reward_source=None, template_kwargs=None, source_kind="unavailable", source_ref="fixture-source",
        system_prompt_mode="row", tokenizer=None, run_plan=None, tool_version=None, environment_version=None,
    )
    result = freeze.build_manifest(args)
    assert result["input_availability"] == "UNAVAILABLE"
    assert result["row_count"] is None
    assert result["tokenization"]["prompt_token_distribution"] is None
    assert not result["formal_measurement_preconditions_met"]
    assert result["identities"]["datasets"][0]["error_type"] == "FileNotFoundError"


def test_synthetic_tag_cannot_be_declared_business(tmp_path):
    dataset = tmp_path / "fixture.jsonl"
    dataset.write_text(json.dumps({"data_source": "synthetic/local_life", "prompt": [{"role": "user", "content": "fixture"}]}) + "\n")
    args = SimpleNamespace(
        dataset=[str(dataset)], system_prompt=None, tool_schema=None, tool_config=None, reward_source=None,
        template_kwargs=None, source_kind="internal_business", source_ref="fixture-source", system_prompt_mode="row",
        tokenizer=None, run_plan=None, tool_version=None, environment_version=None,
    )
    result = freeze.build_manifest(args)
    assert "synthetic_data_cannot_be_declared_internal_business" in result["errors"]
    assert not result["formal_measurement_preconditions_met"]


def test_invalid_tool_object_rejected_without_attribute_error(tmp_path):
    filename = tmp_path / "tools.json"
    filename.write_text('["not a tool"]')
    with pytest.raises(ValueError, match="tool object"):
        freeze.load_tools(filename)


def test_prefix_counts_are_input_properties_and_use_original_order():
    class FixtureTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["fixture_ids"]

    rows = [{"prompt": [{"role": "user", "fixture_ids": ids}]} for ids in ([1, 2, 3], [1, 2, 4], [1, 5])]
    result = freeze.measure_prompts(rows, FixtureTokenizer(), None, {}, None, "row")
    assert result["common_prefix_tokens_all_rows"] == 1
    assert result["adjacent_prefix_distribution_original_order"]["max"] == 2
    assert result["prefix_lengths_are_input_properties_not_actual_cache_hits"]
    assert "fixture_ids" not in json.dumps(result)


def test_empty_tokenizer_result_cannot_be_measured():
    class EmptyTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return []

    with pytest.raises(ValueError, match="actual token IDs"):
        freeze.measure_prompts([{"prompt": [{"role": "user", "content": "fixture"}]}], EmptyTokenizer(), None, {}, None, "row")


def test_uncached_input_is_not_exact_physical_compute():
    summary = receipts.summarize_request(events(manifest())[0])
    assert summary["uncached_input_tokens"] == 4
    assert summary["actual_prefill_compute"] is None
    assert summary["uncached_input_is_not_exact_compute_tokens"]


def test_scalar_only_version_does_not_become_token_span_proof():
    record = events(manifest())[0]
    del record["native_response"]["meta_info"]["weight_versions"]
    summary = receipts.summarize_request(record)
    assert summary["weight_versions"] is None
    assert summary["version_evidence"].startswith("SCALAR_ONLY")


@pytest.mark.parametrize("mutate", [
    lambda meta: meta.update(cached_tokens=7),
    lambda meta: meta.update(completion_tokens=3),
    lambda meta: meta.update(weight_version="unexpected"),
    lambda meta: meta.update(weight_versions=[{"version": "v0", "start": 1, "end": 2}]),
    lambda meta: meta.update(output_token_logprobs=[[-0.1, 99, None], [-0.2, 12, None]]),
    lambda meta: meta.update(prompt_tokens=True),
    lambda meta: meta.update(queue_time=float("nan")),
])
def test_native_corruption_rejected(mutate):
    record = events(manifest())[0]
    mutate(record["native_response"]["meta_info"])
    with pytest.raises(ValueError):
        receipts.summarize_request(record)


def test_invalid_native_response_fails_with_protocol_error():
    record = events(manifest())[0]
    record["native_response"] = []
    with pytest.raises(ValueError, match="native response"):
        receipts.summarize_request(record)


def test_complete_zero_reward_group_is_legal_but_not_business_success():
    frozen = manifest()
    result = receipts.collect(events(frozen), frozen)
    assert result["receipt_status"] == "COMPLETE"
    assert result["windows"][0]["complete_training_legal_groups"] == 1
    assert result["windows"][0]["all_members_business_success_groups"] == 0
    assert not result["eligible_for_performance_review"]
    assert not result["original_training_update_observed"]


def test_missing_native_cache_field_remains_null_and_incomplete():
    frozen = manifest()
    records = events(frozen)
    del records[0]["native_response"]["meta_info"]["cached_tokens"]
    result = receipts.collect(records, frozen)
    assert result["receipt_status"] == "INCOMPLETE"
    assert result["windows"][0]["requests"][0]["cached_tokens"] is None
    assert result["windows"][0]["requests"][0]["uncached_input_tokens"] is None


def test_missing_reward_cannot_count_as_a_legal_training_group():
    frozen = manifest()
    records = events(frozen)
    records[8]["members"][0]["reward"] = None
    result = receipts.collect(records, frozen)
    assert result["receipt_status"] == "INCOMPLETE"
    assert result["windows"][0]["complete_training_legal_groups"] == 0


def test_wrong_member_links_are_not_a_complete_group():
    frozen = manifest()
    records = events(frozen)
    records[0]["member_index"] = 7
    result = receipts.collect(records, frozen)
    assert result["receipt_status"] == "FAILED"
    assert any(item["reason"] == "request_group_member_link_mismatch" for item in result["errors"])


def test_duplicate_final_request_rejected():
    frozen = manifest()
    records = events(frozen)
    records.append(copy.deepcopy(records[0]))
    assert receipts.collect(records, frozen)["receipt_status"] == "FAILED"


def test_consumption_cannot_precede_optimizer_completion():
    record = update(manifest())
    record["ack_monotonic_s"] = 2.0
    with pytest.raises(ValueError, match="before update"):
        receipts.summarize_update(record)


def test_original_update_and_independent_parameter_change_are_separate():
    frozen = manifest()
    result = receipts.collect([*events(frozen), update(frozen)], frozen)
    assert result["original_training_update_observed"]
    actual_update = result["windows"][0]["updates"][0]
    assert actual_update["changed_parameter_elements"] is None
    assert not actual_update["independent_parameter_change_proved"]


def test_same_group_cannot_be_confirmed_twice():
    frozen = manifest()
    result = receipts.collect([*events(frozen), update(frozen), update(frozen)], frozen)
    assert result["receipt_status"] == "FAILED"
    assert any(item["reason"] == "duplicate_or_unknown_consumed_group" for item in result["errors"])


def test_string_is_not_a_group_id_array():
    record = update(manifest())
    record["consumed_group_ids"] = "g0"
    with pytest.raises(ValueError, match="list"):
        receipts.summarize_update(record)


def test_manifest_digest_change_rejected():
    frozen = manifest()
    frozen["formal_measurement_preconditions_met"] = True
    with pytest.raises(ValueError, match="Manifest digest"):
        receipts.collect([], frozen)


def test_fully_observed_b_c_pair_is_reviewable_without_claiming_benefit():
    frozen = manifest(formal=True, with_plan=True)
    phases = ["initial_cold", "natural_reuse", "new_version_cold", "new_version_reuse"]
    result = receipts.collect([*events(frozen, arm="B", phases=phases), *events(frozen, arm="C", phases=phases)], frozen)
    assert result["receipt_status"] == "COMPLETE"
    assert result["eligible_for_performance_review"]
    assert result["performance_benefit"] == "NOT_ADJUDICATED"
    assert result["production_deployment"] == "NOT_PROVED"
    assert not result["original_training_update_observed"]


def test_b_only_window_cannot_claim_native_policy_comparison():
    frozen = manifest(formal=True, with_plan=True)
    phases = ["initial_cold", "natural_reuse", "new_version_cold", "new_version_reuse"]
    result = receipts.collect(events(frozen, arm="B", phases=phases), frozen)
    assert result["receipt_status"] == "COMPLETE"
    assert not result["eligible_for_performance_review"]
    assert "matched_native_B_and_C_windows_not_observed" in result["performance_review_blockers"]


@pytest.mark.parametrize("salt_match", [False, None])
def test_salt_mismatch_or_missing_is_legal_training_but_not_fair_cache_comparison(salt_match):
    frozen = manifest(formal=True, with_plan=True)
    phases = ["initial_cold", "natural_reuse", "new_version_cold", "new_version_reuse"]
    records = [*events(frozen, arm="B", phases=phases), *events(frozen, arm="C", phases=phases)]
    records[0]["salt_match"] = salt_match
    result = receipts.collect(records, frozen)
    assert result["receipt_status"] == "COMPLETE"
    assert result["windows"][0]["complete_training_legal_groups"] == 4
    assert not result["eligible_for_performance_review"]
    assert "cache_salt_mismatch_or_not_observed" in result["performance_review_blockers"]
    assert result["windows"][0]["requests"][0]["salt_match"] is salt_match


def test_input_changed_after_freeze_blocks_comparison():
    frozen = manifest(formal=True, with_plan=True)
    phases = ["initial_cold", "natural_reuse", "new_version_cold", "new_version_reuse"]
    records = [*events(frozen, arm="B", phases=phases), *events(frozen, arm="C", phases=phases)]
    records[0]["input_token_sha256"] = "different-actual-input"
    result = receipts.collect(records, frozen)
    assert not result["eligible_for_performance_review"]
    assert "actual_initial_inputs_not_bound_to_frozen_token_hashes" in result["performance_review_blockers"]


def test_original_rows_must_match_between_native_windows():
    frozen = manifest(formal=True, with_plan=True)
    frozen["tokenization"]["rows"].append({"row_index": 1, "ordered_token_sha256": frozen["tokenization"]["rows"][0]["ordered_token_sha256"]})
    del frozen["manifest_sha256"]
    frozen["manifest_sha256"] = receipts.digest_json(frozen)
    phases = ["initial_cold", "natural_reuse", "new_version_cold", "new_version_reuse"]
    baseline = events(frozen, arm="B", phases=phases)
    treatment = events(frozen, arm="C", phases=phases)
    treatment[8]["input_row_index"] = 1
    result = receipts.collect([*baseline, *treatment], frozen)
    assert not result["eligible_for_performance_review"]
    assert "native_comparison_input_selection_or_order_differs" in result["performance_review_blockers"]


def test_new_version_request_before_update_completion_blocks_performance():
    frozen = manifest(formal=True, with_plan=True)
    phases = ["initial_cold", "natural_reuse", "new_version_cold", "new_version_reuse"]
    records = [*events(frozen, arm="B", phases=phases), *events(frozen, arm="C", phases=phases)]
    records[18]["started_monotonic_s"] = 6.0
    result = receipts.collect(records, frozen)
    assert not result["eligible_for_performance_review"]
    assert "version_update_and_cold_cache_phases_not_observed" in result["performance_review_blockers"]


def test_cross_process_monotonic_times_cannot_prove_phase_order():
    frozen = manifest(formal=True, with_plan=True)
    phases = ["initial_cold", "natural_reuse", "new_version_cold", "new_version_reuse"]
    records = [*events(frozen, arm="B", phases=phases), *events(frozen, arm="C", phases=phases)]
    records[0]["clock_domain"] = "another-process"
    assert not receipts.collect(records, frozen)["eligible_for_performance_review"]


def test_missing_physical_allocation_does_not_produce_GPU_hour_metric():
    frozen = manifest()
    records = events(frozen)
    records[-1]["gpu_uuids"] = None
    result = receipts.collect(records, frozen)
    assert result["windows"][0]["complete_legal_groups_per_allocated_GPU_hour"] is None


def test_window_budget_limit_enforced():
    frozen = manifest(formal=True, with_plan=True)
    records = events(frozen)
    records[-1]["window_seconds"] = 101.0
    result = receipts.collect(records, frozen)
    assert result["receipt_status"] == "FAILED"
    assert not result["eligible_for_performance_review"]
