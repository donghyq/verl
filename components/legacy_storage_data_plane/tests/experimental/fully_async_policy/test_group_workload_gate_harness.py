"""Gate collector contracts; fixtures are never recorded as cluster evidence."""

import asyncio
import importlib.util
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).parents[3]
HARNESSES = ROOT / "project-b-final-validation-20260928/terminal-20261007"
spec = importlib.util.spec_from_file_location("group_gate_harness", HARNESSES / "gate2_group_workload_harness.py")
harness = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = harness
spec.loader.exec_module(harness)


def window(policy, round_index=0, factor=1.0):
    return {
        "policy": policy, "round": round_index,
        "valid_groups": 1, "total_groups": 1, "all_counts_released": True,
        "output_digests": {"g/0": "same_actual_work"},
        "native_work_digests": {"g/0": "same_full_native_work"},
        "treatment_placement_complete": True,
        "group_completion_p50_s": 2.0 * factor,
        "group_completion_p95_s": 3.0 * factor,
    }


def test_fixed_comparator_accepts_only_matched_stable_improvement():
    windows = [window(policy, index, 0.9 if policy == "group_aware" else 1.0)
               for index in range(3) for policy in ("least_inflight", "sticky", "group_aware")]
    assert harness.classify(windows)[0] == "PASS"
    windows[-1]["output_digests"] = {"g/0": "different_actual_work"}
    assert harness.classify(windows)[0] == "NO_BENEFIT"


def test_comparator_stops_on_failure_or_regression():
    windows = [window(policy, factor=1.1 if policy == "group_aware" else 1.0)
               for policy in ("least_inflight", "sticky", "group_aware")]
    assert harness.classify(windows)[0] == "REGRESSED"
    windows[0]["valid_groups"] = 0
    assert harness.classify(windows)[1] == "correctness_or_request_release_failed"


def test_real_call_counter_releases_during_tool_wait():
    router = harness.WindowRouter("sticky", harness.mod.WorkloadCalibration(), ["a", "b"])
    context = harness.mod.GroupRequestContext("g/0", "g", 0, 1, "short", 64, 128)
    router.scheduler.register_group([context], ["a", "b"])
    replica = router.acquire("g/0")
    assert sum(router.counts.values()) == 1
    with pytest.raises(RuntimeError, match="twice"):
        router.acquire("g/0")
    router.release("g/0", replica, 5)
    assert sum(router.counts.values()) == 0
    assert router.scheduler.group_state("g")["running_members"] == 1
    assert router.call_intervals[0]["generated_tokens"] == 5


class ContractRunner:
    async def reset_window(self, **kwargs):
        pass

    async def execute(self, *, context, workload, acquire, release):
        replica = acquire()
        try:
            await asyncio.sleep(0)
            if context.member_index == 0:
                raise OSError("fixture transport failure")
            return {
                "generated_tokens": 1, "valid": True, "output_digest": "fixture",
                "native_work_digest": "CPU_fixture_native_work",
                "evidence_source": "CPU_test_fixture_only",
            }
        finally:
            release(replica, 1 if context.member_index else None)

    async def runtime_metadata(self):
        return {"allocated_gpu_count": None, "rollout_gpu_idle_ratio": None}


def test_failed_transport_retains_null_fields_and_releases_counts(tmp_path):
    groups = [{"group_id": key, "members": [
        {"workload_key": key, "prompt_tokens": 64, "max_new_tokens": 128,
         "raw_prompt": [{"role": "user", "content": f"CPU fixture {key}"}]} for _ in range(8)
    ]} for key in ("short", "medium", "long", "tool_heavy")]
    result = asyncio.run(harness.run_window(
        ContractRunner(), {"groups": groups},
        harness.mod.WorkloadCalibration(), "sticky", 0, ["a", "b"], tmp_path,
    ))
    assert result["valid_groups"] == 0
    assert result["all_counts_released"]
    assert result["valid_samples_per_gpu_hour"] is None
    assert result["rollout_gpu_idle_ratio"] is None
    traces = __import__("json").loads((tmp_path / "round-0-sticky-decisions.json").read_text())
    failed = next(trace for trace in traces if trace["member_index"] == 0)
    assert failed["actual_generated_tokens"] is None
    assert failed["prediction_error_s"] is None


def test_formal_group_cannot_combine_different_prompts_or_future_length_classes():
    groups = [{"group_id": key, "members": [
        {"workload_key": key, "prompt_tokens": 64, "max_new_tokens": 128,
         "raw_prompt": [{"role": "user", "content": key}]} for _ in range(8)
    ]} for key in ("short", "medium", "long", "tool_heavy")]
    assert harness.validate_workload({"groups": groups})
    groups[0]["members"][7]["raw_prompt"] = [{"role": "user", "content": "different task"}]
    with pytest.raises(ValueError, match="same prompt"):
        harness.validate_workload({"groups": groups})
    groups[0]["members"][7]["raw_prompt"] = [{"role": "user", "content": "short"}]
    groups[0]["members"][7]["workload_key"] = "future_long_output"
    with pytest.raises(ValueError, match="common known input"):
        harness.validate_workload({"groups": groups})


def test_treatment_with_fallback_or_unmatched_native_work_cannot_pass():
    windows = [window(policy, index, 0.9 if policy == "group_aware" else 1.0)
               for index in range(3) for policy in ("least_inflight", "sticky", "group_aware")]
    windows[-1]["treatment_placement_complete"] = False
    assert harness.classify(windows) == ("NO_BENEFIT", "formal_treatment_not_executed_for_every_trajectory")
    windows[-1]["treatment_placement_complete"] = True
    windows[-1]["native_work_digests"] = {"g/0": "changed_clipped_away_engine_tokens"}
    assert harness.classify(windows)[1].startswith("full_native_work_differs")
    del windows[-1]["native_work_digests"]
    assert harness.classify(windows)[1].startswith("full_native_work_evidence_missing")


def complete_workload():
    return {"groups": [{"group_id": key, "members": [
        {"workload_key": key, "prompt_tokens": 64, "max_new_tokens": 128,
         "raw_prompt": [{"role": "user", "content": key}]} for _ in range(8)
    ]} for key in ("short", "medium", "long", "tool_heavy")]}


def calibrated_profiles():
    return harness.mod.WorkloadCalibration([
        harness.mod.CalibrationObservation(key, replica, 64, 128, 4, 1.0, None, None,
                                           f"CPU-fixture/{key}/{replica}/{repeat}")
        for key in ("short", "medium", "long", "tool_heavy") for replica in ("a", "b") for repeat in range(3)
    ])


def test_formal_calibration_coverage_requires_bucket_budget_and_every_replica(tmp_path):
    document = complete_workload()
    profiles = calibrated_profiles()
    assert harness.calibration_coverage(document, profiles, ["a", "b"]) == {
        "complete": True, "checked": 64, "missing_profiles": [],
    }
    for member in document["groups"][0]["members"]:
        member["prompt_tokens"] = 1024
    coverage = harness.calibration_coverage(document, profiles, ["a", "b"])
    assert len(coverage["missing_profiles"]) == 16
    assert {row["replica"] for row in coverage["missing_profiles"]} == {"a", "b"}
    assert {row["prompt_bucket"] for row in coverage["missing_profiles"]} == {"lt_4096"}

    class NoCallsRunner:
        async def reset_window(self, **kwargs):
            raise AssertionError("missing profile must be rejected before any cache reset or engine call")

    with pytest.raises(ValueError, match="coverage"):
        asyncio.run(harness.run_window(NoCallsRunner(), document, profiles, "group_aware", 0, ["a", "b"], tmp_path))
    document = complete_workload()
    for member in document["groups"][1]["members"]:
        member["max_new_tokens"] = 256
    assert not harness.calibration_coverage(document, profiles, ["a", "b"])["complete"]
    assert not harness.calibration_coverage(complete_workload(), profiles, ["a", "missing-replica"])["complete"]


def test_formal_router_refuses_silent_least_inflight_fallback():
    router = harness.WindowRouter("group_aware", harness.mod.WorkloadCalibration(), ["a", "b"])
    context = harness.mod.GroupRequestContext("g/0", "g", 0, 1, "short", 64, 128)
    router.scheduler.register_group([context], ["a", "b"])
    with pytest.raises(RuntimeError, match="cannot fall back"):
        router.acquire("g/0")
    assert sum(router.counts.values()) == 0
    assert not router.active


def test_formal_entrypoint_stops_before_runner_creation_when_calibration_missing(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    workload = tmp_path / "workload.json"
    calibration = tmp_path / "calibration.json"
    output = tmp_path / "evidence"
    workload.write_text(json.dumps(complete_workload()))
    calibration.write_text(json.dumps({"role": "independent_calibration", "observations": []}))
    monkeypatch.setattr(harness, "load_plugin", lambda _: pytest.fail("formal runner must not be created"))
    with pytest.raises(ValueError, match="no formal engine calls"):
        asyncio.run(harness.main(SimpleNamespace(
            workload=str(workload), calibration=str(calibration), servers='{"a":"http://a","b":"http://b"}',
            output=str(output), runner="must-not-load.py",
        )))
    coverage = json.loads((output / "calibration-coverage.json").read_text())
    assert not coverage["complete"] and len(coverage["missing_profiles"]) == 64
