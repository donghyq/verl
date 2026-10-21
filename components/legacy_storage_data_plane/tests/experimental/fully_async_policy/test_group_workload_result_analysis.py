"""Saved-result analysis tests; all engine records are CPU fixtures only."""

import copy
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).parents[3]
FILE = ROOT / "project-b-final-validation-20260928/terminal-20261007/gate2_result_analysis.py"
spec = importlib.util.spec_from_file_location("gate_result_report_test", FILE)
reporter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reporter
spec.loader.exec_module(reporter)


def write(filename, value):
    filename.write_text(json.dumps(value))


def raw_record():
    calls = [{"request": {"input_ids": list(range(64)), "sampling_params": {"max_new_tokens": 128, "sampling_seed": 42}},
              "result": {"text": "CPU fixture output", "meta_info": {
                  "output_token_logprobs": [[-0.1, 5], [-0.2, 6], [-0.3, 7]], "completion_tokens": 3,
              }}}]
    tools = [{"name": "query_venue_detail", "parameters": {"venue_id": "CPU-only"},
              "response": {"text": "CPU fixture tool result"}, "reward": 0.0, "error": None}]
    return {"calls": calls, "tools": tools, "native_work": reporter.native.native_work_record(calls, tools),
            "output": {"response_ids": [5, 6], "response_mask": [1, 1]}, "elapsed_s": 1.0}


@pytest.fixture
def evidence(tmp_path):
    acceptance = tmp_path / "acceptance"
    raw_dir = tmp_path / "raw"
    acceptance.mkdir()
    raw_dir.mkdir()
    groups = [{"group_id": key, "members": [{"raw_prompt": [{"role": "user", "content": f"CPU fixture {key}"}],
               "workload_key": key, "prompt_tokens": 64, "max_new_tokens": 128} for _ in range(8)]}
              for key in ("short", "medium", "long", "tool_heavy")]
    document = {"groups": groups, "engine_evidence_dir": str(raw_dir)}
    raw = raw_record()
    native_digest = reporter.native.sha(raw["native_work"])
    output_digest = reporter.native.sha(raw["output"])
    observations = []
    for group in groups:
        for replica in ("a", "b"):
            for repeat in range(3):
                source = raw_dir / f"calibration-{group['group_id']}-{replica}-{repeat}.json"
                write(source, raw)
                observations.append({"workload_key": group["group_id"], "replica_id": replica, "prompt_tokens": 64,
                                     "max_new_tokens": 128, "generated_tokens": 3, "elapsed_s": 1.0,
                                     "first_token_s": None, "tool_time_s": None, "source_id": str(source)})
    calibration = {"role": "independent_calibration", "observations": observations}
    profiles = reporter.harness.mod.WorkloadCalibration([reporter.harness.mod.CalibrationObservation(**row) for row in observations])
    coverage = reporter.harness.calibration_coverage(document, profiles, ["a", "b"])
    freeze = {
        "workload_sha256": reporter.harness.digest(document), "calibration_sha256": reporter.harness.digest(calibration),
        "runner_sha256": __import__("hashlib").sha256(pathlib.Path(reporter.native.__file__).read_bytes()).hexdigest(),
        "scheduler_sha256": __import__("hashlib").sha256(pathlib.Path(reporter.harness.mod.__file__).read_bytes()).hexdigest(),
        "calibration_coverage": coverage, "rounds": 3, "tolerance": 0.05,
    }
    windows = []
    for round_index, order in enumerate(reporter.ORDERS):
        for policy in order:
            rows = []
            prefix = acceptance / f"round-{round_index}-{policy}"
            for group in groups:
                for member in range(8):
                    group_id = f"round-{round_index}/{group['group_id']}"
                    source = raw_dir / f"round-{round_index}-{policy}-{group['group_id']}-{member}.json"
                    write(source, raw)
                    rows.append({"group_id": group_id, "member_index": member, "trajectory_id": f"{group_id}/{member}",
                                 "valid": True, "error": None, "generated_tokens": 3, "output_digest": output_digest,
                                 "native_work_digest": native_digest, "evidence_source": str(source)})
            pathlib.Path(f"{prefix}-trajectories.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            write(pathlib.Path(f"{prefix}-decisions.json"), [{
                "trajectory_id": row["trajectory_id"],
                "reason": "minimum_predicted_group_completion" if policy == "group_aware" else "least_inflight_disabled",
            } for row in rows])
            write(pathlib.Path(f"{prefix}-calls.json"), [{"trajectory_id": row["trajectory_id"]} for row in rows])
            write(raw_dir / f"round-{round_index}-{policy}-cache-reset.json", {
                replica: {"success": True, "attempts": [{"status": 200, "body": "Cache flushed."}]} for replica in ("a", "b")
            })
            windows.append({
                "round": round_index, "policy": policy, "valid_samples": 32, "total_samples": 32,
                "valid_groups": 4, "total_groups": 4, "all_counts_released": True, "treatment_placement_complete": True,
                "group_completion_p50_s": 0.75 if policy == "group_aware" else 1.0,
                "group_completion_p95_s": 1.5 if policy == "group_aware" else 2.0,
                "rollout_gpu_idle_ratio": None, "valid_samples_per_gpu_hour": None,
                "output_digests": {f"{group['group_id']}/{member}": output_digest for group in groups for member in range(8)},
                "native_work_digests": {f"{group['group_id']}/{member}": native_digest for group in groups for member in range(8)},
            })
    workload_path = tmp_path / "workload.json"
    calibration_path = tmp_path / "calibration.json"
    write(workload_path, document)
    write(calibration_path, calibration)
    write(acceptance / "freeze.json", freeze)
    write(acceptance / "windows.json", windows)
    return acceptance, calibration_path, workload_path, raw_dir


def test_complete_saved_fixture_reuses_verdict_and_preserves_actual_counts_and_null_metrics(evidence):
    acceptance, calibration, workload, _ = evidence
    report = reporter.analyze(acceptance, calibration, workload)
    assert report["analysis_status"] == "COMPLETE" and report["GROUP_AWARE_ROUTING"] == "PASS"
    assert report["calibration_coverage"]["checked"] == 64
    assert len(report["windows"]) == 9 and len(report["comparisons"]) == 6
    for counts in report["by_policy_actual_counts"].values():
        assert counts["actual_samples"] == counts["valid_samples"] == counts["observed_native_calls"] == 96
        assert counts["actual_groups"] == counts["valid_groups"] == 12
        assert counts["observed_tool_calls"] == 96 and counts["observed_generated_tokens"] == 288
    assert all(row["rollout_gpu_idle_ratio"] is None and row["valid_samples_per_gpu_hour"] is None for row in report["windows"])
    assert "null" in reporter.markdown(report)


def test_partial_windows_missing_raw_and_false_coverage_cannot_pass(evidence):
    acceptance, calibration, workload, raw_dir = evidence
    windows = json.loads((acceptance / "windows.json").read_text())
    write(acceptance / "windows.json", windows[:-1])
    report = reporter.analyze(acceptance, calibration, workload)
    assert report["analysis_status"] == "INCOMPLETE" and report["GROUP_AWARE_ROUTING"] is None
    assert not report["performance_claim_allowed"]
    write(acceptance / "windows.json", windows)
    # Remove a CPU fixture only; production/historical evidence is untouched.
    (raw_dir / "round-0-group_aware-short-0.json").unlink()
    report = reporter.analyze(acceptance, calibration, workload)
    assert report["GROUP_AWARE_ROUTING"] is None
    assert any("round-0-group_aware-short-0.json" in issue for issue in report["missing_or_inconsistent_evidence"])
    calibration_document = json.loads(calibration.read_text())
    calibration_document["observations"].pop()
    write(calibration, calibration_document)
    report = reporter.analyze(acceptance, calibration, workload)
    assert not report["calibration_coverage"]["complete"] and report["GROUP_AWARE_ROUTING"] is None


def test_actual_native_work_difference_is_no_benefit_even_when_clipped_outputs_match(evidence):
    acceptance, calibration, workload, raw_dir = evidence
    source = raw_dir / "round-0-group_aware-short-0.json"
    raw = json.loads(source.read_text())
    raw["calls"][0]["result"]["meta_info"]["output_token_logprobs"][-1][1] = 99
    raw["native_work"] = reporter.native.native_work_record(raw["calls"], raw["tools"])
    write(source, raw)
    digest = reporter.native.sha(raw["native_work"])
    rowfile = acceptance / "round-0-group_aware-trajectories.jsonl"
    rows = [json.loads(line) for line in rowfile.read_text().splitlines()]
    target = next(row for row in rows if row["group_id"] == "round-0/short" and row["member_index"] == 0)
    target["native_work_digest"] = digest
    rowfile.write_text("\n".join(json.dumps(row) for row in rows))
    windows = json.loads((acceptance / "windows.json").read_text())
    next(row for row in windows if row["round"] == 0 and row["policy"] == "group_aware")["native_work_digests"]["short/0"] = digest
    write(acceptance / "windows.json", windows)
    report = reporter.analyze(acceptance, calibration, workload)
    assert report["analysis_status"] == "COMPLETE" and report["GROUP_AWARE_ROUTING"] == "NO_BENEFIT"
    assert not report["comparisons"][0]["full_native_work_matched"] and report["comparisons"][0]["retained_output_matched"]


def test_corrupted_native_work_is_incomplete_and_observed_request_failure_is_regressed(evidence):
    acceptance, calibration, workload, raw_dir = evidence
    source = raw_dir / "round-0-group_aware-short-0.json"
    original = json.loads(source.read_text())
    changed = copy.deepcopy(original)
    changed["calls"][0]["result"]["meta_info"]["output_token_logprobs"][-1][1] = 99
    write(source, changed)
    assert reporter.analyze(acceptance, calibration, workload)["analysis_status"] == "INCOMPLETE"
    write(source, original)
    windows = json.loads((acceptance / "windows.json").read_text())
    windows[0]["all_counts_released"] = False
    write(acceptance / "windows.json", windows)
    report = reporter.analyze(acceptance, calibration, workload)
    assert report["analysis_status"] == "COMPLETE" and report["GROUP_AWARE_ROUTING"] == "REGRESSED"


def test_no_evidence_does_not_turn_empty_windows_into_pass(tmp_path):
    report = reporter.analyze(tmp_path, tmp_path / "no-calibration.json", tmp_path / "no-workload.json")
    assert report["GROUP_AWARE_ROUTING"] is None and report["observed_windows"] == 0
    assert report["analysis_status"] == "INCOMPLETE"


def test_helper_imports_only_standard_library_and_never_runs_runtime_operations():
    code = f"import runpy, sys; runpy.run_path({str(FILE)!r}, run_name='analysis_import_test'); assert 'torch' not in sys.modules and 'ray' not in sys.modules and 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True)


def test_copied_evidence_resolves_remote_paths_only_under_explicit_raw_root(evidence):
    acceptance, calibration, workload, raw_dir = evidence
    document = json.loads(workload.read_text())
    document["engine_evidence_dir"] = "/CPU-fixture-not-a-real-remote-path/raw"
    write(workload, document)
    calibration_document = json.loads(calibration.read_text())
    for row in calibration_document["observations"]:
        row["source_id"] = "/CPU-fixture-not-a-real-remote-path/raw/" + pathlib.Path(row["source_id"]).name
    write(calibration, calibration_document)
    for filename in acceptance.glob("*-trajectories.jsonl"):
        rows = [json.loads(line) for line in filename.read_text().splitlines()]
        for row in rows:
            row["evidence_source"] = "/CPU-fixture-not-a-real-remote-path/raw/" + pathlib.Path(row["evidence_source"]).name
        filename.write_text("\n".join(json.dumps(row) for row in rows))
    freeze = json.loads((acceptance / "freeze.json").read_text())
    freeze["workload_sha256"] = reporter.harness.digest(document)
    freeze["calibration_sha256"] = reporter.harness.digest(calibration_document)
    write(acceptance / "freeze.json", freeze)
    assert reporter.analyze(acceptance, calibration, workload)["analysis_status"] == "INCOMPLETE"
    assert reporter.analyze(acceptance, calibration, workload, raw_dir)["analysis_status"] == "COMPLETE"


def test_common_business_failures_preserve_negative_verdict_without_scheduler_causality(evidence):
    acceptance, calibration, workload, _ = evidence
    windows = json.loads((acceptance / "windows.json").read_text())
    for window in windows:
        filename = acceptance / f"round-{window['round']}-{window['policy']}-trajectories.jsonl"
        rows = [json.loads(line) for line in filename.read_text().splitlines()]
        for row in rows:
            if row["group_id"].endswith("/tool_heavy"):
                row["valid"] = False
        filename.write_text("\n".join(json.dumps(row) for row in rows))
        window["valid_samples"] = 24
        window["valid_groups"] = 3
    write(acceptance / "windows.json", windows)
    report = reporter.analyze(acceptance, calibration, workload)
    assert report["analysis_status"] == "COMPLETE" and report["GROUP_AWARE_ROUTING"] == "REGRESSED"
    assert all(len(members) == 8 for members in report["shared_formal_invalid_members"].values())
    assert "do not establish a scheduler-induced regression" in report["business_failure_causal_limit"]
    assert not report["performance_claim_allowed"]
