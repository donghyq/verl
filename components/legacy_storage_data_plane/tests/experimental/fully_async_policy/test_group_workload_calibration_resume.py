"""Fixed-slot continuation and failed-task cost boundaries; CPU fixtures only."""

import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "project-b-final-validation-20260928/terminal-20261007"


def module(name):
    spec = importlib.util.spec_from_file_location("calibration_test_" + name, SCRIPTS / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


native = module("gate2_tool_agent_runner")
prep = module("gate2_prepare_calibration")
evidence = prep.evidence


def write(filename, value):
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(value))


def raw_record(member, replica="replica-0", started=10.0):
    sampling = dict(member["sampling_params"], max_new_tokens=member["max_new_tokens"])
    inputs = [10 + list(evidence.CLASSES).index(member["workload_key"])]
    calls = [{"replica_id": replica, "start_at": started, "end_at": started + 0.3,
              "request": {"input_ids": inputs, "sampling_params": sampling,
                          "return_logprob": True, "stream": False},
              "result": {"text": "CPU fixture native termination", "output_ids": [42], "meta_info": {
                  "output_token_logprobs": [[-0.1, 42, None]], "completion_tokens": 1,
                  "prompt_tokens": 1, "cached_tokens": 0, "finish_reason": {"type": "stop"}}}}]
    heavy = member["workload_key"] == "tool_heavy"
    tools = [{"name": name, "parameters": args, "start_at": started + 0.31, "end_at": started + 0.4,
              "response": {"text": json.dumps(body), "image": None, "video": None}, "reward": 0.0, "error": None}
             for name, args, body in [
                 ("search_candidates", {"district": "CPU-only"}, {"candidates": [{"id": "V001"}]}),
                 ("query_venue_detail", {"venue_id": "<venue_id>" if heavy else "V001"},
                  {"error": "unknown venue_id"} if heavy else {"id": "V001"}),
             ]]
    return {"calls": calls, "tools": tools, "native_work": native.native_work_record(calls, tools), "elapsed_s": 1.0,
            "output": {"prompt_ids": inputs, "response_ids": [42], "response_mask": [1], "response_logprobs": [-0.1],
                       "metrics": {"tool_calls": 0.1}, "extra_fields": {
                           "local_life_search_calls": 1, "local_life_detail_ids": [] if heavy else ["V001"]}}}


def receipt(replicas):
    return {replica: {"success": True, "attempts": [{"url": base + "/flush_cache", "method": "GET",
            "status": 200, "body": "Cache flushed."}]} for replica, base in replicas.items()}


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    source, runtime, model = tmp_path / "source-r10", tmp_path / "runtime-r3", tmp_path / "CPU-model"
    runtime.mkdir()
    model.mkdir()
    source.mkdir()
    replicas = {replica: "http://CPU-fixture-" + replica for replica in evidence.REPLICAS}
    members = [{"workload_key": key, "raw_prompt": [{"role": "user", "content": "independent CPU " + key}],
                "prompt_tokens": 1, "max_new_tokens": 128, "min_tool_calls": 7 if key == "tool_heavy" else 2,
                "tool_selection": ["search_candidates", "query_venue_detail"],
                "sampling_params": {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "sampling_seed": 100 + index}}
               for index, key in enumerate(evidence.CLASSES)]
    document = {"model_path": str(model), "groups": [{"group_id": key, "members": [{**copy.deepcopy(member),
                "raw_prompt": [{"role": "user", "content": "formal CPU " + key}]} for _ in range(8)]}
                for key, member in zip(evidence.CLASSES, members, strict=True)]}
    workload, template = source / evidence.REL / "gate2_fixed_workload.json", source / evidence.REL / "gate2_independent_calibration_template.json"
    write(workload, document)
    write(template, {"role": "independent_calibration_template", "calibration_members": members})
    critical = ["verl/experimental/agent_loop/agent_loop.py", "verl/experimental/agent_loop/tool_agent_loop.py",
                "verl/experimental/agent_loop/tool_parser.py", "verl/experimental/agent_loop/utils.py",
                "verl/tools/schemas.py", "verl/tools/function_tool.py",
                "verl/experimental/fully_async_policy/group_workload_scheduler.py",
                "verl/experimental/fully_async_policy/local_life_workload.py", "verl/utils/chat_template.py",
                "verl/utils/tokenizer.py", evidence.REL + "/gate2_group_workload_harness.py",
                evidence.REL + "/launch_gate2_native_2a100.sh"]
    for name in critical:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    files = {name: evidence.file_sha(source / name) for name in critical + [str(workload.relative_to(source)), str(template.relative_to(source))]}
    files[evidence.REL + "/gate2_tool_agent_runner.py"] = "282231418a0cd0de7a79c01b1ee5d5cbe1d31c1e62d8180e95e3d3ae26059d2f"
    files[evidence.REL + "/gate2_prepare_calibration.py"] = "e5563b0ebc2804ed8d99d5371d7b6e7041a2e1607a8cf773a6e5a4679588b070"
    write(tmp_path / "source-r9-readback.json", {"source": "source-r9", "runtime": "runtime-r3", "files": files})
    versions = {"mooncake-transfer-engine": "0.3.12.post1", "accelerate": "1.15.0", "transformers": "5.12.1", "xgrammar": "0.2.1"}
    monkeypatch.setattr(evidence.importlib.metadata, "version", lambda name: versions[name])
    write(runtime / "runtime-readback.json", {"runtime": str(runtime), "actual_rollout_import": "PASS",
                                             "actual_versions": versions, "archive_sha256": evidence.RUNTIME_ARCHIVES})
    (model / "config.json").write_text('"CPU fixture, no model weights"')
    model_receipt = tmp_path / "model-13185-readback.json"
    write(model_receipt, {"status": "completed", "shared_model_path": str(model), "shared_copy_verified_files": ["config.json"],
                          "assets": [{"file": "config.json", "bytes": (model / "config.json").stat().st_size,
                                      "sha256": evidence.file_sha(model / "config.json")}]})
    old_run = tmp_path / "gate2-r2"
    old = old_run / "calibration"
    write(old_run / "launch-freeze.json", {"source": str(tmp_path / "source-r9"), "runtime": str(runtime),
          "formal_windows": 9, "calibration_calls": 24, "extra_rounds": 0, "servers": replicas, "tools_delay_ms": 500,
          "workload_sha256": evidence.file_sha(workload), "calibration_template_sha256": evidence.file_sha(template)})
    write(old_run / "launch-failure.json", {"traceback": "CPU fixture prior business-invalid calibration"})
    write(old_run / "commands.json", [["bash", str(tmp_path / "source-r9" / evidence.REL / "launch_gate2_native_2a100.sh"),
          str(model), str(old_run / "servers"), "30120", "30121", "0", "1"]])
    (old_run / "servers").mkdir()
    native_command = old_run / "servers/frozen-common-command.txt"
    native_command.write_text(shlex.join(evidence.native_command(model)))
    old_inventory = old_run / "servers/actual-gpu-inventory.txt"
    old_inventory.write_text("0, GPU-cpu-fixture-0, NVIDIA A100-SXM4-80GB, 81920 MiB\n1, GPU-cpu-fixture-1, NVIDIA A100-SXM4-80GB, 81920 MiB\n")
    inventory = tmp_path / "actual-current-gpu-inventory.txt"
    inventory.write_text(old_inventory.read_text())
    partial = []
    for position, (replica, index, repeat) in enumerate(evidence.slots(members)[:10]):
        raw = raw_record(members[index], replica, started=10.0 + position * 2)
        directory = old / "actual-engine-records"
        name = f"round-{repeat}-calibration-{replica}-{index}"
        filename = directory / (name + "-old.json")
        write(filename, raw)
        write(directory / (name + "-cache-reset.json"), receipt(replicas))
        if position < 9:
            partial.append(evidence.observation(raw, members[index], replica, filename))
    write(old / "calibration.partial.json", partial)
    return SimpleNamespace(source=source, runtime=runtime, model=model, workload=workload, template=template,
        model_readback=model_receipt, old=old, old_run=old_run, inventory=inventory, native_command=native_command,
        members=members, replicas=replicas, expected=[[10+i] for i in range(4)], new_calls=[])


def verify(frozen):
    return evidence.verify_resume_context(frozen.old, frozen.source, frozen.runtime, frozen.model,
        frozen.workload, frozen.template, frozen.replicas, frozen.model_readback, frozen.inventory, frozen.native_command)


def test_business_failure_is_a_complete_cost_observation_without_formal_eligibility(frozen):
    raw = raw_record(frozen.members[3])
    audit = native.service_observation_audit(raw, frozen.members[3])
    assert audit["service_observation_valid"] and not audit["business_valid"]
    assert audit["business_invalid_reasons"] == ["successful_tool_calls=1<required=7"]
    observation = evidence.observation(raw, frozen.members[3], "replica-0", "CPU-only")
    assert observation["elapsed_s"] == 1.0
    assert not prep.harness.classify([{"valid_groups": 3, "total_groups": 4, "all_counts_released": True}])[0] == "PASS"


@pytest.mark.parametrize("change", ["nan_logprob", "missing_ids", "unfinished_tool", "work_mismatch", "call_count_mismatch"])
def test_incomplete_native_work_is_not_cost_calibration(frozen, change):
    raw = raw_record(frozen.members[0])
    if change == "nan_logprob": raw["calls"][0]["result"]["meta_info"]["output_token_logprobs"][0][0] = float("nan")
    if change == "missing_ids": del raw["calls"][0]["result"]["output_ids"]
    if change == "unfinished_tool": raw["tools"][0]["end_at"] = None
    if change == "work_mismatch": raw["native_work"]["observed_generated_tokens"] += 1
    if change == "call_count_mismatch": raw["calls"][0]["result"]["meta_info"]["completion_tokens"] += 1
    assert not native.service_observation_audit(raw, frozen.members[0])["service_observation_valid"]


@pytest.mark.parametrize("change", ["seed", "input", "replica", "cache", "duplicate", "partial"])
def test_slot_resume_rejects_wrong_identity_even_with_plausible_filename(frozen, change):
    directory = frozen.old / "actual-engine-records"
    target = directory / "round-0-calibration-replica-0-0-old.json"
    raw = json.loads(target.read_text())
    if change == "seed": raw["calls"][0]["request"]["sampling_params"]["sampling_seed"] += 1
    if change == "input":
        raw["calls"][0]["request"]["input_ids"] = [999]
        raw["output"]["prompt_ids"] = [999]
    if change == "replica": raw["calls"][0]["replica_id"] = "replica-1"
    raw["native_work"] = native.native_work_record(raw["calls"], raw["tools"])
    write(target, raw)
    if change == "cache":
        reset = directory / "round-0-calibration-replica-0-0-cache-reset.json"
        value = json.loads(reset.read_text()); value["replica-1"]["attempts"][0]["status"] = 400; write(reset, value)
    if change == "duplicate": write(directory / "same-valid-looking-copy.json", raw)
    if change == "partial":
        value = json.loads((frozen.old / "calibration.partial.json").read_text()); value[0]["elapsed_s"] = 0.1
        write(frozen.old / "calibration.partial.json", value)
    with pytest.raises(ValueError):
        evidence.restore_slots(frozen.old, frozen.members, frozen.replicas, frozen.expected, native)


@pytest.mark.parametrize("change", ["model", "gpu", "runtime", "source", "flags", "workload", "budget"])
def test_context_is_fail_closed_on_model_runtime_source_device_or_budget_drift(frozen, change):
    assert verify(frozen)["legacy_get_server_info"] is None
    if change == "model": (frozen.model / "config.json").write_text("changed")
    if change == "gpu": frozen.inventory.write_text(frozen.inventory.read_text().replace("GPU-cpu-fixture-0", "GPU-another-host"))
    if change == "runtime":
        value = json.loads((frozen.runtime / "runtime-readback.json").read_text()); value["actual_versions"]["xgrammar"] = "0.1"
        write(frozen.runtime / "runtime-readback.json", value)
    if change == "source": (frozen.source / "verl/experimental/fully_async_policy/group_workload_scheduler.py").write_text("changed")
    if change == "flags": frozen.native_command.write_text(frozen.native_command.read_text().replace("--tp-size 1", "--tp-size 2"))
    if change == "workload": frozen.workload.write_text(frozen.workload.read_text() + " ")
    if change == "budget":
        value = json.loads((frozen.old_run / "launch-freeze.json").read_text()); value["calibration_calls"] = 25
        write(frozen.old_run / "launch-freeze.json", value)
    with pytest.raises(ValueError): verify(frozen)


def invoke(frozen, tmp_path, monkeypatch, verify_only=False):
    class Runner:
        def __init__(self, document):
            self.raw_dir = Path(document["engine_evidence_dir"])
            self.raw_dir.mkdir(parents=True)

        def prompt_token_ids(self, member): return [10 + list(evidence.CLASSES).index(member["workload_key"])]
        def prompt_token_count(self, member): return len(self.prompt_token_ids(member))

        async def reset_window(self, policy, round_index):
            self.window_id = f"round-{round_index}-{policy}"
            write(self.raw_dir / (self.window_id + "-cache-reset.json"), receipt(frozen.replicas))

        async def execute(self, context, workload, acquire, release):
            frozen.new_calls.append(context.trajectory_id)
            assert context.trajectory_id not in {f"calibration/{r}/{i}/{n}" for r,i,n in evidence.slots(frozen.members)[:10]}
            replica = acquire()
            release(replica, 1)
            raw = raw_record(workload, replica, started=100.0 + len(frozen.new_calls) * 2)
            raw["task_score"] = 0.0
            raw["trajectory_identity"] = {
                "trajectory_id": context.trajectory_id, "group_id": context.group_id, "member_index": 0, "group_size": 1,
                "workload_key": context.workload_key, "prompt_tokens": context.prompt_tokens, "max_new_tokens": context.max_new_tokens,
            }
            filename = self.raw_dir / (self.window_id + "-new.json")
            write(filename, raw)
            audit = native.service_observation_audit(raw, workload)
            return {**audit, "valid": audit["business_valid"], "task_score": 0.0, "generated_tokens": 1,
                    "elapsed_s": 1.0, "evidence_source": str(filename)}

    async def create_runner(document, server_urls): return Runner(document)

    plugin = SimpleNamespace(create_runner=create_runner, service_observation_audit=native.service_observation_audit,
                             sha=native.sha, CALIBRATION_METHOD_VERSION=native.CALIBRATION_METHOD_VERSION)
    monkeypatch.setattr(prep.harness, "load_plugin", lambda name: plugin)
    output = tmp_path / "new-run" / ("verify" if verify_only else "calibration")
    args = SimpleNamespace(workload=str(frozen.workload), calibration_template=str(frozen.template),
        servers=json.dumps(frozen.replicas), output=str(output), runner="CPU fixture, no engine", resume_calibration=str(frozen.old),
        source=str(frozen.source), runtime=str(frozen.runtime), model_readback=str(frozen.model_readback),
        current_gpu_inventory=str(frozen.inventory), current_native_command=str(frozen.native_command), verify_resume_only=verify_only)
    asyncio.run(prep.main(args))
    return output


def test_fixed_24_slots_reuse_ten_and_execute_only_fourteen_with_failure_states_preserved(frozen, tmp_path, monkeypatch):
    before = {file: evidence.file_sha(file) for file in frozen.old_run.rglob("*") if file.is_file()}
    output = invoke(frozen, tmp_path, monkeypatch)
    result = json.loads((output / "calibration.json").read_text())
    assert len(result["observations"]) == 24 and result["calibration_coverage"]["checked"] == 64
    assert result["reused_slots"] == 10 and result["new_calibration_slots"] == len(frozen.new_calls) == 14
    assert len(set(frozen.new_calls)) == 14
    assert result["business_distribution"]["business_true"] == 18
    assert result["business_distribution"]["business_false"] == 6
    assert result["formal_business_criteria_changed"] is False
    assert all(evidence.file_sha(file) == digest for file, digest in before.items())
    report = module("gate2_result_analysis").analyze(tmp_path / "no-formal-windows", output / "calibration.json", output / "prepared-workload.json")
    assert report["calibration_business_distribution"]["business_false"] == 6
    assert report["GROUP_AWARE_ROUTING"] is None and not report["performance_claim_allowed"]


def test_verify_resume_only_calls_no_engine_or_cache_and_reports_ten_fourteen(frozen, tmp_path, monkeypatch):
    output = invoke(frozen, tmp_path, monkeypatch, verify_only=True)
    value = json.loads((output / "resume-verification.json").read_text())
    assert value["reused_slots"] == 10 and value["remaining_slots"] == 14 and value["new_engine_requests"] == 0
    assert value["business_distribution"]["business_true"] == 9 and value["business_distribution"]["business_false"] == 1
    assert frozen.new_calls == [] and not list((output / "actual-engine-records").glob("*.json"))


def test_a_second_named_continuation_cannot_replay_any_started_original_slot(frozen, tmp_path):
    previous = tmp_path / "another-attempt"
    write(previous / "launch-freeze.json", {"resume_calibration": str(frozen.old)})
    # No request has started, so an earlier CPU identity failure need not burn
    # the remaining budget. Once reset/request evidence exists it is refused.
    verify(frozen)
    write(previous / "calibration/actual-engine-records/reset-with-unknown-request-status.json", {"CPU": "fixture"})
    with pytest.raises(ValueError, match="another continuation"):
        verify(frozen)
