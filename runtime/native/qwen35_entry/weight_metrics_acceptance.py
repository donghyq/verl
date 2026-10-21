"""Join existing Qwen3.5 runtime receipts; never run training or create probes.

CLI: python weight_metrics_acceptance.py --inputs actual-inputs.json --output summary.json
Inputs are artifact paths plus the actual APPID, run_id, model/config fingerprint,
final source archive and per-phase stage identities. Missing receipts stay pending.
The source archive, freeze, owner stage events and native helpers are bound by SHA.
Only the frozen native audit and parameter-delta functions evaluate their proofs.
"""

import argparse
import ast
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path
from types import SimpleNamespace

import weight_metrics_collect as collect


HERE = Path(__file__).resolve().parent
QWEN_MODEL = "qwen3_5_moe"
FAILED_WORDS = ("FAIL", "ERROR", "KILL", "STOP", "REJECT", "NOT_ACCEPTED")


def observation(value, reason=None, *, source=None, scope=None):
    return {"value": value, "reason": reason if value is None else None, "source": source, "scope": scope}


def check(value, reason=None, *, source=None):
    return {"status": "VERIFIED" if value is True else "REJECTED" if value is False else "PENDING",
            "value": value, "reason": reason if value is not True else None, "source": source}


def finite_nonzero(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


class Evidence:
    def __init__(self, base):
        self.base, self.sources, self.errors = Path(base), [], []

    def read(self, reference, role, *, lines=False, raw=False):
        if not reference:
            return None
        ref = {"path": reference} if isinstance(reference, str) else reference
        path = Path(ref["path"])
        path = path if path.is_absolute() else self.base / path
        if not path.is_file():
            self.sources.append({"role": role, "path": str(path), "sha256": None, "reason": "artifact_not_collected"})
            return None
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        self.sources.append({"role": role, "path": str(path.resolve()), "sha256": digest, "reason": None})
        if ref.get("sha256") and ref["sha256"] != digest:
            self.errors.append(role + ": artifact differs from supplied SHA")
            return None
        if raw:
            return data
        try:
            return [json.loads(line) for line in data.decode().splitlines() if line.strip()] if lines else json.loads(data)
        except (ValueError, UnicodeError):
            self.errors.append(role + ": malformed JSON receipt")
            return None


def pure_native_functions(raw, names, source):
    """Use exact frozen pure audit functions without importing model/Ray code."""
    tree = ast.parse(raw, filename=source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise ValueError("Frozen native helper functions are missing: " + source)
    namespace = {"math": math, "json": json, "hashlib": hashlib}
    exec(compile(ast.Module(body=functions, type_ignores=[]), source, "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def source_binding(spec, evidence):
    manifest = evidence.read(spec.get("source_manifest"), "source_manifest")
    archive = evidence.read(spec.get("source_archive"), "source_archive", raw=True)
    expected = spec.get("source_sha256")
    if manifest is None or archive is None:
        return None, None, check(None, "actual final source manifest/archive not collected")
    actual = hashlib.sha256(archive).hexdigest()
    if not expected or actual != expected or manifest.get("archive_sha256") != expected:
        return manifest, None, check(False, "actual source archive and manifest do not match final source SHA")
    import io
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        name = "verl/checkpoint_engine/weight_update_profile.py"
        raw = bundle.extractfile(name).read()
    if hashlib.sha256(raw).hexdigest() != manifest["files"].get(name):
        return manifest, None, check(False, "native profile bytes differ from final source manifest")
    module = pure_native_functions(raw, ("summarize_profile_records",), name)
    return manifest, module, check(True, source="source_archive/source_manifest")


def native_helper(spec, freeze, evidence, *, role, names, default):
    raw = evidence.read(spec.get(role, str(default)), role, raw=True)
    expected = {row["sha256"] for row in freeze.get("files", []) if Path(row["path"]).name == default.name}
    if raw is None or not expected:
        return None
    if hashlib.sha256(raw).hexdigest() not in expected:
        evidence.errors.append(role + ": native helper differs from actual business freeze")
        return None
    return pure_native_functions(raw, names, default.name)


def ranks(rows, count=8):
    if not isinstance(rows, list) or len(rows) != count or any(type(row.get("rank")) is not int for row in rows):
        return None
    mapped = {row["rank"]: row for row in rows}
    return mapped if sorted(mapped) == list(range(count)) and len(mapped) == count else None


def actual_processes(rows, known_hashes):
    result = []
    for row in rows or []:
        process = row.get("_observer", {}).get("process")
        if not process:
            return None
        required = [source for source in process.get("sources", []) if source["module"] in
                    ("verl", "runtime_observer", "native_training_audit", "training_update_audit", "actor_worker")
                    or source["module"].startswith("verl.workers.engine")]
        uuid = process.get("gpu_uuid", {})
        uuid = uuid.get("value") if isinstance(uuid, dict) else None
        if (len(required) < 6 or any(source.get("matches_frozen") is not True or source.get("sha256") not in known_hashes
                                     for source in required) or not uuid or not process.get("ray_node_id")
                or not process.get("ray_actor_id") or type(process.get("pid")) is not int):
            return None
        result.append({"rank": row["rank"], "pid": process["pid"], "socket_hostname": process["socket_hostname"],
                       "ray_node_id": process["ray_node_id"], "ray_actor_id": process["ray_actor_id"], "gpu_uuid": uuid,
                       "physical_mlp_hostname": process.get("physical_mlp_hostname"), "sources": required})
    return result if len({row["gpu_uuid"] for row in result}) == 8 else None


def phase_summary(phase, identity, manifest, profile_module, evidence, runtime_allowed):
    stage = phase["stage_id"]
    events = evidence.read(phase.get("events"), stage + ".events", lines=True) or []
    result = evidence.read(phase.get("result"), stage + ".result")
    freeze = evidence.read(phase.get("freeze"), stage + ".freeze") or {}
    first = evidence.read(phase.get("first_update"), stage + ".first_update")
    final_parameters = evidence.read(phase.get("parameters_final"), stage + ".parameters_final")
    expected_source = identity.get("source_sha256")
    wrong_events = any(row.get("stage_id") != stage or row.get("source_sha") != expected_source for row in events)
    if wrong_events:
        evidence.errors.append(stage + ": events are from a different stage/source archive")
    family = freeze.get("mounted_hf_model_type")
    model_bound = (freeze.get("schema_version") == 35 and freeze.get("data_kind") == "production/business"
                   and family == QWEN_MODEL and freeze.get("model_configuration_fingerprint") == identity.get("model_configuration_fingerprint"))
    if freeze and not model_bound:
        evidence.errors.append(stage + ": freeze is not the actual Qwen3.5 business/model identity")
    resume, target = phase["resume_step"], phase["target_step"]
    if freeze and (freeze.get("native_resume_step") != resume or freeze.get("target_step") != target):
        evidence.errors.append(stage + ": frozen update frontier differs from requested phase")
    known_hashes = set((manifest or {}).get("files", {}).values()) | {row["sha256"] for row in freeze.get("files", [])}
    helper = native_helper(phase, freeze, evidence, role="native_audit", names=("rank_map", "verify_rank_update", "verify_publication"),
                           default=HERE.parents[1] / "continuation-20261012/implementation/continuous-training/native_training_audit.py")
    delta_helper = native_helper(phase, freeze, evidence, role="parameter_observer", names=("parameter_sample_delta",),
                                 default=HERE.parents[1] / "production-entry-20261008/run_v1_functional_gate.py")
    indexed = {}
    for row in events:
        step = row.get("trainer_step")
        if type(step) is int:
            indexed.setdefault((row["event"], step), []).append(row)
    expected_steps = list(range(resume + 1, target + 1))
    verified, publications, learning, details = [], [], [], []
    for step in expected_steps:
        before = indexed.get(("native_actor_update_dispatched", step), [])
        after = indexed.get(("native_actor_update_verified", step), [])
        publication = indexed.get(("native_policy_publication_verified", step), [])
        before = before[0] if len(before) == 1 else None
        after = after[0] if len(after) == 1 else None
        publication = publication[0] if len(publication) == 1 else None
        audit_error = None
        if helper and before and after:
            try:
                helper.verify_rank_update(before["observations"], after["observations"], rank_count=8, step=step)
                verified.append(step)
            except (ValueError, KeyError, TypeError) as error:
                audit_error = type(error).__name__ + ": " + str(error)
        if helper and publication:
            try:
                helper.verify_publication(publication, step, identity.get("model_configuration_fingerprint"))
                if publication["active_policy_record"]["run_id"] != identity.get("run_id"):
                    raise ValueError("Durable policy is from a different actual run")
                publications.append(step)
            except (ValueError, KeyError, TypeError) as error:
                audit_error = type(error).__name__ + ": " + str(error)
        if after and after.get("learning_signal_verified") is True and finite_nonzero(after.get("gradient_norm")):
            learning.append(step)
        per_rank_delta = []
        if delta_helper and before and after and step == resume + 1:
            old, new = ranks(before.get("observations")), ranks(after.get("observations"))
            if old and new:
                for rank in range(8):
                    left, right = old[rank].get("_observer", {}).get("parameters"), new[rank].get("_observer", {}).get("parameters")
                    proof = delta_helper.parameter_sample_delta([left], [right]) if left and right else None
                    per_rank_delta.append({"rank": rank, "proof": proof,
                                           "reason": None if proof and proof.get("observed_parameter_change") is not None else "same-coordinate local shard samples not observed",
                                           "source": "native_actor_update_dispatched/verified._observer.parameters"})
        per_rank_returns = []
        for row in (after or {}).get("observations", []):
            returns = row.get("_observer", {}).get("native_engine_step_returns")
            current = [item for item in returns or [] if item.get("trainer_step") == step]
            per_rank_returns.append({"rank": row.get("rank"), "returns": current or None,
                                     "reason": None if current else "actual native engine return norm not observed"})
        details.append({"step": step, "optimizer_commit_observed": step in verified,
                        "engine_and_durable_version_matched": step in publications, "original_learning_signal_observed": step in learning,
                        "audit_error": audit_error, "per_rank_parameter_delta": per_rank_delta or None,
                        "per_rank_parameter_delta_reason": None if per_rank_delta else "per-step same-coordinate parameter probe unavailable; not inferred from optimizer counters",
                        "per_rank_engine_returns": per_rank_returns or None,
                        "per_rank_engine_returns_reason": None if per_rank_returns else "native engine rank receipts unavailable"})
    initial = next((row for row in events if row["event"] == "original_trainer_initialized"), None)
    first_after = (indexed.get(("native_actor_update_verified", resume + 1)) or [None])[0]
    process_map = actual_processes((first_after or {}).get("observations"), known_hashes)
    first_detail = details[0] if details else {}
    rank_delta = first_detail.get("per_rank_parameter_delta")
    rank_change = bool(rank_delta and len(rank_delta) == 8 and all(row["proof"] and row["proof"].get("observed_parameter_change") is True for row in rank_delta))
    native_status = result.get("status") if isinstance(result, dict) else None
    terminal_failed = bool(native_status and any(word in native_status.upper() for word in FAILED_WORDS))
    terminal_pass = native_status == "PASS_NATIVE_UPDATES_AND_PUBLICATIONS"
    owner_first_pass = isinstance(first, dict) and first.get("status") == "PASS_REQUIRED_FIRST_UPDATE_OBSERVATIONS"
    requests = first.get("actual_update_request_sources", []) if isinstance(first, dict) else []
    native_request_bound = bool(requests and all(row.get("requested_version") == resume + 1 and
                               row.get("source", {}).get("sha256") in known_hashes for row in requests))
    first_returns = first_detail.get("per_rank_engine_returns") or []
    rank_return_signal = (sorted(row["rank"] for row in first_returns if type(row.get("rank")) is int) == list(range(8))
                          and len(first_returns) == 8 and all(len(row["returns"] or []) == 1 and
                          row["returns"][0].get("finite") is True and row["returns"][0].get("nonzero") is True for row in first_returns))
    first_verified = bool(runtime_allowed and not wrong_events and model_bound and helper and delta_helper and process_map
                          and owner_first_pass and native_request_bound and rank_change and resume + 1 in verified and resume + 1 in publications
                          and rank_return_signal and resume + 1 in learning and not terminal_failed)
    phase_complete = bool(first_verified and terminal_pass and verified == expected_steps and publications == expected_steps
                          and result.get("runtime_source_archive_sha256") == expected_source)
    runtime_receipts_bound = bool(runtime_allowed and not wrong_events and model_bound and process_map and helper)

    collector_records = []
    for ref in phase.get("collector_events", []):
        collector_records.extend(evidence.read(ref, stage + ".collector_events", lines=True) or [])
    raw_profile_records = []
    for reference in phase.get("raw_profile_logs", []):
        payload = evidence.read(reference, stage + ".raw_profile_logs", raw=True)
        for line in (payload or b"").decode(errors="replace").splitlines():
            if collect.PROFILE_MARKER in line:
                try:
                    raw_profile_records.append(json.JSONDecoder().raw_decode(line.split(collect.PROFILE_MARKER, 1)[1])[0])
                except ValueError:
                    evidence.errors.append(stage + ": malformed original native profile log record")
    timing = collect.summarize(collector_records, run_id=phase.get("profile_run_id", stage), profile_module=profile_module,
                               result=result) if profile_module else None
    profiles = {row["update_id"]: row for row in (timing or {}).get("profiles", [])}
    if profile_module and raw_profile_records:
        for profile in profile_module.summarize_profile_records(raw_profile_records)["profiles"]:
            profiles.setdefault(profile["update_id"], {**profile, "stages": collect._profile_stages(profile),
                                                       "source": "original_process_profile_logs_bound_by_input_stage_artifact"})
    for row in events:
        if row["event"] == "native_weight_profile_snapshot" and isinstance(row.get("profile"), dict):
            profile = row["profile"]
            if profile["update_id"] not in profiles:
                profile = {**profile, "total_s": profile.get("durations_s", {}).get("total"), "open_phases": [], "event_count": None}
                profiles[profile["update_id"]] = {**profile, "stages": collect._profile_stages(profile),
                                                   "source": "runtime_observer.native_weight_profile_snapshot.profile"}
    spans = []
    begins = {row["span_id"]: row for row in events if row["event"] == "stage_begin"}
    for row in events:
        if row["event"] not in ("stage_end", "stage_exception") or not row.get("stage", "").startswith("native_publication."):
            continue
        start = begins.get(row.get("span_id"))
        elapsed = row.get("elapsed_ns")
        same_process = start and all(start.get(key) == row.get(key) for key in ("pid", "socket_hostname", "stage_id", "source_sha"))
        spans.append({"stage": row["stage"], "span_id": row["span_id"], "status": row["event"],
                      "elapsed_s": elapsed / 1e9 if same_process and type(elapsed) is int and elapsed >= 0 else None,
                      "reason": None if same_process and type(elapsed) is int and elapsed >= 0 else "original process-local completed span unavailable",
                      "requested_version": start.get("requested_version") if start else None,
                      "scope": "original_observer_inclusive_call_elapsed_not_wire_or_serving_pause"})
    segment_phases = {"parameter_generator_prepare": "parameter_export_prepare", "parameter_materialization": "parameter_export_next",
                      "sender_transfer_including_export": "sender_transfer_export_mixed", "transfer_and_apply_mixed": "transfer_apply",
                      "engine_application_RPC": "engine_apply_rpc_inclusive", "cache_invalidation_RPC": "cache_invalidation",
                      "readmission": "gateway.start_admission", "stop_call_to_readmission_return": "admission_pause"}
    components = {}
    for label, native_phase in segment_phases.items():
        measured = [{"update_id": profile["update_id"], "global_steps": profile.get("global_steps"),
                     "hostname": profile["hostname"], "pid": profile["pid"], "operation": profile["operation"],
                     "elapsed_s": profile["durations_s"][native_phase], "native_profile_status": profile["status"],
                     "scope": "original_profile_process_local_host_elapsed_nested_and_pipeline_times_overlap"}
                    for profile in profiles.values() if native_phase in profile.get("durations_s", {})]
        components[label] = observation(measured or None, "actual native " + native_phase + " boundary not observed",
                                        source="original_WeightUpdateProfile", scope="per_update_per_process_no_cross_process_sum")
    completed_pauses = [row for row in components["stop_call_to_readmission_return"]["value"] or []
                        if row["native_profile_status"] == "completed"]
    remote_acks = [row for row in raw_profile_records + [record["fields"]["profile"] for record in collector_records
                    if record.get("event") == collect.PROFILE_EVENT] if row.get("phase") == "engine_apply_remote_ack"]
    first_before = (indexed.get(("native_actor_update_dispatched", resume + 1)) or [{}])[0]
    restored_parameters = [row.get("_observer", {}).get("parameters") for row in first_before.get("observations", [])]
    restored_parameters = restored_parameters if len(restored_parameters) == 8 and all(restored_parameters) else None
    return {"stage_id": stage, "runtime_source_sha256": expected_source, "status": "FAILED_NATIVE_RUN_PARTIAL_EVIDENCE" if terminal_failed else
            "OBSERVED_NATIVE_PHASE_UPDATES" if phase_complete else "PENDING_RUNTIME_EVIDENCE",
            "native_result_status": observation(native_status, "actual native terminal result not collected", source=stage + ".result"),
            "frontiers": {"resume_step": resume, "target_step": target},
            "actual_model_identity": check(model_bound if freeze else None, "actual Qwen3.5 business freeze not collected or mismatched", source=stage + ".freeze"),
            "first_update_chain": check(first_verified if events and first else None, "original group/source, all-rank native returns/optimizer, each-rank delta, model/source identity and engine publication must all agree"),
            "successful_native_optimizer_updates": observation(len(verified) if runtime_receipts_bound else None, "original counter receipts are not yet bound to actual Qwen3.5 model/source/process/APPID allocation", source=stage + ".events",
                                                                  scope="successful all-rank optimizer call/counter receipts; overall phase may still fail"),
            "learning_signal_steps": observation(learning if runtime_receipts_bound else None, "original learning receipts are not yet bound to actual runtime identity", source=stage + ".events"),
            "verified_publication_steps": observation(publications if runtime_receipts_bound else None, "engine/policy receipts are not yet bound to actual runtime identity", source=stage + ".events"),
            "raw_audit_receipt_steps": {"optimizer": verified, "learning": learning, "publication": publications,
                                         "scope": "raw native audit function results; not runtime-qualified until identity joins succeed"},
            "each_rank_parameter_change": check(rank_change if rank_delta else None, "every train rank needs nonzero same-coordinate shard sample delta"),
            "all_rank_native_finite_nonzero_returns": check(rank_return_signal if first_returns else None, "every original train engine must expose its own finite nonzero return"),
            "actual_native_update_request_source": check(native_request_bound if requests else None, "metadata version increments require the actual frozen native update call receipt"),
            "actual_train_rank_process_mapping": observation(process_map, "bound source/Ray actor/node/physical CUDA UUID mapping not collected", source=stage + ".events"),
            "engine_version_before": observation((initial or {}).get("publication", {}).get("actual_parameter_version"), "initial original engine version not observed"),
            "engine_version_after_first_update": observation((indexed.get(("native_policy_publication_verified", resume + 1)) or [{}])[0].get("actual_parameter_version"), "actual post-update engine readback not observed"),
            "post_update_generation_response_version": observation(None, "engine model_info and metadata counters are not actual generation response metadata"),
            "owner_first_update_receipt": observation(first, "frozen runtime_observer first-update.json not collected", source=stage + ".first_update"),
            "step_details": details, "weight_costs": {"profiles": list(profiles.values()), "native_publication_call_spans": spans, "components": components,
                "wire_only_s": observation(None, "NCCL export, pipelined transfer and engine application do not have a wire-only boundary"),
                "engine_apply_only_s": observation(None, "original application RPC includes IPC and remote work; no isolated engine kernel clock"),
                "complete_admission_pause_s": observation(completed_pauses or None, "no completed stop-call to native readmission-return phase observed",
                                                           source="original_WeightUpdateProfile.admission_pause", scope="individual process-local intervals; failure intervals are not completed pause"),
                "engine_application_remote_ACKs": observation(remote_acks or None, "original engine HTTP ACK receipts not observed",
                                                               source="original_WeightUpdateProfile.engine_apply_remote_ack"),
                "nested_times_overlap": True, "cross_process_times_summed": False},
            "final_parameter_receipt": observation(final_parameters, "actual final local shard samples not collected", source=stage + ".parameters_final"),
            "first_dispatch_parameter_receipt": observation(restored_parameters, "actual original local shards before first dispatched update not collected", source=stage + ".events"),
            "initialization_receipt": observation(initial, "actual initialized native trainer receipt not collected"),
            "native_result": result, "phase_complete": phase_complete}


def cold_resume(phases, delta_helper):
    left = next((phase for phase in phases if phase["frontiers"] == {"resume_step": 0, "target_step": 10}), None)
    right = next((phase for phase in phases if phase["frontiers"] == {"resume_step": 10, "target_step": 100}), None)
    if not left or not right or not delta_helper:
        return {"status": "PENDING", "reason": "phase10, phase100 and frozen native parameter helper are all required", "full_restore_equivalence_verified": False}
    initial = right["initialization_receipt"]["value"] or {}
    initial_rows = ranks(initial.get("observations"))
    old_rows = left["final_parameter_receipt"]["value"]
    restored = right["first_dispatch_parameter_receipt"]["value"]
    resume_events = right.get("_resume_events", [])
    proof = next((row.get("parameter_samples") for row in resume_events if row["event"] == "native_cold_resume_samples_verified"), None)
    new_processes = right["actual_train_rank_process_mapping"]["value"]
    old_processes = left["actual_train_rank_process_mapping"]["value"]
    new_process = bool(new_processes and old_processes and all(
        (old_processes[rank]["ray_actor_id"], old_processes[rank]["ray_node_id"], old_processes[rank]["pid"]) !=
        (new_processes[rank]["ray_actor_id"], new_processes[rank]["ray_node_id"], new_processes[rank]["pid"]) for rank in range(8)))
    frontier = bool(initial_rows and all(row.get("initial_optimizer_step_frontier") == 10 for row in initial_rows.values())
                    and restored and all(row.get("optimizer_step_max") == 10 for row in restored))
    compared = delta_helper.parameter_sample_delta(old_rows, restored) if old_rows and restored else None
    same_samples = bool(proof and proof.get("sample_linf") == 0.0 and compared and compared.get("sample_linf") == 0.0)
    complete = bool(left["phase_complete"] and right["phase_complete"] and frontier and new_process and same_samples)
    return {"status": "OBSERVED_NATIVE_COLD_RESUME_10_TO_100" if complete else "PENDING_OR_REJECTED",
            "new_rank_processes": check(new_process if new_processes and old_processes else None, "actual old/new Ray actor and process identities not joined"),
            "restored_adam_frontier_10": check(frontier if initial_rows else None, "all-rank original initial Adam frontier not observed"),
            "original_same_coordinate_resume_sample_proof": observation(proof, "original native cold-resume sample receipt not observed"),
            "independent_receipt_readback_sample_comparison": observation(compared, "phase10 final and phase100 initial local shard receipts not collected",
                                                                           source="frozen_native_parameter_sample_delta"),
            "phase100_new_process_updates": right["successful_native_optimizer_updates"],
            "full_restore_equivalence_verified": False,
            "scope": "original checkpoint/Adam frontier and local shard sample continuity; not uninterrupted numerical or full-weight equivalence"}


def optional_report(reference, name, evidence, stages):
    report = evidence.read(reference, name + ".runtime_report")
    if report is None:
        return {"status": "PENDING", "runtime_report": observation(None, "actual " + name + " runtime report not collected")}
    ref = reference if isinstance(reference, dict) else {}
    bound = ref.get("stage_id") in stages
    return {"status": "RUNTIME_REPORT_COLLECTED" if bound else "PENDING_RUN_IDENTITY_JOIN",
            "runtime_report": observation(report, source=name + ".runtime_report"),
            "reported_status": report.get("status"), "phase_binding": check(bound if ref.get("stage_id") else None,
                "optional report requires actual stage_id from this run; an old model report is not acceptance"),
            "business_benefit_accepted": report.get("business_benefit_accepted", report.get("routing_benefit")),
            "gpu_reclamation_or_hybrid_switching_inferred": False}


def build_summary(inputs, *, base):
    evidence = Evidence(base)
    identity = inputs.get("identity", {})
    runtime_allowed = inputs.get("evidence_kind") == "runtime" and identity.get("model_family") == QWEN_MODEL
    manifest, module, source_check = source_binding(identity, evidence)
    allocation = evidence.read(inputs.get("owned_allocation"), "owned_allocation")
    uuid_pattern = r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
    app_bound = bool(allocation and allocation.get("app_id") == identity.get("app_id") and identity.get("app_id")
                     and len(set(allocation.get("gpu_uuids", []))) == 16 and len(set(allocation.get("kernel_boot_ids", []))) >= 2
                     and all(re.fullmatch(uuid_pattern, value, re.I) for value in allocation.get("gpu_uuids", [])))
    identity_allowed = runtime_allowed and source_check["value"] is True and app_bound
    phases = []
    delta_helper = None
    for phase in inputs.get("phases", []):
        phase_identity, phase_manifest, phase_module, phase_source_check = identity, manifest, module, source_check
        if any(key in phase for key in ("source_archive", "source_manifest", "source_sha256")):
            phase_identity = {**identity, **{key: phase[key] for key in ("source_archive", "source_manifest", "source_sha256") if key in phase}}
            phase_manifest, phase_module, phase_source_check = source_binding(phase_identity, evidence)
            if phase_source_check["value"] is False:
                evidence.errors.append(phase["stage_id"] + ": actual phase archive differs from its declared final source identity")
        summary = phase_summary(phase, phase_identity, phase_manifest, phase_module, evidence,
                                identity_allowed and phase_source_check["value"] is True)
        summary["source_archive_binding"] = phase_source_check
        summary["_resume_events"] = evidence.read(phase.get("events"), phase["stage_id"] + ".resume_events", lines=True) or []
        phases.append(summary)
        mapping = summary["actual_train_rank_process_mapping"]["value"]
        if mapping and allocation and not {row["gpu_uuid"] for row in mapping} <= set(allocation.get("gpu_uuids", [])):
            evidence.errors.append(phase["stage_id"] + ": train rank GPUs are outside actual owned APPID allocation")
        if delta_helper is None:
            freeze = evidence.read(phase.get("freeze"), phase["stage_id"] + ".resume_freeze") or {}
            delta_helper = native_helper(phase, freeze, evidence, role="parameter_observer", names=("parameter_sample_delta",),
                                        default=HERE.parents[1] / "production-entry-20261008/run_v1_functional_gate.py")
    resume = cold_resume(phases, delta_helper)
    for phase in phases:
        phase.pop("_resume_events", None)
    verified_stages = {phase["stage_id"] for phase in phases if phase["first_update_chain"]["value"] is True}
    storage = optional_report(inputs.get("storage_report"), "storage", evidence, verified_stages)
    routing = optional_report(inputs.get("routing_report"), "routing", evidence, verified_stages)
    failed = any(phase["status"] == "FAILED_NATIVE_RUN_PARTIAL_EVIDENCE" for phase in phases)
    accepted_training = bool(identity_allowed and phases and all(phase["phase_complete"] for phase in phases)
                             and resume["status"] == "OBSERVED_NATIVE_COLD_RESUME_10_TO_100" and not evidence.errors)
    return {"schema_version": 1, "status": "NOT_RUNTIME_EVIDENCE" if not runtime_allowed else
            "REJECTED_INPUT_IDENTITY" if evidence.errors else "FAILED_RUNTIME_PARTIAL_EVIDENCE" if failed else
            "OBSERVED_TRAINING_AND_COLD_RESUME" if accepted_training else "PENDING_RUNTIME_EVIDENCE",
            "identity": {**identity, "source_archive_binding": source_check,
                         "caller_identity_scope": "expected bindings; actual model, source, APPID and process checks are reported separately",
                         "APPID_owned_resource_binding": check(app_bound if allocation else None, "owned actual APPID/16-UUID/two-kernel receipt not collected or mismatched"),
                         "owned_allocation": observation(allocation, "actual allocation identity unavailable", source="owned_allocation"),
                         "actual_rollout_rank_process_mapping": observation(None, "existing train-rank observer does not expose rollout rank/node/UUID role mapping"),
                         "model_scope": "Qwen3.5 only; historical Qwen3/1B/CPU receipts cannot satisfy these gates"},
            "training_weights": {"status": "OBSERVED_TRAINING_AND_COLD_RESUME" if accepted_training else "PENDING_OR_FAILED",
                                 "phases": phases, "cold_resume_10_to_100": resume},
            "storage_work_continuation": storage, "KV_routing": routing,
            "total_actual_gpu_hours": observation(None, "actual role UUID allocation lifetimes including failed attempts are not joined; configured counts and publication spans are not GPU-hours"),
            "project_production_acceptance": False,
            "project_acceptance_reason": "training, storage/routing function and attributable task benefit have independent evidence requirements",
            "input_errors": evidence.errors, "sources": evidence.sources,
            "unaccepted_partial_receipts": [observation(evidence.read(reference, "unaccepted_partial_receipt"),
                "historical partial receipt not collected", source="unaccepted_partial_receipt", scope="not a Qwen3.5 optimizer/update or recovery receipt")
                for reference in inputs.get("unaccepted_partial_receipts", [])],
            "execution": {"new_observations": 0, "training_or_service_replayed": False, "GPU_jobs_submitted": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-training-acceptance", action="store_true")
    args = parser.parse_args()
    summary = build_summary(json.loads(args.inputs.read_text()), base=args.inputs.resolve().parent)
    summary["input_descriptor_sha256"] = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        stream.write(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "project_production_acceptance": False, "output": str(args.output)}))
    if args.require_training_acceptance and summary["training_weights"]["status"] != "OBSERVED_TRAINING_AND_COLD_RESUME":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
