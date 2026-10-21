"""Capture existing native weight profiles and audits without timing them again.

Install in the native trainer process before ``trainer.init()`` and mirror its
existing ``event`` callback with ``collector.event(name, **fields)``. Nothing in
this module changes training, reads tensors, calls services, or fences CUDA.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
import socket
import threading
from pathlib import Path


PROFILE_MARKER = "WEIGHT_UPDATE_PROFILE "
PROFILE_EVENT = "native_weight_update_profile"
class WeightMetricsCollector(logging.Handler):
    """Write original receipts per process; never alter optimizer outcomes.

    Logging errors are retained in ``errors`` and do not escape into training.
    The output file is exclusive: a later process cannot append to old evidence.
    ``recorded_at_ns`` from the original profile is kept for diagnosis only; all
    reported durations come from its original process-local perf_counter.
    """

    def __init__(self, output, *, run_id, profile_module=None):
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("An actual native run_id is required")
        self.run_id = run_id
        self.hostname, self.pid = socket.gethostname(), os.getpid()
        self.errors = []
        self._write_lock = threading.Lock()
        self._sequence = 0
        self._profile_module = profile_module
        self._profile_logger = None
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.path = output / f"weight_metrics_{self.hostname}_{self.pid}.jsonl"
        self._stream = self.path.open("x", encoding="utf-8")
        super().__init__(level=logging.INFO)

    def install(self):
        if self._profile_logger is None:
            try:
                module = self._profile_module or importlib.import_module("verl.checkpoint_engine.weight_update_profile")
            except ImportError as error:
                self.errors.append({"event": "weight_metrics_profile_unavailable", "error_type": type(error).__name__})
                self.event("weight_metrics_profile_unavailable", reason="native_runtime_profile_module_missing")
                return self
            self._profile_logger = module.logger
            self._profile_logger.addHandler(self)
            self.event(
                "weight_metrics_collector_installed",
                profile_module=str(Path(module.__file__).resolve()),
                profile_module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                timing_source="existing_WeightUpdateProfile_perf_counter",
                profile_logger_effective_level=self._profile_logger.getEffectiveLevel(),
                profile_info_logging_enabled=self._profile_logger.isEnabledFor(logging.INFO),
            )
        return self

    def event(self, name, **fields):
        """Mirror the existing native event after its original durable write."""
        try:
            with self._write_lock:
                self._sequence += 1
                record = {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "collector_hostname": self.hostname,
                    "collector_pid": self.pid,
                    "collector_sequence": self._sequence,
                    "event": name,
                    "fields": fields,
                }
                self._stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
                self._stream.flush()
        except Exception as error:
            # A failed evidence write cannot repeat or invalidate a real update.
            self.errors.append({"event": name, "error_type": type(error).__name__})

    def emit(self, record):
        try:
            message = record.getMessage()
            if message.startswith(PROFILE_MARKER):
                self.event(PROFILE_EVENT, profile=json.loads(message[len(PROFILE_MARKER):]))
        except Exception as error:
            self.errors.append({"event": PROFILE_EVENT, "error_type": type(error).__name__})

    def close(self):
        if self._profile_logger is not None:
            self._profile_logger.removeHandler(self)
            self._profile_logger = None
        with self._write_lock:
            if not self._stream.closed:
                self._stream.close()
        super().close()


def load_profile_module(path):
    """Read the actual frozen native profile module, without importing torch."""
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("weight_metrics_native_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_records(paths):
    records, sources = [], []
    for path in paths:
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{number}")
                records.append(row)
    return records, sources


def read_native_records(paths, *, run_id, stage_id, source_sha):
    """Bind the owner's real stage events without changing their original data.

    A native file has no collector run_id, so both actual stage_id and final
    source_sha must match the caller's frozen phase. Its wall/monotonic stamps
    remain metadata and are never used to derive weight publication durations.
    """
    records, sources = [], []
    for path in paths:
        native, source = read_records([path])
        sources.extend(source)
        for sequence, row in enumerate(native, 1):
            if row.get("stage_id") != stage_id or row.get("source_sha") != source_sha:
                raise ValueError("Native observation stage_id or source_sha differs from frozen phase")
            if row.get("run_id") not in (None, run_id):
                raise ValueError("Native observation run_id differs")
            if not isinstance(row.get("event"), str) or type(row.get("pid")) is not int or not row.get("socket_hostname"):
                raise ValueError("Native observation lacks actual event/process identity")
            records.append({
                "schema_version": 1, "run_id": run_id,
                "collector_hostname": row["socket_hostname"], "collector_pid": row["pid"],
                "collector_sequence": sequence,
                "receipt_stream": "owner_stage_events:" + stage_id + ":" + source[0]["sha256"],
                "event": row["event"], "fields": {key: value for key, value in row.items() if key != "event"},
            })
    return records, sources


def _measure(profile, phase, missing_reason, *, scope="inclusive_host_elapsed"):
    value = profile["durations_s"].get(phase)
    is_open = phase in profile["open_phases"]
    return {
        "value_s": None if is_open else value,
        "reason": "phase_started_without_matching_completion" if is_open else (missing_reason if value is None else None),
        "source_phase": phase if value is not None or is_open else None,
        "scope": scope,
        "clock": "original_process_perf_counter",
    }


def _profile_stages(profile):
    stages = {
        "parameter_export_prepare": _measure(profile, "parameter_export_prepare", "parameter_export_has_no_separate_observed_boundary"),
        "communication_transfer": _measure(profile, "wire_transfer", "native_transfer_apply_overlaps_parameter_export_and_engine_application"),
        "engine_application": _measure(profile, "engine_apply", "engine_application_has_no_separate_observed_boundary"),
        "cache_invalidation": _measure(profile, "cache_invalidation", "release_kv_is_resource_release_not_an_observed_flush_cache_duration"),
        "readmission": _measure(profile, "gateway.start_admission", "native_gateway_readmission_phase_not_observed"),
        "total_admission_pause": _measure(profile, "admission_pause", "complete_stop_to_readmission_boundary_not_observed"),
        "transfer_apply_mixed": _measure(profile, "transfer_apply", "native_transfer_apply_phase_not_observed"),
        "publication_total": _measure(profile, "total", "publication_did_not_return"),
    }
    # An interrupted pause is only an observed lower bound. It never means that
    # the engines have resumed serving when the failure context exits.
    if profile["status"] != "completed":
        pause = stages["total_admission_pause"]
        if pause["value_s"] is not None:
            pause.update(observed_until_failure_s=pause["value_s"], value_s=None,
                         reason="publication_failed_readmission_not_confirmed")
    return stages


def summarize(records, *, run_id, profile_module, result=None):
    """Reuse native profile parsing; preserve the original audit as evidence.

    This is a receipt parser, not a replacement for the original training gate.
    It does not infer a parameter delta or request version from a publication
    counter. Any new owner audit fields are copied intact into step evidence.
    """
    seen, unique = {}, []
    for row in records:
        if row.get("schema_version") != 1 or row.get("run_id") != run_id:
            raise ValueError("Mixed runs or unsupported collector schema")
        identity = (row.get("receipt_stream", "collector"), row["collector_hostname"],
                    row["collector_pid"], row["collector_sequence"])
        if identity in seen:
            if seen[identity] != row:
                raise ValueError("Conflicting copies of a collector receipt")
            continue
        seen[identity] = row
        unique.append(row)
    raw = [row["fields"]["profile"] for row in unique if row["event"] == PROFILE_EVENT]
    # Reject cross-process update_id reuse before native summarization. Never
    # subtract recorded_at_ns between processes or sum their inclusive times.
    identities = {}
    for row in raw:
        process = (row["hostname"], row["pid"], row["operation"], row["global_steps"])
        previous = identities.setdefault(row["update_id"], process)
        if previous != process:
            raise ValueError("A native profile update_id has conflicting process identities")
    native = profile_module.summarize_profile_records(raw)
    # A snapshot is the original manager's completed process-local timer data,
    # useful even when INFO logs were unavailable. Do not manufacture phase
    # records or derive durations from the owner's event timestamps.
    native_profiles = {p["update_id"]: p for p in native["profiles"]}
    for row in unique:
        snapshot = row["fields"].get("checkpoint_profile")
        if row["event"] != "native_policy_publication_verified" or snapshot is None:
            continue
        if not isinstance(snapshot, dict) or snapshot.get("status") != "completed":
            raise ValueError("Publication profile snapshot is not a completed native profile")
        if snapshot.get("global_steps") != row["fields"].get("trainer_step"):
            raise ValueError("Publication snapshot differs from actual native trainer step")
        durations = snapshot["durations_s"]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in durations.values()):
            raise ValueError("Invalid original snapshot duration")
        previous = native_profiles.get(snapshot["update_id"])
        if previous is not None:
            for key in ("hostname", "pid", "global_steps", "status", "durations_s"):
                if previous[key] != snapshot[key]:
                    raise ValueError("Native snapshot conflicts with original profile logs")
        else:
            native_profiles[snapshot["update_id"]] = {
                **snapshot, "total_s": durations.get("total"), "open_phases": [],
                "event_count": None, "evidence_form": "original_manager_snapshot",
            }
    profiles = []
    for profile in native_profiles.values():
        profiles.append({**profile, "stages": _profile_stages(profile),
                         "total_s_reason": None if profile["total_s"] is not None else "original_publication_timer_did_not_finish",
                         "initialization_only": profile["global_steps"] == 0})

    steps = {}
    initialized = []
    unassigned = []
    for row in unique:
        fields = row["fields"]
        if row["event"] == "original_trainer_initialized":
            initialized.append(fields)
        step = fields.get("trainer_step", fields.get("global_steps", fields.get("step")))
        if type(step) is not int or step <= 0:
            if row["event"] not in (PROFILE_EVENT, "weight_metrics_collector_installed"):
                unassigned.append({"event": row["event"], "fields": fields})
            continue
        entry = steps.setdefault(step, {"trainer_step": step, "native_events": []})
        entry["native_events"].append({"event": row["event"], "fields": fields,
                                       "collector_process": [row["collector_hostname"], row["collector_pid"]],
                                       "collector_sequence": row["collector_sequence"]})
    for step, entry in steps.items():
        entry["publication_profile_ids"] = [p["update_id"] for p in profiles
                                             if p["operation"] == "weight_update" and p["global_steps"] == step]
        by_name = {row["event"]: row["fields"] for row in entry["native_events"]}
        learning = by_name.get("native_learning_signal_observed")
        update = by_name.get("native_actor_update_verified")
        publication = by_name.get("native_policy_publication_verified")
        group_events = [row for row in entry["native_events"] if row["event"] in
                        ("group_dispatch_observed", "training_group_admission_observed")]
        delta_event = by_name.get("first_update_parameter_delta")
        delta = delta_event.get("parameter_sample_delta") if delta_event else None
        norm = update.get("gradient_norm") if update else None
        entry["observed"] = {
            "complete_prompt_groups": learning.get("complete_prompt_groups") if learning else None,
            "groups": learning.get("groups") if learning else None,
            "batch_keys_sha256": learning.get("batch_keys_sha256") if learning else None,
            "gradient_norm": norm,
            "finite_nonzero_gradient": (math.isfinite(norm) and norm > 0) if type(norm) in (float, int) else None,
            "participating_train_ranks": [x["rank"] for x in update.get("observations", [])] if update else None,
            "native_optimizer_proof": update,
            "actual_parameter_version": publication.get("actual_parameter_version") if publication else None,
            "active_policy_record": publication.get("active_policy_record") if publication else None,
            "group_source": group_events or None,
            "group_source_reason": None if group_events else "owner_group_source_audit_not_observed_at_this_step",
            "same_coordinate_parameter_delta": delta,
            "same_coordinate_parameter_delta_reason": None if delta is not None else "owner_per_step_parameter_sample_delta_not_observed",
            "post_update_generation_weight_version": None,
            "post_update_generation_weight_version_reason": "model_info_and_policy_version_counters_are_not_generation_response_metadata",
            "per_rank_gradient_norms": None,
            "per_rank_gradient_norms_reason": "legacy_original_audit_exposes_aggregate_gradient_norm_only",
        }
        reasons = {
            "complete_prompt_groups": "native_learning_signal_event_not_observed",
            "groups": "native_learning_signal_event_has_no_group_receipts",
            "batch_keys_sha256": "native_learning_signal_event_has_no_batch_digest",
            "gradient_norm": "original_actor_gradient_metric_not_observed",
            "finite_nonzero_gradient": "original_actor_gradient_metric_not_observed",
            "participating_train_ranks": "native_optimizer_verified_event_not_observed",
            "native_optimizer_proof": "native_optimizer_verified_event_not_observed",
            "actual_parameter_version": "actual_engine_version_readback_not_observed",
            "active_policy_record": "actual_durable_policy_readback_not_observed",
        }
        for field, reason in reasons.items():
            entry["observed"][field + "_reason"] = reason if entry["observed"][field] is None else None
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "PARSED_OBSERVATIONS_ONLY",
        "native_run_status": result.get("status") if isinstance(result, dict) else None,
        "native_run_status_reason": None if isinstance(result, dict) and result.get("status") else "native_terminal_result_not_supplied",
        "project_production_acceptance": False,
        "first_update_chain_acceptance": None,
        "first_update_chain_acceptance_reason": "owner_must_join_complete_business_group_all_rank_optimizer_delta_publication_and_generation_receipts",
        "durations_never_derived_from_cross_process_wallclock": True,
        "nested_times_overlap": native["nested_times_overlap"],
        "wire_time_s": native["wire_time_s"],
        "wire_time_s_reason": "native_profile_has_no_independent_wire_only_boundary",
        "gpu_time_s": native["gpu_time_s"],
        "gpu_time_s_reason": "native_profile_measures_host_elapsed_without_CUDA_event_fences",
        "profiles": profiles,
        "weight_profile_observation_reason": None if profiles else "native_profile_logs_or_manager_snapshot_not_supplied",
        "steps": [steps[step] for step in sorted(steps)],
        "initialization_receipts": initialized,
        "unassigned_native_events": unassigned,
        "collector_event_count": len(unique),
        "timing_aggregation": "per_update_per_process_inclusive_no_cross_process_sum",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, action="append", default=[])
    parser.add_argument("--native-events", type=Path, action="append", default=[])
    parser.add_argument("--stage-id")
    parser.add_argument("--source-sha")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-module", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.events and not args.native_events:
        parser.error("At least one collector or native events file is required")
    if args.native_events and (not args.stage_id or not args.source_sha):
        parser.error("Native stage events require their actual --stage-id and frozen --source-sha")
    records, sources = read_records(args.events)
    if args.native_events:
        native, native_sources = read_native_records(args.native_events, run_id=args.run_id,
                                                     stage_id=args.stage_id, source_sha=args.source_sha)
        records.extend(native)
        sources.extend(native_sources)
    module = load_profile_module(args.profile_module)
    result = json.loads(args.result.read_text()) if args.result else None
    output = summarize(records, run_id=args.run_id, profile_module=module, result=result)
    output["sources"] = sources
    output["profile_parser_sha256"] = hashlib.sha256(args.profile_module.read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "profiles": len(output["profiles"]), "steps": len(output["steps"])}))


if __name__ == "__main__":
    main()
