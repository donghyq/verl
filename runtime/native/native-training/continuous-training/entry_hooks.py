"""Opt-in glue for a future copy of the sealed native entry."""
import json
import os
from pathlib import Path
from routing_business_compare import Options, RoutingBusinessCompare


def record(report, output, value):
    """Benchmark diagnostics never replace a native result or native exception."""
    report["routing_business_compare"] = value
    try:
        path = Path(output) / "routing-business-benchmark-glue.json"
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    except BaseException:
        pass


def prepare_optional(trainer, read_publication, initial_observation, *, report, output):
    filename = os.environ.get("VERL_ROUTING_BUSINESS_COMPARE_OPTIONS")
    if not filename:
        return None
    try:
        # These are completed native observations already returned before this
        # hook. No polling/RPC/sleep to wait for a not-yet-started publication.
        profile = getattr(trainer.standalone_checkpoint_manager, "last_weight_update_profile", {})
        registrations = initial_observation.get("registrations", [])
        urls = {x if x.startswith(("http://", "https://")) else "http://" + x
                for x in trainer.standalone_server_manager.server_addresses}
        ready = (initial_observation.get("actual_parameter_version") == 0
            and initial_observation.get("policy_identity", {}).get("policy_version") == "0"
            and profile.get("operation") == "weight_update" and profile.get("global_steps") == 0
            and profile.get("status") == "completed" and profile.get("error_type") is None
            and "gateway.start_admission" in profile.get("durations_s", {})
            and {row.get("url") for row in registrations} == urls
            and len(registrations) == len(urls) == 2
            and all(row.get("routing_ready") is True and row.get("worker_id")
                    and str(row.get("engine_weight_version")) == "0" for row in registrations))
        if not ready:
            record(report, output, {"status": "NOT_RUN", "reason": "ACTUAL_V0_PUBLICATION_AND_COLD_HEALTH_NOT_ALREADY_OBSERVED",
                                    "waited_for_publication": False})
            return None
        with open(filename) as stream:
            options = Options(**json.load(stream))
        driver = RoutingBusinessCompare.prepare(trainer, options=options, read_publication=read_publication)
        record(report, output, {"status": "PREPARED_NOT_MEASURED", "actual_v0_initial_observation": initial_observation,
                               "completed_native_publication_profile": profile})
        return driver
    except BaseException as error:
        # RoutingBusinessCompare.prepare cleans only its CPU children before
        # rethrowing. Preserve the original trainer's opportunity to fit.
        record(report, output, {"status": "NOT_RUN", "stage": "benchmark_prepare",
                               "error_type": type(error).__name__, "error": str(error)})
        return None


def finish_optional(driver, *, report, output):
    try:
        witness = report.get("version1_token_witness", {})
        count = witness.get("output_token_count")
        if (witness.get("status") != "PASS_NATIVE_VERSION1_TOKEN"
                or witness.get("actual_engine_weight_version") != "1"
                or witness.get("client_generate_calls") != 1
                or type(count) is not int or not 1 <= count <= 4):
            result = {"status": "NOT_RUN", "reason": "MANDATORY_ACTUAL_V1_TOKEN_WITNESS_NOT_PASSED"}
        else:
            filename = os.environ.get("VERL_ROUTING_NATIVE_RESET_RECEIPTS")
            receipts = {}
            if filename:
                with open(filename) as stream:
                    receipts = json.load(stream)
            result = driver.after_publication(cache_reset_receipts=receipts)
        record(report, output, result)
        return result
    except BaseException as error:
        result = {"status": "NOT_ACCEPTED", "stage": "benchmark_finish",
                  "error_type": type(error).__name__, "error": str(error), "automatic_retries": 0}
        record(report, output, result)
        return result


def close_optional(driver, *, report, output):
    if driver is None:
        return
    try:
        driver.close()
        errors = getattr(driver, "cleanup_errors", [])
    except BaseException as error:
        errors = [{"error_type": type(error).__name__, "error": str(error)}]
    if errors:
        value = dict(report.get("routing_business_compare", {}))
        value["benchmark_cleanup_errors"] = errors
        record(report, output, value)
