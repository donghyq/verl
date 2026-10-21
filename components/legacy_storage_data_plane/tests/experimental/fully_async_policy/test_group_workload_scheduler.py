"""Scheduling invariants; synthetic timings are not GPU performance evidence."""

import importlib.util
import pathlib
import sys

import pytest


MODULE_PATH = pathlib.Path(__file__).parents[3] / "verl/experimental/fully_async_policy/group_workload_scheduler.py"
spec = importlib.util.spec_from_file_location("group_workload_scheduler", MODULE_PATH)
scheduler_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scheduler_module
spec.loader.exec_module(scheduler_module)

CalibrationObservation = scheduler_module.CalibrationObservation
GroupRequestContext = scheduler_module.GroupRequestContext
GroupWorkloadScheduler = scheduler_module.GroupWorkloadScheduler
WorkloadCalibration = scheduler_module.WorkloadCalibration


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


def context(index, workloads=("short", "short", "short", "long"), group_id="g"):
    return GroupRequestContext(f"{group_id}/{index}", group_id, index, len(workloads), workloads[index], 64, 128)


def calibration(durations=None, first=True, tool=True):
    durations = durations or {"short": 1.0, "long": 9.0}
    rows = []
    for replica in ("a", "b"):
        for workload, duration in durations.items():
            for sample in range(3):
                rows.append(
                    CalibrationObservation(
                        workload, replica, 64, 128, 32, duration,
                        0.1 if first else None, 0.0 if tool else None,
                        f"calibration/{replica}/{workload}/{sample}",
                    )
                )
    return WorkloadCalibration(rows)


def test_group_objective_accounts_for_pending_long_member():
    clock = Clock()
    scheduler = GroupWorkloadScheduler(calibration(), enabled=True, clock=clock)
    scheduler.register_group([context(index) for index in range(4)], ["a", "b"])
    decisions = [scheduler.acquire(f"g/{index}", {"a": 0, "b": 0}) for index in range(4)]
    loads = {"a": 0.0, "b": 0.0}
    for decision in decisions:
        loads[decision.chosen_replica] += 9 if decision.member_index == 3 else 1
        assert decision.estimated_group_completion_at == 109.0
    assert sorted(loads.values()) == [3.0, 9.0]
    # Request-count balancing would put two members on each replica, and its
    # long-member replica would take 10 seconds in this declared serial model.
    assert decisions[0].group_state["pending_members"] == 3


def test_default_off_preserves_first_acquire_and_sticky_owner():
    scheduler = GroupWorkloadScheduler(calibration())
    scheduler.register_group([context(index) for index in range(4)], ["a", "b"])
    first = scheduler.acquire("g/0", {"a": 2, "b": 1})
    repeated = scheduler.acquire("g/0", {"a": 0, "b": 9})
    assert first.chosen_replica == repeated.chosen_replica == "b"
    assert first.reason == "least_inflight_disabled"
    assert repeated.reason == "trajectory_sticky"
    assert first.estimated_group_completion_at is None


def test_least_inflight_baseline_can_release_trajectory_sticky():
    scheduler = GroupWorkloadScheduler(calibration(), sticky=False)
    scheduler.register_group([context(index) for index in range(4)], ["a", "b"])
    assert scheduler.acquire("g/0", {"a": 0, "b": 1}).chosen_replica == "a"
    assert scheduler.acquire("g/0", {"a": 2, "b": 0}).chosen_replica == "b"
    with pytest.raises(ValueError, match="live routing"):
        GroupWorkloadScheduler(calibration(), enabled=True, sticky=False)


def test_missing_replica_calibration_falls_back_entire_group():
    scheduler = GroupWorkloadScheduler(calibration(), enabled=True)
    scheduler.register_group([context(index) for index in range(4)], ["a", "unmeasured"])
    decision = scheduler.acquire("g/3", {"a": 3, "unmeasured": 0})
    assert decision.reason == "least_inflight_missing_independent_calibration"
    assert decision.estimated_completion_at is None
    assert all(candidate["remaining_generation_cost_s"] is None for candidate in decision.candidates)


def test_missing_timing_components_remain_null():
    estimate = calibration(first=False).estimate(context(0), "a")
    assert estimate.service_s == 1.0
    assert estimate.prefix_state_acquire_cost_s is None
    assert estimate.remaining_generation_cost_s is None
    assert estimate.unseparated_service_cost_s == 1.0


def test_completion_records_prediction_error_without_recalibration():
    clock = Clock()
    profiles = calibration()
    scheduler = GroupWorkloadScheduler(profiles, enabled=True, clock=clock)
    scheduler.register_group([context(index) for index in range(4)], ["a", "b"])
    decision = scheduler.acquire("g/3", {"a": 0, "b": 0})
    clock.value = 110.5
    state = scheduler.complete("g/3", generated_tokens=70, tool_time_s=0.05)
    trace = scheduler.decision_traces()[0]
    assert trace["actual_completion_at"] == 110.5
    assert trace["prediction_error_s"] == 110.5 - decision.estimated_completion_at
    assert state["completed_members"] == 1
    assert profiles.estimate(context(3), "a").predicted_generated_tokens == 32
    with pytest.raises(ValueError, match="not previously completed"):
        scheduler.complete("g/3", generated_tokens=70)


def test_real_unknown_replica_signals_are_not_filled_with_zero():
    scheduler = GroupWorkloadScheduler(calibration(), enabled=True)
    scheduler.register_group([context(index) for index in range(4)], ["a", "b"])
    decision = scheduler.acquire("g/0", {"a": 1, "b": 2})
    for candidate in decision.candidates:
        assert candidate["resource_pressure_penalty_s"] is None
        assert candidate["reclaim_risk_penalty_s"] is None
        assert candidate["queued_tokens"] is None
        assert candidate["calibration_source_ids"]
        assert candidate["running_requests"] in (1, 2)


@pytest.mark.parametrize("requests", [[context(0), context(0)], [context(0), context(1), context(2)]])
def test_complete_group_required_before_placement(requests):
    scheduler = GroupWorkloadScheduler(calibration(), enabled=True)
    with pytest.raises(ValueError, match="every member exactly once"):
        scheduler.register_group(requests, ["a", "b"])


def test_different_policy_filtering_remains_caller_responsibility():
    scheduler = GroupWorkloadScheduler(calibration(), enabled=True)
    # The caller has excluded replica b using the existing identity/readiness
    # authority. A group forecast cannot put it back into the candidate pool.
    scheduler.register_group([context(index) for index in range(4)], ["a"])
    assert scheduler.acquire("g/0", {"a": 0}).chosen_replica == "a"


def test_admitted_owner_failure_never_silently_migrates():
    scheduler = GroupWorkloadScheduler(calibration(), enabled=True)
    scheduler.register_group([context(index) for index in range(4)], ["a", "b"])
    owner = scheduler.acquire("g/0", {"a": 0, "b": 0}).chosen_replica
    other = "b" if owner == "a" else "a"
    with pytest.raises(RuntimeError, match="handoff path"):
        scheduler.acquire("g/0", {other: 0})


def test_finished_group_state_can_be_bounded():
    scheduler = GroupWorkloadScheduler(calibration())
    scheduler.register_group([context(index) for index in range(4)], ["a", "b"])
    with pytest.raises(ValueError, match="incomplete"):
        scheduler.forget_completed_group("g")
    for index in range(4):
        scheduler.acquire(f"g/{index}", {"a": 0, "b": 0})
        scheduler.complete(f"g/{index}", generated_tokens=1, success=index != 3)
    assert scheduler.group_state("g")["failed_members"] == 1
    scheduler.forget_completed_group("g")
    with pytest.raises(KeyError):
        scheduler.acquire("g/0", {"a": 0, "b": 0})


def test_larger_search_falls_back_with_explicit_budget():
    scheduler = GroupWorkloadScheduler(calibration(), enabled=True, max_assignments=4)
    scheduler.register_group([context(index) for index in range(4)], ["a", "b"])
    assert scheduler.acquire("g/0", {"a": 0, "b": 0}).reason == "least_inflight_planning_budget_exceeded"


@pytest.mark.parametrize("elapsed, first, tool", [(float("nan"), 0.1, 0.0), (1.0, 1.1, 0.0), (1.0, 0.7, 0.4)])
def test_invalid_calibration_rejected(elapsed, first, tool):
    row = CalibrationObservation("short", "a", 64, 128, 32, elapsed, first, tool, "source")
    with pytest.raises(ValueError):
        WorkloadCalibration([row])


def test_prompt_or_generation_budget_mismatch_does_not_guess():
    profiles = calibration()
    request = GroupRequestContext("x", "x", 0, 1, "short", 4096, 128)
    assert profiles.estimate(request, "a") is None
    request = GroupRequestContext("x", "x", 0, 1, "short", 64, 256)
    assert profiles.estimate(request, "a") is None


def test_trace_copy_does_not_modify_scheduler_state():
    scheduler = GroupWorkloadScheduler(calibration(), enabled=True)
    scheduler.register_group([context(index) for index in range(4)], ["a", "b"])
    scheduler.acquire("g/0", {"a": 0, "b": 0})
    scheduler.decision_traces()[0]["candidates"][0]["running_requests"] = 1000
    assert scheduler.decision_traces()[0]["candidates"][0]["running_requests"] == 0


def test_contextvar_scope_restores_after_error():
    assert scheduler_module.group_request_context.get() is None
    with pytest.raises(RuntimeError):
        with scheduler_module.bind_group_request_context(context(0)):
            assert scheduler_module.group_request_context.get().trajectory_id == "g/0"
            raise RuntimeError("loop failed")
    assert scheduler_module.group_request_context.get() is None


def test_duplicate_evidence_does_not_satisfy_calibration_sample_count():
    row = CalibrationObservation("short", "a", 64, 128, 32, 1.0, None, None, "one-source")
    with pytest.raises(ValueError, match="duplicate calibration"):
        WorkloadCalibration([row, row, row])


def test_failed_unknown_generated_work_stays_null():
    scheduler = GroupWorkloadScheduler(calibration())
    scheduler.register_group([context(index) for index in range(4)], ["a", "b"])
    scheduler.acquire("g/0", {"a": 0, "b": 0})
    scheduler.complete("g/0", generated_tokens=None, success=False)
    assert scheduler.decision_traces()[0]["actual_generated_tokens"] is None


def test_cancel_pending_member_releases_forecast_without_marking_it_started():
    scheduler = GroupWorkloadScheduler(calibration(), enabled=True)
    scheduler.register_group([context(index) for index in range(4)], ["a", "b"])
    state = scheduler.cancel("g/0")
    assert state["failed_members"] == 1
    assert state["pending_members"] == 3
    assert state["running_members"] == 0
    assert not scheduler.is_started("g/0")
    with pytest.raises(ValueError, match="failed trajectory"):
        scheduler.acquire("g/0", {"a": 0, "b": 0})
