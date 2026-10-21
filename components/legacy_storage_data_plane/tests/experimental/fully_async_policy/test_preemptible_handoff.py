import json
from dataclasses import asdict

import pytest

from verl.experimental.fully_async_policy.preemptible_handoff import (
    HandoffManifest,
    HandoffManifestRegistry,
    HandoffMemberRecordStore,
    HandoffState,
    PreemptState,
    TrajectoryPreemptionTracker,
)
from verl.experimental.fully_async_policy.rollout_artifact import CanonicalPolicyKVIdentity


def clock_gen():
    now = [0.0]

    def advance(step):
        now[0] += step

    def clock():
        return now[0]

    return clock, advance


def member(group, index, size=8, generation=0, snapshot_ref="inflight/snap-1", record_ref="handoff-member/r1"):
    return {
        "trajectory_id": f"{group}#{index}",
        "group_id": group,
        "member_index": index,
        "group_size": size,
        "expected_generation": generation,
        "snapshot_ref": snapshot_ref,
        "record_ref": record_ref,
    }


def manifest(n_completed=(0, 1), n_checkpointed=(2, 3, 4, 5), n_pending=(6, 7), group="group-1",
             source_generation=3, source_rollouter="fully_async_rollouter"):
    def strip(entry, keys):
        return {key: value for key, value in entry.items() if key not in keys}

    return HandoffManifest(
        handoff_id="handoff-1",
        source_rollouter=source_rollouter,
        source_generation=source_generation,
        policy_identity=asdict(CanonicalPolicyKVIdentity("3", "weights", "template", "kv")),
        environment_version="env-1",
        tool_version="tools-1",
        completed_artifacts=("artifact-earlier",),
        checkpointed_trajectories=tuple(
            {**strip(member(group, index), {"record_ref"}), "snapshot_ref": f"inflight/snap-{index}",
             "expected_generation": 3}
            for index in n_checkpointed
        ),
        completed_members=tuple(
            {**strip(member(group, index), {"expected_generation", "snapshot_ref"}),
             "record_ref": f"handoff-member/r{index}"}
            for index in n_completed
        ),
        pending_samples=tuple(
            {**strip(member(group, index), {"expected_generation", "snapshot_ref", "record_ref"})}
            for index in n_pending
        ),
    )


# ---------------------------------------------------------------- tracker ----

def test_preemption_state_machine_and_metrics():
    clock, advance = clock_gen()
    tracker = TrajectoryPreemptionTracker(clock=clock)
    tracker.register_running("task-1#0")
    assert tracker.state("task-1#0") is PreemptState.RUNNING
    with pytest.raises(ValueError, match="unknown trajectory"):
        tracker.state("missing")

    advance(10)
    tracker.request_preemption("task-1#0")
    assert tracker.state("task-1#0") is PreemptState.PREEMPT_REQUESTED
    tracker.request_preemption("task-1#0")  # idempotent
    with pytest.raises(ValueError, match="reached safe point"):
        tracker.mark_snapshot_committed("task-1#0", "inflight/snap-1")

    advance(2)
    tracker.mark_safe_point("task-1#0")
    assert tracker.state("task-1#0") is PreemptState.SAFE_POINT_REACHED
    with pytest.raises(ValueError, match="hand-off requires"):
        tracker.mark_handed_off("task-1#0")

    advance(1)
    tracker.mark_snapshot_committed("task-1#0", "inflight/snap-1")
    # A committed snapshot may no longer be discarded via the abort fallback.
    with pytest.raises(ValueError, match="aborting"):
        tracker.mark_aborted("task-1#0")
    advance(3)
    tracker.mark_handed_off("task-1#0")
    with pytest.raises(ValueError, match="terminal"):
        tracker.request_preemption("task-1#0")

    row = tracker.metrics()["trajectories"][0]
    assert row["state"] == "handed_off"
    assert row["safe_point_wait_ms"] == pytest.approx(2000.0)
    assert row["snapshot_commit_ms"] == pytest.approx(1000.0)
    assert row["worker_reclaim_latency_ms"] == pytest.approx(6000.0)


def test_safe_point_timeout_abort_fallback_keeps_reclaim_metric():
    clock, advance = clock_gen()
    tracker = TrajectoryPreemptionTracker(clock=clock)
    tracker.register_running("task-1#1")
    advance(5)
    tracker.request_preemption("task-1#1")
    advance(30)  # never reaches a tool boundary
    tracker.mark_aborted("task-1#1")
    row = tracker.metrics()["trajectories"][0]
    assert row["state"] == "aborted"
    assert row["safe_point_wait_ms"] is None
    assert row["worker_reclaim_latency_ms"] == pytest.approx(30000.0)
    # RUNNING trajectories without a preemption request produce no row.
    tracker.register_running("task-1#2")
    assert all(r["trajectory_id"] != "task-1#2" for r in tracker.metrics()["trajectories"])


# --------------------------------------------------------------- manifest ----

def test_manifest_accepts_n8_partition_two_four_two():
    # Requirement 7 shape: 2 completed / 4 preempted / 2 pending in one n=8 group.
    draft = manifest()
    draft.encode()  # validate + canonical JSON round trip
    tables = draft.member_tables()
    assert tables["group-1"]["completed"] == [0, 1]
    assert sorted(tables["group-1"]["checkpointed"]) == [2, 3, 4, 5]
    assert tables["group-1"]["pending"] == [6, 7]


@pytest.mark.parametrize("mutation", [
    "duplicate", "missing", "bad_snapshot_ref", "bad_expected_generation",
    "extra_field", "bad_member_index", "no_checkpointed",
])
def test_manifest_rejects_broken_group_partitions(mutation):
    draft = manifest()
    if mutation == "duplicate":
        broken = draft.__class__(**{**asdict(draft), "checkpointed_trajectories":
            draft.checkpointed_trajectories + (draft.checkpointed_trajectories[0],)})
    elif mutation == "missing":
        broken = draft.__class__(**{**asdict(draft), "pending_samples":
            draft.pending_samples[:1]})
    elif mutation == "bad_snapshot_ref":
        entries = list(draft.checkpointed_trajectories)
        entries[0] = {**entries[0], "snapshot_ref": ""}
        broken = draft.__class__(**{**asdict(draft), "checkpointed_trajectories": tuple(entries)})
    elif mutation == "bad_expected_generation":
        entries = list(draft.checkpointed_trajectories)
        entries[0] = {**entries[0], "expected_generation": -1}
        broken = draft.__class__(**{**asdict(draft), "checkpointed_trajectories": tuple(entries)})
    elif mutation == "extra_field":
        entries = list(draft.completed_members)
        entries[0] = {**entries[0], "unexpected": 1}
        broken = draft.__class__(**{**asdict(draft), "completed_members": tuple(entries)})
    elif mutation == "bad_member_index":
        entries = list(draft.pending_samples)
        entries[0] = {**entries[0], "member_index": 8}
        broken = draft.__class__(**{**asdict(draft), "pending_samples": tuple(entries)})
    else:
        broken = draft.__class__(**{**asdict(draft), "checkpointed_trajectories": ()})
    with pytest.raises(ValueError):
        broken.validate()


# -------------------------------------------------------------- registry ----

def test_registry_commit_requires_all_snapshots_and_current_generation():
    clock, advance = clock_gen()
    tracker = TrajectoryPreemptionTracker(clock=clock)
    generations = {"fully_async_rollouter": 3}
    registry = HandoffManifestRegistry(
        tracker, current_producer_generation=lambda producer: generations.get(producer), clock=clock)

    # Stale source generation fails closed at prepare time.
    with pytest.raises(ValueError, match="not current"):
        registry.prepare(manifest(source_generation=2))
    # Unknown source producer also fails closed.
    with pytest.raises(ValueError, match="not current"):
        registry.prepare(manifest(source_rollouter="ghost_rollouter"))

    draft = manifest()
    # Idempotent prepare of the identical draft.
    assert registry.prepare(draft).state is HandoffState.PREPARING
    assert registry.prepare(draft).manifest == draft
    # COMMITTED requires every listed trajectory to have committed a snapshot.
    for index in (2, 3):
        trajectory = f"group-1#{index}"
        tracker.register_running(trajectory)
        tracker.request_preemption(trajectory)
        advance(1)
        tracker.mark_safe_point(trajectory)
        tracker.mark_snapshot_committed(trajectory, f"inflight/snap-{index}")
    with pytest.raises(ValueError, match="no committed snapshot"):
        registry.commit("handoff-1")
    for index in (4, 5):
        trajectory = f"group-1#{index}"
        tracker.register_running(trajectory)
        tracker.request_preemption(trajectory)
        advance(1)
        tracker.mark_safe_point(trajectory)
        tracker.mark_snapshot_committed(trajectory, f"inflight/snap-{index}")
    record = registry.commit("handoff-1")
    assert record.state is HandoffState.COMMITTED
    with pytest.raises(ValueError, match="not one of"):
        registry.commit("handoff-1")

    advance(2)
    registry.record_source_release("handoff-1")
    metrics = registry.metrics()["handoffs"][0]
    # prepare -> COMMITTED -> source release spans all four snapshot commits
    # (4 x advance(1)) plus the extra advance(2): 6s in monotonic-clock units.
    assert metrics["worker_reclaim_latency_ms"] == pytest.approx(6000.0)

    # Re-entering RESTORING preserves progress; trajectory leases remain the
    # authority that rejects another live owner. FINISHED follows publication.
    registry.begin_restore("handoff-1")
    assert registry.begin_restore("handoff-1").state is HandoffState.RESTORING
    registry.finish("handoff-1", "artifact-1")
    assert registry.get("handoff-1").state is HandoffState.FINISHED


def test_registry_prepare_rejects_conflicting_duplicate_handoff():
    clock, _ = clock_gen()
    tracker = TrajectoryPreemptionTracker(clock=clock)
    registry = HandoffManifestRegistry(
        tracker, current_producer_generation=lambda producer: 3, clock=clock)
    registry.prepare(manifest())
    with pytest.raises(ValueError, match="already exists"):
        registry.prepare(manifest(n_completed=(), n_checkpointed=(0, 1, 2, 3), n_pending=(4, 5, 6, 7)))


def test_member_record_store_round_trip_and_validation():
    store = HandoffMemberRecordStore()
    reference = store.put(json.dumps({"member_index": 0, "response_ids": [1, 2]}).encode())
    assert json.loads(store.get(reference))["response_ids"] == [1, 2]
    with pytest.raises(ValueError, match="non-empty bytes"):
        store.put(b"")
    with pytest.raises(ValueError):
        store.put(b"not-json")
    with pytest.raises(ValueError, match="unknown member record"):
        store.get("handoff-member/missing")
