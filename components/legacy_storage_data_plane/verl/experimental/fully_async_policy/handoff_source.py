"""Source-side preemptible execution over the real ToolAgentLoop.

The loop runs with a checkpoint callback wired to the trajectory snapshot
registry and an owner check wired to the cooperative preemption tracker:

- ``run_preemptible_trajectory`` opens the trajectory lease, registers RUNNING
  in the tracker, and drives ``run_until_checkpoint``.  When a preemption was
  requested, the first completed tool batch raises TrajectoryCheckpointYield:
  the snapshot commits through the registry and the tracker advances
  SAFE_POINT_REACHED -> SNAPSHOT_COMMITTED.  Without a preemption request the
  loop runs to completion (no checkpoints are cut).
- ``resume_preemptible_trajectory`` is the target-side continuation using the
  real ``ToolAgentLoop.resume_from_snapshot`` (token re-prefill; committed
  tool results never re-execute).
"""
from __future__ import annotations

import time
from typing import Any, Callable

from verl.experimental.fully_async_policy.preemptible_handoff import PreemptState


class PreemptionRequested(Exception):
    """Raised by the caller when a preemption request races the run loop."""


async def run_preemptible_trajectory(
    *,
    agent_loop,
    sampling_params: dict[str, Any],
    snapshot_registry,
    tracker,
    trajectory_id: str,
    owner_id: str,
    identity: dict[str, str],
    environment_version: str,
    tool_version: str,
    kwargs: dict[str, Any],
    lease_seconds: float = 600.0,
    snapshot_ttl_seconds: float = 3600.0,
    preempt_immediately: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run one trajectory; return {"status": "yielded", "snapshot_ref": ...}
    after a preemption checkpoint, or {"status": "completed", "output": ...}."""
    lease = snapshot_registry.open(trajectory_id, owner_id, lease_seconds)
    tracker.register_running(trajectory_id)
    if preempt_immediately:
        # The reclamation decision was already pending when this trajectory
        # was admitted; its first completed tool batch is the safe point.
        tracker.request_preemption(trajectory_id)

    async def check_owner():
        snapshot_registry._check(lease)

    uncommitted = {}  # in-memory continuation snapshot when no preemption yet

    async def commit_checkpoint(snapshot):
        # Only cut storage writes once a preemption was actually requested.
        if tracker.state(trajectory_id) is not PreemptState.PREEMPT_REQUESTED:
            uncommitted["snapshot"] = snapshot
            return None
        tracker.mark_safe_point(trajectory_id)  # a completed tool-batch boundary
        key = snapshot_registry.commit(lease, snapshot, snapshot_ttl_seconds)
        tracker.mark_snapshot_committed(trajectory_id, key)
        return key

    try:
        from verl.experimental.agent_loop.tool_agent_loop import TrajectoryCheckpointYield

        resume = None
        while True:
            run_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ("rollout_group_id", "rollout_sample_index", "rollout_group_size")}
            if resume is not None:
                run_kwargs["_trajectory_resume"] = resume
            try:
                output = await agent_loop.run_until_checkpoint(
                    sampling_params,
                    commit_snapshot=commit_checkpoint,
                    check_owner=check_owner,
                    trajectory_id=trajectory_id,
                    rollout_group_id=kwargs["rollout_group_id"],
                    rollout_sample_index=kwargs["rollout_sample_index"],
                    rollout_group_size=kwargs["rollout_group_size"],
                    execution_generation=lease.generation,
                    snapshot_environment_version=environment_version,
                    snapshot_tool_version=tool_version,
                    snapshot_policy_identity=identity,
                    **run_kwargs,
                )
            except TrajectoryCheckpointYield as yield_event:
                if yield_event.reference is None:
                    # No preemption at this boundary: continue in memory from
                    # the yielded snapshot without committing anything.
                    resume = uncommitted["snapshot"]
                    continue
                # Preemption reached a safe point: the snapshot is committed.
                return {"status": "yielded", "snapshot_ref": yield_event.reference,
                        "generation": lease.generation}
            # No preemption was requested through the whole run: completed.
            snapshot_registry.finish(lease)
            tracker.discard(trajectory_id)
            return {"status": "completed", "output": output}
    except BaseException:
        # SAFE_POINT_TIMEOUT / crash before commit: nothing is committed past
        # the registry's own guarantees; the pending path re-rolls the sample.
        try:
            tracker.mark_aborted(trajectory_id)
        except ValueError:
            pass
        raise


async def resume_preemptible_trajectory(
    *,
    agent_loop,
    snapshot_registry,
    tracker,
    manifest_member: dict[str, Any],
    environment_version: str,
    tool_version: str,
    owner_id: str = "target_rollouter",
    lease_seconds: float = 600.0,
    check_owner: Callable[[], Any] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Target-side resume of one checkpointed member via the real agent loop.

    Returns (finished_agent_loop_output, resume_metrics).  tool_replay_count is
    measured by the caller's tool fixture; the resume path structurally cannot
    re-execute committed tool batches (they are restored as results).
    """
    trajectory_id = manifest_member["trajectory_id"]
    lease = snapshot_registry.takeover(
        trajectory_id, manifest_member["expected_generation"], owner_id, lease_seconds)
    snapshot = snapshot_registry.load(lease)

    async def default_check_owner():
        snapshot_registry._check(lease)

    output = await agent_loop.resume_from_snapshot(
        snapshot,
        check_owner=check_owner or default_check_owner,
        execution_generation=lease.generation,
        environment_version=environment_version,
        tool_version=tool_version,
        **(extra_kwargs or {}),
    )
    tracker.mark_handed_off(trajectory_id)
    snapshot_registry.finish(lease)
    tracker.discard(trajectory_id)
    return output, {"generation": lease.generation, "snapshot_ref": manifest_member["snapshot_ref"]}
