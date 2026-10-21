"""Rollouter-side adapter wiring preemptible handoff into the production path.

The coordinator is owned by FullyAsyncRollouter but deliberately holds only
picklable references (the artifact-store client and frozen lease dataclasses),
so checkpoint/owner callbacks injected into ``non_tensor_batch`` columns can
cross the Ray worker boundary via cloudpickle without serializing the actor.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from verl.experimental.fully_async_policy.preemptible_handoff import HandoffManifest


@dataclass
class MemberLease:
    member_index: int
    trajectory_id: str
    lease: Any  # TrajectoryOwnerLease (frozen dataclass, picklable)
    snapshot_ref: str | None = None
    expected_generation: int | None = None


@dataclass
class ParkedSample:
    sample_id: str
    group_id: str
    group_size: int
    member_leases: dict[int, MemberLease]
    preempted_members: list[int]
    completed_members: list[int]
    pending_members: list[int]
    record_ref: str
    producer_generation: int
    policy_identity: dict[str, str]
    snapshot_ttl_seconds: float = 3600.0
    handoff_id: str = ""
    completed_record_refs: dict[int, str] = field(default_factory=dict)
    input_record_refs: dict[int, str] = field(default_factory=dict)


def _object_column(values: list[Any]) -> np.ndarray:
    column = np.empty(len(values), dtype=object)
    column[:] = values
    return column


def bind_handoff_owner_controls(
    full_batch: Any, client: Any, member: dict[str, Any], lease: Any,
    *, policy_identity: dict[str, str], environment_version: str,
    tool_version: str, snapshot: Any = None,
) -> Any:
    """Bind one original request row to its existing target ownership lease."""
    if len(full_batch) != 1:
        raise ValueError("handoff continuation requires one original member row")

    async def check_owner(lease=lease):
        await client.check_trajectory_owner(lease)

    values = {
        "__trajectory_check_owner__": check_owner,
        "__trajectory_id__": member["trajectory_id"],
        "__trajectory_group_id__": member["group_id"],
        "__trajectory_sample_index__": member["member_index"],
        "__trajectory_group_size__": member["group_size"],
        "__trajectory_execution_generation__": lease.generation,
        "__trajectory_environment_version__": environment_version,
        "__trajectory_tool_version__": tool_version,
        "__trajectory_policy_identity__": policy_identity,
    }
    if snapshot is not None:
        values["__trajectory_resume__"] = snapshot
    for key, value in values.items():
        full_batch.non_tensor_batch[key] = _object_column([value])
    return full_batch


async def open_sample_leases(
    client: Any,
    full_batch: Any,
    sample_id: str,
    *,
    owner_id: str,
    lease_seconds: float,
    environment_version: str,
    tool_version: str,
    policy_identity: dict[str, str],
    snapshot_ttl_seconds: float = 3600.0,
    active_member_indices: list[int] | None = None,
) -> dict[int, MemberLease]:
    """Open a registry lease per trajectory before generation starts.

    Injects the ``__trajectory_*`` control columns the AgentLoopWorker maps
    onto ToolAgentLoop kwargs.  The callbacks close over ``client`` and the
    frozen lease only (never over the rollouter actor).
    """
    group_size = len(full_batch.non_tensor_batch["raw_prompt"])
    active_members = set(range(group_size) if active_member_indices is None else active_member_indices)
    if active_member_indices is not None and (
        len(active_members) != len(active_member_indices)
        or any(type(i) is not int or i < 0 or i >= group_size for i in active_members)
    ):
        raise ValueError("initial members must be distinct indices within the complete group")
    checkpoints, owners, ids, groups, indexes, sizes = [], [], [], [], [], []
    generations, envs, tools, identities = [], [], [], []
    leases: dict[int, MemberLease] = {}
    for member_index in range(group_size):
        trajectory_id = f"{sample_id}#{member_index}"
        if member_index not in active_members:
            # An unstarted member has no executing owner or RUNNING record.
            leases[member_index] = MemberLease(member_index, trajectory_id, None)
            checkpoints.append(None)
            owners.append(None)
            ids.append(trajectory_id)
            groups.append(sample_id)
            indexes.append(member_index)
            sizes.append(group_size)
            generations.append(0)
            envs.append(environment_version)
            tools.append(tool_version)
            identities.append(policy_identity)
            continue
        lease = await client.open_trajectory(trajectory_id, owner_id, lease_seconds)
        await client.register_running_trajectory(trajectory_id)

        async def make_checkpoint(snapshot, trajectory_id=trajectory_id, lease=lease,
                                   snapshot_ttl_seconds=snapshot_ttl_seconds):
            state = await client.get_trajectory_preemption(trajectory_id)
            if state != "preempt_requested":
                return None  # no reclamation in flight: continue in place
            await client.mark_trajectory_safe_point(trajectory_id)
            snapshot_ref = await client.commit_trajectory_snapshot(
                lease, snapshot, snapshot_ttl_seconds)
            await client.mark_trajectory_snapshot_committed(trajectory_id, snapshot_ref)
            return snapshot_ref

        async def make_owner_check(lease=lease):
            await client.check_trajectory_owner(lease)

        checkpoints.append(make_checkpoint)
        owners.append(make_owner_check)
        ids.append(trajectory_id)
        groups.append(sample_id)
        indexes.append(member_index)
        sizes.append(group_size)
        generations.append(lease.generation)
        envs.append(environment_version)
        tools.append(tool_version)
        identities.append(policy_identity)
        leases[member_index] = MemberLease(member_index, trajectory_id, lease)
    full_batch.non_tensor_batch["__trajectory_checkpoint__"] = _object_column(checkpoints)
    full_batch.non_tensor_batch["__trajectory_check_owner__"] = _object_column(owners)
    full_batch.non_tensor_batch["__trajectory_id__"] = _object_column(ids)
    full_batch.non_tensor_batch["__trajectory_group_id__"] = _object_column(groups)
    full_batch.non_tensor_batch["__trajectory_sample_index__"] = np.array(indexes, dtype=np.int64)
    full_batch.non_tensor_batch["__trajectory_group_size__"] = np.array(sizes, dtype=np.int64)
    full_batch.non_tensor_batch["__trajectory_execution_generation__"] = np.array(generations, dtype=np.int64)
    full_batch.non_tensor_batch["__trajectory_environment_version__"] = _object_column(envs)
    full_batch.non_tensor_batch["__trajectory_tool_version__"] = _object_column(tools)
    full_batch.non_tensor_batch["__trajectory_policy_identity__"] = _object_column(identities)
    return leases


def read_preempted_rows(full_batch: Any) -> dict[int, dict[str, Any]]:
    """Map member_index -> preemption info for rows that yielded at a safe point."""
    extra = full_batch.non_tensor_batch
    preempted: dict[int, dict[str, Any]] = {}
    flags = extra.get("__preempted__")
    if flags is None:
        return preempted
    member_index = extra["rollout_sample_index"]
    rows = len(flags) if not hasattr(flags, "__len__") else len(flags)
    for row in range(rows):
        if bool(flags[row]):
            preempted[int(member_index[row])] = {
                "snapshot_ref": extra["__preemption_snapshot_ref__"][row],
                "trajectory_id": extra["__preemption_trajectory_id__"][row],
                "expected_generation": int(extra["__preemption_expected_generation__"][row]),
            }
    return preempted


async def finalize_sample(
    client: Any,
    preempted: dict[int, dict[str, Any]],
    leases: dict[int, MemberLease],
) -> None:
    """Release leases for members that finished normally on the source."""
    for member_index, info in preempted.items():
        member_lease = leases[member_index]
        member_lease.snapshot_ref = info["snapshot_ref"]
        member_lease.expected_generation = info["expected_generation"]
    for member_index, member_lease in leases.items():
        if member_index in preempted or member_lease.lease is None:
            continue  # ownership crosses the boundary via committed snapshot
        await client.finish_trajectory(member_lease.lease)
        await client.discard_trajectory_preemption(member_lease.trajectory_id)


async def build_and_commit_manifest(
    client: Any,
    parked: ParkedSample,
    *,
    environment_version: str,
    tool_version: str,
    completed_artifacts: tuple[str, ...] = (),
) -> HandoffManifest:
    checkpointed = tuple({
        "trajectory_id": parked.member_leases[index].trajectory_id,
        "group_id": parked.group_id,
        "member_index": index,
        "group_size": parked.group_size,
        "expected_generation": parked.member_leases[index].expected_generation,
        "snapshot_ref": parked.member_leases[index].snapshot_ref,
        **({"input_ref": parked.input_record_refs[index]}
           if index in parked.input_record_refs else {}),
    } for index in sorted(parked.preempted_members))
    completed = tuple({
        "trajectory_id": parked.member_leases[index].trajectory_id,
        "group_id": parked.group_id,
        "member_index": index,
        "group_size": parked.group_size,
        "record_ref": parked.completed_record_refs.get(index, parked.record_ref),
    } for index in parked.completed_members)
    pending = tuple({
        "trajectory_id": parked.member_leases[index].trajectory_id,
        "group_id": parked.group_id,
        "member_index": index,
        "group_size": parked.group_size,
        **({"record_ref": parked.input_record_refs[index]}
           if index in parked.input_record_refs else {}),
    } for index in parked.pending_members)
    handoff_id = parked.handoff_id or f"handoff-{parked.sample_id}"
    manifest = HandoffManifest(
        handoff_id=handoff_id,
        source_rollouter="fully_async_rollouter",
        source_generation=parked.producer_generation,
        policy_identity=parked.policy_identity,
        environment_version=environment_version,
        tool_version=tool_version,
        completed_artifacts=completed_artifacts,
        checkpointed_trajectories=checkpointed,
        completed_members=completed,
        pending_samples=pending,
    )
    await client.prepare_handoff_manifest(manifest)
    await client.commit_handoff_manifest(manifest.handoff_id)
    await client.record_handoff_source_release(manifest.handoff_id)
    # Source release: surrender the preempted members' write leases so the
    # target can take over (late source writes fail closed from here on).
    for index in parked.preempted_members:
        await client.release_trajectory_owner(parked.member_leases[index].lease)
        await client.mark_trajectory_handed_off(parked.member_leases[index].trajectory_id)
    return manifest
