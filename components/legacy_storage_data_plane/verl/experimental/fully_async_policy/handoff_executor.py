"""Target-side restore_handoff orchestration for preemptible rollouts.

KV Restore is a hard OFF requirement on this path: snapshots are resumed by
token re-prefill only.  ``kv_ref`` stays in the snapshot schema and must
degrade safely (it is ignored here; ``trajectory_recovery`` provides the
``token_only_reprefill`` plan semantics).

The executor talks to the artifact store through the same client surface used
in production (``RayRolloutArtifactStoreClient``); unit tests substitute an
in-process fake.  Group reassembly must go through the Rollout Artifact data
plane: the trainer consumes once through claim/ack, never a bypass channel.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from verl.experimental.fully_async_policy.preemptible_handoff import (
    HandoffManifest,
    HandoffState,
    decode_handoff_member_record,
    encode_handoff_member_record,
)


class HandoffStoreClient(Protocol):
    """The subset of the store-actor client surface used by restore."""

    async def get_handoff_manifest(self, handoff_id: str) -> HandoffManifest: ...
    async def begin_handoff_restore(self, handoff_id: str) -> str: ...
    async def finish_handoff_manifest(self, handoff_id: str, artifact_id: str) -> str: ...
    async def acquire_handoff_trajectory(self, handoff_id: str, trajectory_id: str,
                                         owner_id: str, lease_seconds: float) -> Any: ...
    async def record_handoff_member_result(self, handoff_id: str, member_index: int,
                                           owner_lease: Any, payload: bytes, metrics: dict) -> str: ...
    async def get_handoff_restored_members(self, handoff_id: str) -> dict: ...
    async def load_trajectory_snapshot(self, owner_lease: Any) -> Any: ...
    async def finish_trajectory(self, owner_lease: Any) -> None: ...
    async def mark_trajectory_handed_off(self, trajectory_id: str) -> str: ...
    async def get_handoff_member_record(self, reference: str) -> bytes: ...
    async def discard_trajectory_preemption(self, trajectory_id: str) -> None: ...


@dataclass
class RestoreOutcome:
    handoff_id: str
    group_id: str
    group_size: int
    resumed_members: list[int] = field(default_factory=list)
    completed_members: list[int] = field(default_factory=list)
    pending_members: list[int] = field(default_factory=list)
    finished_pending_members: list[int] = field(default_factory=list)
    resume_input_tokens: int = 0
    recomputed_prefill_tokens: int | None = 0
    tool_replay_count: int | None = 0
    stale_write_rejected: int = 0
    group_completion_latency_ms: float | None = None


def _validate_restore_identity(snapshot: Any, member: dict[str, Any],
                               manifest: HandoffManifest) -> None:
    """Fail closed on identity mismatch before any takeover proceeds."""
    if (snapshot.trajectory_id != member["trajectory_id"]
            or snapshot.group_id != member["group_id"]
            or snapshot.member_index != member["member_index"]
            or snapshot.group_size != member["group_size"]):
        raise ValueError("snapshot identity does not match the handoff manifest member")
    if (snapshot.environment_version != manifest.environment_version
            or snapshot.tool_version != manifest.tool_version):
        raise ValueError("snapshot environment/tool version does not match the manifest")
    if snapshot.policy_identity != manifest.policy_identity:
        raise ValueError("snapshot policy identity does not match the manifest")
    # KV Restore = OFF: the stored reference (if any) must never gate resume.
    # Token re-prefill is the only supported continuation; nothing reads kv_ref.


async def restore_handoff(
    client: HandoffStoreClient,
    handoff_id: str,
    *,
    resume_trajectory: Callable[[Any, Any], Any],
    owner_id: str = "target_rollouter",
    lease_seconds: float = 300.0,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[RestoreOutcome, dict[int, Any]]:
    """Take over one committed handoff and reassemble its groups in order.

    ``resume_trajectory(member, snapshot)`` continues a checkpointed trajectory
    by token re-prefill (ToolAgentLoop.resume_from_snapshot) and returns the
    finished member record. ``member['owner_lease']`` is the acquired lease;
    the callback must use it for ownership checks and the new generation,
    rather than attempting another takeover. The record may include tensors.
    """
    manifest = await client.get_handoff_manifest(handoff_id)
    if not isinstance(manifest, HandoffManifest):
        raise ValueError("handoff manifest object required")
    if await client.begin_handoff_restore(handoff_id) != HandoffState.RESTORING.value:
        raise ValueError("handoff is not restorable")
    started = clock()

    all_members = list(manifest.checkpointed_trajectories) + list(
        manifest.completed_members) + list(manifest.pending_samples)
    if not all_members:
        raise ValueError("handoff manifest has no members")
    group_id = all_members[0]["group_id"]
    group_size = all_members[0]["group_size"]
    if any(m["group_id"] != group_id or m["group_size"] != group_size for m in all_members):
        raise ValueError("manifest group geometry is inconsistent")

    outcome = RestoreOutcome(
        handoff_id=handoff_id, group_id=group_id, group_size=group_size,
    )

    records: dict[int, Any] = {}

    for member in manifest.completed_members:
        # Completed members never re-execute on the target (tool_replay == 0).
        payload = await client.get_handoff_member_record(member["record_ref"])
        records[member["member_index"]] = decode_handoff_member_record(payload)
        outcome.completed_members.append(member["member_index"])

    saved_results = await client.get_handoff_restored_members(handoff_id)
    for member in manifest.checkpointed_trajectories:
        index = member["member_index"]
        saved = saved_results.get(index)
        if saved is not None:
            record = decode_handoff_member_record(
                await client.get_handoff_member_record(saved["record_ref"]))
            resume_metrics = saved["metrics"]
        else:
            lease = await client.acquire_handoff_trajectory(
                handoff_id, member["trajectory_id"], owner_id, lease_seconds)
            snapshot = await client.load_trajectory_snapshot(lease)
            _validate_restore_identity(snapshot, member, manifest)
            record, resume_metrics = await resume_trajectory({**member, "owner_lease": lease}, snapshot)
            resume_metrics = {**resume_metrics,
                              "resume_input_tokens": len(snapshot.agent_state["prompt_ids"])}
            await client.record_handoff_member_result(
                handoff_id, index, lease, encode_handoff_member_record(record), resume_metrics)
        # Keep the checkpoint and saved result until the full group's actual
        # Artifact publication. A target interruption can then skip completed
        # work and reacquire only an expired incomplete member's existing lease.
        outcome.resume_input_tokens += int(resume_metrics["resume_input_tokens"])
        prefill = resume_metrics.get("recomputed_prefill_tokens")
        if prefill is None:
            outcome.recomputed_prefill_tokens = None
        elif outcome.recomputed_prefill_tokens is not None:
            outcome.recomputed_prefill_tokens += int(prefill)
        replay = resume_metrics.get("tool_replay_count")
        if replay is None:
            outcome.tool_replay_count = None
        elif outcome.tool_replay_count is not None:
            outcome.tool_replay_count += int(replay)
        if member["member_index"] in records:
            raise ValueError(f"duplicate member {member['member_index']} after resume")
        records[member["member_index"]] = record
        outcome.resumed_members.append(member["member_index"])

    for member in manifest.pending_samples:
        index = member["member_index"]
        if index in saved_results:
            records[index] = decode_handoff_member_record(
                await client.get_handoff_member_record(saved_results[index]["record_ref"]))
            outcome.finished_pending_members.append(index)
        else:
            outcome.pending_members.append(index)

    # The manifest's validate() already guarantees that completed /
    # checkpointed / pending partition 0..group_size-1.  restore_handoff is
    # only responsible for the completed + checkpointed records; pending
    # members are regenerated through the normal generation path and the full
    # group is validated again when it is published to the artifact plane.
    restored = sorted(outcome.completed_members + outcome.resumed_members + outcome.finished_pending_members)
    if restored != sorted(set(restored)):
        raise ValueError("duplicate members in restored group")
    if set(restored) & set(outcome.pending_members):
        raise ValueError("restored and pending members overlap")
    outcome.group_completion_latency_ms = (clock() - started) * 1000.0
    return outcome, records


async def publish_restored_group(
    client: Any,
    outcome: RestoreOutcome,
    records: dict[int, Any],
    pending_records: dict[int, Any],
    *,
    publish_group: Callable[[dict[int, Any]], Any],
) -> Any:
    """Assemble the full GRPO group and publish it through the artifact plane.

    ``publish_group`` is the single publish entry point on the target
    rollouter's artifact data plane (claim/ack by the Trainer happens exactly
    once through the normal claim path; no bypass channel is allowed).
    """
    merged = dict(records)
    for member_index, record in pending_records.items():
        if member_index in merged:
            raise ValueError(f"duplicate member {member_index} in assembled group")
        merged[member_index] = record
    if sorted(merged) != list(range(outcome.group_size)):
        raise ValueError(
            f"assembled group does not cover members 0..{outcome.group_size - 1}: "
            f"{sorted(merged)}")
    artifact_id = await publish_group(merged)
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("publish_group must return the published artifact id")
    await client.finish_handoff_manifest(outcome.handoff_id, artifact_id)
    return artifact_id
