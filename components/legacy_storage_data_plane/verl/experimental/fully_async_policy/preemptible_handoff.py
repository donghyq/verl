"""Cooperative preemption state and transactional handoff manifests.

Embedded in the serialized metadata actor next to TrajectorySnapshotRegistry.
This module does not introduce a second versioning system: ownership is still
fenced by the existing producer-generation registry, and manifests only
reference snapshot keys committed through TrajectorySnapshotRegistry.

Relevant requirements (Project B, Milestone 1):
- Safe-point preemption states: RUNNING -> PREEMPT_REQUESTED ->
  SAFE_POINT_REACHED -> SNAPSHOT_COMMITTED -> HANDED_OFF (+ ABORTED fallback).
- Transactional handoff manifest: PREPARING -> COMMITTED -> RESTORING ->
  FINISHED, COMMITTED only after every listed trajectory committed a snapshot.
- GRPO group integrity: completed / checkpointed / pending members must
  partition one group with no duplicates and no missing member_index.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from verl.experimental.fully_async_policy.trajectory_snapshot import canonical_bytes


class PreemptState(str, Enum):
    RUNNING = "running"
    PREEMPT_REQUESTED = "preempt_requested"
    SAFE_POINT_REACHED = "safe_point_reached"
    SNAPSHOT_COMMITTED = "snapshot_committed"
    HANDED_OFF = "handed_off"
    ABORTED = "aborted"


_TERMINAL_PREEMPT_STATES = (PreemptState.HANDED_OFF, PreemptState.ABORTED)


@dataclass(frozen=True)
class PreemptionRecord:
    trajectory_id: str
    state: PreemptState
    preempt_requested_at: float
    safe_point_at: float | None = None
    snapshot_committed_at: float | None = None
    handed_off_at: float | None = None
    aborted_at: float | None = None
    snapshot_ref: str | None = None

    def safe_point_wait_ms(self) -> float | None:
        if self.safe_point_at is None:
            return None
        return (self.safe_point_at - self.preempt_requested_at) * 1000.0

    def snapshot_commit_ms(self) -> float | None:
        if self.snapshot_committed_at is None or self.safe_point_at is None:
            return None
        return (self.snapshot_committed_at - self.safe_point_at) * 1000.0

    def worker_reclaim_latency_ms(self) -> float | None:
        end = self.handed_off_at if self.handed_off_at is not None else self.aborted_at
        if end is None:
            return None
        return (end - self.preempt_requested_at) * 1000.0


class TrajectoryPreemptionTracker:
    """Per-trajectory cooperative preemption state on the metadata actor.

    The agent-loop owner callback consults this table through the store actor;
    transitions only advance forward.  A trajectory may only be marked
    handed-off after its snapshot committed, and abort is the SAFE_POINT_TIMEOUT
    fallback for trajectories that never reach a tool-batch boundary.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.records: dict[str, PreemptionRecord] = {}

    def _record(self, trajectory_id: str) -> PreemptionRecord:
        if trajectory_id not in self.records:
            raise ValueError(f"unknown trajectory: {trajectory_id}")
        return self.records[trajectory_id]

    def register_running(self, trajectory_id: str) -> None:
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("trajectory_id must be a non-empty string")
        if trajectory_id in self.records:
            raise ValueError(f"trajectory already registered: {trajectory_id}")
        self.records[trajectory_id] = PreemptionRecord(trajectory_id, PreemptState.RUNNING, 0.0)

    def state(self, trajectory_id: str) -> PreemptState:
        return self._record(trajectory_id).state

    def request_preemption(self, trajectory_id: str) -> PreemptionRecord:
        """RUNNING -> PREEMPT_REQUESTED; idempotent once requested."""
        record = self._record(trajectory_id)
        if record.state in _TERMINAL_PREEMPT_STATES:
            raise ValueError("cannot preempt a terminal trajectory")
        if record.state is PreemptState.RUNNING:
            self.records[trajectory_id] = PreemptionRecord(
                trajectory_id, PreemptState.PREEMPT_REQUESTED, self.clock()
            )
        return self.records[trajectory_id]

    def mark_safe_point(self, trajectory_id: str) -> PreemptionRecord:
        record = self._record(trajectory_id)
        if record.state is not PreemptState.PREEMPT_REQUESTED:
            raise ValueError("safe point requires an active preemption request")
        self.records[trajectory_id] = PreemptionRecord(
            trajectory_id, PreemptState.SAFE_POINT_REACHED,
            record.preempt_requested_at, safe_point_at=self.clock(),
        )
        return self.records[trajectory_id]

    def mark_snapshot_committed(self, trajectory_id: str, snapshot_ref: str) -> PreemptionRecord:
        if not isinstance(snapshot_ref, str) or not snapshot_ref:
            raise ValueError("snapshot_ref must be a non-empty string")
        record = self._record(trajectory_id)
        if record.state is not PreemptState.SAFE_POINT_REACHED:
            raise ValueError("snapshot commit requires a reached safe point")
        self.records[trajectory_id] = PreemptionRecord(
            trajectory_id, PreemptState.SNAPSHOT_COMMITTED,
            record.preempt_requested_at, safe_point_at=record.safe_point_at,
            snapshot_committed_at=self.clock(), snapshot_ref=snapshot_ref,
        )
        return self.records[trajectory_id]

    def mark_handed_off(self, trajectory_id: str) -> PreemptionRecord:
        record = self._record(trajectory_id)
        if record.state is not PreemptState.SNAPSHOT_COMMITTED:
            raise ValueError("hand-off requires a committed snapshot")
        self.records[trajectory_id] = PreemptionRecord(
            record.trajectory_id, PreemptState.HANDED_OFF,
            record.preempt_requested_at, safe_point_at=record.safe_point_at,
            snapshot_committed_at=record.snapshot_committed_at,
            handed_off_at=self.clock(), snapshot_ref=record.snapshot_ref,
        )
        return self.records[trajectory_id]

    def mark_aborted(self, trajectory_id: str) -> PreemptionRecord:
        """SAFE_POINT_TIMEOUT fallback: the trajectory never reached a boundary."""
        record = self._record(trajectory_id)
        if record.state in _TERMINAL_PREEMPT_STATES:
            raise ValueError("cannot abort a terminal trajectory")
        if record.state is PreemptState.SNAPSHOT_COMMITTED:
            raise ValueError("snapshot already committed; hand off instead of aborting")
        self.records[trajectory_id] = PreemptionRecord(
            record.trajectory_id, PreemptState.ABORTED,
            record.preempt_requested_at, safe_point_at=record.safe_point_at,
            aborted_at=self.clock(),
        )
        return self.records[trajectory_id]

    def discard(self, trajectory_id: str) -> None:
        self._record(trajectory_id)
        del self.records[trajectory_id]

    def metrics(self) -> dict[str, Any]:
        rows = []
        for record in self.records.values():
            if record.preempt_requested_at == 0.0:
                continue  # still RUNNING, no preemption involved
            rows.append({
                "trajectory_id": record.trajectory_id,
                "state": record.state.value,
                "safe_point_wait_ms": record.safe_point_wait_ms(),
                "snapshot_commit_ms": record.snapshot_commit_ms(),
                "worker_reclaim_latency_ms": record.worker_reclaim_latency_ms(),
            })
        return {"trajectories": rows}


class HandoffState(str, Enum):
    PREPARING = "preparing"
    COMMITTED = "committed"
    RESTORING = "restoring"
    FINISHED = "finished"


@dataclass(frozen=True)
class HandoffManifest:
    handoff_id: str
    source_rollouter: str
    source_generation: int
    policy_identity: dict[str, str]
    environment_version: str
    tool_version: str
    completed_artifacts: tuple[str, ...]
    checkpointed_trajectories: tuple[dict[str, Any], ...]
    completed_members: tuple[dict[str, Any], ...]
    pending_samples: tuple[dict[str, Any], ...]
    schema_version: int = 1

    def member_tables(self) -> dict[str, dict[str, Any]]:
        """group_id -> {"group_size": int, "completed": [...], "checkpointed": [...], "pending": [...]}"""
        tables: dict[str, dict[str, Any]] = {}
        for category in ("completed_members", "checkpointed_trajectories", "pending_samples"):
            for member in getattr(self, category):
                table = tables.setdefault(member["group_id"], {
                    "group_size": member["group_size"],
                    "completed": [], "checkpointed": [], "pending": []})
                if table["group_size"] != member["group_size"]:
                    raise ValueError(f"group size mismatch for {member['group_id']}")
                key = "completed" if category == "completed_members" else (
                    "checkpointed" if category == "checkpointed_trajectories" else "pending")
                table[key].append(member["member_index"])
        return tables

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported handoff manifest schema")
        if not all(isinstance(name, str) and name for name in (
                self.handoff_id, self.source_rollouter, self.environment_version, self.tool_version)):
            raise ValueError("handoff manifest identity is missing")
        if type(self.source_generation) is not int or self.source_generation < 0:
            raise ValueError("handoff manifest source generation must be non-negative")
        from verl.experimental.fully_async_policy.rollout_artifact import _decode_policy_identity

        _decode_policy_identity(self.policy_identity).validate()
        if not isinstance(self.completed_artifacts, tuple):
            raise ValueError("completed_artifacts must be a tuple")
        seen_members: set[tuple[str, int]] = set()
        for artifact_id in self.completed_artifacts:
            if not isinstance(artifact_id, str) or not artifact_id:
                raise ValueError("completed artifact id must be a non-empty string")

        for member in self.checkpointed_trajectories:
            self._validate_member(member, "checkpointed_trajectories", seen_members)
            if not isinstance(member["snapshot_ref"], str) or not member["snapshot_ref"]:
                raise ValueError("checkpointed trajectory requires a snapshot_ref")
            if type(member["expected_generation"]) is not int or member["expected_generation"] < 0:
                raise ValueError("checkpointed trajectory requires expected_generation")
            if "input_ref" in member and (not isinstance(member["input_ref"], str) or not member["input_ref"]):
                raise ValueError("checkpointed member input_ref must be a non-empty string")
        for member in self.completed_members:
            self._validate_member(member, "completed_members", seen_members)
            if not isinstance(member["record_ref"], str) or not member["record_ref"]:
                raise ValueError("completed member requires a record_ref")
        for member in self.pending_samples:
            self._validate_member(member, "pending_samples", seen_members)
            if "record_ref" in member and (not isinstance(member["record_ref"], str) or not member["record_ref"]):
                raise ValueError("pending member record_ref must be a non-empty string")
        if not self.checkpointed_trajectories:
            raise ValueError("preemptible handoff requires at least one checkpointed trajectory")

        # GRPO group partition invariant: for every touched group the union of
        # completed / checkpointed / pending members must be exactly
        # 0..group_size-1 with no duplicates (missing=0, duplicate=0).
        for group_id, table in self.member_tables().items():
            size = table["group_size"]
            union: list[int] = []
            for members in (table["completed"], table["checkpointed"], table["pending"]):
                union.extend(members)
            if len(union) != size or sorted(union) != list(range(size)):
                raise ValueError(f"handoff members do not partition group {group_id}")

    @staticmethod
    def _validate_member(member: dict[str, Any], category: str, seen: set) -> None:
        if not isinstance(member, dict):
            raise ValueError(f"{category} entries must be dicts")
        required = {
            "trajectory_id", "group_id", "member_index", "group_size",
        } | ({"snapshot_ref", "expected_generation"} if category == "checkpointed_trajectories"
             else ({"record_ref"} if category == "completed_members" else set()))
        optional = ({"input_ref"} if category == "checkpointed_trajectories"
                    else ({"record_ref"} if category == "pending_samples" else set()))
        if not required <= set(member) or set(member) - required - optional:
            raise ValueError(f"{category} entry has unexpected fields: {sorted(member)}")
        if not all(isinstance(member[name], str) and member[name] for name in (
                "trajectory_id", "group_id")):
            raise ValueError(f"{category} entry identity is missing")
        if (type(member["member_index"]) is not int or type(member["group_size"]) is not int
                or not 0 <= member["member_index"] < member["group_size"] or member["group_size"] <= 0):
            raise ValueError(f"{category} entry membership is invalid")
        key = (member["group_id"], member["member_index"])
        if key in seen:
            raise ValueError(f"duplicate member {key} in handoff manifest")
        seen.add(key)

    def encode(self) -> bytes:
        self.validate()
        return canonical_bytes(asdict(self))

    @classmethod
    def decode(cls, payload: bytes) -> "HandoffManifest":
        decoded = json.loads(payload)
        for name in ("completed_artifacts", "checkpointed_trajectories", "completed_members", "pending_samples"):
            decoded[name] = tuple(decoded[name])
        manifest = cls(**decoded)
        manifest.validate()
        return manifest


@dataclass
class HandoffRecord:
    manifest: HandoffManifest
    state: HandoffState
    prepared_at: float
    committed_at: float | None = None
    source_released_at: float | None = None
    restoring_started_at: float | None = None
    finished_at: float | None = None
    restored_member_refs: dict[int, str] = field(default_factory=dict)
    restored_member_metrics: dict[int, dict[str, Any]] = field(default_factory=dict)
    restored_member_leases: dict[int, Any] = field(default_factory=dict)
    published_artifact_id: str | None = None

    def worker_reclaim_latency_ms(self) -> float | None:
        if self.committed_at is None:
            return None
        end = self.source_released_at if self.source_released_at is not None else self.committed_at
        return (end - self.prepared_at) * 1000.0


class HandoffManifestRegistry:
    """Transactional handoff manifests on the metadata actor.

    PREPARING -> COMMITTED is the atomic boundary: commit fails unless every
    listed checkpointed trajectory reached SNAPSHOT_COMMITTED in the
    preemption tracker and the source producer generation is still current
    (late/stale sources fail closed through the existing fencing).
    """

    def __init__(self, tracker: TrajectoryPreemptionTracker,
                 current_producer_generation: Callable[[str], int | None],
                 clock: Callable[[], float] = time.monotonic):
        self.tracker = tracker
        self.current_producer_generation = current_producer_generation
        self.clock = clock
        self.records: dict[str, HandoffRecord] = {}

    def _require(self, handoff_id: str) -> HandoffRecord:
        if not isinstance(handoff_id, str) or not handoff_id or "/" in handoff_id:
            raise ValueError("handoff_id must be a non-empty path-safe string")
        record = self.records.get(handoff_id)
        if record is None:
            raise ValueError(f"unknown handoff: {handoff_id}")
        return record

    def _require_state(self, handoff_id: str, *states: HandoffState) -> HandoffRecord:
        record = self._require(handoff_id)
        if record.state not in states:
            raise ValueError(f"handoff {handoff_id} is {record.state.value}, not one of "
                             f"{[state.value for state in states]}")
        return record

    def prepare(self, manifest: HandoffManifest) -> HandoffRecord:
        manifest.validate()
        existing = self.records.get(manifest.handoff_id)
        if existing is not None:
            if (existing.state is HandoffState.PREPARING
                    and existing.manifest == manifest):
                return existing  # idempotent prepare of the same draft
            raise ValueError(f"handoff already exists: {manifest.handoff_id}")
        current = self.current_producer_generation(manifest.source_rollouter)
        if current != manifest.source_generation:
            # Late or stale source rollouter fails closed; no second epoch
            # system is introduced, the producer fencing decides.
            raise ValueError(
                f"source generation {manifest.source_generation} is not current "
                f"(current={current})")
        record = HandoffRecord(manifest, HandoffState.PREPARING, self.clock())
        self.records[manifest.handoff_id] = record
        return record

    def commit(self, handoff_id: str) -> HandoffRecord:
        """PREPARING -> COMMITTED after every listed snapshot is committed."""
        record = self._require_state(handoff_id, HandoffState.PREPARING)
        current = self.current_producer_generation(record.manifest.source_rollouter)
        if current != record.manifest.source_generation:
            raise ValueError("source generation changed before handoff commit")
        for member in record.manifest.checkpointed_trajectories:
            trajectory = self.tracker.records.get(member["trajectory_id"])
            if trajectory is None or trajectory.state not in (
                    PreemptState.SNAPSHOT_COMMITTED, PreemptState.HANDED_OFF):
                raise ValueError(
                    f"cannot commit handoff; trajectory {member['trajectory_id']} "
                    f"has no committed snapshot")
            if trajectory.snapshot_ref != member["snapshot_ref"]:
                raise ValueError(
                    f"manifest snapshot_ref mismatch for {member['trajectory_id']}")
        record.state = HandoffState.COMMITTED
        record.committed_at = self.clock()
        return record

    def record_source_release(self, handoff_id: str) -> HandoffRecord:
        record = self._require_state(handoff_id, HandoffState.COMMITTED)
        if record.source_released_at is None:
            record.source_released_at = self.clock()
        return record

    def begin_restore(self, handoff_id: str) -> HandoffRecord:
        record = self._require_state(handoff_id, HandoffState.COMMITTED, HandoffState.RESTORING)
        if record.state is HandoffState.RESTORING:
            # Per-trajectory owner leases still reject another live target.
            # A target that died after starting restore may be retried once
            # those leases expire, without rolling back the manifest state.
            return record
        record.state = HandoffState.RESTORING
        record.restoring_started_at = self.clock()
        return record

    def finish(self, handoff_id: str, artifact_id: str) -> HandoffRecord:
        record = self._require_state(handoff_id, HandoffState.RESTORING)
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("finished handoff requires a published artifact id")
        record.published_artifact_id = artifact_id
        record.state = HandoffState.FINISHED
        record.finished_at = self.clock()
        return record

    def cancel(self, handoff_id: str) -> HandoffRecord:
        record = self._require_state(handoff_id, HandoffState.PREPARING, HandoffState.RESTORING)
        del self.records[handoff_id]
        return record

    def get(self, handoff_id: str) -> HandoffRecord:
        return self._require(handoff_id)

    def metrics(self) -> dict[str, Any]:
        rows = []
        for record in self.records.values():
            rows.append({
                "handoff_id": record.manifest.handoff_id,
                "state": record.state.value,
                "prepare_to_commit_ms": None if record.committed_at is None
                    else (record.committed_at - record.prepared_at) * 1000.0,
                "worker_reclaim_latency_ms": record.worker_reclaim_latency_ms(),
                "commit_to_restore_ms": None if record.restoring_started_at is None
                    else (record.restoring_started_at - record.committed_at) * 1000.0,
                "restore_to_finish_ms": None if record.finished_at is None
                    else (record.finished_at - record.restoring_started_at) * 1000.0,
            })
        return {"handoffs": rows}


class HandoffMemberRecordStore:
    """JSON payloads for completed group members, addressed by opaque refs.

    A completed member never re-executes on the target; its serialized
    AgentLoopOutput travels through the manifest's record_ref instead of the
    trajectory-snapshot path (which only models mid-flight tool boundaries).
    """

    def __init__(self):
        self.records: dict[str, bytes] = {}

    def put(self, payload: bytes) -> str:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("member record payload must be non-empty bytes")
        json.loads(payload.decode("utf-8"))  # must be valid JSON
        reference = "handoff-member/" + uuid4().hex
        self.records[reference] = payload
        return reference

    def get(self, reference: str) -> bytes:
        if reference not in self.records:
            raise ValueError(f"unknown member record: {reference}")
        return self.records[reference]

    def discard(self, reference: str) -> None:
        self.records.pop(reference, None)


def encode_handoff_member_record(record: Any) -> bytes:
    """Preserve full runtime outputs using the existing cloudpickle format.

    These bytes are private, trusted inputs to the metadata actor, just like
    RolloutSample payloads. They are not a public deserialization interface.
    The JSON envelope permits integrity checks without deserializing tensors.
    """
    import cloudpickle

    payload = cloudpickle.dumps(record)
    return canonical_bytes({
        "handoff_member_schema": 1,
        "encoding": "cloudpickle",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "payload": base64.b64encode(payload).decode("ascii"),
    })


def decode_handoff_member_record(payload: bytes) -> Any:
    envelope = json.loads(payload)
    if not isinstance(envelope, dict) or "handoff_member_schema" not in envelope:
        return envelope  # existing JSON-only member fixtures remain readable
    if (envelope.get("handoff_member_schema") != 1
            or envelope.get("encoding") != "cloudpickle"
            or set(envelope) != {"handoff_member_schema", "encoding", "sha256", "payload"}):
        raise ValueError("unsupported handoff member encoding")
    raw = base64.b64decode(envelope["payload"], validate=True)
    if hashlib.sha256(raw).hexdigest() != envelope["sha256"]:
        raise ValueError("handoff member payload digest mismatch")
    import cloudpickle

    return cloudpickle.loads(raw)


class HandoffRestoreCoordinator:
    """Reuse manifest progress and trajectory leases on the same actor.

    The actor and its object store must survive the rollout worker failures
    in scope. This coordinator does not claim durable metadata recovery after
    the metadata actor dies, and introduces no additional epoch authority.
    """

    def __init__(self, manifests, snapshots, member_records, artifact_store):
        self.manifests = manifests
        self.snapshots = snapshots
        self.member_records = member_records
        self.artifact_store = artifact_store

    def _member(self, handoff_id, trajectory_id, *, include_pending=False):
        record = self.manifests._require_state(handoff_id, HandoffState.RESTORING)
        members = record.manifest.checkpointed_trajectories
        if include_pending:
            members += record.manifest.pending_samples
        for member in members:
            if member["trajectory_id"] == trajectory_id:
                return record, member
        raise ValueError("trajectory is not a checkpointed handoff member")

    def acquire(self, handoff_id, trajectory_id, owner_id, lease_seconds):
        record, member = self._member(handoff_id, trajectory_id)
        if member["member_index"] in record.restored_member_refs:
            raise ValueError("handoff member already restored")
        current = self.snapshots.owners.get(trajectory_id)
        if current is None or current.generation < member["expected_generation"]:
            raise ValueError("handoff trajectory owner is unavailable")
        if self.snapshots.latest.get(trajectory_id) != member["snapshot_ref"]:
            raise ValueError("handoff snapshot reference changed")
        # Initial takeover uses source N; retry after target death uses the
        # current expired target N+1. The same registry increments generation.
        return self.snapshots.takeover(trajectory_id, current.generation, owner_id, lease_seconds)

    def acquire_pending(self, handoff_id, trajectory_id, owner_id, lease_seconds):
        record, member = self._member(handoff_id, trajectory_id, include_pending=True)
        if member not in record.manifest.pending_samples:
            raise ValueError("trajectory is not a pending handoff member")
        if member["member_index"] in record.restored_member_refs:
            raise ValueError("handoff pending member already completed")
        current = self.snapshots.owners.get(trajectory_id)
        if current is None:
            return self.snapshots.open(trajectory_id, owner_id, lease_seconds)
        if self.snapshots.clock() < current.expires_at:
            raise ValueError("pending trajectory still has a live owner")
        if trajectory_id in self.snapshots.latest:
            raise ValueError("pending trajectory has a checkpoint; blind restart is forbidden")
        # An interrupted pending member has no committed checkpoint. Restart
        # from the original input, explicitly losing any uncommitted work.
        # _assign is the registry's existing producer-generation authority.
        return self.snapshots._assign(trajectory_id, owner_id, lease_seconds)

    def record_result(self, handoff_id, member_index, owner_lease, payload, metrics):
        record, member = self._member(handoff_id, owner_lease.trajectory_id, include_pending=True)
        if member["member_index"] != member_index:
            raise ValueError("restored member index does not match its owner")
        self.snapshots._check(owner_lease)
        json.loads(payload.decode("utf-8"))
        previous = record.restored_member_refs.get(member_index)
        if previous is not None:
            if self.member_records.get(previous) != payload:
                raise ValueError("restored handoff member cannot be rewritten")
            return previous
        reference = self.member_records.put(payload)
        record.restored_member_refs[member_index] = reference
        record.restored_member_metrics[member_index] = deepcopy(metrics)
        record.restored_member_leases[member_index] = owner_lease
        return reference

    def restored(self, handoff_id):
        record = self.manifests._require_state(handoff_id, HandoffState.RESTORING)
        return {
            index: {"record_ref": reference, "metrics": deepcopy(record.restored_member_metrics[index])}
            for index, reference in record.restored_member_refs.items()
        }

    def find_artifact(self, handoff_id, artifact_id):
        """Inspect the original immutable submission, without touching I/O."""
        from verl.experimental.fully_async_policy.rollout_artifact import RolloutArtifactStatus

        record = self.manifests.get(handoff_id)
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("artifact id must be non-empty")
        metadata = getattr(self.artifact_store, "_metadata", self.artifact_store)
        artifact = metadata._artifacts.get(artifact_id)
        if artifact is None:
            return None
        members = (record.manifest.completed_members + record.manifest.checkpointed_trajectories
                   + record.manifest.pending_samples)
        group_id, group_size = members[0]["group_id"], members[0]["group_size"]
        if (artifact.status not in (
                RolloutArtifactStatus.READY, RolloutArtifactStatus.LEASED, RolloutArtifactStatus.CONSUMED)
                or artifact.manifest.rollout_group_id != group_id
                or artifact.manifest.trajectory_count != group_size
                or asdict(artifact.manifest.identity) != record.manifest.policy_identity
                or artifact.manifest.rollout_group_size != 1
                or artifact.manifest.rollout_sample_index != 0):
            raise ValueError("artifact does not match the actual complete-group handoff publication")
        return {"artifact_id": artifact_id, "status": artifact.status.value}

    def finish(self, handoff_id, artifact_id):
        record = self.manifests.get(handoff_id)
        if record.state is HandoffState.FINISHED:
            if record.published_artifact_id != artifact_id:
                raise ValueError("handoff already finished for a different artifact")
            return record
        self.manifests._require_state(handoff_id, HandoffState.RESTORING)
        required = {m["member_index"] for m in record.manifest.checkpointed_trajectories}
        if not required <= set(record.restored_member_refs):
            raise ValueError("cannot finish before every checkpointed member is restored")
        if self.find_artifact(handoff_id, artifact_id) is None:
            raise ValueError("finish requires the actual complete-group artifact publication")
        # Only the actual complete-group publication permits deleting the
        # original checkpoints. Saved outputs survive any target interruption
        # before this boundary; the Trainer keeps its ordinary claim/ack path.
        for member in record.manifest.checkpointed_trajectories + record.manifest.pending_samples:
            trajectory_id = member["trajectory_id"]
            owner = self.snapshots.owners.get(trajectory_id)
            if owner is not None and self.snapshots.clock() < owner.expires_at:
                self.snapshots.finish(owner)
            elif trajectory_id in self.snapshots.latest:
                # The completed artifact supersedes this saved checkpoint.
                # With an expired owner no live execution can write it.
                self.snapshots.expiry[trajectory_id] = self.snapshots.clock()
                self.snapshots.collect_expired()
            if trajectory_id in self.manifests.tracker.records:
                self.manifests.tracker.discard(trajectory_id)
        finished = self.manifests.finish(handoff_id, artifact_id)
        original_refs = {m["record_ref"] for m in record.manifest.completed_members}
        original_refs.update(m["record_ref"] for m in record.manifest.pending_samples if "record_ref" in m)
        original_refs.update(m["input_ref"] for m in record.manifest.checkpointed_trajectories if "input_ref" in m)
        for reference in set(record.restored_member_refs.values()) | original_refs:
            self.member_records.discard(reference)
        return finished
