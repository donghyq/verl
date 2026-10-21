# Copyright 2026 Project B contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact protocol migration from Project B f91c201d; see MIGRATION.md."""
from __future__ import annotations

import base64
import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

from .snapshot import canonical_bytes

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
        from .identity import _decode_policy_identity

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


def _validate_unscored_member(record):
    if not isinstance(record, dict) or record.get("trajectory_runtime_record") != 1:
        raise ValueError("Unsupported unscored member record")
    if (record.get("kind") != "unscored" or record.get("scoring_state") != "unscored"
            or record.get("trainable") is not False or "output" not in record):
        raise ValueError("Record is not an explicitly unscored, non-trainable result")
    uid, index, size = record.get("uid"), record.get("member_index"), record.get("group_size")
    original, score_kwargs = record.get("member_kwargs"), record.get("score_kwargs")
    if (not isinstance(uid, str) or not uid or type(size) is not int or size != 8
            or type(index) is not int or not 0 <= index < size
            or record.get("trajectory_id") != f"{uid}/member-{index}"
            or not isinstance(original, dict) or original.get("uid") != uid
            or original.get("session_id") != index
            or not isinstance(score_kwargs, dict) or score_kwargs.get("uid") != uid
            or score_kwargs.get("session_id") != index):
        raise ValueError("Unscored result differs from its original prompt member")
    return record


async def save_unscored_member(client, index, output, original, *, score_kwargs):
    """PUT immutable generated state before the unchanged original scorer runs."""
    uid = original["uid"]
    record = _validate_unscored_member({
        "trajectory_runtime_record": 1, "kind": "unscored",
        "scoring_state": "unscored", "trainable": False,
        "uid": uid, "member_index": index,
        "group_size": original["__trajectory_group_size__"],
        "trajectory_id": f"{uid}/member-{index}",
        "member_kwargs": original, "score_kwargs": score_kwargs, "output": output,
    })
    # Existing codec preserves all native AgentLoopOutput fields, token IDs,
    # logprobs and structured messages/tool results without reconstructing them.
    reference = await client.put_handoff_member_record(encode_handoff_member_record(record))
    print("PROJECT_B_UNSCORED_MEMBER " + json.dumps({
        "trajectory_id": record["trajectory_id"], "reference": reference,
        "kind": "unscored", "trainable": False,
    }, sort_keys=True), flush=True)
    return reference


async def load_unscored_member(client, reference):
    """Read through the existing store's length/digest-checked object API."""
    return _validate_unscored_member(decode_handoff_member_record(await client.get_handoff_member_record(reference)))


async def rescore_unscored_member(client, reference, worker):
    """Run only the unchanged native score method on a fresh decoded result.

    This does not call the agent loop, a model generator, TransferQueue, or group
    publication. A scored result still needs stock V1 preparation and all eight
    group members before it can be used for training.
    """
    record = await load_unscored_member(client, reference)
    output = record["output"]
    outputs = output if isinstance(output, list) else [output]
    if not outputs:
        raise ValueError("Unscored result has no native outputs")
    await worker._compute_score(outputs, kwargs=deepcopy(record["score_kwargs"]))
    if outputs[-1].reward_score is None:
        raise ValueError("The original scorer returned without a reward")
    return {"source_reference": reference, "scoring_state": "score_call_completed",
            "trainable": False, "output": output,
            "member_kwargs": record["member_kwargs"], "score_kwargs": record["score_kwargs"]}
