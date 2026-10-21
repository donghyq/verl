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

"""Durable coordination for unfinished, text ToolAgentLoop trajectories.

All mutating calls are serialized by one coordinator. Metadata durability is
required before returning a new lease, committed snapshot, member result or
publication reference. Restart does not resurrect execution leases. Finished
samples continue to use the stock V1 TransferQueue and trainer algorithms.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .generation import ProducerGenerationRegistry
from .handoff import (
    HandoffManifest,
    HandoffManifestRegistry,
    HandoffRecord,
    HandoffState,
    PreemptionRecord,
    PreemptState,
    TrajectoryPreemptionTracker,
    decode_handoff_member_record,
)
from .policy_publication import (
    EnginePolicyObservation,
    HandoffPublicationBarrier,
    PolicyPublicationBlocked,
    PolicyPublicationTicket,
    PublishedPolicyRecord,
    empty_policy_state,
    nonempty,
    policy_version,
    validate_policy_state,
)
from .snapshot import TrajectoryOwnerLease, TrajectorySnapshot, canonical_bytes
from .storage import CoordinationStorageError, FileMetadataStore, FileObjectStore, ObjectStore


class PublicationBlocked(RuntimeError):
    """A dispatched write cannot be retried without checking the real queue."""


class DurableTrajectoryCoordinator:
    def __init__(
        self,
        root_dir: str,
        run_id: str,
        *,
        object_store: ObjectStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        fault_hook: Callable[[str], None] | None = None,
    ):
        if (not isinstance(run_id, str) or run_id in {".", ".."}
                or re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) is None):
            raise ValueError("run_id must be a non-empty path-safe identity")
        self.run_id, self.clock, self.wall_clock = run_id, clock, wall_clock
        self.root = Path(root_dir) / run_id
        self._metadata = FileMetadataStore(self.root, fault_hook=fault_hook)
        self._closed = self._poisoned = False
        self._recovered = False
        self._leases: dict[str, TrajectoryOwnerLease] = {}
        self._policy_barrier_deadlines: dict[str, float] = {}
        try:
            self._objects = object_store if object_store is not None else FileObjectStore(self.root / "objects")
            saved = self._metadata.load()
            self._state = saved if saved is not None else {
                "schema_version": 1,
                "run_id": run_id,
                "sequence": 0,
                "producers": {},
                "trajectory_producers": {},
                "trajectories": {},
                "preemptions": {},
                "handoffs": {},
                "objects": {},
                "gc_pending": [],
                "policy_publications": empty_policy_state(),
            }
            # An additive metadata migration; Snapshot/Manifest bytes and
            # original producer/execution generations remain unchanged.
            if isinstance(self._state, dict):
                self._state.setdefault("policy_publications", empty_policy_state())
            self._validate_state()
            if saved is None:
                self._metadata.save(self._state)
            else:
                self._recovered = True
                # A queue write receipt was valid before the coordinator died.
                # Verify the actual queue again before finishing an unfinished
                # handoff. Old monotonic owner deadlines are never hydrated.
                def require_revalidation(draft):
                    policy = draft["policy_publications"]
                    for name in ("publication", "barrier"):
                        if policy[name] is not None:
                            if name == "publication" and policy[name]["state"] != "BLOCKED":
                                policy[name]["failure"] = "COORDINATOR_RESTART_FROM_" + policy[name]["state"]
                            policy[name]["state"] = "BLOCKED"
                    for row in draft["handoffs"].values():
                        publication = row.get("publication")
                        if row["state"] != HandoffState.FINISHED.value and publication is not None:
                            if publication["state"] == "PUBLISHED":
                                publication["state"] = "NEEDS_VERIFICATION"
                self._transaction(require_revalidation)
        except Exception:
            self._metadata.close()
            raise

    def _require_open(self) -> None:
        if self._closed or self._poisoned:
            raise CoordinationStorageError("coordinator is closed or has an uncertain durable write; restart it")

    def _transaction(self, mutate: Callable[[dict], Any]) -> Any:
        self._require_open()
        draft = deepcopy(self._state)
        result = mutate(draft)
        draft["sequence"] += 1
        try:
            self._metadata.save(draft)
        except Exception:
            self._poisoned = True
            raise
        self._state = draft
        return result

    def _validate_state(self) -> None:
        state = self._state
        expected = {
            "schema_version", "run_id", "sequence", "producers", "trajectory_producers",
            "trajectories", "preemptions", "handoffs", "objects", "gc_pending",
            "policy_publications",
        }
        if (not isinstance(state, dict) or set(state) != expected
                or state["schema_version"] != 1 or state["run_id"] != self.run_id
                or type(state["sequence"]) is not int or state["sequence"] < 0
                or not isinstance(state["gc_pending"], list)):
            raise CoordinationStorageError("invalid trajectory coordination schema or run identity")
        for name in expected - {"schema_version", "run_id", "sequence", "gc_pending"}:
            if not isinstance(state[name], dict):
                raise CoordinationStorageError("invalid trajectory coordination table")
        try:
            validate_policy_state(state["policy_publications"], self.run_id)
        except (ValueError, TypeError, KeyError) as error:
            raise CoordinationStorageError("invalid durable policy publication state") from error
        for name in ("producers", "trajectory_producers"):
            for producer, incarnations in state[name].items():
                if (not producer or not isinstance(incarnations, dict) or not incarnations
                        or any(not incarnation or type(generation) is not int or generation < 0
                               for incarnation, generation in incarnations.items())):
                    raise CoordinationStorageError("invalid persisted producer generation history")
        for trajectory_id, row in state["trajectories"].items():
            owner = row.get("owner")
            if (not isinstance(owner, dict) or set(owner) != {"owner_id", "generation", "lease_id"}
                    or type(owner["generation"]) is not int
                    or ProducerGenerationRegistry(state["trajectory_producers"])
                    .get_current_producer_generation(trajectory_id) != owner["generation"]):
                raise CoordinationStorageError("trajectory owner does not match durable generation history")
            reference = row.get("latest_ref")
            if reference is not None:
                snapshot = TrajectorySnapshot.decode(self._read_object(reference))
                if snapshot.trajectory_id != trajectory_id or snapshot.execution_generation > owner["generation"]:
                    raise CoordinationStorageError("persisted snapshot owner identity mismatch")
                if not isinstance(row.get("snapshot_expires_at_unix"), (int, float)):
                    raise CoordinationStorageError("snapshot retention deadline is missing")
        self._registry(state)
        for handoff_id, row in state["handoffs"].items():
            manifest = self._manifest(row)
            if manifest.handoff_id != handoff_id or len(manifest.member_tables()) != 1:
                raise CoordinationStorageError("durable handoff must have exactly one prompt group")
            if row["state"] != HandoffState.FINISHED.value:
                self._verify_manifest_objects(state, manifest, enforce_retention=False)
                for reference in row["restored_member_refs"].values():
                    self._read_object(reference)
            elif row.get("publication", {}).get("state") != "PUBLISHED":
                raise CoordinationStorageError("finished handoff lacks a verified publication reference")

    def _read_object(self, reference: str) -> bytes:
        descriptor = self._state["objects"].get(reference)
        if descriptor is None:
            raise CoordinationStorageError("unknown durable trajectory object")
        try:
            payload = self._objects.get(descriptor["storage_key"])
        except Exception as error:
            raise CoordinationStorageError("durable trajectory object is unavailable") from error
        if (not isinstance(payload, bytes) or len(payload) != descriptor["bytes"]
                or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]):
            raise CoordinationStorageError("durable trajectory object failed integrity verification")
        return payload

    def _put_object(self, draft: dict, category: str, payload: bytes) -> str:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("trajectory object must be non-empty bytes")
        digest = hashlib.sha256(payload).hexdigest()
        storage_key = category + "/" + digest
        # Keep opaque logical references, as in the original codec. Distinct
        # callers can own identical inputs without one handoff's GC deleting
        # another caller's not-yet-attached record.
        reference = category + "/" + uuid4().hex
        self._objects.put(storage_key, payload)
        readback = self._objects.get(storage_key)
        if readback != payload:
            raise CoordinationStorageError("trajectory object PUT did not preserve exact bytes")
        draft["objects"][reference] = {"sha256": digest, "bytes": len(payload), "storage_key": storage_key}
        return reference

    def register_producer(self, producer_id: str, incarnation_id: str) -> int:
        return self._transaction(lambda draft: ProducerGenerationRegistry(draft["producers"])
                                 .register_producer(producer_id, incarnation_id))

    def get_current_producer_generation(self, producer_id: str) -> int | None:
        self._require_open()
        return ProducerGenerationRegistry(self._state["producers"]).get_current_producer_generation(producer_id)

    def _check_owner(self, lease: TrajectoryOwnerLease) -> None:
        self._require_open()
        if not isinstance(lease, TrajectoryOwnerLease):
            raise ValueError("stale or expired trajectory owner")
        # Renewal returns a new immutable lease with the same identity. A
        # callback that still holds the pre-renewal object must remain usable;
        # the in-process current lease, not either caller copy, is authoritative.
        current = self._leases.get(lease.trajectory_id)
        lease_identity = (lease.trajectory_id, lease.owner_id, lease.generation, lease.lease_id)
        if (current is None
                or (current.trajectory_id, current.owner_id, current.generation, current.lease_id) != lease_identity
                or self.clock() >= current.expires_at
                or lease.expires_at > current.expires_at):
            raise ValueError("stale or expired trajectory owner")
        owner = self._state["trajectories"][lease.trajectory_id]["owner"]
        if owner != {"owner_id": current.owner_id, "generation": current.generation, "lease_id": current.lease_id}:
            raise ValueError("stale or expired trajectory owner")

    def check_trajectory_owner(self, lease: TrajectoryOwnerLease) -> None:
        self._check_owner(lease)

    def _assign(self, draft: dict, trajectory_id: str, owner_id: str, seconds: float) -> TrajectoryOwnerLease:
        if (not isinstance(trajectory_id, str) or not trajectory_id
                or not isinstance(owner_id, str) or not owner_id
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("invalid execution lease")
        generation = ProducerGenerationRegistry(draft["trajectory_producers"]).register_producer(
            trajectory_id, uuid4().hex
        )
        lease = TrajectoryOwnerLease(trajectory_id, owner_id, generation, uuid4().hex, self.clock() + seconds)
        row = draft["trajectories"].setdefault(trajectory_id, {})
        row["owner"] = {"owner_id": owner_id, "generation": generation, "lease_id": lease.lease_id}
        row.pop("released", None)
        return lease

    def open_trajectory(self, trajectory_id: str, owner_id: str, lease_seconds: float) -> TrajectoryOwnerLease:
        def open_owner(draft):
            if trajectory_id in draft["trajectories"]:
                raise ValueError("trajectory already registered; use compare-generation takeover")
            return self._assign(draft, trajectory_id, owner_id, lease_seconds)
        lease = self._transaction(open_owner)
        self._leases[trajectory_id] = lease
        return lease

    def renew_trajectory_owner(self, lease: TrajectoryOwnerLease, lease_seconds: float) -> TrajectoryOwnerLease:
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("invalid lease renewal")
        self._check_owner(lease)
        current = self._leases[lease.trajectory_id]
        expires_at = max(current.expires_at, self.clock() + lease_seconds)
        renewed = TrajectoryOwnerLease(
            current.trajectory_id, current.owner_id, current.generation, current.lease_id, expires_at
        )

        def renew(draft):
            # The deadline is intentionally process-local: coordinator restart
            # must not resurrect an old owner. This durable write still confirms
            # the exact identity before new rights are returned.
            owner = draft["trajectories"][current.trajectory_id]["owner"]
            if owner != {"owner_id": current.owner_id, "generation": current.generation,
                         "lease_id": current.lease_id}:
                raise ValueError("stale or expired trajectory owner")

        self._transaction(renew)
        self._leases[lease.trajectory_id] = renewed
        return renewed

    def release_trajectory_owner(self, lease: TrajectoryOwnerLease) -> None:
        self._check_owner(lease)
        self._transaction(lambda draft: draft["trajectories"][lease.trajectory_id].update({"released": True}))
        self._leases.pop(lease.trajectory_id, None)

    def finish_trajectory(self, lease: TrajectoryOwnerLease) -> None:
        """Release a normally finished source member; completed bytes are separate."""
        self._check_owner(lease)
        reference = self._state["trajectories"][lease.trajectory_id].get("latest_ref")
        if reference is not None and self._reference_is_pinned(reference):
            raise ValueError("cannot finish a trajectory pinned by an unfinished handoff")
        def finish(draft):
            row = draft["trajectories"][lease.trajectory_id]
            row["released"] = True
            row.pop("latest_ref", None)
            row.pop("snapshot_expires_at_unix", None)
            if reference is not None:
                draft["gc_pending"].append(reference)
        self._transaction(finish)
        self._leases.pop(lease.trajectory_id, None)

    def _tracker(self, state: dict) -> TrajectoryPreemptionTracker:
        tracker = TrajectoryPreemptionTracker(clock=self.clock)
        for trajectory_id, raw in state["preemptions"].items():
            values = dict(raw)
            values["state"] = PreemptState(values["state"])
            tracker.records[trajectory_id] = PreemptionRecord(**values)
        return tracker

    @staticmethod
    def _save_tracker(draft: dict, tracker: TrajectoryPreemptionTracker) -> None:
        draft["preemptions"] = {
            trajectory_id: {**asdict(record), "state": record.state.value}
            for trajectory_id, record in tracker.records.items()
        }

    def _preemption_transition(self, method: str, trajectory_id: str, *args) -> str | None:
        def advance(draft):
            tracker = self._tracker(draft)
            result = getattr(tracker, method)(trajectory_id, *args)
            self._save_tracker(draft, tracker)
            return result.state.value if result is not None else None
        return self._transaction(advance)

    def _owned_preemption_transition(self, method: str, lease: TrajectoryOwnerLease, *args) -> str | None:
        def advance(draft):
            self._check_owner(lease)
            tracker = self._tracker(draft)
            result = getattr(tracker, method)(lease.trajectory_id, *args)
            self._save_tracker(draft, tracker)
            return result.state.value if result is not None else None
        return self._transaction(advance)

    def register_running_trajectory(self, trajectory_id: str) -> None:
        if trajectory_id not in self._state["trajectories"]:
            raise ValueError("running trajectory requires an execution owner")
        self._preemption_transition("register_running", trajectory_id)

    def register_owned_running_trajectory(self, lease: TrajectoryOwnerLease) -> None:
        self._owned_preemption_transition("register_running", lease)

    def request_trajectory_preemption(self, trajectory_id: str) -> str:
        return self._preemption_transition("request_preemption", trajectory_id)

    def request_owned_trajectory_preemption(self, lease: TrajectoryOwnerLease) -> str:
        return self._owned_preemption_transition("request_preemption", lease)

    def mark_trajectory_safe_point(self, trajectory_id: str) -> str:
        return self._preemption_transition("mark_safe_point", trajectory_id)

    def mark_owned_trajectory_safe_point(self, lease: TrajectoryOwnerLease) -> str:
        return self._owned_preemption_transition("mark_safe_point", lease)

    def get_trajectory_preemption(self, trajectory_id: str) -> str:
        self._require_open()
        return self._tracker(self._state).state(trajectory_id).value

    def mark_trajectory_snapshot_committed(self, trajectory_id: str, snapshot_ref: str) -> str:
        record = self._tracker(self._state)._record(trajectory_id)
        if record.state is PreemptState.SNAPSHOT_COMMITTED and record.snapshot_ref == snapshot_ref:
            return record.state.value
        raise ValueError("snapshot commit marker must come from the durable snapshot operation")

    def mark_trajectory_handed_off(self, trajectory_id: str) -> str:
        return self._preemption_transition("mark_handed_off", trajectory_id)

    def mark_trajectory_aborted(self, trajectory_id: str) -> str:
        return self._preemption_transition("mark_aborted", trajectory_id)

    def abort_owned_trajectory(self, lease: TrajectoryOwnerLease) -> dict:
        """Fence a failed execution in one serialized, durable actor operation.

        Source preemption state and ownership release share a transaction. A
        committed source snapshot is retained for handoff; a target failure
        releases its current rights without rewriting the source's HANDED_OFF
        history. No caller-side check/id-mutation pair is sufficient here.
        """
        def abort(draft):
            self._check_owner(lease)
            tracker = self._tracker(draft)
            record = tracker.records.get(lease.trajectory_id)
            if record is not None and record.state is PreemptState.SNAPSHOT_COMMITTED:
                return {"trajectory_id": lease.trajectory_id,
                        "state": record.state.value, "released": False}
            if record is not None:
                if record.state not in {PreemptState.HANDED_OFF, PreemptState.ABORTED}:
                    record = tracker.mark_aborted(lease.trajectory_id)
                self._save_tracker(draft, tracker)
            draft["trajectories"][lease.trajectory_id]["released"] = True
            return {"trajectory_id": lease.trajectory_id,
                    "state": record.state.value if record is not None else None, "released": True}

        result = self._transaction(abort)
        if result["released"]:
            self._leases.pop(lease.trajectory_id, None)
        return result

    def discard_trajectory_preemption(self, trajectory_id: str) -> None:
        self._preemption_transition("discard", trajectory_id)

    @staticmethod
    def _validate_history(snapshot: TrajectorySnapshot, previous: TrajectorySnapshot) -> None:
        # The same append-only checks as the existing snapshot registry.
        for name in ("group_id", "member_index", "group_size", "environment_version",
                     "tool_version", "sampling_params", "limits", "tool_selection"):
            if getattr(snapshot, name) != getattr(previous, name):
                raise ValueError("trajectory membership/configuration changed across checkpoint")
        for name in ("prompt_ids", "response_mask", "response_logprobs", "messages"):
            before = previous.agent_state[name]
            if snapshot.agent_state[name][:len(before)] != before:
                raise ValueError("committed trajectory history cannot be rewritten")
        segments = previous.agent_state["extra_fields"]["trajectory_segments"]
        if (snapshot.agent_state["extra_fields"]["trajectory_segments"][:len(segments)] != segments
                or snapshot.completed_tool_batches[:len(previous.completed_tool_batches)]
                != previous.completed_tool_batches):
            raise ValueError("committed sampling provenance and tool results cannot be rewritten")

    def commit_trajectory_snapshot(
        self, lease: TrajectoryOwnerLease, snapshot: TrajectorySnapshot, ttl_seconds: float
    ) -> str:
        self._check_owner(lease)
        if snapshot.trajectory_id != lease.trajectory_id or snapshot.execution_generation != lease.generation:
            raise ValueError("snapshot does not belong to the current execution owner")
        if self._state["policy_publications"]["records"]:
            published = PublishedPolicyRecord.decode(
                self.get_published_policy(snapshot.policy_identity["policy_version"])
            )
            if asdict(published.identity) != snapshot.policy_identity:
                raise ValueError("snapshot source identity differs from actual policy publication")
            for segment in snapshot.agent_state["extra_fields"].get("trajectory_segments", []):
                record = PublishedPolicyRecord.decode(self.get_published_policy(segment["policy_version"]))
                supplied = segment.get("policy_identity")
                if supplied is None and segment["policy_version"] == snapshot.policy_identity["policy_version"]:
                    supplied = snapshot.policy_identity  # Existing single-policy snapshot remains exact.
                if supplied != asdict(record.identity):
                    raise ValueError("snapshot segment differs from the actual published policy identity")
        barrier = self._state["policy_publications"]["barrier"]
        if barrier is not None and any(row["trajectory_id"] == lease.trajectory_id
                                      for row in barrier["source_leases"]):
            self._check_policy_barrier_row(barrier)
            if snapshot.group_id != barrier["group_id"] or snapshot.policy_identity != barrier["identity"]:
                raise ValueError("snapshot differs from the bounded handoff policy cut")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("invalid snapshot TTL")
        payload = snapshot.encode()
        previous_ref = self._state["trajectories"][lease.trajectory_id].get("latest_ref")
        if previous_ref is not None:
            if self._read_object(previous_ref) == payload:
                return previous_ref
            self._validate_history(snapshot, TrajectorySnapshot.decode(self._read_object(previous_ref)))
            if self._reference_is_pinned(previous_ref):
                raise ValueError("a committed handoff snapshot cannot be replaced")
        def commit(draft):
            tracker = self._tracker(draft)
            if tracker.state(lease.trajectory_id) is not PreemptState.SAFE_POINT_REACHED:
                raise ValueError("snapshot commit requires a reached safe point")
            reference = self._put_object(draft, "inflight", payload)
            self._check_owner(lease)
            row = draft["trajectories"][lease.trajectory_id]
            row.update(latest_ref=reference, snapshot_expires_at_unix=self.wall_clock() + ttl_seconds)
            tracker.mark_snapshot_committed(lease.trajectory_id, reference)
            self._save_tracker(draft, tracker)
            if previous_ref is not None:
                draft["gc_pending"].append(previous_ref)
            return reference
        # Pointer and tracker marker commit atomically, removing the old gap
        # between PUT success and the separate snapshot-committed marker RPC.
        return self._transaction(commit)

    def load_trajectory_snapshot(self, lease: TrajectoryOwnerLease) -> TrajectorySnapshot:
        self._check_owner(lease)
        row = self._state["trajectories"][lease.trajectory_id]
        if self.wall_clock() >= row.get("snapshot_expires_at_unix", 0):
            raise ValueError("trajectory snapshot expired")
        return TrajectorySnapshot.decode(self._read_object(row["latest_ref"]))

    def put_handoff_member_record(self, payload: bytes) -> str:
        json.loads(payload)
        return self._transaction(lambda draft: self._put_object(draft, "handoff-member", payload))

    def get_handoff_member_record(self, reference: str) -> bytes:
        self._require_open()
        return self._read_object(reference)

    @staticmethod
    def _manifest(row: dict) -> HandoffManifest:
        return HandoffManifest.decode(canonical_bytes(row["manifest"]))

    def _registry(self, state: dict) -> HandoffManifestRegistry:
        registry = HandoffManifestRegistry(
            self._tracker(state), ProducerGenerationRegistry(state["producers"]).get_current_producer_generation,
            clock=self.clock,
        )
        for handoff_id, row in state["handoffs"].items():
            registry.records[handoff_id] = HandoffRecord(
                manifest=self._manifest(row), state=HandoffState(row["state"]),
                prepared_at=row["prepared_at"], committed_at=row["committed_at"],
                source_released_at=row["source_released_at"], restoring_started_at=row["restoring_started_at"],
                finished_at=row["finished_at"], published_artifact_id=row["published_artifact_id"],
                restored_member_refs={int(i): ref for i, ref in row["restored_member_refs"].items()},
                restored_member_metrics={int(i): values for i, values in row["restored_member_metrics"].items()},
            )
        return registry

    @staticmethod
    def _save_registry(draft: dict, registry: HandoffManifestRegistry) -> None:
        for handoff_id, record in registry.records.items():
            publication = draft["handoffs"].get(handoff_id, {}).get("publication")
            draft["handoffs"][handoff_id] = {
                "manifest": asdict(record.manifest), "state": record.state.value,
                "prepared_at": record.prepared_at, "committed_at": record.committed_at,
                "source_released_at": record.source_released_at,
                "restoring_started_at": record.restoring_started_at, "finished_at": record.finished_at,
                "restored_member_refs": {str(i): ref for i, ref in record.restored_member_refs.items()},
                "restored_member_metrics": {str(i): values for i, values in record.restored_member_metrics.items()},
                "published_artifact_id": record.published_artifact_id, "publication": publication,
            }

    def _verify_manifest_objects(self, state: dict, manifest: HandoffManifest, *, enforce_retention: bool) -> None:
        for member in manifest.checkpointed_trajectories:
            row = state["trajectories"].get(member["trajectory_id"])
            if row is None or row.get("latest_ref") != member["snapshot_ref"]:
                raise ValueError("handoff snapshot reference is not current")
            snapshot = TrajectorySnapshot.decode(self._read_object(member["snapshot_ref"]))
            if (snapshot.trajectory_id != member["trajectory_id"] or snapshot.group_id != member["group_id"]
                    or snapshot.member_index != member["member_index"] or snapshot.group_size != member["group_size"]
                    or snapshot.execution_generation != member["expected_generation"]
                    or snapshot.policy_identity != manifest.policy_identity
                    or snapshot.environment_version != manifest.environment_version
                    or snapshot.tool_version != manifest.tool_version):
                raise ValueError("snapshot does not match handoff membership or identity")
            if enforce_retention and self.wall_clock() >= row["snapshot_expires_at_unix"]:
                raise ValueError("trajectory snapshot expired")
            if "input_ref" not in member:
                raise ValueError("durable handoff needs the original checkpointed member input_ref")
            self._read_object(member["input_ref"])
        for member in manifest.completed_members:
            self._read_object(member["record_ref"])
        for member in manifest.pending_samples:
            if "record_ref" not in member:
                raise ValueError("durable handoff needs the original pending member record_ref")
            self._read_object(member["record_ref"])

    def prepare_handoff_manifest(self, manifest: HandoffManifest) -> dict:
        manifest.validate()
        if len(manifest.member_tables()) != 1:
            raise ValueError("one durable handoff must contain exactly one GRPO prompt group")
        self._verify_manifest_objects(self._state, manifest, enforce_retention=True)
        self._check_manifest_policy_cut(manifest)
        def prepare(draft):
            registry = self._registry(draft)
            record = registry.prepare(manifest)
            self._save_registry(draft, registry)
            return {"handoff_id": manifest.handoff_id, "state": record.state.value}
        return self._transaction(prepare)

    def commit_handoff_manifest(self, handoff_id: str) -> str:
        def commit(draft):
            registry = self._registry(draft)
            record = registry.get(handoff_id)
            self._check_manifest_policy_cut(record.manifest)
            self._verify_manifest_objects(draft, record.manifest, enforce_retention=True)
            if record.state is not HandoffState.COMMITTED:
                record = registry.commit(handoff_id)
            self._save_registry(draft, registry)
            return record.state.value
        return self._transaction(commit)

    def get_handoff_manifest(self, handoff_id: str) -> HandoffManifest:
        self._require_open()
        return self._registry(self._state).get(handoff_id).manifest

    def _handoff_status(self, handoff_id: str, row: dict) -> dict:
        manifest = self._manifest(row)
        tables = manifest.member_tables()
        if len(tables) != 1:
            raise CoordinationStorageError("durable handoff must have exactly one prompt group")
        publication = row.get("publication")
        return {
            "handoff_id": handoff_id,
            "state": row["state"],
            "source_rollouter": manifest.source_rollouter,
            "source_released": row["source_released_at"] is not None,
            "source_released_at": row["source_released_at"],
            "group_uid": next(iter(tables)),
            "publication_state": publication.get("state") if publication is not None else None,
        }

    def get_handoff_status(self, handoff_id: str) -> dict:
        """Read control status without exposing owner leases or member payloads."""
        self._require_open()
        row = self._state["handoffs"].get(handoff_id)
        if row is None:
            raise ValueError(f"unknown handoff: {handoff_id}")
        return deepcopy(self._handoff_status(handoff_id, row))

    def list_recoverable_handoffs(self) -> list[dict]:
        """List committed source-released handoffs that can be resumed once."""
        self._require_open()
        rows = []
        for handoff_id, row in sorted(self._state["handoffs"].items()):
            if (row["state"] in {HandoffState.COMMITTED.value, HandoffState.RESTORING.value}
                    and row["source_released_at"] is not None):
                rows.append(self._handoff_status(handoff_id, row))
        return deepcopy(rows)

    def record_handoff_source_release(self, handoff_id: str) -> str:
        def release(draft):
            registry = self._registry(draft)
            record = registry._require_state(handoff_id, HandoffState.COMMITTED)
            for member in record.manifest.checkpointed_trajectories:
                lease = self._leases.get(member["trajectory_id"])
                if lease is not None and self.clock() < lease.expires_at:
                    raise ValueError("source execution leases must be released before recording source release")
            registry.record_source_release(handoff_id)
            self._save_registry(draft, registry)
            return record.state.value
        return self._transaction(release)

    def release_handoff_source(self, handoff_id: str) -> str:
        """Persist source release as one operation, then invalidate runtime rights."""
        released = []
        def release(draft):
            registry = self._registry(draft)
            record = registry._require_state(handoff_id, HandoffState.COMMITTED)
            for member in record.manifest.checkpointed_trajectories:
                trajectory_id = member["trajectory_id"]
                owner = draft["trajectories"][trajectory_id]["owner"]
                if owner["generation"] != member["expected_generation"]:
                    raise ValueError("source trajectory generation changed before release")
                draft["trajectories"][trajectory_id]["released"] = True
                if registry.tracker.state(trajectory_id) is not PreemptState.HANDED_OFF:
                    registry.tracker.mark_handed_off(trajectory_id)
                released.append(trajectory_id)
            registry.record_source_release(handoff_id)
            self._save_tracker(draft, registry.tracker)
            self._save_registry(draft, registry)
            return record.state.value
        status = self._transaction(release)
        for trajectory_id in released:
            self._leases.pop(trajectory_id, None)
        return status

    def begin_handoff_restore(self, handoff_id: str) -> str:
        def begin(draft):
            registry = self._registry(draft)
            record = registry.begin_restore(handoff_id)
            self._save_registry(draft, registry)
            return record.state.value
        return self._transaction(begin)

    def _member(self, handoff_id: str, trajectory_id: str, *, pending: bool = False) -> dict:
        record = self._registry(self._state)._require_state(handoff_id, HandoffState.RESTORING)
        members = record.manifest.pending_samples if pending else record.manifest.checkpointed_trajectories
        for member in members:
            if member["trajectory_id"] == trajectory_id:
                if member["member_index"] in record.restored_member_refs:
                    raise ValueError("handoff member already restored")
                return member
        raise ValueError("trajectory is not a member of this handoff")

    def _live_owner(self, trajectory_id: str) -> bool:
        lease = self._leases.get(trajectory_id)
        return lease is not None and self.clock() < lease.expires_at

    def acquire_handoff_trajectory(
        self, handoff_id: str, trajectory_id: str, owner_id: str, lease_seconds: float
    ) -> TrajectoryOwnerLease:
        member = self._member(handoff_id, trajectory_id)
        row = self._state["trajectories"][trajectory_id]
        if (row["owner"]["generation"] < member["expected_generation"] or self._live_owner(trajectory_id)
                or row.get("latest_ref") != member["snapshot_ref"]):
            raise ValueError("takeover requires committed snapshot and expired or released matching owner")
        if self.wall_clock() >= row["snapshot_expires_at_unix"]:
            raise ValueError("trajectory snapshot expired")
        lease = self._transaction(lambda draft: self._assign(draft, trajectory_id, owner_id, lease_seconds))
        self._leases[trajectory_id] = lease
        return lease

    def acquire_handoff_pending(
        self, handoff_id: str, trajectory_id: str, owner_id: str, lease_seconds: float
    ) -> TrajectoryOwnerLease:
        self._member(handoff_id, trajectory_id, pending=True)
        row = self._state["trajectories"].get(trajectory_id, {})
        if self._live_owner(trajectory_id) or row.get("latest_ref") is not None:
            raise ValueError("pending trajectory has a live owner or checkpoint; blind restart is forbidden")
        lease = self._transaction(lambda draft: self._assign(draft, trajectory_id, owner_id, lease_seconds))
        self._leases[trajectory_id] = lease
        return lease

    def record_handoff_member_result(
        self, handoff_id: str, member_index: int, owner_lease: TrajectoryOwnerLease, payload: bytes, metrics: dict
    ) -> str:
        self._check_owner(owner_lease)
        json.loads(payload)
        canonical_bytes(metrics)
        record = self._registry(self._state)._require_state(handoff_id, HandoffState.RESTORING)
        all_members = record.manifest.checkpointed_trajectories + record.manifest.pending_samples
        if not any(m["trajectory_id"] == owner_lease.trajectory_id and m["member_index"] == member_index
                   for m in all_members):
            raise ValueError("restored member index does not match its owner")
        previous = record.restored_member_refs.get(member_index)
        if previous is not None:
            if self._read_object(previous) != payload:
                raise ValueError("restored handoff member cannot be rewritten")
            return previous
        def complete(draft):
            registry = self._registry(draft)
            record = registry._require_state(handoff_id, HandoffState.RESTORING)
            reference = self._put_object(draft, "handoff-member", payload)
            self._check_owner(owner_lease)
            record.restored_member_refs[member_index] = reference
            record.restored_member_metrics[member_index] = deepcopy(metrics)
            self._save_registry(draft, registry)
            return reference
        return self._transaction(complete)

    def get_handoff_restored_members(self, handoff_id: str) -> dict[int, dict]:
        self._require_open()
        record = self._registry(self._state).get(handoff_id)
        return {
            i: {"record_ref": ref, "metrics": deepcopy(record.restored_member_metrics[i])}
            for i, ref in record.restored_member_refs.items()
        }

    @staticmethod
    def _validate_binding(binding: dict) -> None:
        if (not isinstance(binding, dict) or set(binding) != {"partition_id", "keys", "content_digest"}
                or not isinstance(binding["partition_id"], str) or not binding["partition_id"]
                or not isinstance(binding["keys"], list) or not binding["keys"]
                or len(set(binding["keys"])) != len(binding["keys"])
                or any(not isinstance(key, str) or not key for key in binding["keys"])
                or not isinstance(binding["content_digest"], str)
                or re.fullmatch(r"[0-9a-f]{64}", binding["content_digest"]) is None):
            raise ValueError("publication binding requires stable queue keys, partition and content digest")

    def prepare_publication(self, handoff_id: str, stable_binding: dict) -> dict:
        """Record exact queue identity before any write; this does not publish."""
        self._validate_binding(stable_binding)
        def prepare(draft):
            registry = self._registry(draft)
            record = registry._require_state(handoff_id, HandoffState.RESTORING)
            required = {m["member_index"] for m in record.manifest.checkpointed_trajectories
                        + record.manifest.pending_samples}
            if not required <= set(record.restored_member_refs):
                raise ValueError("cannot publish before all handoff members finish")
            group_size = next(iter(record.manifest.member_tables().values()))["group_size"]
            if len(stable_binding["keys"]) != group_size:
                raise ValueError("this handoff requires one verified final output key per group member")
            row = draft["handoffs"][handoff_id]
            publication_id = hashlib.sha256(canonical_bytes({
                "run_id": self.run_id, "handoff_id": handoff_id, "binding": stable_binding,
            })).hexdigest()
            existing = row.get("publication")
            if existing is not None:
                if existing["publication_id"] != publication_id or existing["binding"] != stable_binding:
                    raise ValueError("handoff publication identity cannot be rewritten")
                return deepcopy(existing)
            row["publication"] = {
                "publication_id": publication_id, "binding": deepcopy(stable_binding),
                "state": "PREPARED", "receipt": None,
            }
            return deepcopy(row["publication"])
        return self._transaction(prepare)

    def mark_publication_started(self, handoff_id: str) -> dict:
        """Durably mark the uncertain window BEFORE calling TransferQueue."""
        def started(draft):
            publication = draft["handoffs"][handoff_id].get("publication")
            if publication is None or publication["state"] != "PREPARED":
                raise PublicationBlocked("publication was already dispatched; query the real queue")
            publication["state"] = "DISPATCHED"
            return deepcopy(publication)
        return self._transaction(started)

    def confirm_publication(self, handoff_id: str, receipt: dict) -> dict:
        """Accept only a receipt obtained by reading and verifying real queue data.

        The V1 adapter must read all bound keys, validate training identities
        and compare the actual fields with the frozen content digest. A put
        return value or a matching cache tag alone is not such a receipt.
        """
        def confirm(draft):
            row = draft["handoffs"][handoff_id]
            publication = row.get("publication")
            if (publication is None or publication["state"] not in
                    {"DISPATCHED", "BLOCKED", "NEEDS_VERIFICATION", "PUBLISHED"}):
                raise PublicationBlocked("no dispatched publication can be confirmed")
            if (not isinstance(receipt, dict)
                    or set(receipt) != {"publication_id", "binding", "backend_reference"}
                    or receipt["publication_id"] != publication["publication_id"]
                    or receipt["binding"] != publication["binding"]
                    or not isinstance(receipt["backend_reference"], dict) or not receipt["backend_reference"]):
                raise ValueError("actual queue receipt does not match the stable publication binding")
            canonical_bytes(receipt)
            publication.update(state="PUBLISHED", receipt=deepcopy(receipt))
            return deepcopy(publication)
        return self._transaction(confirm)

    def reconcile_publication(self, handoff_id: str, receipt: dict | None) -> dict:
        if receipt is not None:
            return self.confirm_publication(handoff_id, receipt)
        def unresolved(draft):
            row = draft["handoffs"][handoff_id]
            if row["state"] == HandoffState.FINISHED.value:
                raise PublicationBlocked("finished publication loss is outside coordinator-only recovery")
            publication = row.get("publication")
            if publication is None:
                raise ValueError("no publication intent")
            if publication["state"] != "PREPARED":
                publication["state"] = "BLOCKED"
            return deepcopy(publication)
        return self._transaction(unresolved)

    def get_publication(self, handoff_id: str) -> dict | None:
        self._require_open()
        return deepcopy(self._state["handoffs"][handoff_id].get("publication"))

    def finish_handoff_manifest(self, handoff_id: str, publication_id: str) -> str:
        def finish(draft):
            registry = self._registry(draft)
            record = registry.get(handoff_id)
            publication = draft["handoffs"][handoff_id].get("publication")
            if (publication is None or publication["state"] != "PUBLISHED"
                    or publication["publication_id"] != publication_id):
                raise PublicationBlocked("finish requires verified complete-group queue publication")
            if record.state is HandoffState.FINISHED:
                return record.state.value
            record = registry.finish(handoff_id, publication_id)
            references = set(record.restored_member_refs.values())
            for member in record.manifest.completed_members + record.manifest.pending_samples:
                references.add(member["record_ref"])
            for member in record.manifest.checkpointed_trajectories:
                references.update((member["snapshot_ref"], member["input_ref"]))
            for member in (record.manifest.checkpointed_trajectories + record.manifest.pending_samples):
                row = draft["trajectories"].get(member["trajectory_id"])
                if row is not None:
                    row.pop("latest_ref", None)
                    row.pop("snapshot_expires_at_unix", None)
                    row["released"] = True
                draft["preemptions"].pop(member["trajectory_id"], None)
            draft["gc_pending"] = sorted(set(draft["gc_pending"]) | references)
            self._save_registry(draft, registry)
            return record.state.value
        status = self._transaction(finish)
        manifest = self.get_handoff_manifest(handoff_id)
        for member in manifest.checkpointed_trajectories + manifest.pending_samples:
            self._leases.pop(member["trajectory_id"], None)
        # Deliberately separate from collect_garbage: FINISHED is durable
        # before any object is deleted; cleanup failure cannot undo delivery.
        return status

    def _reference_is_pinned(self, reference: str) -> bool:
        for row in self._state["handoffs"].values():
            if row["state"] == HandoffState.FINISHED.value:
                continue
            manifest = self._manifest(row)
            all_refs = set(row["restored_member_refs"].values())
            for member in manifest.checkpointed_trajectories:
                all_refs.update((member["snapshot_ref"], member["input_ref"]))
            for member in manifest.completed_members + manifest.pending_samples:
                all_refs.add(member["record_ref"])
            if reference in all_refs:
                return True
        return False

    def queue_completed_group_cleanup(self, uid: str, owner_id: str, source_generation: int,
                                      input_record_refs: dict[int, str],
                                      output_record_refs: dict[int, str]) -> int:
        """Queue one normal n=8 source after its exact queue publication.

        This API never deletes objects. It re-reads the original member-record
        encoding and accepts all sixteen records as one atomic cleanup batch.
        """
        self._require_open()
        if (not isinstance(uid, str) or not uid or not isinstance(owner_id, str) or not owner_id
                or type(source_generation) is not int or source_generation < 0):
            raise ValueError("invalid normal trajectory cleanup identity")
        expected = set(range(8))
        records = {}
        for kind, references in (("input", input_record_refs), ("output", output_record_refs)):
            if (not isinstance(references, dict) or set(references) != expected
                    or any(not isinstance(reference, str) or not reference for reference in references.values())):
                raise ValueError("normal cleanup requires exactly eight input and eight output records")
            records[kind] = references
        references = set(records["input"].values()) | set(records["output"].values())
        if len(references) != 16:
            raise ValueError("normal cleanup member records must be distinct")
        if (ProducerGenerationRegistry(self._state["producers"]).get_current_producer_generation(owner_id)
                != source_generation):
            raise ValueError("normal cleanup source producer is no longer current")

        # A previous queue RPC may have committed before its reply was lost.
        # An idempotent retry must not read objects already deleted by GC.
        for index in range(8):
            row = self._state["trajectories"].get(f"{uid}/member-{index}")
            if (row is None or row.get("released") is not True
                    or row.get("owner", {}).get("owner_id") != owner_id):
                raise ValueError("normal cleanup member owner is not released by this source")
        if not (references & self._state["objects"].keys()):
            return 0
        if references <= set(self._state["gc_pending"]):
            return 0

        decoded = {}
        for kind in ("input", "output"):
            for index in range(8):
                reference = records[kind][index]
                trajectory_id = f"{uid}/member-{index}"
                expected_record = {
                    "trajectory_runtime_record": 1, "kind": kind, "uid": uid,
                    "member_index": index, "group_size": 8, "trajectory_id": trajectory_id,
                }
                value = decode_handoff_member_record(self._read_object(reference))
                if (not isinstance(value, dict) or any(value.get(key) != item for key, item in expected_record.items())
                        or not isinstance(value.get("member_kwargs"), dict)
                        or (kind == "output" and "output" not in value)):
                    raise ValueError("normal cleanup member record identity or kind differs")
                decoded[(kind, index)] = value
                row = self._state["trajectories"].get(trajectory_id)
                if (row is None or row.get("released") is not True
                        or row.get("owner", {}).get("owner_id") != owner_id):
                    raise ValueError("normal cleanup member owner is not released by this source")
        for index in range(8):
            for kind in ("input", "output"):
                kwargs = decoded[(kind, index)]["member_kwargs"]
                if kwargs.get("uid") != uid or kwargs.get("session_id") != index:
                    raise ValueError("normal cleanup input and output member binding differs")
        if any(self._reference_is_pinned(reference) for reference in references):
            raise ValueError("normal cleanup record is pinned by an unfinished handoff")

        def queue(draft):
            before = set(draft["gc_pending"])
            draft["gc_pending"] = sorted(before | references)
            return len((before | references) - before)

        return self._transaction(queue)

    def collect_garbage(self) -> int:
        """Retry only explicitly recorded cleanup; never scan historical evidence."""
        self._require_open()
        cleanable = {
            reference for reference in self._state["gc_pending"]
            if not self._reference_is_pinned(reference)
        }
        if not cleanable:
            return 0

        objects = self._state["objects"]
        selected_keys = {
            objects[reference]["storage_key"]
            for reference in cleanable
            if reference in objects
        }
        references_by_key: dict[str, set[str]] = {}
        for reference, descriptor in objects.items():
            references_by_key.setdefault(descriptor["storage_key"], set()).add(reference)
        # An immutable object may be aliased by several logical references.
        # Delete its bytes only when every alias is in this cleanable batch.
        deletable_keys = sorted(
            key for key in selected_keys if references_by_key[key] <= cleanable
        )
        for storage_key in deletable_keys:
            self._objects.delete(storage_key)

        def remove(draft):
            for reference in cleanable:
                draft["objects"].pop(reference, None)
            draft["gc_pending"] = [
                reference for reference in draft["gc_pending"] if reference not in cleanable
            ]

        self._transaction(remove)
        return len(cleanable)

    def get_trajectory_preemption_metrics(self) -> dict:
        self._require_open()
        metrics = self._tracker(self._state).metrics()
        if self._recovered:
            for row in metrics["trajectories"]:
                for key in ("safe_point_wait_ms", "snapshot_commit_ms", "worker_reclaim_latency_ms"):
                    row[key] = None
        return metrics

    def get_handoff_metrics(self) -> dict:
        self._require_open()
        metrics = self._registry(self._state).metrics()
        if self._recovered:
            for row in metrics["handoffs"]:
                for key in ("prepare_to_commit_ms", "worker_reclaim_latency_ms",
                            "commit_to_restore_ms", "restore_to_finish_ms"):
                    row[key] = None
        return metrics

    def begin_policy_publication(self, identity: dict, model_configuration_fingerprint: str,
                                 publisher_id: str, engine_urls: list[str]) -> PolicyPublicationTicket:
        from .identity import _decode_policy_identity

        canonical = _decode_policy_identity(identity)
        version = policy_version(canonical.policy_version)
        nonempty(model_configuration_fingerprint, "model_configuration_fingerprint")
        nonempty(publisher_id, "publisher_id")
        if (not isinstance(engine_urls, list) or not engine_urls or len(set(engine_urls)) != len(engine_urls)
                or any(not isinstance(url, str) or not url for url in engine_urls)):
            raise ValueError("publication needs all distinct actual engine URLs")
        ticket = PolicyPublicationTicket(self.run_id, uuid4().hex, version)

        def begin(draft):
            policy = draft["policy_publications"]
            if policy["publication"] is not None or policy["barrier"] is not None:
                raise PolicyPublicationBlocked("another publication or handoff cut is unresolved")
            if version in policy["records"]:
                raise PolicyPublicationBlocked("this version was already published; do not repeat a weight update")
            if policy["active_version"] is not None:
                previous = PublishedPolicyRecord.decode(policy["records"][policy["active_version"]])
                if int(version) <= int(previous.identity.policy_version):
                    raise ValueError("published training steps cannot rewind in the same run")
                if (previous.model_configuration_fingerprint != model_configuration_fingerprint
                        or previous.identity.tokenizer_template_fingerprint != canonical.tokenizer_template_fingerprint
                        or previous.identity.kv_cache_schema_fingerprint != canonical.kv_cache_schema_fingerprint):
                    raise ValueError("model/tokenizer/KV configuration changed within the publication run")
            policy["publication"] = {
                "operation_id": ticket.operation_id, "version": version, "identity": asdict(canonical),
                "model_configuration_fingerprint": model_configuration_fingerprint, "publisher_id": publisher_id,
                "engine_urls": list(engine_urls), "state": "PREPARED", "failure": None,
            }
            return ticket
        return self._transaction(begin)

    def _policy_publication_row(self, state: dict, ticket: PolicyPublicationTicket) -> dict:
        row = state["policy_publications"]["publication"]
        if (not isinstance(ticket, PolicyPublicationTicket) or ticket.run_id != self.run_id or row is None
                or row["operation_id"] != ticket.operation_id or row["version"] != ticket.policy_version):
            raise PolicyPublicationBlocked("policy publication operation is no longer current")
        return row

    def mark_policy_update_started(self, ticket: PolicyPublicationTicket) -> None:
        def mark(draft):
            row = self._policy_publication_row(draft, ticket)
            if row["state"] != "PREPARED":
                raise PolicyPublicationBlocked("policy mutation cannot be retried")
            row["state"] = "UPDATING"
        self._transaction(mark)

    def fail_policy_publication(self, ticket: PolicyPublicationTicket, reason: str) -> None:
        nonempty(reason, "failure reason")
        def fail(draft):
            row = self._policy_publication_row(draft, ticket)
            row["state"], row["failure"] = "BLOCKED", reason
        self._transaction(fail)

    def confirm_policy_publication(self, ticket: PolicyPublicationTicket, observations: list[dict]) -> dict:
        from .identity import _decode_policy_identity

        def confirm(draft):
            row = self._policy_publication_row(draft, ticket)
            if row["state"] != "UPDATING":
                raise PolicyPublicationBlocked("policy transfer did not complete in the current publication")
            actual = tuple(EnginePolicyObservation.decode(item) for item in observations)
            if {item.engine_url for item in actual} != set(row["engine_urls"]):
                raise ValueError("actual engine observations do not cover configured replicas exactly")
            record = PublishedPolicyRecord(
                self.run_id, _decode_policy_identity(row["identity"]), row["model_configuration_fingerprint"],
                row["publisher_id"], actual, self.wall_clock())
            policy = draft["policy_publications"]
            policy["records"][ticket.policy_version] = record.encode()
            policy["active_version"], policy["publication"] = ticket.policy_version, None
            return record.encode()
        return self._transaction(confirm)

    def get_published_policy(self, version: int | str) -> dict:
        self._require_open()
        record = self._state["policy_publications"]["records"].get(policy_version(version))
        if record is None:
            raise PolicyPublicationBlocked("generation used an unknown or unsuccessful policy publication")
        return deepcopy(record)

    def get_active_published_policy(self) -> dict:
        self._require_open()
        policy = self._state["policy_publications"]
        if policy["publication"] is not None or policy["active_version"] is None:
            raise PolicyPublicationBlocked("no settled active policy publication is available")
        return deepcopy(policy["records"][policy["active_version"]])

    def authorize_same_version_engines(self, version, engine_urls: list[str], observations: list[dict]) -> dict:
        """Admit copied replicas of the settled version without updating its weights.

        The caller owns the native transfer into isolated engines. This method
        accepts actual model_info readbacks only, and never grants permission
        to update model tensors or to publish a second training version.
        """
        version = policy_version(version)
        if (not isinstance(engine_urls, list) or not engine_urls or len(set(engine_urls)) != len(engine_urls)
                or any(not isinstance(url, str) or not url for url in engine_urls)):
            raise ValueError("engine membership needs distinct configured URLs")
        actual = tuple(EnginePolicyObservation.decode(item) for item in observations)
        if (len(actual) != len(engine_urls) or {item.engine_url for item in actual} != set(engine_urls)
                or any(item.weight_version != version for item in actual)):
            raise ValueError("engine membership readback does not cover the settled version")

        def authorize(draft):
            policy = draft["policy_publications"]
            if (policy["active_version"] != version or policy["publication"] is not None
                    or policy["barrier"] is not None):
                raise PolicyPublicationBlocked("engine membership requires an idle settled policy version")
            previous = PublishedPolicyRecord.decode(policy["records"][version])
            merged = {item.engine_url: item for item in previous.engine_observations}
            merged.update({item.engine_url: item for item in actual})
            record = PublishedPolicyRecord(
                previous.run_id, previous.identity, previous.model_configuration_fingerprint,
                previous.publisher_id, tuple(merged[url] for url in sorted(merged)), previous.committed_at_unix,
            )
            policy["records"][version] = record.encode()
            return record.encode()

        return self._transaction(authorize)

    def get_policy_coordination_state(self) -> dict:
        """Read unresolved intents after an uncertain return; never replay them."""
        self._require_open()
        return deepcopy(self._state["policy_publications"])

    def begin_handoff_policy_barrier(self, group_id: str, source_leases: list[TrajectoryOwnerLease],
                                     timeout_seconds: float) -> HandoffPublicationBarrier:
        nonempty(group_id, "group_id")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("handoff publication timeout must be finite and positive")
        if (not source_leases or len({lease.trajectory_id for lease in source_leases}) != len(source_leases)
                or len({lease.owner_id for lease in source_leases}) != 1):
            raise ValueError("a handoff policy cut needs distinct trajectories with one real source owner")
        for lease in source_leases:
            self._check_owner(lease)
        record = PublishedPolicyRecord.decode(self.get_active_published_policy())
        barrier = HandoffPublicationBarrier(self.run_id, uuid4().hex, group_id, record.identity)

        def begin(draft):
            policy = draft["policy_publications"]
            if policy["publication"] is not None or policy["barrier"] is not None:
                raise PolicyPublicationBlocked("another publication or handoff cut is unresolved")
            policy["barrier"] = {
                "operation_id": barrier.operation_id, "group_id": group_id, "identity": asdict(record.identity),
                "source_leases": [{key: value for key, value in asdict(lease).items() if key != "expires_at"}
                                  for lease in source_leases],
                "deadline_unix": self.wall_clock() + timeout_seconds, "state": "ACTIVE",
            }
            return barrier
        result = self._transaction(begin)
        self._policy_barrier_deadlines[barrier.operation_id] = self.clock() + timeout_seconds
        return result

    def _check_policy_barrier_row(self, row: dict) -> None:
        deadline = self._policy_barrier_deadlines.get(row["operation_id"])
        if row["state"] != "ACTIVE" or deadline is None or self.clock() >= deadline:
            raise PolicyPublicationBlocked("handoff policy cut expired or lost its live controller")

    def _policy_barrier_row(self, state: dict, barrier: HandoffPublicationBarrier) -> dict:
        row = state["policy_publications"]["barrier"]
        if (not isinstance(barrier, HandoffPublicationBarrier) or barrier.run_id != self.run_id or row is None
                or row["operation_id"] != barrier.operation_id or row["group_id"] != barrier.group_id
                or row["identity"] != asdict(barrier.identity)):
            raise PolicyPublicationBlocked("handoff policy cut is no longer current")
        return row

    def check_handoff_policy_barrier(self, barrier: HandoffPublicationBarrier) -> dict:
        self._require_open()
        row = self._policy_barrier_row(self._state, barrier)
        self._check_policy_barrier_row(row)
        return deepcopy(row["identity"])

    def _check_manifest_policy_cut(self, manifest: HandoffManifest) -> None:
        row = self._state["policy_publications"]["barrier"]
        if row is None or row["group_id"] not in manifest.member_tables():
            return
        self._check_policy_barrier_row(row)
        if manifest.policy_identity != row["identity"]:
            raise ValueError("Manifest policy identity differs from the source cut")
        source = {lease["trajectory_id"]: lease for lease in row["source_leases"]}
        members = {member["trajectory_id"] for member in (*manifest.completed_members,
                                                         *manifest.checkpointed_trajectories)}
        for member in manifest.pending_samples:
            trajectory_id = member["trajectory_id"]
            if trajectory_id not in source:
                continue  # A member never admitted on the source has no cut lease.
            cut = source[trajectory_id]
            current = self._state["trajectories"].get(trajectory_id, {})
            expected_owner = {key: cut[key] for key in ("owner_id", "generation", "lease_id")}
            # A pending member retains its immutable original input, not a
            # completed sample. Never erase a snapshot or another generation.
            if (current.get("owner") != expected_owner or current.get("released") is not True
                    or self._live_owner(trajectory_id) or current.get("latest_ref") is not None):
                raise ValueError("Pending cut member still owns execution or retained progress")
            record = decode_handoff_member_record(self._read_object(member["record_ref"]))
            expected = {"trajectory_runtime_record": 1, "kind": "input", "uid": row["group_id"],
                        "trajectory_id": trajectory_id, "member_index": member["member_index"],
                        "group_size": member["group_size"]}
            if (any(record.get(key) != value for key, value in expected.items())
                    or not isinstance(record.get("member_kwargs"), dict)):
                raise ValueError("Pending cut member does not retain its original input")
            cancelled = record.get("cancelled_native_generation")
            if cancelled is not None:
                ranks = cancelled.get("ranks", [])
                if (cancelled.get("status") != "PASS" or not ranks or any(
                        rank.get("source_disposition") != "PENDING_CANCELLED"
                        or rank.get("generated_tokens") != 0 or rank.get("kv_allocated") is not False
                        or rank.get("pending_removed") is not True
                        or any(rank.get("source_binding", {}).get(key) != value for key, value in {
                            "trajectory_id": trajectory_id, "source_owner_id": cut["owner_id"],
                            "source_execution_generation": cut["generation"], "source_lease_id": cut["lease_id"],
                        }.items()) for rank in ranks)):
                    raise ValueError("Pending cut member has no exact zero-output cancellation")
            members.add(trajectory_id)
        if not set(source) <= members:
            raise ValueError("Manifest does not retain every source trajectory in the policy cut")

    def release_handoff_policy_barrier(self, barrier: HandoffPublicationBarrier,
                                       handoff_id: str | None = None) -> None:
        def release(draft):
            row = self._policy_barrier_row(draft, barrier)
            for source in row["source_leases"]:
                lease = self._leases.get(source["trajectory_id"])
                if (lease is not None and lease.lease_id == source["lease_id"]
                        and self.clock() < lease.expires_at):
                    raise PolicyPublicationBlocked(
                        "source execution rights must be released before reopening publication"
                    )
            if handoff_id is not None:
                record = self._registry(draft).get(handoff_id)
                if (record.state not in {HandoffState.COMMITTED, HandoffState.RESTORING, HandoffState.FINISHED}
                        or record.source_released_at is None or record.manifest.policy_identity != row["identity"]
                        or row["group_id"] not in record.manifest.member_tables()):
                    raise ValueError("handoff cut requires its committed Manifest and actual source release")
            # With no Manifest this is explicit failure cleanup, never a commit.
            # Old rights must already be invalid. An elapsed timer alone cannot
            # let a concurrent parameter writer pass this barrier.
            draft["policy_publications"]["barrier"] = None
        self._transaction(release)
        self._policy_barrier_deadlines.pop(barrier.operation_id, None)

    def summary(self) -> dict:
        self._require_open()
        return {
            "run_id": self.run_id, "sequence": self._state["sequence"],
            "root_dir": str(self.root.parent.resolve()),
            "active_execution_leases": sum(self._live_owner(t) for t in self._leases),
            "handoffs": {key: row["state"] for key, row in self._state["handoffs"].items()},
            "unsettled_publications": [key for key, row in self._state["handoffs"].items()
                if row.get("publication") is not None and row["publication"]["state"] != "PUBLISHED"],
            "gc_pending": len(self._state["gc_pending"]),
            "active_policy_version": self._state["policy_publications"]["active_version"],
            "policy_publication_blocked": self._state["policy_publications"]["publication"] is not None,
            "handoff_policy_barrier": self._state["policy_publications"]["barrier"] is not None,
        }

    def export_checkpoint(self, destination: str, checkpoint_step: int) -> dict:
        from ..trajectory_checkpoint import export_trajectory_checkpoint

        return export_trajectory_checkpoint(self, destination, checkpoint_step)

    def close(self) -> None:
        self._closed = True
        self._leases.clear()
        self._metadata.close()
