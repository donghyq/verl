"""Committed text-only tool-boundary state, separate from trainable artifacts.

The metadata owner and shared payload store must survive a compute-worker exit.
This module does not provide metadata-owner election or storage durability.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(canonical_bytes(tokens)).hexdigest()


@dataclass(frozen=True)
class TrajectoryOwnerLease:
    trajectory_id: str
    owner_id: str
    generation: int
    lease_id: str
    expires_at: float


@dataclass(frozen=True)
class TrajectorySnapshot:
    trajectory_id: str
    group_id: str
    member_index: int
    group_size: int
    execution_generation: int
    agent_state: dict[str, Any]
    sampling_params: dict[str, Any]
    limits: dict[str, Any]
    environment_version: str
    tool_version: str
    policy_identity: dict[str, str]
    tool_selection: list[str]
    completed_tool_batches: list[dict[str, Any]]
    kv_ref: dict[str, Any] | None = None
    schema_version: int = 1
    next_stage: str = "generating"

    def validate(self) -> None:
        from verl.experimental.fully_async_policy.rollout_artifact import _decode_policy_identity

        _decode_policy_identity(self.policy_identity).validate()
        if self.schema_version != 1 or self.next_stage != "generating":
            raise ValueError("unsupported trajectory snapshot stage/schema")
        if not all(isinstance(x, str) and x for x in (
            self.trajectory_id, self.group_id, self.environment_version, self.tool_version
        )):
            raise ValueError("snapshot identity/version is missing")
        if (type(self.group_size) is not int or self.group_size <= 0
                or type(self.member_index) is not int or not 0 <= self.member_index < self.group_size
                or type(self.execution_generation) is not int or self.execution_generation < 0):
            raise ValueError("invalid trajectory membership/generation")
        state = self.agent_state
        expected = {"messages", "prompt_ids", "response_ids", "response_mask", "response_logprobs",
            "turn_scores", "tool_rewards", "user_turns", "assistant_turns", "extra_fields",
            "tools_kwargs", "metrics", "request_id", "mm_processor_kwargs"}
        if set(state) != expected or not all(
            isinstance(message, dict) and isinstance(message.get("content"), str)
            for message in state["messages"]
        ):
            raise ValueError("snapshot must contain only the supported text AgentData fields")
        tokens, mask, logprobs = state["prompt_ids"], state["response_mask"], state["response_logprobs"]
        if not tokens or any(type(x) is not int or x < 0 for x in tokens):
            raise ValueError("snapshot must contain actual token history")
        if len(mask) >= len(tokens) or any(type(x) is not int or x not in (0, 1) for x in mask):
            raise ValueError("invalid response mask")
        if len(logprobs) != len(mask) or any(not math.isfinite(x) for x in logprobs):
            raise ValueError("snapshot logprob/mask mismatch")
        if any(logprob != 0 for logprob, bit in zip(logprobs, mask) if bit == 0):
            raise ValueError("tool/context tokens cannot carry behavior logprobs")
        generated = sum(mask)
        cursor = 0
        for segment in state["extra_fields"].get("trajectory_segments", []):
            if segment["start"] != cursor or segment["end"] <= cursor:
                raise ValueError("snapshot sampling segments are not contiguous")
            if (not isinstance(segment.get("policy_version"), str) or not segment["policy_version"]
                    or not segment.get("sampling_config_fingerprint")):
                raise ValueError("snapshot segment sampling provenance is missing")
            cursor = segment["end"]
        if cursor != generated:
            raise ValueError("snapshot sampling segments do not cover generated tokens")
        for segment in state["extra_fields"].get("trajectory_segments", []):
            generation = segment.get("execution_generation")
            if generation is not None and (type(generation) is not int
                    or not 0 <= generation <= self.execution_generation):
                raise ValueError("snapshot segment execution generation is invalid")
        if (not self.completed_tool_batches or any(
                not batch.get("calls") or len(batch["calls"]) != len(batch.get("results", []))
                for batch in self.completed_tool_batches)):
            raise ValueError("snapshot requires a completed tool batch")
        if self.kv_ref is not None:
            from verl.experimental.fully_async_policy.kv_export_proof import validate_export_reference

            validate_export_reference(self.kv_ref, token_ids=tokens, policy_identity=self.policy_identity)
        canonical_bytes(asdict(self))

    def encode(self) -> bytes:
        self.validate()
        return canonical_bytes(asdict(self))

    @classmethod
    def decode(cls, payload: bytes) -> "TrajectorySnapshot":
        snapshot = cls(**json.loads(payload))
        snapshot.validate()
        return snapshot


class TrajectorySnapshotRegistry:
    """Use a separate transactional artifact store under the same authority.

    Snapshot objects are never exposed by the Trainer's artifact claim methods.
    All methods run on the existing serialized metadata Actor. Ownership changes
    compare the current generation and require lease expiry or explicit release.
    """

    def __init__(self, store, clock=time.monotonic):
        self.store = store
        self.clock = clock
        self.owners: dict[str, TrajectoryOwnerLease] = {}
        self.latest: dict[str, str] = {}
        self.expiry: dict[str, float] = {}
        self.versions: dict[str, str] = {}

    def open(self, trajectory_id: str, owner_id: str, lease_seconds: float) -> TrajectoryOwnerLease:
        if trajectory_id in self.owners or trajectory_id in self.latest:
            raise ValueError("trajectory already registered; use compare-generation takeover")
        return self._assign(trajectory_id, owner_id, lease_seconds)

    def _assign(self, trajectory_id: str, owner_id: str, seconds: float) -> TrajectoryOwnerLease:
        if not trajectory_id or not owner_id or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid execution lease")
        generation = self.store.register_producer(trajectory_id, uuid4().hex)
        owner = TrajectoryOwnerLease(trajectory_id, owner_id, generation, uuid4().hex, self.clock() + seconds)
        self.owners[trajectory_id] = owner
        return owner

    def _check(self, lease: TrajectoryOwnerLease) -> None:
        if self.owners.get(lease.trajectory_id) != lease or self.clock() >= lease.expires_at:
            raise ValueError("stale or expired trajectory owner")

    def renew(self, lease: TrajectoryOwnerLease, seconds: float) -> TrajectoryOwnerLease:
        self._check(lease)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid lease renewal")
        new = TrajectoryOwnerLease(lease.trajectory_id, lease.owner_id, lease.generation,
            lease.lease_id, self.clock() + seconds)
        self.owners[lease.trajectory_id] = new
        return new

    def release_owner(self, lease: TrajectoryOwnerLease) -> None:
        self._check(lease)
        self.owners[lease.trajectory_id] = TrajectoryOwnerLease(lease.trajectory_id,
            lease.owner_id, lease.generation, lease.lease_id, self.clock())

    def takeover(self, trajectory_id: str, expected_generation: int, owner_id: str,
                 seconds: float) -> TrajectoryOwnerLease:
        current = self.owners.get(trajectory_id)
        if (current is None or current.generation != expected_generation
                or self.clock() < current.expires_at or trajectory_id not in self.latest):
            raise ValueError("takeover requires committed snapshot and expired matching owner")
        if self.clock() >= self.expiry[trajectory_id]:
            raise ValueError("trajectory snapshot expired")
        return self._assign(trajectory_id, owner_id, seconds)

    def commit(self, lease: TrajectoryOwnerLease, snapshot: TrajectorySnapshot, ttl_seconds: float) -> str:
        from verl.experimental.fully_async_policy.rollout_artifact import RolloutArtifactManifest
        from verl.experimental.fully_async_policy.rollout_artifact import _decode_policy_identity
        self._check(lease)
        if snapshot.trajectory_id != lease.trajectory_id or snapshot.execution_generation != lease.generation:
            raise ValueError("snapshot does not belong to the current execution owner")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("invalid snapshot TTL")
        payload = snapshot.encode()
        if lease.trajectory_id in self.latest:
            previous_snapshot = self.load(lease)
            for name in ("group_id", "member_index", "group_size", "environment_version",
                         "tool_version", "sampling_params", "limits", "tool_selection"):
                if getattr(snapshot, name) != getattr(previous_snapshot, name):
                    raise ValueError("trajectory membership/configuration changed across checkpoint")
            for name in ("prompt_ids", "response_mask", "response_logprobs", "messages"):
                previous_value = previous_snapshot.agent_state[name]
                if snapshot.agent_state[name][:len(previous_value)] != previous_value:
                    raise ValueError("committed trajectory history cannot be rewritten")
            previous_segments = previous_snapshot.agent_state["extra_fields"]["trajectory_segments"]
            if (snapshot.agent_state["extra_fields"]["trajectory_segments"][:len(previous_segments)]
                    != previous_segments or
                    snapshot.completed_tool_batches[:len(previous_snapshot.completed_tool_batches)]
                    != previous_snapshot.completed_tool_batches):
                raise ValueError("committed sampling provenance and tool results cannot be rewritten")
        key = "inflight/" + uuid4().hex
        manifest = RolloutArtifactManifest(artifact_id=key, producer_id=lease.trajectory_id,
            identity=_decode_policy_identity(snapshot.policy_identity), rollout_group_id=key,
            rollout_group_size=1, rollout_sample_index=0, trajectory_count=1,
            payload_digest=hashlib.sha256(payload).hexdigest())
        # Existing reserve -> payload PUT/readback -> commit boundary. Separate
        # store prevents an unfinished trajectory from entering Trainer claims.
        self.store.publish(manifest, payload)
        self.versions[key] = manifest.identity.policy_version
        try:
            self._check(lease)
        except ValueError:
            self._delete(key)
            raise
        previous = self.latest.get(lease.trajectory_id)
        self.latest[lease.trajectory_id] = key
        self.expiry[lease.trajectory_id] = self.clock() + ttl_seconds
        if previous is not None:
            self._delete(previous)
        return key

    def load(self, lease: TrajectoryOwnerLease) -> TrajectorySnapshot:
        self._check(lease)
        if self.clock() >= self.expiry[lease.trajectory_id]:
            raise ValueError("trajectory snapshot expired")
        key = self.latest[lease.trajectory_id]
        # Read using the normal data lease and digest checks, without rewriting
        # historical generation or sampling provenance on the stored payload.
        read = self.store.claim_group_for_artifact(key,
            accepted_policy_versions=set(self.versions.values()), lease_timeout_seconds=30)
        if read is None:
            raise ValueError("committed snapshot is unavailable")
        try:
            if hashlib.sha256(read.payloads[0]).hexdigest() != read.manifests[0].payload_digest:
                raise ValueError("trajectory snapshot payload digest mismatch")
            return TrajectorySnapshot.decode(read.payloads[0])
        finally:
            self.store.release(read)

    def _delete(self, key: str) -> None:
        read = self.store.claim_group_for_artifact(key,
            accepted_policy_versions=set(self.versions.values()), lease_timeout_seconds=30)
        if read is not None:
            self.store.acknowledge(read)
            self.versions.pop(key, None)

    def finish(self, lease: TrajectoryOwnerLease) -> None:
        self._check(lease)
        key = self.latest.pop(lease.trajectory_id, None)
        if key is not None:
            self._delete(key)
        self.expiry.pop(lease.trajectory_id, None)
        self.release_owner(lease)

    def collect_expired(self) -> int:
        count = 0
        for trajectory_id in list(self.latest):
            owner = self.owners[trajectory_id]
            if self.clock() >= self.expiry[trajectory_id] and self.clock() >= owner.expires_at:
                self._delete(self.latest.pop(trajectory_id))
                del self.expiry[trajectory_id]
                count += 1
        return count
