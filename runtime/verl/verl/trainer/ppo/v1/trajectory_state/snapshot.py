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

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

def _is_text_only_message(message: Any) -> bool:
    """Allow the original factory's null text beside structured tool calls."""
    if not isinstance(message, dict):
        return False
    if isinstance(message.get("content"), str):
        return True
    calls = message.get("tool_calls")
    return (message.get("role") == "assistant" and message.get("content") is None
            and isinstance(calls, list) and bool(calls) and all(
                isinstance(call, dict) and call.get("type") == "function"
                and isinstance(call.get("function"), dict) for call in calls))


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
        from .identity import _decode_policy_identity

        _decode_policy_identity(self.policy_identity).validate()
        inflight = self.schema_version == 2 and self.next_stage == "generating_inflight"
        if not inflight and (self.schema_version != 1 or self.next_stage != "generating"):
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
            _is_text_only_message(message)
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
        if ((not self.completed_tool_batches and not inflight) or any(
                not batch.get("calls") or len(batch["calls"]) != len(batch.get("results", []))
                for batch in self.completed_tool_batches)):
            raise ValueError("snapshot requires a completed tool batch")
        if self.kv_ref is not None:
            if self.kv_ref.get("kind") == "token_kv_committed_v1":
                from .token_kv_resume import validate_token_kv_reference

                if not inflight:
                    raise ValueError("token-KV reference requires a generation-interruption snapshot")
                validate_token_kv_reference(self.kv_ref, snapshot=self)
            else:
                from .kv_export_proof import validate_export_reference

                validate_export_reference(self.kv_ref, token_ids=tokens, policy_identity=self.policy_identity)
        if inflight and (self.kv_ref is None or self.kv_ref.get("kind") != "token_kv_committed_v1"):
            raise ValueError("generation-interruption snapshot requires a committed token-KV reference")
        if not inflight and "trajectory_inflight_generation" in state["extra_fields"]:
            raise ValueError("legacy snapshot cannot contain unfinished generation state")
        canonical_bytes(asdict(self))

    def encode(self) -> bytes:
        self.validate()
        return canonical_bytes(asdict(self))

    @classmethod
    def decode(cls, payload: bytes) -> "TrajectorySnapshot":
        snapshot = cls(**json.loads(payload))
        snapshot.validate()
        return snapshot
