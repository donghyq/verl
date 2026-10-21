# Copyright 2026 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Versioned, group-aware rollout artifacts for Fully Async training.

This module deliberately has no hard Ray, DataProto, or Mooncake dependency.  It
defines the artifact lifecycle contract and an in-process reference implementation;
the Mooncake backend imports Mooncake Store only when that backend is selected.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import operator
import os
import secrets
import socket
import time
from pathlib import Path
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Callable

from verl.experimental.fully_async_policy.transfer_admission import (
    TransferAdmissionConfig,
    TransferAdmissionController,
    TransferIntent,
    TransferRequest,
)

class RolloutArtifactStatus(StrEnum):
    WRITING = "writing"
    READY = "ready"
    LEASED = "leased"
    CONSUMED = "consumed"
    REJECTED = "rejected"



def _runtime_config_value(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(name, default)
        except (TypeError, AttributeError):
            pass
    return getattr(config, name, default)


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _config_file_fingerprint(path: Any, filename: str) -> str | None:
    """Fingerprint stable config content instead of a machine-local cache path."""
    if not isinstance(path, (str, Path)) or not path:
        return None
    config_path = Path(path) / filename
    try:
        if not config_path.is_file():
            return None
        return hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError:
        return None


def build_fully_async_policy_kv_identity(
    policy_version: int | str,
    *,
    model_config: Any,
    rollout_config: Any,
    tokenizer: Any | None = None,
) -> "CanonicalPolicyKVIdentity":
    """Build cross-replica KV identity from stable runtime-visible inputs.

    VeRL policy_version is the dynamic weight identity. The static model
    fingerprint identifies which model/runtime configuration that version
    belongs to. Replica and allocator generations are intentionally excluded.
    """

    version = str(policy_version)
    if not version:
        raise ValueError("policy_version must be non-empty")

    model_path = _runtime_config_value(model_config, "path")
    tokenizer_path = _runtime_config_value(model_config, "tokenizer_path", model_path)
    custom_chat_template = _runtime_config_value(model_config, "custom_chat_template")
    trust_remote_code = bool(_runtime_config_value(model_config, "trust_remote_code", False))

    tokenizer_obj = tokenizer or _runtime_config_value(model_config, "tokenizer")
    if tokenizer_obj is not None:
        tokenizer_fields = {
            "tokenizer_class": type(tokenizer_obj).__name__,
            "chat_template": custom_chat_template
            or getattr(tokenizer_obj, "chat_template", None),
            "vocab_size": getattr(tokenizer_obj, "vocab_size", None),
            "bos_token_id": getattr(tokenizer_obj, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer_obj, "eos_token_id", None),
            "pad_token_id": getattr(tokenizer_obj, "pad_token_id", None),
        }
    else:
        tokenizer_identity = _runtime_config_value(model_config, "tokenizer_identity")
        tokenizer_config_fingerprint = _config_file_fingerprint(tokenizer_path, "tokenizer_config.json")
        tokenizer_fields = {
            "tokenizer_identity": tokenizer_identity,
            "tokenizer_config_fingerprint": tokenizer_config_fingerprint,
            "tokenizer_path_fallback": tokenizer_path
            if tokenizer_identity is None and tokenizer_config_fingerprint is None
            else None,
        }

    model_identity = _runtime_config_value(model_config, "identity")
    model_config_fingerprint = _config_file_fingerprint(model_path, "config.json")
    static_model = {
        "model_identity": model_identity,
        "model_config_fingerprint": model_config_fingerprint,
        "model_path_fallback": model_path
        if model_identity is None and model_config_fingerprint is None
        else None,
        "trust_remote_code": trust_remote_code,
        "model_type": _runtime_config_value(model_config, "model_type", "language_model"),
        "override_config": _runtime_config_value(model_config, "override_config", {}),
    }

    kv_schema = {
        "backend": _runtime_config_value(rollout_config, "name", "sglang"),
        "dtype": _runtime_config_value(rollout_config, "dtype"),
        "tensor_model_parallel_size": _runtime_config_value(
            rollout_config, "tensor_model_parallel_size", 1
        ),
        "data_parallel_size": _runtime_config_value(rollout_config, "data_parallel_size", 1),
        "pipeline_model_parallel_size": _runtime_config_value(
            rollout_config, "pipeline_model_parallel_size", 1
        ),
        "expert_parallel_size": _runtime_config_value(
            rollout_config, "expert_parallel_size", 1
        ),
        "moe_tensor_parallel_size": _runtime_config_value(
            rollout_config, "moe_tensor_parallel_size", 1
        ),
        "quantization": _runtime_config_value(rollout_config, "quantization"),
        "quantization_config_file": _runtime_config_value(
            rollout_config, "quantization_config_file"
        ),
        "sglang_engine_mode": _runtime_config_value(
            rollout_config, "sglang_engine_mode", "local"
        ),
    }

    static_model_fingerprint = _canonical_fingerprint(static_model)
    return CanonicalPolicyKVIdentity(
        policy_version=version,
        model_weights_fingerprint=_canonical_fingerprint(
            {
                "static_model_fingerprint": static_model_fingerprint,
                "policy_version": version,
            }
        ),
        tokenizer_template_fingerprint=_canonical_fingerprint(tokenizer_fields),
        kv_cache_schema_fingerprint=_canonical_fingerprint(kv_schema),
    )


@dataclass(frozen=True)
class CanonicalPolicyKVIdentity:
    """The cross-replica identity for a policy and its KV representation.

    Replica-local allocator generations and request handles are deliberately
    excluded. They validate local runtime state, not whether two replicas can
    safely interpret the same KV blocks.
    """

    policy_version: str
    model_weights_fingerprint: str
    tokenizer_template_fingerprint: str
    kv_cache_schema_fingerprint: str

    @classmethod
    def for_fully_async_policy_version(cls, policy_version: int | str) -> "CanonicalPolicyKVIdentity":
        """Legacy test helper; runtime code must use the runtime-config builder."""
        version = str(policy_version)
        return cls(
            policy_version=version,
            model_weights_fingerprint=_canonical_fingerprint(
                {"legacy_test_policy_version": version}
            ),
            tokenizer_template_fingerprint=_canonical_fingerprint(
                {"legacy_test_tokenizer_template": "fully_async"}
            ),
            kv_cache_schema_fingerprint=_canonical_fingerprint(
                {"legacy_test_kv_schema": "fully_async"}
            ),
        )

    def validate(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    def is_complete(self) -> bool:
        try:
            self.validate()
        except ValueError:
            return False
        return True

    @property
    def digest(self) -> str:
        self.validate()
        canonical_identity = json.dumps(vars(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()

    @property
    def cache_salt(self) -> str:
        """The exact digest passed to SGLang's existing cache namespace."""
        return self.digest


@dataclass(frozen=True)
class TrajectorySegment:
    response_index: int
    start: int
    end: int
    policy_version: str
    sampling_config_fingerprint: str
    behavior_logprobs: tuple[float, ...]
    turn_index: int = 0
    execution_generation: int | None = None


def sampling_config_fingerprint(sampling_params: dict[str, Any]) -> str:
    return _canonical_fingerprint(sampling_params)


@dataclass(frozen=True)
class RolloutTrainingSemantics:
    trajectory_policy_version: str
    environment_version: str
    tool_version: str
    producer_generation: int
    generated_token_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    trajectory_token_counts: tuple[int, ...]
    segments: tuple[TrajectorySegment, ...]

    def trainable_token_mask(self, accepted_policy_versions: set[str]) -> tuple[bool, ...]:
        """Mark generated tokens whose behavior policy is currently eligible."""
        self.validate(self.trajectory_policy_version)
        mask = [False] * len(self.generated_token_ids)
        offsets = [0]
        for count in self.trajectory_token_counts:
            offsets.append(offsets[-1] + count)
        for segment in self.segments:
            offset = offsets[segment.response_index]
            mask[offset + segment.start:offset + segment.end] = [
                segment.policy_version in accepted_policy_versions
            ] * (segment.end - segment.start)
        return tuple(mask)

    def validate(self, identity: CanonicalPolicyKVIdentity | str) -> None:
        versions = (self.trajectory_policy_version, self.environment_version, self.tool_version)
        if any(not isinstance(version, str) or not version for version in versions):
            raise ValueError('trajectory, environment and tool versions must be non-empty strings')
        policy_version = identity.policy_version if isinstance(identity, CanonicalPolicyKVIdentity) else identity
        if self.trajectory_policy_version != policy_version:
            raise ValueError('trajectory policy version must match artifact identity')
        if type(self.producer_generation) is not int or self.producer_generation < 0:
            raise ValueError('producer_generation must be a non-negative integer')
        if not isinstance(self.generated_token_ids, tuple) or not self.generated_token_ids or any(
            type(token_id) is not int or token_id < 0 for token_id in self.generated_token_ids
        ):
            raise ValueError('generated_token_ids must be non-empty non-negative integers')
        if (
            not isinstance(self.behavior_logprobs, tuple)
            or len(self.behavior_logprobs) != len(self.generated_token_ids)
            or any(type(logprob) not in (int, float) or not math.isfinite(logprob) for logprob in self.behavior_logprobs)
        ):
            raise ValueError('behavior_logprobs must align with generated tokens and be finite')
        if any(type(count) is not int or count < 1 for count in self.trajectory_token_counts):
            raise ValueError('trajectory token counts must be positive integers')
        if not self.trajectory_token_counts or sum(self.trajectory_token_counts) != len(self.generated_token_ids):
            raise ValueError('trajectory token counts must cover generated tokens')
        if not self.segments:
            raise ValueError('trajectory segments are required')
        offsets = [0]
        for count in self.trajectory_token_counts:
            offsets.append(offsets[-1] + count)
        positions = [0] * len(self.trajectory_token_counts)
        for segment in self.segments:
            if not isinstance(segment, TrajectorySegment):
                raise ValueError('trajectory segment has invalid type')
            index = segment.response_index
            if type(index) is not int or not 0 <= index < len(positions):
                raise ValueError('trajectory segment response index is invalid')
            if type(segment.start) is not int or type(segment.end) is not int or segment.start != positions[index] or not segment.start < segment.end <= self.trajectory_token_counts[index]:
                raise ValueError('trajectory segments must cover each response contiguously')
            if not isinstance(segment.policy_version, str) or not segment.policy_version:
                raise ValueError('trajectory segment policy version is required')
            if type(segment.turn_index) is not int or segment.turn_index < 0:
                raise ValueError('trajectory segment turn index is invalid')
            if segment.execution_generation is not None and (
                type(segment.execution_generation) is not int or segment.execution_generation < 0
            ):
                raise ValueError('trajectory segment execution generation is invalid')
            if not isinstance(segment.sampling_config_fingerprint, str) or len(segment.sampling_config_fingerprint) != 64 or any(ch not in '0123456789abcdef' for ch in segment.sampling_config_fingerprint):
                raise ValueError('trajectory segment sampling fingerprint is invalid')
            values = self.behavior_logprobs[offsets[index] + segment.start:offsets[index] + segment.end]
            if segment.behavior_logprobs != values:
                raise ValueError('trajectory segment logprobs differ from generated tokens')
            positions[index] = segment.end
        if positions != list(self.trajectory_token_counts):
            raise ValueError('trajectory segments do not cover all generated tokens')
        if self.trajectory_policy_version not in {segment.policy_version for segment in self.segments}:
            raise ValueError('artifact policy identity must occur in trajectory segments')


def rollout_training_semantics_from_batch(
    batch: Any,
    *,
    policy_version: str,
    environment_version: str,
    tool_version: str,
    producer_generation: int,
) -> RolloutTrainingSemantics:
    import torch

    tensors = batch.batch
    if not all(name in tensors for name in ("responses", "response_mask", "rollout_log_probs")):
        raise ValueError("training semantics requires responses, response_mask and rollout_log_probs")
    responses = tensors["responses"]
    response_mask = tensors["response_mask"]
    logprobs = tensors["rollout_log_probs"]
    if not all(isinstance(value, torch.Tensor) for value in (responses, response_mask, logprobs)):
        raise ValueError("training semantics requires tensor fields")
    if responses.ndim != 2 or responses.shape != response_mask.shape or responses.shape != logprobs.shape:
        raise ValueError("training semantics response shapes must match")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must contain only 0 or 1")
    selected = response_mask.bool()
    tokens = tuple(int(value) for value in responses[selected].tolist())
    values = tuple(float(value) for value in logprobs[selected].tolist())
    metadata = batch.non_tensor_batch.get('trajectory_segments')
    if metadata is None or len(metadata) != responses.shape[0]:
        raise ValueError('trajectory segments are required for every response')
    counts = tuple(int(value) for value in selected.sum(dim=1).tolist())
    segments = []
    for response_index, row in enumerate(metadata):
        if not isinstance(row, (list, tuple)):
            raise ValueError('trajectory segments must be a sequence per response')
        row_logprobs = tuple(float(value) for value in logprobs[response_index][selected[response_index]].tolist())
        for item in row:
            if not isinstance(item, dict) or set(item) not in ({'start', 'end', 'policy_version', 'sampling_config_fingerprint', 'turn_index'}, {'start', 'end', 'policy_version', 'sampling_config_fingerprint', 'turn_index', 'execution_generation'}):
                raise ValueError('trajectory segment has invalid schema')
            start, end = item['start'], item['end']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(row_logprobs):
                raise ValueError('trajectory segment token range is invalid')
            segments.append(TrajectorySegment(response_index, start, end, str(item['policy_version']), item['sampling_config_fingerprint'], row_logprobs[start:end], item['turn_index'], item.get('execution_generation')))
    semantics = RolloutTrainingSemantics(
        trajectory_policy_version=str(policy_version),
        environment_version=environment_version,
        tool_version=tool_version,
        producer_generation=producer_generation,
        generated_token_ids=tokens,
        behavior_logprobs=values,
        trajectory_token_counts=counts,
        segments=tuple(segments),
    )
    semantics.validate(str(policy_version))
    return semantics


def validate_rollout_training_payload(
    payload: bytes, manifest: RolloutArtifactManifest, *, environment_version: str, tool_version: str,
    accepted_policy_versions: set[str] | None = None,
) -> None:
    import ray

    if manifest.training_semantics is None:
        raise ValueError("rollout training semantics are required")
    decoded = ray.cloudpickle.loads(payload)
    batch = decoded.full_batch
    metadata = batch.non_tensor_batch
    response_count = batch.batch["responses"].shape[0]
    if accepted_policy_versions is not None and not all(
        manifest.training_semantics.trainable_token_mask(accepted_policy_versions)
    ):
        raise ValueError("rollout trajectory contains a stale policy segment")
    actual = rollout_training_semantics_from_batch(
        batch,
        policy_version=manifest.identity.policy_version,
        environment_version=environment_version,
        tool_version=tool_version,
        producer_generation=manifest.training_semantics.producer_generation,
    )
    if actual != manifest.training_semantics:
        raise ValueError("rollout training semantics differ from payload")
    minimum_versions = metadata.get("min_global_steps")
    maximum_versions = metadata.get("max_global_steps")
    if minimum_versions is None or maximum_versions is None or len(minimum_versions) != response_count or len(maximum_versions) != response_count:
        raise ValueError("rollout payload policy version differs from trajectory segments or is missing")
    for response_index in range(response_count):
        row_versions = [segment.policy_version for segment in actual.segments if segment.response_index == response_index]
        if (
            not row_versions
            or str(minimum_versions[response_index]) != row_versions[0]
            or str(maximum_versions[response_index]) != row_versions[-1]
        ):
            raise ValueError("rollout payload policy version differs from trajectory segments")
    producer_generations = metadata.get("producer_generation")
    if producer_generations is None or len(producer_generations) != response_count or any(
        type(generation) is bool or operator.index(generation) != manifest.training_semantics.producer_generation
        for generation in producer_generations
    ):
        raise ValueError("rollout payload producer generation differs from artifact manifest")
    # The manifest describes one prompt-level artifact; its payload still holds
    # every response in that prompt's GRPO group.
    if decoded.sample_id != manifest.rollout_group_id or response_count != manifest.trajectory_count:
        raise ValueError("rollout payload group differs from artifact manifest")
    try:
        group_ids = tuple(metadata["rollout_group_id"])
        group_sizes = tuple(metadata["rollout_group_size"])
        sample_indexes = tuple(metadata["rollout_sample_index"])
        group_roles = tuple(metadata["rollout_group_role"])
        if any(type(value) is bool for value in (*group_sizes, *sample_indexes)):
            raise TypeError("boolean group size or sample index")
        sizes = tuple(operator.index(value) for value in group_sizes)
        indexes = tuple(operator.index(value) for value in sample_indexes)
    except (KeyError, TypeError) as exc:
        raise ValueError("rollout payload group metadata is missing or invalid") from exc
    if (
        len(group_ids) != response_count
        or len(sizes) != response_count
        or len(indexes) != response_count
        or len(group_roles) != response_count
        or any(group_id != manifest.rollout_group_id for group_id in group_ids)
        or any(size != response_count for size in sizes)
        or set(indexes) != set(range(response_count))
        or any(role != "sample" for role in group_roles)
    ):
        raise ValueError("rollout payload group metadata differs from artifact manifest")


@dataclass(frozen=True)
class RolloutArtifactManifest:
    artifact_id: str
    producer_id: str
    identity: CanonicalPolicyKVIdentity
    rollout_group_id: str
    rollout_group_size: int
    rollout_sample_index: int
    trajectory_count: int = 1
    rollout_group_role: str = "sample"
    group_manifest_id: str | None = None
    payload_digest: str | None = None
    kv_ref: None = None
    training_semantics: RolloutTrainingSemantics | None = None

    @property
    def counts_toward_group(self) -> bool:
        return self.rollout_group_role == "sample"

    def validate(self) -> None:
        if not self.artifact_id or not self.producer_id or not self.rollout_group_id:
            raise ValueError("artifact_id, producer_id and rollout_group_id must be non-empty")
        self.identity.validate()
        if self.training_semantics is not None and not isinstance(self.training_semantics, RolloutTrainingSemantics):
            raise ValueError('training_semantics must be RolloutTrainingSemantics')
        if self.training_semantics is not None:
            self.training_semantics.validate(self.identity)
        if self.rollout_group_size < 1:
            raise ValueError("rollout_group_size must be >= 1")
        if self.trajectory_count < 1:
            raise ValueError("trajectory_count must be >= 1")
        if self.counts_toward_group and not 0 <= self.rollout_sample_index < self.rollout_group_size:
            raise ValueError("sample artifact index must be within rollout_group_size")
        if not self.counts_toward_group and self.rollout_sample_index != -1:
            raise ValueError("non-sample artifact must use rollout_sample_index=-1")
        if self.kv_ref is not None:
            raise ValueError("v1 artifact data plane does not accept a kv_ref")


def _encode_policy_identity(identity: CanonicalPolicyKVIdentity) -> dict[str, str]:
    identity.validate()
    return {
        "policy_version": identity.policy_version,
        "model_weights_fingerprint": identity.model_weights_fingerprint,
        "tokenizer_template_fingerprint": identity.tokenizer_template_fingerprint,
        "kv_cache_schema_fingerprint": identity.kv_cache_schema_fingerprint,
    }


def _decode_policy_identity(payload: Any) -> CanonicalPolicyKVIdentity:
    if not isinstance(payload, dict) or set(payload) != {
        "policy_version",
        "model_weights_fingerprint",
        "tokenizer_template_fingerprint",
        "kv_cache_schema_fingerprint",
    }:
        raise ValueError("policy identity snapshot has an invalid schema")
    if any(not isinstance(value, str) for value in payload.values()):
        raise ValueError("policy identity snapshot fields must be strings")
    identity = CanonicalPolicyKVIdentity(**payload)
    identity.validate()
    return identity


def _encode_training_semantics(semantics: RolloutTrainingSemantics) -> dict[str, Any]:
    return {
        "trajectory_policy_version": semantics.trajectory_policy_version,
        "environment_version": semantics.environment_version,
        "tool_version": semantics.tool_version,
        "producer_generation": semantics.producer_generation,
        "generated_token_ids": list(semantics.generated_token_ids),
        "behavior_logprobs": list(semantics.behavior_logprobs),
        "trajectory_token_counts": list(semantics.trajectory_token_counts),
        "segments": [
            {
                "response_index": segment.response_index,
                "start": segment.start,
                "end": segment.end,
                "policy_version": segment.policy_version,
                "sampling_config_fingerprint": segment.sampling_config_fingerprint,
                "behavior_logprobs": list(segment.behavior_logprobs),
                "turn_index": segment.turn_index,
                **({"execution_generation": segment.execution_generation}
                   if segment.execution_generation is not None else {}),
            }
            for segment in semantics.segments
        ],
    }


def _decode_training_semantics(payload: Any) -> RolloutTrainingSemantics:
    required_fields = {
        "trajectory_policy_version",
        "environment_version",
        "tool_version",
        "producer_generation",
        "generated_token_ids",
        "behavior_logprobs",
        "trajectory_token_counts",
        "segments",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError("training semantics snapshot has an invalid schema")
    if any(not isinstance(payload[name], str) for name in required_fields - {
        "producer_generation", "generated_token_ids", "behavior_logprobs", "trajectory_token_counts", "segments"
    }):
        raise ValueError("training semantics version fields must be strings")
    if type(payload["producer_generation"]) is not int or payload["producer_generation"] < 0:
        raise ValueError("training semantics producer generation must be a non-negative integer")
    if (
        not isinstance(payload["generated_token_ids"], list)
        or any(type(token_id) is not int or token_id < 0 for token_id in payload["generated_token_ids"])
    ):
        raise ValueError("training semantics token snapshot must be a list of non-negative integers")
    if (
        not isinstance(payload["behavior_logprobs"], list)
        or len(payload["behavior_logprobs"]) != len(payload["generated_token_ids"])
        or any(
            type(logprob) not in (int, float) or not math.isfinite(logprob)
            for logprob in payload["behavior_logprobs"]
        )
    ):
        raise ValueError("training semantics logprob snapshot is invalid")
    if not isinstance(payload["trajectory_token_counts"], list) or any(type(count) is not int for count in payload["trajectory_token_counts"]):
        raise ValueError("training semantics trajectory counts are invalid")
    if not isinstance(payload["segments"], list):
        raise ValueError("training semantics segments are invalid")
    segments = []
    segment_fields = {"response_index", "start", "end", "policy_version", "sampling_config_fingerprint", "behavior_logprobs", "turn_index"}
    for item in payload["segments"]:
        if not isinstance(item, dict) or set(item) not in (segment_fields, segment_fields | {"execution_generation"}) or not isinstance(item["behavior_logprobs"], list):
            raise ValueError("training semantics segment snapshot has an invalid schema")
        segments.append(TrajectorySegment(**{**item, "behavior_logprobs": tuple(item["behavior_logprobs"])}))
    semantics = RolloutTrainingSemantics(
        trajectory_policy_version=payload["trajectory_policy_version"],
        environment_version=payload["environment_version"],
        tool_version=payload["tool_version"],
        producer_generation=payload["producer_generation"],
        generated_token_ids=tuple(payload["generated_token_ids"]),
        behavior_logprobs=tuple(payload["behavior_logprobs"]),
        trajectory_token_counts=tuple(payload["trajectory_token_counts"]),
        segments=tuple(segments),
    )
    return semantics


def _encode_manifest(manifest: RolloutArtifactManifest) -> dict[str, Any]:
    manifest.validate()
    return {
        "artifact_id": manifest.artifact_id,
        "producer_id": manifest.producer_id,
        "identity": _encode_policy_identity(manifest.identity),
        "rollout_group_id": manifest.rollout_group_id,
        "rollout_group_size": manifest.rollout_group_size,
        "rollout_sample_index": manifest.rollout_sample_index,
        "trajectory_count": manifest.trajectory_count,
        "rollout_group_role": manifest.rollout_group_role,
        "group_manifest_id": manifest.group_manifest_id,
        "payload_digest": manifest.payload_digest,
        "kv_ref": None,
        "training_semantics": (
            None
            if manifest.training_semantics is None
            else _encode_training_semantics(manifest.training_semantics)
        ),
    }


def _decode_manifest(payload: Any) -> RolloutArtifactManifest:
    required_fields = {
        "artifact_id",
        "producer_id",
        "identity",
        "rollout_group_id",
        "rollout_group_size",
        "rollout_sample_index",
        "trajectory_count",
        "rollout_group_role",
        "group_manifest_id",
        "payload_digest",
        "kv_ref",
        "training_semantics",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError("artifact manifest snapshot has an invalid schema")
    for name in ("artifact_id", "producer_id", "rollout_group_id", "rollout_group_role"):
        if not isinstance(payload[name], str):
            raise ValueError(f"artifact manifest {name} must be a string")
    for name in ("rollout_group_size", "rollout_sample_index", "trajectory_count"):
        if type(payload[name]) is not int:
            raise ValueError(f"artifact manifest {name} must be an integer")
    for name in ("group_manifest_id", "payload_digest"):
        if payload[name] is not None and not isinstance(payload[name], str):
            raise ValueError(f"artifact manifest {name} must be a string or null")
    if payload["kv_ref"] is not None:
        raise ValueError("artifact manifest snapshot does not accept a kv_ref")
    if payload["training_semantics"] is not None and not isinstance(payload["training_semantics"], dict):
        raise ValueError("artifact manifest training semantics snapshot must be an object or null")

    manifest = RolloutArtifactManifest(
        artifact_id=payload["artifact_id"],
        producer_id=payload["producer_id"],
        identity=_decode_policy_identity(payload["identity"]),
        rollout_group_id=payload["rollout_group_id"],
        rollout_group_size=payload["rollout_group_size"],
        rollout_sample_index=payload["rollout_sample_index"],
        trajectory_count=payload["trajectory_count"],
        rollout_group_role=payload["rollout_group_role"],
        group_manifest_id=payload["group_manifest_id"],
        payload_digest=payload["payload_digest"],
        kv_ref=None,
        training_semantics=(
            None
            if payload["training_semantics"] is None
            else _decode_training_semantics(payload["training_semantics"])
        ),
    )
    manifest.validate()
    return manifest


@dataclass
class RolloutArtifact:
    manifest: RolloutArtifactManifest
    payload: Any
    status: RolloutArtifactStatus = RolloutArtifactStatus.WRITING
    created_at: float = 0.0
    lease_token: str | None = None
    lease_expires_at: float | None = None
    rejected_reason: str | None = None


@dataclass(frozen=True)
class ArtifactLease:
    """Exclusive claim for one complete training group."""

    lease_token: str
    artifact_ids: tuple[str, ...]
    manifests: tuple[RolloutArtifactManifest, ...]
    payloads: tuple[Any, ...]
    expires_at: float


@dataclass(frozen=True)
class RolloutArtifactNotification:
    """Small control-plane message carried by the existing MessageQueue.

    The payload remains in the artifact data plane.  This notification is only
    a readiness hint, so it is deliberately safe to retry or drop.
    """

    artifact_id: str


@dataclass(frozen=True)
class _MooncakePayloadHandle:
    """Internal reference from metadata state to an offloaded payload."""

    key: str
    digest: str


class InMemoryRolloutArtifactStore:
    """Reference artifact store with atomic complete-group claims.

    It is intentionally synchronous and process-local.  Its API and invariants are
    designed to be used by an asynchronous adapter, and later mapped to a durable
    data-plane backend such as Mooncake Store.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._artifacts: dict[str, RolloutArtifact] = {}
        self._producer_incarnations: dict[str, dict[str, int]] = {}

    @staticmethod
    def _group_key(manifest: RolloutArtifactManifest) -> str:
        """Group IDs are producer-global; version disagreement is a malformed group."""
        return manifest.rollout_group_id

    def register_producer(self, producer_id: str, incarnation_id: str) -> int:
        if not isinstance(producer_id, str) or not producer_id:
            raise ValueError("producer_id must be a non-empty string")
        if not isinstance(incarnation_id, str) or not incarnation_id:
            raise ValueError("incarnation_id must be a non-empty string")
        incarnations = self._producer_incarnations.setdefault(producer_id, {})
        existing_generation = incarnations.get(incarnation_id)
        if existing_generation is not None:
            current_generation = max(incarnations.values())
            if existing_generation != current_generation:
                raise ValueError(
                    f"retired producer incarnation is not current: {producer_id}/{incarnation_id}"
                )
            return existing_generation
        generation = 0 if not incarnations else max(incarnations.values()) + 1
        incarnations[incarnation_id] = generation
        return generation

    def get_current_producer_generation(self, producer_id: str) -> int | None:
        incarnations = self._producer_incarnations.get(producer_id)
        return None if not incarnations else max(incarnations.values())

    def validate_current_producer_generation(self, manifest: RolloutArtifactManifest) -> None:
        generation = self.get_current_producer_generation(manifest.producer_id)
        if generation is None:
            raise ValueError(f"producer is not registered: {manifest.producer_id}")
        if manifest.training_semantics is None:
            raise ValueError("strict rollout artifacts require training semantics")
        if manifest.training_semantics.producer_generation != generation:
            raise ValueError(
                f"stale producer generation for artifact {manifest.artifact_id}: "
                f"expected={generation} actual={manifest.training_semantics.producer_generation}"
            )

    def publish(self, manifest: RolloutArtifactManifest, payload: Any) -> None:
        """Atomically make one artifact ready after fail-closed validation."""
        self.reserve(manifest, payload)
        self.commit(manifest.artifact_id, payload)

    def reserve(self, manifest: RolloutArtifactManifest, payload: Any) -> None:
        """Validate and insert an artifact in WRITING state."""
        manifest.validate()
        if manifest.training_semantics is not None:
            self.validate_current_producer_generation(manifest)
        existing = self._artifacts.get(manifest.artifact_id)
        if existing is not None:
            if (
                existing.manifest == manifest
                and existing.payload == payload
                and existing.status in {RolloutArtifactStatus.WRITING, RolloutArtifactStatus.READY}
            ):
                return
            raise ValueError(f"artifact_id already exists: {manifest.artifact_id}")

        for artifact in self._artifacts.values():
            other = artifact.manifest
            if self._group_key(other) != self._group_key(manifest):
                continue
            if other.identity != manifest.identity or other.rollout_group_size != manifest.rollout_group_size:
                raise ValueError("all artifacts in a rollout group must have the same identity and declared size")
            if other.training_semantics is not None or manifest.training_semantics is not None:
                if (
                    other.training_semantics is None
                    or manifest.training_semantics is None
                    or other.training_semantics.environment_version != manifest.training_semantics.environment_version
                    or other.training_semantics.tool_version != manifest.training_semantics.tool_version
                ):
                    raise ValueError('all artifacts in a rollout group must share environment and tool versions')
            if other.counts_toward_group and manifest.counts_toward_group and (
                other.rollout_sample_index == manifest.rollout_sample_index
            ):
                raise ValueError("duplicate rollout_sample_index in rollout group")

        self._artifacts[manifest.artifact_id] = RolloutArtifact(
            manifest=manifest,
            payload=payload,
            status=RolloutArtifactStatus.WRITING,
            created_at=self._clock(),
        )

    def commit(self, artifact_id: str, payload: Any) -> None:
        """Promote a WRITING artifact to READY after its payload is durable."""
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            raise ValueError(f"artifact is not reserved: {artifact_id}")
        if artifact.manifest.training_semantics is not None:
            self.validate_current_producer_generation(artifact.manifest)
        if artifact.status is RolloutArtifactStatus.WRITING:
            artifact.payload = payload
            artifact.status = RolloutArtifactStatus.READY
            return
        if artifact.status is RolloutArtifactStatus.READY and artifact.payload == payload:
            return
        raise ValueError(f"artifact is not in a committable state: {artifact_id}")

    def _expire_leases(self) -> int:
        now = self._clock()
        expired = 0
        for artifact in self._artifacts.values():
            if artifact.status is RolloutArtifactStatus.LEASED and artifact.lease_expires_at <= now:
                artifact.status = RolloutArtifactStatus.READY
                artifact.lease_token = None
                artifact.lease_expires_at = None
                expired += 1
        return expired

    def reject_unaccepted_policy_versions(self, accepted_policy_versions: set[str]) -> int:
        """Reject unclaimed artifacts that cannot legally enter the next update."""
        rejected = 0
        for artifact in self._artifacts.values():
            if artifact.status is not RolloutArtifactStatus.READY:
                continue
            semantics = artifact.manifest.training_semantics
            reason = None
            if artifact.manifest.identity.policy_version not in accepted_policy_versions:
                reason = "policy_version_not_accepted"
            elif semantics is not None and not all(semantics.trainable_token_mask(accepted_policy_versions)):
                # The final policy version can still be accepted while an
                # earlier continuation segment is too old for this update.
                reason = "policy_segment_not_accepted"
            if reason is not None:
                artifact.status = RolloutArtifactStatus.REJECTED
                artifact.rejected_reason = reason
                rejected += 1
        return rejected

    def _reject_unaccepted_training_versions(
        self,
        accepted_environment_versions: set[str] | None,
        accepted_tool_versions: set[str] | None,
    ) -> None:
        if accepted_environment_versions is None and accepted_tool_versions is None:
            return
        if accepted_environment_versions is not None and not accepted_environment_versions:
            raise ValueError('accepted_environment_versions must not be empty')
        if accepted_tool_versions is not None and not accepted_tool_versions:
            raise ValueError('accepted_tool_versions must not be empty')
        for artifact in self._artifacts.values():
            if artifact.status is not RolloutArtifactStatus.READY:
                continue
            semantics = artifact.manifest.training_semantics
            if semantics is None or (
                accepted_environment_versions is not None
                and semantics.environment_version not in accepted_environment_versions
            ) or (
                accepted_tool_versions is not None and semantics.tool_version not in accepted_tool_versions
            ):
                artifact.status = RolloutArtifactStatus.REJECTED
                artifact.rejected_reason = 'training_version_not_accepted'

    def _reject_stale_producer_generations(self, require_current: bool) -> None:
        if not require_current:
            return
        for artifact in self._artifacts.values():
            if artifact.status is not RolloutArtifactStatus.READY:
                continue
            try:
                self.validate_current_producer_generation(artifact.manifest)
            except ValueError:
                artifact.status = RolloutArtifactStatus.REJECTED
                artifact.rejected_reason = "producer_generation_not_current"

    def claim_complete_group(
        self,
        *,
        accepted_policy_versions: set[str],
        lease_timeout_seconds: float,
        accepted_environment_versions: set[str] | None = None,
        accepted_tool_versions: set[str] | None = None,
        require_current_producer_generation: bool = False,
    ) -> ArtifactLease | None:
        if not accepted_policy_versions:
            raise ValueError("accepted_policy_versions must not be empty")
        if lease_timeout_seconds <= 0:
            raise ValueError("lease_timeout_seconds must be positive")
        self._expire_leases()
        self.reject_unaccepted_policy_versions(accepted_policy_versions)
        self._reject_unaccepted_training_versions(
            accepted_environment_versions,
            accepted_tool_versions,
        )
        self._reject_stale_producer_generations(require_current_producer_generation)

        groups: dict[str, list[RolloutArtifact]] = {}
        for artifact in self._artifacts.values():
            if artifact.status is RolloutArtifactStatus.READY:
                groups.setdefault(self._group_key(artifact.manifest), []).append(artifact)

        for group_key in sorted(groups):
            lease = self._claim_ready_group(groups[group_key], lease_timeout_seconds)
            if lease is not None:
                return lease
        return None

    def claim_group_for_artifact(
        self,
        artifact_id: str,
        *,
        accepted_policy_versions: set[str],
        lease_timeout_seconds: float,
        accepted_environment_versions: set[str] | None = None,
        accepted_tool_versions: set[str] | None = None,
        require_current_producer_generation: bool = False,
    ) -> ArtifactLease | None:
        """Claim only the complete group announced by ``artifact_id``.

        Queue notifications may be delayed, duplicated, or reordered.  They must
        never cause a consumer to claim an unrelated ready group, particularly
        when multiple consumers are enabled in a later deployment.
        """
        if not accepted_policy_versions:
            raise ValueError("accepted_policy_versions must not be empty")
        if lease_timeout_seconds <= 0:
            raise ValueError("lease_timeout_seconds must be positive")
        self._expire_leases()
        self.reject_unaccepted_policy_versions(accepted_policy_versions)
        self._reject_unaccepted_training_versions(
            accepted_environment_versions,
            accepted_tool_versions,
        )
        self._reject_stale_producer_generations(require_current_producer_generation)

        announced = self._artifacts.get(artifact_id)
        if announced is None or announced.status is not RolloutArtifactStatus.READY:
            return None
        group_key = self._group_key(announced.manifest)
        group = [
            artifact
            for artifact in self._artifacts.values()
            if artifact.status is RolloutArtifactStatus.READY and self._group_key(artifact.manifest) == group_key
        ]
        # The store is a concurrent Ray actor: a rollout publisher may flip an
        # artifact to READY after the sweep in reject_unaccepted_policy_versions()
        # and before this group is assembled. Re-check every member here so a
        # stale-segment artifact can never slip into a lease; otherwise the
        # trainer-side validate_rollout_training_payload() raises and kills the
        # training loop instead of simply waiting for a fresh group.
        for artifact in group:
            semantics = artifact.manifest.training_semantics
            if artifact.manifest.identity.policy_version not in accepted_policy_versions:
                artifact.status = RolloutArtifactStatus.REJECTED
                artifact.rejected_reason = "policy_version_not_accepted"
            elif semantics is not None and not all(semantics.trainable_token_mask(accepted_policy_versions)):
                artifact.status = RolloutArtifactStatus.REJECTED
                artifact.rejected_reason = "policy_segment_not_accepted"
        if announced.status is not RolloutArtifactStatus.READY:
            # The announced artifact itself was just rejected above; let the
            # next notification for a fresh group drive another claim.
            return None
        group = [
            artifact
            for artifact in self._artifacts.values()
            if artifact.status is RolloutArtifactStatus.READY and self._group_key(artifact.manifest) == group_key
        ]
        return self._claim_ready_group(group, lease_timeout_seconds)

    def _claim_ready_group(self, artifacts: list[RolloutArtifact], lease_timeout_seconds: float) -> ArtifactLease | None:
        samples = [artifact for artifact in artifacts if artifact.manifest.counts_toward_group]
        if not samples:
            return None
        size = samples[0].manifest.rollout_group_size
        indexes = {artifact.manifest.rollout_sample_index for artifact in samples}
        if len(samples) != size or indexes != set(range(size)):
            return None
        # A group must never mix an incompatible manifest, even if a malformed
        # backend populated it.  publish() already enforces this for this store.
        identity = samples[0].manifest.identity
        if any(
            artifact.manifest.identity != identity or artifact.manifest.rollout_group_size != size
            for artifact in samples
        ):
            return None

        ordered = sorted(samples, key=lambda artifact: artifact.manifest.rollout_sample_index)
        token = secrets.token_urlsafe(24)
        expires_at = self._clock() + lease_timeout_seconds
        for artifact in ordered:
            artifact.status = RolloutArtifactStatus.LEASED
            artifact.lease_token = token
            artifact.lease_expires_at = expires_at
        return ArtifactLease(
            lease_token=token,
            artifact_ids=tuple(artifact.manifest.artifact_id for artifact in ordered),
            manifests=tuple(artifact.manifest for artifact in ordered),
            payloads=tuple(artifact.payload for artifact in ordered),
            expires_at=expires_at,
        )

    def acknowledge(self, lease: ArtifactLease) -> None:
        """Commit consumption of all artifacts in a claimed group exactly once."""
        artifacts = [self._artifacts[artifact_id] for artifact_id in lease.artifact_ids]
        if all(artifact.status is RolloutArtifactStatus.CONSUMED for artifact in artifacts):
            return
        if any(
            artifact.status is not RolloutArtifactStatus.LEASED or artifact.lease_token != lease.lease_token
            for artifact in artifacts
        ):
            raise ValueError("lease does not own every artifact in the group")
        for artifact in artifacts:
            artifact.status = RolloutArtifactStatus.CONSUMED
            artifact.lease_token = None
            artifact.lease_expires_at = None

    def release(self, lease: ArtifactLease) -> None:
        artifacts = [self._artifacts[artifact_id] for artifact_id in lease.artifact_ids]
        if any(
            artifact.status is not RolloutArtifactStatus.LEASED or artifact.lease_token != lease.lease_token
            for artifact in artifacts
        ):
            raise ValueError("lease does not own every artifact in the group")
        for artifact in artifacts:
            artifact.status = RolloutArtifactStatus.READY
            artifact.lease_token = None
            artifact.lease_expires_at = None

    def get_status(self, artifact_id: str) -> RolloutArtifactStatus:
        return self._artifacts[artifact_id].status

    def export_metadata_snapshot(self) -> dict[str, Any]:
        artifacts = []
        for artifact in self._artifacts.values():
            artifacts.append(
                {
                    "manifest": _encode_manifest(artifact.manifest),
                    "status": artifact.status.value,
                    "rejected_reason": artifact.rejected_reason,
                }
            )
        return {
            "schema_version": 2,
            "producers": {
                producer_id: dict(incarnations)
                for producer_id, incarnations in self._producer_incarnations.items()
            },
            "artifacts": artifacts,
        }

    def restore_metadata_snapshot(
        self,
        snapshot: Any,
        *,
        payload_resolver: Callable[[RolloutArtifactManifest], Any] | None = None,
    ) -> None:
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "schema_version",
            "producers",
            "artifacts",
        }:
            raise ValueError("metadata snapshot has an invalid schema")
        if snapshot["schema_version"] != 2 or type(snapshot["schema_version"]) is not int:
            raise ValueError("unsupported metadata snapshot schema version")
        if not isinstance(snapshot["producers"], dict) or not isinstance(snapshot["artifacts"], list):
            raise ValueError("metadata snapshot producers or artifacts have invalid types")

        restored_producers: dict[str, dict[str, int]] = {}
        for producer_id, incarnations in snapshot["producers"].items():
            if not isinstance(producer_id, str) or not producer_id:
                raise ValueError("metadata snapshot producer_id must be a non-empty string")
            if not isinstance(incarnations, dict) or not incarnations:
                raise ValueError("metadata snapshot producer history must be a non-empty object")
            restored_incarnations: dict[str, int] = {}
            for incarnation_id, generation in incarnations.items():
                if not isinstance(incarnation_id, str) or not incarnation_id:
                    raise ValueError("metadata snapshot incarnation_id must be a non-empty string")
                if type(generation) is not int or generation < 0:
                    raise ValueError("metadata snapshot generation must be a non-negative integer")
                restored_incarnations[incarnation_id] = generation
            restored_producers[producer_id] = restored_incarnations

        restored_artifacts: dict[str, RolloutArtifact] = {}
        for artifact_snapshot in snapshot["artifacts"]:
            if not isinstance(artifact_snapshot, dict) or set(artifact_snapshot) != {
                "manifest",
                "status",
                "rejected_reason",
            }:
                raise ValueError("metadata artifact snapshot has an invalid schema")
            if artifact_snapshot["rejected_reason"] is not None and not isinstance(
                artifact_snapshot["rejected_reason"], str
            ):
                raise ValueError("metadata artifact rejected_reason must be a string or null")
            try:
                status = RolloutArtifactStatus(artifact_snapshot["status"])
            except ValueError as exc:
                raise ValueError("metadata artifact snapshot has an invalid status") from exc
            manifest = _decode_manifest(artifact_snapshot["manifest"])
            if manifest.artifact_id in restored_artifacts:
                raise ValueError(f"duplicate artifact in metadata snapshot: {manifest.artifact_id}")
            payload = None
            if payload_resolver is not None:
                payload = payload_resolver(manifest)
            restored_artifacts[manifest.artifact_id] = RolloutArtifact(
                manifest=manifest,
                payload=payload,
                status=(
                    RolloutArtifactStatus.READY
                    if status is RolloutArtifactStatus.LEASED
                    else status
                ),
                created_at=self._clock(),
                rejected_reason=artifact_snapshot["rejected_reason"],
            )

        self._producer_incarnations = restored_producers
        self._artifacts = restored_artifacts

    def get_metrics(self) -> dict[str, int]:
        self._expire_leases()
        counts = {status.value: 0 for status in RolloutArtifactStatus}
        for artifact in self._artifacts.values():
            counts[artifact.status.value] += 1
        counts["total"] = len(self._artifacts)
        return counts


class MooncakePayloadStore:
    """Small synchronous adapter over MooncakeDistributedStore.

    The adapter keeps the artifact state machine independent of the Mooncake SDK.
    It converts Mooncake's integer return codes into explicit exceptions so a failed
    write or read can never be mistaken for a missing artifact.
    """

    def __init__(
        self,
        *,
        local_hostname: str = "",
        metadata_server: str = "",
        global_segment_size: int = 512 * 1024 * 1024,
        local_buffer_size: int = 128 * 1024 * 1024,
        protocol: str = "tcp",
        device_name: str = "",
        master_server_address: str = "",
        preferred_segment: str = "",
        store: Any | None = None,
    ):
        if global_segment_size <= 0:
            raise ValueError("global_segment_size must be positive")
        if local_buffer_size <= 0:
            raise ValueError("local_buffer_size must be positive")

        self._store = store if store is not None else self._create_store()
        self._preferred_segment = preferred_segment
        self._pending_removals: set[str] = set()
        setup_args = (
            local_hostname or os.getenv("MOONCAKE_LOCAL_HOSTNAME") or socket.getfqdn(),
            metadata_server or os.getenv("MOONCAKE_METADATA_SERVER") or "http://localhost:8080/metadata",
            int(global_segment_size),
            int(local_buffer_size),
            protocol,
            device_name,
            master_server_address or os.getenv("MOONCAKE_MASTER_SERVER") or "localhost:50051",
        )
        return_code = self._store.setup(*setup_args)
        if return_code != 0:
            raise RuntimeError(f"failed to initialize Mooncake Store: return_code={return_code}")

    @staticmethod
    def _create_store() -> Any:
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as exc:
            raise RuntimeError("mooncake-store backend requires the mooncake-store Python package") from exc
        return MooncakeDistributedStore()

    def put(self, key: str, payload: bytes) -> None:
        if self._preferred_segment:
            from mooncake.store import ReplicateConfig

            config = ReplicateConfig()
            config.preferred_segment = self._preferred_segment
            return_code = self._store.put(key, payload, config)
        else:
            return_code = self._store.put(key, payload)
        if return_code != 0:
            raise RuntimeError(f"failed to put Mooncake artifact {key}: return_code={return_code}")

    def get(self, key: str) -> bytes:
        if not self._store.is_exist(key):
            raise RuntimeError(f"Mooncake artifact is missing: {key}")
        payload = self._store.get(key)
        if not isinstance(payload, bytes):
            raise RuntimeError(f"Mooncake artifact has an invalid payload type: {key}")
        return payload

    def exists(self, key: str) -> bool:
        return bool(self._store.is_exist(key))

    def remove(self, key: str) -> None:
        if not self._store.is_exist(key):
            self._pending_removals.discard(key)
            return
        try:
            # The artifact store owns this object and has already committed its
            # metadata transition.  Force deletion avoids Mooncake's short object
            # lease blocking cleanup of consumed training data.
            return_code = self._store.remove(key, force=True)
        except TypeError:
            # Compatibility fallback for older Mooncake bindings without force.
            return_code = self._store.remove(key)
        if return_code == -706:
            # Mooncake gives a newly written object a short lease.  The object is
            # no longer owned by the artifact metadata, so defer deletion instead
            # of failing the already-committed training update.
            self._pending_removals.add(key)
            return
        if return_code != 0:
            raise RuntimeError(f"failed to remove Mooncake artifact {key}: return_code={return_code}")
        self._pending_removals.discard(key)

    def retry_pending_removals(self) -> None:
        for key in list(self._pending_removals):
            self.remove(key)

    @property
    def pending_removal_count(self) -> int:
        return len(self._pending_removals)

    def close(self) -> None:
        close = getattr(self._store, "close", None)
        if close is not None:
            close()


class MooncakeRolloutArtifactStore:
    """Artifact store with metadata in Ray state and payloads in Mooncake Store.

    The Ray actor remains the single writer for artifact status and lease ownership.
    Mooncake Store carries only immutable serialized payloads.  This split preserves
    the complete-group and version-safety invariants while moving large rollout
    objects out of trainer and rollouter process memory.
    """

    def __init__(
        self,
        *,
        namespace: str = "verl/rollout-artifacts/v1",
        clock: Callable[[], float] = time.monotonic,
        payload_store: MooncakePayloadStore | None = None,
        **payload_store_config: Any,
    ):
        if not namespace:
            raise ValueError("namespace must be non-empty")
        self._namespace = namespace
        self._metadata = InMemoryRolloutArtifactStore(clock=clock)
        self._payload_store = payload_store or MooncakePayloadStore(**payload_store_config)

    def _payload_handle(self, manifest: RolloutArtifactManifest) -> _MooncakePayloadHandle:
        if not manifest.payload_digest:
            raise ValueError("manifest.payload_digest is required for the Mooncake backend")
        key = f"{self._namespace}/{manifest.artifact_id}/{manifest.identity.digest}"
        return _MooncakePayloadHandle(key=key, digest=manifest.payload_digest)

    def register_producer(self, producer_id: str, incarnation_id: str) -> int:
        return self._metadata.register_producer(producer_id, incarnation_id)

    def get_current_producer_generation(self, producer_id: str) -> int | None:
        return self._metadata.get_current_producer_generation(producer_id)

    def _offloaded_manifest_and_handle(
        self,
        manifest: RolloutArtifactManifest,
        payload: bytes,
    ) -> tuple[RolloutArtifactManifest, _MooncakePayloadHandle]:
        if not isinstance(payload, bytes):
            raise TypeError("Mooncake artifact payloads must already be serialized bytes")
        manifest.validate()
        digest = hashlib.sha256(payload).hexdigest()
        if manifest.payload_digest is not None and manifest.payload_digest != digest:
            raise ValueError("manifest.payload_digest does not match the serialized payload")
        offloaded_manifest = replace(manifest, payload_digest=digest)
        return offloaded_manifest, self._payload_handle(offloaded_manifest)

    def reserve(self, manifest: RolloutArtifactManifest, payload: Any) -> None:
        """Validate and reserve metadata before writing the immutable payload."""
        offloaded_manifest, handle = self._offloaded_manifest_and_handle(manifest, payload)
        if offloaded_manifest.training_semantics is not None:
            self._metadata.validate_current_producer_generation(offloaded_manifest)

        if self._payload_store.exists(handle.key):
            existing_payload = self._payload_store.get(handle.key)
            if hashlib.sha256(existing_payload).hexdigest() != handle.digest:
                raise ValueError(f"Mooncake artifact key already exists with different payload: {handle.key}")

        existing = self._metadata._artifacts.get(offloaded_manifest.artifact_id)
        if existing is not None and existing.status is RolloutArtifactStatus.READY:
            if existing.manifest != offloaded_manifest or not self._payload_store.exists(handle.key):
                raise ValueError(f"artifact_id already exists: {offloaded_manifest.artifact_id}")
            return

        self._metadata.reserve(offloaded_manifest, handle)

    def commit(self, manifest: RolloutArtifactManifest, payload: Any) -> None:
        """Promote a reserved artifact after its immutable payload is durable."""
        offloaded_manifest, handle = self._offloaded_manifest_and_handle(manifest, payload)
        if not self._payload_store.exists(handle.key):
            raise ValueError(f"Mooncake artifact payload is missing before commit: {handle.key}")
        existing_payload = self._payload_store.get(handle.key)
        if hashlib.sha256(existing_payload).hexdigest() != handle.digest:
            raise ValueError(f"Mooncake artifact payload digest mismatch before commit: {handle.key}")
        self._metadata.commit(offloaded_manifest.artifact_id, handle)

    def publish(self, manifest: RolloutArtifactManifest, payload: Any) -> None:
        """Reserve metadata, write the immutable payload, then commit READY."""
        offloaded_manifest, handle = self._offloaded_manifest_and_handle(manifest, payload)
        self.reserve(offloaded_manifest, payload)
        if not self._payload_store.exists(handle.key):
            self._payload_store.put(handle.key, payload)
        self.commit(offloaded_manifest, payload)


    def claim_complete_group(
        self,
        *,
        accepted_policy_versions: set[str],
        lease_timeout_seconds: float,
        accepted_environment_versions: set[str] | None = None,
        accepted_tool_versions: set[str] | None = None,
        require_current_producer_generation: bool = False,
    ) -> ArtifactLease | None:
        self._payload_store.retry_pending_removals()
        lease = self._metadata.claim_complete_group(
            accepted_policy_versions=accepted_policy_versions,
            lease_timeout_seconds=lease_timeout_seconds,
            accepted_environment_versions=accepted_environment_versions,
            accepted_tool_versions=accepted_tool_versions,
            require_current_producer_generation=require_current_producer_generation,
        )
        try:
            return None if lease is None else self._load_payloads(lease)
        finally:
            self.cleanup_rejected_payloads()

    def claim_group_for_artifact(
        self,
        artifact_id: str,
        *,
        accepted_policy_versions: set[str],
        lease_timeout_seconds: float,
        accepted_environment_versions: set[str] | None = None,
        accepted_tool_versions: set[str] | None = None,
        require_current_producer_generation: bool = False,
    ) -> ArtifactLease | None:
        self._payload_store.retry_pending_removals()
        lease = self._metadata.claim_group_for_artifact(
            artifact_id,
            accepted_policy_versions=accepted_policy_versions,
            lease_timeout_seconds=lease_timeout_seconds,
            accepted_environment_versions=accepted_environment_versions,
            accepted_tool_versions=accepted_tool_versions,
            require_current_producer_generation=require_current_producer_generation,
        )
        try:
            return None if lease is None else self._load_payloads(lease)
        finally:
            self.cleanup_rejected_payloads()

    def _load_payloads(self, lease: ArtifactLease) -> ArtifactLease:
        payloads: list[bytes] = []
        try:
            for manifest in lease.manifests:
                handle = self._payload_handle(manifest)
                payload = self._payload_store.get(handle.key)
                digest = hashlib.sha256(payload).hexdigest()
                if digest != handle.digest:
                    raise RuntimeError(f"payload digest mismatch for Mooncake artifact {handle.key}")
                payloads.append(payload)
        except Exception:
            # A claimed group must not remain leased when its payload is missing or
            # corrupted.  Release the metadata lease before surfacing the failure.
            self._metadata.release(lease)
            raise
        return replace(lease, payloads=tuple(payloads))

    def acknowledge(self, lease: ArtifactLease) -> None:
        self._metadata.acknowledge(lease)
        for manifest in lease.manifests:
            self._payload_store.remove(self._payload_handle(manifest).key)

    def release(self, lease: ArtifactLease) -> None:
        # The payload is immutable, so a failed consumer can claim it again after
        # the lease expires or is explicitly released.
        self._metadata.release(lease)

    def reject_unaccepted_policy_versions(self, accepted_policy_versions: set[str]) -> int:
        try:
            return self._metadata.reject_unaccepted_policy_versions(accepted_policy_versions)
        finally:
            self.cleanup_rejected_payloads()

    def cleanup_rejected_payloads(self) -> int:
        """Idempotently remove payload objects whose metadata is REJECTED."""
        handles = [
            artifact.payload
            for artifact in self._metadata._artifacts.values()
            if artifact.status is RolloutArtifactStatus.REJECTED
            and isinstance(artifact.payload, _MooncakePayloadHandle)
        ]
        for handle in handles:
            self._payload_store.remove(handle.key)
        self._payload_store.retry_pending_removals()
        return len(handles)

    def _handoff_snapshot_key(self, handoff_id: str) -> str:
        if not isinstance(handoff_id, str) or not handoff_id or "/" in handoff_id:
            raise ValueError("handoff_id must be a non-empty path-safe string")
        return f"{self._namespace}/_handoff/{handoff_id}"

    def export_metadata_snapshot(self) -> dict[str, Any]:
        return self._metadata.export_metadata_snapshot()

    def persist_handoff_snapshot(self, handoff_id: str) -> str:
        """Persist a small metadata checkpoint in the same shared Mooncake tier."""
        key = self._handoff_snapshot_key(handoff_id)
        envelope = {
            "schema_version": 2,
            "handoff_id": handoff_id,
            "metadata_snapshot": self.export_metadata_snapshot(),
        }
        payload = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if self._payload_store.exists(key):
            if self._payload_store.get(key) != payload:
                raise ValueError(f"handoff snapshot already exists with different state: {handoff_id}")
            return key
        self._payload_store.put(key, payload)
        return key

    def restore_handoff_snapshot(self, handoff_id: str) -> None:
        key = self._handoff_snapshot_key(handoff_id)
        payload = self._payload_store.get(key)
        try:
            envelope = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("handoff snapshot is not valid JSON") from exc
        if (
            not isinstance(envelope, dict)
            or envelope.get("schema_version") != 2
            or envelope.get("handoff_id") != handoff_id
            or "metadata_snapshot" not in envelope
        ):
            raise ValueError("handoff snapshot has an invalid schema")
        self.restore_metadata_snapshot(envelope["metadata_snapshot"])

    def delete_handoff_snapshot(self, handoff_id: str) -> None:
        self._payload_store.remove(self._handoff_snapshot_key(handoff_id))

    def restore_metadata_snapshot(self, snapshot: Any) -> None:
        # Snapshot metadata is JSON-compatible by contract; copy it so recovery
        # reconciliation never mutates the caller's checkpoint object.
        recovered = json.loads(json.dumps(snapshot))
        artifact_state: dict[str, tuple[RolloutArtifactStatus, _MooncakePayloadHandle | None]] = {}

        for artifact_snapshot in recovered.get("artifacts", []):
            manifest = _decode_manifest(artifact_snapshot["manifest"])
            try:
                status = RolloutArtifactStatus(artifact_snapshot["status"])
            except ValueError:
                # Let the canonical metadata restore produce the schema error.
                status = RolloutArtifactStatus.REJECTED

            handle = self._payload_handle(manifest)
            if status in {
                RolloutArtifactStatus.WRITING,
                RolloutArtifactStatus.READY,
                RolloutArtifactStatus.LEASED,
            }:
                reason = None
                if not self._payload_store.exists(handle.key):
                    reason = "payload_missing_after_recovery"
                else:
                    payload = self._payload_store.get(handle.key)
                    if hashlib.sha256(payload).hexdigest() != handle.digest:
                        reason = "payload_corrupt_after_recovery"

                if reason is not None:
                    artifact_snapshot["status"] = RolloutArtifactStatus.REJECTED.value
                    artifact_snapshot["rejected_reason"] = reason
                    status = RolloutArtifactStatus.REJECTED
                elif status is RolloutArtifactStatus.WRITING:
                    # The immutable object is complete; recovery can finish the
                    # interrupted metadata commit without replaying generation.
                    artifact_snapshot["status"] = RolloutArtifactStatus.READY.value
                    artifact_snapshot["rejected_reason"] = None
                    status = RolloutArtifactStatus.READY

            artifact_state[manifest.artifact_id] = (status, handle)

        def resolve_payload(manifest: RolloutArtifactManifest) -> Any:
            status, handle = artifact_state[manifest.artifact_id]
            return handle if status in {RolloutArtifactStatus.READY, RolloutArtifactStatus.LEASED} else None

        self._metadata.restore_metadata_snapshot(recovered, payload_resolver=resolve_payload)
        for manifest, (status, handle) in artifact_state.items():
            if status in {RolloutArtifactStatus.CONSUMED, RolloutArtifactStatus.REJECTED} and handle is not None:
                self._payload_store.remove(handle.key)
        self._payload_store.retry_pending_removals()

    def get_status(self, artifact_id: str) -> RolloutArtifactStatus:
        return self._metadata.get_status(artifact_id)

    def get_metrics(self) -> dict[str, int]:
        self._payload_store.retry_pending_removals()
        metrics = self._metadata.get_metrics()
        metrics["mooncake_pending_removals"] = self._payload_store.pending_removal_count
        return metrics

    def close(self) -> None:
        self._payload_store.close()


def begin_transfer_admission_safely(
    controller: TransferAdmissionController | None, request: TransferRequest
) -> dict[str, Any] | None:
    """Record a shadow admission decision without making it a data-plane dependency."""
    if controller is None:
        return None
    try:
        return controller.begin(request).to_dict()
    except Exception as exc:
        print(f"TRANSFER_ADMISSION_SHADOW_ERROR action=begin error={exc!r}", flush=True)
        return None


def complete_transfer_admission_safely(
    controller: TransferAdmissionController | None, transfer_id: str
) -> None:
    if controller is None:
        return
    try:
        controller.complete(transfer_id)
    except Exception as exc:
        print(f"TRANSFER_ADMISSION_SHADOW_ERROR action=complete error={exc!r}", flush=True)


def create_ray_rollout_artifact_store_actor(
    *,
    backend: str = "ray_memory",
    mooncake_config: dict[str, Any] | None = None,
    transfer_admission_config: dict[str, Any] | None = None,
    trajectory_snapshot_enabled: bool = False,
    optional_restore_admission_config: dict[str, Any] | None = None,
    optional_restore_resource_domain: str | None = None,
):
    """Create the optional Ray actor class without making the contract depend on Ray.

    Keeping the import here preserves fast, dependency-light unit tests for the
    state machine.  Production callers create this actor once in the Fully Async
    task runner and pass its handle to rollout and update actors.
    """
    import ray

    # Correctness baseline for P1: producer registration, artifact publication,
    # and claims must have one deterministic linearization order. P2 may split
    # metadata critical sections from payload I/O, but not before persistence
    # and reconciliation semantics are defined.
    @ray.remote(num_cpus=1, max_concurrency=1)
    class RayRolloutArtifactStore:
        def __init__(self):
            if backend == "ray_memory":
                self._store = InMemoryRolloutArtifactStore()
            elif backend == "mooncake":
                allowed_keys = {
                    "namespace",
                    "local_hostname",
                    "metadata_server",
                    "global_segment_size",
                    "local_buffer_size",
                    "protocol",
                    "device_name",
                    "master_server_address",
                    "preferred_segment",
                }
                config = {key: value for key, value in (mooncake_config or {}).items() if key in allowed_keys}
                self._store = MooncakeRolloutArtifactStore(**config)
            else:
                raise ValueError(f"unsupported rollout artifact backend: {backend}")
            self._transfer_admission: TransferAdmissionController | None = None
            if transfer_admission_config is not None:
                admission = TransferAdmissionConfig.from_mapping(transfer_admission_config)
                if not admission.shadow:
                    raise ValueError("online transfer admission is not implemented; enable shadow mode")
                self._transfer_admission = TransferAdmissionController(admission)
            if type(trajectory_snapshot_enabled) is not bool:
                raise ValueError("trajectory_snapshot_enabled must be a bool")
            self._trajectory_snapshots = None
            if trajectory_snapshot_enabled:
                from verl.experimental.fully_async_policy.trajectory_snapshot import TrajectorySnapshotRegistry
                from verl.experimental.fully_async_policy.preemptible_handoff import (
                    HandoffManifestRegistry,
                    HandoffMemberRecordStore,
                    HandoffRestoreCoordinator,
                    TrajectoryPreemptionTracker,
                )

                if backend == "mooncake":
                    snapshots = MooncakeRolloutArtifactStore(
                        namespace=self._store._namespace + "/inflight",
                        payload_store=self._store._payload_store,
                    )
                else:
                    snapshots = InMemoryRolloutArtifactStore()
                self._trajectory_snapshots = TrajectorySnapshotRegistry(snapshots)
                # Cooperative preemption + transactional handoff manifests live
                # on the same serialized actor; ownership stays fenced by the
                # producer-generation registry (no second epoch system).
                self._trajectory_preemption = TrajectoryPreemptionTracker()
                self._handoff_manifests = HandoffManifestRegistry(
                    self._trajectory_preemption,
                    current_producer_generation=self._store.get_current_producer_generation,
                )
                self._handoff_member_records = HandoffMemberRecordStore()
                self._handoff_restore = HandoffRestoreCoordinator(
                    self._handoff_manifests, self._trajectory_snapshots,
                    self._handoff_member_records, self._store)
            # Separate opt-in resource domain for optional KV movement. Required
            # artifact and weight operations keep their existing shadow path.
            self._optional_restore_admission = None
            if optional_restore_admission_config is not None:
                if not isinstance(optional_restore_resource_domain, str) or not optional_restore_resource_domain:
                    raise ValueError("optional restore admission requires a physical resource domain")
                self._optional_restore_admission = TransferAdmissionController(
                    TransferAdmissionConfig.from_mapping(optional_restore_admission_config)
                )

        def _snapshot_registry(self):
            if self._trajectory_snapshots is None:
                raise ValueError("in-flight trajectory snapshots are disabled")
            return self._trajectory_snapshots

        def open_trajectory(self, trajectory_id, owner_id, lease_seconds):
            return self._snapshot_registry().open(trajectory_id, owner_id, lease_seconds)

        def commit_trajectory_snapshot(self, owner_lease, snapshot, ttl_seconds):
            return self._snapshot_registry().commit(owner_lease, snapshot, ttl_seconds)

        def load_trajectory_snapshot(self, owner_lease):
            return self._snapshot_registry().load(owner_lease)

        def takeover_trajectory(self, trajectory_id, expected_generation, owner_id, lease_seconds):
            return self._snapshot_registry().takeover(
                trajectory_id, expected_generation, owner_id, lease_seconds
            )

        def check_trajectory_owner(self, owner_lease):
            self._snapshot_registry()._check(owner_lease)

        def renew_trajectory_owner(self, owner_lease, lease_seconds):
            return self._snapshot_registry().renew(owner_lease, lease_seconds)

        def release_trajectory_owner(self, owner_lease):
            self._snapshot_registry().release_owner(owner_lease)

        def finish_trajectory(self, owner_lease):
            self._snapshot_registry().finish(owner_lease)

        def collect_expired_trajectory_snapshots(self):
            return self._snapshot_registry().collect_expired()

        # ---------------- cooperative preemption + handoff manifests ----------

        def _preemption_tracker(self):
            if self._trajectory_snapshots is None:
                raise ValueError("in-flight trajectory snapshots are disabled")
            return self._trajectory_preemption

        def _handoff_registry(self):
            if self._trajectory_snapshots is None:
                raise ValueError("in-flight trajectory snapshots are disabled")
            return self._handoff_manifests

        def register_running_trajectory(self, trajectory_id):
            self._preemption_tracker().register_running(trajectory_id)

        def get_trajectory_preemption(self, trajectory_id):
            return self._preemption_tracker().state(trajectory_id).value

        def request_trajectory_preemption(self, trajectory_id):
            record = self._preemption_tracker().request_preemption(trajectory_id)
            return record.state.value

        def mark_trajectory_safe_point(self, trajectory_id):
            record = self._preemption_tracker().mark_safe_point(trajectory_id)
            return record.state.value

        def mark_trajectory_snapshot_committed(self, trajectory_id, snapshot_ref):
            record = self._preemption_tracker().mark_snapshot_committed(trajectory_id, snapshot_ref)
            return record.state.value

        def mark_trajectory_handed_off(self, trajectory_id):
            record = self._preemption_tracker().mark_handed_off(trajectory_id)
            return record.state.value

        def mark_trajectory_aborted(self, trajectory_id):
            record = self._preemption_tracker().mark_aborted(trajectory_id)
            return record.state.value

        def discard_trajectory_preemption(self, trajectory_id):
            self._preemption_tracker().discard(trajectory_id)

        def get_trajectory_preemption_metrics(self):
            return self._preemption_tracker().metrics()

        def prepare_handoff_manifest(self, manifest):
            from verl.experimental.fully_async_policy.preemptible_handoff import HandoffManifest

            if not isinstance(manifest, HandoffManifest):
                raise ValueError("handoff manifest object required")
            record = self._handoff_registry().prepare(manifest)
            return {"handoff_id": record.manifest.handoff_id, "state": record.state.value}

        def commit_handoff_manifest(self, handoff_id):
            record = self._handoff_registry().commit(handoff_id)
            return record.state.value

        def record_handoff_source_release(self, handoff_id):
            record = self._handoff_registry().record_source_release(handoff_id)
            return record.state.value

        def begin_handoff_restore(self, handoff_id):
            record = self._handoff_registry().begin_restore(handoff_id)
            return record.state.value

        def finish_handoff_manifest(self, handoff_id, artifact_id):
            self._handoff_registry()
            record = self._handoff_restore.finish(handoff_id, artifact_id)
            return record.state.value

        def find_handoff_artifact(self, handoff_id, artifact_id):
            self._handoff_registry()
            return self._handoff_restore.find_artifact(handoff_id, artifact_id)

        def acquire_handoff_trajectory(self, handoff_id, trajectory_id, owner_id, lease_seconds):
            self._handoff_registry()
            return self._handoff_restore.acquire(handoff_id, trajectory_id, owner_id, lease_seconds)

        def acquire_handoff_pending(self, handoff_id, trajectory_id, owner_id, lease_seconds):
            self._handoff_registry()
            return self._handoff_restore.acquire_pending(handoff_id, trajectory_id, owner_id, lease_seconds)

        def record_handoff_member_result(self, handoff_id, member_index, owner_lease, payload, metrics):
            self._handoff_registry()
            return self._handoff_restore.record_result(handoff_id, member_index, owner_lease, payload, metrics)

        def get_handoff_restored_members(self, handoff_id):
            self._handoff_registry()
            return self._handoff_restore.restored(handoff_id)

        def cancel_handoff_manifest(self, handoff_id):
            self._handoff_registry().cancel(handoff_id)

        def get_handoff_manifest(self, handoff_id):
            return self._handoff_registry().get(handoff_id).manifest

        def get_handoff_metrics(self):
            return self._handoff_registry().metrics()

        def put_handoff_member_record(self, payload):
            if self._trajectory_snapshots is None:
                raise ValueError("in-flight trajectory snapshots are disabled")
            return self._handoff_member_records.put(payload)

        def get_handoff_member_record(self, reference):
            if self._trajectory_snapshots is None:
                raise ValueError("in-flight trajectory snapshots are disabled")
            return self._handoff_member_records.get(reference)

        def discard_handoff_member_record(self, reference):
            if self._trajectory_snapshots is None:
                raise ValueError("in-flight trajectory snapshots are disabled")
            self._handoff_member_records.discard(reference)

        def begin_optional_restore(self, request: TransferRequest):
            if request.intent not in {TransferIntent.REMOTE_KV_RESTORE, TransferIntent.PREFETCH}:
                raise ValueError("execution admission is restricted to optional KV transfers")
            if self._optional_restore_admission is None:
                return None
            if request.metadata.get("resource_domain") != optional_restore_resource_domain:
                raise ValueError("optional transfer targets a different resource domain")
            return self._optional_restore_admission.begin(request).to_dict()

        def complete_optional_restore(self, transfer_id: str):
            if self._optional_restore_admission is not None:
                self._optional_restore_admission.complete(transfer_id)

        def get_optional_restore_admission_summary(self):
            if self._optional_restore_admission is None:
                return {}
            return self._optional_restore_admission.summary()

        def get_optional_restore_admission_records(self):
            if self._optional_restore_admission is None:
                return []
            return [record.to_dict() for record in self._optional_restore_admission.records]

        def _begin_transfer_admission(self, request: TransferRequest) -> dict[str, Any] | None:
            return begin_transfer_admission_safely(self._transfer_admission, request)

        def _complete_transfer_admission(self, transfer_id: str) -> None:
            complete_transfer_admission_safely(self._transfer_admission, transfer_id)

        def begin_weight_sync_admission(self, request: TransferRequest) -> dict[str, Any] | None:
            return self._begin_transfer_admission(request)

        def complete_transfer_admission(self, transfer_id: str) -> None:
            self._complete_transfer_admission(transfer_id)

        def get_transfer_admission_summary(self) -> dict[str, float]:
            if self._transfer_admission is None:
                return {}
            return self._transfer_admission.summary()

        def get_transfer_admission_records(self) -> list[dict[str, Any]]:
            if self._transfer_admission is None:
                return []
            return [record.to_dict() for record in self._transfer_admission.records]

        def _claim_transfer_request(self, claim: str) -> TransferRequest:
            return TransferRequest(
                transfer_id=f"rollout-artifact-read:{claim}:{time.monotonic_ns()}",
                intent=TransferIntent.ROLLOUT_ARTIFACT,
                bytes=0,
                metadata={
                    "direction": "read",
                    "claim": claim,
                },
            )

        def publish(self, manifest: RolloutArtifactManifest, payload: Any) -> None:
            request = TransferRequest(
                transfer_id=f"rollout-artifact:{manifest.artifact_id}",
                intent=TransferIntent.ROLLOUT_ARTIFACT,
                bytes=len(payload) if isinstance(payload, bytes) else 0,
                policy_version=manifest.identity.policy_version,
                producer_generation=(
                    manifest.training_semantics.producer_generation
                    if manifest.training_semantics is not None
                    else None
                ),
                metadata={"artifact_id": manifest.artifact_id},
            )
            self._begin_transfer_admission(request)
            try:
                self._store.publish(manifest, payload)
            finally:
                self._complete_transfer_admission(request.transfer_id)

        def register_producer(self, producer_id: str, incarnation_id: str) -> int:
            return self._store.register_producer(producer_id, incarnation_id)

        def get_current_producer_generation(self, producer_id: str) -> int | None:
            return self._store.get_current_producer_generation(producer_id)

        def claim_complete_group(
            self, accepted_policy_versions: set[str], lease_timeout_seconds: float,
            accepted_environment_versions: set[str] | None = None, accepted_tool_versions: set[str] | None = None,
            require_current_producer_generation: bool = False,
        ) -> ArtifactLease | None:
            request = self._claim_transfer_request("complete_group")
            self._begin_transfer_admission(request)
            try:
                return self._store.claim_complete_group(
                    accepted_policy_versions=accepted_policy_versions,
                    lease_timeout_seconds=lease_timeout_seconds,
                    accepted_environment_versions=accepted_environment_versions,
                    accepted_tool_versions=accepted_tool_versions,
                    require_current_producer_generation=require_current_producer_generation,
                )
            finally:
                self._complete_transfer_admission(request.transfer_id)

        def claim_group_for_artifact(
            self, artifact_id: str, accepted_policy_versions: set[str], lease_timeout_seconds: float,
            accepted_environment_versions: set[str] | None = None, accepted_tool_versions: set[str] | None = None,
            require_current_producer_generation: bool = False,
        ) -> ArtifactLease | None:
            request = self._claim_transfer_request(artifact_id)
            self._begin_transfer_admission(request)
            try:
                return self._store.claim_group_for_artifact(
                    artifact_id,
                    accepted_policy_versions=accepted_policy_versions,
                    lease_timeout_seconds=lease_timeout_seconds,
                    accepted_environment_versions=accepted_environment_versions,
                    accepted_tool_versions=accepted_tool_versions,
                    require_current_producer_generation=require_current_producer_generation,
                )
            finally:
                self._complete_transfer_admission(request.transfer_id)

        def acknowledge(self, lease: ArtifactLease) -> None:
            self._store.acknowledge(lease)

        def release(self, lease: ArtifactLease) -> None:
            self._store.release(lease)

        def export_metadata_snapshot(self) -> dict[str, Any]:
            return self._store.export_metadata_snapshot()

        def restore_metadata_snapshot(self, snapshot: Any) -> None:
            self._store.restore_metadata_snapshot(snapshot)

        def persist_handoff_snapshot(self, handoff_id: str) -> str:
            method = getattr(self._store, "persist_handoff_snapshot", None)
            if method is None:
                raise RuntimeError("persisted rollout state handoff requires the Mooncake artifact backend")
            return method(handoff_id)

        def restore_handoff_snapshot(self, handoff_id: str) -> None:
            method = getattr(self._store, "restore_handoff_snapshot", None)
            if method is None:
                raise RuntimeError("persisted rollout state handoff requires the Mooncake artifact backend")
            method(handoff_id)

        def delete_handoff_snapshot(self, handoff_id: str) -> None:
            method = getattr(self._store, "delete_handoff_snapshot", None)
            if method is None:
                raise RuntimeError("persisted rollout state handoff requires the Mooncake artifact backend")
            method(handoff_id)

        def get_metrics(self) -> dict[str, int | float]:
            metrics = self._store.get_metrics()
            metrics.update(self.get_transfer_admission_summary())
            return metrics

        def get_status(self, artifact_id: str) -> RolloutArtifactStatus:
            return self._store.get_status(artifact_id)

        def close(self) -> None:
            if hasattr(self._store, "close"):
                self._store.close()

    return RayRolloutArtifactStore


class RayRolloutArtifactStoreClient:
    """Async client for the Ray-backed reference data plane."""

    def __init__(self, store_actor: Any):
        self.store_actor = store_actor

    async def _call(self, method: str, *args: Any) -> Any:
        future = getattr(self.store_actor, method).remote(*args)
        return await asyncio.wrap_future(future.future())

    async def publish(self, manifest: RolloutArtifactManifest, payload: Any) -> None:
        await self._call("publish", manifest, payload)

    async def begin_weight_sync_admission(self, request: TransferRequest) -> dict[str, Any] | None:
        return await self._call("begin_weight_sync_admission", request)

    async def complete_transfer_admission(self, transfer_id: str) -> None:
        await self._call("complete_transfer_admission", transfer_id)

    async def get_transfer_admission_summary(self) -> dict[str, float]:
        return await self._call("get_transfer_admission_summary")

    async def get_transfer_admission_records(self) -> list[dict[str, Any]]:
        return await self._call("get_transfer_admission_records")

    async def register_producer(self, producer_id: str, incarnation_id: str) -> int:
        return await self._call("register_producer", producer_id, incarnation_id)

    async def get_current_producer_generation(self, producer_id: str) -> int | None:
        return await self._call("get_current_producer_generation", producer_id)

    async def claim_complete_group(
        self, *, accepted_policy_versions: set[str], lease_timeout_seconds: float,
        accepted_environment_versions: set[str] | None = None, accepted_tool_versions: set[str] | None = None,
        require_current_producer_generation: bool = False,
    ) -> ArtifactLease | None:
        if (
            accepted_environment_versions is None
            and accepted_tool_versions is None
            and not require_current_producer_generation
        ):
            return await self._call("claim_complete_group", accepted_policy_versions, lease_timeout_seconds)
        return await self._call(
            "claim_complete_group",
            accepted_policy_versions,
            lease_timeout_seconds,
            accepted_environment_versions,
            accepted_tool_versions,
            require_current_producer_generation,
        )

    async def claim_group_for_artifact(
        self, artifact_id: str, *, accepted_policy_versions: set[str], lease_timeout_seconds: float,
        accepted_environment_versions: set[str] | None = None, accepted_tool_versions: set[str] | None = None,
        require_current_producer_generation: bool = False,
    ) -> ArtifactLease | None:
        if (
            accepted_environment_versions is None
            and accepted_tool_versions is None
            and not require_current_producer_generation
        ):
            return await self._call("claim_group_for_artifact", artifact_id, accepted_policy_versions, lease_timeout_seconds)
        return await self._call(
            "claim_group_for_artifact",
            artifact_id,
            accepted_policy_versions,
            lease_timeout_seconds,
            accepted_environment_versions,
            accepted_tool_versions,
            require_current_producer_generation,
        )

    async def acknowledge(self, lease: ArtifactLease) -> None:
        await self._call("acknowledge", lease)

    async def release(self, lease: ArtifactLease) -> None:
        await self._call("release", lease)

    async def export_metadata_snapshot(self) -> dict[str, Any]:
        return await self._call("export_metadata_snapshot")

    async def restore_metadata_snapshot(self, snapshot: Any) -> None:
        await self._call("restore_metadata_snapshot", snapshot)

    async def persist_handoff_snapshot(self, handoff_id: str) -> str:
        return await self._call("persist_handoff_snapshot", handoff_id)

    async def restore_handoff_snapshot(self, handoff_id: str) -> None:
        await self._call("restore_handoff_snapshot", handoff_id)

    async def delete_handoff_snapshot(self, handoff_id: str) -> None:
        await self._call("delete_handoff_snapshot", handoff_id)

    # -------- cooperative preemption + transactional handoff manifests --------

    async def open_trajectory(self, trajectory_id, owner_id, lease_seconds):
        return await self._call("open_trajectory", trajectory_id, owner_id, lease_seconds)

    async def commit_trajectory_snapshot(self, owner_lease, snapshot, ttl_seconds):
        return await self._call("commit_trajectory_snapshot", owner_lease, snapshot, ttl_seconds)

    async def load_trajectory_snapshot(self, owner_lease):
        return await self._call("load_trajectory_snapshot", owner_lease)

    async def takeover_trajectory(self, trajectory_id, expected_generation, owner_id, lease_seconds):
        return await self._call("takeover_trajectory", trajectory_id, expected_generation, owner_id, lease_seconds)

    async def check_trajectory_owner(self, owner_lease):
        return await self._call("check_trajectory_owner", owner_lease)

    async def renew_trajectory_owner(self, owner_lease, lease_seconds):
        return await self._call("renew_trajectory_owner", owner_lease, lease_seconds)

    async def release_trajectory_owner(self, owner_lease):
        return await self._call("release_trajectory_owner", owner_lease)

    async def finish_trajectory(self, owner_lease):
        return await self._call("finish_trajectory", owner_lease)

    async def collect_expired_trajectory_snapshots(self):
        return await self._call("collect_expired_trajectory_snapshots")

    async def register_running_trajectory(self, trajectory_id: str) -> None:
        await self._call("register_running_trajectory", trajectory_id)

    async def get_trajectory_preemption(self, trajectory_id: str) -> str:
        return await self._call("get_trajectory_preemption", trajectory_id)

    async def request_trajectory_preemption(self, trajectory_id: str) -> str:
        return await self._call("request_trajectory_preemption", trajectory_id)

    async def mark_trajectory_safe_point(self, trajectory_id: str) -> str:
        return await self._call("mark_trajectory_safe_point", trajectory_id)

    async def mark_trajectory_snapshot_committed(self, trajectory_id: str, snapshot_ref: str) -> str:
        return await self._call("mark_trajectory_snapshot_committed", trajectory_id, snapshot_ref)

    async def mark_trajectory_handed_off(self, trajectory_id: str) -> str:
        return await self._call("mark_trajectory_handed_off", trajectory_id)

    async def mark_trajectory_aborted(self, trajectory_id: str) -> str:
        return await self._call("mark_trajectory_aborted", trajectory_id)

    async def discard_trajectory_preemption(self, trajectory_id: str) -> None:
        await self._call("discard_trajectory_preemption", trajectory_id)

    async def get_trajectory_preemption_metrics(self) -> dict[str, Any]:
        return await self._call("get_trajectory_preemption_metrics")

    async def prepare_handoff_manifest(self, manifest: Any) -> dict[str, Any]:
        return await self._call("prepare_handoff_manifest", manifest)

    async def commit_handoff_manifest(self, handoff_id: str) -> str:
        return await self._call("commit_handoff_manifest", handoff_id)

    async def record_handoff_source_release(self, handoff_id: str) -> str:
        return await self._call("record_handoff_source_release", handoff_id)

    async def begin_handoff_restore(self, handoff_id: str) -> str:
        return await self._call("begin_handoff_restore", handoff_id)

    async def finish_handoff_manifest(self, handoff_id: str, artifact_id: str) -> str:
        return await self._call("finish_handoff_manifest", handoff_id, artifact_id)

    async def find_handoff_artifact(self, handoff_id: str, artifact_id: str):
        return await self._call("find_handoff_artifact", handoff_id, artifact_id)

    async def acquire_handoff_trajectory(self, handoff_id, trajectory_id, owner_id, lease_seconds):
        return await self._call("acquire_handoff_trajectory", handoff_id, trajectory_id, owner_id, lease_seconds)

    async def acquire_handoff_pending(self, handoff_id, trajectory_id, owner_id, lease_seconds):
        return await self._call("acquire_handoff_pending", handoff_id, trajectory_id, owner_id, lease_seconds)

    async def record_handoff_member_result(self, handoff_id, member_index, owner_lease, payload, metrics):
        return await self._call("record_handoff_member_result", handoff_id, member_index, owner_lease, payload, metrics)

    async def get_handoff_restored_members(self, handoff_id):
        return await self._call("get_handoff_restored_members", handoff_id)

    async def cancel_handoff_manifest(self, handoff_id: str) -> None:
        await self._call("cancel_handoff_manifest", handoff_id)

    async def get_handoff_manifest(self, handoff_id: str) -> Any:
        return await self._call("get_handoff_manifest", handoff_id)

    async def get_handoff_metrics(self) -> dict[str, Any]:
        return await self._call("get_handoff_metrics")

    async def put_handoff_member_record(self, payload: bytes) -> str:
        return await self._call("put_handoff_member_record", payload)

    async def get_handoff_member_record(self, reference: str) -> bytes:
        return await self._call("get_handoff_member_record", reference)

    async def discard_handoff_member_record(self, reference: str) -> None:
        await self._call("discard_handoff_member_record", reference)

    async def get_metrics(self) -> dict[str, int | float]:
        return await self._call("get_metrics")
