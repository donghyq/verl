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

"""Verify a restored group's stock V1 fields through public TransferQueue APIs.

The caller supplies fields/tags already prepared by AgentLoopWorkerTQ and its
training-semantics validator. No reward, behavior logprob or trajectory is
recomputed here. A dispatched publication is reconciled by reading the real
queue; member data are never resent. The stock uid's finished tag must also be
observed and durably confirmed before finish/GC. This is neither durable queue
replay nor an atomic optimizer-update/ack transaction.

Usage::

    result = await publish_trajectory_group(
        coordinator=trajectory_client,
        handoff_id=manifest.handoff_id,
        members=[PublicationMember(uid, i, 0) for i in range(8)],
        fields=original_postprocessed_tensordict,
        tags=original_postprocessed_tags,
        partition_id="train",
        validate_fields=original_training_semantics_validator,
    )

The validator takes ``(fields, members)`` and returns None/True on success or
raises on failure. It must check the original policy/version semantics. Its
return value, a PUT result or tags alone cannot replace actual payload GET.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import struct
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDictBase

from verl.trainer.ppo.v1.trajectory_state.handoff import HandoffState


TQ_VERSION = "0.1.10"


class PublicationValidationError(ValueError):
    """Actual training data does not match its frozen publication identity."""


@dataclass(frozen=True)
class PublicationMember:
    uid: str
    session_id: int
    output_index: int = 0

    @property
    def key(self) -> str:
        return f"{self.uid}_{self.session_id}_{self.output_index}"


def _canonical_value(value: Any) -> Any:
    """Canonicalize container order without changing tensor dtype or contents."""
    if isinstance(value, NonTensorStack):
        return [_canonical_value(item) for item in value.tolist()]
    if isinstance(value, NonTensorData):
        return _canonical_value(value.data)
    if isinstance(value, TensorDictBase):
        return {key: _canonical_value(value.get(key)) for key in sorted(value.keys())}
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {"numpy_object_dtype": str(value.dtype), "shape": list(value.shape),
                    "data": _canonical_value(value.tolist())}
        return value
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PublicationValidationError("Training metadata keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, bytes, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise PublicationValidationError("Training metadata has an unsupported or nonfinite value")


def _logical_rows(fields: TensorDictBase) -> list[dict[str, Any]]:
    if not isinstance(fields, TensorDictBase) or list(fields.batch_size) != [8]:
        raise PublicationValidationError("Expected one stock TensorDict batch of eight members")
    if not fields.keys() or any(not isinstance(key, str) for key in fields.keys()):
        raise PublicationValidationError("Expected nonempty stock V1 field names")
    return [_canonical_value(fields[i]) for i in range(8)]


def training_payload_digest(
    *, keys: Sequence[str], partition_id: str, fields: TensorDictBase, tags: Sequence[dict]
) -> str:
    """Hash complete logical rows with TQ's original tensor serializer.

    Per-row normalization makes the hash independent of regular/jagged batch
    storage, dictionary insertion order and tensor backing-storage addresses.
    The published TensorDict itself keeps the original V1 layout.
    """
    from transfer_queue.utils.serial_utils import encode

    logical = {
        "schema_version": 1, "keys": list(keys), "partition_id": partition_id,
        "fields": _logical_rows(fields), "tags": _canonical_value(list(tags)),
    }
    hasher = hashlib.sha256(b"verl-v1-tq-training-fields-v1\x00")
    for frame in encode(logical):
        raw = memoryview(frame).cast("B")
        hasher.update(struct.pack(">Q", raw.nbytes))
        hasher.update(raw)
    return hasher.hexdigest()


def _validate_members(members: Sequence[PublicationMember], rows: list[dict[str, Any]]) -> tuple[PublicationMember, ...]:
    members = tuple(members)
    if len(members) != 8 or any(not isinstance(member, PublicationMember) for member in members):
        raise PublicationValidationError("Expected exactly eight original member bindings")
    if (len({member.uid for member in members}) != 1
            or any(not isinstance(member.uid, str) or not member.uid for member in members)
            or any(type(member.session_id) is not int or type(member.output_index) is not int
                   or member.session_id < 0 or member.output_index < 0 for member in members)
            or sorted(member.session_id for member in members) != list(range(8))):
        raise PublicationValidationError("Invalid original uid/session/output membership")
    for member, row in zip(members, rows, strict=True):
        session = row.get("session_id")
        if isinstance(session, torch.Tensor) and session.numel() == 1:
            session = session.item()
        if row.get("uid") != member.uid or type(session) is not int or session != member.session_id:
            raise PublicationValidationError("Actual row does not match its original member key")
        required = ("prompts", "responses", "response_mask", "loss_mask", "rollout_log_probs", "input_ids")
        if any(not isinstance(row.get(name), torch.Tensor) or row[name].ndim != 1 for name in required):
            raise PublicationValidationError("Incomplete stock text training fields")
        prompt, response, mask = row["prompts"], row["responses"], row["response_mask"]
        if (prompt.dtype != torch.int64 or response.dtype != torch.int64 or row["input_ids"].dtype != torch.int64
                or not row["rollout_log_probs"].is_floating_point()
                or response.numel() == 0 or mask.numel() != response.numel()
                or row["rollout_log_probs"].numel() != response.numel()
                or not torch.equal(row["loss_mask"], mask)
                or not bool(torch.all((mask == 0) | (mask == 1)))
                or not bool(torch.isfinite(row["rollout_log_probs"]).all())
                or not torch.equal(row["input_ids"], torch.cat((prompt, response)))):
            raise PublicationValidationError("Original token/mask/behavior-logprob alignment failed")
    return members


async def _invoke(function: Callable, *args, timeout: float, **kwargs):
    async def call():
        result = function(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    return await asyncio.wait_for(call(), timeout=timeout)


async def _validate_original(validator, fields, members, timeout):
    result = await _invoke(validator, fields, members, timeout=timeout)
    if result is not None and result is not True:
        raise PublicationValidationError("Original training semantics validator did not accept the batch")


async def _blocked(coordinator, handoff_id, result, reason, error, timeout):
    result.update(state="BLOCKED", reason=reason, error_type=type(error).__name__)
    try:
        unresolved = await _invoke(coordinator.reconcile_publication, handoff_id, None, timeout=timeout)
        result["publication_state"] = unresolved["state"]
    except Exception as metadata_error:
        result["reconciliation_error_type"] = type(metadata_error).__name__
    return result


def _members_from_binding(binding, uid):
    if (not isinstance(binding, dict) or set(binding) != {"partition_id", "keys", "content_digest"}
            or not isinstance(binding["partition_id"], str) or not binding["partition_id"]
            or not isinstance(binding["keys"], list) or len(binding["keys"]) != 8
            or not isinstance(binding["content_digest"], str) or len(binding["content_digest"]) != 64
            or any(char not in "0123456789abcdef" for char in binding["content_digest"])):
        raise PublicationValidationError("Invalid saved publication binding")
    members = []
    prefix = uid + "_"
    for key in binding["keys"]:
        if not isinstance(key, str) or not key.startswith(prefix):
            raise PublicationValidationError("Saved queue key does not belong to the original prompt")
        components = key[len(prefix):].split("_")
        if len(components) != 2 or any(not item.isdecimal() for item in components):
            raise PublicationValidationError("Saved queue key has invalid session/output indices")
        member = PublicationMember(uid, int(components[0]), int(components[1]))
        if member.key != key:
            raise PublicationValidationError("Saved queue key is not in the original V1 format")
        members.append(member)
    if sorted(member.session_id for member in members) != list(range(8)):
        raise PublicationValidationError("Saved publication is missing or duplicating a session")
    return tuple(members)


def _has_ready_proof(intent, binding, uid):
    """A durable data+ready receipt, never a coordinator status by itself."""
    receipt = intent.get("receipt")
    if (not isinstance(receipt, dict) or receipt.get("publication_id") != intent["publication_id"]
            or receipt.get("binding") != binding):
        return False
    backend = receipt.get("backend_reference")
    if (not isinstance(backend, dict) or backend.get("api") != "transfer_queue.async_kv_batch_get"
            or backend.get("partition_id") != binding["partition_id"] or backend.get("keys") != binding["keys"]
            or backend.get("verified_members") != 8 or backend.get("actual_content_digest") != binding["content_digest"]
            or backend.get("tags_verified_by") != "transfer_queue.async_kv_list"):
        return False
    ready = backend.get("prompt_ready")
    return bool(
        isinstance(ready, dict) and ready.get("key") == uid
        and ready.get("partition_id") == binding["partition_id"]
        and ready.get("verified_by") == "transfer_queue.async_kv_list"
        and ready.get("actual_status") in {"finished", "consumed"} and ready.get("verified") is True
    )


async def _prompt_status(tq_api, partition_id, uid, timeout):
    partitions = await _invoke(tq_api.async_kv_list, partition_id=partition_id, timeout=timeout)
    tags = partitions.get(partition_id, {}).get(uid)
    if not isinstance(tags, dict) or tags.get("status") not in {"running", "finished", "consumed"}:
        raise PublicationValidationError("Original prompt key is absent or has an ambiguous status")
    return tags["status"]


async def _confirm(coordinator, handoff_id, receipt, timeout):
    confirmed = await _invoke(coordinator.confirm_publication, handoff_id, receipt, timeout=timeout)
    if confirmed["state"] != "PUBLISHED" or confirmed["binding"] != receipt["binding"]:
        raise PublicationValidationError("Coordinator did not persist the exact verified receipt")
    return confirmed


async def _finish_ready(coordinator, handoff_id, receipt, result, timeout, collect_garbage):
    try:
        finished = await _invoke(coordinator.finish_handoff_manifest, handoff_id, receipt["publication_id"], timeout=timeout)
        if finished != HandoffState.FINISHED.value:
            raise PublicationValidationError("Coordinator did not durably finish the ready handoff")
    except Exception as error:
        result.update(reason="MANIFEST_FINISH_FAILED", error_type=type(error).__name__)
        return result
    result.update(state="FINISHED", publication_state="PUBLISHED", reason=None, receipt=receipt)
    if collect_garbage:
        try:
            result["gc_count"] = await _invoke(coordinator.collect_garbage, timeout=timeout)
        except Exception as error:
            result["gc_error_type"] = type(error).__name__
    return result


async def _publish_or_reconcile(
    *, coordinator, handoff_id, members, fields, tags, partition_id, validate_fields,
    tq_api, rpc_timeout_seconds, collect_garbage, allow_new,
):
    result = {
        "state": "BLOCKED", "publication_id": None, "binding": None,
        "write_attempted": False, "readback_verified": False, "gc_count": None,
        "readback_performed": False, "readback_proof_reused": False,
        "prompt_ready_write_attempted": False, "prompt_ready_verified": False,
        "queue_replay_persistence_proved": False, "optimizer_ack_atomicity_proved": False,
    }
    if isinstance(rpc_timeout_seconds, bool) or not isinstance(rpc_timeout_seconds, (int, float)) or not math.isfinite(rpc_timeout_seconds) or rpc_timeout_seconds <= 0:
        raise ValueError("Publication RPC timeout must be finite and positive")
    timeout = rpc_timeout_seconds
    try:
        if not isinstance(handoff_id, str) or not handoff_id:
            raise PublicationValidationError("Expected the original handoff identity")
        manifest = await _invoke(coordinator.get_handoff_manifest, handoff_id, timeout=timeout)
        tables = manifest.member_tables()
        if len(tables) != 1 or next(iter(tables.values()))["group_size"] != 8:
            raise PublicationValidationError("Expected one original n=8 handoff group")
        uid = next(iter(tables))
        result["handoff_membership_verified"] = True
        result["prompt_key"] = uid
        intent = await _invoke(coordinator.get_publication, handoff_id, timeout=timeout)
        summary = await _invoke(coordinator.summary, timeout=timeout)
        manifest_finished = summary["handoffs"][handoff_id] == HandoffState.FINISHED.value
        if intent is not None:
            binding = intent["binding"]
            saved_members = _members_from_binding(binding, uid)
            if partition_id is not None and partition_id != binding["partition_id"]:
                raise PublicationValidationError("Saved publication partition cannot be changed")
            if members is not None and tuple(member.key for member in members) != tuple(binding["keys"]):
                raise PublicationValidationError("Saved member keys cannot be changed")
            result.update(binding=binding, publication_id=intent["publication_id"], publication_state=intent["state"])
            saved_ready = _has_ready_proof(intent, binding, uid)
        else:
            saved_members, saved_ready = None, False
        if manifest_finished:
            if not saved_ready:
                raise PublicationValidationError("Finished handoff lacks a durable prompt readiness proof")
            # The stock consumer may have cleared TQ data. Do not replay after delivery.
            result.update(state="FINISHED", receipt=deepcopy(intent["receipt"]), reason=None,
                          readback_verified=True, readback_proof_reused=True,
                          prompt_ready_verified=True, duplicate_finished=True,
                          proof_layer=intent["receipt"]["backend_reference"].get("proof_layer"))
            return result
        if tq_api is None:
            import transfer_queue as tq_api
        runtime_is_stock = getattr(tq_api, "__name__", None) == "transfer_queue"
        if runtime_is_stock and version("transferqueue") != TQ_VERSION:
            raise PublicationValidationError("TransferQueue runtime differs from the pinned version")
        result["proof_layer"] = "actual_transferqueue_runtime" if runtime_is_stock else "injected_api_only"
        active_partition = binding["partition_id"] if intent is not None else partition_id
        if not isinstance(active_partition, str) or not active_partition:
            raise PublicationValidationError("Expected an actual publication partition")
        prompt_status = await _prompt_status(tq_api, active_partition, uid, timeout)
        result["prompt_status_observed"] = prompt_status
    except Exception as error:
        result.update(reason="PUBLICATION_INTENT_FAILED", error_type=type(error).__name__)
        return result

    if saved_ready:
        if prompt_status not in {"finished", "consumed"}:
            return await _blocked(coordinator, handoff_id, result, "READY_PROOF_STATUS_REGRESSED",
                                  PublicationValidationError("Previously ready prompt is no longer ready"), timeout)
        # Revalidate actual prompt metadata; a persisted data receipt is not a new GET.
        receipt = deepcopy(intent["receipt"])
        receipt["backend_reference"]["prompt_ready"]["actual_status"] = prompt_status
        result.update(readback_verified=True, readback_proof_reused=True, prompt_ready_verified=True)
        try:
            await _confirm(coordinator, handoff_id, receipt, timeout)
        except Exception as error:
            result.update(reason="READY_PROOF_RECONFIRM_FAILED", error_type=type(error).__name__)
            return result
        return await _finish_ready(coordinator, handoff_id, receipt, result, timeout, collect_garbage)

    frozen_fields = frozen_tags = None
    try:
        if not callable(validate_fields):
            raise PublicationValidationError("An original training fields validator is required")
        if allow_new:
            if not isinstance(tags, Sequence) or len(tags) != 8 or any(not isinstance(tag, dict) for tag in tags):
                raise PublicationValidationError("Expected the original postprocessed tags")
            frozen_fields, frozen_tags = deepcopy(fields), deepcopy(list(tags))
            members = _validate_members(members, _logical_rows(frozen_fields))
            if members[0].uid != uid:
                raise PublicationValidationError("Original queue uid does not match the handoff group")
            await _validate_original(validate_fields, frozen_fields, members, timeout)
            proposed = {"partition_id": active_partition, "keys": [member.key for member in members],
                        "content_digest": training_payload_digest(keys=[member.key for member in members],
                            partition_id=active_partition, fields=frozen_fields, tags=frozen_tags)}
            if intent is not None and proposed != binding:
                raise PublicationValidationError("Existing publication binding cannot be changed")
            binding = proposed
        elif intent is None or intent["state"] == "PREPARED":
            raise PublicationValidationError("Read-only recovery cannot dispatch a new member publication")
        else:
            members = saved_members
        if intent is None:
            if prompt_status != "running":
                raise PublicationValidationError("First publication requires the existing running prompt key")
            intent = await _invoke(coordinator.prepare_publication, handoff_id, binding, timeout=timeout)
        result.update(binding=binding, publication_id=intent["publication_id"], publication_state=intent["state"])
        keys, partition_id = binding["keys"], binding["partition_id"]
    except Exception as error:
        result.update(reason="PREPARATION_VALIDATION_FAILED", error_type=type(error).__name__)
        return result

    if intent["state"] == "PREPARED":
        try:
            if prompt_status != "running":
                raise PublicationValidationError("Already ready prompt cannot receive new member writes")
            intent = await _invoke(coordinator.mark_publication_started, handoff_id, timeout=timeout)
            if intent["state"] != "DISPATCHED" or intent["binding"] != binding:
                raise PublicationValidationError("Coordinator did not persist the exact dispatched binding")
            result["publication_state"] = intent["state"]
        except Exception as error:
            return await _blocked(coordinator, handoff_id, result, "DISPATCH_MARK_FAILED", error, timeout)
        result["write_attempted"] = True
        try:
            # Never use this return value as a publication receipt.
            await _invoke(tq_api.async_kv_batch_put, keys=keys, partition_id=partition_id,
                          fields=frozen_fields, tags=frozen_tags, timeout=timeout)
        except Exception as error:
            # A write may have completed even when its response/metadata failed.
            result["write_error_type"] = type(error).__name__
    elif intent["state"] not in {"DISPATCHED", "BLOCKED", "NEEDS_VERIFICATION", "PUBLISHED"}:
        return await _blocked(coordinator, handoff_id, result, "UNSUPPORTED_PUBLICATION_STATE",
                              PublicationValidationError("Unsupported publication state"), timeout)

    try:
        actual = await _invoke(tq_api.async_kv_batch_get, keys=keys, partition_id=partition_id,
                               select_fields=None, timeout=timeout)
        result["readback_performed"] = True
        _validate_members(members, _logical_rows(actual))
        await _validate_original(validate_fields, actual, members, timeout)
        partitions = await _invoke(tq_api.async_kv_list, partition_id=partition_id, timeout=timeout)
        actual_tags = [partitions[partition_id][key] for key in keys]
        actual_digest = training_payload_digest(keys=keys, partition_id=partition_id, fields=actual, tags=actual_tags)
        if actual_digest != binding["content_digest"]:
            raise PublicationValidationError("Actual complete fields/tags digest differs from the frozen batch")
        result["readback_verified"] = True
        backend_reference = {
            "api": "transfer_queue.async_kv_batch_get", "partition_id": partition_id,
            "keys": keys, "field_names": sorted(actual.keys()), "verified_members": 8,
            "actual_content_digest": actual_digest, "tags_verified_by": "transfer_queue.async_kv_list",
            "proof_layer": result["proof_layer"],
            "tq_version": TQ_VERSION if runtime_is_stock else None,
        }
    except Exception as error:
        return await _blocked(coordinator, handoff_id, result, "ACTUAL_READBACK_FAILED", error, timeout)

    receipt = {"publication_id": intent["publication_id"], "binding": binding,
               "backend_reference": backend_reference}
    try:
        confirmed = await _confirm(coordinator, handoff_id, receipt, timeout)
    except Exception as error:
        result.update(reason="PUBLICATION_CONFIRM_FAILED", error_type=type(error).__name__, publication_state=None)
        return result
    result["publication_state"] = confirmed["state"]
    try:
        prompt_status = await _prompt_status(tq_api, partition_id, uid, timeout)
        if prompt_status == "running":
            result["prompt_ready_write_attempted"] = True
            try:
                await _invoke(tq_api.async_kv_put, key=uid, partition_id=partition_id,
                              tag={"status": "finished"}, timeout=timeout)
            except Exception as error:
                result["prompt_ready_write_error_type"] = type(error).__name__
            # Only the actual ready status, not PUT completion, allows finish/GC.
            prompt_status = await _prompt_status(tq_api, partition_id, uid, timeout)
        if prompt_status not in {"finished", "consumed"}:
            raise PublicationValidationError("Original prompt readiness has not been observed")
        receipt["backend_reference"]["prompt_ready"] = {
            "key": uid, "partition_id": partition_id, "actual_status": prompt_status,
            "verified_by": "transfer_queue.async_kv_list", "verified": True,
        }
        result.update(prompt_ready_verified=True, prompt_status_observed=prompt_status)
    except Exception as error:
        return await _blocked(coordinator, handoff_id, result, "PROMPT_READY_NOT_VERIFIED", error, timeout)
    try:
        await _confirm(coordinator, handoff_id, receipt, timeout)
    except Exception as error:
        result.update(reason="PROMPT_READY_CONFIRM_FAILED", error_type=type(error).__name__, publication_state=None)
        return result
    return await _finish_ready(coordinator, handoff_id, receipt, result, timeout, collect_garbage)


async def publish_trajectory_group(
    *, coordinator, handoff_id: str, members: Sequence[PublicationMember], fields: TensorDictBase,
    tags: Sequence[dict], partition_id: str, validate_fields: Callable,
    tq_api=None, rpc_timeout_seconds: float = 30.0, collect_garbage: bool = True,
) -> dict[str, Any]:
    """Write members once, durably prove data and stock uid readiness, then GC.

    The original uid must already exist as running before a first write. An
    already-finished or consumed uid is never rewritten. On recovery, prefer
    reconcile_trajectory_group_publication so postprocessing is not repeated.
    """
    return await _publish_or_reconcile(
        coordinator=coordinator, handoff_id=handoff_id, members=members, fields=fields, tags=tags,
        partition_id=partition_id, validate_fields=validate_fields, tq_api=tq_api,
        rpc_timeout_seconds=rpc_timeout_seconds, collect_garbage=collect_garbage, allow_new=True,
    )


async def reconcile_trajectory_group_publication(
    *, coordinator, handoff_id: str, validate_fields: Callable | None = None,
    tq_api=None, rpc_timeout_seconds: float = 30.0, collect_garbage: bool = True,
) -> dict[str, Any]:
    """Recover an existing publication without generating or rewriting members.

    With no durable ready proof, a real full member GET/digest and the original
    validator are required. If those data have been consumed/deleted, BLOCKED
    preserves the uncertainty; this function does not repeat an optimizer step.
    An already-FINISHED publication with its durable ready receipt returns that
    historical result without TQ writes or fresh data-persistence claims.
    """
    return await _publish_or_reconcile(
        coordinator=coordinator, handoff_id=handoff_id, members=None, fields=None, tags=None,
        partition_id=None, validate_fields=validate_fields, tq_api=tq_api,
        rpc_timeout_seconds=rpc_timeout_seconds, collect_garbage=collect_garbage, allow_new=False,
    )
