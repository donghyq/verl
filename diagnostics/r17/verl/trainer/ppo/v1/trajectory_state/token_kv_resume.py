# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""An opt-in bridge between durable V1 trajectories and portable SGLang decode.

Only committed request state is imported. The ordinary tool handler receives
one complete assistant output, including its saved prefix, and performs the
existing Continuous Token merge. This module does not publish sample groups or
claim that a source engine has exited; process-death proof belongs to the
runtime. It also never turns an uncertain HTTP outcome into another generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from .snapshot import canonical_bytes, token_digest

KIND = "token_kv_committed_v1"
FRAME = "trajectory_inflight_generation"
STATE_KINDS = {"attention_kv", "mamba_conv", "mamba_temporal"}


def _sha(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _tokens(value, *, nonempty=True):
    return isinstance(value, list) and (bool(value) or not nonempty) and all(
        type(item) is int and item >= 0 for item in value
    )


def _actual_attempts(output, expected_version):
    """Validate native receipts; never infer a receipt from policy identity."""
    ids, probabilities = output.token_ids, output.log_probs
    if not _tokens(ids, nonempty=False) or not isinstance(probabilities, list) or len(ids) != len(probabilities):
        raise ValueError("actual output-token logprobs are required")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in probabilities):
        raise ValueError("actual output logprobs must be finite")
    if getattr(output, "routed_experts", None) is not None:
        raise ValueError("routed-expert replay is unsupported by token-KV recovery")
    extra = output.extra_fields
    attempts = extra.get("native_gateway_attempts")
    if attempts is None and isinstance(extra.get("native_meta_info"), dict):
        attempts = [{**extra, "output_start": 0, "output_end": len(ids)}]
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("actual native generation receipts are missing")
    cursor = 0
    for attempt in attempts:
        start, end = attempt.get("output_start"), attempt.get("output_end")
        if type(start) is not int or type(end) is not int or start != cursor or not start <= end <= len(ids):
            raise ValueError("native attempts do not cover actual generated tokens")
        meta = attempt.get("native_meta_info")
        if not isinstance(meta, dict) or str(meta.get("weight_version")) != expected_version:
            raise ValueError("actual native policy version differs")
        if not isinstance(meta.get("id"), str) or not meta["id"]:
            raise ValueError("actual native request ID is missing")
        spans = meta.get("weight_versions")
        if spans is not None:
            if not isinstance(spans, list) or not spans:
                raise ValueError("actual native version spans are missing")
            local = 0
            for span in spans:
                left, right = span.get("start"), span.get("end")
                if (type(left) is not int or type(right) is not int or left != local
                        or not left <= right <= end - start
                        or (right == left and end > start)
                        or str(span.get("version")) != expected_version):
                    raise ValueError("actual native version spans do not cover one attempt")
                local = right
            if local != end - start:
                raise ValueError("actual native version coverage is incomplete")
        cursor = end
    if cursor != len(ids):
        raise ValueError("native attempts do not cover the complete output")
    return deepcopy(attempts)


def validate_token_kv_reference(ref, *, snapshot):
    """Validate durable descriptor shape without requiring a GPU/file mount.

    The engine separately verifies every payload before loading. The bridge
    checks the pinned manifest and every TP metadata file before admission.
    """
    frame = snapshot.agent_state["extra_fields"].get(FRAME)
    if not isinstance(ref, dict) or not isinstance(frame, dict):
        raise ValueError("token-KV descriptor or unfinished generation state is missing")
    sid, storage, manifest = ref.get("snapshot_id"), ref.get("storage_dir"), ref.get("manifest_ref")
    if not isinstance(sid, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", sid) is None or sid in {".", ".."}:
        raise ValueError("invalid token-KV snapshot ID")
    if not isinstance(storage, str) or not Path(storage).is_absolute() or not isinstance(manifest, dict):
        raise ValueError("token-KV descriptor requires an absolute storage directory")
    expected_path = str(Path(storage) / sid / "COMMITTED.json")
    if (ref.get("kind") != KIND or ref.get("snapshot_committed") is not True
            or manifest.get("path") != expected_path or not _digest(manifest.get("sha256"))
            or type(manifest.get("bytes")) is not int or manifest["bytes"] <= 0):
        raise ValueError("token-KV descriptor lacks an immutable commit receipt")
    identity, binding = ref.get("identity"), ref.get("source_binding")
    if (not isinstance(identity, dict) or type(identity.get("tp_size")) is not int or identity["tp_size"] < 1
            or not isinstance(binding, dict) or ref.get("policy_identity") != snapshot.policy_identity
            or ref.get("token_digest") != token_digest(snapshot.agent_state["prompt_ids"])
            or str(identity.get("weight_version")) != snapshot.policy_identity["policy_version"]
            or str(identity.get("actualweight_version")) != snapshot.policy_identity["policy_version"]):
        raise ValueError("token-KV model/version/prefix identity differs")
    for name in ("model_config_sha256", "cache_layout_sha256", "engine_source_sha256"):
        if not _digest(identity.get(name)):
            raise ValueError(f"actual engine {name} is missing")
    if (binding.get("trajectory_id") != snapshot.trajectory_id
            or binding.get("source_execution_generation") != snapshot.execution_generation
            or type(binding.get("source_execution_generation")) is not int
            or not all(isinstance(binding.get(name), str) and binding[name]
                       for name in ("source_owner_id", "source_lease_id", "source_engine_instance_id", "request_id"))):
        raise ValueError("token-KV source ownership binding differs")
    inputs, outputs, probabilities = frame.get("input_ids"), frame.get("output_ids"), frame.get("output_logprobs")
    if (inputs != snapshot.agent_state["prompt_ids"] or not _tokens(inputs) or not _tokens(outputs)
            or not isinstance(probabilities, list) or len(probabilities) != len(outputs)
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in probabilities)
            or frame.get("logical_request_id") != snapshot.agent_state["request_id"]
            or frame.get("source_engine_request_id") != binding["request_id"]
            or ref.get("generated_prefix_digest") != token_digest(outputs)
            or not _digest(frame.get("request_metadata_sha256"))
            or frame["request_metadata_sha256"] != ref.get("request_metadata_sha256")):
        raise ValueError("unfinished generation has changed its actual prefix or logprobs")
    budget, remaining = frame.get("original_max_new_tokens"), frame.get("remaining_generation_tokens")
    if (type(budget) is not int or type(remaining) is not int or remaining <= 0
            or budget != remaining + len(outputs)
            or not isinstance(frame.get("requested_sampling_params"), dict)
            or frame.get("assistant_output_merged") is not False):
        raise ValueError("unfinished generation budget or merge boundary is invalid")
    from types import SimpleNamespace

    _actual_attempts(SimpleNamespace(token_ids=outputs, log_probs=probabilities,
                                    extra_fields={"native_gateway_attempts": frame.get("source_native_attempts")}),
                     snapshot.policy_identity["policy_version"])


def read_committed_request(ref):
    """Read actual committed JSON, checking all ranks and the shared-state hash."""
    path = Path(ref["manifest_ref"]["path"])
    if path.is_symlink():
        raise ValueError("committed snapshot manifest may not be a symlink")
    raw = path.read_bytes()
    receipt = ref["manifest_ref"]
    if len(raw) != receipt["bytes"] or hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ValueError("committed snapshot manifest changed")
    manifest = json.loads(raw)
    size = ref["identity"]["tp_size"]
    if (manifest.get("schema_version") != 1 or manifest.get("snapshot_id") != ref["snapshot_id"]
            or manifest.get("tp_size") != size
            or [row.get("rank") for row in manifest.get("members", [])] != list(range(size))):
        raise ValueError("committed TP snapshot is incomplete")
    metadata = None
    for rank in range(size):
        rank_path = path.parent / f"rank-{rank}.json"
        if rank_path.is_symlink():
            raise ValueError("rank metadata may not be a symlink")
        value = json.loads(rank_path.read_bytes())
        if (_sha(value) != manifest["members"][rank]["metadata_sha256"]
                or value.get("schema_version") != 1 or value.get("tp_rank") != rank
                or value.get("tp_size") != size or value.get("snapshot_id") != ref["snapshot_id"]
                or value.get("identity") != ref["identity"]
                or value.get("payload_name") != f"rank-{rank}.bin"
                or type(value.get("payload_bytes")) is not int or value["payload_bytes"] <= 0
                or not _digest(value.get("payload_sha256"))):
            raise ValueError("committed rank metadata changed")
        request = value.get("request_metadata")
        if (_sha([value["identity"], request]) != manifest["shared_state_sha256"]
                or (metadata is not None and request != metadata)):
            raise ValueError("committed TP request histories disagree")
        metadata = request
    if not isinstance(metadata, dict) or _sha(metadata) != ref["request_metadata_sha256"]:
        raise ValueError("committed request metadata changed")
    if (metadata.get("request_schema_version") != 2
            or metadata.get("source_binding") != ref["source_binding"]
            or token_digest(metadata.get("origin_input_ids")) != ref["token_digest"]
            or token_digest(metadata.get("output_ids")) != ref["generated_prefix_digest"]):
        raise ValueError("committed request differs from the trajectory descriptor")
    # r3 persists spans produced by the engine's native version event helper.
    # They are never reconstructed from the caller's desired identity.
    spans, cursor = metadata.get("output_weight_versions"), 0
    if not isinstance(spans, list) or not spans:
        raise ValueError("committed native output version spans are missing")
    for span in spans:
        start, end = span.get("start"), span.get("end")
        if (type(start) is not int or type(end) is not int or start != cursor
                or not start < end <= len(metadata["output_ids"])
                or str(span.get("version")) != ref["policy_identity"]["policy_version"]):
            raise ValueError("committed native output version spans are inconsistent")
        cursor = end
    if cursor != len(metadata["output_ids"]):
        raise ValueError("committed native output version coverage is incomplete")
    return metadata


def build_inflight_snapshot(*, before_generate_snapshot, export_receipt, source_lease,
                            partial_output, generation_call_sampling_params):
    """Bind an actual export to the AgentData captured before awaiting generate.

    The caller must export with source_binding already included. This builder
    cannot manufacture a source binding or an observed version receipt.
    """
    base = before_generate_snapshot
    if not _tokens(partial_output.token_ids):
        raise ValueError("a real nonempty generation prefix is required")
    if (source_lease.trajectory_id != base.trajectory_id or source_lease.generation != base.execution_generation
            or export_receipt.get("status") != "PASS" or export_receipt.get("snapshot_committed") is not True):
        raise ValueError("actual committed export and current source lease are required")
    identity = export_receipt["identity"]
    ranks = export_receipt.get("ranks")
    if (not isinstance(ranks, list) or {row.get("tp_rank") for row in ranks} != set(range(identity["tp_size"]))
            or any(set(row.get("state_kinds", [])) != STATE_KINDS for row in ranks)):
        raise ValueError("actual attention KV and all GDN TP state receipts are required")
    manifest_path = Path(export_receipt["manifest_ref"]["path"])
    # Hash-pinned metadata is the authority; do not read speculative ACK fields.
    rank0 = json.loads((manifest_path.parent / "rank-0.json").read_bytes())
    metadata = rank0["request_metadata"]
    binding = metadata.get("source_binding")
    if (not isinstance(binding, dict) or binding.get("source_owner_id") != source_lease.owner_id
            or binding.get("source_lease_id") != source_lease.lease_id):
        raise ValueError("engine export was not bound to the actual source lease")
    if (metadata.get("request_schema_version") != 2
            or metadata.get("origin_input_ids") != base.agent_state["prompt_ids"]
            or metadata.get("output_ids") != partial_output.token_ids
            or metadata.get("output_logprobs") != partial_output.log_probs
            or metadata.get("output_logprob_token_ids") != partial_output.token_ids
            or metadata.get("computed_tokens") != len(base.agent_state["prompt_ids"]) + len(partial_output.token_ids) - 1
            or metadata.get("pending_input_token") != partial_output.token_ids[-1]
            or metadata.get("request_flags", {}).get("return_logprob") is not True):
        raise ValueError("actual engine state and observed partial output differ")
    attempts = _actual_attempts(partial_output, base.policy_identity["policy_version"])
    if attempts[-1]["native_meta_info"]["id"] != metadata["rid"] or binding.get("request_id") != metadata["rid"]:
        raise ValueError("native SSE and exported request identities differ")
    frame = {
        "logical_request_id": base.agent_state["request_id"],
        "source_engine_request_id": metadata["rid"],
        "input_ids": deepcopy(metadata["origin_input_ids"]),
        "output_ids": deepcopy(metadata["output_ids"]),
        "output_logprobs": deepcopy(metadata["output_logprobs"]),
        "source_native_attempts": attempts,
        "requested_sampling_params": deepcopy(generation_call_sampling_params),
        "original_max_new_tokens": metadata["resolved_sampling_params"]["max_new_tokens"],
        "remaining_generation_tokens": metadata["remaining_generation_tokens"],
        "request_metadata_sha256": _sha(metadata),
        "assistant_output_merged": False,
    }
    ref = {
        "kind": KIND, "snapshot_id": export_receipt["snapshot_id"],
        "storage_dir": str(manifest_path.parent.parent),
        "snapshot_committed": True, "manifest_ref": deepcopy(export_receipt["manifest_ref"]),
        "identity": deepcopy(identity), "source_binding": deepcopy(binding),
        "policy_identity": deepcopy(base.policy_identity),
        "token_digest": token_digest(base.agent_state["prompt_ids"]),
        "generated_prefix_digest": token_digest(partial_output.token_ids),
        "request_metadata_sha256": _sha(metadata),
    }
    state = deepcopy(base.agent_state)
    state["extra_fields"][FRAME] = frame
    result = replace(base, agent_state=state, kv_ref=ref, schema_version=2, next_stage="generating_inflight")
    result.validate()
    read_committed_request(ref)
    return result


def capture_before_generate_snapshot(loop, agent_data, *, sampling_params, run_kwargs):
    """Capture actual AgentData before the original generating handler awaits.

    This is a candidate CPU copy, not a committed checkpoint. Only
    build_inflight_snapshot can pair it with an actual export and validate it.
    The original stream/preemption source hook must call this at that boundary.
    """
    from .snapshot import TrajectorySnapshot

    loop._check_text_state(agent_data)
    fields = ("messages", "prompt_ids", "response_ids", "response_mask", "response_logprobs",
              "turn_scores", "tool_rewards", "user_turns", "assistant_turns", "extra_fields",
              "tools_kwargs", "metrics", "request_id", "mm_processor_kwargs")
    return TrajectorySnapshot(
        trajectory_id=run_kwargs["trajectory_id"], group_id=run_kwargs["rollout_group_id"],
        member_index=run_kwargs["rollout_sample_index"], group_size=run_kwargs["rollout_group_size"],
        execution_generation=run_kwargs["execution_generation"],
        agent_state={name: deepcopy(getattr(agent_data, name)) for name in fields},
        sampling_params=deepcopy(sampling_params), limits=loop._trajectory_limits(),
        environment_version=run_kwargs["snapshot_environment_version"],
        tool_version=run_kwargs["snapshot_tool_version"], policy_identity=deepcopy(run_kwargs["snapshot_policy_identity"]),
        tool_selection=list(agent_data._active_tools), completed_tool_batches=deepcopy(agent_data.completed_tool_batches),
    )


async def commit_and_yield_token_checkpoint(snapshot, *, commit_snapshot, check_owner):
    """Use the existing coordinator commit/fencing callback before yielding.

    A None result means no current preemption; the source must continue. KV
    slots/processes may be released only by the runtime after a real reference.
    """
    from .preemptible_tool_agent_loop import TrajectoryCheckpointYield

    snapshot.validate()
    if snapshot.schema_version != 2 or not callable(commit_snapshot) or not callable(check_owner):
        raise ValueError("token checkpoint requires an existing owned commit callback")
    await check_owner()
    reference = await commit_snapshot(snapshot)
    if reference is None:
        return None
    if not isinstance(reference, str) or not reference:
        raise ValueError("coordinator did not return a durable trajectory reference")
    await check_owner()
    raise TrajectoryCheckpointYield(snapshot, reference)


def validate_portable_restore_receipt(observed, *, ref, target_rid):
    """Require actual per-rank load/readback proof, not merely valid output."""
    if observed.get("status") != "PASS" or observed.get("identity") != ref["identity"]:
        raise ValueError("destination engine identity changed during portable restore")
    ranks = observed.get("ranks")
    size = ref["identity"]["tp_size"]
    if not isinstance(ranks, list) or sorted(row.get("tp_rank") for row in ranks) != list(range(size)):
        raise ValueError("portable restore receipts do not cover all TP ranks")
    result = []
    directory = Path(ref["manifest_ref"]["path"]).parent
    request = read_committed_request(ref)
    for rank in ranks:
        proof = rank.get("portable_restore_by_rid", {}).get(target_rid)
        if (not isinstance(proof, dict) or proof.get("status") != "PASS"
                or proof.get("target_rid") != target_rid or proof.get("source_rid") != request["rid"]
                or proof.get("source_binding") != ref["source_binding"]
                or proof.get("identity") != ref["identity"] or proof.get("manifest_ref") != ref["manifest_ref"]
                or any(proof.get(name) is not True for name in (
                    "digestmatch", "loaded_from_snapshot", "allocated_new_slots", "request_reconstructed",
                    "stream_returns_full_output"))
                or proof.get("from_retained_cpu_record") is not False or proof.get("target_stream_offset") != 0
                or set(proof.get("state_kinds", [])) != STATE_KINDS
                or proof.get("computed_tokens") != request["computed_tokens"]
                or proof.get("pending_input_token") != request["pending_input_token"]
                or not isinstance(proof.get("engine_instance_id"), str) or not proof["engine_instance_id"]
                or not isinstance(proof.get("destination_slots"), dict)):
            raise ValueError("portable restore lacks actual state/request reconstruction proof")
        exported = json.loads((directory / f"rank-{rank['tp_rank']}.json").read_bytes())["tensor_metadata"]
        if any(proof.get(name) != exported.get(name) for name in (
            "state_sha256", "state_inventory_sha256", "state_bytes"
        )):
            raise ValueError("portable GPU readback differs from committed source tensors")
        result.append(deepcopy(proof))
    return result


def resumed_generation_segments(segments, *, prefix_tokens, source_generation, target_generation):
    """Retain which execution generated saved tokens even for a full new SSE."""
    result = []
    for segment in segments:
        start, end = segment["start"], segment["end"]
        cut = min(end, max(start, prefix_tokens))
        if start < cut:
            result.append({**segment, "start": start, "end": cut, "execution_generation": source_generation})
        if cut < end:
            result.append({**segment, "start": cut, "end": end, "execution_generation": target_generation})
    return result


class PortableSGLangHTTPBackend:
    """Real SGLang HTTP transport; actual native Gateway parser remains authority."""

    def __init__(self, engine_url, *, timeout_s=600, api_key=None):
        if not isinstance(engine_url, str) or not engine_url.startswith(("http://", "https://")):
            raise ValueError("a real destination SGLang engine URL is required")
        self.engine_url, self.timeout_s, self.api_key = engine_url.rstrip("/"), timeout_s, api_key

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def inspect(self):
        import aiohttp

        async with aiohttp.ClientSession(headers=self._headers(), timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as session:
            async with session.post(self.engine_url + "/token_kv_snapshot", json={"operation": "inspect"}) as response:
                response.raise_for_status()
                value = await response.json()
                if value.get("status") != "PASS" or not isinstance(value.get("identity"), dict):
                    raise ValueError("destination snapshot inspect failed")
                return value

    async def generate_from_snapshot(self, *, ref, request_id, limit, observed_identity):
        import aiohttp
        from verl.workers.rollout.sglang_gateway_client import GatewayHTTPResponse, SGLangGatewayClient

        body = {"rid": request_id, "stream": True, "token_kv_restore": {
            name: deepcopy(ref[name]) for name in (
                "snapshot_id", "storage_dir", "identity", "source_binding", "manifest_ref"
            )
        }}
        body["token_kv_restore"]["expected_weight_version"] = ref["identity"]["weight_version"]
        last = None
        async with aiohttp.ClientSession(headers=self._headers(), timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as session:
            async with session.post(self.engine_url + "/generate", json=body) as response:
                response.raise_for_status()
                headers = dict(response.headers)
                async for line in response.content:
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == b"[DONE]":
                        continue
                    value = json.loads(payload)
                    if value.get("error"):
                        raise ValueError("destination engine returned a stream error")
                    last = value
        if not isinstance(last, dict) or not isinstance(last.get("meta_info"), dict):
            raise ValueError("destination engine returned no actual output")
        meta = last["meta_info"]
        rows = meta.get("output_token_logprobs")
        if not isinstance(rows, list):
            raise ValueError("destination engine returned no actual output-token logprobs")
        actual_ids = [row[1] for row in rows if isinstance(row, (list, tuple)) and len(row) >= 2]
        if len(actual_ids) != len(rows) or (last.get("output_ids") is not None and last["output_ids"] != actual_ids):
            raise ValueError("destination token IDs and actual logprobs differ")
        # This derives IDs only from actual engine output, preserving raw metadata.
        value = {**last, "output_ids": actual_ids}
        output = SGLangGatewayClient._decode_generation(
            GatewayHTTPResponse(200, value, headers), return_logprob=True, limit=limit,
            request_id=request_id, observed_weight_version=int(observed_identity["actualweight_version"]),
            observed_cache_salt="",
        )
        # A direct engine route provides no observed Gateway cache namespace.
        output.extra_fields["cache_salt"] = None
        output.extra_fields["salt_match"] = None
        return output


class _ResumeFirstGeneration:
    def __init__(self, manager, *, snapshot, lease, check_owner, backend):
        self.manager, self.snapshot, self.lease = manager, snapshot, lease
        self.check_owner, self.backend, self.used, self.started = check_owner, backend, False, False

    def __getattr__(self, name):
        return getattr(self.manager, name)

    async def generate(self, *args, **kwargs):
        if self.used:
            return await self.manager.generate(*args, **kwargs)
        if self.started:
            raise ValueError("uncertain portable recovery cannot be retried as a new native generation")
        self.started = True
        if args:
            raise ValueError("token-KV recovery requires explicit generation arguments")
        snapshot, ref = self.snapshot, self.snapshot.kv_ref
        frame = snapshot.agent_state["extra_fields"][FRAME]
        await self.check_owner()
        if (kwargs.get("request_id") != frame["logical_request_id"]
                or kwargs.get("prompt_ids") != frame["input_ids"]
                or kwargs.get("sampling_params") != frame["requested_sampling_params"]
                or any(kwargs.get(name) is not None for name in (
                    "image_data", "video_data", "audio_data", "mm_processor_output"))):
            raise ValueError("target generation does not match the saved pre-merge boundary")
        token_only = bool(getattr(self.check_owner, "token_only_resume", False))
        reason = "published_policy_changed" if token_only else None
        if not token_only:
            if self.backend is None:
                raise ValueError("token-KV destination backend has not been attached to this V1 worker")
            observed = await self.backend.inspect()
            if observed["identity"] != ref["identity"]:
                token_only, reason = True, "engine_identity_incompatible"
        await self.check_owner()
        if token_only:
            params = deepcopy(kwargs["sampling_params"])
            params.pop("max_tokens", None)
            params.pop("max_new_tokens", None)
            params["max_tokens"] = frame["remaining_generation_tokens"]
            # Original FullyAsync client still handles abort retry and its remaining budget.
            output = await self.manager.generate(**{
                **kwargs, "prompt_ids": frame["input_ids"] + frame["output_ids"], "sampling_params": params,
            })
            await self.check_owner()
            target_identity = getattr(self.check_owner, "target_policy_identity", snapshot.policy_identity)
            tail_attempts = _actual_attempts(output, target_identity["policy_version"])
            offset = len(frame["output_ids"])
            if len(output.token_ids) > frame["remaining_generation_tokens"]:
                raise ValueError("native recompute exceeded the remaining cumulative budget")
            output = deepcopy(output)
            output.token_ids = frame["output_ids"] + output.token_ids
            output.log_probs = frame["output_logprobs"] + output.log_probs
            output.extra_fields["native_gateway_attempts"] = frame["source_native_attempts"] + [
                {**attempt, "output_start": offset + attempt["output_start"],
                 "output_end": offset + attempt["output_end"]} for attempt in tail_attempts
            ]
            output.extra_fields["min_global_steps"] = int(snapshot.policy_identity["policy_version"])
            output.extra_fields["max_global_steps"] = int(target_identity["policy_version"])
            output.extra_fields["trajectory_token_kv_recovery"] = {"mode": "NATIVE_TOKEN_RECOMPUTE", "reason": reason}
        else:
            metadata = read_committed_request(ref)
            if (metadata["output_logprobs"] != frame["output_logprobs"]
                    or metadata["remaining_generation_tokens"] != frame["remaining_generation_tokens"]
                    or metadata["resolved_sampling_params"]["max_new_tokens"] != frame["original_max_new_tokens"]):
                raise ValueError("committed request and durable generation state differ")
            # A new engine request ID is essential: the source request is not reused.
            target_rid = uuid4().hex
            output = await self.backend.generate_from_snapshot(
                ref=ref, request_id=target_rid, limit=frame["original_max_new_tokens"],
                observed_identity=observed["identity"],
            )
            await self.check_owner()
            _actual_attempts(output, snapshot.policy_identity["policy_version"])
            raise ValueError("TARGET_NATIVE_OUTPUT_DIAGNOSTIC_ONLY_STOP_BEFORE_TRAINING; "
                             + portable_stop_reason_diagnostic(output, len(frame["output_ids"])))
            if output.stop_reason not in {"stop", "length", "completed"}:
                raise ValueError("portable generation did not complete; no uncertain recompute is allowed" + "; native_diagnostic=" + portable_stop_reason_diagnostic(output, len(frame["output_ids"])))
            count = len(frame["output_ids"])
            if (output.token_ids[:count] != frame["output_ids"] or output.log_probs[:count] != frame["output_logprobs"]
                    or len(output.token_ids) > frame["original_max_new_tokens"]):
                raise ValueError("portable restore changed the saved generated prefix, logprobs, or budget")
            observed = await self.backend.inspect()
            await self.check_owner()
            proofs = validate_portable_restore_receipt(observed, ref=ref, target_rid=target_rid)
            output.extra_fields["trajectory_token_kv_recovery"] = {
                "mode": "PORTABLE_STATE_RESTORED", "snapshot_id": ref["snapshot_id"], "ranks": proofs,
                "physical_restore_input_compute": _physical_input_compute_receipt(
                    observed, target_rid=target_rid),
                "source_process_exit_proven": False,
            }
        self.used = True
        return output


def install_token_kv_resume(loop, *, snapshot, lease, check_owner, backend=None):
    """Install on the actual V1 loop instance before resume_from_snapshot."""
    snapshot.validate()
    if (snapshot.schema_version != 2 or lease.trajectory_id != snapshot.trajectory_id
            or type(lease.generation) is not int or lease.generation <= snapshot.execution_generation
            or not callable(check_owner)):
        raise ValueError("portable recovery requires a newly fenced trajectory owner")
    if not snapshot.policy_identity["policy_version"].isascii() or not snapshot.policy_identity["policy_version"].isdecimal():
        raise ValueError("V1 published policy version must be an actual non-negative integer")
    loop.server_manager = _ResumeFirstGeneration(loop.server_manager, snapshot=snapshot, lease=lease,
                                               check_owner=check_owner, backend=backend)


def attach_token_kv_backend(worker, engine_url, *, timeout_s=600, api_key=None):
    """Explicit opt-in for a worker bound to one actual destination engine URL."""
    worker.token_kv_resume_backend = PortableSGLangHTTPBackend(engine_url, timeout_s=timeout_s, api_key=api_key)
    return worker.token_kv_resume_backend


def _physical_input_compute_receipt(observed, *, target_rid):
    """Read the existing post-restore inspect; TP replicas must agree, never sum."""
    result = {"status": "UNAVAILABLE", "target_rid": target_rid,
        "physical_restore_actual_recomputed_tokens": None,
        "scope": "physical_target_native_forward_after_import_only",
        "Ray_fallback_actual_recomputed_tokens": None, "per_rank": []}
    ranks = observed.get("ranks", [])
    if (observed.get("status") != "PASS" or not isinstance(ranks, list)
            or sorted(row.get("tp_rank") for row in ranks) != [0, 1]):
        return result
    values = []
    for row in ranks:
        metrics = row.get("resume_metrics_by_rid", {}).get(target_rid, {})
        value = metrics.get("actual_recomputed_tokens")
        result["per_rank"].append({"tp_rank": row["tp_rank"], "metrics": deepcopy(metrics)})
        if (metrics.get("actual_input_counter_status") != "OBSERVED_COMPLETED_NATIVE_FORWARD"
                or type(value) is not int or value < 0
                or not metrics.get("actual_completed_forward_iters")
                or type(metrics.get("committed_prefix_tokens")) is not int
                or metrics.get("actual_processed_input_tokens", 0) < 1):
            return result
        values.append((value, metrics["committed_prefix_tokens"]))
    if len(set(values)) != 1:
        return result
    result.update(status="OBSERVED_PHYSICAL_IMPORT_ONLY",
        physical_restore_actual_recomputed_tokens=values[0][0],
        committed_prefix_tokens=values[0][1],
        actual_weight_version=observed.get("identity", {}).get("actualweight_version"))
    return result


import hashlib
import json
import re

def _safe_text(value, limit=600):
    try:
        text = str(value)
    except Exception:
        text = value.__class__.__name__ if value is not None else "None"
    text = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)\b(?:https?|ftp)://\S+", "[REDACTED_URL]", text)
    text = re.sub(
        r"(?i)\b(bearer|token|access[_-]?token|api[_-]?key|secret|password)\s*[=:]\s*\S+",
        lambda match: match.group(1) + "=[REDACTED]", text,
    )
    text = re.sub(r"(?<![A-Za-z0-9_])/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+", "[REDACTED_PATH]", text)
    text = re.sub(r"(?i)\b(?:prompt|input_ids|output_ids|token_ids|log_probs|logprobs|output_token_logprobs|authorization|messages)\s*[:=].*", "[REDACTED_PAYLOAD]", text, flags=re.DOTALL)
    return text[:limit]

def _finish_details(meta):
    if not isinstance(meta, dict):
        return {"present": False}
    finish = meta.get("finish_reason")
    if not isinstance(finish, dict):
        return {"present": False}
    details = {"present": True}
    for key in ("type", "code", "status_code", "err_type"):
        if key in finish:
            details[key] = _safe_text(finish.get(key), 120)
    if "message" in finish:
        message = _safe_text(finish.get("message"), 600)
        details["message"] = message
        details["message_sha256"] = hashlib.sha256(message.encode("utf-8", "replace")).hexdigest()
        details["message_length"] = len(message)
    return details

def portable_stop_reason_diagnostic(output, preserved_token_count=0):
    """Return a bounded, read-only diagnostic for a rejected portable output."""
    try:
        stop_reason = _safe_text(getattr(output, "stop_reason", None), 120)
        token_ids = getattr(output, "token_ids", None)
        count = len(token_ids) if hasattr(token_ids, "__len__") else None
        extra = getattr(output, "extra_fields", None)
        meta = extra.get("native_meta_info") if isinstance(extra, dict) else None
        payload = {
            "output_stop_reason": stop_reason,
            "output_token_count": count,
            "preserved_token_count": preserved_token_count if type(preserved_token_count) is int and preserved_token_count >= 0 else None,
            "new_token_count": count - preserved_token_count if type(count) is int and type(preserved_token_count) is int and 0 <= preserved_token_count <= count else None,
            "native_meta_info": {
                "present": isinstance(meta, dict),
                "finish_reason": _finish_details(meta),
            },
        }
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except Exception:
        return json.dumps(
            {"output_stop_reason": None, "new_token_count": None,
             "native_meta_info": {"present": False, "finish_reason": {"present": False}}},
            sort_keys=True, separators=(",", ":"),
        )
