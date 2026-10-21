# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Configured interruption on the actual V1 generate call path.

This opt-in path allows normal source generations to run concurrently. Only the
pause/export/commit/release/continue control transaction is serialized. Actual
platform reclamation/engine restart remains the existing resource controller's
responsibility.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import copy
import json
import hashlib
import os
import time
import logging
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from .token_kv_resume import PortableSGLangHTTPBackend, attach_token_kv_backend, build_inflight_snapshot


def validated_source_settings(settings, owner_id):
    if settings.get("enabled") is not True:
        raise ValueError("token-KV source requires explicit enabled configuration")
    legacy_singleton = settings.get("functional_singleton", False)
    if type(legacy_singleton) is not bool:
        raise ValueError("legacy functional_singleton must be a boolean when provided")
    urls = settings.get("engine_urls_by_owner")
    if (not isinstance(urls, dict) or owner_id not in urls or not urls
            or any(not isinstance(url, str) or not url.startswith(("http://", "https://")) for url in urls.values())):
        raise ValueError("token-KV requires a valid engine URL for each configured owner")
    directory = settings.get("snapshot_storage_dir")
    threshold = settings.get("interrupt_after_tokens", 16)
    owners = settings.get("interrupt_owner_ids", [])
    if (not isinstance(directory, str) or not Path(directory).is_absolute()
            or type(threshold) is not int or threshold < 1
            or not isinstance(owners, list) or any(owner not in urls for owner in owners)
            or type(settings.get("release_full_gpu_pool", False)) is not bool):
        raise ValueError("invalid token-KV storage, interruption threshold, or source owner list")
    if settings.get("source_group_preemption_budget_s") is not None:
        if (settings.get("source_group_preemption_budget_s") != 600
                or settings.get("source_capture_commit_cap_s") != 45
                or settings.get("source_release_continue_cap_s") != 15
                or not isinstance(settings.get("source_preemption_journal_dir"), str)
                or not Path(settings["source_preemption_journal_dir"]).is_absolute()):
            raise ValueError("joint source requires fixed600/capture45/release15 and an absolute timing journal")
    return {**deepcopy(settings), "engine_url": urls[owner_id], "interrupt_after_tokens": threshold,
            "auto_interrupt": owner_id in owners, "release_full_gpu_pool": settings.get("release_full_gpu_pool", False)}


def configure_worker_token_kv(worker, *, owner_id, settings, run_namespace=None):
    """Called by the existing configure_trajectory_state worker initialization."""
    policy = validated_source_settings(settings, owner_id)
    if run_namespace is not None:
        if not isinstance(run_namespace, str) or not run_namespace:
            raise ValueError("token-KV routing requires its actual run namespace")
        policy["run_namespace"] = run_namespace
    if (policy.get("source_group_preemption_budget_s") is not None
            and worker.trajectory_entry.runtime.preemption_deadline_s != policy["source_group_preemption_budget_s"]):
        raise ValueError("source runtime/publication barrier deadline differs from the admitted group budget")
    timeout = worker.trajectory_entry.runtime.lease_seconds
    attach_token_kv_backend(worker, policy["engine_url"], timeout_s=timeout)
    worker.token_kv_source_backend = TokenKVSourceBackend(worker, policy, timeout_s=timeout)


def generation_payload(arguments, *, rid, response_length, max_model_len):
    if (not isinstance(arguments.get("prompt_ids"), list)
            or any(arguments.get(name) is not None for name in (
                "image_data", "video_data", "audio_data", "mm_processor_output"))
            or arguments.get("mm_processor_kwargs")):
        raise ValueError("token-KV functional source supports the actual plain-text generation path only")
    params = deepcopy(arguments["sampling_params"])
    requested = params.pop("max_tokens", params.get("max_new_tokens", response_length))
    params.pop("max_new_tokens", None)
    params.pop("logprobs", None)
    if type(requested) is not int or requested < 1:
        raise ValueError("generation requires a positive actual cumulative budget")
    limit = min(requested, response_length, max_model_len - len(arguments["prompt_ids"]) - 1)
    if limit < 1:
        raise ValueError("source generation has no model context budget")
    params["max_new_tokens"] = limit
    return {"rid": rid, "stream": True, "input_ids": deepcopy(arguments["prompt_ids"]),
            "sampling_params": params, "return_logprob": True,
            "logprob_start_len": len(arguments["prompt_ids"])}, limit


_SOURCE_SEAL_KEY = "trajectory_source_request_v1"


def seal_source_payload(payload, *, before_generate, lease):
    """Bind native queued state to the lease and input before the original POST."""
    if (payload["input_ids"] != before_generate.agent_state["prompt_ids"]
            or before_generate.trajectory_id != lease.trajectory_id
            or before_generate.execution_generation != lease.generation):
        raise ValueError("source dispatch input/execution differs from its captured trajectory")
    seal = {"schema_version": 1, "request_id": payload["rid"],
            "trajectory_id": lease.trajectory_id, "source_execution_generation": lease.generation,
            "source_owner_id": lease.owner_id, "source_lease_id": lease.lease_id,
            "policy_version": before_generate.policy_identity["policy_version"],
            "policy_identity_sha256": hashlib.sha256(json.dumps(
                before_generate.policy_identity, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
            "origin_input_ids_sha256": hashlib.sha256(json.dumps(
                payload["input_ids"], separators=(",", ":")
            ).encode()).hexdigest()}
    existing = payload["sampling_params"].get("custom_params")
    custom = {} if existing is None else deepcopy(existing)
    if not isinstance(custom, dict) or _SOURCE_SEAL_KEY in custom:
        raise ValueError("source request provenance uses a reserved custom_params key")
    # JSON string is immutable and survives the original typed msgpack IPC.
    custom[_SOURCE_SEAL_KEY] = json.dumps(seal, sort_keys=True, separators=(",", ":"))
    payload["sampling_params"]["custom_params"] = custom
    return seal


def observed_partial_from_export(exported):
    """Normalize actual committed native request metadata, without version defaults."""
    from verl.workers.rollout.replica import TokenOutput

    # Each rank's save_receipt is the actual store.write_rank result; its hash
    # is subsequently checked by build_inflight_snapshot/read_committed_request.
    ranks = exported.get("ranks", [])
    rank0 = next((row for row in ranks if row.get("tp_rank") == 0), None)
    if rank0 is None:
        raise ValueError("actual source TP rank zero receipt is missing")
    metadata = rank0["save_receipt"]["request_metadata"]
    spans = metadata.get("output_weight_versions")
    if not isinstance(spans, list) or not spans:
        raise ValueError("native exported output version spans are missing")
    actual_meta = {"id": metadata["rid"], "weight_version": spans[-1]["version"],
                   "weight_versions": deepcopy(spans), "observation_source": "COMMITTED_ENGINE_REQUEST_METADATA"}
    return TokenOutput(token_ids=list(metadata["output_ids"]), log_probs=list(metadata["output_logprobs"]),
                       extra_fields={"native_meta_info": actual_meta,
                                     "committed_native_request_metadata": deepcopy(metadata)})


def completed_sse_output(last, *, rid, limit, observed_identity, headers, observed_cache_salt=None):
    from verl.workers.rollout.sglang_gateway_client import GatewayHTTPResponse, SGLangGatewayClient

    rows = last.get("meta_info", {}).get("output_token_logprobs")
    if not isinstance(rows, list) or any(not isinstance(row, (list, tuple)) or len(row) < 2 for row in rows):
        raise ValueError("source SSE has no actual output-token logprobs")
    ids = [row[1] for row in rows]
    if last.get("output_ids") is not None and last["output_ids"] != ids:
        raise ValueError("source SSE IDs and logprobs differ")
    result = SGLangGatewayClient._decode_generation(
        GatewayHTTPResponse(200, {**last, "output_ids": ids}, headers), return_logprob=True,
        limit=limit, request_id=rid, observed_weight_version=int(observed_identity["actualweight_version"]),
        observed_cache_salt=observed_cache_salt or "",
    )
    if result.extra_fields["engine_weight_version"] != str(observed_identity["actualweight_version"]):
        raise ValueError("source generation used a different policy version")
    if observed_cache_salt is None:
        result.extra_fields["cache_salt"] = result.extra_fields["salt_match"] = None
    elif result.extra_fields["salt_match"] is not True:
        raise ValueError("routed source generation differs from its versioned cache namespace")
    if result.stop_reason not in {"stop", "length", "completed"}:
        raise ValueError("functional source did not complete or commit a checkpoint")
    return result


class SnapshotControlConflict(RuntimeError):
    """A native refusal with retained body; this exception never authorizes replay."""

    def __init__(self, *, engine_url, path, payload, http_status, raw_body, body):
        self.engine_url, self.path, self.http_status = engine_url, path, http_status
        self.operation = payload.get("operation")
        self.request_id = payload.get("target_rid") or payload.get("rid")
        self.snapshot_id = payload.get("snapshot_id")
        self.response_body = body
        self.response_body_sha256 = hashlib.sha256(raw_body).hexdigest()
        self.response_body_bytes = len(raw_body)
        self.response_body_excerpt = raw_body[:8192].decode("utf-8", errors="replace")
        super().__init__(json.dumps(self.diagnostic(), sort_keys=True, ensure_ascii=True))

    def diagnostic(self):
        return {"engine_url": self.engine_url, "path": self.path,
                "operation": self.operation, "request_id": self.request_id,
                "snapshot_id": self.snapshot_id, "http_status": self.http_status,
                "body_bytes": self.response_body_bytes, "body_sha256": self.response_body_sha256,
                "body_excerpt": self.response_body_excerpt,
                "body_excerpt_truncated": self.response_body_bytes > 8192}


class TokenKVSourceBackend(PortableSGLangHTTPBackend):
    def __init__(self, worker, policy, *, timeout_s):
        super().__init__(policy["engine_url"], timeout_s=timeout_s)
        self.worker, self.policy = worker, policy
        self._control_lock = asyncio.Lock()
        # Read-only evidence for an external controller. A status of running
        # never becomes generation progress: only an SSE row with a real
        # integer completion_tokens updates this table.
        self._generation_progress = {}

    async def control(self, path, payload):
        import aiohttp

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as session:
            async with session.post(self.engine_url + path, json=payload) as response:
                # SGLang uses 409 for several distinct native state refusals.
                # Read that result before raising; a status alone cannot classify it.
                raw = await response.read()
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    value = None
                if (response.status >= 400 or not isinstance(value, dict)
                        or value.get("error") or value.get("status") == "FAILED"):
                    raise SnapshotControlConflict(
                        engine_url=self.engine_url, path=path, payload=payload,
                        http_status=response.status, raw_body=raw, body=value,
                    )
                return value

    def group_state(self, uid):
        runtime = self.worker.trajectory_entry.runtime
        return runtime.group_status(uid)

    def generation_progress(self):
        """Return actual SSE-observed generation progress, grouped by UID."""
        return deepcopy(self._generation_progress)

    def _observe_generation(self, before_generate, *, completion_tokens, active, lease, native_request_id=None, source_instance=None, engine_weight_version=None):
        if type(completion_tokens) is not int or completion_tokens < 0:
            raise ValueError("generation progress requires an actual nonnegative token count")
        row = self._generation_progress.setdefault(before_generate.group_id, {})
        previous = row.get(before_generate.member_index, {})
        row[before_generate.member_index] = {
            "native_request_id": native_request_id or previous.get("native_request_id"),
            "source_owner_id": lease.owner_id, "source_lease_id": lease.lease_id,
            "source_engine_instance_id": source_instance or previous.get("source_engine_instance_id"),
            "engine_weight_version": engine_weight_version if engine_weight_version is not None else previous.get("engine_weight_version"),
            "completion_tokens": completion_tokens,
            "active": bool(active),
            "engine_url": self.engine_url,
            "trajectory_id": lease.trajectory_id,
            "execution_generation": lease.generation,
            "source_preemption_events": deepcopy(previous.get("source_preemption_events", []))[-16:],
            "source_preemption_sequence": previous.get("source_preemption_sequence", 0),
        }

    def _mark_generation_inactive(self, before_generate, lease):
        row = self._generation_progress.get(before_generate.group_id, {})
        current = row.get(before_generate.member_index, {}).get("completion_tokens", 0)
        self._observe_generation(before_generate, completion_tokens=current, active=False, lease=lease)

    @staticmethod
    def _validate_release(released, exported):
        ranks = released.get("ranks", [])
        if (released.get("status") != "PASS" or released.get("identity") != exported["identity"]
                or sorted(row.get("tp_rank") for row in ranks) != list(range(exported["identity"]["tp_size"]))
                or any(
                    row.get("old_slots_released") is not True
                    or row.get("detached") is not True
                    or row.get("engine_generation_unblocked") is not True
                    or row.get("native_release_mode") != "release_kv_cache(is_insert=False)"
                    or row.get("destroyed_old_gpu_slots") is True
                    for row in ranks
                )):
            raise ValueError("actual source slots were not safely detached on every TP rank")

    async def settle_unstarted(self, before_generate, *, lease, commit_snapshot, check_owner):
        from .preemptible_tool_agent_loop import TrajectoryCheckpointYield, TrajectoryPendingYield

        state = self.group_state(before_generate.group_id)
        if not state or not state["preemption_requested"]:
            return False
        await check_owner()
        if not any(before_generate.agent_state["response_mask"]) and not before_generate.completed_tool_batches:
            raise TrajectoryPendingYield(lease.trajectory_id)
        # A later turn may be waiting while its completed tool batch is durable.
        # Use the existing tool-boundary contract rather than inventing empty KV.
        reference = await commit_snapshot(before_generate)
        if not isinstance(reference, str) or not reference:
            raise ValueError("waiting completed-tool state did not commit")
        await check_owner()
        raise TrajectoryCheckpointYield(before_generate, reference)

    async def _persist_preemption_event(self, trace, stage, event, **fields):
        if trace is None:
            return
        payload = {**trace, "stage": stage, "event": event,
                   "monotonic_s": time.monotonic(), "pid": os.getpid(), **fields}
        # Reuse the existing progress RPC so phase terminal export retains the
        # branch even when the source journal itself is outside the seven files.
        row = self._generation_progress.get(trace.get("uid"), {}).get(trace.get("member_index"))
        if row is not None:
            allowed = ("stage", "event", "monotonic_s", "request_id", "row_kind", "error_type")
            evidence = {key: payload[key] for key in allowed if key in payload}
            evidence["sequence"] = row.get("source_preemption_sequence", 0) + 1
            row["source_preemption_sequence"] = evidence["sequence"]
            row.setdefault("source_preemption_events", []).append(evidence)
            del row["source_preemption_events"][:-16]
        directory = self.policy.get("source_preemption_journal_dir")
        if not directory:
            return
        def append():
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            name = hashlib.sha256((trace["trajectory_id"] + ":" + trace["request_id"]).encode()).hexdigest()
            raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
            if len(raw) > 16384:
                raise ValueError("preemption timing record exceeds bounded metadata limit")
            target = root / ("source-" + name + ".jsonl")
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(target, flags, 0o600)
            try:
                with os.fdopen(fd, "ab", closefd=False) as stream:
                    stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            finally:
                os.close(fd)
        try:
            await asyncio.to_thread(append)
        except Exception as error:
            # Timing I/O cannot replace a native control/commit failure. The
            # durable trajectory and physical snapshot remain authoritative.
            logging.getLogger(__name__).warning("source preemption timing write failed: %s", type(error).__name__)

    @asynccontextmanager
    async def _preemption_stage(self, trace, stage, *, deadline=None):
        started = time.monotonic()
        await self._persist_preemption_event(trace, stage, "start")
        try:
            if deadline is None:
                yield
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("source member capture/release phase budget exhausted: " + stage)
                async with asyncio.timeout(remaining):
                    yield
                if time.monotonic() > deadline:
                    raise TimeoutError("source synchronous stage exceeded its finite capture deadline: " + stage)
        except BaseException as error:
            outcome = "yield" if type(error).__name__ in {"TrajectoryCheckpointYield", "TrajectoryPendingYield"} else "error"
            details = ({"native_control_conflict": error.diagnostic()}
                       if isinstance(error, SnapshotControlConflict) else {})
            await self._persist_preemption_event(trace, stage, outcome, elapsed_s=time.monotonic()-started,
                                                 error_type=type(error).__name__, **details)
            raise
        else:
            await self._persist_preemption_event(trace, stage, "end", elapsed_s=time.monotonic()-started)

    async def commit_export_and_release(self, exported, *, before_generate, arguments, lease,
                                        commit_snapshot, check_owner, guard, trace=None, capture_deadline=None):
        from .preemptible_tool_agent_loop import TrajectoryCheckpointYield

        async with self._preemption_stage(trace, "snapshot_build", deadline=capture_deadline):
            snapshot = build_inflight_snapshot(
                before_generate_snapshot=before_generate, export_receipt=exported, source_lease=lease,
                partial_output=observed_partial_from_export(exported),
                generation_call_sampling_params=arguments["sampling_params"],
            )
        async with self._preemption_stage(trace, "trajectory_commit", deadline=capture_deadline):
            await check_owner()
            reference = await commit_snapshot(snapshot)
        release_cap = self.policy.get("source_release_continue_cap_s")
        release_deadline = time.monotonic() + release_cap if release_cap is not None else None
        if reference is None:
            async with self._preemption_stage(trace, "continue_generation", deadline=release_deadline):
                await self.control("/continue_generation", {"torch_empty_cache": False})
            return
        if not isinstance(reference, str) or not reference:
            raise ValueError("source cannot release before its durable trajectory reference exists")
        async with self._preemption_stage(trace, "native_release", deadline=release_deadline):
            await check_owner()
            released = await self.control("/token_kv_snapshot", {
                **guard, "operation": "release", "release_full_gpu_pool": self.policy["release_full_gpu_pool"],
            })
            self._validate_release(released, exported)
        async with self._preemption_stage(trace, "continue_generation", deadline=release_deadline):
            await self.control("/continue_generation", {"torch_empty_cache": False})
        raise TrajectoryCheckpointYield(snapshot, reference)

    @staticmethod
    def _source_identity(observed, before_generate):
        if observed.get("supported") is not True:
            raise ValueError("source engine did not satisfy the actual token-KV runtime constraints")
        identity = observed["identity"]
        if (str(identity.get("weight_version")) != before_generate.policy_identity["policy_version"]
                or str(identity.get("actualweight_version")) != before_generate.policy_identity["policy_version"]):
            raise ValueError("source engine differs from the actual published trajectory policy")
        instance = observed.get("engine_instance_id")
        if not isinstance(instance, str) or not instance:
            raise ValueError("actual source engine process identity is missing")
        return identity, instance

    @asynccontextmanager
    async def source_generation_channel(self, payload, *, before_generate, arguments):
        """Bind snapshot control to this request's actual routed engine only."""
        import aiohttp
        from verl.workers.rollout.sglang_gateway_lifecycle import make_gateway_client, native_gateway_enabled

        if native_gateway_enabled(self.worker.config):
            from verl.workers.rollout.sglang_gateway_client import SGLangGatewayClient
            from .routed_token_kv_stream import open_routed_token_kv_stream

            urls = tuple(dict.fromkeys(getattr(self.worker.llm_client, "engine_urls", ())))
            if not urls:
                raise ValueError("routed token-KV requires actual native manager engine endpoints")
            namespace = self.policy.get("run_namespace")
            if not isinstance(namespace, str) or not namespace:
                raise ValueError("routed token-KV requires the actual trajectory run namespace")
            identity = before_generate.policy_identity
            fingerprint = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            request = {**payload, "cache_salt": arguments.get("cache_salt") or
                       SGLangGatewayClient.build_policy_cache_salt(
                           run_namespace=namespace, policy_version=identity["policy_version"],
                           policy_identity_salt=fingerprint)}
            gateway = make_gateway_client(self.worker.config, list(urls))
            try:
                async with open_routed_token_kv_stream(
                    gateway, request, expected_weight_version=int(identity["policy_version"])
                ) as routed:
                    # Every active stream gets a separate binding. Mutating the
                    # shared backend would send concurrent controls to the wrong GPU.
                    bound = copy(self)
                    bound.engine_url = routed.engine_url
                    actual_identity, instance = self._source_identity(await bound.inspect(), before_generate)
                    yield bound, routed.response, actual_identity, instance, request["cache_salt"]
            finally:
                await gateway.close()
            return

        raise ValueError("Production token_kv_resume requires the configured native Gateway routed stream")

    async def generate_owned(self, arguments, *, before_generate, lease, commit_snapshot, check_owner):
        if not callable(commit_snapshot) or not callable(check_owner) or lease is None:
            raise ValueError("functional source requires the original owned commit/check callbacks")
        await self.settle_unstarted(before_generate, lease=lease, commit_snapshot=commit_snapshot, check_owner=check_owner)
        await check_owner()
        rid = uuid4().hex
        rollout = self.worker.config.actor_rollout_ref.rollout
        maximum = getattr(rollout, "max_model_len", None) or rollout.prompt_length + rollout.response_length
        payload, limit = generation_payload(arguments, rid=rid, response_length=rollout.response_length, max_model_len=maximum)
        seal_source_payload(payload, before_generate=before_generate, lease=lease)
        async with self.source_generation_channel(payload, before_generate=before_generate, arguments=arguments) as channel:
            bound, response, identity, instance, salt = channel
            return await bound._consume_source_stream(
                response, arguments=arguments, before_generate=before_generate, lease=lease,
                commit_snapshot=commit_snapshot, check_owner=check_owner,
                rid=rid, limit=limit, identity=identity, instance=instance, observed_cache_salt=salt,
            )

    @staticmethod
    def _validate_cancelled_pending(receipt, *, guard, arguments):
        ranks = receipt.get("ranks", [])
        expected_ranks = list(range(guard["identity"]["tp_size"]))
        prompt_digest = hashlib.sha256(json.dumps(
            arguments["prompt_ids"], sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if (receipt.get("status") != "PASS" or receipt.get("identity") != guard["identity"]
                or sorted(row.get("tp_rank") for row in ranks) != expected_ranks
                or any(row.get("status") != "PASS"
                       or row.get("source_disposition") != "PENDING_CANCELLED"
                       or row.get("identity") != guard["identity"]
                       or row.get("source_binding") != guard["source_binding"]
                       or row.get("origin_input_ids_sha256") != prompt_digest
                       or row.get("generated_tokens") != 0
                       or row.get("pending_removed") is not True
                       or row.get("kv_allocated") is not False
                       or row.get("prepared_while_paused") is not True
                       or type(row.get("overlap_results_drained")) is not int
                       or row["overlap_results_drained"] < 0
                       or row.get("native_cancel_mode") != "EXACT_WAITING_QUEUE_REMOVE_AFTER_TP_PREPARE"
                       for row in ranks)):
            raise ValueError("source did not atomically cancel the exact zero-output request on every TP rank")

    @staticmethod
    def _ready_source_row_kind(generation_read_task):
        """Peek at readiness without consuming/replacing the parser's Task."""
        if generation_read_task is None or not generation_read_task.done():
            return "not_ready"
        try:
            line = generation_read_task.result()
        except StopAsyncIteration:
            return "eof"
        except asyncio.CancelledError:
            return "cancelled"
        except BaseException:
            return "read_error"
        if not isinstance(line, (bytes, bytearray)):
            return "invalid_row"
        if not line.startswith(b"data:"):
            return "keepalive"
        raw = bytes(line[5:]).strip()
        if not raw:
            return "keepalive"
        if raw == b"[DONE]":
            return "done"
        try:
            row = json.loads(raw)
        except (TypeError, ValueError, UnicodeDecodeError):
            return "invalid_json"
        if not isinstance(row, dict):
            return "invalid_row"
        if row.get("error"):
            return "native_error"
        meta = row.get("meta_info", {})
        if not isinstance(meta, dict):
            return "invalid_meta"
        if meta.get("finish_reason") is not None:
            return "finish"
        if type(meta.get("completion_tokens")) is not int or meta["completion_tokens"] < 0:
            return "invalid_meta"
        return "nonterminal_progress"

    async def _preempt_source_request(self, *, arguments, before_generate, lease,
                                      commit_snapshot, check_owner, rid, identity, instance,
                                      generation_read_task=None):
        trace = {"schema_version": 1, "uid": before_generate.group_id,
                 "trajectory_id": lease.trajectory_id, "member_index": before_generate.member_index,
                 "source_execution_generation": lease.generation, "request_id": rid,
                 "engine_url": self.engine_url, "engine_instance_id": instance,
                 "weight_version": identity["weight_version"], "group_cap_s": self.policy.get("source_group_preemption_budget_s")}
        waiting = time.monotonic()
        await self._persist_preemption_event(trace, "control_lock_wait", "start")
        await self._control_lock.acquire()
        try:
            await self._persist_preemption_event(trace, "control_lock_wait", "end", elapsed_s=time.monotonic()-waiting)
            # The SSE reader stays live while another member owns this lock.
            # Its ready row belongs to the normal parser, including completion
            # and errors. A merely ready progress row, however, is not a
            # completion race and must not starve capture.
            ready_kind = self._ready_source_row_kind(generation_read_task)
            await self._persist_preemption_event(trace, "ready_row_before_pause", "observed", row_kind=ready_kind)
            if ready_kind not in {"not_ready", "keepalive", "nonterminal_progress"}:
                await self._persist_preemption_event(trace, "source_row_ready_after_control_wait", "observed")
                return False
            capture_cap = self.policy.get("source_capture_commit_cap_s")
            capture_deadline = time.monotonic() + capture_cap if capture_cap is not None else None
            async with self._preemption_stage(trace, "pause_generation", deadline=capture_deadline):
                await check_owner()
                await self.control("/pause_generation", {"mode": "in_place"})
            ready_kind = self._ready_source_row_kind(generation_read_task)
            await self._persist_preemption_event(trace, "ready_row_after_pause", "observed", row_kind=ready_kind)
            if ready_kind not in {"not_ready", "keepalive", "nonterminal_progress"}:
                # A completed forward can reach the SSE reader while pause is
                # being acknowledged. Resume before returning it to its parser.
                async with self._preemption_stage(trace, "continue_generation_before_ready_row", deadline=capture_deadline):
                    await self.control("/continue_generation", {"torch_empty_cache": False})
                return False
            binding = {"trajectory_id": lease.trajectory_id, "source_execution_generation": lease.generation,
                       "source_owner_id": lease.owner_id, "source_lease_id": lease.lease_id,
                       "source_engine_instance_id": instance, "request_id": rid,
                       "origin_input_ids_sha256": hashlib.sha256(json.dumps(arguments["prompt_ids"], separators=(",", ":")).encode()).hexdigest(),
                       "policy_identity_sha256": hashlib.sha256(json.dumps(before_generate.policy_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
            guard = {"rid": rid, "snapshot_id": "trajectory-" + uuid4().hex,
                     "storage_dir": self.policy["snapshot_storage_dir"],
                     "identity": identity, "expected_weight_version": identity["weight_version"],
                     "source_binding": binding}
            async with self._preemption_stage(trace, "native_export_or_pending_cancel_rpc", deadline=capture_deadline):
                exported = await self.control("/token_kv_snapshot", {**guard, "operation": "preempt"})
            await self._persist_preemption_event(trace, "native_rank_durations", "observed",
                native_return_status=exported.get("status"),
                tokenizer_manifest_commit_s=exported.get("manifest_commit_s"),
                tokenizer_clock_scope="same_native_tokenizer_handler_local_duration",
                clock_scope="each_native_TP_rank_local_duration", ranks=[{
                    key: row.get(key) for key in ("tp_rank", "copy_to_host_s", "persist_rank_s", "operation_s", "source_disposition")
                } for row in exported.get("ranks", [])])
            dispositions = {row.get("source_disposition") for row in exported.get("ranks", [])}
            if dispositions == {"PENDING_CANCELLED"}:
                self._validate_cancelled_pending(exported, guard=guard, arguments=arguments)
                commit_pending = getattr(check_owner, "commit_unstarted_generation", None)
                if not callable(commit_pending):
                    raise ValueError("source runtime cannot durably record its cancelled pending request")
                async with self._preemption_stage(trace, "pending_input_commit", deadline=capture_deadline):
                    reference = await commit_pending(exported)
                if not isinstance(reference, str) or not reference:
                    raise ValueError("cancelled pending request has no durable original-input record")
                from .preemptible_tool_agent_loop import TrajectoryCheckpointYield, TrajectoryPendingYield
                try:
                    async with self._preemption_stage(trace, "pending_or_completed_tool_snapshot", deadline=capture_deadline):
                        await self.settle_unstarted(before_generate, lease=lease,
                            commit_snapshot=commit_snapshot, check_owner=check_owner)
                except (TrajectoryCheckpointYield, TrajectoryPendingYield):
                    release_cap = self.policy.get("source_release_continue_cap_s")
                    release_deadline = time.monotonic() + release_cap if release_cap is not None else None
                    async with self._preemption_stage(trace, "continue_generation", deadline=release_deadline):
                        await self.control("/continue_generation", {"torch_empty_cache": False})
                    raise
                raise ValueError("cancelled source request lost its live preemption")
            if dispositions != {"ACTIVE_EXPORTED"}:
                raise ValueError("source preemption has no unanimous native request disposition")
            await self.commit_export_and_release(
                exported, before_generate=before_generate, arguments=arguments, lease=lease,
                commit_snapshot=commit_snapshot, check_owner=check_owner, guard=guard,
                trace=trace, capture_deadline=capture_deadline,
            )
        finally:
            self._control_lock.release()

    async def _consume_source_stream(self, response, *, arguments, before_generate, lease,
                                     commit_snapshot, check_owner, rid, limit, identity,
                                     instance, observed_cache_salt):
        last = None
        headers = dict(response.headers)
        lines = response.content.__aiter__()
        read_task = asyncio.create_task(anext(lines))
        runtime = self.worker.trajectory_entry.runtime
        signal_task = (asyncio.create_task(runtime.wait_for_preemption(before_generate.group_id))
                       if self.group_state(before_generate.group_id) is not None else None)
        naturally_finished = False
        trace = {"uid": before_generate.group_id, "trajectory_id": lease.trajectory_id,
                 "member_index": before_generate.member_index, "request_id": rid}
        self._observe_generation(before_generate, completion_tokens=0, active=True, lease=lease,
            native_request_id=rid, source_instance=instance, engine_weight_version=identity["weight_version"])
        try:
            while True:
                waits = {read_task}
                if signal_task is not None:
                    waits.add(signal_task)
                done, _ = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
                interrupt = False
                # Consume a ready native completion first. It must not become
                # a spurious export of a request the engine already finished.
                if read_task in done:
                    try:
                        line = read_task.result()
                    except StopAsyncIteration:
                        await self._persist_preemption_event(trace, "source_stream_terminal", "observed", row_kind="eof")
                        break
                    read_task = asyncio.create_task(anext(lines))
                    if line.startswith(b"data:"):
                        raw = line[5:].strip()
                        if raw and raw != b"[DONE]":
                            last = json.loads(raw)
                            if not isinstance(last, dict) or last.get("error"):
                                raise ValueError("source generation returned an error")
                            meta = last.get("meta_info", {})
                            naturally_finished = meta.get("finish_reason") is not None
                            if naturally_finished:
                                await self._persist_preemption_event(trace, "source_stream_terminal", "observed", row_kind="finish")
                            if not naturally_finished:
                                count = meta.get("completion_tokens")
                                if type(count) is not int:
                                    raise ValueError("source SSE omitted its actual generated-token count")
                                self._observe_generation(before_generate, completion_tokens=count, active=True, lease=lease,
                                    native_request_id=rid, source_instance=instance, engine_weight_version=identity["weight_version"])
                                if self.policy["auto_interrupt"] and count >= self.policy["interrupt_after_tokens"]:
                                    if await self.worker.request_trajectory_preemption(before_generate.group_id) is not True:
                                        raise ValueError("original worker did not accept the actual group preemption")
                                    interrupt = True
                if signal_task is not None and signal_task in done:
                    interrupt = signal_task.result() or interrupt
                    signal_task = None
                state = self.group_state(before_generate.group_id)
                interrupt = interrupt or bool(state and state["preemption_requested"])
                if naturally_finished:
                    if signal_task is not None:
                        signal_task.cancel()
                        await asyncio.gather(signal_task, return_exceptions=True)
                        signal_task = None
                    continue
                if interrupt:
                    await self._preempt_source_request(
                        arguments=arguments, before_generate=before_generate, lease=lease,
                        commit_snapshot=commit_snapshot, check_owner=check_owner, rid=rid,
                        identity=identity, instance=instance, generation_read_task=read_task,
                    )
            if last is None:
                raise ValueError("source engine returned no actual generation output")
            final_count = last.get("meta_info", {}).get("completion_tokens")
            if type(final_count) is not int:
                final_count = len(last.get("meta_info", {}).get("output_token_logprobs") or [])
            self._observe_generation(before_generate, completion_tokens=final_count, active=False, lease=lease)
            await check_owner()
            return completed_sse_output(last, rid=rid, limit=limit, observed_identity=identity,
                                        headers=headers, observed_cache_salt=observed_cache_salt)
        finally:
            tasks = [task for task in (read_task, signal_task) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._mark_generation_inactive(before_generate, lease)


# Compatibility for configurations and tests that still refer to the old name.
SingletonTokenKVSourceBackend = TokenKVSourceBackend
