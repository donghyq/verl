"""Optional finite native-PartialRollout versus durable CPU-worker campaign.

The audited trainer freezes real prepared business records. Each arm uses the
same factory, sampler, reward and published weights. Only the durable arm binds
a snapshot runtime. Experimental data go to separate TransferQueue partitions;
they never reach the optimizer. There is one verified CPU actor exit per arm.
No GPU replica or unverified process is stopped.
"""
from __future__ import annotations

import asyncio
from copy import copy, deepcopy
from contextvars import ContextVar
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4

from storage_handoff_driver import digest, event, install_worker_observer, wait_for, write_private

COMPLETED = [0, 1]
CHECKPOINTED = [2, 3, 4, 5]
PENDING = [6, 7]
ARMS = ["native_partial", "durable"]
_CALL = ContextVar("storage_native_generate_call", default=None)


def normalize_spec(value):
    """A bounded experiment specification, never manually supplied model data."""
    if not isinstance(value, dict) or value.get("enabled") is not True:
        raise ValueError("Campaign requires an explicitly enabled materials specification")
    allowed = {"enabled", "at_step", "dataset_indices", "source_worker_index", "target_worker_index",
               "total_timeout_s", "arm_timeout_s", "arm_order"}
    if set(value) - allowed:
        raise ValueError("Unknown storage experiment fields: " + str(sorted(set(value) - allowed)))
    result = {"enabled": True, "at_step": value.get("at_step"),
              "dataset_indices": value.get("dataset_indices", [0]),
              "source_worker_index": value.get("source_worker_index", 0),
              "target_worker_index": value.get("target_worker_index", 1),
              "total_timeout_s": value.get("total_timeout_s", 1800),
              "arm_timeout_s": value.get("arm_timeout_s", 600),
              "arm_order": value.get("arm_order", ARMS)}
    if type(result["at_step"]) is not int or result["at_step"] < 1:
        raise ValueError("Campaign must follow a verified optimizer update")
    indices = result["dataset_indices"]
    if (not isinstance(indices, list) or not 1 <= len(indices) <= 2 or len(set(indices)) != len(indices)
            or any(type(i) is not int or i < 0 for i in indices)):
        raise ValueError("Freeze one or two distinct real prepared dataset indices")
    if any(type(result[k]) is not int or result[k] < 0 for k in ("source_worker_index", "target_worker_index")):
        raise ValueError("Worker indices must be nonnegative integers")
    if result["source_worker_index"] == result["target_worker_index"]:
        raise ValueError("Source and surviving target must be different workers")
    for k in ("total_timeout_s", "arm_timeout_s"):
        if type(result[k]) not in (int, float) or not math.isfinite(result[k]) or not 1 <= result[k] <= 3600:
            raise ValueError("Campaign wall-clock budgets must be finite and at most 3600 seconds")
    order = result["arm_order"]
    if result["arm_timeout_s"] > result["total_timeout_s"] or not isinstance(order, list) or sorted(order) != sorted(ARMS):
        raise ValueError("Run each actual arm once within the total budget")
    return deepcopy(result)


class MergedCallObserver:
    """Keep the native PartialRollout loop and verify its real merged receipts."""
    def __init__(self, client, spec, owner, member):
        self.client, self.spec, self.owner, self.member = client, spec, owner, dict(member)

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def generate(self, *args, **kwargs):
        observed = {"id": uuid4().hex, "attempts": []}
        handle = _CALL.set(observed)
        try:
            output = await self.client.generate(*args, **kwargs)
            attempts = observed["attempts"]
            tokens = [token for a in attempts for token in a["token_ids"]]
            logs = [log for a in attempts for log in (a["behavior_logprobs"] or [])]
            if tokens != output.token_ids or logs != output.log_probs:
                raise ValueError("Original native PartialRollout changed actual attempt token/logprob history")
            receipts = output.extra_fields.get("native_gateway_attempts", [])
            cursor = 0
            if len(receipts) != len(attempts):
                raise ValueError("Original native PartialRollout omitted an actual attempt receipt")
            for receipt, attempt in zip(receipts, attempts, strict=True):
                if receipt["output_start"] != cursor or receipt["output_end"] != cursor + len(attempt["token_ids"]):
                    raise ValueError("Original native attempt ranges do not preserve generation history")
                cursor = receipt["output_end"]
            aborted = sum(a["stop_reason"] in ("abort", "aborted") for a in attempts)
            event(self.spec["root"], self.owner, "native_partial_call_verified", **self.member,
                  logical_call_id=observed["id"], actual_attempts=len(attempts),
                  actual_aborted_attempts=aborted, retained_token_logprob_history_verified=True,
                  actual_partial_resume_verified=aborted > 0 and len(attempts) > 1,
                  internal_retry_count=max(0, len(attempts) - 1))
            return output
        finally:
            _CALL.reset(handle)


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    # Arrow/numpy containers are normalized before transport, preserving values.
    if type(value).__module__.startswith("numpy"):
        return _json_value(value.tolist())
    raise ValueError("Business campaign input must be exact text/JSON data: " + type(value).__name__)


def freeze_inputs(trainer, spec, proof, root):
    """Read the actual prepared dataset without advancing its training sampler."""
    if trainer.global_steps != spec["at_step"]:
        raise ValueError("Freeze must use the publication's actual trainer step")
    rollout = trainer.config.actor_rollout_ref.rollout
    if type(rollout.n) is not int or rollout.n != 8 or not rollout.calculate_log_probs:
        raise ValueError("Campaign requires the original n=8 text/logprob configuration")
    sampling = {"temperature": rollout.temperature, "top_p": rollout.top_p,
                "top_k": rollout.top_k, "repetition_penalty": 1.0, "logprobs": rollout.calculate_log_probs}
    if rollout.topk_log_probs or getattr(rollout, "enable_rollout_routing_replay", False):
        raise ValueError("Campaign does not support top-k heads or expert replay")
    rows = []
    for index in spec["dataset_indices"]:
        if index >= len(trainer.train_dataset):
            raise ValueError("Requested real business dataset index is absent")
        original = deepcopy(trainer.train_dataset[index])
        original.pop("dummy_tensor", None)  # Dataset transport placeholder; no model or reward input.
        row = _json_value(original)
        required = ("raw_prompt", "agent_name", "data_source", "reward_model", "extra_info")
        if any(k not in row for k in required) or not row["raw_prompt"] or "ground_truth" not in row["reward_model"]:
            raise ValueError("Prepared business record is incomplete; never substitute a fixture")
        if row.get("__rollout_n__", 8) != 8:
            raise ValueError("Prepared business record overrides the original group size")
        if any(row.get(k) for k in ("multi_modal_data", "images", "videos", "audios")):
            raise ValueError("The durable text path cannot freeze multimodal input")
        rows.append({"dataset_index": index, "business_row": row, "business_row_sha256": digest(row)})
    payload = {"schema_version": 1, "source": "original_trainer.train_dataset.__getitem__",
               "sampler_advanced": False, "optimizer_updates_added": 0, "spec": spec,
               "verified_publication": proof, "sampling_params": _json_value(sampling), "groups": rows,
               "mixed_group": {"completed": COMPLETED, "tool_boundary": CHECKPOINTED, "pending": PENDING}}
    write_private(Path(root) / "frozen-business-input.json", json.dumps(payload, ensure_ascii=False,
                  sort_keys=True, allow_nan=False, indent=2).encode())
    return payload


def _members(frozen, uid, step, sampling):
    row = deepcopy(frozen["business_row"])
    row["uid"], row["global_steps"] = uid, step
    row.pop("__rollout_n__", None)
    params = deepcopy(sampling)
    if not row.pop("__do_sample__", True):
        from verl.trainer.ppo.v1.agent_loop_tq import apply_greedy_sampling_params
        apply_greedy_sampling_params(params)
    trajectory = {"step": step, "sample_index": row.get("index", frozen["dataset_index"]),
                  "rollout_n": 8, "validate": False}
    return [{**deepcopy(row), "session_id": i, "sampling_params": deepcopy(params),
             "trajectory": deepcopy(trajectory), "trace": False, "__trajectory_group_size__": 8}
            for i in range(8)]


class AttemptObserver:
    """Observe each real native transport return inside FullyAsync.generate."""
    def __init__(self, transport, spec, owner, member):
        self.transport, self.spec, self.owner, self.member = transport, spec, owner, dict(member)

    def __getattr__(self, name):
        return getattr(self.transport, name)

    async def generate(self, *args, **kwargs):
        request = {"request_id": kwargs.get("request_id"), "prompt_ids": deepcopy(kwargs["prompt_ids"]),
                   "sampling_params": deepcopy(kwargs["sampling_params"])}
        attempt = uuid4().hex
        call = _CALL.get()
        if call is None:
            raise ValueError("Native attempt has no original FullyAsync generate call")
        event(self.spec["root"], self.owner, "native_attempt_dispatched", **self.member,
              attempt_id=attempt, logical_call_id=call["id"], request_id=request["request_id"],
              input_token_ids_sha256=digest(request["prompt_ids"]),
              sampling_params_sha256=digest(request["sampling_params"]))
        started = time.perf_counter()
        output = await self.transport.generate(*args, **kwargs)
        elapsed = time.perf_counter() - started
        extra = output.extra_fields or {}
        meta = extra.get("native_meta_info") or {}
        version = extra.get("engine_weight_version", meta.get("weight_version"))
        if str(version) != self.spec["policy_identity"]["policy_version"]:
            raise ValueError("Actual native attempt ran a different or unknown weight version")
        cached = meta.get("cached_tokens")
        if type(cached) is not int or not 0 <= cached <= len(request["prompt_ids"]):
            cached = None
        observed = {**self.member, **request, "token_ids": list(output.token_ids),
                    "behavior_logprobs": output.log_probs, "stop_reason": output.stop_reason,
                    "extra_fields": _json_value(extra), "attempt_id": attempt, "logical_call_id": call["id"],
                    "actual_weight_version": str(version), "elapsed_s": elapsed}
        path = Path(self.spec["root"]) / ("attempt-" + attempt + ".json")
        io_started = time.perf_counter()
        write_private(path, json.dumps(observed, allow_nan=False, ensure_ascii=False).encode())
        call["attempts"].append(observed)
        event(self.spec["root"], self.owner, "native_attempt_returned", **self.member,
              attempt_id=attempt, logical_call_id=call["id"], request_id=request["request_id"], elapsed_s=elapsed,
              generated_tokens=len(output.token_ids), input_tokens=len(request["prompt_ids"]),
              cached_tokens=cached, uncached_input_tokens=len(request["prompt_ids"]) - cached if cached is not None else None,
              input_token_ids_sha256=digest(request["prompt_ids"]),
              sampling_params_sha256=digest(request["sampling_params"]),
              actual_weight_version=str(version), stop_reason=output.stop_reason,
              gateway_worker_id=extra.get("gateway_worker_id"), evidence_io_s=time.perf_counter() - io_started)
        return output


def _observe_client(loop, spec, owner, member):
    from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
    if not isinstance(loop.server_manager, FullyAsyncLLMServerClient):
        raise ValueError("Both arms must call the actual FullyAsyncLLMServerClient")
    native = loop.server_manager
    if getattr(native, "_generation_transport", None) is None:
        raise ValueError("Attempt observations require the current native Gateway transport")
    observed = copy(native)
    observed._generation_transport = AttemptObserver(native._generation_transport, spec, owner, member)
    loop.server_manager = MergedCallObserver(observed, spec, owner, member)
    event(spec["root"], owner, "actual_loop_bound", **member,
          loop_class=type(loop).__module__ + "." + type(loop).__qualname__,
          loop_mro=[c.__module__ + "." + c.__qualname__ for c in type(loop).__mro__],
          client_class=type(native).__module__ + "." + type(native).__qualname__,
          partial_rollout_enabled=not (hasattr(native.config, "async_training") and not native.config.async_training.partial_rollout),
          transport_class=type(native._generation_transport).__module__ + "." + type(native._generation_transport).__qualname__,
          snapshot_runtime_bound=spec["arm"] == "durable", gateway_session_affinity_asserted=False)


def _observe_tools(loop, spec, owner, member):
    for name, tool in loop.tools.items():
        if getattr(tool, "_storage_handoff_measured", False):
            raise ValueError("Business tools must be trajectory-local instances")
        tool._storage_handoff_measured = True
        execute = tool.execute

        async def measured(*args, _execute=execute, _name=name, **kwargs):
            started = time.perf_counter()
            result = await _execute(*args, **kwargs)
            parameters = kwargs.get("parameters", args[1] if len(args) > 1 else None)
            turn = getattr(kwargs.get("agent_data"), "assistant_turns", None)
            event(spec["root"], owner, "tool_returned", **member, name=_name,
                  turn_index=turn - 1 if type(turn) is int else None,
                  arguments_sha256=digest(parameters), result_text_sha256=digest(getattr(result[0], "text", None)),
                  elapsed_s=time.perf_counter() - started)
            return result
        tool.execute = measured


def _row_digest(fields):
    """Hash a single actual TQ row with its pinned serializer, without rescoring."""
    from transfer_queue.utils.serial_utils import encode
    from verl.trainer.ppo.v1.trajectory_publication import _canonical_value
    hasher = hashlib.sha256()
    for frame in encode(_canonical_value(fields[0])):
        raw = memoryview(frame).cast("B")
        hasher.update(raw.nbytes.to_bytes(8, "big"))
        hasher.update(raw)
    return hasher.hexdigest()


async def _read_rows(partition, uid, indices):
    import transfer_queue as tq
    from verl.trainer.ppo.v1.trajectory_entry import validate_trajectory_fields
    tags = (await tq.async_kv_list(partition_id=partition))[partition]
    rows = {}
    for i in indices:
        key = f"{uid}_{i}_0"
        if tags.get(key, {}).get("status") != "success":
            raise ValueError("Completed original TQ member is absent")
        fields = await tq.async_kv_batch_get(keys=[key], partition_id=partition, select_fields=None)
        validate_trajectory_fields(fields)
        rows[i] = {"fields": fields, "tags": [tags[key]], "key": key, "content_sha256": _row_digest(fields)}
    return rows


async def _baseline_member(worker, state, index, stage, *, hold=False):
    from verl.trainer.ppo.v1.trajectory_entry import V1TrajectoryEntry
    from verl.experimental.agent_loop.tool_agent_loop import AgentState
    import transfer_queue as tq
    original = deepcopy(state["members"][index])
    # This entry is only the existing Hydra factory and stock output preparer.
    # NO runtime, lease, commit callback or snapshot restore exists in this arm.
    entry = V1TrajectoryEntry(worker, None)
    loop = entry._new_loop(original)
    member = {"uid": original["uid"], "trajectory_id": f"{original['uid']}/member-{index}",
              "member_index": index, "stage": stage, "generation": 0 if stage == "source" else 1}
    _observe_client(loop, state["spec"], state["owner"], member)
    _observe_tools(loop, state["spec"], state["owner"], member)
    if hold:
        tools_method = loop._handle_processing_tools_state

        async def completed_tool_boundary(data):
            before = len(data.messages)
            calls = [{"name": c.name, "arguments": c.arguments} for c in data.tool_calls[:loop.max_parallel_calls]]
            next_state = await tools_method(data)
            if next_state == AgentState.GENERATING and index not in state["ready"]:
                results = deepcopy(data.messages[before:])
                if not calls or len(calls) != len(results) or any(r.get("role") != "tool" for r in results):
                    raise ValueError("Native arm has no legal completed real tool batch")
                history = {"prompt_ids": list(data.prompt_ids), "response_mask": list(data.response_mask),
                           "response_logprobs": list(data.response_logprobs), "messages": deepcopy(data.messages),
                           "batch": {"turn_index": data.assistant_turns - 1, "calls": calls, "results": results}}
                write_private(Path(state["spec"]["root"]) / f"native-boundary-{index}.json",
                              json.dumps(history, allow_nan=False, ensure_ascii=False).encode())
                event(state["spec"]["root"], state["owner"], "native_tool_boundary_observed", **member,
                      history_sha256=digest(history), generated_tokens=sum(data.response_mask),
                      completed_tool_calls=len(calls), recoverable_snapshot_created=False)
                state["ready"].add(index)
                await state["hold"].wait()  # The actual CPU exit ends this execution.
            return next_state
        loop._handle_processing_tools_state = completed_tool_boundary
    kwargs = entry._prompt_kwargs(original)
    kwargs.pop("__trajectory_group_size__", None)
    if any(k.startswith("_trajectory_") for k in kwargs):
        raise ValueError("Native PartialRollout arm must bypass every durable control")
    output = await loop.run(original["sampling_params"], **kwargs)
    prepared = await entry._prepare_member(index, output, original)
    await tq.async_kv_batch_put(keys=prepared.keys, fields=prepared.fields, tags=prepared.tags,
                              partition_id=state["spec"]["partition"])
    actual = (await _read_rows(state["spec"]["partition"], original["uid"], [index]))[index]
    state["completed"][index] = actual["content_sha256"]
    event(state["spec"]["root"], state["owner"], "member_completed_and_read_back", **member,
          actual_content_sha256=actual["content_sha256"], original_reward_preparer_used=True)


def _durable_entry(worker, state, client):
    from storage_handoff_driver import _MEMBER
    from verl.trainer.ppo.v1.trajectory_entry import V1TrajectoryEntry
    from verl.trainer.ppo.v1.trajectory_state.runtime import PromptGroupHandoffRuntime
    primary = worker.trajectory_entry.runtime
    measured_client = MeasuredStateClient(client, state)
    runtime = PromptGroupHandoffRuntime(measured_client, owner_id=state["owner"], incarnation_id=uuid4().hex,
        policy_identity=state["spec"]["policy_identity"], environment_version=primary.environment_version,
        tool_version=primary.tool_version, lease_seconds=primary.lease_seconds,
        snapshot_ttl_seconds=primary.snapshot_ttl_seconds, preemption_deadline_s=primary.preemption_deadline_s,
        published_policy_client=measured_client)
    entry = V1TrajectoryEntry(worker, runtime)
    factory = entry._new_loop

    def observed_factory(original):
        loop = factory(original)
        member = _MEMBER.get()
        if member is None:
            raise ValueError("Durable observer has no actual execution identity")
        _observe_client(loop, state["spec"], state["owner"], member)
        return loop
    entry._new_loop = observed_factory
    install_worker_observer(entry, state["spec"]["root"])
    state["entry"] = entry
    return entry


class MeasuredStateClient:
    """Time the actual isolated native store RPCs, including snapshot loads."""
    def __init__(self, client, state):
        self.client, self.state = client, state

    def __getattr__(self, name):
        function = getattr(self.client, name)
        if not callable(function):
            return function
        async def observed(*args, **kwargs):
            started = time.perf_counter()
            result = await function(*args, **kwargs)
            returned_bytes = len(result) if isinstance(result, bytes) else None
            event(self.state["spec"]["root"], self.state["owner"], "native_state_rpc_returned",
                  method=name, elapsed_s=time.perf_counter() - started,
                  returned_payload_bytes=returned_bytes)
            return result
        return observed


async def _durable_source(worker, state, client):
    entry = _durable_entry(worker, state, client)
    entry.enable_acceptance_source_observation(state["uid"], checkpoint_members=CHECKPOINTED)
    result = await entry.runtime.run_source(state["uid"], state["members"], entry._run_source_member,
                                            start_members=COMPLETED + CHECKPOINTED)
    if result.status != "handed_off" or not result.handoff_id:
        raise ValueError("Durable source did not produce a committed mixed-group handoff")
    state["handoff_id"] = result.handoff_id


async def _native_finish(worker, state):
    import transfer_queue as tq
    from verl.trainer.ppo.v1.trajectory_entry import assemble_prepared_fields, validate_trajectory_fields
    from verl.trainer.ppo.v1.trajectory_publication import PublicationMember, _logical_rows, _validate_members, training_payload_digest
    existing = await _read_rows(state["spec"]["partition"], state["uid"], COMPLETED)
    expected = {int(i): v for i, v in state["spec"]["completed_digests"].items()}
    if {i: row["content_sha256"] for i, row in existing.items()} != expected:
        raise ValueError("CPU exit changed an already completed native TQ member")
    # Reexecute only the four interrupted and two never-started members.
    await asyncio.gather(*[_baseline_member(worker, state, i, "target") for i in CHECKPOINTED + PENDING])
    rows = await _read_rows(state["spec"]["partition"], state["uid"], list(range(8)))
    if {i: rows[i]["content_sha256"] for i in COMPLETED} != expected:
        raise ValueError("Native replay rewrote completed members")
    fields = assemble_prepared_fields([rows[i]["fields"] for i in range(8)])
    members = [PublicationMember(state["uid"], i) for i in range(8)]
    _validate_members(members, _logical_rows(fields))
    validate_trajectory_fields(fields)
    tags = [rows[i]["tags"][0] for i in range(8)]
    actual_digest = training_payload_digest(keys=[m.key for m in members], partition_id=state["spec"]["partition"],
                                             fields=fields, tags=tags)
    await tq.async_kv_put(key=state["uid"], partition_id=state["spec"]["partition"], tag={"status": "finished"})
    listed = await tq.async_kv_list(partition_id=state["spec"]["partition"])
    if listed[state["spec"]["partition"]][state["uid"]].get("status") != "finished":
        raise ValueError("Native full-group readiness was not read back")
    state["receipt"] = {"state": "FINISHED", "readback_verified": True, "prompt_ready_verified": True,
                        "verified_members": 8, "completed_members_reused": COMPLETED,
                        "rerun_members": CHECKPOINTED, "first_run_members": PENDING,
                        "actual_content_digest": actual_digest, "snapshot_runtime_bound": False}


async def _durable_finish(worker, state, client):
    from verl.trainer.ppo.v1.trajectory_entry import assemble_prepared_fields, validate_trajectory_fields
    from verl.trainer.ppo.v1.trajectory_publication import PublicationMember, publish_trajectory_group
    entry = _durable_entry(worker, state, client)
    started = time.perf_counter()
    result = await entry.runtime.run_target(state["spec"]["handoff_id"], entry._resume_member,
                                            entry._run_pending_member, token_only=False)
    restore_s = time.perf_counter() - started
    expected = {int(i): value for i, value in state["spec"]["completed_digests"].items()}
    if {i: _row_digest(result.outputs[i].fields) for i in COMPLETED} != expected:
        raise ValueError("Durable target changed an already completed source member or its reward")
    fields = assemble_prepared_fields([o.fields for o in result.outputs])
    tags = [tag for o in result.outputs for tag in o.tags]
    members = [PublicationMember(state["uid"], i) for i in range(8)]
    publication_started = time.perf_counter()
    receipt = await publish_trajectory_group(coordinator=entry.runtime.client, handoff_id=state["spec"]["handoff_id"],
        members=members, fields=fields, tags=tags, partition_id=state["spec"]["partition"],
        validate_fields=validate_trajectory_fields, collect_garbage=False)
    if receipt.get("state") != "FINISHED" or not receipt.get("readback_verified") or not receipt.get("prompt_ready_verified"):
        raise ValueError("Original durable publication did not verify a complete n=8 group")
    receipt.update(restore_and_member_completion_s=restore_s,
                   native_group_publication_s=time.perf_counter() - publication_started, snapshot_runtime_bound=True,
                   completed_members_reused=COMPLETED, checkpointed_members_restored=CHECKPOINTED,
                   first_run_members=PENDING)
    state["receipt"] = receipt


async def worker_rpc(worker, operation, spec=None, client=None):
    """Small RPC added to the existing owned CPU worker; no new job is created."""
    if operation in {"start", "resume"}:
        if spec is None or getattr(worker, "_storage_campaign", None) is not None:
            raise ValueError("Worker already has an experiment or has no frozen specification")
        if worker.background_tasks or worker.trajectory_entry is None or worker.trajectory_entry.runtime._groups:
            raise ValueError("Primary worker must be quiescent before the optional experiment")
        if spec["arm"] not in ARMS or (spec["arm"] == "durable") != (client is not None):
            raise ValueError("Only the durable arm may bind the native storage client")
        owner = "storage-" + spec["arm"] + "-" + uuid4().hex
        state = {"spec": spec, "owner": owner, "uid": spec["uid"], "members": spec["members"],
                 "completed": {}, "ready": set(), "hold": asyncio.Event(), "entry": None}
        worker._storage_campaign = state
        from omegaconf import OmegaConf
        event(spec["root"], owner, "arm_worker_started", arm=spec["arm"], operation=operation,
              primary_runtime_bypassed=True, experiment_runtime_bound=client is not None,
              policy_identity=spec["policy_identity"], business_row_sha256=spec["business_row_sha256"],
              actual_rollout_config=OmegaConf.to_container(worker.config.actor_rollout_ref.rollout, resolve=True),
              source_calls=("factory → loop.run → Durable.run upstream → ToolAgentLoop.run → FullyAsync.generate → native transport"
                            if spec["arm"] == "native_partial" else
                            "runtime.run_source → entry._run_source_member → loop.run_until_checkpoint → native store"),
              target_calls=("read completed TQ rows → loop.run only for unfinished members → original prepare → TQ full readback"
                            if spec["arm"] == "native_partial" else
                            "runtime.run_target → entry._resume_member/_run_pending_member → native publish_trajectory_group"))
        async def execute():
            if operation == "resume":
                return await (_durable_finish(worker, state, client) if client else _native_finish(worker, state))
            import transfer_queue as tq
            await tq.async_kv_put(key=state["uid"], partition_id=spec["partition"], tag={"status": "running"})
            if client is not None:
                return await _durable_source(worker, state, client)
            await asyncio.gather(*[_baseline_member(worker, state, i, "source", hold=i in CHECKPOINTED)
                                   for i in COMPLETED + CHECKPOINTED])
        state["task"] = asyncio.create_task(execute())
        return {"owner": owner, "snapshot_runtime_bound": client is not None}
    primary_configured = worker.trajectory_entry is not None
    state = getattr(worker, "_storage_campaign", None)
    if state is None:
        return {"status": "absent", "cleanup_verified": True, "primary_runtime_configured": primary_configured}
    if operation == "cancel":
        if state["task"].done():
            # A failed finished task is recorded by status; cleanup does not rerun it.
            worker._storage_campaign = None
            return {"status": "settled", "cleanup_verified": True, "primary_runtime_configured": primary_configured}
        state["task"].cancel()
        try:
            await asyncio.wait_for(asyncio.gather(state["task"], return_exceptions=True), 20)
        except asyncio.TimeoutError:
            return {"status": "cleanup_pending", "cleanup_verified": False, "primary_runtime_configured": primary_configured}
        worker._storage_campaign = None
        return {"status": "cancelled", "cleanup_verified": True, "primary_runtime_configured": primary_configured}
    if operation == "preempt":
        if state["spec"]["arm"] != "durable":
            raise ValueError("Native arm cannot use durable preemption")
        await state["entry"].runtime.request_preemption(state["uid"])
        state["entry"].release_acceptance_boundaries(state["uid"])
    elif operation != "status":
        raise ValueError("Unknown campaign worker operation")
    task = state["task"]
    if task.done():
        task.result()  # Preserve the actual source/target exception.
    row = {"status": "finished" if task.done() else "running", "owner": state["owner"],
           "completed": state["completed"], "ready": sorted(state["ready"]),
           "handoff_id": state.get("handoff_id"), "receipt": state.get("receipt")}
    entry = state["entry"]
    if entry is not None and state["uid"] in entry.runtime._groups:
        group = entry.runtime._groups[state["uid"]]
        row["native_runtime_status"] = entry.runtime._status(group)
        row["ready"] = sorted(entry._acceptance_boundaries[state["uid"]]["ready"])
    return row


def summarize_arm(root, arm, records=()):
    """Separate exact repeated requests/tool calls from merely similar output."""
    rows = [json.loads(line) for p in Path(root).glob("events-*.jsonl") for line in p.read_text().splitlines() if line]
    attempts = [r for r in rows if r["event"] == "native_attempt_returned"]
    dispatched = [r for r in rows if r["event"] == "native_attempt_dispatched"]
    source = [r for r in attempts if r["stage"] == "source"]
    target = [r for r in attempts if r["stage"].startswith("target")]
    keys = {(r["member_index"], r["input_token_ids_sha256"], r["sampling_params_sha256"], r["actual_weight_version"])
            for r in source}
    repeated = sum((r["member_index"], r["input_token_ids_sha256"], r["sampling_params_sha256"], r["actual_weight_version"])
                   in keys for r in target)
    tools = [r for r in rows if r["event"] == "tool_returned"]
    committed = {(r["member_index"], r["turn_index"], r["name"], r["arguments_sha256"])
                 for r in tools if r["stage"] == "source" and r["member_index"] in CHECKPOINTED}
    target_tools = [r for r in tools if r["stage"].startswith("target")]
    replay = sum((r["member_index"], r["turn_index"], r["name"], r["arguments_sha256"]) in committed for r in target_tools)
    if any(r["turn_index"] is None for r in tools):
        replay = None
    cached = [r["uncached_input_tokens"] for r in target]
    proofs = [r for r in rows if r["event"] == "target_history_verified"]
    if arm == "durable" and {r["member_index"] for r in proofs} != set(CHECKPOINTED):
        raise ValueError("Every real checkpoint needs an immutable target history proof")
    aborted = [r for r in attempts if r["stop_reason"] in ("abort", "aborted")]
    calls = [r for r in rows if r["event"] == "native_partial_call_verified"]
    state_calls = [r for r in rows if r["event"] == "native_state_rpc_returned"]
    routed = {}
    for r in attempts:
        routed.setdefault(r["logical_call_id"], []).append({"request_id": r["request_id"],
                                                          "worker_id": r.get("gateway_worker_id")})
    return {"native_transport_attempts": len(attempts), "actual_aborted_attempts": len(aborted),
            "native_attempts_dispatched_without_return": len({r["attempt_id"] for r in dispatched}
                                                            - {r["attempt_id"] for r in attempts}),
            "native_partial_retry_observation": "No engine abort occurred" if not aborted else "See actual attempt files",
            "actual_partial_resume_verified": any(r["actual_partial_resume_verified"] for r in calls),
            "native_partial_rollout_internal_retry_count": sum(r["internal_retry_count"] for r in calls),
            "source_generated_tokens": sum(r["generated_tokens"] for r in source),
            "target_generated_tokens": sum(r["generated_tokens"] for r in target),
            "total_generated_tokens": sum(r["generated_tokens"] for r in attempts),
            "repeated_complete_model_requests": repeated, "repeated_completed_tool_calls": replay,
            "source_completed_tool_calls": sum(r["stage"] == "source" for r in tools),
            "target_completed_tool_calls": len(target_tools),
            "uncached_target_input_tokens": sum(cached) if cached and all(v is not None for v in cached) else None,
            "native_generation_s_sum": sum(r["elapsed_s"] for r in attempts),
            "tool_execution_s_sum": sum(r["elapsed_s"] for r in tools),
            "snapshot_bytes": sum(record["snapshot_bytes"] for record, _ in records),
            "snapshot_commit_s_sum": sum(record["snapshot_commit_s"] for record, _ in records),
            "native_snapshot_load_s_sum": sum(r["elapsed_s"] for r in state_calls if r["method"] == "load_trajectory_snapshot")
                                          if arm == "durable" else None,
            "native_state_rpc_s_sum": sum(r["elapsed_s"] for r in state_calls),
            "native_member_record_get_bytes": sum(r["returned_payload_bytes"] for r in state_calls
                                                   if r["method"] == "get_handoff_member_record"),
            "restore_and_continuation_s_sum": sum(r["restore_and_continuation_s"] for r in proofs) if proofs else None,
            "history_proofs": proofs, "actual_routed_workers_by_native_call": routed,
            "session_affinity_verified": None, "gpu_prefill_time_s": None,
            "evidence_io_s_sum": sum(r["evidence_io_s"] for r in attempts),
            "discarded_source_generated_tokens": sum(r["generated_tokens"] for r in source if r["member_index"] in CHECKPOINTED)
                                                   if arm == "native_partial" else 0}


async def _replace_cpu_worker(manager, index):
    from verl.trainer.ppo.v1.trajectory_entry import state_settings
    replacement = manager.agent_loop_workers_class.options(name="agent_loop_worker_" + str(index) + "_" + uuid4().hex[:8]).remote(
        manager.config, manager.llm_client, manager.teacher_client, manager.reward_loop_worker_handles)
    settings = state_settings(manager.config)
    owner = settings.run_id + "/worker-" + str(index) + "-" + uuid4().hex
    # Keep the actual handle as soon as constructed, even if configure is uncertain.
    manager.agent_loop_workers[index] = replacement
    await replacement.configure_trajectory_state.remote(manager.trajectory_coordinator_client, owner, settings)
    await replacement.set_trajectory_checkpoint_paused.remote(True)
    manager.trajectory_worker_owner_ids[owner] = index
    return replacement


async def _exit_cpu(source, root, deadline):
    import ray
    from ray.util.state import get_actor
    actor_id = source._actor_id.hex()
    actual = await asyncio.to_thread(get_actor, actor_id)
    resources = actual.get("required_resources") if isinstance(actual, dict) else getattr(actual, "required_resources", None)
    if not isinstance(resources, dict) or resources.get("GPU", 0) != 0:
        raise ValueError("Fault target is not a verified task-owned CPU agent worker")
    event(root, "driver", "actual_cpu_exit_dispatched", actor_id=actor_id, required_resources=resources,
          gpu_replica_exit_verified=False, physical_gpu_release_verified=False)
    ray.kill(source, no_restart=True)
    async def dead():
        row = await asyncio.to_thread(get_actor, actor_id)
        state = row.get("state") if isinstance(row, dict) else getattr(row, "state", None)
        return {"actor_id": actor_id, "state": state, "required_resources": resources} if state == "DEAD" else None
    return await wait_for(dead, deadline, "Actual CPU actor exit is not observable")


async def run_campaign(manager, frozen, root):
    """Pause native admission, run the finite arms, restore CPU worker admission."""
    import ray
    from verl.trainer.ppo.v1.trajectory_state import create_ray_trajectory_coordinator_actor, RayTrajectoryCoordinatorClient
    from verl.trainer.ppo.v1.trajectory_state.policy_publication import PublishedPolicyRecord
    from verl.trainer.ppo.v1.trajectory_state.snapshot import TrajectorySnapshot, TrajectoryOwnerLease
    from verl.workers.rollout.llm_server import observe_owned_engine_versions
    spec = frozen["spec"]
    si, ti = spec["source_worker_index"], spec["target_worker_index"]
    if any(i >= len(manager.agent_loop_workers) or i in manager._draining_worker_indices for i in (si, ti)):
        raise ValueError("Need two available original task-owned CPU workers")
    deadline = time.monotonic() + spec["total_timeout_s"]
    campaign_id = "storage-" + uuid4().hex
    result = {"status": "PENDING", "campaign_id": campaign_id, "arms": [], "optimizer_updates_added": 0,
              "gpu_replica_exit_verified": False, "physical_gpu_release_verified": False,
              "hybrid_switching_verified": False, "business_benefit_accepted": False,
              "gpu_lifecycle_gap": "Current SGLangHttpServer exposes no owned child-process shutdown/relaunch receipt; KV sleep is not process exit"}
    coordinator = None
    paused = False
    pause_attempted = False
    workers = []
    try:
        if getattr(manager, "_trajectory_checkpoint_paused", False):
            raise ValueError("The primary owner already paused admission; campaign cannot take over that control")
        pause_attempted = True
        await asyncio.wait_for(manager.prepare_trajectory_checkpoint(timeout_s=min(60, spec["total_timeout_s"])),
                               timeout=min(65, spec["total_timeout_s"]))
        paused = True
        record = PublishedPolicyRecord.decode(await manager.trajectory_coordinator_client.get_active_published_policy())
        if record.encode() != frozen["verified_publication"]["active_policy_record"]:
            raise ValueError("Published policy changed after the audited freeze")
        urls = [r.engine_url for r in record.engine_observations]
        async def check_weights():
            observations = await observe_owned_engine_versions(urls, timeout_s=10)
            if any(r["weight_version"] != record.identity.policy_version for r in observations):
                raise ValueError("Actual engine weights changed within the campaign")
            return observations
        result["actual_engine_observations_before"] = await check_weights()
        for group in frozen["groups"]:
            for arm in spec["arm_order"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Finite campaign total budget exhausted")
                arm_started = time.perf_counter()
                arm_deadline = min(deadline, time.monotonic() + spec["arm_timeout_s"])
                uid = campaign_id + "-" + str(group["dataset_index"]) + "-" + arm
                arm_root = Path(root) / uid
                arm_spec = {"arm": arm, "uid": uid, "root": str(arm_root), "partition": uid,
                            "policy_identity": asdict(record.identity), "business_row_sha256": group["business_row_sha256"],
                            "members": _members(group, uid, spec["at_step"], frozen["sampling_params"])}
                write_private(arm_root / "actual-arm-config.json", json.dumps({**arm_spec,
                    "primary_trajectory_state_enabled": True, "experiment_snapshot_runtime_bound": arm == "durable",
                    "original_completed_members_preserved": COMPLETED,
                    "checkpoint_members": CHECKPOINTED, "never_started_members": PENDING,
                    "optimizer_consumes_experiment_partition": False}, ensure_ascii=False, allow_nan=False, indent=2).encode())
                client = None
                if arm == "durable":
                    # Independent native metadata imports only the already verified real publication.
                    # These coordinator calls do NOT transfer weights or invoke an optimizer.
                    coordinator = create_ray_trajectory_coordinator_actor().remote(str(Path(root) / "native-state"), uid)
                    client = RayTrajectoryCoordinatorClient(coordinator)
                    ticket = await client.begin_policy_publication(asdict(record.identity), record.model_configuration_fingerprint,
                        "import-existing-audited-publication", urls)
                    await client.mark_policy_update_started(ticket)
                    imported = await client.confirm_policy_publication(ticket, await check_weights())
                    event(arm_root, "driver", "isolated_policy_proof_imported", actual_primary_record=record.encode(),
                          imported_record=imported, weight_transfers_added=0, optimizer_updates_added=0)
                source, target = manager.agent_loop_workers[si], manager.agent_loop_workers[ti]
                workers = [source, target]
                source_id, target_id = source._actor_id.hex(), target._actor_id.hex()
                await source.storage_handoff_campaign.remote("start", arm_spec, client)
                async def ready():
                    row = await source.storage_handoff_campaign.remote("status")
                    if row["status"] == "finished" and row["ready"] != CHECKPOINTED:
                        raise ValueError("This real business group finished without four legal tool boundaries; no saved-work claim")
                    if arm == "native_partial":
                        return row if sorted(map(int, row["completed"])) == COMPLETED and row["ready"] == CHECKPOINTED else None
                    native = row.get("native_runtime_status", {})
                    return row if native.get("completed_members") == COMPLETED and native.get("pending_members") == PENDING \
                        and row["ready"] == CHECKPOINTED else None
                boundary = await wait_for(ready, arm_deadline, "Actual source did not reach the legal mixed tool boundary")
                records = []
                if arm == "durable":
                    await source.storage_handoff_campaign.remote("preempt")
                    async def committed():
                        row = await source.storage_handoff_campaign.remote("status")
                        return row if row["status"] == "finished" and row["handoff_id"] else None
                    handoff = await wait_for(committed, arm_deadline, "Native durable handoff did not commit")
                    manifest = await client.get_handoff_manifest(handoff["handoff_id"])
                    manifest.validate()
                    table = manifest.member_tables()[uid]
                    if table != {"group_size": 8, "completed": COMPLETED, "checkpointed": CHECKPOINTED, "pending": PENDING}:
                        raise ValueError("Actual native manifest changed the fixed mixed membership")
                    if not (await client.get_handoff_status(handoff["handoff_id"]))["source_released"]:
                        raise ValueError("Actual source rights were not released")
                    arm_spec["handoff_id"] = handoff["handoff_id"]
                    for member in manifest.checkpointed_trajectories:
                        filename = "snapshot-" + digest([member["trajectory_id"], member["expected_generation"]]) + ".json"
                        observed = json.loads((arm_root / (filename + ".source")).read_text())
                        payload = (arm_root / filename).read_bytes()
                        if observed["snapshot_ref"] != member["snapshot_ref"] or hashlib.sha256(payload).hexdigest() != observed["snapshot_sha256"]:
                            raise ValueError("Source snapshot evidence differs from the actual native manifest")
                        records.append((observed, TrajectorySnapshot.decode(payload)))
                    from verl.trainer.ppo.v1.trajectory_state import decode_handoff_member_record
                    completed_digests = {}
                    for member in manifest.completed_members:
                        payload = await client.get_handoff_member_record(member["record_ref"])
                        actual_output = decode_handoff_member_record(payload)["output"]
                        completed_digests[member["member_index"]] = _row_digest(actual_output.fields)
                    arm_spec["completed_digests"] = completed_digests
                    storage_before = _storage_bytes(Path(root) / "native-state" / uid)
                else:
                    arm_spec["completed_digests"] = boundary["completed"]
                exited = await _exit_cpu(source, arm_root, arm_deadline)
                # Replace only the verified owned CPU worker. No GPU resources are reclaimed.
                replacement = await _replace_cpu_worker(manager, si)
                workers = [replacement, target]
                rejected = []
                for observed, snapshot in records:
                    try:
                        await client.commit_trajectory_snapshot(TrajectoryOwnerLease(**observed["owner_lease"]), snapshot, 30)
                    except Exception as error:
                        cause = getattr(error, "cause", error)
                        if not isinstance(cause, ValueError) or str(cause) != "stale or expired trajectory owner":
                            raise RuntimeError("Native stale writer rejection was not proved") from error
                        rejected.append(snapshot.member_index)
                    else:
                        raise ValueError("Native coordinator accepted an exited source write")
                takeover_started = time.perf_counter()
                await target.storage_handoff_campaign.remote("resume", arm_spec, client)
                async def finished():
                    row = await target.storage_handoff_campaign.remote("status")
                    return row if row["status"] == "finished" and row["receipt"] else None
                completed = await wait_for(finished, arm_deadline, "Surviving worker did not verify a complete n=8 group")
                arm_result = {"arm": arm, "status": "ACTUAL_CPU_EXIT_AND_COMPLETE_GROUP_OBSERVED",
                    "business_row_sha256": group["business_row_sha256"], "policy_identity": asdict(record.identity),
                    "source_actor_id": source_id, "target_actor_id": target_id,
                    "replacement_actor_id": replacement._actor_id.hex(), "actual_cpu_exit": exited,
                    "source_owner": boundary["owner"], "target_owner": completed["owner"],
                    "completed_members_reused": COMPLETED, "source_tool_boundaries": CHECKPOINTED,
                    "pending_members": PENDING, "old_writes_rejected": rejected if arm == "durable" else None,
                    "publication": completed["receipt"], "observed_work": summarize_arm(arm_root, arm, records),
                    "takeover_and_publication_s": time.perf_counter() - takeover_started,
                    "arm_total_wall_s": time.perf_counter() - arm_started,
                    "engine_observations_after": await check_weights(), "gpu_release_verified": False}
                if arm == "durable":
                    arm_result["actual_native_storage_before_recovery"] = storage_before
                    arm_result["actual_native_storage_after_publication"] = _storage_bytes(Path(root) / "native-state" / uid)
                write_private(arm_root / "actual-arm-result.json", json.dumps(arm_result, allow_nan=False, indent=2).encode())
                result["arms"].append(arm_result)
                cleanup = await target.storage_handoff_campaign.remote("cancel")
                if not cleanup["cleanup_verified"]:
                    raise RuntimeError("Completed target task did not settle")
                if coordinator is not None:
                    await coordinator.close.remote()
                    ray.kill(coordinator, no_restart=True)
                    coordinator = None
        result["status"] = "FINITE_NATIVE_PARTIAL_AND_DURABLE_WORKER_CAMPAIGN_OBSERVED"
        result["actual_engine_observations_after"] = await check_weights()
    except (Exception, asyncio.CancelledError) as error:
        result.update(status="FAILED_OR_PENDING", error_type=type(error).__name__, error=str(error),
                      primary_optimizer_and_publication_result_overwritten=False, automatic_fault_retry_allowed=False)
    finally:
        # Replacement handles are retained in the manager before configure awaits.
        # This avoids addressing the dead old source when construction was uncertain.
        workers = [manager.agent_loop_workers[i] for i in (si, ti)]
        cleanup = []
        for worker in workers:
            try:
                row = await asyncio.wait_for(worker.storage_handoff_campaign.remote("cancel"), 25)
                cleanup.append({"actor_id": worker._actor_id.hex(), **row})
            except Exception as error:
                cleanup.append({"actor_id": worker._actor_id.hex(), "cleanup_verified": False, "error_type": type(error).__name__})
        result["worker_cleanup"] = cleanup
        # Settling Python tasks alone does not prove engine-side request abort.
        result["gpu_request_cancellation_verified"] = False
        unmatched = []
        for directory in Path(root).iterdir() if Path(root).exists() else []:
            if not directory.is_dir() or directory.name == "native-state":
                continue
            observations = [json.loads(line) for p in directory.glob("events-*.jsonl")
                            for line in p.read_text().splitlines() if line]
            dispatched = {r["attempt_id"] for r in observations if r["event"] == "native_attempt_dispatched"}
            returned = {r["attempt_id"] for r in observations if r["event"] == "native_attempt_returned"}
            unmatched.extend(sorted(dispatched - returned))
        result["native_attempts_requiring_readback"] = unmatched
        if coordinator is not None:
            # Only our isolated CPU coordinator is addressed. Native object files are retained.
            try:
                await asyncio.wait_for(coordinator.close.remote(), 10)
                ray.kill(coordinator, no_restart=True)
            except Exception as error:
                result["coordinator_cleanup_error_type"] = type(error).__name__
        result["primary_admission_restored"] = False
        # prepare_trajectory_checkpoint sets the pause before its first await.
        # On a timeout it may still be ours even though prepare did not return.
        paused = paused or (pause_attempted and getattr(manager, "_trajectory_checkpoint_paused", False))
        if paused and not unmatched and all(row["cleanup_verified"] and row.get("primary_runtime_configured") for row in cleanup):
            try:
                await asyncio.wait_for(manager.resume_trajectory_checkpoint_admission(), 15)
                result["primary_admission_restored"] = True
            except Exception as error:
                result["admission_restore_error_type"] = type(error).__name__
        result["campaign_total_budget_s"] = spec["total_timeout_s"]
        if not result["primary_admission_restored"]:
            result["status"] = "FAILED_OR_PENDING"
            result["cleanup_requires_readback"] = True
        write_private(Path(root) / "campaign-result.json", json.dumps(result, allow_nan=False, indent=2).encode())
    return result


def _storage_bytes(directory):
    files = [p for p in Path(directory).rglob("*") if p.is_file()]
    return {"files": len(files), "actual_file_bytes": sum(p.stat().st_size for p in files),
            "scope": "This isolated native coordinator metadata and object directory; no KV/GDN export"}


def maybe_run_campaign_after_publication(trainer, proof):
    """Default off; called by the existing verified post-publication owner hook."""
    path = os.environ.get("VERL_STORAGE_HANDOFF_CAMPAIGN_SPEC")
    if not path:
        return None
    root = Path(os.environ["VERL_STORAGE_HANDOFF_EVIDENCE_ROOT"])
    try:
        raw = json.loads(Path(path).read_text())
        if raw.get("enabled") is not True:
            return None
        spec = normalize_spec(raw)
        if trainer.global_steps != spec["at_step"]:
            return None
        if getattr(trainer, "_storage_handoff_attempted", False):
            return None  # An uncertain dispatch never authorizes another fault.
        from native_training_audit import verify_publication
        verify_publication(proof, spec["at_step"],
                           trainer.config.actor_rollout_ref.rollout.trajectory_state.model_configuration_fingerprint)
        if spec["at_step"] not in trainer.audit_verified_steps or spec["at_step"] not in trainer.audit_publication_steps:
            raise ValueError("Original optimizer/publication audit did not confirm this step")
        trainer._storage_handoff_attempted = True
        frozen = freeze_inputs(trainer, spec, proof, root)
        from verl.trainer.ppo.v1.trajectory_control_dispatch import _dispatcher
        async def bounded():
            return await asyncio.wait_for(run_campaign(trainer.agent_loop_manager, frozen, root),
                                          spec["total_timeout_s"])
        future = _dispatcher(trainer.agent_loop_manager).submit(bounded())
        trainer._storage_handoff_future = future
        result = future.result(timeout=spec["total_timeout_s"] + 90)
        trainer._storage_handoff_campaign_result = result
        return result
    except Exception as error:
        # The optional experiment cannot relabel an already audited optimizer update.
        # Outstanding remote work/admission needs readback; never retry automatically.
        event(root, "driver", "optional_campaign_failed", error_type=type(error).__name__,
              trainer_step=trainer.global_steps, primary_training_result_overwritten=False,
              automatic_fault_retry_allowed=False, remote_outcome_requires_readback=True)
        return {"status": "FAILED_OR_PENDING", "error_type": type(error).__name__,
                "primary_training_result_overwritten": False}
