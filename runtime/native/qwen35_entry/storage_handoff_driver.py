"""Opt-in CPU executor exit and durable takeover on the original V1 interfaces.

Install the companion runtime hooks in the owner's frozen source. Enable only
after the model can update/publish: VERL_STORAGE_HANDOFF_EVIDENCE_ROOT is a shared
private evidence directory; VERL_STORAGE_HANDOFF_AT_STEP is a positive update
step. SOURCE_INDEX and TARGET_INDEX default to 0 and 1. No GPU process is stopped,
no shared-GPU switching is enabled, and no external service is called directly.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

_MEMBER = ContextVar("storage_handoff_member", default=None)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_private(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def event(root, owner, event_name, **fields):
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(directory / ("events-" + digest(owner) + ".jsonl"),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a") as stream:
        stream.write(json.dumps({"event": event_name, "owner": owner, "wall_time_unix": time.time(),
            **fields}, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def preservation(snapshot, output, generation):
    """Exact committed history is immutable even when new tokens differ."""
    state = snapshot.agent_state
    history = list(output.prompt_ids) + list(output.response_ids)
    count = len(state["response_mask"])
    tools = output.extra_fields.get("completed_tool_batches", [])
    checks = {
        "token_prefix_unchanged": history[:len(state["prompt_ids"])] == state["prompt_ids"],
        "mask_prefix_unchanged": list(output.response_mask)[:count] == state["response_mask"],
        "behavior_logprob_prefix_unchanged": list(output.response_logprobs)[:count] == state["response_logprobs"],
        "tool_results_prefix_unchanged": tools[:len(snapshot.completed_tool_batches)] == snapshot.completed_tool_batches,
        "sampling_segments_prefix_unchanged": output.extra_fields.get("trajectory_segments", [])[
            :len(state["extra_fields"].get("trajectory_segments", []))] == state["extra_fields"].get("trajectory_segments", []),
        "execution_generation_advanced": type(generation) is int and generation > snapshot.execution_generation,
    }
    if not all(checks.values()):
        raise ValueError("Restored output changed committed source history: " + str(checks))
    return {**checks, "trajectory_id": snapshot.trajectory_id, "member_index": snapshot.member_index,
        "source_generation": snapshot.execution_generation, "target_generation": generation,
        "preserved_history_tokens": len(state["prompt_ids"]),
        "preserved_generated_tokens": sum(state["response_mask"]),
        "preserved_tool_batches": len(snapshot.completed_tool_batches),
        "history_sha256": digest({k: state[k] for k in ("prompt_ids", "response_mask", "response_logprobs")}),
        "tool_batches_sha256": digest(snapshot.completed_tool_batches),
        "source_policy_identity": snapshot.policy_identity,
        "target_segments": output.extra_fields.get("trajectory_segments"),
        "resume_mode": output.extra_fields.get("trajectory_resume_mode")}


class MeasuredServer:
    """Observe actual native results; keep its request and return value intact."""
    def __init__(self, server, root, owner, member):
        self.server, self.root, self.owner, self.member = server, root, owner, dict(member)

    def __getattr__(self, name):
        return getattr(self.server, name)

    async def generate(self, *args, **kwargs):
        started = time.perf_counter()
        output = await self.server.generate(*args, **kwargs)
        native = (output.extra_fields or {}).get("native_meta_info") or {}
        prompt = kwargs.get("prompt_ids")
        input_tokens = native.get("prompt_tokens")
        if type(input_tokens) is not int:
            input_tokens = len(prompt) if isinstance(prompt, list) else None
        cached = native.get("cached_tokens")
        if type(cached) is not int or input_tokens is None or not 0 <= cached <= input_tokens:
            cached = None
        event(self.root, self.owner, "native_generation_returned", **self.member,
            elapsed_s=time.perf_counter() - started, input_tokens=input_tokens, cached_tokens=cached,
            uncached_input_tokens=input_tokens - cached if cached is not None else None,
            generated_tokens=len(output.token_ids), generated_tokens_sha256=digest(output.token_ids),
            input_token_ids_sha256=digest(prompt) if isinstance(prompt, list) else None,
            sampling_params_sha256=digest(kwargs.get("sampling_params")),
            request_id=kwargs.get("request_id"), actual_weight_version=native.get("weight_version"),
            native_weight_segments=native.get("weight_versions"),
            gpu_prefill_time_s=None)
        return output


def install_worker_observer(entry, root):
    """Called once just after native configure_trajectory_state, before admission."""
    if getattr(entry, "_storage_handoff_observed", False):
        raise ValueError("Storage handoff observer is already attached")
    entry._storage_handoff_observed = True
    owner = entry.runtime.owner_id
    source_method, resume_method = entry._run_source_member, entry._resume_member
    pending_method, new_loop_method = entry._run_pending_member, entry._new_loop

    async def source(index, original, lease, commit_snapshot, check_owner):
        member = {"uid": original["uid"], "trajectory_id": lease.trajectory_id,
            "member_index": index, "generation": lease.generation, "stage": "source"}
        async def measured_commit(snapshot):
            started = time.perf_counter()
            reference = await commit_snapshot(snapshot)
            committed_s = time.perf_counter() - started
            if reference is not None:
                payload = snapshot.encode()
                filename = "snapshot-" + digest([lease.trajectory_id, lease.generation]) + ".json"
                record = {"snapshot_ref": reference, "owner_lease": asdict(lease),
                    "snapshot_file": filename, "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
                    "snapshot_commit_s": committed_s, "snapshot_bytes": len(payload), **member}
                # Immutable evidence is independent of native snapshot GC.
                write_private(Path(root) / filename, payload)
                write_private(Path(root) / (filename + ".source"), json.dumps(record, allow_nan=False).encode())
                event(root, owner, "source_snapshot_committed", **record)
            return reference
        for name in ("resolve_published_policy", "get_snapshot_policy_identity"):
            if hasattr(commit_snapshot, name):
                setattr(measured_commit, name, getattr(commit_snapshot, name))
        token = _MEMBER.set(member)
        try:
            return await source_method(index, original, lease, measured_commit, check_owner)
        finally:
            _MEMBER.reset(token)

    async def resume(index, snapshot, original, lease, check_owner):
        token = _MEMBER.set({"uid": original["uid"], "trajectory_id": lease.trajectory_id,
            "member_index": index, "generation": lease.generation, "stage": "target"})
        event(root, owner, "target_acquired_snapshot", trajectory_id=lease.trajectory_id,
            member_index=index, generation=lease.generation, source_generation=snapshot.execution_generation,
            source_snapshot_sha256=hashlib.sha256(snapshot.encode()).hexdigest())
        try:
            return await resume_method(index, snapshot, original, lease, check_owner)
        finally:
            _MEMBER.reset(token)

    async def pending(index, original, lease, check_owner):
        token = _MEMBER.set({"uid": original["uid"], "trajectory_id": lease.trajectory_id,
            "member_index": index, "generation": lease.generation, "stage": "target_pending"})
        try:
            return await pending_method(index, original, lease, check_owner)
        finally:
            _MEMBER.reset(token)

    def new_loop(original):
        loop = new_loop_method(original)
        member = _MEMBER.get()
        if member is None:
            raise ValueError("Observed native loop has no execution member")
        loop.server_manager = MeasuredServer(loop.server_manager, root, owner, member)
        for name, tool in loop.tools.items():
            if getattr(tool, "_storage_handoff_measured", False):
                raise ValueError("Tool observations require the business factory's trajectory-local tool instances")
            tool._storage_handoff_measured = True
            execute = tool.execute
            async def measured_tool(*args, _execute=execute, _name=name, **kwargs):
                started = time.perf_counter()
                result = await _execute(*args, **kwargs)
                parameters = kwargs.get("parameters", args[1] if len(args) > 1 else None)
                data = kwargs.get("agent_data")
                turn = getattr(data, "assistant_turns", None)
                event(root, owner, "tool_returned", **member, name=_name,
                    turn_index=turn - 1 if type(turn) is int else None,
                    arguments_sha256=digest(parameters), result_text_sha256=digest(getattr(result[0], "text", None)),
                    elapsed_s=time.perf_counter() - started)
                return result
            tool.execute = measured_tool
        native_resume = loop.resume_from_snapshot
        async def measured_resume(snapshot, **kwargs):
            started = time.perf_counter()
            output = await native_resume(snapshot, **kwargs)
            proof = preservation(snapshot, output, kwargs["execution_generation"])
            event(root, owner, "target_history_verified", **proof, restore_and_continuation_s=time.perf_counter() - started)
            return output
        loop.resume_from_snapshot = measured_resume
        return loop

    entry._run_source_member, entry._resume_member = source, resume
    entry._run_pending_member, entry._new_loop = pending, new_loop
    event(root, owner, "observer_attached", fault_scope="cpu_agent_worker", gpu_release_verified=False)


async def wait_for(predicate, deadline, description):
    while time.monotonic() < deadline:
        value = await predicate()
        if value:
            return value
        await asyncio.sleep(0.05)
    raise TimeoutError(description)


def summarize_observed_work(root, records):
    """Read this run's observations; missing native cache/retry fields stay null."""
    snapshots = {snapshot.trajectory_id: snapshot for _, snapshot in records}
    rows = []
    for filename in Path(root).glob("events-*.jsonl"):
        rows.extend(json.loads(line) for line in filename.read_text().splitlines() if line)
    rows = [row for row in rows if row.get("trajectory_id") in snapshots]
    proofs = [row for row in rows if row["event"] == "target_history_verified"]
    if {row["trajectory_id"] for row in proofs} != set(snapshots):
        raise ValueError("Every committed source snapshot needs an observed target history proof")
    generations = [row for row in rows if row["event"] == "native_generation_returned"]
    target = [row for row in generations if row["stage"] == "target"]
    source_requests = {(row["trajectory_id"], row.get("input_token_ids_sha256"),
        row.get("sampling_params_sha256"), row.get("actual_weight_version"))
        for row in generations if row["stage"] == "source" and row.get("input_token_ids_sha256") is not None}
    replay = sum((row["trajectory_id"], row.get("input_token_ids_sha256"), row.get("sampling_params_sha256"),
        row.get("actual_weight_version")) in source_requests for row in target)
    if any(row.get("input_token_ids_sha256") is None or row.get("actual_weight_version") is None for row in generations):
        replay = None
    committed_tools = set()
    for trajectory, snapshot in snapshots.items():
        for batch in snapshot.completed_tool_batches:
            for call in batch["calls"]:
                committed_tools.add((trajectory, batch["turn_index"], call["name"], digest(json.loads(call["arguments"]))))
    target_tools = [row for row in rows if row["event"] == "tool_returned" and row["stage"] == "target"]
    tool_replay = sum((row["trajectory_id"], row["turn_index"], row["name"], row["arguments_sha256"])
        in committed_tools for row in target_tools)
    if any(row["turn_index"] is None for row in target_tools):
        tool_replay = None
    recomputed = sum(row["uncached_input_tokens"] for row in target) if target and all(
        row["uncached_input_tokens"] is not None for row in target) else None
    return {"preserved_generated_tokens": sum(sum(s.agent_state["response_mask"]) for s in snapshots.values()),
        "preserved_tool_calls": sum(len(b["calls"]) for s in snapshots.values() for b in s.completed_tool_batches),
        "snapshot_bytes": sum(record["snapshot_bytes"] for record, _ in records),
        "snapshot_commit_s_sum": sum(record["snapshot_commit_s"] for record, _ in records),
        "restore_and_continuation_s_sum": sum(row["restore_and_continuation_s"] for row in proofs),
        "target_generated_tokens": sum(row["generated_tokens"] for row in target),
        "completed_model_request_replay_count": replay, "committed_tool_call_replay_count": tool_replay,
        "uncached_target_input_tokens": recomputed, "native_partial_rollout_internal_retry_count": None,
        "gpu_prefill_time_s": None, "history_proofs": proofs,
        "measurement_scope": "Outer native generate calls; native HTTP retries are not observed by this wrapper"}


async def exit_and_resume(manager, root, source_index, target_index, timeout_s):
    """Drain accepted groups, kill the actual CPU Actor, then resume once."""
    import ray
    from ray.util.state import get_actor
    from verl.trainer.ppo.v1.trajectory_entry import state_settings
    from verl.trainer.ppo.v1.trajectory_state.snapshot import TrajectoryOwnerLease, TrajectorySnapshot

    if source_index == target_index or any(type(i) is not int or not 0 <= i < len(manager.agent_loop_workers)
        or i in manager._draining_worker_indices for i in (source_index, target_index)):
        raise ValueError("Choose two distinct available native CPU workers")
    source = manager.agent_loop_workers[source_index]
    source_id = source._actor_id.hex()
    source_state = await asyncio.to_thread(get_actor, source_id)
    resources = source_state.get("required_resources") if isinstance(source_state, dict) else getattr(source_state, "required_resources", None)
    if not isinstance(resources, dict) or resources.get("GPU", 0) != 0:
        raise ValueError("The selected exit target must be a verified CPU Actor")
    deadline = time.monotonic() + timeout_s
    root = Path(root)
    await source.set_trajectory_checkpoint_paused.remote(True)
    async def live_groups():
        row = await source.trajectory_checkpoint_status.remote()
        if row["errors"]:
            raise ValueError("Source already has blocked accepted work")
        return row["group_uids"]
    # Existing groups continue while new admission is paused. Do not fabricate a
    # group or kill an idle worker and call that trajectory recovery.
    groups = await live_groups()
    if not groups:
        await source.set_trajectory_checkpoint_paused.remote(False)
        raise ValueError("No live native prompt group at the requested fault point")
    handoffs = []
    for uid in groups:
        selected = await manager.request_trajectory_preemption(uid)
        if selected is None:
            continue  # Group completed normally; it is not a saved-work claim.
        if selected != source_index:
            raise ValueError("Native prompt ownership differs from the selected source")
        async def ready():
            rows = await manager.list_recoverable_handoffs()
            return next((r for r in rows if r["group_uid"] == uid and r["source_released"]), None)
        handoffs.append(await wait_for(ready, deadline, "Native tool-boundary handoff did not commit/release"))
    if not handoffs:
        raise ValueError("Accepted groups completed without a durable handoff")
    async def drained():
        row = await source.trajectory_checkpoint_status.remote()
        if row["errors"]:
            raise ValueError("Source failed before a committed release")
        return row if not any(row[k] for k in ("background_tasks", "active_groups", "cleanup_pending")) else None
    await wait_for(drained, deadline, "Accepted source work did not settle")

    records, manifests = [], []
    for row in handoffs:
        manifest = await manager.trajectory_coordinator_client.get_handoff_manifest(row["handoff_id"])
        manifest.validate()
        manifests.append(manifest)
        table = manifest.member_tables()[row["group_uid"]]
        if table["group_size"] != 8 or sorted(table["completed"] + table["checkpointed"] + table["pending"]) != list(range(8)):
            raise ValueError("The native n=8 handoff has missing or duplicate members")
        for member in manifest.checkpointed_trajectories:
            filename = "snapshot-" + digest([member["trajectory_id"], member["expected_generation"]]) + ".json"
            record = json.loads((root / (filename + ".source")).read_text())
            payload = (root / filename).read_bytes()
            if record["snapshot_ref"] != member["snapshot_ref"] or hashlib.sha256(payload).hexdigest() != record["snapshot_sha256"]:
                raise ValueError("Observed committed snapshot differs from the native manifest")
            records.append((record, TrajectorySnapshot.decode(payload)))
    if not records:
        raise ValueError("Fault point has no committed tool-result snapshot")

    event(root, "driver", "cpu_worker_exit_dispatched", source_actor_id=source_id,
        handoff_ids=[r["handoff_id"] for r in handoffs], snapshot_count=len(records),
        actual_actor_resources=resources, gpu_release_verified=False)
    ray.kill(source, no_restart=True)
    async def exited():
        state = await asyncio.to_thread(get_actor, source_id)
        actual = state.get("state") if isinstance(state, dict) else getattr(state, "state", None)
        return {"actor_id": source_id, "state": actual} if actual == "DEAD" else None
    exited_state = await wait_for(exited, deadline, "Actual CPU Actor death was not observed")
    event(root, "driver", "cpu_worker_exit_verified", **exited_state, gpu_release_verified=False)

    rejected = []
    for record, snapshot in records:
        lease = TrajectoryOwnerLease(**record["owner_lease"])
        try:
            await manager.trajectory_coordinator.commit_trajectory_snapshot.remote(lease, snapshot, 30.0)
        except Exception as error:
            cause = getattr(error, "cause", error)
            if not isinstance(cause, ValueError) or str(cause) != "stale or expired trajectory owner":
                raise RuntimeError("Old write probe did not prove native owner fencing") from error
            rejected.append({"trajectory_id": snapshot.trajectory_id, "error_type": type(cause).__name__, "message": str(cause)})
        else:
            raise ValueError("Native coordinator accepted a released source write")
    event(root, "driver", "released_source_writes_rejected", probes=rejected)

    # Use the original worker constructor. No GPU actor/service is recreated.
    replacement = manager.agent_loop_workers_class.options(name="agent_loop_worker_" + str(source_index) + "_" + uuid4().hex[:8]).remote(
        manager.config, manager.llm_client, manager.teacher_client, manager.reward_loop_worker_handles)
    settings = state_settings(manager.config)
    owner = settings.run_id + "/worker-" + str(source_index) + "-" + uuid4().hex
    await replacement.configure_trajectory_state.remote(manager.trajectory_coordinator_client, owner, settings)
    manager.agent_loop_workers[source_index] = replacement
    manager.trajectory_worker_owner_ids[owner] = source_index
    receipts = []
    for row in handoffs:
        started = time.perf_counter()
        # One remote resume dispatch. Uncertain outcomes are never retried here.
        receipt = await manager.resume_trajectory_handoff(row["handoff_id"], target_worker_index=target_index)
        if receipt.get("state") != "FINISHED" or not receipt.get("prompt_ready_verified") or not receipt.get("readback_verified"):
            raise ValueError("Native takeover did not expose a complete verified prompt group")
        receipts.append({"handoff_id": row["handoff_id"], "target_worker_index": target_index,
            "takeover_and_publication_s": time.perf_counter() - started, "publication": receipt})
    work = summarize_observed_work(root, records)
    manager._draining_worker_indices.discard(source_index)
    result = {"status": "ACTUAL_CPU_EXIT_AND_NATIVE_TAKEOVER_OBSERVED", "fault_scope": "cpu_agent_worker",
        "source_actor_id": source_id, "actual_cpu_exit": exited_state, "replacement_actor_id": replacement._actor_id.hex(),
        "snapshots": len(records), "old_write_probes_rejected": len(rejected), "complete_groups": len(receipts),
        "receipts": receipts, "observed_work": work, "gpu_replica_exit_verified": False, "physical_gpu_release_verified": False,
        "hybrid_switching_verified": False, "baseline_comparison": None, "business_benefit_accepted": False}
    write_private(root / "cpu-handoff-result.json", json.dumps(result, allow_nan=False, indent=2).encode())
    return result


def maybe_run_after_publication(trainer, proof):
    """Owner hook immediately after verify_publication in audited on_step_end."""
    if os.environ.get("VERL_STORAGE_HANDOFF_CAMPAIGN_SPEC"):
        from storage_handoff_campaign import maybe_run_campaign_after_publication

        return maybe_run_campaign_after_publication(trainer, proof)
    configured = os.environ.get("VERL_STORAGE_HANDOFF_AT_STEP")
    if configured is None:
        return None
    step = int(configured)
    if step < 1:
        raise ValueError("A real optimizer update must precede the storage fault")
    if trainer.global_steps != step:
        return None
    if getattr(trainer, "_storage_handoff_attempted", False):
        raise ValueError("Never repeat an uncertain storage handoff fault")
    from native_training_audit import verify_publication
    verify_publication(proof, step, trainer.config.actor_rollout_ref.rollout.trajectory_state.model_configuration_fingerprint)
    if step not in trainer.audit_verified_steps or step not in trainer.audit_publication_steps:
        raise ValueError("Original audit did not confirm this optimizer update and publication")
    root = os.environ["VERL_STORAGE_HANDOFF_EVIDENCE_ROOT"]
    trainer._storage_handoff_attempted = True
    from verl.trainer.ppo.v1.trajectory_control_dispatch import _dispatcher
    future = _dispatcher(trainer.agent_loop_manager).submit(exit_and_resume(trainer.agent_loop_manager, root,
        int(os.environ.get("VERL_STORAGE_HANDOFF_SOURCE_INDEX", "0")),
        int(os.environ.get("VERL_STORAGE_HANDOFF_TARGET_INDEX", "1")),
        float(os.environ.get("VERL_STORAGE_HANDOFF_TIMEOUT_S", "180"))))
    # Keep the future attached. A timeout may leave remote execution live and
    # does not authorize a second dispatch or mutation of the published policy.
    trainer._storage_handoff_future = future
    try:
        return future.result(timeout=float(os.environ.get("VERL_STORAGE_HANDOFF_TIMEOUT_S", "180")) + 10)
    except BaseException as error:
        event(root, "driver", "handoff_failed_without_retry", error_type=type(error).__name__,
            optimizer_step=step, gpu_release_verified=False, remote_outcome_requires_readback=True)
        raise
