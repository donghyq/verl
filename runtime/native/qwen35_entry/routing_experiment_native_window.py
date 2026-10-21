"""Attach bounded windows to the live native Trainer before TransferQueue closes.

No model loading, optimizer update, resource controller or new routing algorithm.
The optional patch calls before_teardown after native fit and its acceptance
checks. The existing Worker gets a passive observer and a paused read-only
render seam that delegates to its original business loop factory.
OwnedGatewayClient is constructed here from the original parent's private
endpoint. Only that parent restarts the CPU Gateway through OwnedPhase.launch;
Ray receives no live OwnedPhase/Popen/stream. Original replicas stay running.
Default-off settings come from the new descriptor's routing_experiment object.
Actual model, tokenizer and business rows come from the original prepared
config/freeze, after the first native fit has passed its own verification.

MLP allocation intervals can be read through allocation_reader(start, end, spec)
or supplied afterwards. Missing allocation or incomparable monotonic clocks
remain null. Version-switch windows with identical native invalidation behavior
are functional observations only and do not publish a routing benefit.
"""

import asyncio
import contextvars
import csv
from copy import deepcopy
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import socket
import time
import types

_MEMBER = contextvars.ContextVar("routing_native_observed_member", default=None)
_OBSERVER_ERRORS = []
OWNED_GATEWAY_EXPORT_FIELDS = (
    "binding(app_id,boot_epoch,phase,rank)", "owner_hostname/ip/kernel_boot_id",
    "gateway_root_identity(pid,start_ticks,uid,ppid)", "gateway_only_registered_descendant_identities",
    "gateway_url", "original_native_gateway_argv", "original_launch_argv",
    "source_manifest_sha256", "roots_file", "registry_file", "running_marker",
    "phase_deadline_monotonic", "replacement_root_registration_and_parent_cleanup_contract")


class NativeInterfaceGap(RuntimeError):
    pass


class NativeBudgetInsufficient(NativeInterfaceGap):
    pass


def _clock():
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        domain = path.read_text().strip()
    except OSError:
        domain = None
    return {"utc_unix_s": time.time(), "monotonic_ns": time.monotonic_ns(),
            "clock_domain": domain,
            "pid": os.getpid(), "socket_hostname": socket.gethostname()}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _write(path, event, **fields):
    row = {"event": event, **_clock(), **fields}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
    return row


def _scalar(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _jsonable(value):
    from runtime_observer import plain
    value = plain(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _observe(path, event, fields):
    """Observer failures never replace a core result or its original exception."""
    try:
        return _write(path, event, **_jsonable(fields() if callable(fields) else fields))
    except Exception as error:
        _OBSERVER_ERRORS.append({"event": event, "error_type": type(error).__name__})
        return None


def install_worker_observer(worker):
    """Observe the original source callback, lease generation and completion.

    Called only by the optional patch after V1TrajectoryEntry was configured.
    All wrappers return the exact original object and re-raise original failures.
    They do not implement a replacement legality/acceptance decision.
    """
    directory = os.environ.get("ROUTING_EXPERIMENT_OBSERVATION_DIR")
    if not directory or getattr(worker, "_routing_native_observed", False):
        return False
    worker._routing_native_observed = True
    path = Path(directory) / f"routing_experiment_worker_{socket.gethostname()}_{os.getpid()}.jsonl"
    runtime = worker.trajectory_entry.runtime
    source_member = runtime._source_member
    generate = worker.llm_client.generate

    @functools.wraps(generate)
    async def observed_generate(*args, **kwargs):
        member = _MEMBER.get()
        started = _clock()
        try:
            output = await generate(*args, **kwargs)
        except BaseException as error:
            _observe(path, "native_request_failed", {"member": member, "started": started,
                     "failure": type(error).__name__, "generated_tokens": None})
            raise
        _observe(path, "native_request_returned", lambda: {
            "member": member, "started": started,
            "request_id": args[0] if args else kwargs.get("request_id"),
            "prompt_tokens": len(kwargs["prompt_ids"]), "prompt_sha256": _digest(kwargs["prompt_ids"]),
            "generated_tokens": len(output.token_ids),
            "native_attempts": output.extra_fields.get("native_gateway_attempts"),
            "initial_prefill_cached_tokens": output.extra_fields.get("num_cached_tokens"),
            "stop_reason": output.stop_reason})
        return output

    async def observed_source_member(self, group, member, run_member):
        captured = {}

        @functools.wraps(run_member)
        async def observed_callback(index, original, lease, commit_snapshot, check_owner):
            identity = {"uid": group.uid, "member_index": index,
                        "trajectory_id": lease.trajectory_id, "execution_generation": lease.generation}
            captured["identity"] = identity
            token = _MEMBER.set(identity)
            try:
                output = await run_member(index, original, lease, commit_snapshot, check_owner)
                try:
                    value = output.output
                    captured["summary"] = {"keys": output.keys,
                        "source_reward": _scalar(value.reward_score),
                        "generated_model_tokens": sum(value.response_mask),
                        "response_tokens_including_tools": len(value.response_ids),
                        "trajectory_segments": value.extra_fields.get("trajectory_segments"),
                        "source_execution_generation": value.extra_fields.get("trajectory_execution_generation"),
                        "reward_extra_info": value.extra_fields.get("reward_extra_info"),
                        "committed_history_check": None,
                        "history_reason": "Original core owns restore/history checks; this observer returns the exact original object"}
                except Exception as error:
                    captured["summary"] = None
                    _observe(path, "native_observer_metadata_gap", {"uid": group.uid, "error_type": type(error).__name__})
                return output
            finally:
                _MEMBER.reset(token)

        try:
            result = await source_member(group, member, observed_callback)
        except BaseException as error:
            _observe(path, "native_source_member_failed", {"uid": group.uid, "member_index": member.index,
                     "attempt": captured.get("identity"), "failure": type(error).__name__})
            raise
        # The original runtime has checked ownership, persisted the output and
        # called finish_trajectory before it sets this state to completed.
        _observe(path, "native_source_member_returned", lambda: {
            "uid": group.uid, "member_index": member.index, "state": member.status,
            "output_record_ref": member.output_ref, "attempt": captured.get("identity"),
            "prepared": captured.get("summary"), "observer_io_errors": list(_OBSERVER_ERRORS)})
        return result

    worker.llm_client.generate = observed_generate
    runtime._source_member = types.MethodType(observed_source_member, runtime)
    _observe(path, "native_worker_observer_installed", lambda: {"runtime_owner_id": runtime.owner_id,
             "original_source_member_sha256": hashlib.sha256(inspect.getsource(source_member).encode()).hexdigest(),
             "original_generate_sha256": hashlib.sha256(inspect.getsource(generate).encode()).hexdigest()})
    return True


def _events(directory):
    rows = []
    for path in sorted(Path(directory).glob("routing_experiment_worker_*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def _dispatches(path, uids):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {row["uid"]: row for row in rows
            if row.get("event") == "group_dispatch_observed" and row.get("uid") in uids}


def _complete_groups(worker_events, uids, dispatches):
    grouped = {uid: [] for uid in uids}
    for row in worker_events:
        if row.get("event") == "native_source_member_returned" and row.get("uid") in grouped:
            grouped[row["uid"]].append(row)
    evidence, latencies = [], []
    timing_complete = True
    for uid, rows in grouped.items():
        completed = [row for row in rows if row["state"] == "completed"]
        members = sorted(row["member_index"] for row in completed)
        if members != list(range(8)) or any(row.get("attempt") is None or row.get("prepared") is None or
                                           row.get("observer_io_errors") for row in completed):
            raise NativeInterfaceGap("Original source member completion/attempt coverage is incomplete")
        start = dispatches.get(uid)
        domains = {row.get("clock_domain") for row in completed}
        duration = None
        if start is not None and len(domains) == 1 and start.get("clock_domain") in domains and None not in domains:
            duration = (max(row["monotonic_ns"] for row in completed) - start["monotonic_ns"]) / 1e9
            if duration < 0:
                raise ValueError("Inconsistent original monotonic clock boundary")
            latencies.append(duration)
        else:
            timing_complete = False
        evidence.append({"uid": uid, "members": completed, "dispatch": start,
                         "first_dispatch_to_final_legal_attempt_s": duration,
                         "timing_reason": None if duration is not None else "Different or unavailable kernel clock domains"})
    return evidence, latencies if timing_complete else None


def _run_async(call):
    return asyncio.run(call())


async def render_original_worker_prompts(worker, batch):
    """Read-only seam on the existing paused Worker and its actual loop factory."""
    import torch
    from tensordict import NonTensorData, NonTensorStack
    from verl.experimental.agent_loop.tool_agent_loop import AgentData

    if (not os.environ.get("ROUTING_EXPERIMENT_OBSERVATION_DIR") or
            not worker.trajectory_checkpoint_paused or worker.trajectory_entry is None):
        raise NativeInterfaceGap("Read-only original prompt rendering requires the opt-in paused native Worker")
    rendered = []
    for index in range(len(batch)):
        original = {}
        # Same exact TensorDict row extraction as _generate_sequences_accepted.
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                original[key] = deepcopy(value[index])
            elif isinstance(value, NonTensorStack):
                original[key] = deepcopy(value[index].data)
            elif isinstance(value, NonTensorData):
                original[key] = deepcopy(value.data)
            else:
                raise NativeInterfaceGap("Unsupported original Worker input type: " + key)
        original.setdefault("agent_name", worker.config.actor_rollout_ref.rollout.agent.default_agent_loop)
        loop = worker.trajectory_entry._new_loop(original)
        kwargs = worker.trajectory_entry._prompt_kwargs(original)
        data = AgentData(deepcopy(kwargs["raw_prompt"]), None, None, None, {}, {},
                        loop._make_request_id(0, loop.rollout_config.full_determinism), deepcopy(kwargs.get("tools_kwargs", {})))
        loop._check_text_state(data)
        selection = (kwargs.get("extra_info") or {}).get("tool_selection")
        if hasattr(loop, "business_initial_tools"):
            # The actual BusinessTrajectoryLoop.run uses this immutable initial
            # selection. Its ct_build_initial_tokens removes duplicate schemas.
            if selection is not None and selection != loop.business_initial_tools:
                raise ValueError("Original business initial tool selection differs")
            selection = loop.business_initial_tools
        loop._select_tools(data, selection)
        await loop._handle_pending_state(data, {})  # No GENERATING state, tool execution or model request.
        rendered.append({"prompt_ids": data.prompt_ids, "tool_schema_sha256": _digest(data._active_tool_schemas),
                         "original_record_sha256": _digest({key: _jsonable(value) for key, value in original.items() if key != "uid"}),
                         "actual_loop_class": type(loop).__module__ + "." + type(loop).__name__,
                         "actual_loop_source": str(Path(inspect.getfile(type(loop))).resolve()),
                         "actual_loop_source_sha256": hashlib.sha256(Path(inspect.getfile(type(loop))).read_bytes()).hexdigest(),
                         "factory": "worker.trajectory_entry._new_loop(actual_original_record)"})
    return rendered


def _render_original_prompts(trainer, batch, timeout_s):
    import ray
    manager = trainer.agent_loop_manager
    workers = [worker for index, worker in enumerate(manager.agent_loop_workers)
               if index not in manager._draining_worker_indices and index != manager._trajectory_reserved_target]
    if not workers:
        raise NativeInterfaceGap("No original native Worker is available for read-only rendering")
    return ray.get(workers[0].routing_experiment_render_prompts.remote(deepcopy(batch)), timeout=timeout_s)


def capture_live_fixture(trainer, groups_per_window, timeout_s):
    """Render at most two original batches, restore their sampler state, generate zero tokens."""
    from runtime_observer import fingerprint, plain
    start = time.monotonic()
    state = deepcopy(trainer.train_dataloader.state_dict())
    records, first_order = [], None
    manager = trainer.agent_loop_manager
    manager.prepare_trajectory_checkpoint(timeout_s=timeout_s)
    try:
        for _ in range(2):
            batch = trainer._next_train_batch(groups_per_window)
            content = {key: plain(value) for key, value in batch.items() if key != "uid"}
            if first_order is None:
                first_order = fingerprint(content)
            remaining = timeout_s - (time.monotonic() - start)
            if remaining <= 0:
                raise NativeInterfaceGap("Read-only native business rendering exhausted its owner budget")
            records.extend(_render_original_prompts(trainer, batch, remaining))
            if len({_digest(row["prompt_ids"]) for row in records}) >= 2 and len(records) >= groups_per_window:
                break
    finally:
        trainer.train_dataloader.load_state_dict(state)
        trainer.train_dataloader_it = None
    return {"original_records": records, "window_input_order_sha256": first_order,
            "factory": "worker.trajectory_entry._new_loop(actual_original_record)",
            "sampled_batches_at_most": 2, "formal_model_calls": 0, "tool_executions": 0,
            "sampler_state_restored": True}


class OwnerAllocationObservation:
    """Read actual job device assignments and the original active phase files.

    This measures assigned device time, including idle setup. It does not
    measure CUDA utilization or infer devices from the experiment budget.
    """
    def __init__(self, client, shared):
        self.binding = client.description["binding"]
        self.shared = Path(shared)
        self.root = self.shared / (self.binding["phase"] + "-ownership")
        if client.endpoint.parent.parent.resolve() != self.root.resolve():
            raise NativeInterfaceGap("Owner allocation files differ from the actual private parent endpoint")

    def snapshot(self):
        sources = []
        def read(path, as_json=True):
            path = Path(path)
            if path.is_symlink():
                raise NativeInterfaceGap("Indirect owner allocation evidence")
            raw = path.read_bytes()
            sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
            return json.loads(raw) if as_json else raw
        ticket = read(self.shared / "platform-host-allocation.json")
        if (ticket.get("status") != "PASS_DISTINCT_PHYSICAL_HOSTS" or
                any(ticket.get(key) != self.binding[key] for key in ("app_id", "boot_epoch"))):
            raise NativeInterfaceGap("Actual platform allocation is not bound to this original owner")
        uuids = []
        for rank, inventory_sha in enumerate(ticket["inventory_sha256"]):
            operator_raw = read(self.shared / ("operator-rank-%d.json" % rank), False)
            operator = json.loads(operator_raw)
            if (hashlib.sha256(operator_raw).hexdigest() != ticket["operator_sha256"][rank] or
                    operator.get("boot_id") != ticket["kernel_boot_ids"][rank] or
                    any(operator.get(key) != value for key, value in dict(self.binding, rank=rank).items() if key != "phase")):
                raise NativeInterfaceGap("Actual native operator/kernel binding changed")
            inventory = read(self.shared / ("gpus-rank-%d.csv" % rank), False)
            if hashlib.sha256(inventory).hexdigest() != inventory_sha:
                raise NativeInterfaceGap("Actual native device inventory changed")
            uuids.extend(row[2].strip() for row in csv.reader(inventory.decode().splitlines()))
            gate = read(self.shared / ("accepted-host-gate-%d.json" % rank))
            if (gate.get("status") != "PASS_DISTINCT_PHYSICAL_HOSTS" or
                    any(gate.get(key) != ticket[key] for key in ("app_id", "boot_epoch", "gpu_uuids", "kernel_boot_ids"))):
                raise NativeInterfaceGap("Actual native host/device gate is missing")
            roots = read(self.root / ("rank-%d" % rank) / "roots.json")
            if any(roots.get(key) != value for key, value in dict(self.binding, rank=rank).items()):
                raise NativeInterfaceGap("Actual phase process owner changed")
            if (self.root / ("rank-%d" % rank) / "watch.stop").exists() or (self.root / ("owner-cleanup-%d.json" % rank)).exists():
                raise NativeInterfaceGap("Native owned phase has already begun releasing assigned devices")
            stages = [json.loads(line) for line in read(self.shared / ("stages-rank-%d.jsonl" % rank), False).decode().splitlines() if line]
            stages = [row for row in stages if row.get("phase") == self.binding["phase"]]
            if (not stages or stages[-1].get("app_id") != self.binding["app_id"] or
                    stages[-1].get("rank") != rank or stages[-1].get("event") != "start" or
                    stages[-1].get("actual_stage") != ("original_native_fit" if rank == 0 else "await_original_native_fit")):
                raise NativeInterfaceGap("Original phase stage is not still holding its assigned devices")
        if self.root.joinpath("cleanup-request.json").exists() or self.root.joinpath("driver-exit.json").exists():
            raise NativeInterfaceGap("Original driver has ended the observed allocation interval")
        if not uuids or len(set(uuids)) != len(uuids) or uuids != ticket["gpu_uuids"]:
            raise NativeInterfaceGap("Actual allocated UUID coverage is incomplete or duplicated")
        return {"gpu_uuids": uuids, "binding": self.binding, "sources": sources,
                "basis": "Original native exclusive visible-device allocation, immutable platform assignment and still-active owned phase; assigned time includes CPU/idle time, not utilization"}


class NativeWindowBridge:
    """Existing live Trainer + existing native methods; no model/process controller."""

    def __init__(self, trainer, config, *, phase_events, worker_observations,
                 replace_owned_gateway=None, allocation_reader=None):
        self.trainer = trainer
        self.config = config
        self.phase_events = Path(phase_events)
        self.worker_observations = Path(worker_observations)
        self.replace_owned_gateway = replace_owned_gateway
        self.allocation_reader = allocation_reader
        self.original_loader_state = deepcopy(trainer.train_dataloader.state_dict())
        self.input_order = None

    async def run_window(self, spec, freeze, output_dir):
        # Keep native synchronous APIs outside the driver's asyncio event loop.
        return await asyncio.to_thread(self._run_window, spec, freeze, Path(output_dir))

    def _publication(self):
        import ray
        from verl.trainer.ppo.v1.trajectory_state.policy_publication import PublishedPolicyRecord
        manager = self.trainer.agent_loop_manager
        record = ray.get(manager.trajectory_coordinator.get_active_published_policy.remote(), timeout=180)
        identity = PublishedPolicyRecord.decode(record)
        lifecycle = self.trainer.standalone_checkpoint_manager.gateway_control
        from run_v1_functional_gate import _read_trainer_active_weight_version
        async def actual():
            return await _read_trainer_active_weight_version(self.trainer)
        version = _run_async(actual)
        if str(version) != identity.identity.policy_version:
            raise ValueError("Actual engine version differs from the original published policy")
        return version, identity, lifecycle

    def _run_window(self, spec, freeze, output):
        from routing_experiment_entry import NativeR2Primitives, shared_prefix
        from runtime_observer import fingerprint, group_membership, input_observation, plain
        import transfer_queue as tq

        if not callable(self.replace_owned_gateway):
            raise NativeInterfaceGap("Missing strict original-parent endpoint adapter from the opt-in bootstrap hook")
        trainer, manager = self.trainer, self.trainer.agent_loop_manager
        if not trainer.audit_verified_steps or not trainer.audit_publication_steps:
            raise NativeInterfaceGap("A real native optimizer update and verified publication must precede the window")
        if spec["groups_per_window"] != self.config.data.train_batch_size:
            raise NativeInterfaceGap("Original audited advantage hook requires the configured complete native batch size")
        output.mkdir(parents=True, exist_ok=True)
        journal = output / "events.jsonl"
        beginning = _clock()
        allocation_before = None
        allocation_reason = None
        if isinstance(self.allocation_reader, OwnerAllocationObservation):
            try:
                allocation_before = self.allocation_reader.snapshot()
            except Exception as error:
                allocation_reason = type(error).__name__ + ": " + str(error)
        costs = []
        deadline = time.monotonic() + spec["max_window_s"]

        def remaining():
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise NativeInterfaceGap("Frozen native window wall-time budget exhausted; no next window or optimizer retry")
            return seconds

        completed = False
        restored = False
        selected_keys = []
        result = {"status": "STARTED_NOT_PASS", "optimizer_updates_in_window": 0,
                  "model_reloads": 0, "resource_controller_added": False}
        try:
            manager.prepare_trajectory_checkpoint(timeout_s=remaining())
            version, published, lifecycle = self._publication()
            if set(lifecycle.urls) != set(freeze["owned_runtime"]["worker_urls"]):
                raise ValueError("Live replica URLs differ from the exact owned MLP freeze")
            before = time.monotonic()
            _run_async(lifecycle.stop_admission)
            receipt = self.replace_owned_gateway(spec["native_policy"], tuple(lifecycle.urls), version)
            if (receipt.get("actual_native_policy") != spec["native_policy"] or
                    receipt.get("old_owned_gateway_exited") is not True or not receipt.get("new_gateway_pid_identity") or
                    not receipt.get("ownership_binding") or receipt.get("empty_registry_confirmed") is not True):
                raise ValueError("Owned CPU Gateway policy replacement was not actually confirmed")
            native_argv = receipt.get("native_gateway_argv", [])
            if (native_argv[:2] != ["smg", "launch"] or "--policy" not in native_argv or
                    native_argv[native_argv.index("--policy") + 1] != spec["native_policy"] or
                    "--worker-urls" in native_argv or not receipt.get("source_receipt")):
                raise ValueError("Receipt must preserve the actual native policy launch argv and empty-registry bootstrap")
            if receipt.get("gateway_url") != freeze["owned_runtime"]["gateway_url"]:
                raise ValueError("Gateway URL changed during the owned native window")
            _run_async(lambda: lifecycle.start_admission(version))
            verification = getattr(self.replace_owned_gateway, "verify_after_registration", None)
            if not callable(verification):
                raise NativeInterfaceGap("Owned parent must freshly observe its actual replacement after original registration")
            registered = verification(receipt)
            if {row.get("url") for row in registered["workers"]} != set(lifecycle.urls):
                raise ValueError("Actual restored native Gateway registry differs from original live replicas")
            registered_version, registered_policy, _ = self._publication()
            if registered_version != version or registered_policy.encode() != published.encode():
                raise ValueError("Original registered engine/publication identity changed during CPU policy replacement")
            receipt["post_registration_owner_observation"] = registered
            costs.append({"operation": "native_gateway_policy_switch", "elapsed_s": time.monotonic() - before,
                          "receipt": receipt})
            trainer.train_dataloader.load_state_dict(deepcopy(self.original_loader_state))
            trainer.train_dataloader_it = None
            batch = trainer._next_train_batch(spec["groups_per_window"])
            uids = set(plain(batch["uid"]))
            content = {key: plain(value) for key, value in batch.items() if key != "uid"}
            actual_order = fingerprint(content)
            if actual_order != freeze["window_input_order_sha256"]:
                raise NativeInterfaceGap("Original native sampler differs from the live prepared first batch")
            if self.input_order is not None and actual_order != self.input_order:
                raise ValueError("Restored original native DataLoader did not repeat the same business inputs")
            self.input_order = actual_order
            rendered = _render_original_prompts(trainer, batch, remaining())
            prompt_ids = [row["prompt_ids"] for row in rendered]
            prefix_length = freeze["common_prefix_tokens"]
            prefix = prompt_ids[0][:prefix_length] if prompt_ids else []
            if (not prefix or _digest(prefix) != freeze["common_prefix_sha256"]
                    or any(tokens[:prefix_length] != prefix or not 6000 <= len(tokens) <= 7000 for tokens in prompt_ids)):
                raise NativeInterfaceGap("Actual native sampler/tool-rendered batch is outside the original 6–7K shared-prefix window; do not pad or rewrite it")

            # Reuse r2's actual flush protocol. Both arms get the same controlled
            # cache eviction and one genuine business-prompt prewarm on the hot worker.
            primitives = NativeR2Primitives(freeze)
            before = time.monotonic()
            flushes = [primitives.flush(url) for url in lifecycle.urls]
            costs.append({"operation": "flush", "elapsed_s": time.monotonic() - before, "receipts": flushes})
            salt = lifecycle.client.build_policy_cache_salt(run_namespace=published.run_id,
                policy_version=version, policy_identity_salt=published.identity.cache_salt)
            before = time.monotonic()
            warm = primitives.r2.http(spec["hot_worker_url"] + "/generate", {
                "rid": f"owned-routing-prewarm-{spec['window_index']}", "input_ids": prompt_ids[0],
                "sampling_params": {"max_new_tokens": 1, "temperature": self.config.actor_rollout_ref.rollout.temperature,
                                    "top_p": self.config.actor_rollout_ref.rollout.top_p, "top_k": self.config.actor_rollout_ref.rollout.top_k},
                "cache_salt": salt, "stream": False})
            meta = warm.get("meta_info", {})
            if str(meta.get("weight_version")) != str(version):
                raise ValueError("Actual prewarm engine version changed")
            costs.append({"operation": "cache_prewarm", "elapsed_s": time.monotonic() - before,
                          "worker_url": spec["hot_worker_url"], "discarded_output_tokens": meta.get("completion_tokens"),
                          "native_meta_info": meta})
            manager.resume_trajectory_checkpoint_admission()
            remaining()
            dispatch_clock = _clock()
            trainer._submit_batch_to_rollout(batch)
            manager.prepare_trajectory_checkpoint(timeout_s=remaining())

            # Apply the original sampler rules to this window's original UIDs.
            # No new staleness/reward criteria, refill or generation scheduler.
            replay = trainer.replay_buffer
            replay._sync_metadata_from_transfer_queue()
            reasons = replay._terminal_eviction_reasons(trainer.global_steps, "train")
            eligible = replay._sampleable_terminal_keys("train", reasons) & uids
            if eligible != uids or not replay._has_enough_samples(trainer.global_steps, "train", len(uids), eligible):
                raise NativeInterfaceGap("Original ReplayBuffer did not accept every original group in this bounded window")
            chosen, snapshot, _ = replay._select_prompt_uids("train", eligible, len(uids))
            original_batch = replay._materialize_batch("train", chosen, snapshot)
            selected_keys = original_batch.keys
            original_batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            native_metrics = {}
            if trainer.reward_loop_manager.reward_loop_worker_handles is None:
                original_batch = trainer._compute_reward_colocate(original_batch, metrics=native_metrics)
            original_batch = trainer._balance_batch(original_batch, metrics=native_metrics)
            original_batch = trainer._compute_old_log_prob(original_batch, native_metrics)
            if trainer.use_reference_policy:
                original_batch = trainer._compute_ref_log_prob(original_batch, native_metrics)
            if trainer.use_critic:
                original_batch = trainer._compute_values(original_batch, native_metrics)
            original_batch = trainer._compute_advantage(original_batch, native_metrics)
            group_membership(original_batch.keys)  # Same observer's actual member 0..7 audit.
            dispatches = _dispatches(self.phase_events, uids)
            # Add kernel clock identity only to this observer's own dispatch;
            # the original event is preserved and its timestamp is not replaced.
            for row in dispatches.values():
                if row.get("socket_hostname") == dispatch_clock["socket_hostname"]:
                    row["clock_domain"] = dispatch_clock["clock_domain"]
            members, latencies = _complete_groups(_events(self.worker_observations), uids, dispatches)
            admission = input_observation(original_batch, dispatches)
            if admission.get("status") != "OBSERVED_ACTUAL_GROUP_METADATA":
                raise NativeInterfaceGap("Original sampler acceptance exists but original member/mask metadata is unavailable")
            available_clock = _clock()
            driver_latencies = []
            for uid in sorted(uids):
                start = dispatches.get(uid) or {}
                elapsed = ((available_clock["monotonic_ns"] - start["monotonic_ns"]) / 1e9
                           if start.get("pid") == available_clock["pid"] and start.get("monotonic_ns") is not None else None)
                if elapsed is None or elapsed < 0:
                    driver_latencies = None
                    break
                driver_latencies.append(elapsed)
                _write(journal, "driver_observed_legal_group_available", uid=uid, original_dispatch=start,
                       original_confirmation=admission, first_dispatch_to_driver_observed_group_available_s=elapsed,
                       scope="Same Trainer process after original ReplayBuffer/TQ/advantage member-mask verification; includes native drain/poll/verification and batch barrier overhead, not precise worker completion")
            _write(journal, "training_group_admission_observed", metadata=admission,
                   original_sampler_method="ReplayBufferAsync._sampleable_terminal_keys/_select_prompt_uids/_materialize_batch",
                   original_advantage_method="PPOTrainer._compute_advantage", actual_member_attempts=members)
            for row in dispatches.values():
                with journal.open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            after_version, after_published, _ = self._publication()
            if after_version != version or after_published.encode() != published.encode():
                raise ValueError("Actual policy changed during the fixed-model window")
            source_members = [row for group in members for row in group["members"]]
            summaries = [row["prepared"] for row in source_members]
            rewards = [row.get("source_reward") for row in summaries]
            requests = [row for row in _events(self.worker_observations)
                        if (row.get("member") or {}).get("uid") in uids and row["event"].startswith("native_request_")]
            failures = [row for row in requests if row["event"] == "native_request_failed"]
            first_requests = {}
            for row in sorted(requests, key=lambda row: row["monotonic_ns"]):
                member = row.get("member") or {}
                if row["event"] == "native_request_returned":
                    first_requests.setdefault((member.get("uid"), member.get("member_index")), row)
            for uid, tokens in zip(plain(batch["uid"]), prompt_ids, strict=True):
                if any(first_requests.get((uid, index), {}).get("prompt_sha256") != _digest(tokens) for index in range(8)):
                    raise NativeInterfaceGap("Actual first model request differs from native prewarm prompt rendering, or original request observation is absent")
            generated = sum(row["generated_tokens"] for row in requests) if requests and not failures else None
            model_lengths = [row["generated_model_tokens"] for row in summaries]
            native_attempts = [attempt for row in requests for attempt in (row.get("native_attempts") or [])]
            attempt_lengths = [row["output_end"] - row["output_start"] for row in native_attempts]
            attempt_coverage = bool(requests) and not failures and all(row.get("native_attempts") for row in requests)
            finish_counts = {}
            for attempt in native_attempts:
                finish = (attempt.get("native_meta_info") or {}).get("finish_reason")
                key = json.dumps(finish, sort_keys=True)
                finish_counts[key] = finish_counts.get(key, 0) + 1
            result.update(status="OBSERVED_NATIVE_GROUPS_NO_OPTIMIZER_UPDATE", native_policy=spec["native_policy"],
                          actual_parameter_version=version, actual_policy_record=published.encode(),
                          accepted_original_groups=len(chosen), source_member_attempts=members,
                          initial_prompt_tokens=[len(tokens) for tokens in prompt_ids], common_prefix_tokens=len(prefix),
                          common_prefix_fraction=[len(prefix) / len(tokens) for tokens in prompt_ids],
                          actual_tool_schema_sha256=[row["tool_schema_sha256"] for row in rendered],
                          actual_business_loop_classes=[row["actual_loop_class"] for row in rendered],
                          original_prompt_factory=rendered[0]["factory"], source_native_metrics=plain(native_metrics))
            remaining()
            completed = True
            metrics = {"legal_complete_groups": len(chosen), "first_dispatch_to_final_legal_attempt_s": latencies,
                "first_dispatch_to_driver_observed_group_available_s": driver_latencies,
                "group_timing_scope": "Driver observation of original legal complete group availability, with original drain/poll/verification and batch barrier overhead; exact worker completion remains separate",
                "mean_reward": sum(rewards) / len(rewards) if all(value is not None for value in rewards) else None,
                "business_success_rate": None,
                "business_success_reason": "Source rewards/extra_info preserved; no independent business-success verdict is invented",
                "mean_generated_tokens_per_attempt": sum(attempt_lengths) / len(attempt_lengths) if attempt_coverage else None,
                "mean_generated_tokens_per_completed_member": sum(model_lengths) / len(model_lengths),
                "generated_tokens_including_failed_attempts": generated,
                "effective_training_tokens": sum(member["training_effective_model_tokens"] for group in admission.get("groups", []) for member in group["members"])
                    if admission.get("status") == "OBSERVED_ACTUAL_GROUP_METADATA" else None,
                "failure_distribution": {"native_request_failed": len(failures), "native_attempt_finish_reasons": finish_counts},
                "cached_tokens": sum(row["initial_prefill_cached_tokens"] for row in requests)
                    if requests and not failures and all(row.get("initial_prefill_cached_tokens") is not None for row in requests) else None,
                "operation_costs": costs,
                "multi_turn_history_growth_tokens": [row["prompt_tokens"] for row in requests if row["event"] == "native_request_returned"]}
            # Charge original TQ clearing and loader/admission restoration too.
            if selected_keys:
                tq.kv_clear(keys=selected_keys, partition_id="train")
                selected_keys = []
            trainer.train_dataloader.load_state_dict(deepcopy(self.original_loader_state))
            trainer.train_dataloader_it = None
            manager.resume_trajectory_checkpoint_admission()
            restored = True
            end_clock = _clock()
            allocation_intervals = None
            if allocation_before is not None:
                try:
                    allocation_after = self.allocation_reader.snapshot()
                    if allocation_after["gpu_uuids"] != allocation_before["gpu_uuids"]:
                        raise NativeInterfaceGap("Actual device allocation changed inside the window")
                    duration = (end_clock["monotonic_ns"] - beginning["monotonic_ns"]) / 1e9
                    allocation_receipt = output / "actual-owner-allocation.json"
                    allocation_receipt.write_text(json.dumps({"before": allocation_before, "after": allocation_after,
                        "begin_clock": beginning, "end_clock": end_clock, "observed_assigned_duration_s": duration}, allow_nan=False, indent=2) + "\n")
                    allocation_intervals = [{"gpu_uuid": uuid, "start_wall_s": beginning["utc_unix_s"],
                        "end_wall_s": beginning["utc_unix_s"] + duration, "exclusive": True,
                        "source_receipt": str(allocation_receipt), "allocation_basis": allocation_before["basis"]}
                        for uuid in allocation_before["gpu_uuids"]]
                except Exception as error:
                    allocation_reason = type(error).__name__ + ": " + str(error)
            elif callable(self.allocation_reader):
                allocation_intervals = self.allocation_reader(beginning["utc_unix_s"], end_clock["utc_unix_s"], spec)
            result["actual_allocation_intervals"] = allocation_intervals
            result["actual_allocation_reason"] = allocation_reason
            result["cache_applicability"] = {**freeze.get("cache_applicability", {}),
                "actual_cached_tokens": metrics["cached_tokens"],
                "observed_reuse": metrics["cached_tokens"] > 0 if metrics["cached_tokens"] is not None else None,
                "interpretation": "Only actual returned prefix-cache counters describe these requests; zero hits do not prove universal model support or a routing benefit"}
            initialization = {"checkpoint_sha256": _digest(published.encode()),
                "checkpoint_binding_scope": "Original immutable published policy record; not a checkpoint-file checksum",
                "input_order_sha256": actual_order, "sampling_sha256": _digest(freeze["original_sampling_params"]),
                "engine_cache_initial_state_sha256": _digest({"flushed_urls": sorted(lifecycle.urls), "warm_prompt": _digest(prompt_ids[0]),
                                                              "hot_worker": spec["hot_worker_url"], "cache_salt": salt}),
                "router_initial_state_sha256": _digest({"fresh_owned_gateway": True, "empty_before_registration": True,
                                                        "owned_worker_urls": sorted(lifecycle.urls)}), "hot_worker_url": spec["hot_worker_url"]}
            return {"events_file": str(journal), "terminal_result_file": str(output / "result.json"),
                "original_group_accounting_complete": True, "original_metrics": metrics, "initial_conditions": initialization,
                "runtime_contract": {"native_policy": receipt["actual_native_policy"], "partial_rollout_preserved": True,
                    "session_affinity_preserved": True, "old_policy_worker_admission_blocked": True,
                    "committed_history_preserved": True, "controlled_restart_or_eviction_confirmed": True,
                    "engine_cache_invalidated_at_switch": True, "router_prediction_invalidated_at_switch": True,
                    "shared_cache_present": None},
                "independent_run_id": receipt.get("independent_reset_receipt_id"),
                "independent_reset_scope": "Fresh owned native CPU Gateway plus confirmed engine eviction; fixed live policy, no independent model launch",
                "allocation_intervals": allocation_intervals, "allocation_reason": allocation_reason}
        except BaseException as error:
            result.update(status="NOT_ACCEPTED_NATIVE_WINDOW", error_type=type(error).__name__,
                          interface_gap=str(error) if isinstance(error, NativeInterfaceGap) else None,
                          automatic_optimizer_or_window_retry_allowed=False, operation_costs=costs)
            raise
        finally:
            (output / "result.json").write_text(json.dumps(_jsonable(result), allow_nan=False, indent=2) + "\n")
            if selected_keys:
                tq.kv_clear(keys=selected_keys, partition_id="train")
            # No auto-reopening on an uncertain original runtime failure.
            if completed and not restored:
                trainer.train_dataloader.load_state_dict(deepcopy(self.original_loader_state))
                trainer.train_dataloader_it = None
                manager.resume_trajectory_checkpoint_admission()


def _prepare_live_freeze(trainer, config, frozen, control, options):
    """Derive inputs from the actual native prepare, never a hand-made workload."""
    from types import SimpleNamespace
    from routing_experiment_entry import prepare
    native_out = Path(control["output"])
    preparation = native_out / "routing-preparation"
    preparation.mkdir(exist_ok=False)
    probe = NativeWindowBridge(trainer, config, phase_events=native_out.parent / (native_out.name + "-observation") / "events.jsonl",
                               worker_observations=options["worker_observations"])
    version, published, lifecycle = probe._publication()
    from routing_experiment_owned_gateway import OwnedGatewayClient
    owner = OwnedGatewayClient(options["owned_gateway_endpoint"], source_sha256=control["runtime_source_sha256"],
        gateway_url=options["gateway_url"], timeout_s=options.get("owner_control_timeout_s", 90))
    if len(lifecycle.urls) != 2 or int(config.actor_rollout_ref.rollout.tensor_model_parallel_size) != 2:
        raise NativeInterfaceGap("Routing evaluation requires exactly two live original TP2 replicas")
    native_argv = owner.description.get("native_gateway_argv", [])
    if ("--policy" not in native_argv or native_argv[native_argv.index("--policy") + 1] != "least_load"):
        raise NativeInterfaceGap("The actual original native route is not the frozen least_load baseline")
    budget = owner.observe_original_budget()
    if budget["remaining_driver_seconds"] < 6 * options["max_window_s"] + 30:
        raise NativeBudgetInsufficient("Original owner deadline cannot cover preparation plus bounded AA+ABBA")
    preparation_clock = _clock()
    allocation_reader = OwnerAllocationObservation(owner, native_out.parent)
    allocation_start, allocation_reason = None, None
    try:
        allocation_start = allocation_reader.snapshot()
    except Exception as error:
        allocation_reason = type(error).__name__ + ": " + str(error)
    fixture = capture_live_fixture(trainer, int(config.data.train_batch_size),
        min(options.get("owner_control_timeout_s", 90), budget["remaining_driver_seconds"] - 6 * options["max_window_s"]))
    source = Path(control["observer_repo"]).resolve(strict=True)
    for row in fixture["original_records"]:
        loop_source = Path(row["actual_loop_source"]).resolve(strict=True)
        if not loop_source.is_relative_to(source) or hashlib.sha256(loop_source.read_bytes()).hexdigest() != row["actual_loop_source_sha256"]:
            raise NativeInterfaceGap("Live business loop source differs from this native run source")
    owned = {"scope": "owned_mlp", "run_id": published.run_id, "source_receipt": str(owner.endpoint),
             "gateway_url": owner.description["gateway_url"], "worker_urls": list(lifecycle.urls),
             "ownership_binding": owner.description["binding"], "actual_parameter_version": version,
             "actual_policy_record": published.encode()}
    owned_path, limits_path = preparation / "owned-runtime.json", preparation / "quality-limits.json"
    owned_path.write_text(json.dumps(owned, allow_nan=False) + "\n")
    limits_path.write_text(json.dumps(options["quality_limits"], allow_nan=False) + "\n")
    freeze_path = preparation / "freeze.json"
    args = SimpleNamespace(config=options["native_config"], native_freeze=options["native_freeze"],
        runtime_source_root=control["observer_repo"], owned_runtime=str(owned_path), executor=__file__,
        quality_limits=str(limits_path), output=str(freeze_path), repo=control["observer_repo"],
        scenario=options.get("scenario", "cold_worker"), native_policy=options.get("native_policy", "cache_aware"),
        groups_per_window=int(config.data.train_batch_size), min_p95_groups=options.get("min_p95_groups", 400),
        gpu_budget_hours=options["gpu_budget_hours"], max_window_s=options["max_window_s"],
        allocated_gpus=options["allocated_gpus_for_budget_reservation_only"],
        verified_native_freeze=frozen, live_rendered_fixture=fixture)
    status = prepare(args)
    if status["status"] != "PREPARED_NOT_EXECUTED":
        raise NativeInterfaceGap("Actual original native prepare did not supply the business prompt window: " + status["status"])
    actual = json.loads(freeze_path.read_text())
    model_config = json.loads((Path(config.actor_rollout_ref.model.path) / "config.json").read_text())
    from omegaconf import OmegaConf
    actual_rollout = OmegaConf.to_container(config.actor_rollout_ref.rollout, resolve=True)
    actual["cache_applicability"] = {"enable_prefix_caching": config.actor_rollout_ref.rollout.get("enable_prefix_caching"),
        "engine_kwargs_sglang": actual_rollout.get("engine_kwargs", {}).get("sglang", {}),
        "model_type": model_config.get("model_type"), "model_classes": model_config.get("architectures"),
        "gdn_cache_support": None, "actual_engine_prefix_cache_support": None,
        "support_reason": "Original published version metadata does not expose actual Qwen/GDN cache capability; do not infer support from config alone",
        "model_configuration_changed_for_experiment": False}
    actual["preparation_allocation_intervals"] = None
    try:
        allocation_end = allocation_reader.snapshot()
        if allocation_start is None or allocation_end["gpu_uuids"] != allocation_start["gpu_uuids"]:
            raise NativeInterfaceGap("Preparation allocation coverage is unavailable or changed")
        end = _clock()
        duration = (end["monotonic_ns"] - preparation_clock["monotonic_ns"]) / 1e9
        cost_path = preparation / "actual-preparation-allocation.json"
        cost_path.write_text(json.dumps({"before": allocation_start, "after": allocation_end,
            "begin_clock": preparation_clock, "end_clock": end, "observed_assigned_duration_s": duration}, indent=2) + "\n")
        actual["preparation_allocation_intervals"] = [{"gpu_uuid": uuid,
            "start_wall_s": preparation_clock["utc_unix_s"], "end_wall_s": preparation_clock["utc_unix_s"] + duration,
            "exclusive": True, "source_receipt": str(cost_path), "allocation_basis": allocation_start["basis"]}
            for uuid in allocation_start["gpu_uuids"]]
    except Exception as error:
        allocation_reason = type(error).__name__ + ": " + str(error)
    actual["preparation_allocation_reason"] = allocation_reason
    freeze_path.write_text(json.dumps(actual, allow_nan=False, indent=2) + "\n")
    if actual["cache_applicability"]["enable_prefix_caching"] is False:
        raise NativeInterfaceGap("Actual prefix caching is disabled: mechanism not applicable; original model config is unchanged")
    return actual, owner


def before_teardown(trainer, config, frozen, control):
    """Optional before-tq.close hook; disabled unless explicitly frozen by owner."""
    options = control.get("routing_experiment")
    if not options:
        return {"status": "DISABLED_NOT_REQUIRED_FOR_FIRST_TRAINING_RUN"}
    if not isinstance(options, dict):
        return {"status": "NOT_ACCEPTED_INVALID_OPTIONAL_WINDOW_SPEC", "model_reloads": 0, "optimizer_updates": 0}
    if not options.get("owned_gateway_endpoint"):
        return {"status": "NOT_ACCEPTED_MISSING_OWNED_PARENT_ENDPOINT",
                "missing_interface": "Apply the original-parent wait hook and forward its private endpoint in routing_experiment settings",
                "model_reloads": 0, "optimizer_updates": 0, "routing_benefit": None}
    try:
        from routing_experiment_entry import file_sha, owned_urls, run_campaign
        freeze, callback = _prepare_live_freeze(trainer, config, frozen, control, options)
        observed_budget = callback.observe_original_budget()
        if observed_budget["remaining_driver_seconds"] < 6 * freeze["max_window_s"]:
            raise NativeBudgetInsufficient("Original driver deadline cannot cover bounded AA + ABBA; no new generation is admitted")
        if freeze.get("status") != "PREPARED_NOT_EXECUTED":
            raise NativeInterfaceGap("Original real-business prompt freeze has not passed preparation")
        owned_urls(freeze["owned_runtime"])
        for key in ("executor", "r2_driver", "native_config", "native_freeze"):
            if file_sha(freeze[key]["path"]) != freeze[key]["sha256"]:
                raise ValueError("Frozen original input changed: " + key)
        if freeze["scenario"] == "version_switch":
            return {"status": "FUNCTIONAL_ONLY_IDENTICAL_NATIVE_INVALIDATION",
                    "reason": "Both native arms delete/re-register workers; no stale-tree treatment difference exists",
                    "routing_benefit": None, "model_reloads": 0,
                    "new_version_rejection_test_performed": False,
                    "old_version_rejection_runtime_proof": None,
                    "proof_reason": "No extra weight update or stale-version model request is issued by this fixed-policy hook; use the original publication/rejection receipts"}
        native_out = Path(control["output"])
        bridge = NativeWindowBridge(trainer, config,
            phase_events=native_out.parent / (native_out.name + "-observation") / "events.jsonl",
            worker_observations=options["worker_observations"], replace_owned_gateway=callback,
            allocation_reader=OwnerAllocationObservation(callback, native_out.parent))
        return asyncio.run(run_campaign(freeze, str(native_out / "routing-comparison"), bridge))
    except BaseException as error:
        # Preserve the completed native training result. This optional experiment
        # cannot trigger a repeated optimizer update or another model launch.
        return {"status": "NOT_RUN" if isinstance(error, NativeBudgetInsufficient) else "NOT_ACCEPTED_NATIVE_WINDOW", "error_type": type(error).__name__,
                "error": str(error),
                "interface_gap": str(error) if isinstance(error, NativeInterfaceGap) else None,
                "routing_benefit": None, "automatic_retries": 0, "model_reloads": 0}


def after_joint_acceptance(trainer, config, frozen, control, report, output):
    """Preserve full joint proof first; optional routing cannot replace it."""
    joint = report.get("joint_handoff") or {}
    elastic = report.get("elastic_hybrid_acceptance") or {}
    checks = joint.get("acceptance_checks") or {}
    accepted = (report.get("status") == "PASS_NATIVE_UPDATES_AND_PUBLICATIONS"
        and report.get("verified_optimizer_updates_this_process", 0) >= 1
        and report.get("learning_signal_updates_this_process", 0) >= 1
        and (report.get("parameter_sample_proof") or {}).get("observed_parameter_change") is True
        and (report.get("version1_token_witness") or {}).get("status") == "PASS_NATIVE_VERSION1_TOKEN"
        and joint.get("status") == "PASS_NATIVE_ELASTIC_HYBRID_TOKEN_KV_HANDOFF"
        and joint.get("actual_gpu_reuse_verified") is True
        and elastic.get("accepted") is True and bool(checks) and all(value is True for value in checks.values()))
    prior = {"status": "PASS_JOINT_ACCEPTANCE_BEFORE_OPTIONAL_ROUTING" if accepted else "NOT_PASS_JOINT_ACCEPTANCE",
        "native_status": report.get("status"), "verified_optimizer_updates_this_process": report.get("verified_optimizer_updates_this_process"),
        "learning_signal_updates_this_process": report.get("learning_signal_updates_this_process"),
        "parameter_sample_proof": report.get("parameter_sample_proof"),
        "version1_token_witness": report.get("version1_token_witness"),
        "joint_handoff_status": joint.get("status"), "joint_acceptance_checks": checks,
        "actual_gpu_reuse_verified": joint.get("actual_gpu_reuse_verified"),
        "original_member_lifecycle_proof": joint.get("original_member_lifecycle_proof"),
        "elastic_hybrid_acceptance": elastic}
    path = Path(output) / "joint-acceptance-before-routing.json"
    path.write_text(json.dumps(prior, allow_nan=False, indent=2) + "\n")
    if not accepted:
        return {"status": "NOT_RUN", "reason": "Full original native joint acceptance has not passed",
                "joint_acceptance_receipt": str(path), "routing_benefit": None, "model_reloads": 0}
    try:
        from routing_experiment_owned_gateway import OwnedGatewayClient
        options = control["routing_experiment"]
        owner = OwnedGatewayClient(options["owned_gateway_endpoint"], source_sha256=control["runtime_source_sha256"],
            gateway_url=options["gateway_url"], timeout_s=options.get("owner_control_timeout_s", 90))
        budget = owner.observe_original_budget()
        required = 6 * options["max_window_s"] + options.get("owner_control_timeout_s", 90) + 15
        if budget["remaining_driver_seconds"] < required:
            return {"status": "NOT_RUN", "reason": "Insufficient original driver budget for AA+ABBA plus preparation and final receipt",
                "remaining_original_driver_seconds": budget["remaining_driver_seconds"], "required_seconds": required,
                "joint_acceptance_receipt": str(path), "routing_benefit": None, "model_reloads": 0,
                "new_assigned_budget": False, "joint_result_overridden": False}
        result = before_teardown(trainer, config, frozen, control)
        result.update(joint_acceptance_receipt=str(path), joint_result_overridden=False, new_assigned_budget=False)
        return result
    except BaseException as error:
        return {"status": "NOT_ACCEPTED_OPTIONAL_ROUTING", "error_type": type(error).__name__, "error": str(error),
            "joint_acceptance_receipt": str(path), "routing_benefit": None, "automatic_retries": 0,
            "model_reloads": 0, "joint_result_overridden": False}
