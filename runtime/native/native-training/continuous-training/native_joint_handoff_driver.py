"""Controlled one-shot Token/KV interruption driver for the native V1 entry.

The driver is intentionally an observer/controller around the existing V1
components.  It does not construct training samples, rewards, tool results, or
a replacement scheduler.  A native observed_run_original_fit wrapper can call
``install_on_trainer`` after ``AgentLoopManagerTQ`` has been created and before
``trainer.fit`` is invoked, then call ``finalize`` with the original entry's
final report.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DriverOptions:
    min_generated_tokens: int = 18
    poll_interval_s: float = 0.05
    control_timeout_s: float = 1800.0
    source_group_preemption_budget_s: float = 600.0
    target_completion_reserve_s: float = 600.0
    target_worker_index: int | None = None
    result_path: str | Path | None = None


class NativeJointHandoffDriver:
    """Wait for a real source stream, request one preemption, and observe it."""

    def __init__(self, manager: Any, trainer: Any | None = None, *, options: DriverOptions | None = None,
                 actual_topology: dict | None = None):
        self.manager = manager
        self.trainer = trainer
        self.options = options or DriverOptions()
        if type(self.options.min_generated_tokens) is not int or self.options.min_generated_tokens < 18:
            raise ValueError("the native joint driver requires a snapshot threshold of at least 18 tokens")
        if (self.options.source_group_preemption_budget_s != 600
                or self.options.target_completion_reserve_s != 600
                or self.options.control_timeout_s != 1800):
            raise ValueError("joint preemption requires fixed source600/target600/controller1800 admission limits")
        self._started_ns = time.time_ns()
        self._monitor_future = None
        self._monitor_deadline = None
        self._dispatch_lock = threading.Lock()
        self._receipt_lock = threading.RLock()
        topology = actual_topology or getattr(trainer, "_joint_actual_topology", None)
        if not isinstance(topology, dict) or not isinstance(topology.get("bindings"), list):
            raise ValueError("elastic driver requires actual hybrid/standalone replica bindings")
        self._bindings = topology["bindings"]
        self._roles = {row["engine_url"].rstrip("/"): row["role"] for row in self._bindings}
        if (sum(role == "hybrid" for role in self._roles.values()) != 2
                or sum(role == "standalone" for role in self._roles.values()) != 2):
            raise ValueError("elastic driver requires two actual hybrid and two standalone TP2 engines")
        self.result: dict[str, Any] = {
            "status": "NOT_RUN",
            "selected_uid": None,
            "source_worker_index": None,
            "target_worker_index": None,
            "handoff_id": None,
            "snapshot_observed_tokens_before_request": None,
            "recovery_receipts": [],
            "published_member_keys": [],
            "consumed_member_keys": [],
            "native_training_uid_consumed": False,
            "native_entry_update_and_publication_accepted": False,
            "dynamic_gpu_resource_reallocation_verified": False,
            "actual_gpu_reuse_verified": False,
            "elastic": {"controller_events": [], "source_process_observations": [],
                        "target_process_observations": [], "sleep_observations": [],
                        "native_compute_rows": [], "native_optimizer_audit_rows": [], "observation_errors": []},
        }

    @classmethod
    def install_on_trainer(
        cls, trainer: Any, *, options: DriverOptions | None = None, actual_topology: dict | None = None
    ) -> "NativeJointHandoffDriver":
        """Install the controller without replacing stock fit/update logic."""
        manager = getattr(trainer, "agent_loop_manager", None)
        if manager is None:
            raise ValueError("install after the original AgentLoopManagerTQ is created")
        driver = cls(manager, trainer, options=options, actual_topology=actual_topology)
        driver._install_manager_wrapper()
        driver._install_training_observer()
        driver._install_elastic_observers()
        from native_grpo_group_metrics import install_group_metrics
        driver._group_metrics = install_group_metrics(driver)
        try:
            from native_grpo_group_metrics import install_routing_window_metrics
            driver._routing_metrics = install_routing_window_metrics(driver)
        except Exception as error:
            driver._routing_metrics = None
            driver.result["routing_metrics"] = {"status": "NOT_RUN", "stage": "routing_window_install",
                "error_type": type(error).__name__, "native_result_overridden": False}
        return driver

    def _install_manager_wrapper(self) -> None:
        if getattr(self.manager, "_native_joint_handoff_driver", None) is not None:
            raise ValueError("the manager already has a native joint handoff driver")
        original = self.manager.generate_sequences
        if asyncio.iscoroutinefunction(original):
            raise ValueError("native AgentLoopManagerTQ.generate_sequences must remain synchronous")
        def generate_sequences_with_controller(prompts):
            with self._dispatch_lock:
                if self._monitor_future is None:
                    from verl.trainer.ppo.v1.trajectory_control_dispatch import _dispatcher
                    allowed = self._prompt_uids(prompts)
                    # Read identities before any original business generation.
                    _dispatcher(self.manager).submit(self._prepare_identity_cache()).result(
                        timeout=min(180, self.options.control_timeout_s))
                    self._monitor_deadline = time.monotonic() + self.options.control_timeout_s
                    self._monitor_future = _dispatcher(self.manager).submit(self._monitor(allowed))
            self.raise_if_control_failed()
            result = original(prompts)
            self.raise_if_control_failed()
            return result
        self.manager._before_trajectory_handoff_resume = self._require_physical_before_resume
        self.manager.generate_sequences = generate_sequences_with_controller
        self.manager._native_joint_handoff_driver = self
        self.result["status"] = "INSTALLED"

    def raise_if_control_failed(self):
        future = self._monitor_future
        if future is not None and future.done():
            future.result()  # Preserve the actual controller cause in native fit.


    def _install_training_observer(self) -> None:
        trainer = self.trainer
        if trainer is None or getattr(trainer, "_native_joint_handoff_driver", None) is not None:
            raise ValueError("install the training observer exactly once")
        original = trainer._update_actor
        if asyncio.iscoroutinefunction(original):
            raise ValueError("the original native actor update must remain synchronous")
        def update_actor_and_observe_keys(batch, metrics):
            self.raise_if_control_failed()
            keys = list(batch.keys)
            self._record_consumed_keys(keys, returned=False)
            self._elastic_event("native_actor_update_dispatched", trainer_step=trainer.global_steps)
            try:
                result = original(batch, metrics)
            except BaseException:
                self.result["native_training_update_returned"] = False
                self._persist_observation()
                raise
            self._observe_native_compute()
            from joint_gpu_return_observer import capture
            capture(self, "after_train")
            self._record_consumed_keys(keys, returned=True)
            self.raise_if_control_failed()
            return result
        trainer._update_actor = update_actor_and_observe_keys
        replay = trainer.replay_buffer
        sync_metadata = replay._sync_metadata_from_transfer_queue
        def checked_metadata_sync(*args, **kwargs):
            self.raise_if_control_failed()
            result = sync_metadata(*args, **kwargs)
            self.raise_if_control_failed()
            return result
        replay._sync_metadata_from_transfer_queue = checked_metadata_sync
        trainer._native_joint_handoff_driver = self

    def _observation_error(self, stage, error):
        self.result["elastic"]["observation_errors"].append(
            {"stage": stage, "error_type": type(error).__name__, "message": str(error)[:240]})

    def _elastic_event(self, event, **fields):
        events = self.result["elastic"]["controller_events"]
        events.append({"event": event, "sequence": len(events), "controller_pid": os.getpid(),
                       "monotonic_ns": time.monotonic_ns(), "observed_unix_s": time.time(), **fields})
        self._persist_observation()

    def _persist_observation(self):
        with self._receipt_lock:
            self._persist_observation_locked()

    def _persist_observation_locked(self):
        if self.options.result_path is not None:
            try:
                path = Path(self.options.result_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.result, ensure_ascii=False, allow_nan=False) + "\n")
                os.replace(temporary, path)
            except Exception as error:
                self._observation_error("driver_receipt_io", error)

    def _worker_rpc(self, function, *args):
        import ray
        return ray.get(self.trainer.actor_rollout_wg.execute_all_async(
            "execute_with_func_generator", function, *args), timeout=self.options.control_timeout_s)

    def _server_handles(self, role):
        manager = self.trainer.llm_server_manager if role == "hybrid" else self.trainer.standalone_server_manager
        result = []
        for url, replica in zip(manager.server_addresses, manager.rollout_replicas, strict=True):
            if not url.startswith(("http://", "https://")):
                url = "http://" + url
            for server in replica.servers:
                result.append((url, server))
        return result

    async def _observe_processes_before_switch(self):
        from elastic_gpu_observer import source_scheduler_observations
        from verl.trainer.ppo.v1.trajectory_state.token_kv_resume import PortableSGLangHTTPBackend
        # Ray owner queries enable the existing sleep-return observation only.
        # Physical UUIDs are read inside each actual CUDA scheduler, never here.
        for role, key in (("hybrid", "source_process_observations"),
                          ("standalone", "target_process_observations")):
            try:
                handles = self._server_handles(role)
                owners = await asyncio.gather(*[server.get_elastic_resource_observation.remote()
                                                for _, server in handles])
                self.result["elastic"][role + "_owner_observations"] = owners
                rows = []
                for url in sorted(url for url, actual_role in self._roles.items() if actual_role == role):
                    observed = self._actual_inspects.get(url)
                    if observed is None:
                        observed = await PortableSGLangHTTPBackend(url,
                            timeout_s=min(30, self.options.control_timeout_s)).inspect()
                    if str(observed.get("identity", {}).get("actualweight_version")) != self.result["selected_actual_identity"]["actual_weight_version"]:
                        raise ValueError("actual scheduler identity version differs across the four engines")
                    rows.extend(source_scheduler_observations(observed, url))
                self.result["elastic"][key] = rows
            except Exception as error:
                self._observation_error("read_" + role + "_scheduler_identity", error)
        self._persist_observation()

    async def _prepare_identity_cache(self):
        from verl.trainer.ppo.v1.trajectory_state.token_kv_resume import PortableSGLangHTTPBackend
        urls = sorted(self._roles)
        rows = await asyncio.gather(*[PortableSGLangHTTPBackend(url, timeout_s=30).inspect() for url in urls])
        self._actual_inspects = dict(zip(urls, rows, strict=True))
        versions = set()
        for row in rows:
            identity = row.get("identity", {})
            version = identity.get("actualweight_version")
            if identity.get("tp_size") != 2 or version is None or str(version) != str(identity.get("weight_version")):
                raise ValueError("Pre-business engine identity must prove coherent actual TP2 version")
            versions.add(str(version))
        if len(versions) != 1:
            raise ValueError("Pre-business engines have different actual versions")
        diagnostic_keys = ("weight_version", "actualweight_version", "tp_size", "model_config_sha256",
                           "cache_layout_sha256", "engine_source_sha256", "actual_model_config_file_sha256")
        try:
            self.result["pre_business_cached_identities"] = [
                {"engine_url": url, "value_source": "original pre-business HTTP inspect response",
                 "fields": [{"field": key, "present": key in row["identity"],
                             "value": row["identity"].get(key), "value_type": type(row["identity"].get(key)).__name__}
                            for key in diagnostic_keys],
                 "pool_geometry": {key: row["identity"].get("cache_layout", {}).get(key)
                                   for key in ("allocator", "request_pool", "page_size", "full_attention_layer_ids", "mamba_layer_ids")},
                 "conv_geometry": [{"shape": item.get("shape"), "dtype": item.get("dtype")}
                                   for item in row["identity"].get("cache_layout", {}).get("conv", [])],
                 "temporal_geometry": {key: row["identity"].get("cache_layout", {}).get("temporal", {}).get(key)
                                       for key in ("shape", "dtype")},
                 "rank_instances": [rank.get("engine_instance_id") for rank in row.get("ranks", [])],
                 "rank_local_cache_layouts": [rank.get("local_cache_layout") for rank in row.get("ranks", [])]}
                for url, row in zip(urls, rows, strict=True)]
        except Exception as diagnostic_error:
            self.result["identity_diagnostic_error"] = {"error_type": type(diagnostic_error).__name__}
        self.result["selected_actual_identity"] = {"actual_weight_version": next(iter(versions))}
        await self._observe_processes_before_switch()
        for key in ("source_process_observations", "target_process_observations"):
            rows = self.result["elastic"].get(key, [])
            if len(rows) != 4 or any(row.get("status") != "OBSERVED" for row in rows):
                raise ValueError("All original scheduler process identities are required before business dispatch")
        self.result["pre_business_identity_cache_ready"] = True
        self._persist_observation()

    async def _record_source_preemption_branches(self, uid):
        """Retain bounded metadata through the existing phase-result export."""
        try:
            progress = await asyncio.wait_for(self.manager.trajectory_generation_progress(uid), timeout=10)
            observed = []
            for worker_index, groups in progress.items():
                for member_index, row in groups.get(uid, {}).items():
                    if not isinstance(row, dict):
                        continue
                    observed.append({"worker_index": int(worker_index), "member_index": int(member_index),
                        "trajectory_id": row.get("trajectory_id"), "native_request_id": row.get("native_request_id"),
                        "active": row.get("active"), "source_preemption_events": row.get("source_preemption_events", [])[-16:]})
            self.result["source_preemption_branch_observations"] = observed[:8]
        except Exception as error:
            self.result["source_preemption_branch_observation_error"] = {"error_type": type(error).__name__}
        self._persist_observation()

    async def _require_physical_before_resume(self, uid, handoff_id, source_index, target_index):
        if uid != self.result.get("selected_uid"):
            return  # Other original resource-reclaim handoffs keep their existing semantics.
        from joint_physical_admission import require_original_physical_handoff
        await self._record_source_preemption_branches(uid)
        try:
            proof = await require_original_physical_handoff(self.manager._trajectory_coordinator_client,
                handoff_id=handoff_id, uid=uid, expected=self.result["native_source_request_binding"])
            # After capture, fresh read-only checks cannot consume the active-stream window.
            from verl.trainer.ppo.v1.trajectory_state.token_kv_resume import PortableSGLangHTTPBackend
            for url, before in self._actual_inspects.items():
                after = await PortableSGLangHTTPBackend(url, timeout_s=30).inspect()
                instances = lambda row: {r.get("engine_instance_id") for r in row.get("ranks", [])}
                if after.get("identity") != before.get("identity") or instances(after) != instances(before):
                    raise ValueError("Actual engine/process changed since pre-business identity observation")
            self.result["physical_pre_target_admission"] = proof
        except BaseException as error:
            self.result["physical_pre_target_admission"] = {"status": "REJECTED_BEFORE_TARGET_OR_TRAINING",
                "uid": uid, "handoff_id": handoff_id, "error_type": type(error).__name__, "error": str(error)}
            self._persist_observation()
            raise
        self._persist_observation()

    async def _inspect_actual_selection(self, source_index, target_index):
        source = self._actual_inspects[self.result["source_engine_url"].rstrip("/")]
        target = self._actual_inspects[self.result["target_engine_url"].rstrip("/")]
        owners = {index: owner for owner, index in self.manager.trajectory_worker_owner_ids.items()}
        binding = self.result["native_source_request_binding"]
        instances = {row.get("engine_instance_id") for row in source.get("ranks", [])}
        if (binding["source_owner_id"] != owners[source_index]
                or binding["source_engine_instance_id"] not in instances
                or str(binding["engine_weight_version"]) != str(source["identity"]["actualweight_version"])):
            raise ValueError("Live native request binding differs from pre-business source identity")
        identity_keys = ("weight_version", "actualweight_version", "tp_size", "model_config_sha256",
                         "cache_layout_sha256", "engine_source_sha256", "actual_model_config_file_sha256")
        self.result["cached_identity_comparison"] = {
            "value_source": "selected entries of the original pre-business HTTP inspect cache",
            "source_url": self.result["source_engine_url"], "target_url": self.result["target_engine_url"],
            "fields": [{"field": key, "source_present": key in source["identity"],
                        "target_present": key in target["identity"],
                        "source_value": source["identity"].get(key), "target_value": target["identity"].get(key),
                        "source_type": type(source["identity"].get(key)).__name__,
                        "target_type": type(target["identity"].get(key)).__name__,
                        "accepted_by_existing_gate": source["identity"].get(key) is not None
                            and source["identity"].get(key) == target["identity"].get(key)}
                       for key in identity_keys]}
        if any(source["identity"].get(key) is None or source["identity"].get(key) != target["identity"].get(key) for key in identity_keys):
            raise ValueError("Cached source/target model identity differs")
        self._elastic_event("selected_same_version_handoff", actual_weight_version=str(source["identity"]["actualweight_version"]))
        self.result["selected_actual_identity"] = {"source_owner_id": owners[source_index],
            "target_owner_id": owners[target_index], "source_role": "hybrid", "target_role": "standalone",
            "actual_weight_version": str(source["identity"]["actualweight_version"]),
            "source_inspect": source, "target_inspect": target}


    def _install_elastic_observers(self):
        # All original methods execute exactly once and keep their result/cause.
        def wrap(obj, name, event, predicate=None, after=None, before=None):
            original = getattr(obj, name)
            def observed(*args, **kwargs):
                if before is not None:
                    try:
                        before()
                    except Exception as error:
                        self._observation_error("before_" + event, error)
                result = original(*args, **kwargs)
                try:
                    if predicate is None or predicate(result):
                        self._elastic_event(event, trainer_step=self.trainer.global_steps)
                        if after is not None:
                            after(result)
                except Exception as error:
                    self._observation_error(event, error)
                return result
            setattr(obj, name, observed)
        wrap(self.trainer, "remove_replicas_from_balancer", "routing_removed")
        wrap(self.trainer.checkpoint_manager, "stop_generation_routing", "routing_stopped")
        wrap(self.manager, "prepare_trajectory_resource_reclaim", "quiesced",
             lambda receipt: isinstance(receipt, dict) and receipt.get("status") == "quiesced")
        wrap(self.trainer.checkpoint_manager, "abort_replicas", "aborted")
        from joint_gpu_return_observer import capture
        def before_replica_sleep():
            self._elastic_event("replica_sleep_begin")
            capture(self, "before_sleep")
        wrap(self.trainer.checkpoint_manager, "sleep_replicas", "sleep_completed",
             after=self._observe_sleep, before=before_replica_sleep)
        wrap(self.trainer, "switch_to_trainer", "switch_to_trainer_returned",
             before=lambda: self._elastic_event("switch_to_trainer_begin"))
        try:
            from elastic_gpu_observer import install_native_compute_observer
            directory = (Path(self.options.result_path).parent if self.options.result_path is not None
                         else Path(os.environ["VERL_TRAJECTORY_RECEIPT_DIR"])) / "elastic-native-compute"
            self.result["elastic"]["native_compute_install"] = self._worker_rpc(
                install_native_compute_observer, str(directory))
        except Exception as error:
            # An identity observer failure does not alter native training outcome.
            self._observation_error("install_native_compute_observer", error)

    def _observe_sleep(self, unused_result):
        import ray
        handles = self._server_handles("hybrid")
        rows = ray.get([server.get_elastic_resource_observation.remote() for _, server in handles],
                       timeout=self.options.control_timeout_s)
        self.result["elastic"]["sleep_observations"] = [
            {"engine_url": url, **row} for (url, _), row in zip(handles, rows, strict=True)]
        from joint_gpu_return_observer import capture
        capture(self, "after_sleep")
        self._persist_observation()

    def _observe_native_compute(self):
        try:
            from elastic_gpu_observer import read_native_compute_observer
            from training_update_audit import read_worker_update_audit
            self.result["elastic"]["native_compute_rows"] = self._worker_rpc(read_native_compute_observer)
            self.result["elastic"]["native_optimizer_audit_rows"] = self._worker_rpc(read_worker_update_audit)
        except Exception as error:
            self._observation_error("read_native_compute", error)
        self._persist_observation()


    def _prompt_uids(self, prompts: Any) -> set[str]:
        extractor = getattr(self.manager, "_trajectory_prompt_uids", None)
        if extractor is None:
            raise ValueError("AgentLoopManagerTQ prompt UID extraction is unavailable")
        values = extractor(prompts)
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ValueError("the native batch did not expose valid prompt UIDs")
        if not values:
            raise ValueError("the native batch has no prompt groups")
        return set(values)

    def _control_wait_stage(self, stage, deadline):
        self.result["controller_wait"] = {
            "stage": stage, "started_monotonic_s": time.monotonic(),
            "remaining_total_budget_s": max(0.0, deadline - time.monotonic()),
            "worker_indices": [row["worker_index"] for row in self._bindings],
        }
        self._persist_observation()

    async def _await_control_stage(self, stage, awaitable, deadline):
        self._control_wait_stage(stage, deadline)
        return await asyncio.wait_for(awaitable, timeout=max(0.0, deadline-time.monotonic()))

    async def _read_generation_progress(self, deadline):
        self._control_wait_stage("generation_progress", deadline)
        # Keep one read-only RPC batch in flight. A slow owner must not turn a
        # five-second observation interval into a fatal control deadline.
        task = asyncio.ensure_future(self.manager.trajectory_generation_progress())
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("joint generation-progress total control budget exhausted")
                try:
                    return await asyncio.wait_for(asyncio.shield(task), timeout=min(5.0, remaining))
                except asyncio.TimeoutError:
                    if task.done():
                        return task.result()  # Preserve a real RPC error, including TimeoutError.
                    wait = self.result["controller_wait"]
                    wait["observation_intervals_elapsed"] = wait.get("observation_intervals_elapsed", 0) + 1
                    wait["elapsed_s"] = time.monotonic() - wait["started_monotonic_s"]
                    self._persist_observation()
        finally:
            if not task.done():
                task.cancel()

    async def _monitor(self, allowed_uids: set[str]) -> None:
        try:
            await self._monitor_once(allowed_uids)
        except BaseException as error:
            import traceback
            wait = self.result.get("controller_wait", {})
            self.result["controller_failure"] = {
                "stage": wait.get("stage", "select_source"),
                "worker_indices": wait.get("worker_indices", []),
                "elapsed_stage_s": max(0.0, time.monotonic()-wait.get("started_monotonic_s", time.monotonic())),
                "total_control_budget_s": self.options.control_timeout_s,
                "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(),
            }
            self._persist_observation()
            raise

    def _admit_preemption(self, deadline, *, stage):
        required = self.options.source_group_preemption_budget_s + self.options.target_completion_reserve_s
        remaining = max(0.0, deadline - time.monotonic())
        self.result["preemption_admission"] = {
            "stage": stage, "controller_monotonic_s": time.monotonic(),
            "remaining_controller_s": remaining, "required_remaining_s": required,
            "source_group_cap_s": self.options.source_group_preemption_budget_s,
            "target_completion_reserve_s": self.options.target_completion_reserve_s,
            "minimum_physical_kv_members": 1, "admitted": remaining >= required,
            "performance_improvement_claimed": False,
        }
        if remaining < required:
            self.result["status"] = "NOT_ADMITTED_INSUFFICIENT_BUDGET"
            self.result["preemption_admission"]["generation_or_control_retry"] = False
            self._persist_observation()
            return False
        self._persist_observation()
        return True

    async def _monitor_once(self, allowed_uids: set[str]) -> None:
        deadline = self._monitor_deadline or (time.monotonic() + self.options.control_timeout_s)
        while time.monotonic() < deadline:
            progress = await self._read_generation_progress(deadline)
            candidate = self._select_candidate(progress, allowed_uids)
            if candidate is not None:
                if not self._admit_preemption(deadline, stage="before_selection_inspect"):
                    return
                uid, worker_index, tokens, member_index, source_id, source_generation, source_engine, request_row = candidate
                self.result.update(
                    native_source_request_binding={"trajectory_id": source_id,
                        "source_execution_generation": source_generation, "member_index": member_index,
                        "source_owner_id": request_row["source_owner_id"], "source_lease_id": request_row["source_lease_id"],
                        "source_engine_instance_id": request_row["source_engine_instance_id"],
                        "request_id": request_row["native_request_id"], "engine_weight_version": request_row["engine_weight_version"]},
                    status="CANDIDATE_SELECTED_NOT_PREEMPTED",
                    selected_uid=uid,
                    source_worker_index=worker_index,
                    snapshot_observed_tokens_before_request=tokens,
                    source_member_index=member_index, source_trajectory_id=source_id,
                    source_execution_generation=source_generation, source_engine_url=source_engine,
                )
                targets = [int(index) for index, groups in progress.items()
                           if int(index) != worker_index and isinstance(groups, dict)
                           and groups.get("__admission_open__") is True
                           and isinstance(groups.get("__target_engine_url__"), str)
                           and groups["__target_engine_url__"] != source_engine
                           and self._roles.get(groups["__target_engine_url__"].rstrip("/")) == "standalone"]
                target = self.options.target_worker_index
                if target is None:
                    target = min(targets) if targets else None
                if target not in targets:
                    raise ValueError("joint KV handoff requires an available distinct actual target engine")
                self.result["target_engine_url"] = progress[target]["__target_engine_url__"]
                groups = progress.get(worker_index, progress.get(str(worker_index)))
                members = groups[uid]
                self.result["native_source_request_binding"]["active_requests"] = {
                    row["trajectory_id"]: {"trajectory_id": row["trajectory_id"],
                        "member_index": int(index), "source_execution_generation": row["execution_generation"],
                        "source_owner_id": row["source_owner_id"], "source_lease_id": row["source_lease_id"],
                        "source_engine_instance_id": row["source_engine_instance_id"],
                        "request_id": row["native_request_id"], "engine_weight_version": row["engine_weight_version"]}
                    for index, row in members.items()
                    if isinstance(row, dict) and row.get("active") is True
                    and type(row.get("completion_tokens")) is int and row["completion_tokens"] >= self.options.min_generated_tokens
                    and all(isinstance(row.get(key), str) and row[key] for key in
                        ("trajectory_id", "source_owner_id", "source_lease_id", "source_engine_instance_id", "native_request_id"))}
                await self._await_control_stage("inspect_source_and_target",
                    self._inspect_actual_selection(worker_index, target), deadline)
                # Recheck after inspection; never spend a source mutation on a
                # window which already lacks its complete fixed source reserve.
                if not self._admit_preemption(deadline, stage="before_preempt_rpc"):
                    return
                self.result["status"] = "PREEMPTION_REQUESTED"
                self._persist_observation()
                receipt = await self._await_control_stage("preempt_and_resume",
                    self.manager.preempt_and_resume_trajectory(
                        uid, target_worker_index=target,
                        timeout_s=max(0.001, deadline - time.monotonic()),
                    ), deadline)
                self._record_resume(receipt)
                return
            await asyncio.sleep(self.options.poll_interval_s)
        raise TimeoutError("no real source stream reached the required 18-token threshold")

    def _select_candidate(self, progress: Any, allowed_uids: set[str]) -> tuple[str, int, int] | None:
        if not isinstance(progress, dict):
            raise ValueError("trajectory generation progress RPC returned an invalid object")
        candidates = []
        for worker_index, groups in progress.items():
            if not isinstance(groups, dict):
                raise ValueError("worker generation progress is not a UID mapping")
            for uid, members in groups.items():
                if uid not in allowed_uids or not isinstance(members, dict):
                    continue
                active_tokens = [
                    row.get("completion_tokens") for row in members.values()
                    if isinstance(row, dict) and row.get("active") is True
                ]
                if active_tokens and all(type(value) is int for value in active_tokens):
                    tokens = max(active_tokens)
                    if tokens >= self.options.min_generated_tokens:
                        for index, row in members.items():
                            if (row.get("active") is True and type(row.get("completion_tokens")) is int
                                    and row["completion_tokens"] >= self.options.min_generated_tokens
                                    and isinstance(row.get("trajectory_id"), str)
                                    and type(row.get("execution_generation")) is int
                                    and isinstance(row.get("engine_url"), str)
                                    and all(isinstance(row.get(key), str) and row[key] for key in
                                        ("native_request_id", "source_owner_id", "source_lease_id", "source_engine_instance_id"))
                                    and row.get("engine_weight_version") is not None):
                                if self._roles.get(row["engine_url"].rstrip("/")) != "hybrid":
                                    continue
                                eligible = [int(target) for target, status in progress.items()
                                            if int(target) != int(worker_index) and isinstance(status, dict)
                                            and status.get("__admission_open__") is True
                                            and isinstance(status.get("__target_engine_url__"), str)
                                            and status["__target_engine_url__"] != row["engine_url"]
                                            and self._roles.get(status["__target_engine_url__"].rstrip("/")) == "standalone"]
                                if (self.options.target_worker_index is not None
                                        and self.options.target_worker_index not in eligible) or not eligible:
                                    continue
                                candidates.append((uid, int(worker_index), row["completion_tokens"], int(index),
                                                   row["trajectory_id"], row["execution_generation"], row["engine_url"], dict(row)))
        if not candidates:
            return None
        # The prompt order is already deterministic; progress order is not a
        # scheduling policy. Pick the first real qualifying stream in worker order.
        return min(candidates, key=lambda row: (row[1], row[0]))

    def _record_resume(self, receipt: Any) -> None:
        if not isinstance(receipt, dict) or receipt.get("status") != "resumed":
            raise ValueError("the existing handoff controller did not return a resumed result")
        published = receipt.get("receipt")
        if not isinstance(published, dict) or published.get("state") != "FINISHED":
            raise ValueError("the existing target publication path did not finish")
        binding = published.get("binding")
        keys = binding.get("keys") if isinstance(binding, dict) else None
        uid = self.result["selected_uid"]
        expected = [f"{uid}_{index}_0" for index in range(8)]
        if keys != expected:
            raise ValueError("the published handoff does not contain the exact original eight members")
        self.result.update(
            status="TARGET_PUBLISHED",
            handoff_id=receipt.get("handoff_id"),
            target_worker_index=receipt.get("target_worker_index"),
            published_member_keys=keys,
        )
        self.result["recovery_receipts"] = self._load_recovery_receipts(uid)
        self.result["member_lifecycle_receipt"] = self._load_member_lifecycle_receipt(uid)
        self._elastic_event("selected_handoff_target_published", uid=uid, handoff_id=self.result["handoff_id"])

    def _record_consumed_keys(self, keys: Any, *, returned: bool = False) -> None:
        uid = self.result["selected_uid"]
        actual = list(keys)
        expected = {f"{uid}_{index}_0" for index in range(8)} if isinstance(uid, str) and uid else set()
        matching = [key for key in actual if isinstance(key, str) and uid and key.startswith(uid + "_")]
        valid = bool(expected) and len(matching) == 8 and set(matching) == expected
        self.result["native_training_batch_keys_before_dispatch"] = actual
        self.result["native_training_batch_observation_source"] = "original _update_actor batch.keys"
        self.result["native_selected_uid_batch_gate"] = valid
        if not valid:
            self.result["native_training_uid_consumed"] = False
            self._persist_observation()
            raise ValueError("Selected original eight-member UID is absent or duplicated in actual native training batch")
        if returned:
            self.result["consumed_member_keys"] = matching
            self.result["native_training_uid_consumed"] = True
            self.result["native_training_update_returned"] = True
        self._persist_observation()



    def _load_recovery_receipts(self, uid: str) -> list[dict[str, Any]]:
        directory = os.environ.get("VERL_TRAJECTORY_RECEIPT_DIR")
        if not directory:
            return []
        rows = []
        for path in sorted(Path(directory).glob("recovery-*.json")):
            try:
                if path.stat().st_mtime_ns < self._started_ns:
                    continue
                row = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if (row.get("uid") == uid and row.get("mode") == "PORTABLE_STATE_RESTORED"
                    and row.get("handoff_id") == self.result["handoff_id"]):
                rows.append({"path":str(path), **row})
        return rows

    def _load_member_lifecycle_receipt(self, uid):
        directory = os.environ.get("VERL_TRAJECTORY_RECEIPT_DIR")
        rows = []
        for path in sorted(Path(directory).glob("member-lifecycle-*.json")) if directory else []:
            try:
                if path.stat().st_mtime_ns < self._started_ns:
                    continue
                row = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if row.get("uid") == uid and row.get("handoff_id") == self.result["handoff_id"]:
                rows.append({"path": str(path), **row})
        return rows[0] if len(rows) == 1 else None

    def _verify_original_member_lifecycle(self):
        """Require a durable result for each original member and real active restores."""
        receipt = self.result.get("member_lifecycle_receipt") or {}
        uid, handoff = self.result.get("selected_uid"), self.result.get("handoff_id")
        version = str(self.result.get("selected_actual_identity", {}).get("actual_weight_version"))
        source = receipt.get("source_policy_identity")
        rows, restores = receipt.get("members"), self.result.get("recovery_receipts", [])
        checks = {
            "exact_original_eight_member_records": (
                receipt.get("schema") == 1 and receipt.get("status") == "VERIFIED_DURABLE_MEMBER_RECORDS"
                and receipt.get("uid") == uid and receipt.get("handoff_id") == handoff
                and receipt.get("group_size") == 8 and isinstance(rows, list) and len(rows) == 8
                and all(isinstance(row, dict) and type(row.get("member_index")) is int for row in rows)
                and sorted(row["member_index"] for row in rows) == list(range(8))),
            "same_actual_policy_identity": (isinstance(source, dict)
                and set(source) == {"policy_version", "model_weights_fingerprint",
                                    "tokenizer_template_fingerprint", "kv_cache_schema_fingerprint"}
                and all(isinstance(v, str) and v for v in source.values())
                and source == receipt.get("target_policy_identity") and source.get("policy_version") == version),
        }
        if not checks["exact_original_eight_member_records"]:
            return {"verified": False, "checks": checks, "physical_members": [], "members": []}
        details, physical, matched = [], [], []
        for row in rows:
            index, category = row["member_index"], row.get("source_state")
            digest, generation = row.get("output_record_sha256"), row.get("output_execution_generation")
            valid = (row.get("trajectory_id") == f"{uid}/member-{index}"
                and isinstance(row.get("output_record_ref"), str) and bool(row["output_record_ref"])
                and isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
                and type(generation) is int and generation >= 0 and row.get("actual_policy_versions") == [version])
            if category == "checkpointed":
                source_generation = row.get("source_execution_generation")
                candidates = [r for r in restores if r.get("member_index") == index]
                checkpoint = row.get("checkpoint") or {}
                snapshot_digest = checkpoint.get("snapshot_sha256")
                valid = (valid and isinstance(row.get("input_record_ref"), str) and bool(row["input_record_ref"])
                    and isinstance(row.get("snapshot_ref"), str) and bool(row["snapshot_ref"])
                    and type(source_generation) is int and source_generation >= 0 and generation > source_generation
                    and isinstance(snapshot_digest, str) and len(snapshot_digest) == 64
                    and all(c in "0123456789abcdef" for c in snapshot_digest))
                if checkpoint.get("kind") == "TOKEN_KV":
                    valid = (valid and checkpoint.get("schema_version") == 2
                        and checkpoint.get("next_stage") == "generating_inflight"
                        and checkpoint.get("snapshot_id") == row.get("snapshot_id")
                        and row.get("recovery_mode") == "PORTABLE_STATE_RESTORED"
                        and isinstance(row.get("snapshot_id"), str) and bool(row["snapshot_id"])
                        and len(candidates) == 1)
                    if len(candidates) == 1:
                        r = candidates[0]
                        valid = (valid and r.get("uid") == uid and r.get("handoff_id") == handoff
                            and r.get("trajectory_id") == row["trajectory_id"]
                            and r.get("mode") == "PORTABLE_STATE_RESTORED"
                            and r.get("snapshot_id") == row["snapshot_id"]
                            and r.get("source_execution_generation") == source_generation
                            and type(r.get("source_execution_generation")) is int
                            and r.get("target_execution_generation") == generation
                            and type(r.get("target_execution_generation")) is int
                            and isinstance(r.get("ranks"), list) and bool(r["ranks"]))
                        matched.append(index)
                    if valid:
                        physical.append(index)
                elif checkpoint.get("kind") == "TOOL_BOUNDARY":
                    valid = (valid and checkpoint.get("schema_version") == 1
                        and checkpoint.get("next_stage") == "generating" and checkpoint.get("kv_ref_absent") is True
                        and type(checkpoint.get("completed_tool_batches")) is int and checkpoint["completed_tool_batches"] > 0
                        and type(checkpoint.get("retained_generated_tokens")) is int and checkpoint["retained_generated_tokens"] > 0
                        and row.get("recovery_mode") is None and row.get("snapshot_id") is None and not candidates)
                else:
                    valid = False
            elif category in {"completed", "pending"}:
                valid = (valid and row.get("snapshot_ref") is None and row.get("recovery_mode") is None
                    and (category == "completed" or
                         isinstance(row.get("input_record_ref"), str) and bool(row["input_record_ref"])))
            else:
                valid = False  # Aborted, failed and missing members cannot become completed.
            details.append({"member_index": index, "source_state": category,
                            "checkpoint_kind": (row.get("checkpoint") or {}).get("kind"), "verified": bool(valid)})
        selected = self.result.get("source_member_index")
        checks.update({"every_original_member_has_matching_result_or_restore": all(r["verified"] for r in details),
                       "at_least_one_real_portable_member": bool(physical),
                       "no_unmapped_or_duplicate_restore": len(matched) == len(restores),
                       "original_selected_active_member_retained": any(
                           row["member_index"] == selected and row.get("source_state") in {"completed", "checkpointed"}
                           and type(self.result.get("source_execution_generation")) is int
                           and row.get("output_execution_generation" if row["source_state"] == "completed"
                                       else "source_execution_generation") == self.result["source_execution_generation"]
                           for row in rows)})
        return {"verified": all(checks.values()), "checks": checks,
                "physical_members": physical, "members": details}


    def finalize(self, native_entry_report: Any) -> dict[str, Any]:
        """Fold in acceptance from the original native entry; never replace it."""
        self.result["group_metrics"] = self._group_metrics.finalize()
        try:
            if getattr(self, "_routing_metrics", None) is not None:
                self.result["routing_metrics"] = self._routing_metrics.report()
        except Exception as error:
            self.result["routing_metrics"] = {"status": "NOT_RUN", "stage": "routing_window_report",
                "error_type": type(error).__name__, "native_result_overridden": False}
        if self._monitor_future is None:
            raise ValueError("the native joint controller was never dispatched")
        self._monitor_future.result(timeout=max(0.01, self._monitor_deadline-time.monotonic()))
        if not isinstance(native_entry_report, dict):
            raise ValueError("the original native entry report is required")
        status = native_entry_report.get("status")
        updates = native_entry_report.get("verified_optimizer_updates_this_process")
        parameter = native_entry_report.get("parameter_sample_proof")
        accepted = (
            status == "PASS_NATIVE_UPDATES_AND_PUBLICATIONS"
            and type(updates) is int and updates >= 1
            and isinstance(parameter, dict)
            and parameter.get("observed_parameter_change") is True
        )
        self.result.update(
            native_entry_status=status,
            native_entry_verified_optimizer_updates=updates,
            native_entry_parameter_change=parameter if isinstance(parameter, dict) else None,
            native_entry_update_and_publication_accepted=accepted,
        )
        try:
            from elastic_gpu_observer import verify_elastic_reuse
            elastic = self.result["elastic"]
            proof = verify_elastic_reuse(source_observations=elastic["source_process_observations"],
                target_observations=elastic["target_process_observations"], sleep_observations=elastic["sleep_observations"],
                compute_rows=elastic["native_compute_rows"], native_audit_rows=elastic["native_optimizer_audit_rows"],
                controller_events=elastic["controller_events"])
            selected = self.result.get("selected_actual_identity", {})
            ranks = selected.get("source_inspect", {}).get("ranks", [])
            observed = {(row.get("hostname"), row.get("pid"))
                        for row in elastic["source_process_observations"]
                        if row.get("identity_source") == "scheduler_cuda_current_device"
                        and row.get("engine_url", "").rstrip("/") == self.result.get("source_engine_url", "").rstrip("/")}
            source_processes = set()
            for row in ranks:
                instance = row.get("engine_instance_id", "")
                host, _, pid = instance.rpartition(":")
                if host and pid.isdigit():
                    source_processes.add((host, int(pid)))
            proof["checks"]["selected_real_source_tp_processes_joined"] = (
                len(source_processes) == 2 and source_processes.issubset(observed))
            proof["verified"] = all(proof["checks"].values())
            self.result["elastic_reuse_proof"] = proof
            self.result["dynamic_gpu_resource_reallocation_verified"] = proof["verified"] and not elastic["observation_errors"]
        except Exception as error:
            self._observation_error("verify_elastic_reuse", error)
            self.result["elastic_reuse_proof"] = {"verified": False, "error_type": type(error).__name__}
            self.result["dynamic_gpu_resource_reallocation_verified"] = False
        self.result["actual_gpu_reuse_verified"] = self.result["dynamic_gpu_resource_reallocation_verified"]
        from elastic_gpu_observer import verify_portable_handoff, native_version1_witness_passed
        portable = verify_portable_handoff(self.result["recovery_receipts"],
            selected=self.result.get("selected_actual_identity", {}))
        self.result["portable_state_handoff_proof"] = portable
        lifecycle = self._verify_original_member_lifecycle()
        self.result["original_member_lifecycle_proof"] = lifecycle
        self.result["native_version1_token_witness"] = native_entry_report.get("version1_token_witness")
        checks = {
            "same_version_42_tensor_KV_GDN_and_other_target_requests": portable["verified"],
            "actual_native_version1_token_generated": native_version1_witness_passed(native_entry_report),
            "snapshot_after_18_real_tokens": (
                type(self.result["snapshot_observed_tokens_before_request"]) is int
                and self.result["snapshot_observed_tokens_before_request"] >= 18
            ),
            "actual_kv_recovery": bool(self.result["recovery_receipts"]),
            "every_original_member_completed_or_restored": lifecycle["verified"],
            "at_least_one_active_member_native_physical_KV_restored": (
                portable["verified"] and bool(lifecycle["physical_members"])),
            "complete_eight_member_delivery": self.result["published_member_keys"] == [
                f"{self.result['selected_uid']}_{index}_0" for index in range(8)
            ],
            "interrupted_uid_in_native_training_batch": self.result["native_training_uid_consumed"],
            "original_entry_update_and_publication": accepted,
            "actual_hybrid_gpu_reused_by_native_trainer": self.result["dynamic_gpu_resource_reallocation_verified"],
        }
        try:
            from joint_gpu_return_observer import finalize as gpu_return_finalize
            self.result["gpu_return_verdict"] = gpu_return_finalize(self, native_entry_report)
            checks["native_replica_memory_return_and_real_trainer_reuse"] = self.result["gpu_return_verdict"]["verified"]
        except Exception as error:
            self._observation_error("gpu_return_verdict", error)
            checks["native_replica_memory_return_and_real_trainer_reuse"] = False
        self.result["acceptance_checks"] = checks
        self.result["status"] = (
            "PASS_NATIVE_ELASTIC_HYBRID_TOKEN_KV_HANDOFF" if all(checks.values()) else "NOT_PASS_MISSING_EVIDENCE"
        )
        try:
            from physical_restore_metrics import physical_restore_summary
            self.result.setdefault("routing_metrics", {}).update(
                physical_restore_summary(self.result["recovery_receipts"]))
        except Exception as error:
            self.result.setdefault("routing_metrics", {}).update(
                physical_restore_actual_recomputed_tokens=None,
                physical_restore_compute_missing_reason=type(error).__name__,
                all_path_actual_recomputed_tokens=None)
        self._persist_observation()
        return deepcopy_result(self.result)


def deepcopy_result(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))

