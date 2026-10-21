"""Optional two-phase hook around an owner-initialized native training run.

prepare() and after_publication() never load an engine or publish weights.
Only their two stock CPU Gateway children belong to this object.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

from business_routing_metrics import summarize
from original_business_sampler import OriginalBusinessSampler, save


@dataclass
class Options:
    output: str
    business_contract: str
    business_auth: str
    train_parquet: str
    train_sha256: str
    smg_executable: str
    deadline_unix_s: float
    allocation_receipt: str
    allocated_gpu_count: int
    baseline_port: int = 32371
    cache_aware_port: int = 32372
    source_indices: tuple[int, ...] = tuple(range(8))
    require_version_transition: bool = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RoutingBusinessCompare:
    def __init__(self, trainer, options, read_publication):
        self.trainer, self.options, self.read_publication = trainer, options, read_publication
        self.root = Path(options.output)
        self.children = []
        self.closed = False
        self.cleanup_errors = []

    @classmethod
    def prepare(cls, trainer, *, options, read_publication):
        """Call after original native init + v0 publication, before native fit."""
        self = cls(trainer, options, read_publication)
        self.root.mkdir(parents=True, exist_ok=False)
        try:
            self.before = read_publication(trainer)
            asyncio.run(self._prepare())
            return self
        except BaseException as error:
            try:
                save(self.root / "result.json", {"status": "NOT_RUN", "error_type": type(error).__name__, "error": str(error)})
            except BaseException:
                pass
            self.close()  # Best-effort, nonthrowing; retain the prepare cause.
            raise

    def remaining(self):
        value = self.options.deadline_unix_s - time.time()
        if value < 35:
            raise TimeoutError("Owner deadline cleanup reserve reached; no additional window")
        return value - 30

    def policy(self, publication):
        from verl.trainer.ppo.v1.trajectory_state.identity import _decode_policy_identity
        version = publication.get("actual_parameter_version")
        identity = publication.get("active_policy_record", {}).get("identity")
        decoded = _decode_policy_identity(identity)
        if type(version) is not int or decoded.policy_version != str(version):
            raise ValueError("Actual owner publication and policy identity disagree")
        return version, decoded.digest

    def client(self, arm, urls=None):
        from verl.workers.rollout.sglang_gateway_client import GatewayClientConfig, SGLangGatewayClient
        rollout = self.trainer.config.actor_rollout_ref.rollout
        return SGLangGatewayClient(GatewayClientConfig(base_url=self.gateway_urls[arm],
            expected_worker_urls=tuple(urls or self.urls), max_model_len=rollout.max_model_len,
            prompt_length=rollout.prompt_length, response_length=rollout.response_length,
            request_timeout_s=min(300, self.remaining()), control_timeout_s=30, admission_timeout_s=30))

    async def registry(self, client, version):
        if await client.get_active_weight_version(client.config.expected_worker_urls) != version:
            raise ValueError("Actual engine version changed or differs across replicas")
        rows = await client.list_workers()
        return {row["id"]: row["url"].rstrip("/") for row in rows}

    def launch(self, arm, urls, port):
        executable = Path(self.options.smg_executable).resolve(strict=True)
        raw = executable.read_bytes()
        first = raw.split(b"\n", 1)[0]
        command = [str(executable)] if raw[:4] == b"\x7fELF" or (first.startswith(b"#!") and b"python" not in first.lower()) else [sys.executable, str(executable)]
        argv = command + ["launch", "--host", "0.0.0.0", "--port", str(port), "--policy",
                          "least_load" if arm == "baseline" else "cache_aware",
                          "--enable-rl", "--disable-retries", "--worker-urls", *urls]
        log = (self.root / (arm + "-gateway.log")).open("xb")
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        row = {"arm": arm, "pid": process.pid, "argv": argv,
               "executable_sha256": hashlib.sha256(raw).hexdigest(), "started_unix_s": time.time()}
        self.children.append((process, log, row))
        save(self.root / "owned-cpu-gateways.json", [x[2] for x in self.children])

    async def wait_registry(self, client, version):
        end = time.monotonic() + min(35, self.remaining())
        while time.monotonic() < end:
            if any(process.poll() is not None for process, _, _ in self.children):
                raise RuntimeError("Owned CPU Gateway exited")
            try:
                return await self.registry(client, version)
            except (OSError, ValueError, RuntimeError):
                await asyncio.sleep(.25)
        raise TimeoutError("Owned stock Gateway did not become ready")

    async def inspect_engines(self, expected_version):
        from joint_topology import _get, _find
        from verl.trainer.ppo.v1.trajectory_state.token_kv_resume import PortableSGLangHTTPBackend
        result = []
        for replica, url in zip(self.trainer.standalone_server_manager.rollout_replicas, self.urls):
            if replica.world_size != 2 or replica.config.tensor_model_parallel_size != 2:
                raise ValueError("Exactly two existing true TP2 replicas are required")
            info = await asyncio.to_thread(_get, url, "/get_server_info")
            for key, expected in (("tp_size", 2), ("disable_radix_cache", False), ("disable_overlap_schedule", False)):
                values = list(_find(info, key))
                if not values or any(x != expected for x in values):
                    raise ValueError("Actual engine setting differs: " + key)
            values = list(_find(info, "max_running_requests"))
            if not values or any(type(x) is not int or x < 4 for x in values):
                raise ValueError("Actual max_running_requests must be at least four")
            observed = await PortableSGLangHTTPBackend(url, timeout_s=20).inspect()
            ranks = observed.get("ranks", [])
            if (len(ranks) != 2 or sorted(x.get("tp_rank") for x in ranks) != [0, 1]
                    or any(x.get("status") != "PASS" or any(str(x.get("identity", {}).get(key)) != str(expected_version)
                           for key in ("weight_version", "actualweight_version")) for x in ranks)):
                raise ValueError("Actual snapshot inspect does not prove both TP ranks hold the owner version")
            result.append({"engine_url": url, "server_info": info, "inspect": observed})
        return result

    async def _prepare(self):
        import pyarrow.parquet as pq
        from verl.workers.rollout.sglang_gateway_client import SGLangGatewayClient
        self.remaining()
        o = self.options
        if (o.baseline_port == o.cache_aware_port or not 1024 <= o.baseline_port <= 65535
                or not 1024 <= o.cache_aware_port <= 65535 or type(o.allocated_gpu_count) is not int):
            raise ValueError("Two distinct owned CPU ports and the actual integer allocation are required")
        if o.allocated_gpu_count < 4 or not Path(o.allocation_receipt).is_file():
            raise ValueError("Owner's actual allocation receipt and whole-run GPU count are required")
        allocation = json.loads(Path(o.allocation_receipt).read_text())
        if (allocation.get("workers") != 4 or allocation.get("gpus_per_worker") != 2
                or o.allocated_gpu_count != 8 or not allocation.get("app_id")):
            raise ValueError("Existing owner binding receipt must prove four allocated workers x two GPUs")
        rollout = self.trainer.config.actor_rollout_ref.rollout
        if self.trainer.config.data.train_batch_size != 8 or rollout.n != 8 or rollout.calculate_log_probs is not True:
            raise ValueError("The unchanged original eight-by-eight real-logprob business configuration is required")
        if digest(o.train_parquet) != o.train_sha256:
            raise ValueError("Actual original training parquet differs")
        table = pq.read_table(o.train_parquet)
        if table.num_rows != 7388 or len(o.source_indices) != 8 or len(set(o.source_indices)) != 8:
            raise ValueError("Keep the original 7388 rows and freeze exactly eight source indices")
        all_rows = table.to_pylist()
        self.rows = [all_rows[i] for i in o.source_indices]
        addresses = self.trainer.standalone_server_manager.server_addresses
        self.urls = [x.rstrip("/") if x.startswith(("http://", "https://")) else "http://" + x for x in addresses]
        if len(self.urls) != 2 or len(set(self.urls)) != 2:
            raise ValueError("Owner manager must expose two distinct actual engines")
        self.version0, self.identity0 = self.policy(self.before)
        self.initial_engines = await self.inspect_engines(self.version0)
        if len({x["inspect"]["engine_instance_id"] for x in self.initial_engines}) != 2:
            raise ValueError("The two URLs do not prove distinct actual engine instances")
        sampler = OriginalBusinessSampler(self.trainer, contract_path=o.business_contract, auth_path=o.business_auth,
            rows=self.rows, policy_record=self.before, output=self.root)
        prefixes = await sampler.initial_prefixes()
        self.prefix = os.path.commonprefix(prefixes)
        if len(self.prefix) < 6000 or any(not 6000 <= len(x) < 8000 for x in prefixes):
            raise ValueError("Actual original business inputs do not satisfy the frozen 6–7K common-prefix workload")
        self.gateway_urls = {"baseline": f"http://127.0.0.1:{o.baseline_port}",
                             "cache_aware": f"http://127.0.0.1:{o.cache_aware_port}"}
        self.launch("cache_aware", self.urls[:1], o.cache_aware_port)
        one = self.client("cache_aware", self.urls[:1])
        try:
            await self.wait_registry(one, self.version0)
            salt = one.build_policy_cache_salt(run_namespace=str(self.root), policy_version=self.version0,
                                             policy_identity_salt=self.identity0)
            warm = await one.generate("routing-v0-prefix-warm", prompt_ids=self.prefix,
                sampling_params={"max_new_tokens": 1, "temperature": 0., "logprobs": True}, cache_salt=salt)
            save(self.root / "v0-router-prefix-warm.json", warm.model_dump())
        finally:
            await one.close()
        full = self.client("cache_aware")
        try:
            await full.register_worker({"url": self.urls[1], "worker_type": "regular",
                "connection_mode": "http", "runtime_type": "sglang"}, expected_weight_version=self.version0)
            self.cache_registry = await self.registry(full, self.version0)
        finally:
            await full.close()
        self.launch("baseline", self.urls, o.baseline_port)
        baseline = self.client("baseline")
        try:
            self.base_registry = await self.wait_registry(baseline, self.version0)
        finally:
            await baseline.close()
        save(self.root / "prepared.json", {"status": "PREPARED_NOT_MEASURED", "actual_publication": self.before,
            "actual_engines": self.initial_engines, "original_train_sha256": o.train_sha256,
            "source_indices": o.source_indices, "initial_prompt_lengths": [len(x) for x in prefixes],
            "common_prefix_tokens": len(self.prefix), "common_prefix_sha256": hashlib.sha256(json.dumps(self.prefix).encode()).hexdigest(),
            "allocation_receipt_sha256": digest(o.allocation_receipt), "allocated_gpu_count": o.allocated_gpu_count,
            "cache_registry": self.cache_registry, "baseline_registry": self.base_registry,
            "router_tree_internal_read_interface_available": False, "token_equality_prewaived": True})

    def after_publication(self, *, cache_reset_receipts=None):
        """Call from the owner after its unchanged native version publication.

        cache_reset_receipts maps each engine URL to an existing original
        update_weight_version JSON response file; this method never calls it.
        """
        try:
            self.after = self.read_publication(self.trainer)
            return asyncio.run(self._after(cache_reset_receipts or {}))
        except BaseException as error:
            result = {"status": "NOT_ACCEPTED", "error_type": type(error).__name__, "error": str(error),
                      "additional_windows_allowed": False}
            save(self.root / "result.json", result)
            raise
        finally:
            self.close()

    async def _after(self, reset_receipts):
        from verl.workers.rollout.sglang_gateway_client import SGLangGatewayClient
        version, identity = self.policy(self.after)
        if self.options.require_version_transition and version != self.version0 + 1:
            raise ValueError("Version-transition comparison requires the owner's actual next native version")
        after_engines = await self.inspect_engines(version)
        if any(a["inspect"]["engine_instance_id"] != b["inspect"]["engine_instance_id"]
               for a, b in zip(self.initial_engines, after_engines)):
            raise ValueError("An engine was reloaded; comparison no longer reuses the same replicas")
        reset_proofs = []
        if self.options.require_version_transition:
            for url in self.urls:
                path = reset_receipts.get(url)
                if path is None:
                    reset_proofs.append({"engine_url": url, "status": "NOT_RUN",
                        "reason": "OWNER_NATIVE_CACHE_RESET_RESPONSE_MISSING"})
                    continue
                value = json.loads(Path(path).read_text())
                if value.get("success") is not True or str(value.get("new_version")) != str(version):
                    raise ValueError("Owner reset receipt is not the actual successful version publication response")
                reset_proofs.append({"engine_url": url, "status": "OBSERVED_NATIVE_SUCCESS_RESPONSE",
                    "path": str(path), "sha256": digest(path), "original_response": value})
        manager = self.trainer.agent_loop_manager
        if getattr(manager, "_trajectory_checkpoint_paused", False):
            raise ValueError("Another owner already paused trajectory admission")
        clients = {arm: self.client(arm) for arm in self.gateway_urls}
        pause_owned = False
        try:
            pause_owned = True
            drained = await manager.prepare_trajectory_checkpoint(timeout_s=min(300, self.remaining()))
            save(self.root / "original-manager-drain-receipt.json", drained)
            if any(process.poll() is not None for process, _, _ in self.children):
                raise ValueError("A comparison Gateway PID changed")
            registries = {arm: await self.registry(client, version) for arm, client in clients.items()}
            if registries["cache_aware"] != self.cache_registry or registries["baseline"] != self.base_registry:
                raise ValueError("Worker registrations changed; old router state fixture was reset")
            salt = SGLangGatewayClient.build_policy_cache_salt(run_namespace=str(self.root), policy_version=version,
                                                             policy_identity_salt=identity)
            stale_salt = SGLangGatewayClient.build_policy_cache_salt(run_namespace=str(self.root),
                policy_version=self.version0, policy_identity_salt=self.identity0)
            rejected = False
            if version != self.version0:
                try: SGLangGatewayClient._compose_cache_salt(version, stale_salt)
                except Exception: rejected = True
                if not rejected: raise ValueError("Existing client did not reject the old version cache namespace")
            save(self.root / "version-transition.json", {"actual_publication": self.after,
                "actual_engine_inspects": after_engines, "same_gateway_pids_and_workers": True,
                "original_reset_responses": reset_proofs,
                "cache_reset_condition_status": "OBSERVED_NATIVE_SUCCESS_RESPONSE" if reset_proofs
                    and all(x["status"] == "OBSERVED_NATIVE_SUCCESS_RESPONSE" for x in reset_proofs) else "NOT_RUN",
                "cache_reset_proof": "Pinned all-TP update_weight_version handler success semantics; no cache occupancy counter exposed",
                "direct_per_rank_flush_responses_exported": False,
                "old_cache_salt_rejected_before_inference": rejected,
                "router_tree_internal_retention_directly_observed": False,
                "fixture_boundary": "Same stock cache-aware process and worker IDs retain v0 routing history; no tree read API is available"})
            sampler = OriginalBusinessSampler(self.trainer, contract_path=self.options.business_contract,
                auth_path=self.options.business_auth, rows=self.rows, policy_record=self.after, output=self.root)
            windows = []
            for index, arm in enumerate(("baseline", "baseline", "cache_aware", "cache_aware")):
                self.remaining()
                # Flush only cold replica 1; warm only replica 0. Never flush both.
                cold = await asyncio.to_thread(self.flush_cold)
                warm = await self.warm_actual_engine(self.urls[0], version, salt, index)
                save(self.root / f"cache-setup-{index}.json", {"cold_flush": cold, "warm_output": warm,
                    "measured_window_includes_setup": False, "cold_engine_url": self.urls[1], "warm_engine_url": self.urls[0]})
                before = await self.registry(clients[arm], version)
                result = await sampler.window(clients[arm], arm=arm, index=index, version=version, salt=salt,
                    worker_map=before, allocated_gpu_count=self.options.allocated_gpu_count,
                    deadline=self.options.deadline_unix_s - 30)
                if await self.registry(clients[arm], version) != before:
                    result["status"] = "REGISTRY_CHANGED"
                windows.append(result)
                if result["status"] != "COMPLETE": break
            result = summarize(windows)
            result["version_fixture_internal_tree_observation"] = "NOT_DIRECTLY_AVAILABLE"
            result["routing_metric_observation_status"] = result["status"]
            result["cross_version_stale_tree_acceptance"] = (
                "NOT_ACCEPTED_INTERNAL_TREE_RETENTION_NOT_OBSERVABLE" if self.options.require_version_transition
                else "NOT_RUN_BY_EXPLICIT_OPTION")
            result["business_comparison_is_independent_of_internal_tree_readability"] = True
            save(self.root / "result.json", result)
            return result
        finally:
            for client in clients.values(): await client.close()
            if pause_owned:
                await manager.resume_trajectory_checkpoint_admission()

    def flush_cold(self):
        with urlopen(self.urls[1] + "/flush_cache", timeout=30) as response:
            text = response.read().decode()
            if response.status != 200 or not text.startswith("Cache flushed."):
                raise ValueError("Idle cold engine did not confirm the existing flush_cache operation")
            return {"engine_url": self.urls[1], "http_status": response.status, "original_body": text}

    async def warm_actual_engine(self, url, version, salt, index):
        import aiohttp
        from verl.workers.rollout.sglang_gateway_client import GatewayHTTPResponse, SGLangGatewayClient
        rid = f"routing-current-prefix-warm-{index}"
        request = {"rid": rid, "input_ids": self.prefix, "sampling_params": {"max_new_tokens": 1, "temperature": 0.},
                   "stream": False, "return_logprob": True, "cache_salt": salt}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=min(60, self.remaining()))) as session:
            async with session.post(url + "/generate", json=request) as response:
                body = await response.json()
                output = SGLangGatewayClient._decode_generation(GatewayHTTPResponse(response.status, body, dict(response.headers)),
                    return_logprob=True, limit=1, request_id=rid, observed_weight_version=version, observed_cache_salt=salt)
                if output.extra_fields["engine_weight_version"] != str(version):
                    raise ValueError("Warm request used an unexpected actual model version")
                return {"actual_request": request, "actual_response": body}

    def close(self):
        if self.closed: return
        cleanup = []
        for process, log, row in reversed(self.children):
            try:
                if process.poll() is None and os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
            except ProcessLookupError:
                process.poll()
            except BaseException as error:
                self.cleanup_errors.append({"pid": process.pid, "error_type": type(error).__name__, "error": str(error)})
            try:
                log.close()
                cleanup.append({"pid": process.pid, "returncode": process.poll(), "engine_signal_sent": False})
            except BaseException as error:
                self.cleanup_errors.append({"pid": process.pid, "error_type": type(error).__name__, "error": str(error)})
        self.closed = True
        try:
            save(self.root / "owned-cpu-cleanup.json", {"children": cleanup, "errors": self.cleanup_errors})
        except BaseException as error:
            self.cleanup_errors.append({"error_type": type(error).__name__, "error": str(error)})
