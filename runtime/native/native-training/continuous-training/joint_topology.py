"""Bind native hybrid/standalone replicas without moving the four trainer ranks."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import urllib.request


OVERRIDES = {
    "actor_rollout_ref.hybrid_engine": True,
    "trainer.v1.separate_async.hybrid_rollout.enable_switch": True,
    "actor_rollout_ref.rollout.trajectory_state.shared_gpu_handoff": True,
    # Fixed source drain and whole controller limits. These are failure/admission
    # ceilings, not measured speedups; the publication barrier uses the same 600s.
    "actor_rollout_ref.rollout.trajectory_state.safe_point_timeout_s": 600.0,
    "actor_rollout_ref.rollout.trajectory_state.control_timeout_s": 1800.0,
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 2,
    "actor_rollout_ref.rollout.data_parallel_size": 1,
    "actor_rollout_ref.rollout.pipeline_model_parallel_size": 1,
    "actor_rollout_ref.rollout.agent.num_workers": 4,
    # This is an owner index, never a device ordinal. bind_actual_replicas
    # replaces it from the actual role table before CPU workers are created.
    "actor_rollout_ref.rollout.trajectory_state.handoff_target_worker_index": 2,
    "actor_rollout_ref.rollout.max_num_seqs": 4,
    "actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests": 4,
    "actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache": False,
    "actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule": False,
    "actor_rollout_ref.rollout.free_cache_engine": True,
}


def override_config(base, *, output_root=None):
    value = copy.deepcopy(base)
    actor, trainer, data = value["actor_rollout_ref"], value["trainer"], value["data"]
    rollout = actor["rollout"]
    if (trainer["nnodes"], trainer["n_gpus_per_node"], rollout["nnodes"], rollout["n_gpus_per_node"]) != (4, 1, 4, 1):
        raise ValueError("Keep four fixed trainer ranks and four dedicated rollout GPUs")
    if (data["train_batch_size"], rollout["n"]) != (8, 8):
        raise ValueError("Keep the original eight groups of eight members")
    if data["shuffle"] is not True or data["seed"] != 42:
        raise ValueError("Original business input sampling changed")
    if trainer["v1"]["separate_async"]["parameter_sync_step"] != 1:
        raise ValueError("Native shared-GPU switching requires per-step publication")
    if rollout["sglang_gateway"]["enabled"] is not True:
        raise ValueError("The original native Gateway must arbitrate hybrid membership")
    for section, dynamic, micro in (
        (actor["actor"], "use_dynamic_bsz", "ppo_micro_batch_size_per_gpu"),
        (rollout, "log_prob_use_dynamic_bsz", "log_prob_micro_batch_size_per_gpu"),
        (actor["ref"], "log_prob_use_dynamic_bsz", "log_prob_micro_batch_size_per_gpu"),
    ):
        if section[dynamic] is not False or section[micro] != 1:
            raise ValueError("Preserve the existing original static micro-batch policy")
    for name, setting in OVERRIDES.items():
        node = value
        keys = name.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = setting
    if output_root is not None:
        root = Path(output_root).resolve()
        if not str(root).startswith("/mnt/shared/project-b"):
            raise ValueError("All cross-pod state must use this boot's shared CFS root")
        trainer["default_local_dir"] = str(root / "checkpoints")
        rollout["trajectory_state"]["root_dir"] = str(root / "trajectory-objects")
        value["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["VERL_TRAJECTORY_RECEIPT_DIR"] = str(root / "recovery-receipts")
    return value


def preflight(repo):
    """Static source presence only; this never establishes GPU acceptance."""
    repo = Path(repo)
    checks = {
        "verl/trainer/ppo/v1/trainer_separate_async.py": ["switch_to_trainer", "switch_to_rollout"],
        "verl/trainer/ppo/v1/agent_loop_tq.py": ["trajectory_generation_progress", "prepare_trajectory_resource_reclaim"],
        "verl/workers/rollout/replica.py": ["init_hybrid", "init_standalone"],
        "verl/workers/rollout/sglang_gateway_client.py": ["resolve_routed_worker"],
    }
    missing = []
    for relative, methods in checks.items():
        target = repo / relative
        names = {n.name for n in ast.walk(ast.parse(target.read_text())) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} if target.is_file() else set()
        missing.extend(relative + "::" + name for name in methods if name not in names)
    for name in ("token_kv_source.py", "token_kv_resume.py", "routed_token_kv_stream.py"):
        relative = "verl/trainer/ppo/v1/trajectory_state/" + name
        if not (repo / relative).is_file():
            missing.append(relative)
    return {"status": "BLOCKED_MISSING_NATIVE_INTERFACE" if missing else "SOURCE_PREFLIGHT_ONLY_PASS", "missing": missing}


def _get(url, path, *, headers=None):
    request = urllib.request.Request(url.rstrip("/") + path, headers=headers or {})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def _find(body, key):
    if isinstance(body, dict):
        if key in body:
            yield body[key]
        for child in body.values():
            yield from _find(child, key)
    elif isinstance(body, list):
        for child in body:
            yield from _find(child, key)


def _url(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Native manager returned an empty engine URL")
    return (value if value.startswith(("http://", "https://")) else "http://" + value).rstrip("/")


def _registry(config, expected):
    settings = config.actor_rollout_ref.rollout.sglang_gateway
    headers = {}
    if settings.api_key_env:
        headers["authorization"] = "Bearer " + os.environ[settings.api_key_env]
    body = _get(settings.url, "/workers", headers=headers)
    workers = body.get("workers") if isinstance(body, dict) else None
    if not isinstance(workers, list):
        raise ValueError("Original Gateway registry did not return worker records")
    result = {}
    seen_ids = set()
    for worker in workers:
        if not isinstance(worker, dict):
            raise ValueError("Invalid original Gateway worker record")
        url, identity = _url(worker.get("url")), worker.get("id")
        if (url in result or not isinstance(identity, str) or not identity or identity in seen_ids
                or worker.get("is_healthy") is not True):
            raise ValueError("Gateway worker identity or ready state is not confirmed")
        result[url] = worker
        seen_ids.add(identity)
    if set(result) != set(expected):
        raise ValueError("Actual Gateway membership differs from the two native replica pools")
    return result


def bind_actual_replicas(trainer, config, output):
    """Read actual initialized managers/registry before AgentLoopManagerTQ.create."""
    from omegaconf import open_dict

    pools = [("hybrid", trainer.llm_server_manager), ("standalone", trainer.standalone_server_manager)]
    allocations = []
    for role, manager in pools:
        urls = [_url(address) for address in manager.server_addresses]
        if len(urls) != 2 or len(manager.rollout_replicas) != 2 or len(set(urls)) != 2:
            raise ValueError("Each native pool must contain exactly two actual TP2 replicas")
        for replica, url in zip(manager.rollout_replicas, urls, strict=True):
            mode = getattr(replica.rollout_mode, "value", replica.rollout_mode)
            if (replica.world_size != 2 or replica.config.tensor_model_parallel_size != 2
                    or str(mode).lower() != role):
                raise ValueError("Native replica TP size or pool role differs")
            allocations.append((role, replica, url))
    urls = [row[2] for row in allocations]
    if len(set(urls)) != 4:
        raise ValueError("Hybrid and standalone native engine URLs must be distinct")
    registry_before = _registry(config, urls)
    state = config.actor_rollout_ref.rollout.trajectory_state
    expected_version = str(state.policy_identity["policy_version"])
    bindings, rows = [], []
    for index, (role, replica, url) in enumerate(allocations):
        info = _get(url, "/get_server_info")
        for key, required in (("tp_size", 2), ("disable_radix_cache", False), ("disable_overlap_schedule", False)):
            actual = list(_find(info, key))
            if not actual or any(value != required for value in actual):
                raise ValueError("Actual engine setting differs or is absent: " + key)
        running = list(_find(info, "max_running_requests"))
        if not running or any(type(value) is not int or value < 4 for value in running):
            raise ValueError("Actual max_running_requests must be at least four")
        model = _get(url, "/model_info")
        version = str(model.get("weight_version")) if isinstance(model, dict) else ""
        if version != expected_version:
            raise ValueError("Actual native engine version differs from the published policy")
        binding = {"worker_index": index, "owner_id": f"{state.run_id}/worker-{index}", "role": role,
                   "engine_url": url, "version": version, "replica_rank": replica.replica_rank,
                   "gateway_worker_id": registry_before[url]["id"], "world_size": replica.world_size}
        bindings.append(binding)
        rows.append({**binding, "server_info": info, "model_info": model,
                     "gateway_registration": registry_before[url]})
    registry_after = _registry(config, urls)
    if {url: row["id"] for url, row in registry_before.items()} != {url: row["id"] for url, row in registry_after.items()}:
        raise ValueError("Native Gateway registration changed during topology readback")
    directory = Path(output) / "joint-physical-snapshots"
    receipt_dir = config.ray_kwargs.ray_init.runtime_env.env_vars.VERL_TRAJECTORY_RECEIPT_DIR
    if any(not str(path).startswith("/mnt/shared/project-b") for path in (directory, state.root_dir, receipt_dir)):
        raise ValueError("All snapshots, trajectory objects, and receipts must use this boot's shared CFS root")
    directory.mkdir(parents=True, exist_ok=False)
    target = next(row["worker_index"] for row in bindings if row["role"] == "standalone")
    settings = {"enabled": True, "functional_singleton": False,
                "engine_urls_by_owner": {row["owner_id"]: row["engine_url"] for row in bindings},
                "snapshot_storage_dir": str(directory.resolve()), "interrupt_after_tokens": 18,
                "interrupt_owner_ids": [], "release_full_gpu_pool": False,
                "source_group_preemption_budget_s": 600.0,
                "source_capture_commit_cap_s": 45.0,
                "source_release_continue_cap_s": 15.0,
                "source_preemption_journal_dir": str(Path(receipt_dir) / "source-preemption")}
    with open_dict(state):
        state.token_kv_resume = settings
        state.handoff_target_worker_index = target
    receipt = {"schema_version": 1, "bindings": bindings, "actual_replicas": rows,
               "engine_roles": {row["engine_url"]: row["role"] for row in bindings},
               "owner_roles": {row["owner_id"]: row["role"] for row in bindings},
               "actual_owner_bindings": settings["engine_urls_by_owner"],
               "reserved_target_worker_index": target, "two_tp2_hybrid_replicas_verified": True,
               "two_tp2_standalone_replicas_verified": True, "trainer_world_size": 4,
               "gpu_reallocation_claimed": False, "same_gpu_training_verified": False,
               "release_scope": "request_kv_detach_then_original_native_hybrid_sleep",
               "shared_paths": {"physical_snapshots": str(directory), "trajectory_objects": state.root_dir,
                                "recovery_receipts": receipt_dir}, "cross_pod_readability_verified": False}
    trainer._joint_actual_topology = receipt
    (Path(output) / "joint-actual-topology.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def make_sampling_client(native_client, actual_topology):
    """Expose observed pool endpoints to the existing routed token-KV backend."""
    rows = actual_topology["bindings"]
    urls = tuple(row["engine_url"] for row in rows)
    if len(urls) != 4 or len(set(urls)) != 4:
        raise ValueError("Sampling requires the four actual initialized native engines")
    client = copy.copy(native_client)
    client._engine_urls = urls
    # The original native Ray client remains untouched for the mandatory v1
    # witness. Existing TokenKVSourceBackend selects the guarded Gateway stream.
    return client


def install_driver(trainer, output):
    from native_joint_handoff_driver import DriverOptions, NativeJointHandoffDriver

    topology = trainer._joint_actual_topology
    manager = trainer.agent_loop_manager
    if len(manager.agent_loop_workers) != len(topology["bindings"]):
        raise ValueError("Actual CPU owner count differs from the observed native pool bindings")
    target = topology["reserved_target_worker_index"]
    if manager._trajectory_reserved_target != target:
        raise ValueError("Original reclaim target differs from the actual standalone owner")
    return NativeJointHandoffDriver.install_on_trainer(trainer, options=DriverOptions(
        min_generated_tokens=18, target_worker_index=None, control_timeout_s=1800,
        source_group_preemption_budget_s=600, target_completion_reserve_s=600,
        result_path=Path(output) / "joint-handoff-result.json"), actual_topology=topology)
