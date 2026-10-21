"""Prepare a bounded native run from the existing frozen functional workload."""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

from run_native_training import validate_config


def prepare_config(base, *, target, checkpoint_root, durable_root, run_id, entry_dir, audit_dir):
    config = copy.deepcopy(base)
    trainer, actor = config["trainer"], config["actor_rollout_ref"]
    rollout, data = actor["rollout"], config["data"]
    trainer.update(
        total_training_steps=target,
        total_epochs=target,
        nnodes=4,
        n_gpus_per_node=1,
        resume_mode="disable" if target in (1, 10) else "resume_path",
        val_before_train=target != 1,
        save_freq=-1 if target == 1 else 50,
        test_freq=0 if target == 1 else 10,
        default_local_dir=str(checkpoint_root),
        max_actor_ckpt_to_keep=2,
        max_critic_ckpt_to_keep=2,
        del_local_ckpt_after_load=False,
        experiment_name=run_id,
    )
    if target == 100:
        trainer["resume_from_path"] = str(Path(checkpoint_root) / "global_step_10")
    trainer["v1"]["separate_async"].update(parameter_sync_step=1, num_warmup_batches=1)
    trainer["v1"]["separate_async"]["hybrid_rollout"]["enable_switch"] = True
    actor["hybrid_engine"] = True
    actor["actor"].update(ppo_mini_batch_size=8, ppo_epochs=1)
    # Native FSDP2 CPUOffloadPolicy keeps the first Adam state allocation on CPU.
    actor["actor"]["fsdp_config"]["offload_policy"] = True
    actor["actor"].setdefault("checkpoint", {})["async_save"] = False
    # Constant learning rate preserves scheduler behavior across the two processes.
    actor["actor"]["optim"].update(lr_scheduler_type="constant", lr_warmup_steps=0, lr_warmup_steps_ratio=0.0)
    rollout.update(nnodes=4, n_gpus_per_node=1, tensor_model_parallel_size=2)
    rollout.setdefault("engine_kwargs", {}).setdefault("sglang", {})["sampling_backend"] = "pytorch"
    rollout["checkpoint_engine"]["backend"] = "nccl"
    rollout["checkpoint_engine"].setdefault("engine_kwargs", {}).setdefault("nccl", {})["multi_sender"] = False
    rollout["agent"]["num_workers"] = 4
    rollout["sglang_gateway"].update(enabled=True, allow_partial_worker_set=True)
    rollout["trajectory_state"].update(
        enabled=True,
        continuous_policy_publication=True,
        shared_gpu_handoff=True,
        handoff_target_worker_index=2,
        root_dir=str(durable_root),
        run_id=run_id,
    )
    data.update(train_batch_size=8, gen_batch_size=8)
    config["transfer_queue"]["backend"]["SimpleStorage"]["total_storage_size"] = 4096
    environment = config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]
    repo_path = Path(environment["PYTHONPATH"].split(os.pathsep)[0])
    functional_entry = repo_path / "project-b-final-validation-20260928/production-entry-20261008"
    environment["PYTHONPATH"] = os.pathsep.join(
        (environment["PYTHONPATH"], str(functional_entry), str(entry_dir), str(audit_dir))
    )
    from joint_topology import override_config
    config = override_config(config, output_root=Path(checkpoint_root).parent)
    # Optional benchmark path must reach the actual remote trainer through frozen Ray runtime_env.
    routing_options = os.environ.get("VERL_ROUTING_BUSINESS_COMPARE_OPTIONS")
    if routing_options:
        config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["VERL_ROUTING_BUSINESS_COMPARE_OPTIONS"] = routing_options
    validate_config(config, target=target, resume_step=0 if target in (1, 10) else 10)
    return config


def prepare_experiment_manifest(frozen, config, *, target):
    """Describe this active experiment; retain old initialization metadata as a reference."""
    result = copy.deepcopy(frozen)
    reference = {key: result.pop(key) for key in ("controlled_resource_profile", "controlled_configuration")
                 if key in result}
    if reference:
        result["input_experiment_reference"] = reference
    trainer, actor = config["trainer"], config["actor_rollout_ref"]
    rollout = actor["rollout"]
    result["native_training_resource_profile"] = {
        "trainer_nodes": trainer["nnodes"], "trainer_gpus_per_node": trainer["n_gpus_per_node"],
        "rollout_nodes": rollout["nnodes"], "rollout_gpus_per_node": rollout["n_gpus_per_node"],
        "rollout_tp": rollout["tensor_model_parallel_size"],
        "hybrid_rollout_replicas": 2, "standalone_rollout_replicas": 2,
        "trainer_world_size_fixed": 4, "same_gpu_training_verified": False,
        "actor_strategy": actor["actor"]["strategy"],
        "actor_model_dtype": actor["actor"]["fsdp_config"]["model_dtype"],
        "planned_updates_this_process": target - (10 if target == 100 else 0),
        "optimizer_updates_verified": 0,
    }
    result["native_training_sampling"] = {
        "temperature": rollout["temperature"], "top_p": rollout["top_p"], "top_k": rollout["top_k"],
        "do_sample": rollout.get("do_sample", True), "members_per_prompt": rollout["n"],
        "sampling_backend": rollout["engine_kwargs"]["sglang"]["sampling_backend"],
        "learning_signal_verified": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target", type=int, choices=(1, 10, 100), required=True)
    parser.add_argument("--runtime-source-manifest", type=Path, required=True)
    options = parser.parse_args()
    if sys.platform == "darwin" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Prepare inside the owned Linux CPU staging process with no visible CUDA devices")
    directory = Path(__file__).resolve().parent
    sys.path[:0] = [
        str(options.repo.resolve()),
        str(options.repo / "project-b-final-validation-20260928/production-entry-20261008"),
    ]
    from omegaconf import OmegaConf
    from prepare_v1_functional_gate import canonical_hash
    from run_v1_functional_gate import verify_freeze

    frozen = verify_freeze(options.config, options.freeze_manifest)
    base = OmegaConf.to_container(OmegaConf.load(options.config), resolve=True)
    config = prepare_config(
        base,
        target=options.target,
        checkpoint_root=options.checkpoint_root,
        durable_root=options.durable_root,
        run_id=options.run_id,
        entry_dir=directory,
        audit_dir=directory.parent / "update-audit",
    )
    frozen = prepare_experiment_manifest(frozen, config, target=options.target)
    source_manifest = options.runtime_source_manifest.resolve()
    source_digest = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    from run_native_training import verify_runtime_source

    source = json.loads(source_manifest.read_text())
    verify_runtime_source(options.repo, source_manifest, source["archive_sha256"])
    frozen["files"] = [row for row in frozen["files"] if row["role"] != "runtime_source_manifest"]
    frozen["files"].append({"role": "runtime_source_manifest", "path": str(source_manifest), "sha256": source_digest})
    frozen["kv_fast_path"]["tp"] = 2
    identity = frozen["policy_identity"]
    identity["kv_cache_schema_fingerprint"] = canonical_hash(frozen["kv_fast_path"])
    config["actor_rollout_ref"]["rollout"]["trajectory_state"]["policy_identity"] = identity
    frozen["continuous_policy_publication"] = True
    frozen["model_configuration_fingerprint"] = config["actor_rollout_ref"]["rollout"]["trajectory_state"][
        "model_configuration_fingerprint"
    ]
    options.output.mkdir(parents=True, exist_ok=False)
    config_file = options.output / "frozen-native.yaml"
    OmegaConf.save(OmegaConf.create(config), config_file, resolve=True)
    frozen["config_sha256"] = hashlib.sha256(config_file.read_bytes()).hexdigest()
    frozen.update(
        experiment_scope="synthetic correctness/endurance, separate from actual task trial",
        production_task_trial_completed=False,
        target_step=options.target,
        native_resume_step=0 if options.target in (1, 10) else 10,
    )
    (options.output / "freeze.json").write_text(json.dumps(frozen, allow_nan=False, indent=2) + "\n")
    receipt = {
        "status": "PREPARED_NOT_GPU_VALIDATED",
        "config_sha256": frozen["config_sha256"],
        "target_step": options.target,
        "train_gpus": 4,
        "standalone_generation_gpus": 4,
        "train_nodes": 4,
        "generation_nodes": 4,
        "gpus_per_node": 1,
        "data_kind": frozen["data_kind"],
        "train_rows": frozen["data"]["train"].get("rows"),
        "production_task_trial_completed": False,
        "models_loaded": 0,
        "checkpoint_cadence": (
            "deferred for the first-update scope; continuous-training checkpoints remain required"
            if options.target == 1
            else "native final step plus step 50; retain two actor checkpoints"
        ),
        "physical_gpu_release_verified": False,
    }
    (options.output / "prepare-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
