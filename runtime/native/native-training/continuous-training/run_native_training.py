"""Run original VeRL V1 fit for the prepared 10/100-step configuration."""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

from native_training_audit import learning_signal_summary, make_audited_trainer, rank_map


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_runtime_source(repo, manifest_path, expected_archive_sha256):
    """Confirm all active native source bytes against the CPU-validated snapshot."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("archive_sha256") != expected_archive_sha256:
        raise ValueError("The declared runtime differs from the tested source snapshot")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("The runtime source manifest is empty")
    root = Path(repo).resolve()
    for name, digest in files.items():
        relative = Path(name)
        source = root / relative
        if relative.is_absolute() or ".." in relative.parts or source.is_symlink():
            raise ValueError("Runtime source manifest escapes its frozen checkout")
        if not source.is_file() or sha256(source) != digest:
            raise ValueError(f"Active native source differs: {name}")
    return {"source_archive_sha256": expected_archive_sha256, "verified_source_files": len(files)}


def validate_config(config, *, target, resume_step):
    if (resume_step, target) not in ((0, 1), (0, 10), (10, 100)):
        raise ValueError("Use the bounded 0->1, 0->10 or 10->100 native experiment")
    trainer, actor = config["trainer"], config["actor_rollout_ref"]
    rollout, data = actor["rollout"], config["data"]
    if trainer["v1"]["trainer_mode"] != "separate_async" or not trainer["use_v1"]:
        raise ValueError("The original V1 separate-async trainer is required")
    if trainer["total_training_steps"] != target:
        raise ValueError("Prepared native stopping step differs from the requested target")
    if trainer["v1"]["separate_async"]["parameter_sync_step"] != 1:
        raise ValueError("Every update must use the original per-step publication")
    if data["train_batch_size"] != 8 or actor["actor"]["ppo_mini_batch_size"] != 8:
        raise ValueError("Eight complete prompt groups form one native update")
    if rollout["n"] != 8 or actor["actor"]["ppo_epochs"] != 1:
        raise ValueError("Use eight fresh members per group and one original PPO epoch")
    if (rollout.get("do_sample", True) is not True or rollout.get("temperature", 0) <= 0
            or rollout.get("top_k") == 1):
        raise ValueError("Reuse the original stochastic training sampling; reject a greedy initialization-only input")
    if actor["actor"]["fsdp_config"]["model_dtype"] != "fp32":
        raise ValueError("Keep the existing FP32 training weights")
    fsdp = actor["actor"]["fsdp_config"]
    if fsdp.get("offload_policy") is not True:
        raise ValueError("The first native FSDP2 Adam update requires CPUOffloadPolicy")
    if any(fsdp.get(name, False) is not False for name in ("param_offload", "optimizer_offload")):
        raise ValueError("Use native CPUOffloadPolicy rather than manual tensor offload flags")
    if rollout["checkpoint_engine"]["backend"] != "nccl":
        raise ValueError("Preserve the native standalone NCCL publication")
    if not rollout["sglang_gateway"]["enabled"]:
        raise ValueError("Native engine version readback requires the configured Gateway")
    state = rollout["trajectory_state"]
    if not state["enabled"] or not state["continuous_policy_publication"]:
        raise ValueError("Durable state must be connected to every native publication")
    if actor["actor"].get("checkpoint", {}).get("async_save", False):
        raise ValueError("Preserve synchronous native checkpoint plus trajectory sidecar")
    first_update_only = (resume_step, target) == (0, 1)
    expected_save_freq = -1 if first_update_only else 50
    if trainer["save_freq"] != expected_save_freq or (first_update_only and trainer["test_freq"] != 0) or (not first_update_only and trainer["test_freq"] <= 0):
        raise ValueError("Preserve native saves and the declared phase-specific validation policy")
    if first_update_only and trainer["val_before_train"]:
        raise ValueError("First-update scope must defer the full held-out evaluation")
    expected_resume = "disable" if resume_step == 0 else "resume_path"
    if trainer["resume_mode"] != expected_resume:
        raise ValueError("Cold resume must explicitly load the saved native checkpoint")
    if resume_step and Path(trainer["resume_from_path"]).name != f"global_step_{resume_step}":
        raise ValueError("The declared native resume checkpoint differs")
    ranks = trainer["nnodes"] * trainer["n_gpus_per_node"]
    if (ranks != 4 or trainer["nnodes"] != 4 or trainer["n_gpus_per_node"] != 1
            or rollout["nnodes"] != 4 or rollout["n_gpus_per_node"] != 1):
        raise ValueError("The candidate requires four training and four independent standalone generation GPUs")
    if (rollout["tensor_model_parallel_size"], rollout["data_parallel_size"], rollout["pipeline_model_parallel_size"]) != (2, 1, 1):
        raise ValueError("Each native hybrid/standalone pool must contain exactly two TP2 replicas")
    if (actor["hybrid_engine"] is not True or state["shared_gpu_handoff"] is not True
            or trainer["v1"]["separate_async"]["hybrid_rollout"]["enable_switch"] is not True
            or rollout["free_cache_engine"] is not True or rollout["agent"]["num_workers"] != 4):
        raise ValueError("Keep original four-rank hybrid switching and four actual CPU owners")
    return ranks


def verify_native_version_token(trainer, llm_client, *, read_publication, output, timeout_s=60):
    """Observe one <=4-token native Ray generation after the verified first update."""
    import asyncio
    import pickle
    from uuid import uuid4

    from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)  # Never repeat a previous witness automatically.
    receipt = {"status": "STARTED_NOT_PASS", "expected_policy_version": "1",
               "generation_path": "native_ray_actor", "max_new_tokens": 4,
               "client_generate_calls": 0, "automatic_update_retry_allowed": False}

    def json_value(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Unsupported native output value: {type(value).__name__}")

    def save(name, value):
        with (directory / name).open("w") as stream:
            json.dump(value, stream, default=json_value, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def validate_publication(observed):
        if (str(observed["actual_parameter_version"]) != "1"
                or observed["active_policy_record"]["identity"]["policy_version"] != "1"):
            raise ValueError("Version-token witness requires the actually published version 1")

    try:
        if not isinstance(llm_client, FullyAsyncLLMServerClient) or llm_client._generation_transport is not None:
            raise ValueError("Version-token witness requires the initialized native Ray client")
        if not 0 < timeout_s <= 60:
            raise ValueError("Version-token generation timeout must be within 60 seconds")
        before = read_publication(trainer)
        receipt["publication_before"] = before
        validate_publication(before)
        prompt_ids = trainer.tokenizer.apply_chat_template(
            [{"role": "user", "content": "请回答1。"}], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if not isinstance(prompt_ids, list) or not prompt_ids or any(type(token) is not int for token in prompt_ids):
            raise ValueError("Native tokenizer did not return a nonempty token list")
        request = {"request_id": f"post-update-version-1-{uuid4().hex}",
                   "prompt_ids": prompt_ids,
                   "sampling_params": {"max_new_tokens": 4, "temperature": 0.0, "logprobs": True}}
        # Native SGLang has no expected-version request kwarg. Bind the expectation in
        # this receipt, then require engine-returned metadata and stable publication.
        save("request.json", {**request, "expected_policy_version": "1", "publication": before,
                              "native_expected_version_kwarg_supported": False,
                              "generation_path": "native_ray_actor", "timeout_s": timeout_s})
        receipt["client_generate_calls"] = 1
        save("result.json", receipt)

        async def generate_once():
            return await asyncio.wait_for(llm_client.generate(**request), timeout=timeout_s)

        native_output = asyncio.run(generate_once())
        with (directory / "token-output.pkl").open("wb") as stream:
            pickle.dump(native_output, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        native_view = native_output.model_dump()
        save("token-output.json", native_view)
        save("output-ids.json", native_output.token_ids)
        receipt["output_token_count"] = len(native_output.token_ids)
        receipt["token_output_path"] = str(directory / "token-output.pkl")
        if not 1 <= len(native_output.token_ids) <= 4:
            raise ValueError("Version-token witness must return between one and four real tokens")
        extra = native_output.extra_fields
        attempts = extra.get("native_generation_attempts", [])
        if extra.get("generation_path") != "native_ray_actor" or not attempts:
            raise ValueError("Native Ray engine metadata is missing from the returned TokenOutput")
        cursor = 0
        for attempt in attempts:
            if str(attempt.get("native_meta_info", {}).get("weight_version")) != "1":
                raise ValueError("Returned native engine weight_version is not the requested version 1")
            if (attempt.get("output_start") != cursor or type(attempt.get("output_end")) is not int
                    or not cursor <= attempt["output_end"] <= len(native_output.token_ids)):
                raise ValueError("Native version metadata does not cover the returned token ranges")
            cursor = attempt["output_end"]
        if cursor != len(native_output.token_ids) or any(
            str(extra.get(name)) != "1" for name in ("global_steps", "min_global_steps", "max_global_steps")
        ):
            raise ValueError("Returned token ranges do not all belong to native version 1")
        after = read_publication(trainer)
        receipt["publication_after"] = after
        validate_publication(after)
        if before["active_policy_record"] != after["active_policy_record"]:
            raise ValueError("Published policy changed during the version-token witness")
        receipt.update(status="PASS_NATIVE_VERSION1_TOKEN", actual_engine_weight_version="1")
        return receipt
    except BaseException as error:
        receipt.update(status="FAILED_NOT_PASS", error_type=type(error).__name__, error=str(error))
        (directory / "error_stack.txt").write_text(traceback.format_exc())
        raise
    finally:
        save("result.json", receipt)


def run_original_fit(config, frozen, control):
    import asyncio

    import ray
    import transfer_queue as tq
    from actor_worker import observe_original_parameters
    from omegaconf import OmegaConf
    from real_batch_replay import capture_trainer_class
    from run_v1_functional_gate import _initialized_parameter_observation, parameter_sample_delta
    from training_update_audit import (
        begin_worker_update_audit_step,
        install_worker_update_audit,
        read_worker_update_audit,
    )

    from verl.trainer.ppo.v1 import AgentLoopManagerTQ, get_trainer_cls
    from verl.trainer.ppo.v1.trajectory_state.policy_publication import PublishedPolicyRecord

    output = Path(control["output"])
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "STARTED_NOT_PASS",
        "data_kind": frozen["data_kind"],
        "project_production_acceptance": False,
        "native_fit_called": False,
        "physical_gpu_release_verified": False,
        "full_restore_equivalence_verified": False,
        "config_sha256": frozen["config_sha256"],
        "resume_step": control["resume_step"],
        "target_step": control["target"],
        "runtime_source_archive_sha256": control["runtime_source_sha256"],
    }

    def event(name, **fields):
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": name, **fields}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def rpc(trainer, function, *args):
        return ray.get(
            trainer.actor_rollout_wg.execute_all_async("execute_with_func_generator", function, *args),
            timeout=control["rpc_timeout_s"],
        )

    def publication(trainer):
        from run_v1_functional_gate import _read_trainer_active_weight_version

        actual = asyncio.run(_read_trainer_active_weight_version(trainer))
        record = ray.get(
            trainer.agent_loop_manager.trajectory_coordinator.get_active_published_policy.remote(),
            timeout=control["rpc_timeout_s"],
        )
        return {
            "actual_parameter_version": actual,
            "active_policy_record": PublishedPolicyRecord.decode(record).encode(),
        }

    def learning_signal(trainer, batch):
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                               select_fields=["uid", "response_mask", "advantages", "rm_scores", "responses"])
        return learning_signal_summary(data.to_padded_tensor(), batch.keys,
                                       group_size=config.actor_rollout_ref.rollout.n,
                                       group_count=config.data.train_batch_size)

    rank_count = validate_config(
        OmegaConf.to_container(config, resolve=True), target=control["target"], resume_step=control["resume_step"]
    )
    base = get_trainer_cls("separate_async")
    if base.__module__ != "verl.trainer.ppo.v1.trainer_separate_async":
        raise ValueError("The original trainer class is required")
    trainer = None
    routing_compare, close_optional = None, None
    tq.init(config.transfer_queue)
    try:
        cls = make_audited_trainer(
            base,
            rpc=rpc,
            event=event,
            read_publication=publication,
            rank_count=rank_count,
            begin_audit=begin_worker_update_audit_step,
            read_audit=read_worker_update_audit,
            read_learning_signal=learning_signal,
            model_configuration_fingerprint=config.actor_rollout_ref.rollout.trajectory_state.model_configuration_fingerprint,
        )
        cls = capture_trainer_class(cls, output / "native-batch-captures", read_identity=publication)
        trainer = cls(config=config)
        trainer.audit_update_steps, trainer.audit_verified_steps, trainer.audit_publication_steps = [], [], []
        trainer.init()
        if trainer.global_steps != control["resume_step"]:
            raise ValueError("Native model/optimizer restore did not reach the declared starting step")
        initial = _initialized_parameter_observation(trainer)
        config.actor_rollout_ref.rollout.trajectory_state.policy_identity = initial["policy_identity"]
        from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
        llm_client = trainer.standalone_server_manager.get_client(
            client_cls=FullyAsyncLLMServerClient, generation_transport=None
        )
        if llm_client._generation_transport is not None:
            raise ValueError("Native actor baseline must use the original Ray generation path")
        from joint_topology import bind_actual_replicas, install_driver, make_sampling_client
        actual_topology = bind_actual_replicas(trainer, config, output)
        sampling_client = make_sampling_client(llm_client, actual_topology)
        event("generation_path_selected", generation_path="native_gateway_routed_token_kv",
              gateway_generation_used=True, native_ray_version1_witness_retained=True,
              hybrid_replicas=2, standalone_replicas=2, trainer_world_size=4)
        trainer.agent_loop_manager = AgentLoopManagerTQ.create(
            config=config,
            llm_client=sampling_client,
            teacher_client=trainer.get_teacher_client(),
            reward_loop_worker_handles=trainer.get_reward_handles(),
        )
        joint_driver = install_driver(trainer, output)
        if os.environ.get("VERL_ROUTING_BUSINESS_COMPARE_OPTIONS"):
            try:
                from entry_hooks import prepare_optional, finish_optional, close_optional
                routing_compare = prepare_optional(trainer, publication, initial,
                    report=report, output=output)
            except BaseException as benchmark_error:
                report["routing_business_compare"] = {"status": "NOT_RUN", "stage": "benchmark_prepare_import",
                    "error_type": type(benchmark_error).__name__, "error": str(benchmark_error)}
        before = rpc(trainer, observe_original_parameters)
        rank_map(before, rank_count)
        if any(row.get("error_type") for row in before):
            raise ValueError("Actual initialized parameter observation failed")
        if control["resume_step"]:
            if any(row.get("optimizer_step_max") != control["resume_step"] for row in before):
                raise ValueError("Original Adam state did not restore the saved update frontier")
            prior = json.loads(Path(control["prior_parameters"]).read_text())
            proof = parameter_sample_delta(prior, before)
            if proof.get("sample_linf") != 0.0:
                raise ValueError("Actual restored parameter samples differ from the previous process")
            event(
                "native_cold_resume_samples_verified",
                parameter_samples=proof,
                restored_prompt_groups=trainer._restored_tq_prompt_count,
                full_restore_equivalence_verified=False,
            )
        observations = rpc(trainer, install_worker_update_audit, str(output / "optimizer-audit"))
        rank_map(observations, rank_count)
        event(
            "original_trainer_initialized",
            publication=initial,
            observations=observations,
            original_fit=f"{base.__module__}.{base.__name__}.fit",
        )
        report["native_fit_called"] = True
        trainer.fit(trainer.agent_loop_manager)
        after = rpc(trainer, observe_original_parameters)
        parameter_proof = parameter_sample_delta(before, after)
        (output / "parameters-final.json").write_text(json.dumps(after, allow_nan=False) + "\n")
        expected_steps = list(range(control["resume_step"] + 1, control["target"] + 1))
        final_observations = rpc(trainer, read_worker_update_audit)
        for row in rank_map(final_observations, rank_count).values():
            if row["verified_counter_advancements_since_attach"] != len(expected_steps):
                raise ValueError("Final native optimizer observation is incomplete")
        if trainer.audit_verified_steps != expected_steps or trainer.audit_publication_steps != expected_steps:
            raise ValueError("Native fit did not verify every requested update and publication")
        if parameter_proof.get("observed_parameter_change") is not True:
            raise ValueError("Actual original parameter samples did not confirm a training change")
        learning_steps = getattr(trainer, "audit_learning_update_steps", [])
        report["learning_signal_update_steps"] = learning_steps
        report["learning_signal_updates_this_process"] = len(learning_steps)
        if not learning_steps:
            raise ValueError("Completed optimizer updates did not verify any original GRPO learning signal")
        checkpoint = None
        if (control["resume_step"], control["target"]) != (0, 1):
            checkpoint = Path(config.trainer.default_local_dir) / f"global_step_{control['target']}"
            if any(not (checkpoint / name).exists() for name in ("actor", "data.pt", "transfer_queue", "trajectory_state")):
                raise ValueError("The original full checkpoint components were not all saved")
            marker = Path(config.trainer.default_local_dir) / "latest_checkpointed_iteration.txt"
            if marker.read_text().strip() != str(control["target"]):
                raise ValueError("Original latest checkpoint marker does not confirm the final save")
        report.update(
            status="PASS_NATIVE_UPDATES_AND_PUBLICATIONS",
            verified_optimizer_updates_this_process=len(expected_steps),
            native_trainer_final_step=control["target"],
            observations=final_observations,
            parameter_sample_proof=parameter_proof,
            native_checkpoint=str(checkpoint) if checkpoint is not None else None,
            native_checkpoint_saved=checkpoint is not None,
            controller_reclaims=list(getattr(trainer, "trajectory_resource_reclaims", [])),
        )
        if (control["resume_step"], control["target"]) == (0, 1):
            # Preserve completed training evidence even if the later inference witness fails.
            (output / "optimizer-evidence-before-version-token.json").write_text(
                json.dumps(report, allow_nan=False, indent=2) + "\n"
            )
            try:
                report["version1_token_witness"] = verify_native_version_token(
                    trainer, llm_client, read_publication=publication,
                    output=output / "version1-token-witness", timeout_s=min(control["rpc_timeout_s"], 60),
                )
            except BaseException:
                report["version1_token_witness"] = {"status": "FAILED_NOT_PASS",
                    "receipt": str(output / "version1-token-witness" / "result.json")}
                raise
            # Keep the established bootstrap status; the mandatory witness raises on failure.
            report["status"] = "PASS_NATIVE_UPDATES_AND_PUBLICATIONS"
        if routing_compare is not None:
            try:
                report["routing_business_compare"] = finish_optional(routing_compare,
                    report=report, output=output)
            except BaseException as benchmark_error:
                report["routing_business_compare"] = {"status": "NOT_ACCEPTED", "stage": "benchmark_finish",
                    "error_type": type(benchmark_error).__name__, "error": str(benchmark_error),
                    "automatic_retries": 0}
        first_update_reader = globals().get("_runtime_first_update_reader")
        if not callable(first_update_reader):
            raise ValueError("Joint native acceptance requires the bound original first-update observation producer")
        first_update_original = first_update_reader()
        report["first_update_observation"] = first_update_original["status"]
        report["first_update_observation_original"] = first_update_original
        report["joint_handoff"] = joint_driver.finalize(report)
        elastic = report["joint_handoff"]
        report["elastic_hybrid_acceptance"] = {
            "status": elastic["status"],
            "actual_gpu_reuse_verified": elastic.get("actual_gpu_reuse_verified") is True,
            "native_entry_update_and_publication_accepted": elastic.get("native_entry_update_and_publication_accepted") is True,
            "accepted": (elastic["status"] == "PASS_NATIVE_ELASTIC_HYBRID_TOKEN_KV_HANDOFF"
                         and elastic.get("actual_gpu_reuse_verified") is True),
            "automatic_retries": 0,
        }
        if control.get("routing_experiment"):
            try:
                from routing_experiment_native_window import after_joint_acceptance
                report["routing_experiment"] = after_joint_acceptance(trainer, config, frozen, control, report, output)
            except BaseException as window_error:
                report["routing_experiment"] = {"status": "NOT_ACCEPTED_OPTIONAL_HOOK",
                    "error_type": type(window_error).__name__, "automatic_retries": 0,
                    "joint_result_overridden": False}
        # Missing identity evidence rejects elastic acceptance independently of
        # the completed native update. Original control exceptions still propagate.
        return report
    except BaseException as error:
        (output / "error_stack.txt").write_text(traceback.format_exc())
        report.update(
            status="FAILED_NOT_PASS",
            error_type=type(error).__name__,
            automatic_update_retry_allowed=False,
            dispatched_update_steps=getattr(trainer, "audit_update_steps", []),
            verified_update_steps=getattr(trainer, "audit_verified_steps", []),
            learning_signal_update_steps=getattr(trainer, "audit_learning_update_steps", []),
        )
        raise
    finally:
        if routing_compare is not None:
            try:
                if close_optional is not None:
                    close_optional(routing_compare, report=report, output=output)
                else:
                    routing_compare.close()
            except BaseException as cleanup_error:
                try:
                    benchmark = dict(report.get("routing_business_compare") or {})
                    benchmark.setdefault("benchmark_cleanup_errors", []).append({
                        "error_type": type(cleanup_error).__name__, "error": str(cleanup_error)})
                    report["routing_business_compare"] = benchmark
                except BaseException:
                    pass  # Never replace the native training result or active exception.
        (output / "result.json").write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
        try:
            if trainer is not None:
                trainer._shutdown_dump_executor()
                tracking = getattr(trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if report["status"].startswith("PASS_") else 1)
        finally:
            tq.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, choices=(1, 10, 100), required=True)
    parser.add_argument("--resume-step", type=int, choices=(0, 10), required=True)
    parser.add_argument("--prior-parameters", type=Path)
    parser.add_argument("--runtime-source-sha256", required=True)
    parser.add_argument("--runtime-source-manifest", type=Path, required=True)
    parser.add_argument("--execute-native", action="store_true")
    options = parser.parse_args()
    if not options.execute_native or sys.platform == "darwin":
        raise RuntimeError("Execute only inside the owned native Linux GPU allocation")
    if options.resume_step and not options.prior_parameters:
        raise ValueError("Cold resume requires parameter observations from the previous process")
    if len(options.runtime_source_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in options.runtime_source_sha256
    ):
        raise ValueError("A verified native source archive SHA is required")
    directory = Path(__file__).resolve().parent
    sys.path[:0] = [
        str(options.repo.resolve()),
        str(options.repo / "project-b-final-validation-20260928/production-entry-20261008"),
        str(directory.parent / "update-audit"),
    ]
    import ray
    from omegaconf import OmegaConf
    from run_v1_functional_gate import verify_freeze

    frozen = verify_freeze(options.config, options.freeze_manifest)
    source_files = [row for row in frozen["files"] if row["role"] == "runtime_source_manifest"]
    if len(source_files) != 1 or Path(source_files[0]["path"]).resolve() != options.runtime_source_manifest.resolve():
        raise ValueError("The source manifest must be one of the frozen verified inputs")
    verify_runtime_source(options.repo, options.runtime_source_manifest, options.runtime_source_sha256)
    config = OmegaConf.load(options.config)
    validate_config(
        OmegaConf.to_container(config, resolve=True), target=options.target, resume_step=options.resume_step
    )
    environment = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env, resolve=True)
    ray.init(address=os.environ["RAY_ADDRESS"], runtime_env=environment)
    control = {
        "output": str(options.output.resolve()),
        "target": options.target,
        "resume_step": options.resume_step,
        "prior_parameters": str(options.prior_parameters.resolve()) if options.prior_parameters else None,
        "runtime_source_sha256": options.runtime_source_sha256,
        "rpc_timeout_s": 180,
    }
    # The same original remote driver pattern as TaskRunnerV1; model workers remain native.
    runner = ray.remote(num_cpus=1)(run_original_fit)
    result = ray.get(runner.remote(config, frozen, control))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "data_kind",
                    "project_production_acceptance",
                    "verified_optimizer_updates_this_process",
                    "native_checkpoint",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
