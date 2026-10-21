import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from prepare_native_training import prepare_config, prepare_experiment_manifest
from run_native_training import validate_config

REPO = Path("/workspace/verl")


@pytest.fixture
def original_config(tmp_path):
    # Read the actual stock preparation function, rather than copying its defaults.
    repo = Path(__import__("os").environ.get("NATIVE_AUDIT_REPO", str(REPO)))
    source = repo / "project-b-final-validation-20260928/production-entry-20261008/prepare_v1_functional_gate.py"
    spec = importlib.util.spec_from_file_location("original_functional_preparation", source)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    model_config = tmp_path / "model-config.json"
    model_config.write_text('{"model_type":"qwen3_moe"}\n')
    paths = {
        "train_data": str(tmp_path / "train.parquet"),
        "validation_data": str(tmp_path / "test.parquet"),
        "model_path": str(tmp_path / "model"),
        "model_config": str(model_config),
        "support_dir": tmp_path,
        "tools_config": str(tmp_path / "tools.yaml"),
        "reward_implementation": str(tmp_path / "reward.py"),
    }
    identity = {
        name: "test-input-only"
        for name in (
            "policy_version",
            "model_weights_fingerprint",
            "tokenizer_template_fingerprint",
            "kv_cache_schema_fingerprint",
        )
    }
    config = original.prepare_config(
        repo,
        paths,
        identity,
        "environment",
        "tool",
        "http://127.0.0.1:1234",
        "native-test-only",
        tmp_path,
        continuous_policy_publication=True,
    )
    return OmegaConf.to_container(config, resolve=True)


def candidate(original, tmp_path, target=10):
    return prepare_config(
        original,
        target=target,
        checkpoint_root=tmp_path / "checkpoints",
        durable_root=tmp_path / "durable",
        run_id="test-only",
        entry_dir=tmp_path / "entry",
        audit_dir=tmp_path / "audit",
    )


def test_candidate_uses_native_16_gpu_layout_and_original_training_precision(original_config, tmp_path):
    config = candidate(original_config, tmp_path)
    assert validate_config(config, target=10, resume_step=0) == 8
    assert (
        config["actor_rollout_ref"]["actor"]["fsdp_config"]
        == original_config["actor_rollout_ref"]["actor"]["fsdp_config"]
    )
    assert config["algorithm"] == original_config["algorithm"]
    assert original_config["data"]["train_batch_size"] == 1  # Preparation did not mutate frozen input.


def test_cold_resume_preserves_checkpoint_and_model_inputs(original_config, tmp_path):
    first = candidate(original_config, tmp_path)
    resumed = candidate(original_config, tmp_path, target=100)
    assert resumed["trainer"]["resume_mode"] == "resume_path"
    assert resumed["trainer"]["resume_from_path"].endswith("global_step_10")
    assert not resumed["trainer"]["del_local_ckpt_after_load"]
    assert first["actor_rollout_ref"] == resumed["actor_rollout_ref"]
    assert first["data"] == resumed["data"]


def test_long_run_requires_the_verified_ten_step_checkpoint(original_config, tmp_path):
    config = candidate(original_config, tmp_path, target=100)
    with pytest.raises(ValueError, match="bounded"):
        validate_config(config, target=100, resume_step=0)


def test_original_probe_module_is_importable_on_remote_workers(original_config, tmp_path):
    import os

    config = candidate(original_config, tmp_path)
    paths = config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["PYTHONPATH"].split(os.pathsep)
    repo = Path(original_config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["PYTHONPATH"].split(os.pathsep)[0])
    assert str(repo / "project-b-final-validation-20260928/production-entry-20261008") in paths
    assert str(tmp_path / "entry") in paths and str(tmp_path / "audit") in paths


@pytest.mark.parametrize(
    "path,value,match",
    [
        (("actor_rollout_ref", "actor", "ppo_epochs"), 2, "PPO epoch"),
        (("actor_rollout_ref", "actor", "ppo_mini_batch_size"), 4, "prompt groups"),
        (("actor_rollout_ref", "actor", "fsdp_config", "model_dtype"), "bfloat16", "FP32"),
        (("actor_rollout_ref", "actor", "checkpoint", "async_save"), True, "synchronous"),
        (("actor_rollout_ref", "rollout", "n"), 4, "fresh members"),
        (("actor_rollout_ref", "rollout", "temperature"), 0.0, "stochastic"),
        (("actor_rollout_ref", "rollout", "do_sample"), False, "stochastic"),
        (("actor_rollout_ref", "rollout", "top_k"), 1, "stochastic"),
        (("actor_rollout_ref", "rollout", "n_gpus_per_node"), 2, "eight training"),
        (("actor_rollout_ref", "rollout", "checkpoint_engine", "backend"), "naive", "standalone NCCL"),
        (("trainer", "v1", "separate_async", "parameter_sync_step"), 2, "per-step publication"),
    ],
)
def test_expensive_invalid_configuration_is_rejected_before_native_allocation(
    original_config, tmp_path, path, value, match
):
    config = candidate(original_config, tmp_path)
    destination = config
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        validate_config(config, target=10, resume_step=0)


def test_old_initialization_scope_cannot_describe_active_training(original_config, tmp_path):
    old = {"controlled_resource_profile": {"actor_gpus": 2, "optimizer_calls_max": 0},
           "controlled_configuration": {"fixed_arms": ["drain", "abort", "handoff"]}}
    config = candidate(original_config, tmp_path, target=100)
    prepared = prepare_experiment_manifest(old, config, target=100)
    assert "controlled_resource_profile" not in prepared and "controlled_configuration" not in prepared
    assert prepared["input_experiment_reference"] == old
    assert prepared["native_training_resource_profile"]["planned_updates_this_process"] == 90
    assert prepared["native_training_resource_profile"]["trainer_gpus_per_node"] == 4
    assert prepared["native_training_sampling"]["temperature"] == 1.0
    assert prepared["native_training_sampling"]["learning_signal_verified"] is False
    assert old["controlled_resource_profile"]["optimizer_calls_max"] == 0
