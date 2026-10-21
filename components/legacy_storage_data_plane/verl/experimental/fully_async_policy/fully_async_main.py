# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import math
import os
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.fully_async_policy.rollout_artifact import (
    RayRolloutArtifactStoreClient,
    create_ray_rollout_artifact_store_actor,
)
from verl.experimental.fully_async_policy.transfer_admission import TransferAdmissionConfig
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local


def validate_artifact_data_plane_config(config) -> None:
    """Validate trajectory-artifact eligibility independently from KV reuse.

    A nonzero asynchronous staleness budget permits training on trajectories
    produced by a previous policy version. If the artifact backend rejects that
    version before the trainer can claim it, finite rollouts can exhaust their
    queue after the first update. This check does not alter the policy KV
    identity or permit cross-version cache reuse.
    """

    artifact_config = config.async_training.get("artifact_data_plane", {})
    if not artifact_config.get("enabled", False):
        return
    if artifact_config.get("training_semantics_enabled", False):
        for field_name in ("environment_version", "tool_version"):
            version = artifact_config.get(field_name)
            if not isinstance(version, str) or not version:
                raise ValueError(f"artifact_data_plane.{field_name} must be a non-empty string")

    accepted_policy_lag = int(artifact_config.get("accepted_policy_lag", 0))
    if accepted_policy_lag < 0:
        raise ValueError("artifact_data_plane.accepted_policy_lag must be >= 0")
    staleness_threshold = float(config.async_training.get("staleness_threshold", 0))
    if staleness_threshold < 0:
        raise ValueError("async_training.staleness_threshold must be >= 0")
    required_policy_lag = math.ceil(staleness_threshold)
    if accepted_policy_lag < required_policy_lag:
        raise ValueError(
            "artifact_data_plane.accepted_policy_lag must cover async_training.staleness_threshold; "
            "this controls trajectory-consumption eligibility only and never relaxes KV cache identity isolation"
        )


def validate_transfer_admission_config(config) -> None:
    """Require the shared shadow owner before admission telemetry is enabled."""
    transfer_config = config.async_training.get("transfer_admission", {})
    enabled = transfer_config.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("async_training.transfer_admission.enabled must be a bool")
    if not enabled:
        return
    if not config.async_training.get("artifact_data_plane", {}).get("enabled", False):
        raise ValueError(
            "async_training.transfer_admission.enabled requires artifact_data_plane.enabled; "
            "the shared Ray admission owner is created with the artifact store"
        )
    admission = TransferAdmissionConfig.from_mapping(
        {
            key: value
            for key, value in transfer_config.items()
            if key not in {"enabled", "weight_sync_bytes_estimate"}
        }
    )
    if not admission.shadow:
        raise ValueError("online transfer admission is not implemented; enable shadow mode")


def validate_preemptible_handoff_config(config) -> None:
    handoff = config.async_training.get("preemptible_handoff", {})
    enabled = handoff.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("preemptible_handoff.enabled must be a bool")
    if not enabled:
        return
    artifact = config.async_training.get("artifact_data_plane", {})
    if not artifact.get("enabled", False) or not artifact.get("training_semantics_enabled", False):
        raise ValueError("preemptible handoff requires the Artifact training-semantic data plane")
    for name in ("lease_seconds", "snapshot_ttl_seconds", "safe_point_timeout_seconds"):
        value = float(handoff.get(name, 60.0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"preemptible_handoff.{name} must be finite and positive")


def validate_group_workload_scheduling_config(config) -> None:
    scheduling = config.async_training.get("group_workload_scheduling", {})
    enabled = scheduling.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("group_workload_scheduling.enabled must be a bool")
    if not enabled:
        return
    calibration_path = scheduling.get("calibration_path", "")
    if not isinstance(calibration_path, str) or not calibration_path:
        raise ValueError("group workload scheduling requires an independent calibration file")
    min_samples = scheduling.get("min_samples", 3)
    if type(min_samples) is not int or min_samples < 1:
        raise ValueError("group_workload_scheduling.min_samples must be a positive integer")


@ray.remote(num_cpus=1)
class FullyAsyncTaskRunner:
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        print("[ASYNC MAIN] Starting fully async PPO training...")
        # Keep the historical experimental config shape compatible with the
        # formal main_ppo entry point.
        config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
        config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config) -> None:
        print(f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        print("[ASYNC MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[ASYNC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        print("[ASYNC MAIN] Creating FullyAsyncTrainer first (needed for hybrid worker group injection)...")
        self._create_trainer(config)

        print("[ASYNC MAIN] Injecting trainer's worker group into rollouter for hybrid replicas...")
        self._setup_hybrid_worker_group(config)

        print("[ASYNC MAIN] Creating FullyAsyncRollouter...")
        self._create_rollouter(config)

        print("[ASYNC MAIN] Setting up rollouter reference on trainer")
        ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

        # sync total_train_steps between rollouter and trainer
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

        # max_queue_size
        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(self.components["rollouter"].set_message_queue_client.remote(self.components["message_queue_client"]))
        ray.get(self.components["trainer"].set_message_queue_client.remote(self.components["message_queue_client"]))

        artifact_config = config.async_training.get("artifact_data_plane", {})
        validate_artifact_data_plane_config(config)
        validate_transfer_admission_config(config)
        validate_preemptible_handoff_config(config)
        validate_group_workload_scheduling_config(config)
        if artifact_config.get("enabled", False):
            artifact_backend = artifact_config.get("backend", "ray_memory")
            if artifact_backend not in {"ray_memory", "mooncake"}:
                raise ValueError("artifact_data_plane.backend must be ray_memory or mooncake")
            artifact_store_actor = create_ray_rollout_artifact_store_actor(
                backend=artifact_backend,
                mooncake_config=artifact_config.get("mooncake", {}),
                trajectory_snapshot_enabled=bool(
                    config.async_training.get("preemptible_handoff", {}).get("enabled", False)
                ),
                transfer_admission_config=(
                    {
                        key: value
                        for key, value in config.async_training.get("transfer_admission", {}).items()
                        if key not in {"enabled", "weight_sync_bytes_estimate"}
                    }
                    if config.async_training.get("transfer_admission", {}).get("enabled", False)
                    else None
                ),
            ).remote()
            artifact_store_client = RayRolloutArtifactStoreClient(artifact_store_actor)
            self.components["artifact_store"] = artifact_store_actor
            self.components["artifact_store_client"] = artifact_store_client
            print(f"[ASYNC MAIN] Enabling {artifact_backend}-backed rollout artifact data plane")

            handoff_config = config.async_training.get("state_handoff", {})
            restore_id = handoff_config.get("restore_id", "")
            if restore_id:
                if artifact_backend != "mooncake":
                    raise ValueError("state_handoff.restore_id requires artifact_data_plane.backend=mooncake")
                # Restore producer history and artifact metadata before the new
                # rollouter registers its Ray worker incarnation.  Registration
                # then advances the producer generation and fences the old owner.
                ray.get(artifact_store_actor.restore_handoff_snapshot.remote(str(restore_id)))
                print(f"[ASYNC MAIN] Restored rollout state handoff: {restore_id}")

            ray.get(self.components["rollouter"].set_artifact_store_client.remote(artifact_store_client))
            ray.get(self.components["trainer"].set_artifact_store_client.remote(artifact_store_client))

        # param_version resume from ckpt or default 0
        ray.get(self.components["trainer"].load_checkpoint.remote())
        ray.get(self.components["rollouter"].load_checkpoint.remote())

        print("[ASYNC MAIN] Param sync before fit..")
        ray.get(self.components["trainer"]._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(self.components["trainer"]._fit_validate.remote(True))

        print("[ASYNC MAIN] All components initialized successfully")

    def _create_rollouter(self, config) -> None:
        print("[ASYNC MAIN] Starting create rollouter...")
        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        # set_hybrid_worker_group must be called BEFORE init_workers() so that
        # _init_async_rollout_manager can pass the hybrid WG to ALM.create().
        if "hybrid_worker_group" in self.components:
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
            print("[ASYNC MAIN] Hybrid worker group injected into rollouter")

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        print("[ASYNC MAIN] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")

    def _setup_hybrid_worker_group(self, config) -> None:
        """
        Extract the trainer's actor_rollout_wg and store it for later injection
        into the rollouter. This WG backs the hybrid rollout replicas
        used during trainer-side validation (use_trainer_do_validate).
        """
        trainer = self.components["trainer"]
        if config.async_training.use_trainer_do_validate:
            trainer_wg = ray.get(trainer.get_actor_wg.remote())
            self.components["hybrid_worker_group"] = trainer_wg
            print(
                f"[ASYNC MAIN] Hybrid worker group extracted from trainer "
                f"(world_size={getattr(trainer_wg, 'world_size', '?')})"
            )
        else:
            print("[ASYNC MAIN] use_trainer_do_validate=False, skipping hybrid worker group setup")

    def persist_rollout_state_handoff(self, handoff_id: str, timeout_seconds: float = 60.0):
        """Quiesce rollout publication and persist metadata into shared Mooncake."""
        if "artifact_store" not in self.components:
            raise RuntimeError("rollout state handoff requires artifact_data_plane")
        config = self.components["config"]
        artifact_config = config.async_training.get("artifact_data_plane", {})
        if artifact_config.get("backend", "ray_memory") != "mooncake":
            raise RuntimeError("persisted rollout state handoff requires the Mooncake artifact backend")
        try:
            quiesce = ray.get(
                self.components["rollouter"].quiesce_for_handoff.remote(timeout_seconds)
            )
        except Exception:
            ray.get(self.components["rollouter"].resume_after_handoff.remote())
            raise
        try:
            key = ray.get(
                self.components["artifact_store"].persist_handoff_snapshot.remote(handoff_id)
            )
        except Exception:
            # Snapshot persistence failed; return the existing owner to service
            # rather than leaving the training run silently frozen.
            ray.get(self.components["rollouter"].resume_after_handoff.remote())
            raise
        return {"handoff_id": handoff_id, "snapshot_key": key, **quiesce}

    def resume_rollout_after_handoff(self) -> None:
        if "rollouter" not in self.components:
            raise RuntimeError("rollouter is not initialized")
        ray.get(self.components["rollouter"].resume_after_handoff.remote())

    def delete_rollout_state_handoff(self, handoff_id: str) -> None:
        if "artifact_store" not in self.components:
            raise RuntimeError("rollout state handoff requires artifact_data_plane")
        ray.get(self.components["artifact_store"].delete_handoff_snapshot.remote(handoff_id))

    def _run_training_loop(self):
        self.running = True

        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollouter_future, trainer_future]

        try:
            while futures:
                # Use ray.wait to monitor all futures and return when any one is completed.
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    try:
                        ray.get(future)
                        print("[ASYNC MAIN] One component completed successfully")
                    except Exception as e:
                        print(f"[ASYNC MAIN] Component failed with error: {e}")
                        for remaining_future in remaining_futures:
                            ray.cancel(remaining_future)
                        raise e

                futures = remaining_futures

        except Exception as e:
            print(f"[ASYNC MAIN] Training failed: {e}")
            for future in futures:
                ray.cancel(future)
            raise
        finally:
            asyncio.run(self.components["message_queue_client"].clear_queue())
            print("[ASYNC MAIN] Training completed or interrupted")


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    from time import time

    start_time = time()
    auto_set_device(config)
    # TODO: unify rollout config with actor_rollout_ref
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
