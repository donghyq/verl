# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""

import asyncio
import hashlib
import json
import logging
import os
import time
import traceback
from dataclasses import replace
from typing import Any
from uuid import uuid4

import ray
import torch
import transfer_queue as tq
from pathlib import Path
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from .trajectory_control_dispatch import trajectory_control_api
from .trajectory_handoff_controller import TrajectoryControlError, TrajectoryHandoffController

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


async def _settle_session_tasks(tasks: list[asyncio.Task[Any]]) -> list[BaseException]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


def bind_token_kv_engine_urls(settings, engine_urls, *, worker_count: int):
    """Bind trajectory owners to actual native engine URLs without guessing.

    This is a static worker-to-engine configuration for Token/KV source control.
    It does not add cache-aware routing to the direct source SSE path. Routed
    streaming requests need separate actual-worker resolution before snapshots.
    """
    token_kv = settings.token_kv_resume
    if not isinstance(token_kv, dict) or not token_kv.get("enabled"):
        return token_kv
    if token_kv.get("engine_urls_by_owner"):
        return token_kv
    if (not isinstance(engine_urls, (list, tuple)) or not engine_urls
            or any(not isinstance(url, str) or not url for url in engine_urls)):
        raise ValueError("token-KV requires actual native engine URLs; refusing to guess an address")
    if type(worker_count) is not int or worker_count < 1:
        raise ValueError("token-KV owner binding requires at least one worker")
    # More workers than engines may share an engine. Preserve deterministic
    # worker order and leave source/target distinction to the caller's receipt.
    owner_urls = {
        f"{settings.run_id}/worker-{index}": engine_urls[index % len(engine_urls)]
        for index in range(worker_count)
    }
    return {**token_kv, "engine_urls_by_owner": owner_urls}


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()
        self.trajectory_entry = None
        self.trajectory_draining = False
        self.trajectory_checkpoint_paused = False
        self._trajectory_admissions_in_flight = 0
        self.trajectory_errors = {}

    async def configure_trajectory_state(self, client, owner_id, settings):
        """Bind the shared coordinator after the normal V1 worker initializes."""
        from .trajectory_entry import V1TrajectoryEntry
        from .trajectory_state.runtime import PromptGroupHandoffRuntime

        if self.trajectory_entry is not None:
            raise ValueError("trajectory state may only be configured once per worker incarnation")
        runtime = PromptGroupHandoffRuntime(
            client, owner_id=owner_id, incarnation_id=uuid4().hex,
            policy_identity=dict(settings.policy_identity),
            environment_version=settings.environment_version, tool_version=settings.tool_version,
            lease_seconds=settings.lease_seconds, snapshot_ttl_seconds=settings.snapshot_ttl_seconds,
            preemption_deadline_s=settings.safe_point_timeout_s,
            published_policy_client=client if settings.continuous_policy_publication else None,
        )
        self.trajectory_entry = V1TrajectoryEntry(self, runtime)
        if settings.token_kv_resume.get("enabled", False):
            from .trajectory_state.token_kv_source import configure_worker_token_kv

            configure_worker_token_kv(
                self, owner_id=owner_id, settings=settings.token_kv_resume,
                run_namespace=settings.run_id,
            )
        import os
        if os.environ.get("ROUTING_EXPERIMENT_OBSERVATION_DIR"):
            from routing_experiment_native_window import install_worker_observer
            install_worker_observer(self)

        evidence_root = os.environ.get("VERL_STORAGE_HANDOFF_EVIDENCE_ROOT")
        if evidence_root and not os.environ.get("VERL_STORAGE_HANDOFF_CAMPAIGN_SPEC"):
            from storage_handoff_driver import install_worker_observer

            install_worker_observer(self.trajectory_entry, evidence_root)

    async def routing_experiment_render_prompts(self, batch):
        """Read-only rendering through this paused Worker's original business factory."""
        from routing_experiment_native_window import render_original_worker_prompts
        return await render_original_worker_prompts(self, batch)
    async def storage_handoff_campaign(self, operation, spec=None, client=None):
        """Optional isolated campaign on this existing task-owned CPU worker."""
        if not os.environ.get("VERL_STORAGE_HANDOFF_CAMPAIGN_SPEC"):
            raise ValueError("storage handoff campaign is disabled")
        from storage_handoff_campaign import worker_rpc

        return await worker_rpc(self, operation, spec, client)

    async def request_trajectory_preemption(self, uid):
        if self.trajectory_entry is None:
            return False
        state = self.trajectory_entry.runtime.group_status(uid)
        if state is None or not any(state.get(key) for key in (
            "pending_members", "starting_members", "running_members", "checkpointed_members"
        )):
            return False
        # Close worker admission before the first remote await. A trajectory can
        # complete while the signal is being sent; the source still stays closed.
        self.trajectory_draining = True
        await self.trajectory_entry.runtime.request_preemption(uid)
        self.trajectory_entry.release_acceptance_boundaries(uid)
        return True

    def trajectory_group_status(self, uid):
        state = self.trajectory_entry.runtime.group_status(uid) if self.trajectory_entry is not None else None
        if uid in self.trajectory_errors:
            return {"state": "blocked", "entry_error": self.trajectory_errors[uid], "runtime": state}
        if state is not None:
            state = {**state, **self.trajectory_entry.acceptance_boundary_status(uid)}
        return state

    def trajectory_generation_progress(self, uid=None):
        """Read actual SSE generation progress from this worker's source backend."""
        backend = getattr(self, "token_kv_source_backend", None)
        progress = backend.generation_progress() if backend is not None else {}
        if uid is not None:
            progress = {uid: progress.get(uid, {})}
        target = getattr(self, "token_kv_resume_backend", None)
        progress["__target_engine_url__"] = getattr(target, "engine_url", None)
        progress["__admission_open__"] = not getattr(self, "trajectory_draining", False)
        return progress

    def set_trajectory_checkpoint_paused(self, paused):
        if type(paused) is not bool:
            raise ValueError("checkpoint admission state must be boolean")
        self.trajectory_checkpoint_paused = paused

    async def trajectory_checkpoint_status(self):
        runtime = self.trajectory_entry.runtime if self.trajectory_entry is not None else None
        active = len(runtime._groups) + len(runtime._active_targets) if runtime is not None else 0
        pending = len(self.background_tasks) + getattr(self, "_trajectory_admissions_in_flight", 0)
        if runtime is not None and not pending and not active:
            await runtime.retry_normal_cleanup()
        return {"background_tasks": pending, "active_groups": active,
                "group_uids": list(runtime._groups) if runtime is not None else [],
                "cleanup_pending": len(runtime._normal_cleanup_pending) if runtime is not None else 0,
                "errors": dict(self.trajectory_errors)}

    async def resume_after_trajectory_reclaim(self):
        status = await self.trajectory_checkpoint_status()
        if status["errors"] or any(status[key] for key in ("background_tasks", "active_groups", "cleanup_pending")):
            raise RuntimeError("a source worker cannot reopen before its accepted work has settled")
        self.trajectory_draining = False

    def trajectory_acceptance_source_records(self, uid):
        """Fixture-only record of the actual source lease and committed snapshot."""
        if self.trajectory_entry is None:
            return None
        return self.trajectory_entry.acceptance_source_records(uid)

    async def resume_trajectory_handoff(self, handoff_id):
        if self.trajectory_entry is None or self.trajectory_draining:
            raise ValueError("takeover requires a configured worker that still accepts work")
        result, receipt = await self.trajectory_entry.resume_handoff(handoff_id)
        # The publication adapter verifies the original prompt's finished tag
        # before persisting FINISHED/GC. Never publish that tag a second time.
        return receipt

    async def run_trajectory_acceptance_fixture(self, prompt, sampling_params, trajectory, *, start_members,
                                              checkpoint_members=None):
        """Driver-only admission control for the fixed mixed-group acceptance.

        Default training never supplies start_members. No scheduling policy or
        synthetic tokens are added by this fixture.
        """
        if (self.trajectory_entry is None or self.trajectory_draining
                or getattr(self, "trajectory_checkpoint_paused", False)):
            raise ValueError("the fixture needs a configured source worker")
        uid = prompt["uid"]
        self.trajectory_entry.enable_acceptance_source_observation(uid, checkpoint_members=checkpoint_members)
        await tq.async_kv_put(key=uid, partition_id="train", tag={"status": "running"})
        result = await self.trajectory_entry.run_prompt(
            prompt, sampling_params, trajectory, start_members=start_members
        )
        if result.status == "completed":
            await tq.async_kv_put(key=uid, partition_id="train", tag={"status": "finished"})
            await self.trajectory_entry.confirm_normal_publication(result)
        return {"status": result.status, "handoff_id": result.handoff_id, "metrics": result.metrics}

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        if getattr(self, "trajectory_draining", False):
            raise RuntimeError("source worker has stopped admission for trajectory handoff")
        if getattr(self, "trajectory_checkpoint_paused", False):
            raise RuntimeError("trajectory checkpoint has stopped new admission")
        self._trajectory_admissions_in_flight = getattr(self, "_trajectory_admissions_in_flight", 0) + 1
        try:
            await self._generate_sequences_accepted(batch)
        finally:
            self._trajectory_admissions_in_flight -= 1

    async def _generate_sequences_accepted(self, batch: TensorDict) -> None:
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        # the trainer never consumes the sampler head on validation rollouts
        if config.topk_log_probs and not validate:
            sampling_params["topk_log_probs"] = config.topk_log_probs

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        if getattr(self, "trajectory_entry", None) is not None and not trajectory["validate"]:
            await self._run_durable_prompt(prompt, sampling_params, trajectory, trace)
            return
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        tasks = []
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params, trajectory=trajectory, trace=trace, session_id=i, **prompt
                    )
                )
                tasks.append(task)

            # Publish a terminal status only after every session settles, so no sibling can write after
            # ReplayBuffer clears a failed group.
            session_errors = await _settle_session_tasks(tasks)
            if session_errors:
                for error in session_errors:
                    logger.error(
                        f"Error in _run_prompt for uid={uid}",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            if tasks:
                await _settle_session_tasks(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _run_durable_prompt(self, prompt, sampling_params, trajectory, trace):
        uid = prompt["uid"]
        await tq.async_kv_put(key=uid, partition_id="train", tag={"status": "running"})
        try:
            result = await self.trajectory_entry.run_prompt(prompt, sampling_params, trajectory, trace=trace)
            if result.status == "completed":
                await tq.async_kv_put(key=uid, partition_id="train", tag={"status": "finished"})
                await self.trajectory_entry.confirm_normal_publication(result)
            elif result.status != "handed_off":
                raise ValueError("unexpected durable trajectory result")
            # A handed-off group stays running. Only the target's verified final
            # TQ publication may expose its completed prompt status.
        except Exception as error:
            logger.exception("Durable prompt failed before completed publication: uid=%s", uid)
            self.trajectory_draining = True
            self.trajectory_errors[uid] = type(error).__name__
            details = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_traceback": "".join(traceback.format_exception(error)),
            }
            self.trajectory_error_details = getattr(self, "trajectory_error_details", {})
            self.trajectory_error_details[uid] = details
            # A lost coordinator reply may hide a committed publication. Keep
            # the group running, preserving its durable ownership/history and
            # forbidding automatic eviction or reissue. Tell the trainer to
            # stop this run explicitly instead of waiting for samples forever.
            await tq.async_kv_put(key=uid, partition_id="train", tag={
                "status": "running", "durable_source_error": True, **details,
            })

    async def _prepare_agent_loop_outputs(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ):
        """Build the stock V1 fields; publication may verify these exact bytes."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        # TODO: Support output:list[AgentLoopOutput]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    # These tags are used for off-policy staleness control, if a trajectory
                    # spans too many global steps, we need to filter it out.
                    # global_steps: which global steps this sample is from dataloader
                    "global_steps": kwargs["global_steps"],
                    # min_global_steps: start generation model weights version of this trajectory
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    # max_global_steps: end generation model weights version of this trajectory
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )

        return keys, list_of_dict_to_tensordict(fields), tags

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        """Put agent loop outputs into TransferQueue using the unchanged schema."""
        prepared = await self._prepare_agent_loop_outputs(output, validate, **kwargs)
        if prepared is not None:
            keys, fields, tags = prepared
            await tq.async_kv_batch_put(
                keys=keys, fields=fields, tags=tags, partition_id="train" if not validate else "val"
            )


class AgentLoopManagerTQ(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        self.trajectory_coordinator = None
        self._trajectory_coordinator_client = None
        self._draining_worker_indices = set()
        self.trajectory_worker_owner_ids = {}
        self.trajectory_control_timeout_s = None
        self.trajectory_resume_monitor_timeout_s = None
        self._trajectory_monitor_tasks = set()
        self._trajectory_monitor_uids = set()
        self.trajectory_status_poll_interval_s = 0.05
        self._trajectory_checkpoint_paused = False
        self._trajectory_reserved_target = None
        self._trajectory_reclaim_receipt = None
        self._trajectory_controller = TrajectoryHandoffController(self)
        super().__init__(*args, **kwargs)

    @property
    def trajectory_coordinator_client(self):
        # Keep the historical raw ActorHandle available to validation drivers.
        return getattr(self, "_trajectory_coordinator_client", None) or self.trajectory_coordinator

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        await instance._init_trajectory_state()
        return instance

    async def _init_trajectory_state(self):
        from .trajectory_entry import state_settings

        settings = state_settings(self.config)
        if settings is None:
            return
        if settings.shared_gpu_handoff:
            target = settings.handoff_target_worker_index
            if len(self.agent_loop_workers) < 2 or not 0 <= target < len(self.agent_loop_workers):
                raise ValueError("shared GPU handoff requires a distinct reserved takeover worker")
            self._trajectory_reserved_target = target
        from .trajectory_state import RayTrajectoryCoordinatorClient, create_ray_trajectory_coordinator_actor

        # A duplicate live coordinator for the same run fails closed. Ray may
        # restart this exact actor; metadata and objects survive in root_dir.
        Actor = create_ray_trajectory_coordinator_actor(max_restarts=settings.coordinator_max_restarts)
        import ray

        from .trajectory_checkpoint import coordinator_actor_name

        actor_name = coordinator_actor_name(settings.run_id, settings.root_dir)
        try:
            self.trajectory_coordinator = ray.get_actor(actor_name, namespace="verl-trajectory-state")
        except ValueError:
            self.trajectory_coordinator = Actor.options(
                name=actor_name, namespace="verl-trajectory-state"
            ).remote(settings.root_dir, settings.run_id)
        client = RayTrajectoryCoordinatorClient(self.trajectory_coordinator)
        self._trajectory_coordinator_client = client
        summary = await client.summary()
        from pathlib import Path

        if (summary.get("run_id") != settings.run_id
                or summary.get("root_dir") != str(Path(settings.root_dir).resolve())):
            raise ValueError("existing trajectory coordinator run_id/root_dir differs from this rollout run")
        settings = replace(settings, token_kv_resume=bind_token_kv_engine_urls(
            settings, getattr(self.llm_client, "engine_urls", ()),
            worker_count=len(self.agent_loop_workers),
        ))
        await asyncio.gather(*[
            worker.configure_trajectory_state.remote(client, f"{settings.run_id}/worker-{i}", settings)
            for i, worker in enumerate(self.agent_loop_workers)
        ])
        self.trajectory_worker_owner_ids = {
            f"{settings.run_id}/worker-{i}": i for i in range(len(self.agent_loop_workers))
        }
        self.trajectory_control_timeout_s = settings.control_timeout_s
        self.trajectory_resume_monitor_timeout_s = settings.resume_monitor_timeout_s
        try:
            await client.collect_garbage()
        except Exception:
            logger.warning("Previously queued trajectory cleanup remains pending", exc_info=True)

    @auto_await
    async def request_trajectory_preemption(self, uid):
        results = await asyncio.gather(*[
            worker.request_trajectory_preemption.remote(uid) for worker in self.agent_loop_workers
        ])
        owners = [i for i, accepted in enumerate(results) if accepted]
        if len(owners) > 1:
            raise RuntimeError("more than one source worker claimed the same prompt")
        self._draining_worker_indices.update(owners)
        return owners[0] if owners else None

    @auto_await
    async def trajectory_group_status(self, uid):
        results = await asyncio.gather(*[
            worker.trajectory_group_status.remote(uid) for worker in self.agent_loop_workers
        ])
        return {i: result for i, result in enumerate(results) if result is not None}

    @auto_await
    async def trajectory_generation_progress(self, uid=None):
        """Read-only per-worker progress backed by actual generation SSE rows."""
        results = await asyncio.gather(*[
            worker.trajectory_generation_progress.remote(uid) for worker in self.agent_loop_workers
        ])
        return {i: result for i, result in enumerate(results) if result}

    @auto_await
    async def resume_trajectory_handoff(self, handoff_id, *, target_worker_index):
        if (type(target_worker_index) is not int or not 0 <= target_worker_index < len(self.agent_loop_workers)
                or target_worker_index in self._draining_worker_indices):
            raise ValueError("driver must select a currently available target worker")
        return await self.agent_loop_workers[target_worker_index].resume_trajectory_handoff.remote(handoff_id)

    @staticmethod
    def _trajectory_prompt_uids(prompts):
        if prompts is None or "uid" not in prompts:
            return []
        values = prompts["uid"]
        if isinstance(values, NonTensorStack):
            return [row.data for row in values]
        if isinstance(values, NonTensorData):
            return [values.data]
        if isinstance(values, torch.Tensor):
            return values.tolist()
        if isinstance(values, (list, tuple)):
            return list(values)
        raise ValueError("prompt uid must be a tensor or non-tensor batch")

    @trajectory_control_api
    async def _start_trajectory_completion_monitors(self, prompts):
        """Finish a handed-off prompt in the same normal training admission.

        The source worker may commit a token-KV snapshot itself. This manager
        monitor never requests preemption; it only waits for either normal
        completion or a released handoff, then dispatches the original target
        resume/publication path once.
        """
        if self.trajectory_coordinator is None:
            return
        loop = asyncio.get_running_loop()
        timeout_s = self.trajectory_resume_monitor_timeout_s
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("trajectory resume monitor timeout is invalid")
        for uid in self._trajectory_prompt_uids(prompts):
            if not isinstance(uid, str) or not uid or uid in self._trajectory_monitor_uids:
                continue
            self._trajectory_monitor_uids.add(uid)

            async def monitor(current_uid):
                try:
                    result = await self._trajectory_controller.run(
                        current_uid, target_worker_index=None, timeout_s=timeout_s, require_active=False
                    )
                    receipt_dir = os.environ.get("VERL_TRAJECTORY_RECEIPT_DIR")
                    if result.get("status") == "resumed" and receipt_dir:
                        path = Path(receipt_dir) / (
                            hashlib.sha256(current_uid.encode("utf-8")).hexdigest()[:16] + ".json"
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps({
                            "uid": current_uid, "status": result["status"],
                            "handoff_id": result.get("handoff_id"),
                            "source_worker_index": result.get("source_worker_index"),
                            "target_worker_index": result.get("target_worker_index"),
                            "receipt": result.get("receipt"),
                        }, ensure_ascii=False, indent=2) + "\n")
                except Exception:
                    logger.exception("Normal trajectory completion monitor failed: uid=%s", current_uid)
                finally:
                    self._trajectory_monitor_uids.discard(current_uid)

            task = loop.create_task(monitor(uid))
            self._trajectory_monitor_tasks.add(task)
            task.add_done_callback(self._trajectory_monitor_tasks.discard)

    @trajectory_control_api
    async def preempt_and_resume_trajectory(self, uid, *, target_worker_index=None, timeout_s=None):
        """Cooperatively hand off one active prompt group and publish once."""
        if self.trajectory_coordinator is None:
            raise TrajectoryControlError("disabled", uid=uid)
        if (getattr(self, "_trajectory_checkpoint_paused", False)
                and self._trajectory_controller.status(uid) is None):
            raise TrajectoryControlError("checkpoint_paused", uid=uid)
        return await self._trajectory_controller.run(
            uid, target_worker_index=target_worker_index,
            timeout_s=self.trajectory_control_timeout_s if timeout_s is None else timeout_s,
        )

    @auto_await
    async def get_handoff_status(self, handoff_id):
        if self.trajectory_coordinator is None:
            raise TrajectoryControlError("disabled")
        return await self.trajectory_coordinator_client.get_handoff_status(handoff_id)

    @auto_await
    async def list_recoverable_handoffs(self):
        if self.trajectory_coordinator is None:
            raise TrajectoryControlError("disabled")
        return await self.trajectory_coordinator_client.list_recoverable_handoffs()

    @trajectory_control_api
    async def recover_trajectory_handoffs(self, *, target_worker_index=None, timeout_s=None):
        """Controlled startup recovery after the first engine identity publication."""
        if self.trajectory_coordinator is None:
            raise TrajectoryControlError("disabled")
        try:
            await self.trajectory_coordinator_client.get_active_published_policy()
        except Exception as error:
            raise TrajectoryControlError(
                "policy_not_published",
                detail="call this only after the initial engine weights are actually published",
            ) from error
        handoffs = await self.trajectory_coordinator_client.list_recoverable_handoffs()
        from .trajectory_state.handoff import HandoffState

        summary = await self.trajectory_coordinator_client.summary()
        ready_ids = {row["handoff_id"] for row in handoffs}
        unresolved = [handoff_id for handoff_id, state in summary["handoffs"].items()
                      if state != HandoffState.FINISHED.value and handoff_id not in ready_ids]
        if unresolved:
            raise TrajectoryControlError(
                "startup_unsettled",
                detail="uncommitted or unreleased handoffs require reconciliation before prompt reissue",
            )
        results = []
        for handoff in handoffs:
            source_index = self.trajectory_worker_owner_ids.get(handoff.get("source_rollouter"))
            results.append(await self._trajectory_controller.run(
                handoff["group_uid"], target_worker_index=target_worker_index,
                timeout_s=self.trajectory_control_timeout_s if timeout_s is None else timeout_s,
                source_worker_index=source_index, require_active=False,
            ))
        return results

    @trajectory_control_api
    async def trajectory_operation_status(self, uid):
        if self.trajectory_coordinator is None:
            raise TrajectoryControlError("disabled", uid=uid)
        return self._trajectory_controller.status(uid)

    @trajectory_control_api
    async def prepare_trajectory_checkpoint(self, *, timeout_s=None):
        """Stop new admission and wait for every accepted group to be published."""
        if self.trajectory_coordinator is None:
            raise TrajectoryControlError("disabled")
        timeout = self._trajectory_controller._validate_timeout(
            self.trajectory_control_timeout_s if timeout_s is None else timeout_s
        )
        if getattr(self, "_trajectory_checkpoint_paused", False):
            raise TrajectoryControlError("checkpoint_already_paused")
        self._trajectory_checkpoint_paused = True
        deadline = time.monotonic() + timeout

        async def wait_until_ready():
            await asyncio.gather(*[
                worker.set_trajectory_checkpoint_paused.remote(True) for worker in self.agent_loop_workers
            ])
            while True:
                workers = await asyncio.gather(*[
                    worker.trajectory_checkpoint_status.remote() for worker in self.agent_loop_workers
                ])
                if any(row["errors"] for row in workers):
                    raise TrajectoryControlError("checkpoint_worker_failed")
                summary = await self.trajectory_coordinator_client.summary()
                controls = any(not operation.task.done()
                               for operation in self._trajectory_controller._operations.values())
                if (not controls and all(not any(row[key] for key in
                        ("background_tasks", "active_groups", "cleanup_pending")) for row in workers)
                        and not summary["active_execution_leases"] and not summary["unsettled_publications"]
                        and not summary["policy_publication_blocked"] and not summary["handoff_policy_barrier"]
                        and all(state == "finished" for state in summary["handoffs"].values())):
                    return summary
                await asyncio.sleep(min(self.trajectory_status_poll_interval_s,
                                        max(0, deadline - time.monotonic())))

        try:
            return await asyncio.wait_for(wait_until_ready(), timeout)
        except asyncio.TimeoutError as error:
            raise TrajectoryControlError("checkpoint_timeout") from error

    @trajectory_control_api
    async def export_trajectory_checkpoint(self, destination, checkpoint_step):
        if not getattr(self, "_trajectory_checkpoint_paused", False):
            raise TrajectoryControlError("checkpoint_not_paused")
        return await self.trajectory_coordinator_client.export_checkpoint(destination, checkpoint_step)

    @trajectory_control_api
    async def resume_trajectory_checkpoint_admission(self):
        # Source workers keep their independent draining flag after a handoff.
        await asyncio.gather(*[
            worker.set_trajectory_checkpoint_paused.remote(False) for worker in self.agent_loop_workers
        ])
        self._trajectory_checkpoint_paused = False

    @trajectory_control_api
    async def prepare_trajectory_resource_reclaim(self, *, timeout_s=None):
        """Publish every accepted group before permitting physical GPU reclamation."""
        target = getattr(self, "_trajectory_reserved_target", None)
        if self.trajectory_coordinator is None or target is None:
            raise TrajectoryControlError("shared_gpu_handoff_disabled")
        if getattr(self, "_trajectory_checkpoint_paused", False):
            raise TrajectoryControlError("checkpoint_already_paused")
        timeout = self._trajectory_controller._validate_timeout(
            self.trajectory_control_timeout_s if timeout_s is None else timeout_s
        )
        self._trajectory_checkpoint_paused = True
        deadline = time.monotonic() + timeout
        originally_draining = set(self._draining_worker_indices)
        receipts = []

        async def settle():
            await asyncio.gather(*[
                worker.set_trajectory_checkpoint_paused.remote(True) for worker in self.agent_loop_workers
            ])
            while True:
                workers = await asyncio.gather(*[
                    worker.trajectory_checkpoint_status.remote() for worker in self.agent_loop_workers
                ])
                if any(row["errors"] for row in workers):
                    raise TrajectoryControlError("reclaim_worker_failed")
                # A separate takeover worker remains available while sources drain.
                # One durable policy barrier is held at a time; each receipt is
                # verified by the controller before the next source is signaled.
                for index, row in enumerate(workers):
                    if index == target:
                        continue
                    for uid in row["group_uids"]:
                        receipt = await self._trajectory_controller.run(
                            uid, target_worker_index=target,
                            timeout_s=max(.001, deadline - time.monotonic()),
                        )
                        if receipt["status"] not in {"resumed", "already_completed"}:
                            raise TrajectoryControlError("reclaim_handoff_unconfirmed", uid=uid)
                        receipts.append(receipt)
                summary = await self.trajectory_coordinator_client.summary()
                controls = any(not operation.task.done()
                               for operation in self._trajectory_controller._operations.values())
                if (not controls and all(not any(row[key] for key in
                        ("background_tasks", "active_groups", "cleanup_pending")) for row in workers)
                        and not summary["active_execution_leases"] and not summary["unsettled_publications"]
                        and not summary["policy_publication_blocked"] and not summary["handoff_policy_barrier"]
                        and all(state == "finished" for state in summary["handoffs"].values())):
                    result = {"status": "quiesced", "operation_id": uuid4().hex, "receipts": receipts,
                              "reopen_worker_indices": sorted(self._draining_worker_indices - originally_draining)}
                    self._trajectory_reclaim_receipt = result
                    return result
                await asyncio.sleep(min(self.trajectory_status_poll_interval_s,
                                        max(0, deadline - time.monotonic())))

        try:
            return await asyncio.wait_for(settle(), timeout)
        except asyncio.TimeoutError as error:
            # Admission stays closed. An uncertain target PUT must settle before
            # retry or memory release; timeout never authorizes sleep_replicas.
            raise TrajectoryControlError("reclaim_timeout") from error

    @trajectory_control_api
    async def finish_trajectory_resource_reclaim(self, receipt):
        expected = getattr(self, "_trajectory_reclaim_receipt", None)
        if (expected is None or receipt != expected or receipt.get("status") != "quiesced"
                or not getattr(self, "_trajectory_checkpoint_paused", False)):
            raise TrajectoryControlError("reclaim_not_quiesced")
        for index in expected["reopen_worker_indices"]:
            await self.agent_loop_workers[index].resume_after_trajectory_reclaim.remote()
            self._draining_worker_indices.discard(index)
        await self.resume_trajectory_checkpoint_admission()
        self._trajectory_reclaim_receipt = None

    def generate_sequences(self, prompts: TensorDict) -> None:
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        if getattr(self, "_trajectory_checkpoint_paused", False):
            raise RuntimeError("trajectory checkpoint has stopped new admission")
        workers = [worker for i, worker in enumerate(self.agent_loop_workers)
                   if i not in self._draining_worker_indices
                   and i != getattr(self, "_trajectory_reserved_target", None)]
        if not workers:
            raise RuntimeError("all trajectory workers have stopped admission")
        chunkes = prompts.chunk(len(workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(workers, chunkes, strict=False)
            ]
        )
        self._start_trajectory_completion_monitors(prompts)
