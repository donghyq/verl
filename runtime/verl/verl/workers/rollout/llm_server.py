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
"""
Utility classes for manage and request LLM servers:
- LLMServerManager: manage life-cycle of LLM servers, including launch, tear-down replicas.
- LLMServerClient: proxy client to request LLM servers, used by AgentLoopWorker.
"""

import asyncio
import logging
import os
from copy import deepcopy
from typing import Any, Iterable, Optional
from uuid import uuid4

import numpy as np
import ray
from omegaconf import DictConfig

from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils import normalize_token_ids
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tracking import RLInsightLogger
from verl.workers.rollout.replica import RolloutReplica, TokenOutput, get_rollout_replica_class
from verl.workers.rollout.router import GlobalRequestLoadBalancer  # noqa: F401
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _engine_url(address: str) -> str:
    return address.rstrip("/") if address.startswith(("http://", "https://")) else f"http://{address}"


def _trajectory_state_settings(config: DictConfig):
    rollout = getattr(getattr(config, "actor_rollout_ref", None), "rollout", None)
    settings = getattr(rollout, "trajectory_state", None)
    return settings


def continuous_policy_publication_enabled(config: DictConfig) -> bool:
    settings = _trajectory_state_settings(config)
    return bool(settings is not None and getattr(settings, "enabled", False)
                and getattr(settings, "continuous_policy_publication", False))


def _policy_publication_requirements(config: DictConfig) -> None:
    if not continuous_policy_publication_enabled(config):
        return
    rollout = config.actor_rollout_ref.rollout
    settings = rollout.trajectory_state
    if not getattr(rollout.sglang_gateway, "enabled", False):
        raise ValueError("continuous_policy_publication requires sglang_gateway.enabled=True")
    if rollout.name != "sglang":
        raise ValueError("continuous_policy_publication requires rollout.name=sglang")
    for name in ("run_id", "model_configuration_fingerprint"):
        value = getattr(settings, name, None)
        if not isinstance(value, str) or not value:
            raise ValueError(f"trajectory_state.{name} is required for continuous_policy_publication")


def _policy_identity_dict(config: DictConfig) -> dict[str, str]:
    settings = _trajectory_state_settings(config)
    identity = getattr(settings, "policy_identity", None)
    if not isinstance(identity, dict):
        identity = dict(identity)
    required = {
        "policy_version",
        "model_weights_fingerprint",
        "tokenizer_template_fingerprint",
        "kv_cache_schema_fingerprint",
    }
    if set(identity) != required or any(not isinstance(value, str) or not value for value in identity.values()):
        raise ValueError("trajectory_state.policy_identity must contain the complete canonical policy identity")
    return dict(identity)


def _coordinator_actor_name(run_id: str, root_dir: str) -> str:
    from verl.trainer.ppo.v1.trajectory_checkpoint import coordinator_actor_name

    return coordinator_actor_name(run_id, root_dir)


def get_or_create_trajectory_coordinator(config: DictConfig):
    settings = _trajectory_state_settings(config)
    if settings is None or not getattr(settings, "enabled", False):
        return None
    from verl.trainer.ppo.v1.trajectory_state import (
        RayTrajectoryCoordinatorClient,
        create_ray_trajectory_coordinator_actor,
    )

    actor_name = _coordinator_actor_name(settings.run_id, settings.root_dir)
    try:
        actor = ray.get_actor(actor_name, namespace="verl-trajectory-state")
        client = RayTrajectoryCoordinatorClient(actor)
        return actor, client
    except ValueError:
        Actor = create_ray_trajectory_coordinator_actor(max_restarts=settings.coordinator_max_restarts)
        actor = Actor.options(name=actor_name, namespace="verl-trajectory-state").remote(
            settings.root_dir, settings.run_id
        )
        client = RayTrajectoryCoordinatorClient(actor)
        return actor, client


async def _validate_coordinator_binding(client, *, run_id: str, root_dir: str):
    from pathlib import Path

    summary = await client.summary()
    if summary.get("run_id") != run_id or summary.get("root_dir") != str(Path(root_dir).resolve()):
        raise ValueError("trajectory coordinator run_id/root_dir differs from the configured rollout run")
    return summary


def _validate_coordinator_binding_sync(client, *, run_id: str, root_dir: str):
    from pathlib import Path

    summary = ray.get(client.actor.summary.remote())
    if summary.get("run_id") != run_id or summary.get("root_dir") != str(Path(root_dir).resolve()):
        raise ValueError("trajectory coordinator run_id/root_dir differs from the configured rollout run")
    return summary


async def observe_owned_engine_versions(addresses: list[str], *, timeout_s: float) -> list[dict[str, str]]:
    import aiohttp

    urls = [address.rstrip("/") if address.startswith(("http://", "https://")) else f"http://{address}"
            for address in addresses]
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def read(url: str) -> dict[str, str]:
            async with session.get(f"{url}/model_info", allow_redirects=False) as response:
                if response.status != 200:
                    raise RuntimeError(f"owned engine {url} /model_info returned HTTP {response.status}")
                payload = await response.json()
                version = payload.get("weight_version")
                if not isinstance(version, str) or not version:
                    raise ValueError("owned engine /model_info returned no weight_version")
                return {"engine_url": url, "weight_version": version, "source": "model_info"}

        return await asyncio.gather(*(read(url) for url in urls))


class ContinuousPublicationTransport:
    """Bind active published policy identity into native Gateway cache_salt."""

    def __init__(self, transport, publication_client, *, run_id: str):
        self._transport = transport
        self._publication_client = publication_client
        self._run_id = run_id

    def __getattr__(self, name):
        try:
            transport = object.__getattribute__(self, "_transport")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(transport, name)

    async def generate(self, *, cache_salt: str | None = None, **kwargs):
        from verl.trainer.ppo.v1.trajectory_state.identity import CanonicalPolicyKVIdentity
        from verl.workers.rollout.sglang_gateway_client import GatewayProtocolError

        record = await self._publication_client.get_active_published_policy()
        identity = record["identity"]
        version = identity["policy_version"]
        digest = CanonicalPolicyKVIdentity(**identity).cache_salt
        expected = self._transport.build_policy_cache_salt(
            run_namespace=self._run_id,
            policy_version=version,
            policy_identity_salt=digest,
        )
        if cache_salt is not None and cache_salt != expected:
            raise GatewayProtocolError("caller cache_salt differs from the active published policy namespace")
        return await self._transport.generate(cache_salt=expected, **kwargs)


class LLMServerClient:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least in-flight requests load balancing via global coordination
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle = None,
        generation_transport: Any = None,
        engine_urls: Iterable[str] | None = None,
        **kwargs,
    ):
        """Initialize the LLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor
                that also holds the server-handle registry. Optional; subclasses that
                manage server routing externally can pass None.
        """
        self.config = config
        self._load_balancer = load_balancer_handle
        self._generation_transport = generation_transport
        self._engine_urls = tuple(engine_urls) if engine_urls is not None else ()
        # Balancer field declarations, queried lazily once.
        self._lb_require_acquire_fields: list[str] | None = None
        self._lb_require_release_fields: list[str] | None = None

    @property
    def engine_urls(self) -> tuple[str, ...]:
        """Native manager endpoints available to explicit recovery control."""
        return self._engine_urls


    async def _acquire_server(self, request_id: str, **extra) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        # Only the declared fields are serialized.
        if self._lb_require_acquire_fields is None:
            acquire_fields, release_fields = await asyncio.gather(
                self._load_balancer.require_acquire_fields.remote(),
                self._load_balancer.require_release_fields.remote(),
            )
            self._lb_require_acquire_fields = list(acquire_fields)
            self._lb_require_release_fields = list(release_fields)
        fields = {name: extra[name] for name in self._lb_require_acquire_fields if name in extra}
        return await self._load_balancer.acquire_server.remote(request_id=request_id, **fields)

    def _release_server(self, server_id: str, request_id: str | None = None) -> None:
        # Fire-and-forget: release is just a counter decrement, no need to await.
        # Awaiting here risks blocking the finally clause if the LB actor is unresponsive.
        pool = {"request_id": request_id}
        fields = {name: pool[name] for name in self._lb_require_release_fields if name in pool}
        self._load_balancer.release_server.remote(server_id=server_id, **fields)

    def _vllm_request_id(self, request_id: str) -> str:
        # request_id passed to vLLM. Default: a fresh uuid per turn so each turn
        # is an independent vLLM request. Under full_determinism the caller's
        # request_id is passed straight through so vLLM sees a stable id across runs.
        if getattr(self.config.actor_rollout_ref.rollout, "full_determinism", False):
            return request_id
        return uuid4().hex

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput | DiffusionOutput: token or diffusion output
        """
        if getattr(self, "_generation_transport", None) is not None:
            # Replica choice and counters belong to the native Gateway. The
            # FullyAsync subclass still owns cumulative tokens and abort/resume.
            mm_processor_output = kwargs.pop("mm_processor_output", None)
            if mm_processor_output is not None:
                raise ValueError("Multimodal processor output is not enabled for native Gateway transport")
            return await self._generation_transport.generate(
                request_id=self._vllm_request_id(request_id),
                prompt_ids=normalize_token_ids(prompt_ids),
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **kwargs,
            )

        server_id, server = await self._acquire_server(
            request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            **kwargs,
        )
        try:
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            # SGLang cannot take raw frames: its video_data accepts only a path/url/base64 or a
            # processor_output dict. Always hand it the pre-computed payload (None when the model's
            # processor produced no pixel_values_videos, e.g. a non-Qwen-VL family) and never the raw
            # frames -- dropping video beats sending SGLang tensors it cannot parse. vLLM keeps the
            # frames and never enters this branch. Neither server signature accepts **kwargs, so pop.
            mm_processor_output = kwargs.pop("mm_processor_output", None)
            if self.config.actor_rollout_ref.rollout.name == "sglang":
                video_data = mm_processor_output
            # priority is only supported by vLLM rollout server.
            priority = kwargs.pop("priority", 0)
            priority_kwargs = (
                {"priority": priority} if priority != 0 and self.config.actor_rollout_ref.rollout.name == "vllm" else {}
            )
            output: TokenOutput = await server.generate.remote(
                request_id=self._vllm_request_id(request_id),  # use new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **priority_kwargs,
                **kwargs,
            )
            global_steps = output.extra_fields.get("global_steps")
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            return output
        finally:
            self._release_server(
                server_id,
                request_id=request_id,
            )


class FullyAsyncLLMServerClient(LLMServerClient):
    """FullyLLMServerClient supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle = None,
        only_hybrid: bool = False,
        **kwargs,
    ):
        """Initialize the FullyAsyncLLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor
                that also holds the server-handle registry.
            only_hybrid (bool): When ``True``, hybrid replicas are the *only* rollout
                resource.  If the load balancer is temporarily empty (e.g. during
                weight synchronisation) :meth:`_acquire_server` will keep retrying
                every 1 second instead of raising immediately.
        """
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._only_hybrid = only_hybrid

    async def _acquire_server(self, request_id: str, **extra) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        # When only_hybrid is True, hybrid replicas are the sole rollout resource and
        # the LB may be temporarily empty during weight sync / scaling transitions.
        # In that case keep retrying every 1 s until a server becomes available.
        # Otherwise raise immediately so callers see the error right away.
        while True:
            try:
                return await super()._acquire_server(request_id, **extra)
            except RuntimeError as e:
                if "No available servers in load balancer" in str(e) and self._only_hybrid:
                    await asyncio.sleep(1)
                else:
                    raise

    def _configured_response_length(self) -> Optional[int]:
        """Per-response token budget from the rollout config, or ``None`` when unavailable.

        Tests and lightweight callers may pass a config stub without the rollout section; in that
        case the resume loop keeps its previous behaviour of deferring to the server default.
        """
        rollout_config = getattr(getattr(self.config, "actor_rollout_ref", None), "rollout", None)
        response_length = getattr(rollout_config, "response_length", None)
        if isinstance(response_length, int) and response_length > 0:
            return response_length
        return None

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Image data for the chat completion.
            video_data (Optional[List[Any]]): Video data for the chat completion.
            audio_data (Optional[List[Any]]): Audio data for the chat completion.
            mm_processor_kwargs (Optional[Dict[str, Any]]): Multimodal processor kwargs.

        Returns:
            TokenOutput: token output
        """
        prompt_ids = normalize_token_ids(prompt_ids)

        limit_key = None
        if "max_tokens" in sampling_params:
            limit_key = "max_tokens"
        elif "max_new_tokens" in sampling_params:
            limit_key = "max_new_tokens"
        original_max_tokens = sampling_params.get(limit_key) if limit_key else None

        # The budget below is rewritten on every attempt, and the caller reuses its dict across
        # turns, so never mutate the caller's copy.
        sampling_params = dict(sampling_params)

        if original_max_tokens is None:
            # Without an explicit limit each attempt falls back to the server-side default, which is
            # derived from len(prompt_ids) and is only correct on the first attempt: a resume passes
            # prompt + tokens generated so far, so the default charges generated tokens against the
            # *prompt* budget and re-permits close to a full response_length every time. Pin the
            # cumulative budget here instead, which also makes the bookkeeping in step 3 effective.
            response_length = self._configured_response_length()
            if response_length is not None:
                limit_key = "max_tokens"
                original_max_tokens = response_length
                sampling_params[limit_key] = response_length

        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            num_preempted=0,
        )
        min_global_steps, max_global_steps = None, None
        # Prefix-cache hits are reported per prefill. The base client returns the
        # server's TokenOutput directly (so sync mode surfaces num_cached_tokens),
        # but here we rebuild a fresh TokenOutput across resume iterations, so we
        # must carry it forward explicitly or the consumer sees 0. Take the first
        # (initial-prompt) prefill's hit count, matching single-prefill semantics.
        num_cached_tokens = None
        native_attempts = []
        native_actor_attempts = []

        while True:
            # 1. generate tokens
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **kwargs,
            )

            if getattr(self, "_generation_transport", None) is not None:
                receipt = deepcopy(output.extra_fields)
                receipt["output_start"] = len(final_output.token_ids)
                receipt["output_end"] = len(final_output.token_ids) + len(output.token_ids)
                native_attempts.append(receipt)

            elif "native_meta_info" in output.extra_fields:
                # This is an actual TokenizerManager response reached through
                # the original Ray actor; it is not a Gateway routing receipt.
                receipt = deepcopy(output.extra_fields)
                receipt["output_start"] = len(final_output.token_ids)
                receipt["output_end"] = len(final_output.token_ids) + len(output.token_ids)
                native_actor_attempts.append(receipt)

            # 2. merge output into final_output
            final_output.token_ids.extend(output.token_ids)
            if output.log_probs is not None:
                final_output.log_probs.extend(output.log_probs)
            # On partial rollout resume the model version may differ, so keep
            # existing routing and only append routing for newly generated tokens.
            if output.routed_experts is not None and len(output.token_ids) > 0:
                if final_output.routed_experts is None:
                    final_output.routed_experts = output.routed_experts
                else:
                    final_output.routed_experts = np.concatenate(
                        [final_output.routed_experts, output.routed_experts[-len(output.token_ids) :]]
                    )
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            # sampler top-k heads for score centering, one row per newly generated token
            for key in ("response_topk_ids", "response_topk_log_probs"):
                if key in output.extra_fields:
                    previous = final_output.extra_fields.get(key)
                    final_output.extra_fields[key] = (
                        output.extra_fields[key]
                        if previous is None
                        else np.concatenate([previous, output.extra_fields[key]])
                    )
            final_output.stop_reason = output.stop_reason

            # carry the initial prefill's prefix-cache hit count forward
            if num_cached_tokens is None:
                num_cached_tokens = output.extra_fields.get("num_cached_tokens")

            # update model weights version
            global_steps = output.extra_fields.get("global_steps", None)
            if min_global_steps is None:
                min_global_steps = global_steps
            max_global_steps = global_steps

            # 3. update max_new_tokens
            if original_max_tokens is not None:
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                if len(final_output.token_ids) >= original_max_tokens:
                    final_output.stop_reason = "length"
                    break

            # 4. check stop reason
            # If partial rollout not enable, aborted samples will be dropped.
            # For v1 trainer, should_retry is always True. Since self.config.async_training is not exist.
            should_retry = True
            if hasattr(self.config, "async_training") and not self.config.async_training.partial_rollout:
                should_retry = False
            if output.stop_reason not in ("aborted", "abort") or not should_retry:
                break

            await asyncio.sleep(1)

        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        final_output.extra_fields["num_cached_tokens"] = num_cached_tokens
        if native_attempts:
            final_output.extra_fields["native_gateway_attempts"] = native_attempts
        if native_actor_attempts:
            final_output.extra_fields["native_generation_attempts"] = native_actor_attempts
            final_output.extra_fields["generation_path"] = "native_ray_actor"
        return final_output


class LLMServerManager:
    """LLMServerManager is responsible for:
    - Launch server replicas
    - Launch global load balancer
    - Elastic launch/tear-down new replicas

    Args:
        config (DictConfig): Config for the trainer entrypoint.
        worker_group (RayWorkerGroup): Worker group for the server replicas. If not none, init hybrid server,
            else init standalone server with a new resource pool.
        rollout_resource_pool (RayResourcePool): Resource pool for the server replicas, only needed for TensorRT-LLM.
        start_rank (int): First ``replica_rank`` to assign.  Defaults to 0.
        load_balancer_cls: Optional subclass of the default router strategy's
            load balancer to use as the routing actor (wrapped with
            ``ray.remote`` at instantiation). When given, it takes precedence
            over ``rollout.router_config_path``. Pass a subclass that
            overrides :meth:`acquire_server` to take full control of routing.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        start_rank: int = 0,
        load_balancer_cls: type | None = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool
        self.start_rank = start_rank
        self._load_balancer_cls = load_balancer_cls

        assert worker_group is not None or self.rollout_config.nnodes > 0, "nnodes must be > 0 in standalone mode"

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(
                self.rollout_config.name,
                disaggregation_enabled=self.rollout_config.disaggregation.enabled,
            )

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the LLMServerManager."""
        instance = cls(*args, **kwargs)
        await instance._initialize_llm_servers()
        await instance._init_global_load_balancer()
        return instance

    @classmethod
    @auto_await
    async def create_empty(cls, *args, **kwargs):
        """Create an empty LLMServerManager: no replicas are launched and the
        load balancer starts empty.

        Used by trainers that serve rollout exclusively from standalone
        resources (e.g. v1 separate_async with ``actor_rollout_ref.hybrid_engine=False``)
        so that no inference engine is initialized on the training GPUs.
        """
        instance = cls(*args, **kwargs)
        instance.rollout_replicas = []
        instance.server_handles = []
        instance.server_addresses = []
        await instance._init_global_load_balancer()
        return instance

    async def _initialize_llm_servers(self, start_rank: int = None):
        """Initialize the LLM server replicas.

        Args:
            start_rank: First ``replica_rank`` to assign.  Defaults to ``self.start_rank``
                so standalone replicas can avoid Ray named-actor collisions with hybrid
                replicas (which start at 0) when both coexist (e.g. separate async).
        """
        if start_rank is None:
            start_rank = self.start_rank
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        # PD inflates per-replica footprint; miss this and init_hybrid slices
        # past worker_group → empty workers on replica_rank>=1.
        disagg = getattr(self.rollout_config, "disaggregation", None)
        if disagg is not None and getattr(disagg, "enabled", False):
            prefill_tp = self.rollout_config.tensor_model_parallel_size
            # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            rollout_world_size = (
                (prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas)
                * self.rollout_config.data_parallel_size
                * self.rollout_config.pipeline_model_parallel_size
            )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        )
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=start_rank + replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and self.rollout_config.name != "trtllm":
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        # TODO: unify trtllm to init_hybrid
        elif self.worker_group and self.rollout_config.name == "trtllm":
            await asyncio.gather(
                *[
                    server.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        print(f"LLMServerManager: {self.server_addresses}")

        # Update Prometheus / rl-insight metrics with server addresses
        needs_metrics = self.rollout_config.prometheus.enable or RLInsightLogger.enabled()
        if self.rollout_config.disable_log_stats:
            if needs_metrics:
                raise ValueError("Metrics monitoring requires disable_log_stats=False, but it is currently True.")
        if not self.rollout_config.disable_log_stats:
            if self.rollout_config.prometheus.enable:
                update_prometheus_config(
                    self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name
                )
            if RLInsightLogger.enabled():
                RLInsightLogger.register_rollout_metrics(
                    self.server_addresses,
                    self.rollout_config.name,
                    labels=[{"replica": server.replica_rank} for server in self.rollout_replicas],
                )

    async def _init_global_load_balancer(self) -> None:
        from verl.workers.rollout.router import get_router_handle

        self.global_load_balancer = get_router_handle(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            router_config_path=getattr(self.rollout_config, "router_config_path", None),
            full_determinism=getattr(self.rollout_config, "full_determinism", False),
            load_balancer_cls=self._load_balancer_cls,
        )

    def _continuous_publication_client(self):
        if not continuous_policy_publication_enabled(self.config):
            return None
        _policy_publication_requirements(self.config)
        _, client = get_or_create_trajectory_coordinator(self.config)
        _validate_coordinator_binding_sync(client, run_id=self.rollout_config.trajectory_state.run_id,
                                           root_dir=self.rollout_config.trajectory_state.root_dir)
        return client

    @property
    def native_engine_urls(self) -> tuple[str, ...]:
        """Return the actual native engine URLs exposed by this manager."""
        return tuple(_engine_url(address) for address in self.server_addresses)


    def get_client(self, client_cls: type[LLMServerClient] | None = None, **kwargs) -> LLMServerClient:
        """Get the LLMServerClient to request LLM server replicas.

        Args:
            client_cls: The client class to instantiate. Defaults to
                :class:`LLMServerClient`. Pass a subclass to customize
                request-id handling (e.g. a deterministic client that forwards
                the caller's ``request_id`` straight to vLLM), or
                :class:`FullyAsyncLLMServerClient` for abort-resume support.
            **kwargs: Forwarded to the client constructor.
        """
        client_cls = client_cls or LLMServerClient
        from verl.workers.rollout.sglang_gateway_lifecycle import make_gateway_client, native_gateway_enabled

        if native_gateway_enabled(self.config):
            transport = make_gateway_client(
                self.config, getattr(self, "gateway_expected_addresses", self.server_addresses)
            )
            publication_client = self._continuous_publication_client()
            if publication_client is not None:
                transport = ContinuousPublicationTransport(
                    transport,
                    publication_client,
                    run_id=self.rollout_config.trajectory_state.run_id,
                )
            kwargs.setdefault("generation_transport", transport)
        client = client_cls(
            config=self.config,
            load_balancer_handle=self.global_load_balancer,
            **kwargs,
        )
        # Preserve custom client constructors that accept only the legacy args.
        if isinstance(client, LLMServerClient):
            client._engine_urls = self.native_engine_urls
        return client

    def get_gateway_lifecycle(self):
        """Return opt-in native control for this manager's exact replica URLs."""
        from verl.trainer.ppo.v1.trajectory_policy_publication import TrajectoryPolicyPublicationGatewayHooks
        from verl.workers.rollout.sglang_gateway_lifecycle import (
            NativeGatewayLifecycle,
            TrajectoryAwareGatewayLifecycle,
            make_gateway_client,
            native_gateway_enabled,
        )

        if not native_gateway_enabled(self.config):
            return None
        if not self.server_addresses:
            return None
        lifecycle = NativeGatewayLifecycle(
            make_gateway_client(self.config, getattr(self, "gateway_expected_addresses", self.server_addresses)),
            self.server_addresses,
        )
        publication_client = self._continuous_publication_client()
        if publication_client is None:
            return lifecycle
        hooks = TrajectoryPolicyPublicationGatewayHooks(
            publication_client,
            run_id=self.rollout_config.trajectory_state.run_id,
            runtime_policy_identity=_policy_identity_dict(self.config),
            model_configuration_fingerprint=self.rollout_config.trajectory_state.model_configuration_fingerprint,
            publisher_id=f"{self.rollout_config.trajectory_state.run_id}/checkpoint-manager",
            engine_urls=list(lifecycle.urls),
            engine_observer=lambda: observe_owned_engine_versions(
                list(lifecycle.urls), timeout_s=self.rollout_config.sglang_gateway.operation_timeout_s
            ),
        )
        return TrajectoryAwareGatewayLifecycle(lifecycle, hooks)

    def get_addresses(self) -> list[str]:
        """Get the OpenAI chat completion API http addresses of the LLM server replicas."""
        return self.server_addresses

    def get_replicas(self) -> list[RolloutReplica]:
        """Get the LLM server replicas."""
        return self.rollout_replicas

    @auto_await
    async def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        await asyncio.gather(*[replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    @auto_await
    async def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        await asyncio.gather(*[replica.stop_profile() for replica in self.rollout_replicas])
