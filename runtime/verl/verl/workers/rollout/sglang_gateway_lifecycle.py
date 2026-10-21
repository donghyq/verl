# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Native Gateway registration around VeRL's existing weight-sync lifecycle."""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass
from typing import Any

from verl.trainer.ppo.v1.trajectory_policy_publication import TrajectoryPolicyPublicationGatewayHooks
from verl.workers.rollout.sglang_gateway_client import (
    GatewayClientConfig,
    GatewayControlError,
    GatewayRegistrationAcceptanceUnknown,
    GatewayRegistrationOperationError,
    GatewayRegistrationOperationTimeout,
    GatewayRegistrationRejected,
    SGLangGatewayClient,
)


def engine_url(address: str) -> str:
    return address.rstrip("/") if address.startswith(("http://", "https://")) else f"http://{address}"


def native_gateway_enabled(config: Any) -> bool:
    gateway = getattr(config.actor_rollout_ref.rollout, "sglang_gateway", None)
    return gateway is not None and bool(getattr(gateway, "enabled", False))


def make_gateway_client(config: Any, addresses: list[str]) -> SGLangGatewayClient:
    rollout = config.actor_rollout_ref.rollout
    gateway = rollout.sglang_gateway
    if getattr(config.trainer, "use_v1", None) is not True:
        raise ValueError("The native SGLang Gateway requires trainer.use_v1=true")
    trainer_mode = getattr(getattr(config.trainer, "v1", None), "trainer_mode", None)
    if trainer_mode not in {"sync", "colocate_async", "separate_async"}:
        raise ValueError(
            "The native SGLang Gateway requires trainer.v1.trainer_mode in {sync, colocate_async, separate_async}"
        )
    if rollout.mode != "async":
        raise ValueError("The native SGLang Gateway requires actor_rollout_ref.rollout.mode=async")
    if rollout.name != "sglang":
        raise ValueError("The native SGLang Gateway requires rollout.name=sglang")
    if getattr(rollout.disaggregation, "enabled", False):
        raise ValueError("The initial native Gateway integration does not support PD disaggregation")
    if getattr(rollout.mtp, "enable", False) or getattr(rollout, "enable_rollout_routing_replay", False):
        raise ValueError("MTP and expert routing replay are not enabled for native Gateway transport")
    if getattr(config.actor_rollout_ref.model, "lora_rank", 0):
        raise ValueError("LoRA routing is not enabled for native Gateway transport")
    if getattr(rollout, "full_determinism", False):
        raise ValueError("Native cache routing does not preserve VeRL full_determinism replica placement")
    switch = getattr(getattr(getattr(config.trainer, "v1", None), "separate_async", None), "hybrid_rollout", None)
    state = getattr(rollout, "trajectory_state", None)
    shared_gpu = bool(getattr(state, "shared_gpu_handoff", False))
    if shared_gpu and (
        trainer_mode != "separate_async" or not getattr(state, "enabled", False)
        or not getattr(state, "continuous_policy_publication", False)
        or getattr(config.trainer.v1.separate_async, "parameter_sync_step", None) != 1
        or not getattr(config.actor_rollout_ref, "hybrid_engine", True)
        or switch is None or not getattr(switch, "enable_switch", False)
        or type(getattr(state, "handoff_target_worker_index", None)) is not int
        or not 0 <= state.handoff_target_worker_index < rollout.agent.num_workers
        or rollout.agent.num_workers < 2
    ):
        raise ValueError("Shared GPU Gateway transport requires guarded separate_async switching and a reserved target")
    if switch is not None and getattr(switch, "enable_switch", False) and not shared_gpu:
        raise ValueError("Dynamic hybrid switching must be disabled for the initial native Gateway integration")
    trainer_mode = getattr(getattr(config.trainer, "v1", None), "trainer_mode", None)
    if trainer_mode == "separate_async" and getattr(config.actor_rollout_ref, "hybrid_engine", True) and not shared_gpu:
        raise ValueError("Native separate_async requires hybrid_engine=False and one authoritative replica set")
    if rollout.topk_log_probs:
        raise ValueError("Sampler top-k heads are not enabled for native Gateway transport")
    maximum = rollout.max_model_len
    if maximum is None:
        from verl.workers.rollout.utils import get_max_position_embeddings

        maximum = get_max_position_embeddings(config.actor_rollout_ref.model.hf_config)
    if gateway.api_key_env and not os.environ.get(gateway.api_key_env):
        raise ValueError("The configured Gateway credential environment variable is absent")
    return SGLangGatewayClient(
        GatewayClientConfig(
            base_url=gateway.url,
            max_model_len=maximum,
            prompt_length=rollout.prompt_length,
            response_length=rollout.response_length,
            request_timeout_s=gateway.request_timeout_s,
            control_timeout_s=gateway.control_timeout_s,
            operation_timeout_s=gateway.operation_timeout_s,
            poll_interval_s=gateway.poll_interval_s,
            api_key=os.environ.get(gateway.api_key_env) if gateway.api_key_env else None,
            expected_worker_urls=tuple(engine_url(address) for address in addresses),
            allow_partial_worker_set=shared_gpu,
        )
    )


class GatewayIsolationUnconfirmed(GatewayControlError):
    """Routing isolation cannot be proved for one or more submitted registrations."""


@dataclass
class _RegistrationAttempt:
    url: str
    post_submitted: bool = False
    acceptance_known: bool = False
    accepted_worker_id: str | None = None
    terminal_state: str | None = None


class NativeGatewayLifecycle:
    def __init__(self, client: SGLangGatewayClient, addresses: list[str]):
        self.client = client
        self.urls = tuple(engine_url(address) for address in addresses)
        if len(set(self.urls)) != len(self.urls):
            raise ValueError("Native Gateway replica URLs must be unique")
        self.removal_observations: list[dict[str, Any]] = []
        self.registration_observations: list[dict[str, Any]] = []
        self.registration_attempts: list[_RegistrationAttempt] = []
        self._permanently_quarantined = False

    def _workers_by_url(self, workers: list[dict]) -> dict[str, dict]:
        indexed = {}
        allowed = set(getattr(getattr(self.client, "config", None), "expected_worker_urls", ()) or self.urls)
        seen = set()
        identities = set()
        for worker in workers:
            url = worker.get("url")
            identity = worker.get("id")
            if (url not in allowed or url in seen or not isinstance(identity, str)
                    or not identity or identity in identities):
                raise GatewayControlError("Gateway contains an unexpected or duplicate replica URL")
            seen.add(url)
            identities.add(identity)
            if url in self.urls:
                indexed[url] = worker
        return indexed

    async def stop_admission(self) -> None:
        if self._permanently_quarantined:
            raise GatewayIsolationUnconfirmed(
                "This native Gateway lifecycle is permanently quarantined after unconfirmed routing isolation"
            )
        try:
            workers = self._workers_by_url(await self.client.list_workers())
            own_ids = {observation["worker_id"] for observation in self.registration_observations}
            unexpected = [
                worker
                for worker in workers.values()
                if worker.get("id") not in own_ids or not isinstance(worker.get("id"), str) or not worker["id"]
            ]
            if unexpected:
                raise GatewayControlError(
                    "Native routing contains a worker identity that this lifecycle did not register"
                )
            results = await asyncio.gather(
                *(self.client.remove_worker(worker["id"], url=url) for url, worker in workers.items()),
                return_exceptions=True,
            )
            for result in results:
                if not isinstance(result, BaseException):
                    self.removal_observations.append(asdict(result))
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise GatewayControlError("At least one native replica removal failed") from errors[0]
            remaining = await self.client.list_workers()
            if any(worker.get("id") in own_ids or worker.get("url") in self.urls for worker in remaining):
                raise GatewayControlError("Native replicas are still registered after removal")
        except (Exception, asyncio.CancelledError) as error:
            self._permanently_quarantined = True
            raise GatewayIsolationUnconfirmed("Native removal failed; routing isolation is unconfirmed") from error
        finally:
            await self.client.close()

    async def start_admission(self, global_steps: int) -> None:
        if self._permanently_quarantined:
            raise GatewayControlError(
                "This native Gateway lifecycle is permanently quarantined after unconfirmed routing isolation"
            )
        if type(global_steps) is not int or global_steps < 0:
            raise GatewayControlError("Registration requires the actual committed RL parameter version")
        attempts = {url: _RegistrationAttempt(url) for url in self.urls}
        self.registration_attempts.extend(attempts.values())

        async def register(attempt: _RegistrationAttempt):
            attempt.post_submitted = True
            try:
                observation = await self.client.register_worker(
                    {"url": attempt.url, "worker_type": "regular", "connection_mode": "http", "runtime_type": "sglang"},
                    expected_weight_version=global_steps,
                )
            except GatewayRegistrationRejected:
                attempt.acceptance_known = True
                attempt.terminal_state = "rejected"
                raise
            except GatewayRegistrationOperationTimeout as error:
                attempt.acceptance_known = True
                attempt.accepted_worker_id = error.worker_id
                raise
            except GatewayRegistrationOperationError as error:
                attempt.acceptance_known = True
                attempt.accepted_worker_id = error.worker_id
                attempt.terminal_state = error.terminal_state
                raise
            except GatewayRegistrationAcceptanceUnknown:
                raise
            attempt.acceptance_known = True
            attempt.accepted_worker_id = observation.worker_id
            attempt.terminal_state = "ready"
            return observation

        try:
            workers = self._workers_by_url(await self.client.list_workers())
            if workers:
                raise GatewayControlError("Native registration requires prior confirmed routing isolation")
            versions = await self.client.observe_unregistered_engine_versions(self.urls)
            if set(versions) != set(self.urls) or any(version != global_steps for version in versions.values()):
                raise GatewayControlError("Unregistered engines do not have the committed RL parameter version")
            results = await asyncio.gather(
                *(register(attempt) for attempt in attempts.values()),
                return_exceptions=True,
            )
            for result in results:
                if not isinstance(result, BaseException):
                    self.registration_observations.append(asdict(result))
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise GatewayControlError("At least one native registration failed") from errors[0]
            if set(self._workers_by_url(await self.client.list_workers())) != set(self.urls):
                raise GatewayControlError("Registered native replicas disappeared before admission verification")
            expected = getattr(getattr(self.client, "config", None), "expected_worker_urls", ()) or self.urls
            actual = await self.client.get_active_weight_version(expected)
            if actual != global_steps:
                raise GatewayControlError("Ready native engines do not have the committed RL parameter version")
        except asyncio.CancelledError as error:
            if any(attempt.post_submitted for attempt in attempts.values()):
                self._permanently_quarantined = True
                raise GatewayIsolationUnconfirmed(
                    "Native registration was cancelled; routing isolation is unconfirmed"
                ) from error
            raise
        except Exception as error:
            if any(attempt.post_submitted for attempt in attempts.values()):
                try:
                    await self._settle_and_remove_after_failed_registration(list(attempts.values()))
                except asyncio.CancelledError:
                    self._permanently_quarantined = True
                    raise
                except Exception as cleanup_error:
                    self._permanently_quarantined = True
                    raise GatewayIsolationUnconfirmed(
                        "Native registration failed and routing isolation is unconfirmed "
                        f"({type(cleanup_error).__name__}: {cleanup_error})"
                    ) from error
            raise
        finally:
            await self.client.close()

    async def _settle_and_remove_after_failed_registration(self, attempts: list[_RegistrationAttempt]) -> None:
        accepted = [attempt for attempt in attempts if attempt.post_submitted and attempt.accepted_worker_id]

        async def settle(attempt):
            if attempt.terminal_state is not None:
                return attempt.terminal_state
            observation = await self.client.settle_registration(attempt.accepted_worker_id, url=attempt.url)
            return observation.state

        settlements = await asyncio.gather(*(settle(attempt) for attempt in accepted), return_exceptions=True)
        cleanup_errors = []
        removable = []
        for attempt, state in zip(accepted, settlements, strict=True):
            if isinstance(state, BaseException):
                cleanup_errors.append(state)
                continue
            attempt.terminal_state = state
            if state == "ready":
                removable.append(attempt)
            elif state == "failed":
                # An explicitly failed AddWorker is terminal, but a partial
                # registration may already exist. Remove only our exact identity.
                try:
                    workers = await self.client.list_workers()
                    for worker in workers:
                        same_id = worker.get("id") == attempt.accepted_worker_id
                        same_url = worker.get("url") == attempt.url
                        if same_id != same_url:
                            raise GatewayControlError("Failed registration identity conflicts with registry")
                        if same_id:
                            removable.append(attempt)
                except Exception as error:
                    cleanup_errors.append(error)
            else:
                cleanup_errors.append(GatewayControlError("Unknown registration terminal state"))

        results = await asyncio.gather(
            *(self.client.remove_worker(attempt.accepted_worker_id, url=attempt.url) for attempt in removable),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                cleanup_errors.append(result)
            else:
                self.removal_observations.append(asdict(result))

        for attempt in attempts:
            if not attempt.post_submitted:
                continue
            try:
                if not attempt.acceptance_known:
                    await self.client.observe_unknown_registration(attempt.url)
                    raise GatewayIsolationUnconfirmed("Native registration acceptance remains unknown")
                if attempt.accepted_worker_id is None or attempt.terminal_state is not None:
                    await self._assert_identity_absent(attempt.accepted_worker_id, attempt.url)
            except Exception as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise GatewayIsolationUnconfirmed(
                "Native registration settlement or cleanup remains unconfirmed "
                f"({type(cleanup_errors[0]).__name__}: {cleanup_errors[0]})"
            ) from cleanup_errors[0]

    async def _assert_identity_absent(self, worker_id: str | None, url: str) -> None:
        workers = await self.client.list_workers()
        if any(worker.get("id") == worker_id or worker.get("url") == url for worker in workers):
            raise GatewayControlError("Native registration leaves an unexpected worker at an owned URL")


class TrajectoryAwareGatewayLifecycle:
    """Compose native Gateway lifecycle with trajectory publication hooks."""

    def __init__(self, lifecycle: NativeGatewayLifecycle, publication_hooks: TrajectoryPolicyPublicationGatewayHooks):
        self._lifecycle = lifecycle
        self._publication_hooks = publication_hooks

    def __getattr__(self, name: str):
        try:
            lifecycle = object.__getattribute__(self, "_lifecycle")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(lifecycle, name)

    async def prepare_policy_publication(self, global_steps: int):
        return await self._publication_hooks.prepare_policy_publication(global_steps)

    async def mark_policy_update_started(self, ticket) -> None:
        await self._publication_hooks.mark_policy_update_started(ticket)

    async def confirm_policy_publication_before_admission(self, ticket):
        return await self._publication_hooks.confirm_policy_publication_before_admission(ticket)

    async def fail_policy_publication(self, ticket, reason: str) -> None:
        await self._publication_hooks.fail_policy_publication(ticket, reason)
