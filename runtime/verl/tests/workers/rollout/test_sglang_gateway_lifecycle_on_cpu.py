"""Fault boundaries at native admission and existing VeRL weight updates."""

import asyncio
import math
from types import MethodType, SimpleNamespace

import cloudpickle
import pytest
from omegaconf import OmegaConf

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.workers.rollout.llm_server import (
    ContinuousPublicationTransport,
    FullyAsyncLLMServerClient,
    LLMServerClient,
    LLMServerManager,
)
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.sglang_gateway_client import (
    GatewayControlError,
    GatewayProtocolError,
    GatewayRegistrationAcceptanceUnknown,
    GatewayRegistrationOperationError,
    GatewayRegistrationOperationTimeout,
    GatewayRegistrationRejected,
    WorkerRegistrationObservation,
    WorkerRegistrationSettlement,
    WorkerRemovalObservation,
)
from verl.workers.rollout.sglang_gateway_lifecycle import (
    GatewayIsolationUnconfirmed,
    NativeGatewayLifecycle,
    TrajectoryAwareGatewayLifecycle,
    make_gateway_client,
)


class Registry:
    def __init__(self, urls):
        self.workers = {url: {"id": str(i), "url": url} for i, url in enumerate(urls)}
        self.events = []
        self.version = 7
        self.fail_remove = None
        self.fail_register = None

    async def list_workers(self):
        return list(self.workers.values())

    async def remove_worker(self, worker_id, *, url):
        self.events.append(("remove", url))
        if url == self.fail_remove:
            raise GatewayControlError("injected deletion failure")
        assert self.workers[url]["id"] == worker_id
        del self.workers[url]
        return WorkerRemovalObservation(worker_id, url)

    async def register_worker(self, spec, *, expected_weight_version):
        self.events.append(("register", spec["url"], expected_weight_version))
        if spec["url"] == self.fail_register:
            raise GatewayRegistrationRejected("injected registration rejection")
        assert expected_weight_version == self.version
        worker_id = "new-" + str(len(self.workers))
        self.workers[spec["url"]] = {"id": worker_id, **spec}
        return WorkerRegistrationObservation(worker_id, spec["url"], str(self.version))

    async def settle_registration(self, worker_id, *, url):
        self.events.append(("settle", url, worker_id))
        if url not in self.workers:
            raise GatewayRegistrationOperationTimeout("settlement unknown", worker_id, url)
        return WorkerRegistrationSettlement(worker_id, url, "ready")

    async def observe_unknown_registration(self, url):
        self.events.append(("observe_unknown", url))
        raise GatewayRegistrationAcceptanceUnknown("acceptance remains unknown")

    async def get_active_weight_version(self, urls):
        assert set(urls) == set(self.workers)
        return self.version

    async def observe_unregistered_engine_versions(self, urls):
        assert self.workers == {}
        self.events.append(("observe_unregistered",))
        return {url: self.version for url in urls}

    async def close(self):
        self.events.append(("close",))


@pytest.mark.asyncio
async def test_successful_registration_and_removal_do_not_claim_cache_purge():
    urls = ["http://replica-a:30000", "http://replica-b:30000"]
    client = Registry([])
    control = NativeGatewayLifecycle(client, urls)
    await control.start_admission(7)
    assert set(client.workers) == set(urls)
    await control.stop_admission()
    assert len(control.removal_observations) == 2
    assert all(row["cache_cleanup_complete"] is None for row in control.removal_observations)
    assert client.workers == {}
    assert all(row["engine_weight_version"] == "7" for row in control.registration_observations)
    kinds = [event[0] for event in client.events]
    assert kinds.index("observe_unregistered") < kinds.index("register")


@pytest.mark.asyncio
async def test_old_unregistered_engine_is_rejected_before_it_can_receive_requests():
    urls = ["http://replica-a:30000", "http://replica-b:30000"]
    client = Registry([])
    control = NativeGatewayLifecycle(client, urls)
    client.version = 6
    with pytest.raises(GatewayControlError, match="Unregistered engines"):
        await control.start_admission(7)
    assert client.workers == {}
    assert not any(event[0] == "register" for event in client.events)


@pytest.mark.asyncio
async def test_missing_engine_observation_does_not_register_any_replica():
    urls = ["http://replica-a:30000", "http://replica-b:30000"]
    client = Registry([])
    control = NativeGatewayLifecycle(client, urls)

    async def incomplete(_urls):
        return {urls[0]: 7}

    client.observe_unregistered_engine_versions = incomplete
    with pytest.raises(GatewayControlError, match="Unregistered engines"):
        await control.start_admission(7)
    assert client.workers == {}
    assert not any(event[0] == "register" for event in client.events)


@pytest.mark.asyncio
async def test_partial_removal_is_failure_and_never_starts_weight_update():
    urls = ["http://a", "http://b"]
    client = Registry([])
    control = NativeGatewayLifecycle(client, urls)
    await control.start_admission(7)
    registration_count = [event[0] for event in client.events].count("register")
    client.fail_remove = urls[1]
    with pytest.raises(GatewayControlError):
        await control.stop_admission()
    assert set(client.workers) == {urls[1]}
    assert registration_count == len(urls)


@pytest.mark.asyncio
async def test_registered_replica_cannot_skip_isolation():
    client = Registry(["http://a"])
    control = NativeGatewayLifecycle(client, ["http://a"])
    with pytest.raises(GatewayControlError):
        await control.start_admission(7)
    assert set(client.workers) == {"http://a"}
    assert not any(event[0] == "remove" for event in client.events)


@pytest.mark.asyncio
async def test_partial_registration_failure_removes_the_successful_owned_replica():
    urls = ["http://a", "http://b"]
    client = Registry([])
    client.fail_register = urls[1]
    control = NativeGatewayLifecycle(client, urls)
    with pytest.raises(GatewayControlError, match="registration failed"):
        await control.start_admission(7)
    assert client.workers == {}
    assert [event[0] for event in client.events].count("register") == len(urls)


def _gateway_factory_config(*, use_v1=True, trainer_mode="sync", rollout_mode="async", rollout_name="sglang"):
    return OmegaConf.create(
        {
            "trainer": {
                "use_v1": use_v1,
                "v1": {
                    "trainer_mode": trainer_mode,
                    "separate_async": {"hybrid_rollout": {"enable_switch": False}},
                },
            },
            "actor_rollout_ref": {
                "hybrid_engine": True,
                "model": {"lora_rank": 0, "hf_config": {}},
                "rollout": {
                    "name": rollout_name,
                    "mode": rollout_mode,
                    "max_model_len": 128,
                    "prompt_length": 64,
                    "response_length": 32,
                    "full_determinism": False,
                    "enable_rollout_routing_replay": False,
                    "topk_log_probs": 0,
                    "disaggregation": {"enabled": False},
                    "mtp": {"enable": False},
                    "sglang_gateway": {
                        "enabled": True,
                        "url": "http://gateway:30000",
                        "request_timeout_s": 1.0,
                        "control_timeout_s": 1.0,
                        "operation_timeout_s": 1.0,
                        "poll_interval_s": 0.01,
                        "api_key_env": None,
                    },
                },
            },
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_v1", False),
        ("trainer_mode", "unknown"),
        ("rollout_mode", "sync"),
        ("rollout_name", "vllm"),
    ],
)
def test_gateway_factory_rejects_unsupported_configuration_before_transport(monkeypatch, field, value):
    import verl.workers.rollout.sglang_gateway_client as client_module
    import verl.workers.rollout.sglang_gateway_lifecycle as lifecycle_module

    transports = []

    def fail_client(*args, **kwargs):
        raise AssertionError("Unsupported configuration must not construct a client")

    def fail_transport(*args, **kwargs):
        transports.append(args)
        raise AssertionError("Unsupported configuration must not construct a transport")

    monkeypatch.setattr(lifecycle_module, "SGLangGatewayClient", fail_client)
    monkeypatch.setattr(client_module, "_AiohttpTransport", fail_transport)
    config = _gateway_factory_config(**{field: value})
    with pytest.raises(ValueError, match="native SGLang Gateway requires"):
        make_gateway_client(config, ["engine:30001"])
    assert transports == []


@pytest.mark.parametrize("trainer_mode", ["sync", "colocate_async", "separate_async"])
def test_gateway_factory_accepts_actual_v1_modes(monkeypatch, trainer_mode):
    import verl.workers.rollout.sglang_gateway_client as client_module
    from verl.workers.rollout.sglang_gateway_client import SGLangGatewayClient

    transports = []

    class Transport:
        def __init__(self, *args):
            transports.append(args)

    monkeypatch.setattr(client_module, "_AiohttpTransport", Transport)
    config = _gateway_factory_config(trainer_mode=trainer_mode)
    if trainer_mode == "separate_async":
        config.actor_rollout_ref.hybrid_engine = False
    client = make_gateway_client(config, ["engine:30001"])
    assert type(client) is SGLangGatewayClient
    assert client.config.expected_worker_urls == ("http://engine:30001",)
    assert [args[0] for args in transports] == ["http://gateway:30000", "http://gateway:30000"]


class RegistrationFlow:
    def __init__(self, url="http://engine:30001"):
        self.url = url
        self.workers = {}
        self.events = []
        self.register_failure = None
        self.settlement_failure = None
        self.late_worker_id = None
        self.settlement_state = "ready"
        self.fail_remove = False

    async def list_workers(self):
        return list(self.workers.values())

    async def observe_unregistered_engine_versions(self, urls):
        self.events.append(("observe", tuple(urls)))
        return {url: 7 for url in urls}

    async def register_worker(self, spec, *, expected_weight_version):
        self.events.append(("post", spec["url"], expected_weight_version))
        if self.register_failure is not None:
            raise self.register_failure
        self.workers[self.url] = {"id": "accepted", "url": self.url}
        return WorkerRegistrationObservation("accepted", self.url, "7")

    async def settle_registration(self, worker_id, *, url):
        self.events.append(("settle", worker_id, url))
        if self.settlement_failure is not None:
            if isinstance(self.settlement_failure, GatewayRegistrationOperationError):
                self.workers[url] = {"id": "foreign", "url": url}
            raise self.settlement_failure
        if self.late_worker_id is not None and url not in self.workers:
            self.workers[url] = {"id": self.late_worker_id, "url": url}
        return WorkerRegistrationSettlement(worker_id, url, self.settlement_state)

    async def remove_worker(self, worker_id, *, url):
        self.events.append(("delete", worker_id, url))
        if self.fail_remove:
            raise GatewayControlError("injected cleanup failure")
        assert self.workers[url]["id"] == worker_id
        del self.workers[url]
        return WorkerRemovalObservation(worker_id, url)

    async def get_active_weight_version(self, urls):
        assert set(urls) == set(self.workers)
        return 7

    async def observe_unknown_registration(self, url):
        self.events.append(("observe_unknown", url))
        raise GatewayRegistrationAcceptanceUnknown("acceptance remains unknown")

    async def close(self):
        self.events.append(("close",))


@pytest.mark.asyncio
async def test_late_registration_after_timeout_is_settled_and_removed_once():
    client = RegistrationFlow()
    client.register_failure = GatewayRegistrationOperationTimeout("registration timeout", "late", client.url)
    client.late_worker_id = "late"
    control = NativeGatewayLifecycle(client, [client.url])
    with pytest.raises(GatewayControlError, match="At least one native registration failed"):
        await control.start_admission(7)
    assert client.workers == {}
    assert [event[0] for event in client.events].count("post") == 1
    assert ("settle", "late", client.url) in client.events
    assert ("delete", "late", client.url) in client.events

    client.register_failure = None
    await control.start_admission(7)
    assert set(client.workers) == {client.url}


@pytest.mark.asyncio
async def test_unknown_registration_terminal_state_quarantines_the_lifecycle():
    client = RegistrationFlow()
    client.register_failure = GatewayRegistrationOperationTimeout("registration timeout", "accepted", client.url)
    client.settlement_failure = GatewayRegistrationOperationTimeout("settlement timeout", "accepted", client.url)
    control = NativeGatewayLifecycle(client, [client.url])
    with pytest.raises(GatewayIsolationUnconfirmed, match="routing isolation is unconfirmed"):
        await control.start_admission(7)
    assert client.workers == {}
    assert not any(event[0] == "delete" for event in client.events)
    with pytest.raises(GatewayIsolationUnconfirmed, match="permanently quarantined"):
        await control.stop_admission()
    with pytest.raises(GatewayControlError, match="permanently quarantined"):
        await control.start_admission(7)
    assert [event[0] for event in client.events].count("post") == 1


@pytest.mark.asyncio
async def test_explicit_registration_failure_is_terminal_without_removal():
    client = RegistrationFlow()
    client.register_failure = GatewayRegistrationOperationError("control job failed", "accepted", client.url)
    client.settlement_state = "failed"
    control = NativeGatewayLifecycle(client, [client.url])
    with pytest.raises(GatewayControlError, match="At least one native registration failed"):
        await control.start_admission(7)
    assert client.workers == {}
    assert ("settle", "accepted", client.url) in client.events
    assert not any(event[0] == "delete" for event in client.events)
    assert [event[0] for event in client.events].count("post") == 1


@pytest.mark.asyncio
async def test_identity_conflict_is_unconfirmed_and_never_deletes_the_foreign_worker():
    client = RegistrationFlow()
    client.register_failure = GatewayRegistrationOperationTimeout("registration timeout", "accepted", client.url)
    client.settlement_failure = GatewayRegistrationOperationError("identity conflict", "accepted", client.url)
    control = NativeGatewayLifecycle(client, [client.url])
    with pytest.raises(GatewayIsolationUnconfirmed):
        await control.start_admission(7)
    assert client.workers == {client.url: {"id": "foreign", "url": client.url}}
    assert not any(event[0] == "delete" for event in client.events)
    assert [event[0] for event in client.events].count("post") == 1
    with pytest.raises(GatewayControlError, match="permanently quarantined"):
        await control.start_admission(7)


@pytest.mark.asyncio
async def test_cleanup_failure_is_unconfirmed_and_quarantined():
    client = RegistrationFlow()
    client.register_failure = GatewayRegistrationOperationTimeout("registration timeout", "accepted", client.url)
    client.late_worker_id = "accepted"
    client.fail_remove = True
    control = NativeGatewayLifecycle(client, [client.url])
    with pytest.raises(GatewayIsolationUnconfirmed):
        await control.start_admission(7)
    assert client.workers == {client.url: {"id": "accepted", "url": client.url}}
    assert [event[0] for event in client.events].count("post") == 1
    with pytest.raises(GatewayControlError, match="permanently quarantined"):
        await control.start_admission(7)


@pytest.mark.asyncio
async def test_lost_post_acceptance_is_always_unconfirmed_and_never_resubmitted():
    client = RegistrationFlow()
    client.register_failure = GatewayRegistrationAcceptanceUnknown("lost POST response")
    control = NativeGatewayLifecycle(client, [client.url])
    with pytest.raises(GatewayIsolationUnconfirmed, match="acceptance remains unknown"):
        await control.start_admission(7)
    assert [event[0] for event in client.events].count("post") == 1
    assert ("observe_unknown", client.url) in client.events
    assert not any(event[0] == "delete" for event in client.events)
    with pytest.raises(GatewayControlError, match="permanently quarantined"):
        await control.start_admission(7)


@pytest.mark.asyncio
async def test_unknown_registration_still_cleans_successful_own_replicas_independently():
    urls = ["http://a", "http://b"]
    client = Registry([])
    original_register = client.register_worker

    async def register_with_lost_post(spec, *, expected_weight_version):
        if spec["url"] == urls[1]:
            client.events.append(("register", spec["url"], expected_weight_version))
            raise GatewayRegistrationAcceptanceUnknown("lost POST response")
        return await original_register(spec, expected_weight_version=expected_weight_version)

    client.register_worker = register_with_lost_post
    control = NativeGatewayLifecycle(client, urls)
    with pytest.raises(GatewayIsolationUnconfirmed, match="acceptance remains unknown"):
        await control.start_admission(7)
    assert client.workers == {}
    assert [event[0] for event in client.events].count("register") == 2
    assert not any(event[0] == "settle" and event[1] == urls[0] for event in client.events)
    assert ("remove", urls[0]) in client.events
    assert ("observe_unknown", urls[1]) in client.events


@pytest.mark.asyncio
async def test_stop_admission_never_deletes_a_foreign_identity_at_an_owned_url():
    client = Registry(["http://a"])
    control = NativeGatewayLifecycle(client, ["http://a"])
    with pytest.raises(GatewayIsolationUnconfirmed):
        await control.stop_admission()
    assert set(client.workers) == {"http://a"}
    assert not any(event[0] == "remove" for event in client.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_weight_sync_reopens_admission_only_after_success(failure):
    events = []

    class Control:
        async def stop_admission(self):
            events.append("isolated")

        async def start_admission(self, step):
            events.append(("ready", step))

    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.gateway_control = Control()

    async def update(self, global_steps):
        events.append(("weights", global_steps))
        if failure:
            raise RuntimeError("weight transfer failed")
        return {"bytes": 17}

    manager._update_weights = MethodType(update, manager)
    if failure:
        with pytest.raises(RuntimeError, match="weight transfer failed"):
            await manager.update_weights(7)
        assert events == ["isolated", ("weights", 7)]
    else:
        result = await manager.update_weights(7)
        payload = {key: value for key, value in result.items() if not key.startswith("timing_s/checkpoint/")}
        assert payload == {"bytes": 17}
        assert all(math.isfinite(value) and value >= 0 for key, value in result.items()
                   if key.startswith("timing_s/checkpoint/"))
        assert events == ["isolated", ("weights", 7), ("ready", 7)]


@pytest.mark.asyncio
async def test_native_sync_without_version_fails_before_external_action():
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.gateway_control = object()
    with pytest.raises(ValueError, match="global_steps"):
        await manager.update_weights(None)


@pytest.mark.asyncio
async def test_default_weight_sync_keeps_optional_version_contract():
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.gateway_control = None
    versions = []

    async def update(self, version):
        versions.append(version)
        return {}

    manager._update_weights = MethodType(update, manager)
    result = await manager.update_weights(None)
    assert set(result) == {"timing_s/checkpoint/total"}
    assert math.isfinite(result["timing_s/checkpoint/total"]) and result["timing_s/checkpoint/total"] >= 0
    assert versions == [None]


@pytest.mark.asyncio
async def test_native_transport_bypasses_ray_and_preserves_partial_budget(monkeypatch):
    grants = []

    class Transport:
        async def generate(self, *, request_id, prompt_ids, sampling_params, **kwargs):
            assert "mm_processor_output" not in kwargs
            grants.append((list(prompt_ids), dict(sampling_params)))
            step = len(grants)
            return TokenOutput(
                token_ids=[40 + step],
                log_probs=[-0.5],
                stop_reason="aborted" if step == 1 else "length",
                extra_fields={"global_steps": step, "min_global_steps": step, "max_global_steps": step},
            )

    async def fail_ray(*args, **kwargs):
        raise AssertionError("Ray load balancer must not choose a native Gateway replica")

    async def no_wait(*args):
        return None

    monkeypatch.setattr(LLMServerClient, "_acquire_server", fail_ray)
    monkeypatch.setattr("verl.workers.rollout.llm_server.asyncio.sleep", no_wait)
    config = OmegaConf.create(
        {"actor_rollout_ref": {"rollout": {"name": "sglang", "full_determinism": False, "response_length": 2}}}
    )
    caller = {"temperature": 0.0}
    client = FullyAsyncLLMServerClient(config, generation_transport=Transport())
    result = await client.generate("trajectory", prompt_ids=[11], sampling_params=caller, mm_processor_output=None)
    assert result.token_ids == [41, 42]
    assert [row["global_steps"] for row in result.extra_fields["native_gateway_attempts"]] == [1, 2]
    assert grants == [([11], {"temperature": 0.0, "max_tokens": 2}), ([11, 41], {"temperature": 0.0, "max_tokens": 1})]


def test_disabled_gateway_factory_preserves_original_client():
    manager = LLMServerManager.__new__(LLMServerManager)
    manager.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"name": "sglang"}}})
    manager.rollout_config = manager.config.actor_rollout_ref.rollout
    manager.global_load_balancer = object()
    manager.server_addresses = []
    client = manager.get_client()
    assert type(client) is LLMServerClient
    assert client._generation_transport is None
    assert manager.get_gateway_lifecycle() is None


@pytest.mark.asyncio
async def test_unknown_accepted_sibling_does_not_skip_successful_owned_cleanup():
    urls = ["http://a", "http://b"]
    client = Registry([])
    original_register = client.register_worker

    async def accepted_but_pending(spec, *, expected_weight_version):
        if spec["url"] == urls[1]:
            client.events.append(("register", spec["url"], expected_weight_version))
            raise GatewayRegistrationOperationTimeout("pending timeout", "pending-b", urls[1])
        return await original_register(spec, expected_weight_version=expected_weight_version)

    client.register_worker = accepted_but_pending
    control = NativeGatewayLifecycle(client, urls)
    with pytest.raises(GatewayIsolationUnconfirmed):
        await control.start_admission(7)
    assert client.workers == {}
    assert ("remove", urls[0]) in client.events
    assert not any(event[0] == "remove" and event[1] == urls[1] for event in client.events)
    assert [event[0] for event in client.events].count("register") == 2
    with pytest.raises(GatewayControlError, match="permanently quarantined"):
        await control.start_admission(7)


@pytest.mark.asyncio
async def test_cancelled_registration_quarantines_instead_of_inferring_empty_is_safe():
    client = RegistrationFlow()
    posted = asyncio.Event()

    async def blocked_post(spec, *, expected_weight_version):
        client.events.append(("post", spec["url"], expected_weight_version))
        posted.set()
        await asyncio.Event().wait()

    client.register_worker = blocked_post
    control = NativeGatewayLifecycle(client, [client.url])
    task = asyncio.create_task(control.start_admission(7))
    await asyncio.wait_for(posted.wait(), timeout=1)
    task.cancel()
    with pytest.raises(GatewayIsolationUnconfirmed, match="cancelled"):
        await task
    assert client.workers == {}
    assert not any(event[0] == "delete" for event in client.events)
    with pytest.raises(GatewayControlError, match="permanently quarantined"):
        await control.start_admission(7)
    assert [event[0] for event in client.events].count("post") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_stage", ["settlement", "removal"])
async def test_cancelled_failed_registration_cleanup_forbids_reuse_and_weight_update(cancel_stage):
    client = RegistrationFlow()
    client.register_failure = GatewayRegistrationOperationTimeout("registration timeout", "accepted", client.url)
    cleanup_entered = asyncio.Event()

    async def blocked_settlement(worker_id, *, url):
        cleanup_entered.set()
        await asyncio.Event().wait()

    async def blocked_removal(worker_id, *, url):
        cleanup_entered.set()
        await asyncio.Event().wait()

    if cancel_stage == "settlement":
        client.settle_registration = blocked_settlement
    else:
        client.late_worker_id = "accepted"
        client.remove_worker = blocked_removal
    control = NativeGatewayLifecycle(client, [client.url])
    task = asyncio.create_task(control.start_admission(7))
    await asyncio.wait_for(cleanup_entered.wait(), timeout=1)
    task.cancel("cleanup interrupted")
    with pytest.raises(asyncio.CancelledError, match="cleanup interrupted"):
        await task

    client.workers.clear()  # Transient absence does not settle the cancelled native operation.
    with pytest.raises(GatewayIsolationUnconfirmed, match="permanently quarantined"):
        await control.stop_admission()
    with pytest.raises(GatewayControlError, match="permanently quarantined"):
        await control.start_admission(8)
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.gateway_control = control

    async def forbidden_update(self, global_steps):
        raise AssertionError("Cancelled native cleanup must not enter weight update")

    manager._update_weights = MethodType(forbidden_update, manager)
    with pytest.raises(GatewayIsolationUnconfirmed, match="permanently quarantined"):
        await manager.update_weights(8)


@pytest.mark.asyncio
async def test_failed_removal_forbids_empty_registry_reuse_and_weight_update():
    client = Registry([])
    control = NativeGatewayLifecycle(client, ["http://a"])
    await control.start_admission(7)
    client.fail_remove = "http://a"
    with pytest.raises(GatewayIsolationUnconfirmed):
        await control.stop_admission()
    client.workers.clear()  # transient absence cannot prove failed native DELETE terminal
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.gateway_control = control

    async def forbidden_update(self, global_steps):
        raise AssertionError("Unknown native isolation must not enter weight update")

    manager._update_weights = MethodType(forbidden_update, manager)
    with pytest.raises(GatewayIsolationUnconfirmed, match="permanently quarantined"):
        await manager.update_weights(8)
    with pytest.raises(GatewayControlError, match="permanently quarantined"):
        await control.start_admission(8)


def _continuous_publication_manager_config(*, root_dir="/tmp/root", run_id="run-a"):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"lora_rank": 0, "hf_config": {}},
                "hybrid_engine": False,
                "rollout": {
                    "name": "sglang",
                    "mode": "async",
                    "sglang_gateway": {
                        "enabled": True,
                        "url": "http://gateway:30000",
                        "request_timeout_s": 1,
                        "control_timeout_s": 1,
                        "operation_timeout_s": 1,
                        "poll_interval_s": 0.1,
                        "api_key_env": None,
                    },
                    "prompt_length": 8,
                    "response_length": 8,
                    "max_model_len": 16,
                    "topk_log_probs": 0,
                    "full_determinism": False,
                    "disaggregation": {"enabled": False},
                    "mtp": {"enable": False},
                    "trajectory_state": {
                        "enabled": True,
                        "continuous_policy_publication": True,
                        "root_dir": root_dir,
                        "run_id": run_id,
                        "policy_identity": {
                            "policy_version": "0",
                            "model_weights_fingerprint": "weights-0",
                            "tokenizer_template_fingerprint": "tok",
                            "kv_cache_schema_fingerprint": "kv",
                        },
                        "model_configuration_fingerprint": "model-config",
                        "environment_version": "env",
                        "tool_version": "tool",
                        "coordinator_max_restarts": 1,
                    },
                },
            },
            "trainer": {
                "use_v1": True,
                "v1": {
                    "trainer_mode": "separate_async",
                    "separate_async": {"hybrid_rollout": {"enable_switch": False}},
                },
            },
        }
    )


@pytest.mark.asyncio
async def test_continuous_publication_transport_builds_active_identity_cache_salt():
    identity = {
        "policy_version": "7",
        "model_weights_fingerprint": "weights-7",
        "tokenizer_template_fingerprint": "tok",
        "kv_cache_schema_fingerprint": "kv",
    }

    class PublicationClient:
        async def get_active_published_policy(self):
            return {"identity": identity}

    seen = []

    class Transport:
        @staticmethod
        def build_policy_cache_salt(*, run_namespace, policy_version, policy_identity_salt):
            return f"{run_namespace}:{policy_version}:{policy_identity_salt}"

        async def generate(self, **kwargs):
            seen.append(kwargs["cache_salt"])
            return TokenOutput(token_ids=[1], log_probs=[-0.1], stop_reason="stop", extra_fields={"global_steps": 7})

    transport = ContinuousPublicationTransport(Transport(), PublicationClient(), run_id="run-a")
    await transport.generate(request_id="r", prompt_ids=[1], sampling_params={"logprobs": True})
    from verl.trainer.ppo.v1.trajectory_state import CanonicalPolicyKVIdentity

    expected = CanonicalPolicyKVIdentity(**identity).cache_salt
    assert seen == [f"run-a:7:{expected}"]


@pytest.mark.asyncio
async def test_continuous_publication_transport_blocks_foreign_caller_salt():
    class PublicationClient:
        async def get_active_published_policy(self):
            return {
                "identity": {
                    "policy_version": "7",
                    "model_weights_fingerprint": "weights-7",
                    "tokenizer_template_fingerprint": "tok",
                    "kv_cache_schema_fingerprint": "kv",
                }
            }

    class Transport:
        @staticmethod
        def build_policy_cache_salt(*, run_namespace, policy_version, policy_identity_salt):
            return f"{run_namespace}:{policy_version}:{policy_identity_salt}"

        async def generate(self, **kwargs):
            raise AssertionError("foreign caller salt must be rejected before dispatch")

    transport = ContinuousPublicationTransport(Transport(), PublicationClient(), run_id="run-a")
    with pytest.raises(GatewayProtocolError, match="cache_salt"):
        await transport.generate(
            request_id="r", prompt_ids=[1], sampling_params={"logprobs": True}, cache_salt="run-b:7:x"
        )


def test_continuous_publication_gateway_lifecycle_wraps_native_control(monkeypatch):
    class FakeClient:
        actor = SimpleNamespace(summary=SimpleNamespace(remote=lambda: {"run_id": "run-a", "root_dir": "/tmp/root"}))

    monkeypatch.setattr(
        "verl.workers.rollout.llm_server.get_or_create_trajectory_coordinator", lambda config: (object(), FakeClient())
    )
    monkeypatch.setattr(
        "verl.workers.rollout.llm_server.observe_owned_engine_versions",
        lambda addresses, timeout_s: [
            {"engine_url": "http://engine:30001", "weight_version": "7", "source": "model_info"}
        ],
    )
    monkeypatch.setattr("verl.workers.rollout.llm_server.ray.get", lambda value: value)
    manager = LLMServerManager.__new__(LLMServerManager)
    manager.config = _continuous_publication_manager_config()
    manager.rollout_config = manager.config.actor_rollout_ref.rollout
    manager.model_config = SimpleNamespace()
    manager.server_addresses = ["http://engine:30001"]
    lifecycle = manager.get_gateway_lifecycle()
    assert isinstance(lifecycle, TrajectoryAwareGatewayLifecycle)


def test_continuous_publication_gateway_lifecycle_uses_normalized_engine_urls(monkeypatch):
    class FakeClient:
        actor = SimpleNamespace(summary=SimpleNamespace(remote=lambda: {"run_id": "run-a", "root_dir": "/tmp/root"}))

    monkeypatch.setattr(
        "verl.workers.rollout.llm_server.get_or_create_trajectory_coordinator", lambda config: (object(), FakeClient())
    )
    monkeypatch.setattr("verl.workers.rollout.llm_server.ray.get", lambda value: value)
    manager = LLMServerManager.__new__(LLMServerManager)
    manager.config = _continuous_publication_manager_config()
    manager.rollout_config = manager.config.actor_rollout_ref.rollout
    manager.model_config = SimpleNamespace()
    manager.server_addresses = ["engine:30001"]
    lifecycle = manager.get_gateway_lifecycle()
    assert lifecycle._lifecycle.urls == ("http://engine:30001",)
    assert tuple(lifecycle._publication_hooks.engine_urls) == ("http://engine:30001",)


def test_continuous_publication_client_factory_wraps_transport(monkeypatch):
    class FakeClient:
        actor = SimpleNamespace(summary=SimpleNamespace(remote=lambda: {"run_id": "run-a", "root_dir": "/tmp/root"}))

    monkeypatch.setattr(
        "verl.workers.rollout.llm_server.get_or_create_trajectory_coordinator", lambda config: (object(), FakeClient())
    )
    monkeypatch.setattr("verl.workers.rollout.llm_server.ray.get", lambda value: value)
    manager = LLMServerManager.__new__(LLMServerManager)
    manager.config = _continuous_publication_manager_config()
    manager.rollout_config = manager.config.actor_rollout_ref.rollout
    manager.global_load_balancer = object()
    manager.server_addresses = ["http://engine:30001"]
    client = manager.get_client()
    assert isinstance(client._generation_transport, ContinuousPublicationTransport)


def test_foreign_root_dir_is_rejected_before_reusing_named_coordinator(monkeypatch):
    class FakeClient:
        actor = SimpleNamespace(summary=SimpleNamespace(remote=lambda: {"run_id": "run-a", "root_dir": "/tmp/foreign"}))

    monkeypatch.setattr(
        "verl.workers.rollout.llm_server.get_or_create_trajectory_coordinator", lambda config: (object(), FakeClient())
    )
    monkeypatch.setattr("verl.workers.rollout.llm_server.ray.get", lambda value: value)
    manager = LLMServerManager.__new__(LLMServerManager)
    manager.config = _continuous_publication_manager_config(root_dir="/tmp/root", run_id="run-a")
    manager.rollout_config = manager.config.actor_rollout_ref.rollout
    manager.global_load_balancer = object()
    manager.server_addresses = ["http://engine:30001"]
    with pytest.raises(ValueError, match="root_dir"):
        manager.get_client()


@pytest.mark.asyncio
async def test_continuous_publication_transport_blocks_unknown_publication_before_dispatch():
    class PublicationClient:
        async def get_active_published_policy(self):
            raise RuntimeError("unknown publication")

    class Transport:
        async def generate(self, **kwargs):
            raise AssertionError("unknown publication must fail before dispatch")

    transport = ContinuousPublicationTransport(Transport(), PublicationClient(), run_id="run-a")
    with pytest.raises(RuntimeError, match="unknown publication"):
        await transport.generate(request_id="r", prompt_ids=[1], sampling_params={"logprobs": True})


def test_continuous_publication_transport_cloudpickle_round_trip_preserves_delegate_and_fields():
    class PublicationClient:
        async def get_active_published_policy(self):
            return {
                "identity": {
                    "policy_version": "7",
                    "model_weights_fingerprint": "weights-7",
                    "tokenizer_template_fingerprint": "tok",
                    "kv_cache_schema_fingerprint": "kv",
                }
            }

    class SerializableTransport:
        @staticmethod
        def build_policy_cache_salt(**kwargs):
            return "unused"

        def close(self):
            return None

    transport = ContinuousPublicationTransport(SerializableTransport(), PublicationClient(), run_id="run-a")
    restored = cloudpickle.loads(cloudpickle.dumps(transport))
    assert restored._run_id == "run-a"
    assert callable(restored.build_policy_cache_salt)
    assert restored.close() is None


def test_continuous_publication_transport_uninitialized_delegate_raises_attribute_error_not_recursion():
    restored = cloudpickle.loads(
        cloudpickle.dumps(ContinuousPublicationTransport.__new__(ContinuousPublicationTransport))
    )
    with pytest.raises(AttributeError, match="close"):
        assert restored.close is None


def test_sglang_gateway_client_cloudpickle_round_trip_with_unopened_transports():
    from verl.workers.rollout.sglang_gateway_client import GatewayClientConfig, SGLangGatewayClient

    client = SGLangGatewayClient(
        GatewayClientConfig(
            base_url="http://gateway:30000",
            max_model_len=16,
            prompt_length=8,
            response_length=8,
            expected_worker_urls=("http://engine:30001",),
        )
    )
    restored = cloudpickle.loads(cloudpickle.dumps(client))
    assert restored.config.base_url == "http://gateway:30000"
    assert restored.config.expected_worker_urls == ("http://engine:30001",)


def test_trajectory_aware_gateway_lifecycle_cloudpickle_round_trip_preserves_delegate_and_hooks():
    class SerializableLifecycle:
        urls = ("http://engine:30001",)

        async def stop_admission(self):
            return None

    class SerializableHooks:
        async def prepare_policy_publication(self, global_steps):
            return global_steps

    wrapper = TrajectoryAwareGatewayLifecycle(SerializableLifecycle(), SerializableHooks())
    restored = cloudpickle.loads(cloudpickle.dumps(wrapper))
    assert restored.urls == ("http://engine:30001",)
    assert restored._publication_hooks.__class__.__name__ == "SerializableHooks"


def test_trajectory_aware_gateway_lifecycle_uninitialized_delegate_raises_attribute_error_not_recursion():
    restored = cloudpickle.loads(
        cloudpickle.dumps(TrajectoryAwareGatewayLifecycle.__new__(TrajectoryAwareGatewayLifecycle))
    )
    with pytest.raises(AttributeError, match="urls"):
        assert restored.urls == ()
