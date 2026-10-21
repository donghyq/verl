# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Native Gateway contracts without a model, GPU, or external Gateway process."""

from __future__ import annotations

from collections import deque
from copy import deepcopy

import pytest

from verl.workers.rollout.sglang_gateway_client import (
    GatewayClientConfig,
    GatewayControlError,
    GatewayHTTPError,
    GatewayHTTPResponse,
    GatewayMixedVersionError,
    GatewayOperationTimeout,
    GatewayProtocolError,
    GatewayRegistrationAcceptanceUnknown,
    SGLangGatewayClient,
)


def _response(body, status=200, headers=None):
    return GatewayHTTPResponse(status, body, headers or {})


def _generation(*, request_id="request-1", version="4", finish="stop", tokens=None, spans=None):
    tokens = [11, 12] if tokens is None else tokens
    meta = {
        "id": request_id,
        "weight_version": version,
        "finish_reason": {"type": finish},
        "output_token_logprobs": [[-0.1, token, None] for token in tokens],
        "cached_tokens": 0,
        "num_retractions": 0,
    }
    if spans is not None:
        meta["weight_versions"] = spans
    return {"output_ids": tokens, "meta_info": meta}


class ScriptedTransport:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.calls = []
        self.closed = 0

    async def request(self, method, path, *, body=None):
        self.calls.append((method, path, deepcopy(body)))
        item = self.responses.popleft()
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self):
        self.closed += 1


def _config(**overrides):
    values = {
        "base_url": "http://gateway:30000",
        "max_model_len": 1024,
        "prompt_length": 512,
        "response_length": 128,
        "poll_interval_s": 0.001,
        "operation_timeout_s": 1,
        "expected_worker_urls": ("http://engine:30001",),
    }
    values.update(overrides)
    return GatewayClientConfig(**values)


def _client(inference=None, control=None, **config):
    return SGLangGatewayClient(
        _config(**config),
        inference_transport=inference or ScriptedTransport(),
        control_transport=control or ScriptedTransport(*_active_reads()),
    )


@pytest.mark.asyncio
async def test_engine_versions_are_read_directly_before_gateway_registration(monkeypatch):
    import verl.workers.rollout.sglang_gateway_client as module

    transports = {
        "http://engine-a": ScriptedTransport(_response({"weight_version": "7"})),
        "http://engine-b": ScriptedTransport(_response({"weight_version": "7"})),
    }
    monkeypatch.setattr(module, "_AiohttpTransport", lambda url, timeout, key: transports[url])
    client = _client()
    assert await client.observe_unregistered_engine_versions(tuple(transports)) == {
        "http://engine-a": 7,
        "http://engine-b": 7,
    }
    assert all(t.calls == [("GET", "/model_info", None)] and t.closed == 1 for t in transports.values())
    assert client._control.calls == []


@pytest.mark.asyncio
async def test_unregistered_engine_missing_version_fails_and_closes_transport(monkeypatch):
    import verl.workers.rollout.sglang_gateway_client as module

    transport = ScriptedTransport(_response({"model_path": "model"}))
    monkeypatch.setattr(module, "_AiohttpTransport", lambda url, timeout, key: transport)
    client = _client()
    with pytest.raises(GatewayProtocolError):
        await client.observe_unregistered_engine_versions(("http://engine-a",))
    assert transport.closed == 1
    assert client._control.calls == []


async def _generate(client, **kwargs):
    return await client.generate(
        "request-1", prompt_ids=[1, 2, 3], sampling_params=kwargs.pop("sampling_params", {"logprobs": True}), **kwargs
    )


@pytest.mark.asyncio
async def test_one_attempt_preserves_tokens_sampling_seed_and_native_version():
    inference = ScriptedTransport(
        _response(
            _generation(spans=[{"version": "4", "start": 0, "end": 2}]),
            headers={"X-SMG-Routed-Worker-ID": "worker-4"},
        )
    )
    params = {"max_tokens": 7, "temperature": 0.7, "sampling_seed": 73, "logprobs": True}
    original = deepcopy(params)
    output = await _generate(_client(inference), sampling_params=params, cache_salt="4")

    assert params == original
    assert inference.calls == [
        (
            "POST",
            "/generate",
            {
                "rid": "request-1",
                "input_ids": [1, 2, 3],
                "sampling_params": {"temperature": 0.7, "sampling_seed": 73, "max_new_tokens": 7},
                "return_logprob": True,
                "stream": False,
                "cache_salt": "4",
            },
        )
    ]
    assert output.token_ids == [11, 12]
    assert output.log_probs == [-0.1, -0.1]
    assert output.extra_fields["global_steps"] == 4
    assert output.extra_fields["engine_weight_versions"] == [{"version": "4", "start": 0, "end": 2}]
    assert output.extra_fields["gateway_worker_id"] == "worker-4"
    assert output.stop_reason == "stop"


@pytest.mark.asyncio
async def test_unknown_worker_and_cached_tokens_remain_unknown():
    body = _generation()
    del body["meta_info"]["cached_tokens"]
    output = await _generate(_client(ScriptedTransport(_response(body))))
    assert output.extra_fields["num_cached_tokens"] is None
    assert output.extra_fields["gateway_worker_id"] is None


@pytest.mark.asyncio
async def test_abort_returns_partial_tokens_without_internal_resume():
    inference = ScriptedTransport(_response(_generation(finish="abort")))
    output = await _generate(_client(inference))
    assert output.stop_reason == "abort"
    assert output.token_ids == [11, 12]
    assert len(inference.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError("unknown dispatch"), ConnectionError("lost response")])
async def test_uncertain_generation_failure_is_not_retried(failure):
    inference = ScriptedTransport(failure, _response(_generation()))
    with pytest.raises(type(failure)):
        await _generate(_client(inference))
    assert len(inference.calls) == 1


@pytest.mark.asyncio
async def test_last_version_does_not_hide_mixed_version_generation():
    body = _generation(
        version="5",
        spans=[{"version": "4", "start": 0, "end": 1}, {"version": "5", "start": 1, "end": 2}],
    )
    with pytest.raises(GatewayMixedVersionError):
        await _generate(
            _client(ScriptedTransport(_response(body)), control=ScriptedTransport(*_active_reads(version="5")))
        )


@pytest.mark.asyncio
async def test_zero_budget_has_no_phantom_logprob():
    body = _generation(tokens=[], finish="length", spans=[{"version": "4", "start": 0, "end": 0}])
    output = await _generate(
        _client(ScriptedTransport(_response(body))), sampling_params={"max_tokens": 0, "logprobs": True}
    )
    assert output.token_ids == output.log_probs == []


@pytest.mark.asyncio
async def test_unsupported_semantics_fail_before_dispatch():
    inference = ScriptedTransport()
    client = _client(inference)
    for kwargs in ({"audio_data": [1]}, {"priority": 3}, {"trajectory_kv_ref": {}}, {"mm_processor_kwargs": {"x": 1}}):
        with pytest.raises(GatewayProtocolError):
            await _generate(client, **kwargs)
    assert not inference.calls


def _worker(*, worker_id="worker-1", url="http://engine:30001", status="ready", job=None, healthy=True):
    worker = {
        "id": worker_id,
        "url": url,
        "worker_type": "regular",
        "connection_mode": "http",
        "runtime_type": "unspecified" if status == "pending" else "sglang",
        "priority": 50,
        "cost": 1.0,
        "max_connection_attempts": 20,
        "status": status,
        "is_healthy": healthy,
        "load": 0,
        "http2": False,
    }
    if job is not None:
        worker["job_status"] = {
            "job_type": "AddWorker",
            "worker_url": url,
            "message": None,
            "timestamp": 0,
            **job,
        }
    return worker


def _active_reads(version="4"):
    return [
        _response({"workers": [_worker()]}),
        _response(_worker()),
        _response({"worker_id": "worker-1", "status": 200, "body": {"weight_version": version}}),
        _response({"workers": [_worker()]}),
    ]


def _registration_receipt():
    return _response({"worker_id": "worker-1", "url": "http://engine:30001"}, status=202)


@pytest.mark.asyncio
async def test_private_dynamic_membership_accepts_only_nonempty_configured_replica_subset():
    expected = ("http://engine:30001", "http://borrowed:30001")
    client = _client(control=ScriptedTransport(*_active_reads()),
                     expected_worker_urls=expected, allow_partial_worker_set=True)
    assert await client.get_active_weight_version(expected) == 4

    foreign = _worker(url="http://foreign:30001")
    client = _client(control=ScriptedTransport(_response({"workers": [foreign]})),
                     expected_worker_urls=expected, allow_partial_worker_set=True)
    with pytest.raises(GatewayControlError, match="unexpected worker"):
        await client.get_active_weight_version(expected)


@pytest.mark.asyncio
async def test_default_membership_still_requires_every_configured_replica():
    from verl.workers.rollout.sglang_gateway_client import GatewayNotReady

    expected = ("http://engine:30001", "http://borrowed:30001")
    client = _client(control=ScriptedTransport(_response({"workers": [_worker()]})), expected_worker_urls=expected)
    with pytest.raises(GatewayNotReady, match="not yet ready"):
        await client.get_active_weight_version(expected)


@pytest.mark.asyncio
async def test_dynamic_membership_cannot_dispatch_when_every_engine_is_removed():
    from verl.workers.rollout.sglang_gateway_client import GatewayNotReady

    client = _client(control=ScriptedTransport(_response({"workers": []})), allow_partial_worker_set=True)
    with pytest.raises(GatewayNotReady):
        await client.get_active_weight_version(client.config.expected_worker_urls)


@pytest.mark.asyncio
async def test_registration_job_gap_keeps_the_accepted_identity_and_verifies_engine_version():
    control = ScriptedTransport(
        _registration_receipt(),
        _response(_worker(status="pending", job={"status": "processing"}, healthy=False)),
        _response({"error": "Worker not found"}, status=404),
        _response({"workers": []}),
        _response(_worker()),
        _response({"worker_id": "worker-1", "status": 200, "body": {"weight_version": "7"}}),
    )
    observation = await _client(control=control).register_worker(
        {"url": "http://engine:30001", "worker_type": "regular"}, expected_weight_version=7
    )
    assert observation.worker_id == "worker-1" and observation.engine_weight_version == "7"
    assert [call[0] for call in control.calls].count("POST") == 1
    assert control.calls[-1][1] == "/v1/rl/workers/worker-1/engine/model_info"


@pytest.mark.asyncio
async def test_missing_registration_is_bounded_and_never_resubmitted():
    class MissingTransport(ScriptedTransport):
        async def request(self, method, path, *, body=None):
            self.calls.append((method, path, deepcopy(body)))
            if method == "POST":
                return _registration_receipt()
            if path == "/workers":
                return _response({"workers": []})
            return _response({"error": "Worker not found"}, status=404)

    control = MissingTransport()
    with pytest.raises(GatewayOperationTimeout, match="Registration did not become ready"):
        await _client(control=control, operation_timeout_s=0.02).register_worker(
            {"url": "http://engine:30001"}, expected_weight_version=7
        )
    assert [call[0] for call in control.calls].count("POST") == 1


@pytest.mark.asyncio
async def test_registration_can_finish_between_missing_job_read_and_registry_read():
    control = ScriptedTransport(
        _registration_receipt(),
        _response({"error": "Worker not found"}, status=404),
        _response({"workers": [_worker()]}),
        _response(_worker()),
        _response({"worker_id": "worker-1", "status": 200, "body": {"weight_version": "7"}}),
    )
    observation = await _client(control=control).register_worker(
        {"url": "http://engine:30001"}, expected_weight_version=7
    )
    assert observation.worker_id == "worker-1" and observation.engine_weight_version == "7"
    assert [call[0] for call in control.calls].count("POST") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conflicting_worker",
    [_worker(worker_id="another-worker"), _worker(url="http://another-engine:30002")],
)
async def test_registration_gap_rejects_conflicting_registry_identity(conflicting_worker):
    control = ScriptedTransport(
        _registration_receipt(),
        _response({"error": "Worker not found"}, status=404),
        _response({"workers": [conflicting_worker]}),
    )
    with pytest.raises(GatewayControlError, match="accepted identity"):
        await _client(control=control).register_worker({"url": "http://engine:30001"}, expected_weight_version=7)
    assert [call[0] for call in control.calls].count("POST") == 1


@pytest.mark.asyncio
async def test_registration_native_failure_is_not_treated_as_a_job_gap():
    control = ScriptedTransport(
        _registration_receipt(),
        _response(_worker(status="pending", job={"status": "failed"}, healthy=False)),
    )
    with pytest.raises(GatewayControlError, match="control job failed"):
        await _client(control=control).register_worker({"url": "http://engine:30001"}, expected_weight_version=7)
    assert len(control.calls) == 2


@pytest.mark.asyncio
async def test_registration_list_during_detail_gap_cannot_prove_job_terminal():
    control = ScriptedTransport(
        _registration_receipt(),
        _response({"error": "Worker not found"}, status=404),
        _response({"workers": [_worker()]}),
        _response(_worker(status="pending", job={"status": "processing"}, healthy=False)),
        _response(_worker(status="pending", job={"status": "failed"}, healthy=False)),
    )
    with pytest.raises(GatewayControlError, match="control job failed"):
        await _client(control=control).register_worker({"url": "http://engine:30001"}, expected_weight_version=7)
    assert len(control.calls) == 5
    assert [call[0] for call in control.calls].count("POST") == 1


@pytest.mark.asyncio
async def test_registration_after_job_gap_still_rejects_wrong_weight_version():
    control = ScriptedTransport(
        _registration_receipt(),
        _response({"error": "Worker not found"}, status=404),
        _response({"workers": []}),
        _response(_worker()),
        _response({"worker_id": "worker-1", "status": 200, "body": {"weight_version": "6"}}),
    )
    with pytest.raises(GatewayControlError, match="different weight version"):
        await _client(control=control).register_worker({"url": "http://engine:30001"}, expected_weight_version=7)


@pytest.mark.asyncio
async def test_registration_settlement_waits_for_exact_terminal_identity():
    control = ScriptedTransport(
        _response({"error": "Worker not found"}, status=404),
        _response({"workers": [_worker()]}),
        _response(_worker(status="pending", job={"status": "processing"}, healthy=False)),
        _response(_worker()),
    )
    settlement = await _client(control=control).settle_registration("worker-1", url="http://engine:30001")
    assert settlement.worker_id == "worker-1"
    assert settlement.state == "ready"
    assert len(control.calls) == 4
    assert not any(call[0] == "POST" for call in control.calls)


@pytest.mark.asyncio
async def test_registration_settlement_reports_explicit_failure_as_terminal():
    control = ScriptedTransport(
        _response(_worker(status="pending", job={"status": "failed"}, healthy=False)),
    )
    settlement = await _client(control=control).settle_registration("worker-1", url="http://engine:30001")
    assert settlement.state == "failed"
    assert not any(call[0] == "POST" for call in control.calls)


@pytest.mark.asyncio
async def test_registration_settlement_rejects_a_foreign_identity_at_the_same_url():
    control = ScriptedTransport(
        _response({"error": "Worker not found"}, status=404),
        _response({"workers": [_worker(worker_id="foreign-worker")]}),
    )
    with pytest.raises(GatewayControlError, match="observations disagree"):
        await _client(control=control).settle_registration("worker-1", url="http://engine:30001")
    assert not any(call[0] == "POST" for call in control.calls)


@pytest.mark.asyncio
async def test_registration_settlement_unknown_404_and_pending_stays_unknown():
    class UnknownSettlementTransport(ScriptedTransport):
        async def request(self, method, path, *, body=None):
            self.calls.append((method, path, deepcopy(body)))
            if path == "/workers":
                return _response({"workers": []})
            return _response({"error": "Worker not found"}, status=404)

    control = UnknownSettlementTransport()
    with pytest.raises(GatewayOperationTimeout, match="settlement remains unknown"):
        await _client(control=control, operation_timeout_s=0.02).settle_registration(
            "worker-1", url="http://engine:30001"
        )
    assert not any(call[0] == "POST" for call in control.calls)


@pytest.mark.asyncio
async def test_unknown_registration_acceptance_never_becomes_a_terminal_claim():
    class EmptyRegistryTransport(ScriptedTransport):
        async def request(self, method, path, *, body=None):
            self.calls.append((method, path, deepcopy(body)))
            return _response({"workers": []})

    control = EmptyRegistryTransport()
    with pytest.raises(GatewayRegistrationAcceptanceUnknown, match="acceptance remains unknown"):
        await _client(control=control, operation_timeout_s=0.02).observe_unknown_registration("http://engine:30001")
    assert control.calls
    assert not any(call[0] == "POST" for call in control.calls)


@pytest.mark.asyncio
async def test_active_version_is_read_from_engine_and_registry_rechecked():
    control = ScriptedTransport(*_active_reads(version="8"))
    version = await _client(control=control).get_active_weight_version(["http://engine:30001"])
    assert version == 8
    assert [path for _, path, _ in control.calls] == [
        "/workers",
        "/workers/worker-1",
        "/v1/rl/workers/worker-1/engine/model_info",
        "/workers",
    ]


@pytest.mark.asyncio
async def test_native_version_observation_is_not_cached_between_generations():
    control = ScriptedTransport(*_active_reads(version="4"), *_active_reads(version="5"))
    inference = ScriptedTransport(_response(_generation(version="4")), _response(_generation(version="5")))
    client = _client(inference=inference, control=control)
    first = await _generate(client)
    second = await _generate(client)
    assert first.extra_fields["global_steps"] == 4
    assert second.extra_fields["global_steps"] == 5
    assert [body["cache_salt"] for _, _, body in inference.calls] == ["4", "5"]


@pytest.mark.asyncio
async def test_engine_version_change_after_read_preserves_legitimate_generation():
    inference = ScriptedTransport(_response(_generation(version="5")))
    output = await _generate(_client(inference))
    assert output.extra_fields["global_steps"] == 5
    assert output.extra_fields["observed_weight_version"] == 4
    assert output.extra_fields["cache_salt"] == "4"
    assert output.extra_fields["salt_match"] is False


@pytest.mark.asyncio
async def test_wrong_caller_cache_salt_does_not_override_the_actual_version():
    inference = ScriptedTransport()
    with pytest.raises(GatewayProtocolError, match="cache_salt"):
        await _generate(_client(inference), cache_salt="7")
    assert not inference.calls


@pytest.mark.asyncio
async def test_generate_waits_for_native_admission_before_its_single_post():
    control = ScriptedTransport(
        _response({"workers": []}),
        _response({"workers": [_worker()]}),
        _response(_worker(job={"status": "processing"})),
        *_active_reads(),
    )
    inference = ScriptedTransport(_response(_generation()))
    output = await _generate(_client(inference=inference, control=control, admission_timeout_s=1))
    assert output.token_ids == [11, 12]
    assert len(inference.calls) == 1


@pytest.mark.asyncio
async def test_admission_wait_is_bounded_without_a_generation_retry():
    class UnavailableTransport:
        async def request(self, method, path, *, body=None):
            return _response({"workers": []})

        async def close(self):
            pass

    inference = ScriptedTransport()
    with pytest.raises(GatewayOperationTimeout, match="admission deadline"):
        await _generate(_client(inference=inference, control=UnavailableTransport(), admission_timeout_s=0.01))
    assert not inference.calls


@pytest.mark.asyncio
async def test_invalid_membership_is_not_admission_wait():
    control = ScriptedTransport(
        _response({"workers": [_worker(), _worker(worker_id="unknown", url="http://unowned:30002")]})
    )
    inference = ScriptedTransport()
    with pytest.raises(GatewayControlError):
        await _generate(_client(inference=inference, control=control))
    assert not inference.calls


@pytest.mark.asyncio
async def test_generate_requires_actual_completed_http_response():
    inference = ScriptedTransport(_response(_generation(), 503))
    with pytest.raises(GatewayHTTPError):
        await _generate(_client(inference))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        _response({"error": "unavailable"}, status=503),
        _response({"error": "timeout"}, status=408),
        _response(None, status=202),
        _response({"url": "http://engine:30001"}, status=202),
    ],
)
async def test_unverifiable_registration_ack_is_unknown_and_posted_once(response):
    control = ScriptedTransport(response)
    with pytest.raises(GatewayRegistrationAcceptanceUnknown):
        await _client(control=control).register_worker({"url": "http://engine:30001"}, expected_weight_version=7)
    assert [method for method, _, _ in control.calls] == ["POST"]


@pytest.mark.asyncio
async def test_non_addworker_failure_cannot_prove_registration_terminal():
    control = ScriptedTransport(
        _registration_receipt(),
        _response(_worker(status="pending", job={"status": "failed", "job_type": "RemoveWorker"}, healthy=False)),
    )
    with pytest.raises(GatewayControlError) as error:
        await _client(control=control).register_worker({"url": "http://engine:30001"}, expected_weight_version=7)
    assert error.value.worker_id == "worker-1"
    assert error.value.terminal_state is None


@pytest.mark.asyncio
async def test_transport_failure_after_ack_keeps_accepted_identity_for_cleanup():
    control = ScriptedTransport(_registration_receipt(), OSError("injected read failure"))
    with pytest.raises(GatewayControlError) as error:
        await _client(control=control).register_worker({"url": "http://engine:30001"}, expected_weight_version=7)
    assert error.value.worker_id == "worker-1"
    assert error.value.url == "http://engine:30001"
    assert error.value.terminal_state is None
    assert [call[0] for call in control.calls].count("POST") == 1
