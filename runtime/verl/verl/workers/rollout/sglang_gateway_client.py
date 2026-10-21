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
"""Single-attempt SGLang generation through the native Gateway.

The fully async client still owns abort/resume and the cumulative response
budget. This module only performs one HTTP generation attempt and the native
worker-management calls required to observe readiness and worker lifecycle.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import weakref
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote, urlencode, urlsplit

from verl.workers.rollout.replica import TokenOutput

SMG_VERSION = "1.11.0"
SMG_SOURCE_COMMIT = "3be823a700fabaff3add8a390cf78f163479d686"
SMG_LINUX_X86_64_WHEEL_SHA256 = "f5c0e8f73f5770009fdab204256e7ce96dbed0ac434a4d4e27f59baf4e675831"
logger = logging.getLogger(__name__)

# Only concurrent reads share work. No completed model/version result is cached.
_MODEL_INFO_INFLIGHT = weakref.WeakKeyDictionary()


ERROR_BODY_SUMMARY_MAX_LENGTH = 2048
_SENSITIVE_BODY_FIELDS = frozenset({"auth", "authorization", "token", "api_key", "password", "cookie"})
_SENSITIVE_TEXT_FIELDS = re.compile(
    r"(?i)\b(authorization|auth|token|api[_-]?key|password|cookie)\b(\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^\s,;&]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")


def _is_sensitive_response_field(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in _SENSITIVE_BODY_FIELDS or any(
        normalized.endswith(f"_{field}") for field in _SENSITIVE_BODY_FIELDS
    )


def _redact_sensitive_response_text(value: str) -> str:
    value = _SENSITIVE_TEXT_FIELDS.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>", value
    )
    return _BEARER_TOKEN.sub("bearer <redacted>", value)


def _sanitize_error_response_body(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_sensitive_response_text(value)
    if isinstance(value, Mapping):
        return {
            key: "<redacted>" if _is_sensitive_response_field(key) else _sanitize_error_response_body(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_sanitize_error_response_body(item) for item in value]
    return value


def _error_response_body_summary(body: Any) -> str:
    """Return a bounded, control-character-safe response-body summary."""

    if body is None or (isinstance(body, bytes | bytearray | memoryview) and not body):
        return "<empty response body>"
    if isinstance(body, bytes | bytearray | memoryview):
        text = bytes(body).decode("utf-8", errors="replace")
        text = _redact_sensitive_response_text(text)
        try:
            summary = json.dumps(text, ensure_ascii=True, separators=(",", ":"))
        except (TypeError, ValueError):
            summary = json.dumps(repr(text), ensure_ascii=True, separators=(",", ":"))
    else:
        try:
            summary = json.dumps(
                _sanitize_error_response_body(body),
                ensure_ascii=True,
                separators=(",", ":"),
                default=str,
            )
        except (TypeError, ValueError):
            summary = json.dumps(repr(body), ensure_ascii=True, separators=(",", ":"))
    if len(summary) > ERROR_BODY_SUMMARY_MAX_LENGTH:
        marker = "...<truncated>"
        summary = summary[: ERROR_BODY_SUMMARY_MAX_LENGTH - len(marker)] + marker
    return summary


class GatewayError(RuntimeError):
    """Generation or control failed; the caller must not infer completion."""


class GatewayHTTPError(GatewayError):
    def __init__(self, method: str, path: str, status: int, body: Any = None):
        self.method, self.path, self.status = method, path, status
        self.body_summary = _error_response_body_summary(body)
        # Retain only the structured identity required for a read-only recheck.
        self.upstream_error = body.get("error") if isinstance(body, Mapping) else None
        self.upstream_worker_id = body.get("worker_id") if isinstance(body, Mapping) else None
        self.upstream_url = body.get("url") if isinstance(body, Mapping) else None
        super().__init__(
            f"Gateway {method} {path} returned HTTP {status}; "
            f"response_body_summary={self.body_summary}"
        )


class GatewayProtocolError(GatewayError):
    """A response cannot be mapped to verified rollout semantics."""


class GatewayMixedVersionError(GatewayProtocolError):
    """This integration does not accept multiple weight versions in one attempt."""


class GatewayControlError(GatewayError):
    """A native operation failed, was partial, or addressed the wrong workers."""


class GatewayNotReady(GatewayControlError):
    """Expected native workers are temporarily unavailable before dispatch."""


class GatewayOperationTimeout(GatewayControlError):
    """An accepted operation did not reach its required observed state in time."""


class GatewayRegistrationRejected(GatewayControlError):
    """The Gateway completed the registration request without accepting it."""


class GatewayRegistrationAcceptanceUnknown(GatewayControlError):
    """The registration request was submitted, but acceptance is not observable."""


class GatewayRegistrationOperationError(GatewayControlError):
    """An accepted registration reached a non-ready or inconsistent state."""

    def __init__(self, message: str, worker_id: str, url: str, terminal_state: str | None = None):
        self.worker_id, self.url = worker_id, url
        self.terminal_state = terminal_state
        super().__init__(message)


class GatewayRegistrationOperationTimeout(GatewayOperationTimeout):
    """An accepted registration did not reach a terminal state in time."""

    def __init__(self, message: str, worker_id: str, url: str):
        self.worker_id, self.url = worker_id, url
        super().__init__(message)


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise GatewayProtocolError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def _weight_step(value: Any) -> int:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    raise GatewayProtocolError("The engine did not report a numeric weight version")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GatewayProtocolError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class GatewayClientConfig:
    base_url: str
    max_model_len: int
    prompt_length: int
    response_length: int
    request_timeout_s: float = 14400.0
    control_timeout_s: float = 600.0
    operation_timeout_s: float = 60.0
    poll_interval_s: float = 0.1
    api_key: str | None = None
    expected_worker_urls: tuple[str, ...] = ()
    allow_partial_worker_set: bool = False
    admission_timeout_s: float | None = None
    cold_start_health_timeout_s: float = 120.0

    def __post_init__(self):
        if type(self.allow_partial_worker_set) is not bool:
            raise ValueError("allow_partial_worker_set must be boolean")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("Gateway base_url must be an HTTP(S) URL without query or fragment")
        for name in ("max_model_len", "prompt_length", "response_length"):
            _positive_int(getattr(self, name), name)
        for name in ("request_timeout_s", "control_timeout_s", "operation_timeout_s", "poll_interval_s", "cold_start_health_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, float | int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.admission_timeout_s is not None:
            value = self.admission_timeout_s
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError("admission_timeout_s must be finite and positive")


@dataclass(frozen=True)
class GatewayHTTPResponse:
    status: int
    body: Any
    headers: Mapping[str, str]


class GatewayHTTPTransport(Protocol):
    async def request(self, method: str, path: str, *, body: Any = None) -> GatewayHTTPResponse: ...

    async def close(self) -> None: ...


class _AiohttpTransport:
    def __init__(self, base_url: str, timeout_s: float, api_key: str | None):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.api_key = api_key
        self._session = None

    async def request(self, method: str, path: str, *, body: Any = None) -> GatewayHTTPResponse:
        import aiohttp

        if self._session is None:
            headers = {"accept": "application/json"}
            if self.api_key:
                headers["authorization"] = f"Bearer {self.api_key}"
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout_s),
            )
        async with self._session.request(
            method, f"{self.base_url}{path}", json=body, allow_redirects=False
        ) as response:
            raw = await response.read()
            try:
                payload = json.loads(raw) if raw else None
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if response.status != 200:
                    raise GatewayHTTPError(method, path, response.status, raw) from exc
                raise GatewayProtocolError("Gateway response is not valid JSON") from exc
            return GatewayHTTPResponse(response.status, payload, dict(response.headers))

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


@dataclass(frozen=True)
class WorkerRemovalObservation:
    worker_id: str
    url: str
    routing_removed: bool = True
    cache_cleanup_complete: None = None


@dataclass(frozen=True)
class WorkerRegistrationObservation:
    worker_id: str
    url: str
    engine_weight_version: str
    routing_ready: bool = True


@dataclass(frozen=True)
class WorkerRegistrationSettlement:
    worker_id: str
    url: str
    state: str  # "ready" or "failed"


@dataclass(frozen=True)
class RoutedWorkerResolution:
    worker_id: str
    engine_url: str
    weight_version: int


class SGLangGatewayClient:
    def __init__(
        self,
        config: GatewayClientConfig,
        *,
        inference_transport: GatewayHTTPTransport | None = None,
        control_transport: GatewayHTTPTransport | None = None,
    ):
        self.config = config
        self._inference = inference_transport or _AiohttpTransport(
            config.base_url, config.request_timeout_s, config.api_key
        )
        self._owns_control_transport = control_transport is None
        self._control = control_transport or _AiohttpTransport(
            config.base_url, config.control_timeout_s, config.api_key
        )
        self._cold_start_warmed_urls: set[str] = set()
        self.cold_start_observations: list[dict[str, Any]] = []

    async def close(self) -> None:
        await self._inference.close()
        if self._control is not self._inference:
            await self._control.close()

    async def observe_unregistered_engine_versions(self, worker_urls: Sequence[str]) -> dict[str, int]:
        """Read engine versions before registering them for request routing."""

        async def read_version(url: str) -> tuple[str, int]:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
                raise GatewayControlError("Invalid native engine URL")
            transport = _AiohttpTransport(url, self.config.control_timeout_s, self.config.api_key)
            try:
                response = await transport.request("GET", "/model_info")
                if response.status != 200:
                    raise GatewayHTTPError("GET", "/model_info", response.status, response.body)
                body = _mapping(response.body, "engine model_info")
                return url, _weight_step(body.get("weight_version"))
            finally:
                await transport.close()

        cold_urls = tuple(url for url in worker_urls if url not in self._cold_start_warmed_urls)
        if not cold_urls:
            return dict(await asyncio.gather(*(read_version(url) for url in worker_urls)))

        # Initial kernel compilation is outside the Gateway's registration
        # operation. Keep one total cold-start budget for all replicas, while
        # retaining the normal registration timeout and native health checks.
        loop = asyncio.get_running_loop()
        started = loop.time()
        logger.info("GATEWAY_COLD_START_HEALTH %s", json.dumps({"event": "started", "engine_count": len(cold_urls)}))
        try:
            async with asyncio.timeout(self.config.cold_start_health_timeout_s):
                before = dict(await asyncio.gather(*(read_version(url) for url in worker_urls)))
                tasks = [asyncio.create_task(self._warmup_unregistered_engine(url)) for url in cold_urls]
                try:
                    observations = await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                after = dict(await asyncio.gather(*(read_version(url) for url in worker_urls)))
                if before != after:
                    raise GatewayControlError("Engine weight version changed during cold-start health verification")
        except asyncio.TimeoutError as error:
            logger.info("GATEWAY_COLD_START_HEALTH %s", json.dumps({"event": "timeout", "elapsed_s": loop.time() - started}))
            raise GatewayOperationTimeout("Cold-start real generation did not complete within its independent budget") from error
        # Mark all replicas together only after real health and version checks.
        # Failure or cancellation leaves the cold-start requirement unsatisfied.
        self._cold_start_warmed_urls.update(cold_urls)
        receipt = {"event": "completed", "elapsed_s": loop.time() - started,
                   "weight_versions": after, "engines": observations}
        self.cold_start_observations.append(receipt)
        logger.info("GATEWAY_COLD_START_HEALTH %s", json.dumps(receipt, sort_keys=True))
        return after

    async def _warmup_unregistered_engine(self, url: str) -> dict[str, Any]:
        # SGLang's /health_generate executes a one-token generation and has its
        # own 20-second health timeout. Do not inherit the Gateway's short probe
        # timeout: a cold request may need a full health interval before retry.
        transport = _AiohttpTransport(url, min(30.0, self.config.cold_start_health_timeout_s), self.config.api_key)
        attempts = 0
        try:
            while True:
                attempts += 1
                try:
                    response = await transport.request("GET", "/health_generate")
                    if response.status == 200:
                        return {"url": url, "attempts": attempts, "health_generate_status": 200}
                    if response.status not in {408, 503, 504}:
                        raise GatewayHTTPError("GET", "/health_generate", response.status, response.body)
                except GatewayHTTPError as error:
                    if error.status not in {408, 503, 504}:
                        raise
                except (asyncio.TimeoutError, OSError):
                    pass
                await asyncio.sleep(min(1.0, self.config.poll_interval_s))
        finally:
            await transport.close()


    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: Sequence[int],
        sampling_params: Mapping[str, Any],
        image_data: Any = None,
        video_data: Any = None,
        audio_data: Any = None,
        mm_processor_kwargs: Mapping[str, Any] | None = None,
        mm_processor_output: Any = None,
        priority: int = 0,
        cache_salt: str | None = None,
        **kwargs: Any,
    ) -> TokenOutput:
        if not isinstance(request_id, str) or not request_id:
            raise GatewayProtocolError("request_id must be a nonempty string")
        if audio_data is not None or mm_processor_kwargs or priority != 0:
            raise GatewayProtocolError("Audio, processor overrides, and request priority are not enabled")
        if mm_processor_output is not None:
            video_data = mm_processor_output
        if kwargs:
            raise GatewayProtocolError(f"Unsupported native Gateway arguments: {sorted(kwargs)}")
        tokens = list(prompt_ids)
        for token in tokens:
            _positive_int(token, "prompt token", allow_zero=True)
        params = dict(sampling_params)
        if "prompt_logprobs" in params:
            raise GatewayProtocolError("Teacher prompt-logprob transport is not enabled in this integration")
        if "max_new_tokens" in params and "max_tokens" in params:
            raise GatewayProtocolError("Specify only one generation limit")
        available = self.config.max_model_len - len(tokens) - 1
        if available < 0:
            raise GatewayProtocolError("Prompt exceeds the configured model context")
        limit = params.pop("max_new_tokens", params.pop("max_tokens", None))
        if limit is None:
            limit = min(
                self.config.response_length, self.config.prompt_length + self.config.response_length - len(tokens)
            )
        else:
            _positive_int(limit, "generation limit", allow_zero=True)
        limit = max(0, min(limit, available))
        params["max_new_tokens"] = limit
        return_logprob = params.pop("logprobs", False)
        if type(return_logprob) is not bool:
            raise GatewayProtocolError("logprobs must be a boolean")
        request = {
            "rid": request_id,
            "input_ids": tokens,
            "sampling_params": params,
            "return_logprob": return_logprob,
            "stream": False,
        }
        if image_data is not None:
            request["image_data"] = image_data
        if video_data is not None:
            request["video_data"] = video_data
        if cache_salt is not None and not isinstance(cache_salt, str):
            raise GatewayProtocolError("cache_salt must be a string")
        active_step = await self._wait_for_active_weight_version()
        actual_salt = self._compose_cache_salt(active_step, cache_salt)
        request["cache_salt"] = actual_salt
        response = await self._inference.request("POST", "/generate", body=request)
        if response.status != 200:
            raise GatewayHTTPError("POST", "/generate", response.status, response.body)
        return self._decode_generation(
            response,
            return_logprob=return_logprob,
            limit=limit,
            request_id=request_id,
            observed_weight_version=active_step,
            observed_cache_salt=actual_salt,
        )

    @staticmethod
    def _compose_cache_salt(observed_weight_version: int, caller_cache_salt: str | None) -> str:
        if caller_cache_salt is None:
            return str(observed_weight_version)
        if caller_cache_salt == str(observed_weight_version):
            return caller_cache_salt
        parts = caller_cache_salt.split(":", 2)
        if len(parts) != 3 or parts[1] != str(observed_weight_version):
            raise GatewayProtocolError("cache_salt does not match the observed engine weight version")
        prefix, _, identity_digest = parts
        if not prefix or not identity_digest:
            raise GatewayProtocolError("cache_salt must include stable run and identity namespaces")
        return caller_cache_salt

    @staticmethod
    def build_policy_cache_salt(*, run_namespace: str, policy_version: int | str, policy_identity_salt: str) -> str:
        if not isinstance(run_namespace, str) or not run_namespace:
            raise ValueError("run_namespace must be a nonempty string")
        if type(policy_version) is int and policy_version >= 0:
            version = str(policy_version)
        elif isinstance(policy_version, str) and policy_version.isascii() and policy_version.isdecimal():
            version = str(int(policy_version))
        else:
            raise ValueError("policy_version must be a non-negative integer string")
        if not isinstance(policy_identity_salt, str) or not policy_identity_salt:
            raise ValueError("policy_identity_salt must be a nonempty string")
        namespace = hashlib.sha256(run_namespace.encode("utf-8")).hexdigest()
        return f"{namespace}:{version}:{policy_identity_salt}"

    @staticmethod
    def _decode_generation(
        response: GatewayHTTPResponse,
        *,
        return_logprob: bool,
        limit: int,
        request_id: str,
        observed_weight_version: int,
        observed_cache_salt: str,
    ) -> TokenOutput:
        body = _mapping(response.body, "generate response")
        output_ids = body.get("output_ids")
        if not isinstance(output_ids, list):
            raise GatewayProtocolError("generate response has no output_ids list")
        for token in output_ids:
            _positive_int(token, "output token", allow_zero=True)
        if len(output_ids) > limit:
            raise GatewayProtocolError("Engine generated more tokens than the requested limit")
        meta = _mapping(body.get("meta_info"), "meta_info")
        if meta.get("id") != request_id:
            raise GatewayProtocolError("Engine returned a different request identity")
        step = _weight_step(meta.get("weight_version"))
        spans = meta.get("weight_versions")
        if spans is not None:
            if not isinstance(spans, list) or not spans:
                raise GatewayProtocolError("weight_versions must contain a covering span")
            cursor = 0
            steps = set()
            for raw_span in spans:
                span = _mapping(raw_span, "weight version span")
                start = _positive_int(span.get("start"), "span start", allow_zero=True)
                end = _positive_int(span.get("end"), "span end", allow_zero=True)
                if start != cursor or end < start or end > len(output_ids) or (end == start and output_ids):
                    raise GatewayProtocolError("weight version spans do not cover output_ids exactly")
                cursor = end
                steps.add(_weight_step(span.get("version")))
            if cursor != len(output_ids) or step != _weight_step(spans[-1].get("version")):
                raise GatewayProtocolError("weight version metadata is inconsistent")
            if len(steps) != 1:
                raise GatewayMixedVersionError("One generation attempt contains multiple weight versions")

        log_probs = None
        if return_logprob:
            rows = meta.get("output_token_logprobs")
            if not isinstance(rows, list) or len(rows) != len(output_ids):
                raise GatewayProtocolError("output token and logprob counts differ")
            log_probs = []
            for token, row in zip(output_ids, rows, strict=True):
                if not isinstance(row, list | tuple) or len(row) < 2 or type(row[1]) is not int or row[1] != token:
                    raise GatewayProtocolError("output logprob token identity differs")
                value = row[0]
                if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                    raise GatewayProtocolError("output logprob is not finite")
                log_probs.append(float(value))

        finish = _mapping(meta.get("finish_reason"), "finish_reason").get("type")
        if finish not in {"stop", "length", "abort", "aborted", "completed"}:
            raise GatewayProtocolError("Engine did not report a finish reason")
        cached = meta.get("cached_tokens")
        if cached is not None:
            _positive_int(cached, "cached_tokens", allow_zero=True)
        preempted = meta.get("num_retractions")
        if preempted is not None:
            _positive_int(preempted, "num_retractions", allow_zero=True)
        return TokenOutput(
            token_ids=list(output_ids),
            log_probs=log_probs,
            stop_reason=finish,
            num_preempted=preempted,
            extra_fields={
                "global_steps": step,
                "min_global_steps": step,
                "max_global_steps": step,
                "engine_weight_version": str(meta["weight_version"]),
                "engine_weight_versions": spans,
                "native_meta_info": deepcopy(dict(meta)),
                "observed_weight_version": observed_weight_version,
                "cache_salt": observed_cache_salt,
                "salt_match": step == observed_weight_version,
                "num_cached_tokens": cached,
                "gateway_worker_id": next(
                    (value for key, value in response.headers.items() if key.lower() == "x-smg-routed-worker-id"),
                    None,
                ),
            },
        )

    async def _control_request(self, method: str, path: str, *, body: Any = None) -> GatewayHTTPResponse:
        response = await self._control.request(method, path, body=body)
        if response.status not in {200, 202, 404}:
            raise GatewayHTTPError(method, path, response.status, response.body)
        return response

    async def list_workers(self) -> list[Mapping[str, Any]]:
        response = await self._control_request("GET", "/workers")
        if response.status != 200:
            raise GatewayHTTPError("GET", "/workers", response.status, response.body)
        body = _mapping(response.body, "worker list")
        workers = body.get("workers")
        if not isinstance(workers, list):
            raise GatewayProtocolError("Worker list is missing")
        return [_mapping(worker, "worker") for worker in workers]

    @staticmethod
    def _worker_url(worker: Mapping[str, Any]) -> str:
        url = worker.get("url")
        if not isinstance(url, str) or not url:
            raise GatewayProtocolError("Native worker has no URL")
        return url

    @staticmethod
    def _worker_job_pending(worker: Mapping[str, Any]) -> bool:
        job = worker.get("job_status")
        if job is None:
            return False
        job = _mapping(job, "worker job")
        if job.get("worker_url") != SGLangGatewayClient._worker_url(worker):
            raise GatewayControlError("Native worker job addresses a different URL")
        status = job.get("status")
        if status == "failed":
            raise GatewayControlError("Native worker control job failed")
        if status not in {"pending", "processing"}:
            raise GatewayProtocolError("Native worker job status is unknown")
        return True

    @staticmethod
    def _worker_failed(worker: Mapping[str, Any]) -> bool:
        job = worker.get("job_status")
        if job is not None:
            job = _mapping(job, "worker job")
            if job.get("worker_url") != SGLangGatewayClient._worker_url(worker):
                raise GatewayControlError("Native worker job addresses a different URL")
            if job.get("status") == "failed" and job.get("job_type") == "AddWorker":
                return True
        return False

    @staticmethod
    def _worker_is_ready(worker: Mapping[str, Any]) -> bool:
        if type(worker.get("is_healthy")) is not bool:
            raise GatewayProtocolError("Native worker readiness fields are incomplete")
        pending = SGLangGatewayClient._worker_job_pending(worker)
        status = worker.get("status")
        if status == "failed":
            raise GatewayControlError("Native engine readiness failed")
        if status not in {"ready", "pending", "not_ready", "draining"}:
            raise GatewayProtocolError("Native worker status is unknown")
        return status == "ready" and worker["is_healthy"] is True and not pending

    async def _wait_for_active_weight_version(self) -> int:
        timeout = self.config.admission_timeout_s or self.config.control_timeout_s
        deadline = asyncio.timeout(timeout)
        last_not_ready = None
        try:
            async with deadline:
                while True:
                    try:
                        return await self.get_active_weight_version(self.config.expected_worker_urls)
                    except GatewayNotReady as error:
                        last_not_ready = error
                        await asyncio.sleep(self.config.poll_interval_s)
        except asyncio.TimeoutError as exc:
            if deadline.expired():
                raise GatewayOperationTimeout(
                    "Expected engines did not become ready before the admission deadline"
                    + (f"; last_readiness={last_not_ready}" if last_not_ready is not None else "")
                ) from exc
            raise

    async def _read_model_info_bounded(
        self, worker_id: str, engine_url: str, *, expected_weight_version: int | None,
        membership: tuple[tuple[str, str], ...],
    ) -> Mapping[str, Any]:
        """Allow one identity-checked read after a proxy reachability failure.

        Both reads and the registry recheck share the original control deadline.
        No generation/control POST is retried and no completed version is cached.
        """
        first_error = None
        attempts = 0
        deadline = asyncio.timeout(self.config.control_timeout_s)
        try:
            async with deadline:
                while True:
                    attempts += 1
                    try:
                        model = await self.call_engine(
                            worker_id, "model_info", method="GET", require_success=False,
                        )
                    except GatewayHTTPError as error:
                        path = f"/v1/rl/workers/{quote(worker_id, safe='')}/engine/model_info"
                        eligible = (
                            error.method == "GET" and error.path == path and error.status == 502
                            and error.upstream_error == "upstream_unreachable"
                            and error.upstream_worker_id == worker_id and error.upstream_url == engine_url
                        )
                        if first_error is not None or not eligible:
                            if first_error is not None:
                                raise error from first_error
                            raise
                        first_error = error
                        # A proxy failure does not authorize a new registry identity,
                        # a draining worker, or a pending publication/registration.
                        try:
                            workers = await self.list_workers()
                            observed = tuple(sorted((self._worker_url(w), w.get("id")) for w in workers))
                            if observed != membership:
                                raise GatewayControlError("Native routing membership changed before model_info recheck")
                            detail = await self._control_request("GET", f"/workers/{quote(worker_id, safe='')}")
                            if detail.status != 200:
                                raise GatewayControlError("Native worker lost its detail identity before model_info recheck")
                            worker = _mapping(detail.body, "worker")
                            if worker.get("id") != worker_id or self._worker_url(worker) != engine_url:
                                raise GatewayControlError("Native worker identity changed before model_info recheck")
                            if not self._worker_is_ready(worker):
                                raise GatewayNotReady("Native worker is not ready for model_info recheck")
                        except GatewayError as recheck_error:
                            raise recheck_error from first_error
                        logger.warning(
                            "GATEWAY_MODEL_INFO_RECHECK worker_id=%s engine_url=%s first_cause=%s",
                            worker_id, engine_url, first_error,
                        )
                        await asyncio.sleep(self.config.poll_interval_s)
                        continue
                    actual = _weight_step(model.get("weight_version"))
                    if expected_weight_version is not None and actual != expected_weight_version:
                        error = GatewayControlError(
                            f"Engine weight version changed from {expected_weight_version} to {actual}"
                        )
                        if first_error is not None:
                            raise error from first_error
                        raise error
                    if first_error is not None:
                        logger.info(
                            "GATEWAY_MODEL_INFO_RECHECK_COMPLETED worker_id=%s engine_url=%s attempts=%s weight_version=%s",
                            worker_id, engine_url, attempts, actual,
                        )
                    return model
        except asyncio.TimeoutError as error:
            if deadline.expired():
                failure = GatewayOperationTimeout("model_info recheck exceeded the original control deadline")
                failure.model_info_attempts = attempts
                raise failure from (first_error or error)
            raise

    async def _read_model_info_inflight(
        self, worker_id: str, engine_url: str, *, expected_weight_version: int | None,
        membership: tuple[tuple[str, str], ...],
    ) -> Mapping[str, Any]:
        """Merge identical in-flight reads across clients on this event loop.

        The dedicated reader owns its transport, so a canceled stream can close
        its client without canceling a different stream's identity observation.
        Worker IDs and the complete registry snapshot isolate publication and
        membership epochs. Completed reads (including failures) are never reused.
        """
        # Preserve ownership and behavior for caller-injected transports.
        if not self._owns_control_transport:
            return await self._read_model_info_bounded(
                worker_id, engine_url, expected_weight_version=expected_weight_version, membership=membership,
            )
        loop = asyncio.get_running_loop()
        flights = _MODEL_INFO_INFLIGHT.setdefault(loop, {})
        key = (
            self.config.base_url.rstrip("/"), self.config.api_key,
            self.config.control_timeout_s, tuple(sorted(self.config.expected_worker_urls)),
            self.config.allow_partial_worker_set, membership,
            worker_id, engine_url, expected_weight_version,
        )
        task = flights.get(key)
        if task is None or task.done():
            async def observe():
                reader = SGLangGatewayClient(self.config)
                try:
                    return await reader._read_model_info_bounded(
                        worker_id, engine_url, expected_weight_version=expected_weight_version, membership=membership,
                    )
                finally:
                    await reader.close()

            task = asyncio.create_task(observe())
            flights[key] = task

            def forget(completed):
                if flights.get(key) is completed:
                    del flights[key]
                # An abandoned read still owns its bounded request and transport.
                # Retrieve its error even if every individual waiter was canceled.
                if not completed.cancelled():
                    completed.exception()

            task.add_done_callback(forget)
        return deepcopy(await asyncio.shield(task))

    async def get_active_weight_version(
        self, expected_worker_urls: Sequence[str], *, expected_weight_version: int | None = None,
    ) -> int:
        if expected_weight_version is not None:
            _positive_int(expected_weight_version, "expected weight version", allow_zero=True)
        expected = set(expected_worker_urls)
        if (
            not expected
            or len(expected) != len(expected_worker_urls)
            or any(not isinstance(url, str) or not url for url in expected)
        ):
            raise GatewayControlError("Generation requires the exact nonempty expected worker URL set")

        def membership_snapshot(workers):
            by_url = {}
            seen_ids = set()
            for worker in workers:
                url = self._worker_url(worker)
                worker_id = worker.get("id")
                if not isinstance(worker_id, str) or not worker_id or url in by_url or worker_id in seen_ids:
                    raise GatewayControlError("Native workers contain a duplicate or invalid identity")
                by_url[url] = worker_id
                seen_ids.add(worker_id)
            if set(by_url) - expected:
                raise GatewayControlError("Native routing membership contains an unexpected worker")
            if not by_url or (expected - set(by_url) and not self.config.allow_partial_worker_set):
                raise GatewayNotReady(
                    "Expected native workers are not yet ready for dispatch "
                    f"(branch=missing_expected_membership expected_count={len(expected)} "
                    f"listed_count={len(by_url)} missing_count={len(expected - set(by_url))})"
                )
            return by_url

        async def check_detail(url, worker_id):
            path = f"/workers/{quote(worker_id, safe='')}"
            response = await self._control_request("GET", path)
            if response.status != 200:
                raise GatewayControlError("Listed native worker lost its detail identity")
            worker = _mapping(response.body, "worker")
            if worker.get("id") != worker_id or self._worker_url(worker) != url:
                raise GatewayControlError("Native worker detail returned a different identity")
            return self._worker_is_ready(worker)

        before = membership_snapshot(await self.list_workers())
        ready = await asyncio.gather(*[check_detail(url, worker_id) for url, worker_id in before.items()])
        if not all(ready):
            raise GatewayNotReady(
                "Expected native workers are not yet ready for dispatch "
                f"(branch=worker_detail_not_ready expected_count={len(expected)} "
                f"listed_count={len(before)} unready_count={sum(not value for value in ready)})"
            )
        models = await asyncio.gather(
            *[
                self._read_model_info_inflight(
                    worker_id, url, expected_weight_version=expected_weight_version,
                    membership=tuple(sorted(before.items())),
                )
                for url, worker_id in before.items()
            ]
        )
        steps = {_weight_step(model.get("weight_version")) for model in models}
        if len(steps) != 1:
            raise GatewayControlError("Active engines do not hold one consistent weight version")
        if expected_weight_version is not None and steps != {expected_weight_version}:
            raise GatewayControlError("Active engines do not hold the expected weight version")
        after = membership_snapshot(await self.list_workers())
        if before != after:
            raise GatewayControlError("Native routing membership changed while reading weight versions")
        return steps.pop()

    async def resolve_routed_worker(self, worker_id: str, *, expected_weight_version: int) -> RoutedWorkerResolution:
        """Resolve a response-header worker ID to its verified native engine URL.

        The Gateway response header can carry a worker ID or URL. This method
        resolves either form through the expected registry before any control RPC. It verifies registry membership, detail
        identity, readiness, the engine's actual ``model_info`` version, and a
        stable before/after registry snapshot before returning the engine URL.
        """
        if not isinstance(worker_id, str) or not worker_id:
            raise GatewayProtocolError("Routed generation response has no worker ID")
        expected_step = _positive_int(expected_weight_version, "expected weight version", allow_zero=True)
        expected = tuple(self.config.expected_worker_urls)
        expected_urls = set(expected)
        if (not expected_urls or len(expected_urls) != len(expected)
                or any(not isinstance(url, str) or not url for url in expected)):
            raise GatewayControlError("Routed generation requires the exact expected worker URL set")

        def membership_snapshot(workers: Sequence[Mapping[str, Any]]) -> dict[str, str]:
            by_id: dict[str, str] = {}
            seen_urls: set[str] = set()
            for worker in workers:
                url = self._worker_url(worker)
                current_id = worker.get("id")
                if (not isinstance(current_id, str) or not current_id
                        or current_id in by_id or url in seen_urls):
                    raise GatewayControlError("Native workers contain a duplicate or invalid identity")
                by_id[current_id] = url
                seen_urls.add(url)
            present = set(by_id.values())
            if present - expected_urls:
                raise GatewayControlError("Native routing membership differs from the expected worker set")
            if not present or (expected_urls - present and not self.config.allow_partial_worker_set):
                raise GatewayControlError("Expected routed workers are not yet available")
            return by_id

        before = membership_snapshot(await self.list_workers())
        # Some native Gateway response paths identify the selected worker by
        # its URL. Resolve either representation through the verified registry;
        # never issue a control request to an address supplied by the header.
        candidates = [current_id for current_id, url in before.items()
                      if worker_id == current_id or worker_id == url]
        if len(candidates) != 1:
            raise GatewayControlError(f"Routed worker {worker_id!r} is unknown or ambiguous in the expected native registry")
        worker_id = candidates[0]
        engine_url = before[worker_id]

        path = f"/workers/{quote(worker_id, safe='')}"
        response = await self._control_request("GET", path)
        if response.status != 200:
            raise GatewayControlError("Routed worker lost its native detail identity")
        worker = _mapping(response.body, "worker")
        if worker.get("id") != worker_id or self._worker_url(worker) != engine_url:
            raise GatewayControlError("Routed worker detail returned a different identity")
        if not self._worker_is_ready(worker):
            raise GatewayControlError("Routed worker is not ready for native generation")

        model = await self._read_model_info_inflight(
            worker_id, engine_url, expected_weight_version=expected_step,
            membership=tuple(sorted((url, current_id) for current_id, url in before.items())),
        )
        model = _mapping(model, "routed engine model_info")
        actual_step = _weight_step(model.get("weight_version"))
        if actual_step != expected_step:
            raise GatewayControlError(
                f"Routed engine weight version changed from {expected_step} to {actual_step}"
            )

        after = membership_snapshot(await self.list_workers())
        if after.get(worker_id) != engine_url or before != after:
            raise GatewayControlError("Native routing membership changed while resolving the routed worker")
        return RoutedWorkerResolution(worker_id=worker_id, engine_url=engine_url, weight_version=actual_step)

    async def remove_worker(self, worker_id: str, *, url: str) -> WorkerRemovalObservation:
        path = f"/workers/{quote(worker_id, safe='')}"
        response = await self._control_request("DELETE", path)
        accepted = _mapping(response.body, "worker deletion receipt")
        if response.status != 202 or accepted.get("worker_id") != worker_id:
            raise GatewayControlError("Native Gateway did not accept the requested deletion")

        async def wait_removed():
            while True:
                current = await self._control_request("GET", path)
                if current.status == 404:
                    workers = await self.list_workers()
                    if any(worker.get("id") == worker_id or self._worker_url(worker) == url for worker in workers):
                        raise GatewayControlError("Worker deletion observations disagree")
                    return WorkerRemovalObservation(worker_id, url)
                worker = _mapping(current.body, "worker")
                if worker.get("id") != worker_id or self._worker_url(worker) != url:
                    raise GatewayControlError("Worker deletion returned a different identity")
                self._worker_job_pending(worker)
                await asyncio.sleep(self.config.poll_interval_s)

        try:
            return await asyncio.wait_for(wait_removed(), timeout=self.config.operation_timeout_s)
        except asyncio.TimeoutError as exc:
            raise GatewayOperationTimeout("Worker still routable or removal unconfirmed") from exc

    async def register_worker(
        self, spec: Mapping[str, Any], *, expected_weight_version: int
    ) -> WorkerRegistrationObservation:
        _positive_int(expected_weight_version, "expected_weight_version", allow_zero=True)
        spec = dict(spec)
        url = spec.get("url")
        if not isinstance(url, str) or not url:
            raise GatewayControlError("Registration requires the engine URL")
        try:
            response = await self._control_request("POST", "/workers", body=spec)
        except GatewayHTTPError as exc:
            if 400 <= exc.status < 500 and exc.status != 408:
                raise GatewayRegistrationRejected("Native Gateway rejected the registration request") from exc
            raise GatewayRegistrationAcceptanceUnknown(
                "Native registration acceptance is unknown after HTTP failure"
            ) from exc
        except Exception as exc:
            raise GatewayRegistrationAcceptanceUnknown(
                "Native registration acceptance is unknown after a transport failure"
            ) from exc
        try:
            accepted = _mapping(response.body, "worker registration receipt")
        except GatewayProtocolError as exc:
            raise GatewayRegistrationAcceptanceUnknown("Native registration receipt is not verifiable") from exc
        worker_id = accepted.get("worker_id")
        if response.status != 202 or not isinstance(worker_id, str) or not worker_id or accepted.get("url") != url:
            raise GatewayRegistrationAcceptanceUnknown("Native registration receipt does not verify acceptance")
        path = f"/workers/{quote(worker_id, safe='')}"

        terminal_state = None

        async def wait_ready():
            nonlocal terminal_state
            while True:
                current = await self._control_request("GET", path)
                if current.status == 404:
                    # The async job view can disappear before registration reaches
                    # the worker registry. Poll the accepted ID within the existing
                    # deadline; never resubmit or adopt another worker at this URL.
                    workers = await self.list_workers()
                    for worker in workers:
                        same_id = worker.get("id") == worker_id
                        same_url = self._worker_url(worker) == url
                        if same_id != same_url:
                            raise GatewayControlError("Registration observations disagree on the accepted identity")
                    await asyncio.sleep(self.config.poll_interval_s)
                    continue
                if current.status != 200:
                    raise GatewayControlError("Accepted registration lost its native worker identity")
                worker = _mapping(current.body, "worker")
                if worker.get("id") != worker_id or self._worker_url(worker) != url:
                    raise GatewayControlError("Registration returned a different worker identity")
                if self._worker_failed(worker):
                    terminal_state = "failed"
                    raise GatewayControlError("Native worker control job failed")
                if self._worker_is_ready(worker):
                    model = await self.call_engine(worker_id, "model_info", method="GET", require_success=False)
                    if _weight_step(model.get("weight_version")) != expected_weight_version:
                        raise GatewayControlError("Registered engine has a different weight version")
                    return WorkerRegistrationObservation(worker_id, url, str(model["weight_version"]))
                await asyncio.sleep(self.config.poll_interval_s)

        try:
            return await asyncio.wait_for(wait_ready(), timeout=self.config.operation_timeout_s)
        except asyncio.TimeoutError as exc:
            raise GatewayRegistrationOperationTimeout(
                "Registration did not become ready with verified weights", worker_id, url
            ) from exc
        except Exception as exc:
            raise GatewayRegistrationOperationError(str(exc), worker_id, url, terminal_state) from exc

    async def settle_registration(self, worker_id: str, *, url: str) -> WorkerRegistrationSettlement:
        """Observe the terminal state of one already accepted registration.

        A 404 or an empty registry is not proof of completion. The method returns
        only after the exact accepted identity is ready, or its AddWorker job has
        explicitly failed. It never submits another POST.
        """
        if not isinstance(worker_id, str) or not worker_id or not isinstance(url, str) or not url:
            raise GatewayControlError("Registration settlement requires the accepted worker identity")
        path = f"/workers/{quote(worker_id, safe='')}"

        async def wait_terminal() -> WorkerRegistrationSettlement:
            while True:
                current = await self._control_request("GET", path)
                if current.status == 404:
                    workers = await self.list_workers()
                    for worker in workers:
                        same_id = worker.get("id") == worker_id
                        same_url = self._worker_url(worker) == url
                        if same_id != same_url:
                            raise GatewayControlError("Registration settlement observations disagree")
                    await asyncio.sleep(self.config.poll_interval_s)
                    continue
                if current.status != 200:
                    raise GatewayControlError("Accepted registration lost its native worker identity")
                worker = _mapping(current.body, "worker")
                if worker.get("id") != worker_id or self._worker_url(worker) != url:
                    raise GatewayControlError("Registration settlement returned a different worker identity")
                if self._worker_failed(worker):
                    return WorkerRegistrationSettlement(worker_id, url, "failed")
                if self._worker_is_ready(worker):
                    return WorkerRegistrationSettlement(worker_id, url, "ready")
                await asyncio.sleep(self.config.poll_interval_s)

        try:
            return await asyncio.wait_for(wait_terminal(), timeout=self.config.operation_timeout_s)
        except asyncio.TimeoutError as exc:
            raise GatewayRegistrationOperationTimeout(
                "Accepted registration settlement remains unknown", worker_id, url
            ) from exc
        except GatewayControlError as exc:
            raise GatewayRegistrationOperationError(str(exc), worker_id, url) from exc

    async def observe_unknown_registration(self, url: str) -> None:
        """Boundedly observe a URL after registration acceptance became unknown.

        This method never submits another POST and never returns a terminal
        conclusion. If a worker appears, its identity cannot be tied to the lost
        request; if none appears, the request may still have been accepted.
        """
        if not isinstance(url, str) or not url:
            raise GatewayControlError("Unknown registration observation requires an engine URL")

        async def observe() -> None:
            while True:
                workers = await self.list_workers()
                if any(self._worker_url(worker) == url for worker in workers):
                    raise GatewayControlError("A worker appeared at a URL with unknown registration acceptance")
                await asyncio.sleep(self.config.poll_interval_s)

        try:
            await asyncio.wait_for(observe(), timeout=self.config.operation_timeout_s)
        except asyncio.TimeoutError as exc:
            raise GatewayRegistrationAcceptanceUnknown("Native registration acceptance remains unknown") from exc

    async def call_engine(
        self,
        worker_id: str,
        operation: str,
        *,
        method: str = "POST",
        body: Any = None,
        require_success: bool = True,
    ) -> Mapping[str, Any]:
        if not operation or not operation.replace("_", "").isalnum() or method not in {"GET", "POST"}:
            raise GatewayProtocolError("Invalid native engine operation")
        path = f"/v1/rl/workers/{quote(worker_id, safe='')}/engine/{operation}"
        response = await self._control_request(method, path, body={} if body is None and method == "POST" else body)
        wrapped = _mapping(response.body, "engine control response")
        if response.status != 200 or wrapped.get("worker_id") != worker_id or wrapped.get("status") != 200:
            raise GatewayControlError("Engine operation did not complete successfully")
        result = _mapping(wrapped.get("body"), "engine result")
        if result.get("success") is False or (require_success and result.get("success") is not True):
            raise GatewayControlError("Engine did not confirm operation success")
        return result

    async def fanout_engine(
        self,
        operation: str,
        *,
        expected_worker_ids: Sequence[str],
        selector: str = "engine=sglang",
        body: Any = None,
    ) -> Mapping[str, Any]:
        expected = set(expected_worker_ids)
        if (
            not expected
            or len(expected) != len(expected_worker_ids)
            or any(not isinstance(worker_id, str) or not worker_id for worker_id in expected)
        ):
            raise GatewayControlError("Expected fleet must be nonempty and contain unique worker IDs")
        if not operation or not operation.replace("_", "").isalnum():
            raise GatewayProtocolError("Invalid native engine operation")
        path = f"/v1/rl/engine/{operation}?{urlencode({'selector': selector})}"
        response = await self._control.request("POST", path, body={} if body is None else body)
        if response.status != 200:
            raise GatewayHTTPError("POST", path, response.status, response.body)
        result = _mapping(response.body, "fanout response")
        results = _mapping(result.get("results"), "fanout results")
        _positive_int(result.get("total"), "fanout total")
        _positive_int(result.get("succeeded"), "fanout succeeded", allow_zero=True)
        if (
            result.get("failed") != []
            or result.get("total") != len(expected)
            or result.get("succeeded") != len(expected)
            or set(results) != expected
        ):
            raise GatewayControlError("Fanout did not cover and complete the expected fleet")
        for raw in results.values():
            worker = _mapping(raw, "fanout worker result")
            if worker.get("status") != 200 or _mapping(worker.get("body"), "engine result").get("success") is not True:
                raise GatewayControlError("One engine did not confirm fanout success")
        return result
