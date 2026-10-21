# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""One Token/KV source SSE generation attempt through the native Gateway.

The stream is addressed only to the Gateway. After the Gateway returns its
routed worker ID, the public resolver verifies the worker's real engine URL and
weight version. This adapter never falls back to a static owner URL or calls an
engine directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from verl.workers.rollout.sglang_gateway_client import GatewayProtocolError


@dataclass(frozen=True)
class RoutedTokenKVStream:
    response: Any
    worker_id: str
    engine_url: str
    weight_version: int


@asynccontextmanager
async def open_routed_token_kv_stream(gateway, payload, *, expected_weight_version: int):
    """Open one Gateway-routed Token/KV SSE request and resolve its engine.

    ``payload`` is copied. Its ``rid`` is passed unchanged, ``stream`` is forced
    to true, and the existing Gateway cache-salt compatibility rule is reused.
    The HTTP response and its private ``aiohttp`` session are always closed on
    normal exit, failure, or cancellation.
    """
    if not isinstance(payload, Mapping):
        raise GatewayProtocolError("Token/KV generation payload must be an object")
    expected_step = expected_weight_version
    if type(expected_step) is not int or expected_step < 0:
        raise GatewayProtocolError("expected weight version must be a nonnegative integer")

    request_id = payload.get("rid")
    if not isinstance(request_id, str) or not request_id:
        raise GatewayProtocolError("Token/KV generation requires a request ID")

    active_step = await gateway._wait_for_active_weight_version()
    if active_step != expected_step:
        raise GatewayProtocolError(
            f"Gateway active weight version {active_step} differs from expected {expected_step}"
        )

    request = dict(payload)
    request["stream"] = True
    request["cache_salt"] = gateway._compose_cache_salt(active_step, request.get("cache_salt"))

    import aiohttp

    headers = {"accept": "text/event-stream"}
    if gateway.config.api_key:
        headers["authorization"] = f"Bearer {gateway.config.api_key}"
    timeout = aiohttp.ClientTimeout(total=gateway.config.request_timeout_s)
    session = aiohttp.ClientSession(timeout=timeout, headers=headers)
    response = None
    try:
        response = await session.post(
            gateway.config.base_url.rstrip("/") + "/generate",
            json=request,
            allow_redirects=False,
        )
        if response.status != 200:
            raise GatewayProtocolError(f"Gateway Token/KV generation returned HTTP {response.status}")
        worker_id = next(
            (value for key, value in response.headers.items()
             if key.lower() == "x-smg-routed-worker-id"),
            None,
        )
        if not isinstance(worker_id, str) or not worker_id:
            raise GatewayProtocolError("Gateway Token/KV generation response has no routed worker ID")
        resolved = await gateway.resolve_routed_worker(
            worker_id, expected_weight_version=expected_step
        )
        yield RoutedTokenKVStream(
            response=response,
            worker_id=resolved.worker_id,
            engine_url=resolved.engine_url,
            weight_version=resolved.weight_version,
        )
    finally:
        if response is not None:
            response.close()
        await session.close()
