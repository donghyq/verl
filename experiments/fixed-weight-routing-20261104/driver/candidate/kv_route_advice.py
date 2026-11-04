# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""V1 shadow advice only; native Gateway still owns every real selection.

Directory coverage and calibration are producer declarations, not independent
GPU measurements. Nothing here authorizes state installation or training data.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

from .conductor_hint_client import ConductorHTTPClient
from .native_cache_directory_client import NativeCacheDirectoryClient, native_query_digest
from .identity import _decode_policy_identity
from .policy_publication import PublishedPolicyRecord


class AdvicePublicationInvalid(ValueError):
    """No settled publication matching the request authorizes this estimate."""


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid nonnegative cost")
    return value


def _prefix(row, length):
    values = tuple(row.get(key, 0) for key in ("device", "cpu_local", "cpu_share", "disk"))
    if any(type(value) is not int or not 0 <= value <= length for value in values) or values != tuple(sorted(values)):
        raise ValueError("invalid cumulative local prefix")
    return values[:2]


class V1KVRouteAdvisor:
    """Bound one directory query per worker and compare advice to actual routing."""
    def __init__(self, settings, *, client=None, clock=time.time, publication_client=None, run_id=None):
        if settings.get("mode") != "shadow":
            raise ValueError("V1 KV routing currently supports shadow only; active requires a stable Gateway target API")
        self.timeout = _number(settings.get("timeout_ms", 50)) / 1000
        self.ttl = _number(settings.get("max_view_age_s", 5))
        if not 0 < self.timeout <= .1 or not 0 < self.ttl <= 10:
            raise ValueError("invalid shadow query/view bounds")
        self.path = Path(settings["view_path"])
        if not self.path.is_absolute():
            raise ValueError("KV routing view_path must be absolute")
        self.context = deepcopy(settings["cache_context"])
        if (set(self.context) != {"model", "tenant_id", "lora_name", "block_size", "tp_size", "dp_size", "hybrid_model"}
                or not isinstance(self.context["model"], str) or not self.context["model"]
                or type(self.context["block_size"]) is not int or self.context["block_size"] <= 0
                or type(self.context["tp_size"]) is not int or self.context["tp_size"] < 1
                or self.context["dp_size"] != 1 or type(self.context["dp_size"]) is not int
                or type(self.context["hybrid_model"]) is not bool):
            raise ValueError("shadow cache context requires explicit model/layout and DP=1")
        self.directory_source = settings.get("directory_source", "conductor_declared")
        if self.directory_source not in {"conductor_declared", "native_prefix_probe"}:
            raise ValueError("unknown directory source")
        if self.directory_source == "native_prefix_probe":
            if not self.context["hybrid_model"] or self.context["lora_name"]:
                raise ValueError("native prefix directory v1 requires hybrid plain-text requests without LoRA")
            self.client = client or NativeCacheDirectoryClient(
                settings["native_instances"], tp_size=self.context["tp_size"], timeout_ms=self.timeout*1000)
        else:
            self.client = client or ConductorHTTPClient(settings["endpoint"], self.timeout * 1000)
        self.clock = clock
        self.publication_client = publication_client
        self.run_id = run_id
        self.pending = None
        self.tickets = set()
        self.records = deque(maxlen=128)
        self._closed = False

    async def _assert_current_publication(self, identity, deadline):
        if self.publication_client is None:
            raise AdvicePublicationInvalid("publication authority is not bound")
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise asyncio.TimeoutError
        try:
            # The original coordinator refuses this read from publication
            # prepare until actual engine observations have been committed.
            raw = await asyncio.wait_for(
                self.publication_client.get_active_published_policy(), remaining)
            record = PublishedPolicyRecord.decode(raw)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            raise
        except Exception as error:
            raise AdvicePublicationInvalid("publication is unavailable or unsettled") from error
        if vars(record.identity) != identity or (self.run_id is not None and record.run_id != self.run_id):
            raise AdvicePublicationInvalid("request identity/run differs from the current publication")
        return record

    def _assert_native_publication_membership(self, record, urls):
        if self.directory_source == "native_prefix_probe":
            observed = {row.engine_url.rstrip("/") for row in record.engine_observations}
            if not {url.rstrip("/") for url in urls} <= observed:
                raise AdvicePublicationInvalid("native candidate is absent from the current engine publication")

    def _view(self, identity, cache_salt, allowed_urls):
        with self.path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("routing view exceeds 1 MiB")
        view = json.loads(raw)
        age = self.clock() - _number(view["observed_at_unix"])
        if (type(view.get("schema_version")) is not int or view["schema_version"] != 1
                or not isinstance(view.get("generation"), str) or not view["generation"]
                or not -1 <= age <= self.ttl or view.get("policy_identity") != identity
                or json.dumps(view.get("cache_context"), sort_keys=True) != json.dumps(
                    {**self.context, "cache_salt": cache_salt}, sort_keys=True)
                or not isinstance(view.get("instances"), dict) or not 0 < len(view["instances"]) <= 128):
            raise ValueError("stale or incompatible routing view")
        calibration = view["calibration"]
        proof = calibration.get("source_receipt_sha256", "")
        if len(proof) != 64 or any(char not in "0123456789abcdef" for char in proof):
            raise ValueError("unbound cost calibration")
        for key in ("prefill_s_per_token", "host_read_s_per_token", "host_read_fixed_s"):
            _number(calibration[key])
        low, high = calibration["prompt_tokens_min"], calibration["prompt_tokens_max"]
        if type(low) is not int or type(high) is not int or not 1 <= low <= high <= 32768:
            raise ValueError("invalid calibration range")
        # Native membership is still checked by the Gateway. The directory is
        # never allowed to add an endpoint outside the original manager pool.
        view["instances"] = {key: row for key, row in view["instances"].items()
                             if row.get("available") is True and row.get("engine_url") in allowed_urls}
        return view, hashlib.sha256(raw).hexdigest()

    def start(self, payload, *, policy_identity, cache_salt, allowed_urls, request_context=None):
        if self._closed or len(self.tickets) >= 128:
            return None
        # Take the copy synchronously: later caller mutation cannot rebind the
        # query to different tokens or weights while it is in flight.
        identity, tokens = deepcopy(policy_identity), deepcopy(payload.get("input_ids"))
        task = asyncio.create_task(self._advise(identity, tokens, cache_salt, tuple(allowed_urls),
                                               payload.get("rid"), deepcopy(request_context), native_eligible=(
                                                   not any(payload.get(key) is not None for key in (
                                                       "input_embeds", "positional_embed_overrides", "image_data",
                                                       "lora_path", "extra_key", "session_params"))
                                                   and (not payload.get("return_logprob")
                                                        or payload.get("logprob_start_len", -1)==-1))))
        self.tickets.add(task)
        task.add_done_callback(self.tickets.discard)
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return task

    async def _advise(self, identity, tokens, salt, allowed_urls, request_id, request_context, *, native_eligible=True):
        started = time.perf_counter()
        deadline = started + self.timeout
        result = {"status": "fallback", "routing_changed": False, "reason": "unavailable", "candidates": {}}
        result.update(request_id=request_id, request_context=request_context)
        try:
            _decode_policy_identity(identity)
            if self.directory_source == "native_prefix_probe" and not native_eligible:
                result["reason"] = "native_probe_unsupported_request_shape"
                return result
            if (not isinstance(tokens, list) or not tokens or len(tokens) > 32768
                    or any(type(token) is not int or token < 0 for token in tokens)):
                raise ValueError("plain-text token history required")
            result.update(policy_identity=identity, cache_salt=salt, prompt_tokens=len(tokens),
                          token_ids_sha256=hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest())
            publication = await self._assert_current_publication(identity, deadline)
            view, digest = self._view(identity, salt, allowed_urls)
            self._assert_native_publication_membership(publication,
                [row["engine_url"] for row in view["instances"].values()])
            result.update(view_generation=view["generation"], view_sha256=digest)
            if self.pending is not None and not self.pending.done():
                result["reason"] = "query_busy"
                return result
            context = {key: self.context[key] for key in ("model", "tenant_id", "lora_name", "block_size")}
            query = {**context, "cache_salt": salt, "token_ids": tokens}
            if self.directory_source == "native_prefix_probe":
                query.update(expected_weight_version=str(identity["policy_version"]),
                             native_allowed_instances={key:row["engine_url"] for key,row in view["instances"].items()},
                             native_expected_rank_generations={key:row["native_rank_generations"]
                                for key,row in view["instances"].items() if "native_rank_generations" in row})
            self.pending = asyncio.create_task(asyncio.to_thread(self.client.query, query))
            self.pending.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise asyncio.TimeoutError
            hits = await asyncio.wait_for(asyncio.shield(self.pending), remaining)
            result["query_seconds"] = time.perf_counter() - started
            result["directory_source"] = self.directory_source
            length, calibration = len(tokens), view["calibration"]
            if not calibration["prompt_tokens_min"] <= length <= calibration["prompt_tokens_max"]:
                raise ValueError("prompt outside calibrated range")
            for instance_id, row in view["instances"].items():
                try:
                    if (not isinstance(instance_id, str) or not instance_id
                            or not isinstance(row.get("engine_instance_id"), str) or not row["engine_instance_id"]):
                        raise ValueError("missing native instance identity")
                    hit = hits.get(instance_id, {})
                    device, host = _prefix(hit.get("rank_matches", {}).get("0", {}), length)
                    if self.directory_source == "native_prefix_probe":
                        proof = hit["native_query"]
                        ranks = {str(rank) for rank in range(self.context["tp_size"])}
                        age = self.clock() - _number(proof["observed_at_unix"])
                        if (proof["query_digest"] != native_query_digest(tokens, salt)
                                or proof["engine_instance_id"] != row["engine_instance_id"]
                                or proof["actual_weight_version"] != str(identity["policy_version"])
                                or set(hit["rank_matches"]) != ranks or set(proof["rank_generations"]) != ranks
                                or not -1 <= age <= self.ttl):
                            raise ValueError("native mixed-state proof is stale or belongs to another request/instance")
                        expected = row.get("native_rank_generations")
                        if expected is not None and expected != proof["rank_generations"]:
                            raise ValueError("stable native target/cache generation changed")
                        if any(set(value) != {"device", "cpu_local", "cpu_share", "disk"}
                               for value in hit["rank_matches"].values()):
                            raise ValueError("native rank coverage lacks explicit tier fields")
                        for value in proof["rank_generations"].values():
                            if (set(value) != {"engine_instance_id", "cache_generation"}
                                    or not isinstance(value["engine_instance_id"], str) or not value["engine_instance_id"]
                                    or not isinstance(value["cache_generation"], str) or len(value["cache_generation"])!=64
                                    or any(c not in "0123456789abcdef" for c in value["cache_generation"])):
                                raise ValueError("native rank generation is malformed")
                        native_prefixes = [_prefix(value, length) for value in hit["rank_matches"].values()]
                        if any(a != b or not 0 <= a < length for a,b in native_prefixes):
                            raise ValueError("native GPU-only coverage is inconsistent")
                        device = host = min(value[0] for value in native_prefixes)
                    elif self.context["hybrid_model"]:
                        coverage = row["hybrid_prefix_by_tp_rank"]
                        if set(coverage) != {str(rank) for rank in range(self.context["tp_size"])}:
                            raise ValueError("missing mixed-state rank coverage")
                        if any(not {"device", "cpu_local"} <= value.keys() for value in coverage.values()):
                            raise ValueError("missing explicit Attention/Mamba coverage")
                        ranks = [_prefix(value, length) for value in coverage.values()]
                        device = min(device, *(value[0] for value in ranks))
                        host = min(host, *(value[1] for value in ranks))
                    prefill = (length - device) * calibration["prefill_s_per_token"]
                    host_cost = (calibration["host_read_fixed_s"]
                                 + (host - device) * calibration["host_read_s_per_token"]
                                 + (length - host) * calibration["prefill_s_per_token"])
                    cost = (_number(row["estimated_queue_seconds"])
                            + _number(row["estimated_remaining_generation_seconds"]) + min(prefill, host_cost))
                    result["candidates"][instance_id] = {
                        "engine_url": row["engine_url"], "engine_instance_id": row["engine_instance_id"],
                        "estimated_seconds": cost, "uncertainty_seconds": _number(row["uncertainty_seconds"]),
                        "declared_device_prefix": device, "declared_host_prefix": host,
                    }
                    if self.directory_source == "native_prefix_probe":
                        result["candidates"][instance_id]["native_target"] = {
                            "schema_version": 1, "policy_version": str(identity["policy_version"]),
                            "query_digest": proof["query_digest"],
                            "rank_generations": deepcopy(proof["rank_generations"]),
                            "minimum_device_prefix": device,
                        }
                except (KeyError, TypeError, ValueError):
                    continue  # A malformed candidate cannot authorize advice.
            if len(result["candidates"]) < 2:
                result["reason"] = "insufficient_compatible_candidates"
            else:
                result["preferred_instance"] = min(result["candidates"], key=lambda key: (
                    result["candidates"][key]["estimated_seconds"], key))
                result.update(status="shadow_estimate", reason="declared_local_cache_and_load")
            # The file must still represent exactly the same membership/cache
            # generation after a slow query. Never recycle an earlier result.
            _, current = self._view(identity, salt, allowed_urls)
            if current != digest:
                result.update(status="fallback", reason="view_changed_during_query", candidates={})
            publication = await self._assert_current_publication(identity, deadline)
            self._assert_native_publication_membership(publication,
                [row["engine_url"] for row in view["instances"].values()])
        except AdvicePublicationInvalid:
            result.update(status="fallback", reason="publication_unavailable_or_mismatched", candidates={})
        except asyncio.TimeoutError:
            result.update(status="fallback", reason="query_timeout", candidates={})
        except (OSError, ValueError, KeyError, TypeError):
            result.update(status="fallback", reason="invalid_or_unavailable_view", candidates={})
        except Exception as error:
            result.update(status="fallback", reason="query_error:" + type(error).__name__, candidates={})
        finally:
            result["advice_seconds"] = time.perf_counter() - started
            if result["status"] == "fallback":
                result.pop("preferred_instance", None)
        return result

    async def observe(self, ticket, *, engine_url, worker_id, engine_instance_id):
        if ticket is None:
            return
        started = time.perf_counter()
        try:
            result = deepcopy(await asyncio.shield(ticket))
            result.update(actual_engine_url=engine_url, actual_worker_id=worker_id,
                          actual_engine_instance_id=engine_instance_id)
            if result["status"] == "shadow_estimate":
                publication = await self._assert_current_publication(
                    result["policy_identity"], started + self.timeout)
                self._assert_native_publication_membership(publication,
                    [row["engine_url"] for row in result["candidates"].values()])
                # Recheck at observation too: weights/retirement may change
                # after the query completed but before the actual SSE returned.
                _, current = self._view(result["policy_identity"], result["cache_salt"],
                                         [row["engine_url"] for row in result["candidates"].values()])
                if current != result["view_sha256"]:
                    raise ValueError("view changed before observation")
                actual = [key for key, row in result["candidates"].items()
                          if row["engine_url"] == engine_url and row["engine_instance_id"] == engine_instance_id]
                if len(actual) != 1:
                    raise ValueError("actual routed instance is not in the compatible view")
                best, native = result["candidates"][result["preferred_instance"]], result["candidates"][actual[0]]
                gain = (native["estimated_seconds"] - best["estimated_seconds"] - result["advice_seconds"]
                        - (time.perf_counter() - started))
                uncertainty = native["uncertainty_seconds"] + best["uncertainty_seconds"]
                result.update(actual_instance=actual[0], predicted_net_seconds=gain,
                              recommendation="consider_alternative" if gain > uncertainty else "retain_native")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            result = {"status": "fallback", "routing_changed": False, "reason": "observation_invalidated",
                      "error_type": type(error).__name__}
        result["observation_wait_seconds"] = time.perf_counter() - started
        self.records.append(result)

    def discard(self, ticket):
        if ticket is not None and not ticket.done():
            ticket.cancel()  # The one bounded HTTP query remains single-flight.

    def close(self):
        self._closed = True
        for ticket in tuple(self.tickets):
            ticket.cancel()
        self.client.close()


def create_route_advisor(settings, *, publication_client=None):
    value = settings.get("local_kv_routing", {})
    if value.get("mode", "disabled") == "disabled":
        return None
    run_id = settings.get("run_namespace")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("V1 shadow routing requires the actual run namespace")
    if publication_client is None:
        raise ValueError("V1 shadow routing requires the continuous publication authority")
    return V1KVRouteAdvisor(value, publication_client=publication_client, run_id=run_id)
