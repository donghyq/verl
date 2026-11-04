"""Fixed-loaded-weight component screen. No Trainer or parameter publication."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import threading
import time
import urllib.error
import urllib.request

from candidate.kv_route_advice import V1KVRouteAdvisor
from candidate.native_cache_directory_client import NativeCacheDirectoryClient
from gateway_readiness import wait_workers
from observed_directory import ObservedDirectory


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sample(body, budget):
    meta = body.get("meta_info", {})
    ids, logs = body.get("output_ids"), meta.get("output_token_logprobs")
    if (meta.get("finish_reason", {}).get("type") != "length" or meta.get("weight_version") != "1"
            or meta.get("completion_tokens") != budget or not isinstance(ids, list) or len(ids) != budget
            or not isinstance(logs, list) or len(logs) != budget
            or any(len(row) < 2 or type(row[1]) is not int or row[1] != token
                   or not math.isfinite(row[0]) for token, row in zip(ids, logs))):
        raise ValueError("incomplete, rejected or wrong-weight sample")
    return ids, [row[0] for row in logs]


def request_body(tokens, salt, budget):
    return {"input_ids": tokens, "cache_salt": salt,
            "sampling_params": {"temperature": 0, "max_new_tokens": budget, "ignore_eos": True,
                                "frequency_penalty": 0, "presence_penalty": 0, "repetition_penalty": 1},
            "return_logprob": True, "logprob_start_len": -1, "stream": False}


class Wire:
    def __init__(self, root, deadline):
        self.root, self.deadline = Path(root), deadline
        self.lock = threading.Lock()
        self.count = 0

    def sync(self, label, url, path, body=None, headers=None, method=None, request_timeout=120):
        remaining = self.deadline - time.time()
        if remaining < 2:
            raise TimeoutError("absolute request deadline")
        req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
              headers={"Content-Type": "application/json", **(headers or {})}, method=method)
        started, tick = time.time(), time.perf_counter()
        failure = None
        status, raw = None, b""
        try:
            try:
                with urllib.request.urlopen(req, timeout=min(request_timeout, remaining)) as response:
                    status, raw = response.status, response.read()
            except urllib.error.HTTPError as error:
                status, raw = error.code, error.read()
        except Exception as error:
            failure = error
        row = {"label": label, "url": url, "path": path, "request": body, "request_headers": headers or {},
               "status": status, "started": started, "finished": time.time(),
               "wall_seconds": time.perf_counter() - tick, "raw_response": raw.decode(errors="replace"),
               "outcome": "UNKNOWN_NO_RETRY" if failure else "RESPONSE_RECEIVED",
               "error": None if failure is None else {"type": type(failure).__name__, "message": str(failure)}}
        with self.lock:
            self.count += 1
            row["sequence"] = self.count
            with (self.root / "wire.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if failure is not None:
            raise failure
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = {"plain_text": raw.decode(errors="replace")}
        return status, value, row

    async def call(self, *args, **kwargs):
        # Exactly one external attempt. Unknown transport outcomes propagate.
        return await asyncio.to_thread(self.sync, *args, **kwargs)


class FixedLoadObservations:
    """Observe owned /model_info for a static-load diagnostic, never publish weights.

    The record format is consumed by the unchanged SHADOW advisor. There is no
    Trainer/coordinator here, and this observer must never be used as one.
    """
    def __init__(self, wire, urls, identity, run_id):
        self.wire, self.urls, self.identity, self.run_id = wire, urls, identity, run_id

    async def get_active_published_policy(self):
        observations = []
        for url in self.urls:
            status, info, _ = await self.wire.call("fixed-load-model-info", url, "/model_info",request_timeout=.1)
            if status != 200 or str(info.get("weight_version")) != "1":
                raise ValueError("fixed loaded version changed")
            observations.append({"engine_url": url, "weight_version": "1", "source": "model_info"})
        return {"run_id": self.run_id, "identity": deepcopy(self.identity),
                "model_configuration_fingerprint": digest({"urls": self.urls, "version": "1"}),
                "publisher_id": "STATIC_LOAD_COMPONENT_OBSERVER_NOT_TRAINER_PUBLICATION",
                "engine_observations": observations, "committed_at_unix": time.time()}


class JointRouter:
    """Use the unchanged V1 shadow candidate, then dispatch only in this screen."""
    def __init__(self, root, wire, urls, gateway, mapping, calibration, model, run_id, *, fixed_weights):
        if fixed_weights is not True:
            raise ValueError("only owned fixed-loaded-weight component experiments allowed")
        self.root, self.wire, self.urls, self.gateway = Path(root), wire, urls, gateway
        self.mapping, self.calibration, self.model, self.run_id = mapping, calibration, model, run_id
        self.identity = {"policy_version": "1", "model_weights_fingerprint": "static-load:" + model,
                         "tokenizer_template_fingerprint": "owned-model-tokenizer",
                         "kv_cache_schema_fingerprint": "pinned-native-full-mamba-tp1"}
        self.client = NativeCacheDirectoryClient({str(i): u for i, u in enumerate(urls)}, tp_size=1, timeout_ms=100)
        self.authority = FixedLoadObservations(wire, urls, self.identity, run_id)
        self.lock = asyncio.Lock()
        self.decisions = []
        self.advisor=None
        self.advice_ticket=None
        self.producer_task=None
        self.closed=False
        self.route_budget_s=.1

    async def headers(self, body):
        at = time.perf_counter()
        try:
            headers,row=await asyncio.wait_for(self._headers(body),self.route_budget_s)
        except asyncio.TimeoutError:
            headers,row={}, {'status':'fallback','reason':'total_route_budget_exhausted','component_dispatch':False}
        except (OSError,ValueError,RuntimeError) as error:
            headers,row={}, {'status':'fallback','reason':'directory_unavailable:'+type(error).__name__,'component_dispatch':False}
        row['route_seconds']=time.perf_counter()-at
        self.decisions.append(row)
        return headers,row

    async def _headers(self,body):
        async with self.lock:
            if self.closed:return {},{'status':'fallback','reason':'router_closed','component_dispatch':False}
            if any(task is not None and not task.done() for task in (self.producer_task,self.advice_ticket,
                    None if self.advisor is None else self.advisor.pending)):
                return {},{'status':'fallback','reason':'query_in_flight','component_dispatch':False}
            query = {"model":self.model,"tenant_id":"component-screen","lora_name": "","block_size":1,
                     "token_ids": body["input_ids"], "cache_salt": body["cache_salt"],
                     "expected_weight_version": "1", "native_allowed_instances": self.client.instances}
            self.producer_task=asyncio.create_task(asyncio.to_thread(self.client.query,query))
            self.producer_task.add_done_callback(lambda t:None if t.cancelled() else t.exception())
            observations=await asyncio.shield(self.producer_task)
            if len(observations) != 2:
                row = {"status": "fallback", "reason": "producer_query_failed", "errors": deepcopy(self.client.last_errors),
                       "component_dispatch":False}
                return {}, row
            service = self.calibration["service_seconds"]
            instances = {}
            for key, obs in observations.items():
                counters = list(obs["native_query"]["queue_counters_by_rank"].values())
                waiting = max(x["waiting_requests"] for x in counters)
                running = max(x["running_requests"] for x in counters)
                instances[key] = {"engine_url": self.client.instances[key], "available": True,
                    "engine_instance_id": obs["native_query"]["engine_instance_id"],
                    "native_rank_generations": obs["native_query"]["rank_generations"],
                    "estimated_queue_seconds": waiting * service / 4,
                    "estimated_remaining_generation_seconds": max(0, running - 3) * service / 4,
                    "uncertainty_seconds": service / 4}
            view = {"schema_version": 1, "generation": digest(observations), "observed_at_unix": time.time(),
                    "policy_identity": self.identity,
                    "cache_context": {"model": self.model, "tenant_id": "component-screen", "lora_name": "",
                                      "block_size": 1, "tp_size": 1, "dp_size": 1, "hybrid_model": True,
                                      "cache_salt": body["cache_salt"]},
                    "calibration": self.calibration["advisor"], "instances": instances}
            path = (self.root / "route-view.json").resolve()
            save(path, view)
            observed=ObservedDirectory(observations,query)
            if self.advisor is None:
                self.advisor = V1KVRouteAdvisor({"mode": "shadow", "view_path": str(path), "cache_context": {
                k: v for k, v in view["cache_context"].items() if k != "cache_salt"},
                "timeout_ms": 100, "max_view_age_s": 2, "directory_source": "native_prefix_probe",
                "native_instances": self.client.instances}, client=observed,
                publication_client=self.authority, run_id=self.run_id)
            else:self.advisor.client=observed
            self.advice_ticket=self.advisor.start(body,policy_identity=self.identity,
                cache_salt=body['cache_salt'],allowed_urls=tuple(self.urls))
            if self.advice_ticket is None:return {},{'status':'fallback','reason':'advisor_unavailable','component_dispatch':False}
            row=await asyncio.shield(self.advice_ticket)
            headers = {}
            if row["status"] == "shadow_estimate":
                key = row["preferred_instance"]
                target = row["candidates"][key]
                headers = {"x-smg-target-worker-id": self.mapping[target["engine_url"]],
                           "x-sglang-native-target": json.dumps(target["native_target"], separators=(",", ":"))}
            row.update(producer_observations=observations,
                       component_dispatch=bool(headers), production_advisor_mode="shadow")
            return headers, row

    def close(self):
        self.closed=True
        if self.advisor is not None:self.advisor.close()
        self.client.close()


def complete_groups(rows, expected_members, turns):
    groups = {}
    for row in rows:
        if (type(row["group"]) is not int or type(row["member"]) is not int
                or type(row["turn"]) is not int or not 0 <= row["member"] < expected_members
                or not 0 <= row["turn"] < turns or not math.isfinite(row["finished"])):
            raise ValueError("invalid group member completion")
        key = (row["group"], row["member"], row["turn"])
        if key in groups:
            raise ValueError("duplicate member/turn completion")
        groups[key] = row
    ids = sorted({row["group"] for row in rows})
    for group in ids:
        expected = {(group, member, turn) for member in range(expected_members) for turn in range(turns)}
        actual = {key for key in groups if key[0] == group}
        if actual != expected:
            raise ValueError("incomplete GRPO-shaped component group")
    return [{"group": group, "ready_at": max(r["finished"] for r in rows if r["group"] == group)} for group in ids]


async def run_arm(root, wire, gateway, urls, arm, prompts, suffix, calibration, model, run_id, groups=2):
    started = time.time()
    mapping = await wait_workers(wire,gateway,urls,arm)
    joint = JointRouter(root, wire, urls, gateway, mapping, calibration, model, run_id,
                        fixed_weights=True) if arm == "joint" else None
    rows = []
    salt = "project-b-screen:" + run_id
    async def episode(group, member):
        tokens = list(prompts[group])
        for turn in range(2):
            # A fixed long-tail member; same rule, IDs and budgets in every arm.
            budget = (128 if member == 7 else 32) if turn == 0 else 32
            body = request_body(tokens, salt, budget)
            at = time.time()
            headers = {"x-smg-routing-key": f"group-{group}-member-{member}"}
            decision = None
            if joint:
                targets, decision = await joint.headers(body)
                headers.update(targets)
            code, result, receipt = await wire.call(f"{arm}-g{group}-m{member}-t{turn}", gateway,
                                                   "/generate", body, headers)
            if code != 200:
                raise ValueError("generation rejected; no automatic retry: " + str(code))
            ids, logs = sample(result, budget)
            rows.append({"group": group, "member": member, "turn": turn, "budget": budget,
                         "input_digest": digest(tokens), "output_ids": ids, "output_logprobs": logs,
                         "started": at, "finished": time.time(), "wire_seconds": receipt["wall_seconds"],
                         "cached_tokens": result["meta_info"].get("cached_tokens"),
                         "native_e2e_seconds": result["meta_info"].get("e2e_latency"),
                         "route_seconds": 0 if decision is None else decision["route_seconds"],
                         "scheduler_queue_seconds": None})
            tokens = tokens + ids + suffix
    try:
        # Every task is awaited even when another one fails. Outstanding native
        # requests are not silently abandoned before cleanup or a cache flush.
        tasks = [asyncio.create_task(episode(g, m)) for g in range(groups) for m in range(8)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [str(x) for x in results if isinstance(x, BaseException)]
        if errors:
            raise ValueError("arm failed: " + repr(errors))
        ready = complete_groups(rows, 8, 2)
        assert len(ready) == groups
        finished = time.time()
        return {"arm": arm, "started": started, "finished": finished, "wall_seconds": finished - started,
                "rows": rows, "groups_ready": ready, "group_ready_gpu_hours": (finished - started) * 2 / 3600,
                "route_decisions": [] if joint is None else joint.decisions,
                "optimizer_updates": 0, "formal_rl_metric": None,
                "limits": "Synthetic two-turn GRPO-shaped fixed-weight component load, not a real RL optimizer update. Exact scheduler wait unavailable; native queue counters and full client/group wall times are recorded separately."}
    finally:
        save(Path(root) / (run_id + "-" + arm + "-partial.json"), {"rows": rows,
             "route_decisions": [] if joint is None else joint.decisions, "finished": time.time()})
        if joint:
            joint.close()


def compare_samples(reference, candidate):
    def index(result):
        return {(r["group"], r["member"], r["turn"]): r for r in result["rows"]}
    left, right = index(reference), index(candidate)
    if left.keys() != right.keys():
        return {"equivalent": False, "reason": "different completed members"}
    failures, max_delta = [], 0.0
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["input_digest"] != b["input_digest"] or a["output_ids"] != b["output_ids"]:
            failures.append(list(key))
        if len(a["output_logprobs"]) != len(b["output_logprobs"]):
            failures.append(list(key))
        else:
            max_delta = max(max_delta, *(abs(x-y) for x,y in zip(a["output_logprobs"], b["output_logprobs"])))
    return {"equivalent": not failures and max_delta == 0, "token_mismatch_members": failures,
            "max_reported_logprob_abs_delta": max_delta, "logprob_tolerance": 0.0}
