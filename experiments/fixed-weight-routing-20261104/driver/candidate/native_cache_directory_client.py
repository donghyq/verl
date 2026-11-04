# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Bounded native per-request prefix observations; shadow only, no dispatch."""
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
import math
import threading
import time

from .conductor_hint_client import ConductorHTTPClient


def native_query_digest(tokens, salt):
    return hashlib.sha256(json.dumps({'token_ids': tokens, 'cache_salt': salt},
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class NativeEnginePrefixClient(ConductorHTTPClient):
    """Reuse the existing bounded/cancellable HTTP transport, with JSON wire format."""
    def _encode_request(self, payload):
        return json.dumps(payload).encode()

    def _wire_request(self, body):
        query = json.loads(body)
        return (self._url.path.rstrip('/') + '/token_kv_snapshot', json.dumps({
            'operation': 'inspect_cache_prefix',
            'expected_weight_version': query['expected_weight_version'],
            'cache_query': {key: query[key] for key in (
                'token_ids', 'cache_salt', 'query_digest', 'expected_rank_generations') if key in query},
        }).encode(), {'Content-Type': 'application/json'})

    def _decode_response(self, data):
        result = json.loads(data)
        if not isinstance(result, dict) or result.get('status') != 'PASS':
            raise ValueError('native cache query failed')
        return result


def validate_native_prefix_response(result, query, tp_size):
    tokens = query['token_ids'];digest = native_query_digest(tokens, query['cache_salt'])
    ranks = result.get('ranks')
    if (result.get('status') != 'PASS' or not isinstance(ranks, list) or len(ranks) != tp_size
            or any(type(row.get('tp_rank')) is not int for row in ranks)
            or {row['tp_rank'] for row in ranks} != set(range(tp_size))):
        raise ValueError('native prefix response lacks every unique TP rank')
    ordered = sorted(ranks, key=lambda row: row['tp_rank'])
    matches, generations = {}, {}
    for row in ordered:
        if (row.get('status') != 'PASS' or row.get('query_digest') != digest
                or type(row.get('prompt_tokens')) is not int or row.get('prompt_tokens') != len(tokens) or row.get('cache_salt') != query['cache_salt']
                or row.get('actual_weight_version') != query['expected_weight_version']
                or row.get('coverage_source') != 'native_read_only_full_mamba_tree'
                or row.get('host_state_advertised') is not False
                or row.get('generation_admission_authorized') is not False):
            raise ValueError('native prefix proof is not bound to this request and weight version')
        instance, generation = row.get('engine_instance_id'), row.get('cache_generation')
        if (not isinstance(instance, str) or not instance or not isinstance(generation, str)
                or len(generation) != 64 or any(c not in '0123456789abcdef' for c in generation)):
            raise ValueError('native target generation is unavailable')
        match = row.get('rank_match')
        if (not isinstance(match, dict) or set(match) != {'device', 'cpu_local', 'cpu_share', 'disk'}
                or any(type(v) is not int or not 0 <= v < len(tokens) for v in match.values())
                or len(set(match.values())) != 1
                or type(row.get('native_complete_device_prefix')) is not int
                or row.get('native_complete_device_prefix') != match['device']):
            raise ValueError('invalid device-only native coverage')
        for key in ('waiting_requests', 'running_requests'):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError('invalid native queue counter')
        observed = row.get('observed_at_unix')
        if isinstance(observed, bool) or not isinstance(observed, (int,float)) or not math.isfinite(observed):
            raise ValueError('invalid native observation time')
        key = str(row['tp_rank']);matches[key] = match
        generations[key] = {'engine_instance_id': instance, 'cache_generation': generation}
    expected = query.get('expected_rank_generations')
    if expected is not None and expected != generations:
        raise ValueError('native stable target or cache generation changed')
    return {'rank_matches': matches, 'native_query': {
        'query_digest': digest, 'actual_weight_version': query['expected_weight_version'],
        'engine_instance_id': ordered[0]['engine_instance_id'], 'rank_generations': generations,
        'observed_at_unix': min(row['observed_at_unix'] for row in ordered),
        'queue_counters_by_rank': {str(row['tp_rank']): {k:row[k] for k in ('waiting_requests','running_requests')}
                                   for row in ordered},
    }}


class NativeCacheDirectoryClient:
    """Observe configured native engines concurrently, with one bounded query in flight."""
    def __init__(self, instances, *, tp_size, timeout_ms=50):
        if (not isinstance(instances, dict) or not 1 <= len(instances) <= 8
                or type(tp_size) is not int or not 1 <= tp_size <= 8
                or isinstance(timeout_ms,bool) or not 0 < timeout_ms <= 100):
            raise ValueError('bounded native instance map, TP size and timeout required')
        if any(not isinstance(k,str) or not k or not isinstance(v,str) for k,v in instances.items()):
            raise ValueError('native instance keys/URLs must be explicit')
        if len(set(instances.values())) != len(instances):
            raise ValueError('native instance URLs must be unique')
        self.instances = dict(instances);self.tp_size = tp_size;self.timeout = timeout_ms/1000
        self.clients = {k:NativeEnginePrefixClient(v,timeout_ms) for k,v in instances.items()}
        self.pool = ThreadPoolExecutor(max_workers=len(instances),thread_name_prefix='native-prefix')
        self.lock = threading.Lock();self.closed = False;self.last_errors = {}

    def query(self, payload):
        if self.closed or not self.lock.acquire(blocking=False):
            raise TimeoutError('native prefix directory is closed or busy')
        try:
            if payload.get('lora_name') != '':
                raise ValueError('native directory v1 does not support LoRA')
            allowed = payload['native_allowed_instances']
            if (not isinstance(allowed,dict) or not allowed or len(allowed)>8
                    or any(self.instances.get(k)!=v for k,v in allowed.items())):
                raise ValueError('native query membership differs from original pool')
            query = {key:payload[key] for key in ('token_ids','cache_salt','expected_weight_version')}
            query['query_digest'] = native_query_digest(query['token_ids'],query['cache_salt'])
            expected = payload.get('native_expected_rank_generations',{})
            futures={}
            for key in allowed:
                per_instance = dict(query)
                if key in expected:per_instance['expected_rank_generations'] = expected[key]
                futures[self.pool.submit(self.clients[key].query,per_instance)] = (key,per_instance)
            done,pending = wait(futures,timeout=self.timeout)
            self.last_errors = {futures[future][0]: "query_timeout" for future in pending}
            for future in pending:future.cancel()
            output = {}
            for future in done:
                key,per_instance=futures[future]
                try:output[key] = validate_native_prefix_response(future.result(),per_instance,self.tp_size)
                except Exception as error:
                    self.last_errors[key] = type(error).__name__ + ": " + str(error)[:256]
                    continue  # Failed peers never inherit prior coverage.
            return output
        finally:
            self.lock.release()

    def close(self):
        self.closed = True
        for client in self.clients.values():client.close()
        self.pool.shutdown(wait=False,cancel_futures=True)
