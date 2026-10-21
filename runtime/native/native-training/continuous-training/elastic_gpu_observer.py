"""Read actual process/device identities around existing sleep and native compute.

No allocation, scheduling, model construction, generation or optimizer invocation
is added. Observation failure never authorizes retrying native computation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time


def canonical_uuid(value):
    if isinstance(value, bytes):
        try:
            value = value.decode('ascii')
        except UnicodeDecodeError:
            value = value.hex()
    value = str(value).lower().removeprefix('gpu-').replace('-', '')
    return value if value and value not in {'none', 'unknown'} else None


def observe_source_process():
    """Ray owner metadata only; GPU identity comes from scheduler inspect."""
    return {'pid': os.getpid(), 'hostname': socket.gethostname(), 'observed_ns': time.time_ns(),
            'gpu_identity_source': 'separate actual scheduler token-KV inspect',
            'pid_namespace_numeric_matching_performed': False}


def source_scheduler_observations(observed, engine_url):
    """Validate the existing all-rank inspect response, without host-PID lookup."""
    identity, ranks = observed.get('identity', {}), observed.get('ranks', [])
    if (observed.get('status') != 'PASS' or identity.get('tp_size') != 2
            or identity.get('actualweight_version') is None
            or str(identity.get('actualweight_version')) != str(identity.get('weight_version'))
            or sorted(row.get('tp_rank') for row in ranks) != [0, 1]):
        raise ValueError('Actual TP2 scheduler inspect is incomplete')
    result = []
    for row in ranks:
        actual = row.get('scheduler_process_identity') or {}
        uuid = canonical_uuid(actual.get('gpu_uuid'))
        instance = row.get('engine_instance_id')
        if (actual.get('status') != 'OBSERVED' or actual.get('cuda_context_initialized') is not True
                or type(actual.get('pid')) is not int or actual['pid'] <= 0
                or actual.get('tp_rank') != row['tp_rank'] or uuid is None
                or not isinstance(actual.get('hostname'), str)
                or instance != f"{actual['hostname']}:{actual['pid']}"):
            raise ValueError('Actual scheduler CUDA UUID/PID/TP rank is unavailable or inconsistent')
        result.append({**actual, 'gpu_uuid': uuid, 'engine_url': engine_url.rstrip('/'),
                       'engine_instance_id': instance, 'weight_version': identity.get('actualweight_version'),
                       'identity_source': 'scheduler_cuda_current_device'})
    return result


def _cuda_identity(worker, engine):
    import torch
    if not torch.cuda.is_initialized():
        raise ValueError('Actual training CUDA context is not initialized')
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    uuid = canonical_uuid(getattr(props, 'uuid', None))
    if uuid is None:
        raise ValueError('Torch did not expose the actual current CUDA context UUID')
    parameters = [p for group in engine.optimizer.param_groups for p in group['params']]
    devices = sorted({str(p.device) for p in parameters})
    cuda_devices = {p.device.index for p in parameters if getattr(p.device, 'type', None) == 'cuda'}
    if cuda_devices and cuda_devices != {device}:
        raise ValueError('CUDA optimizer parameters differ from the current native compute device')
    audit = getattr(worker, '_production_update_audit', None)
    return {'rank': worker.rank, 'pid': os.getpid(), 'hostname': socket.gethostname(),
            'gpu_uuid': uuid, 'cuda_device': 'cuda:%d' % device,
            'cuda_context_initialized': True,
            'identity_source': 'torch_cuda_current_device_at_original_engine_boundary',
            'optimizer_parameter_devices': devices,
            'parameter_device': 'cuda:%d' % device if cuda_devices else None,
            'optimizer_execution_device_inferred': False,
            'scope': 'Original train_batch and optimizer-return boundaries joined by PID; CPU parameter/Adam offload remains possible',
            'trainer_step': getattr(audit, 'current_step', None), 'observed_ns': time.time_ns(),
            'process_monotonic_ns': time.monotonic_ns()}



def install_native_compute_observer(outer_worker, output_dir):
    """Use the native arbitrary-function dispatch; observe original engine calls."""
    from actor_worker import resolve_training_worker
    worker = resolve_training_worker(outer_worker)
    if getattr(worker, '_elastic_compute_observer', None) is not None:
        raise ValueError('Elastic compute observer already installed')
    engine = worker.actor.engine if hasattr(worker, 'actor') else worker.engine
    state = {'rank': worker.rank, 'pid': os.getpid(), 'events': [], 'errors': []}
    worker._elastic_compute_observer = state
    path = Path(output_dir) / f'rank-{worker.rank}-pid-{os.getpid()}.json'

    def save(event, **fields):
        try:
            row = {'event': event, **_cuda_identity(worker, engine), **fields}
            state['events'].append(row)
        except Exception as error:
            state['errors'].append({'event': event, 'error_type': type(error).__name__, 'message': str(error)[:240]})
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, allow_nan=False) + '\n')
            os.replace(temporary, path)
        except Exception as error:
            state['errors'].append({'event': 'receipt_io', 'error_type': type(error).__name__})

    original = engine.train_batch
    first = {'train': False, 'optimizer': False}

    def observed_train_batch(*args, **kwargs):
        selected = not first['train']
        if selected:
            first['train'] = True
            save('native_engine_train_batch_entered')
        try:
            result = original(*args, **kwargs)
        except BaseException:
            if selected:
                save('native_engine_train_batch_raised')
            raise
        if selected:
            save('native_engine_train_batch_returned')
        return result

    def optimizer_post(optimizer, args, kwargs):
        if not first['optimizer']:
            first['optimizer'] = True
            try:
                from training_update_audit import _step_frontier
                save('native_optimizer_step_returned', optimizer_step_frontier=_step_frontier(optimizer))
            except Exception as error:
                state['errors'].append({'event': 'optimizer_observation', 'error_type': type(error).__name__})

    engine.train_batch = observed_train_batch
    worker._elastic_optimizer_hook = engine.optimizer.register_step_post_hook(optimizer_post)
    return {'rank': worker.rank, 'pid': os.getpid(), 'installed': True}


def read_native_compute_observer(outer_worker):
    from actor_worker import resolve_training_worker
    worker = resolve_training_worker(outer_worker)
    return getattr(worker, '_elastic_compute_observer', None)


def verify_elastic_reuse(*, source_observations, target_observations, sleep_observations,
                         compute_rows, native_audit_rows, controller_events):
    """Join real evidence; a configured UUID, idle card or RPC dispatch cannot pass."""
    def uuids(observations):
        return {value for row in observations
                if row.get('identity_source') == 'scheduler_cuda_current_device'
                and row.get('status') == 'OBSERVED' and row.get('cuda_context_initialized') is True
                and (value := canonical_uuid(row.get('gpu_uuid'))) is not None}
    source, target = uuids(source_observations), uuids(target_observations)
    expected_events = ['switch_to_trainer_begin', 'quiesced', 'routing_removed', 'routing_stopped',
                       'aborted', 'replica_sleep_begin', 'sleep_completed',
                       'switch_to_trainer_returned', 'native_actor_update_dispatched']
    observed_names = [event['event'] for event in controller_events]
    end = observed_names.index('native_actor_update_dispatched') + 1 if 'native_actor_update_dispatched' in observed_names else len(observed_names)
    observed_names = observed_names[:end]
    ordered = all(observed_names.count(name) == 1 for name in expected_events)
    if ordered:
        positions = [observed_names.index(name) for name in expected_events]
        ordered = positions == sorted(set(positions))
    audits = {row['rank']: row for row in native_audit_rows if isinstance(row, dict)}
    computes = {row['rank']: row for row in compute_rows if isinstance(row, dict)}
    per_rank, trained = [], set()
    for rank in range(4):
        state, audit = computes.get(rank, {}), audits.get(rank, {})
        events = state.get('events', [])
        by_event = {row['event']: row for row in events}
        names = ['native_engine_train_batch_entered', 'native_optimizer_step_returned',
                 'native_engine_train_batch_returned']
        rows = [by_event.get(name, {}) for name in names]
        identities = {(row.get('pid'), canonical_uuid(row.get('gpu_uuid')), row.get('trainer_step')) for row in rows}
        valid = (not state.get('errors') and len(identities) == 1 and all(row.get('identity_source') == 'torch_cuda_current_device_at_original_engine_boundary'
                 and row.get('cuda_context_initialized') is True and row.get('cuda_device', '').startswith('cuda:') for row in rows)
                 and type(rows[0].get('trainer_step')) is int and rows[0]['trainer_step'] >= 1
                 and [row.get('event') for row in events] == names
                 and all(type(row.get('process_monotonic_ns')) is int for row in rows)
                 and [row['process_monotonic_ns'] for row in rows] == sorted(set(row['process_monotonic_ns'] for row in rows))
                 and audit.get('pid') == rows[0].get('pid') and audit.get('trainer_step') == rows[0].get('trainer_step')
                 and audit.get('verified_counter_advancements_since_attach', 0) >= 1
                 and audit.get('optimizer_calls_since_attach', 0) >= 1)
        if valid:
            trained.add(canonical_uuid(rows[0]['gpu_uuid']))
        per_rank.append({'rank': rank, 'actual_train_and_optimizer_same_process_verified': valid,
                         'events': events, 'native_optimizer_audit': audit})
    # A successful primary sleep releases both tags for its actual TP replica.
    primary = [row for row in sleep_observations if row.get('node_rank') == 0]
    sleep_ok = (len(primary) == 2
        and {row.get('engine_url', '').rstrip('/') for row in primary}
            == {row.get('engine_url', '').rstrip('/') for row in source_observations}
        and all((row.get('last_sleep') or {}).get('completed') is True
        and set(row['last_sleep'].get('tags', [])) == {'weights', 'kv_cache'}
        and row['last_sleep'].get('rollout_mode') == 'HYBRID'
        and row['last_sleep'].get('owner_pid') == row.get('pid')
        and type(row['last_sleep'].get('completed_ns')) is int for row in primary))
    checks = {'original_switch_order_observed': ordered, 'both_hybrid_replica_sleep_confirmed': sleep_ok,
              'source_four_actual_process_gpu_uuids': len(source) == 4,
              'target_four_actual_process_gpu_uuids': len(target) == 4,
              'four_native_compute_ranks_verified': all(row['actual_train_and_optimizer_same_process_verified'] for row in per_rank),
              'trainer_reused_all_hybrid_source_uuids': len(trained) == 4 and trained == source,
              'standalone_target_physically_disjoint': bool(source and target) and source.isdisjoint(target)}
    return {'verified': all(checks.values()), 'checks': checks, 'per_rank': per_rank,
            'source_gpu_uuids': sorted(source), 'target_gpu_uuids': sorted(target), 'trainer_gpu_uuids': sorted(trained),
            'identity_scope': 'actual scheduler CUDA current-device UUID and process-local PID; no NVML/host PID matching'}


def verify_portable_handoff(receipts, *, selected):
    """Validate existing all-rank load/readback proof; never infer from output tokens."""
    expected = selected.get('actual_weight_version')
    target_ranks = selected.get('target_inspect', {}).get('ranks', [])
    target_instances = {row.get('engine_instance_id') for row in target_ranks}
    target_tp_ranks = {row.get('engine_instance_id'): row.get('tp_rank') for row in target_ranks}
    checks, details = [], []
    for receipt in receipts:
        ranks = receipt.get('ranks', [])
        complete = (len(ranks) == 2 and {target_tp_ranks.get(row.get('engine_instance_id')) for row in ranks} == {0, 1})
        per_rank = []
        for row in ranks:
            inventory = row.get('state_inventory', [])
            identity = row.get('identity', {})
            before = row.get('other_active_request_ids_before_import')
            after = row.get('active_request_ids_after_import')
            fields = {
                'actual_same_version': (expected is not None
                    and str(identity.get('actualweight_version')) == str(expected)
                    and str(identity.get('weight_version')) == str(expected)),
                'real_fresh_target': (row.get('engine_instance_id') in target_instances
                    and row.get('from_retained_cpu_record') is False
                    and row.get('status') == 'PASS'
                    and all(row.get(key) is True for key in ['loaded_from_snapshot', 'allocated_new_slots',
                        'request_reconstructed', 'stream_returns_full_output', 'digestmatch'])),
                'all_42_actual_tensors': (len(inventory) == 42
                    and len({item.get('name') for item in inventory}) == 42
                    and all(isinstance(item.get('sha256'), str) and len(item['sha256']) == 64
                            and type(item.get('bytes')) is int and item['bytes'] > 0 for item in inventory)
                    and set(row.get('state_kinds', [])) == {'attention_kv', 'mamba_conv', 'mamba_temporal'}),
                'other_concurrent_requests_retained': (isinstance(before, list) and bool(before)
                    and isinstance(after, list) and set(before).issubset(after)
                    and row.get('target_rid') in after),
            }
            per_rank.append({'tp_rank': target_tp_ranks.get(row.get('engine_instance_id')), 'checks': fields,
                'actual_tensor_count': len(inventory), 'other_request_ids_before': before,
                'active_request_ids_after': after})
        valid = complete and all(all(row['checks'].values()) for row in per_rank)
        checks.append(valid)
        details.append({'snapshot_id': receipt.get('snapshot_id'), 'verified': valid, 'ranks': per_rank})
    return {'verified': bool(checks) and all(checks), 'receipts': details,
        'missing_concurrent_target_requests_means': 'NOT_PASS_NOT_OBSERVED; never synthesize background work',
        'source': 'Original native portable restore validated readback and all-rank receipts'}


def native_version1_witness_passed(report):
    witness = report.get('version1_token_witness') or {}
    return (witness.get('status') == 'PASS_NATIVE_VERSION1_TOKEN'
            and witness.get('generation_path') == 'native_ray_actor'
            and str(witness.get('actual_engine_weight_version')) == '1'
            and witness.get('client_generate_calls') == 1
            and type(witness.get('output_token_count')) is int
            and 1 <= witness['output_token_count'] <= 4
            and all(str((witness.get(key) or {}).get('actual_parameter_version')) == '1'
                    for key in ['publication_before', 'publication_after'])
            and (witness['publication_before'].get('active_policy_record')
                 == witness['publication_after'].get('active_policy_record')))
