"""Read-only evidence at the original joint drain/sleep/train boundaries.

No release, scheduling, request generation, PID signalling or GPU allocation is
performed here. Native per-member release receipts keep their original fields.
"""
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid


def read_owned_training_gpu(outer_worker):
    from actor_worker import resolve_training_worker
    from elastic_gpu_observer import canonical_uuid
    import torch

    worker = resolve_training_worker(outer_worker)
    result = {'rank': worker.rank, 'pid': os.getpid(), 'hostname': socket.gethostname(),
              'observed_unix_s': time.time(), 'status': 'UNAVAILABLE', 'gpu_uuid': None,
              'native_engine_step_returns': getattr(worker, '_qwen35_engine_step_returns', None)}
    try:
        if not torch.cuda.is_initialized():
            raise ValueError('Existing training CUDA context is not initialized; observer will not initialize it')
        actual = canonical_uuid(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        result['gpu_uuid'] = actual
        identifier = 'GPU-' + str(uuid.UUID(hex=actual))
        command = ['nvidia-smi', '--id=' + identifier,
                   '--query-gpu=uuid,index,memory.used,memory.free', '--format=csv,noheader,nounits']
        response = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        lines = [line for line in response.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError('Exactly one actual assigned UUID must match NVML')
        fields = [part.strip() for part in lines[0].split(',')]
        if len(fields) != 4 or canonical_uuid(fields[0]) != actual:
            raise ValueError('NVML response differs from this process actual CUDA UUID')
        result.update(status='OBSERVED', gpu_uuid=actual, nvml_uuid=fields[0],
                      memory_used_bytes=int(fields[2]) * 1024**2,
                      memory_free_bytes=int(fields[3]) * 1024**2,
                      raw_stdout=response.stdout, query_argv=command,
                      native_engine_step_returns=getattr(worker, '_qwen35_engine_step_returns', None),
                      observer_allocated_GPU_memory=False, signalled_PIDs=[])
    except Exception as error:
        result.update(error_type=type(error).__name__, reason=str(error)[:1000])
    return result


def capture(driver, stage):
    evidence = driver.result.setdefault('gpu_return_observations', {})
    # Observe the one selected handoff transition. Initial construction or a
    # later publication sleep must not overwrite its actual before/after pair.
    if not driver.result.get('selected_actual_identity') or stage in evidence:
        return
    try:
        evidence[stage] = {'driver_pid': os.getpid(), 'driver_unix_s': time.time(),
                           'rows': driver._worker_rpc(read_owned_training_gpu)}
    except Exception as error:
        evidence[stage] = {'driver_pid': os.getpid(), 'driver_unix_s': time.time(),
                           'rows': [], 'error_type': type(error).__name__, 'reason': str(error)[:1000]}
    driver._persist_observation()


def finalize(driver, native_entry_report):
    """Keep the existing native proof and the CatPaw private-slot profile apart."""
    from assert_source_memory_saver import check_file
    from gpu_return_verdict import make_verdict

    root = Path(driver.options.result_path).parent
    raw = driver.result.get('gpu_return_observations', {})
    elastic = driver.result['elastic']
    checks, owners = {}, []
    for index, row in enumerate(elastic.get('hybrid_owner_observations', [])):
        if row.get('node_rank') != 0:
            continue
        path = root / ('gpu-return-source-server_args-%d.json' % index)
        path.write_text(json.dumps(row.get('server_args'), ensure_ascii=False, indent=2) + '\n')
        try:
            owners.append(check_file(str(path), role='source'))
        except Exception as error:
            owners.append({'file': str(path), 'pass': False, 'error_type': type(error).__name__, 'reason': str(error)})
    checks['actual_source_server_args_memory_saver_and_eager'] = len(owners) == 2 and all(row['pass'] for row in owners)
    source = set(driver.result.get('elastic_reuse_proof', {}).get('source_gpu_uuids', []))
    before = {row['gpu_uuid']: row for row in raw.get('before_sleep', {}).get('rows', []) if row.get('status') == 'OBSERVED'}
    after = {row['gpu_uuid']: row for row in raw.get('after_sleep', {}).get('rows', []) if row.get('status') == 'OBSERVED'}
    deltas = {key: after[key]['memory_free_bytes'] - before[key]['memory_free_bytes']
              for key in source if key in before and key in after}
    checks['same_four_source_UUIDs_independently_observed_before_and_after_sleep'] = len(source) == 4 and set(before) == set(after) == source
    checks['NVML_free_memory_increased_on_each_source_GPU'] = len(deltas) == 4 and all(value > 0 for value in deltas.values())
    primary = driver.result.get('elastic_reuse_proof', {})
    checks['original_full_replica_sleep_and_actual_UUID_training_proof'] = primary.get('verified') is True
    returns = []
    for row in raw.get('after_train', {}).get('rows', []):
        for item in row.get('native_engine_step_returns') or []:
            value = (item.get('grad_norm') or {}).get('value')
            if item.get('status') == 'ORIGINAL_ENGINE_RETURNED' and item.get('finite') is True and type(value) in (int, float) and math.isfinite(value) and value > 0:
                returns.append(item)
    checks['four_real_rank_gradient_returns'] = {row.get('rank') for row in returns} == set(range(4))
    checks['original_parameter_shard_and_first_update_guard'] = native_entry_report.get('first_update_observation') == 'PASS_REQUIRED_FIRST_UPDATE_OBSERVATIONS'
    checks['original_physical_handoff_digest_proof'] = driver.result.get('portable_state_handoff_proof', {}).get('verified') is True
    events = elastic.get('controller_events', [])
    before_s = raw.get('before_sleep', {}).get('driver_unix_s')
    begin = next((row for row in reversed(events) if row['event'] == 'replica_sleep_begin'
                  and before_s is not None and row.get('observed_unix_s', float('inf')) <= before_s), {})
    begin_s = begin.get('observed_unix_s')
    safe = next((row for row in reversed(events) if row['event'] == 'quiesced'
                 and begin_s is not None and row.get('observed_unix_s', float('inf')) < begin_s), {})
    end = next((row for row in events if row['event'] == 'sleep_completed'
                and begin_s is not None and row.get('observed_unix_s', float('-inf')) > begin_s), {})
    end_s = end.get('observed_unix_s')
    train = next((row for row in events if row['event'] == 'switch_to_trainer_begin'
                  and end_s is not None and row.get('observed_unix_s', float('-inf')) > end_s), {})
    timeline = {'safe_point_all_members_at': safe.get('observed_unix_s'),
                'release_begin': begin_s, 'release_end': end_s,
                'trainer_alloc_begin': train.get('observed_unix_s')}
    timeline['trainer_step_end'] = raw.get('after_train', {}).get('driver_unix_s')
    sequence = list(timeline.values())
    checks['same_driver_observed_native_operation_order'] = all(type(value) in (int, float) for value in sequence) and all(sequence[i] < sequence[i+1] for i in range(len(sequence)-1))
    # The existing CatPaw A1-A7 tool targets a different operation: a per-request
    # full-KV-pool release. This run uses native HYBRID sleep(weights,kv_cache).
    # Never translate the original False private-slot flags into full release.
    legacy = {'release_receipt': None, 'kv_pool': None, 'timeline': timeline,
              'nvml': {'before': [dict(row, uuid=row['gpu_uuid']) for row in before.values()],
                       'after': [dict(row, uuid=row['gpu_uuid']) for row in after.values()]},
              'trainer': {'alloc_gpu_uuids': primary.get('trainer_gpu_uuids', [])},
              'target_resume': {'resume_success': checks['original_physical_handoff_digest_proof'],
                                'digest_match': checks['original_physical_handoff_digest_proof']}}
    result = {'scope': 'NATIVE_HYBRID_DRAIN_FULL_REPLICA_SLEEP_AND_TRAINER_UUID_REUSE',
              'verified': all(checks.values()), 'checks': checks, 'timeline': timeline,
              'timeline_scope': 'Same Trainer driver observes original operation boundaries; includes RPC and observation overhead',
              'per_source_GPU_NVML_free_bytes_delta': deltas, 'raw': raw,
              'source_launch_assertions': owners, 'actual_rank_gradient_returns': returns,
              'original_sleep_receipts': elastic.get('sleep_observations'),
              'original_private_slot_release_receipts_unchanged': True,
              'per_request_full_gpu_pool_request': None,
              'per_request_full_gpu_pool_request_missing_reason': 'Original per-member policy is unchanged; this observer does not infer its request flag from native replica sleep',
              'KV_pool_capacity_formula': None,
              'KV_pool_capacity_formula_missing_reason': 'Native replica sleep releases weights and KV regions; per-request KV-only formula is not measured by this interface',
              'catpaw_private_slot_profile_applicable': False,
              'catpaw_original_profile_result': make_verdict(legacy),
              'R88_original_memory_saver_false_does_not_prove_GPU_return': True,
              'signalled_PIDs': []}
    path = root / 'gpu_returned_for_training_verdict.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result
