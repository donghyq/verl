import math
from datetime import datetime, timezone


def _percentile(sorted_values, pct):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * pct
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def _stage_observations(segments):
    observations = {}
    counters = {}
    seen_request_ids = set()
    for seg in segments or []:
        stage = seg.get('stage')
        if not isinstance(stage, str) or not stage:
            continue
        if seg.get('status') != 'success':
            continue
        if stage not in observations:
            observations[stage] = {'elapsed_s': [], 'sample_count': 0, 'pids': set(), 'boot_ids': set()}
        obs = observations[stage]
        elapsed = seg.get('elapsed_s')
        if isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0:
            obs['elapsed_s'].append(float(elapsed))
        obs['sample_count'] += 1
        if 'pid' in seg:
            obs['pids'].add(seg.get('pid'))
        if 'boot_id' in seg:
            obs['boot_ids'].add(seg.get('boot_id'))
        # Physical recomputation is only counted from an explicit native target counter,
        # one time per unique request_id, and only for successful segments.
        tokens = seg.get('actual_recomputed_tokens')
        if stage == 'physical_target' and isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
            rid = seg.get('request_id')
            if isinstance(rid, str) and rid:
                key = ('physical_target', rid)
                if key not in seen_request_ids:
                    seen_request_ids.add(key)
                    counters['physical_recomputed_tokens'] = counters.get('physical_recomputed_tokens', 0) + tokens
    return observations, counters


def summarize_window(row, segments, allocation_start_unix_s=None, allocation_end_unix_s=None):
    if row is None:
        raise ValueError('row is required')

    groups = row.get('groups') or []
    window_elapsed_s = row.get('window_elapsed_s')
    allocated_seconds = row.get('total_allocated_seconds')
    gpu_uuids = row.get('actual_gpu_uuids') or []
    physical_new_tokens = row.get('actual_new_model_tokens')

    valid, rejected = [], []
    nonzero_signal_groups = 0
    for group in groups:
        members = group.get('members') or []
        gate = group.get('original_grpo_gate') or {}
        reasons = []
        if group.get('native_runtime_status') != 'completed':
            reasons.append('native_runtime_not_completed')
        if len(members) != 8:
            reasons.append('member_count_not_8')
        if any(not m.get('complete') for m in members):
            reasons.append('incomplete_member')
        if any(not m.get('history_preserved') for m in members):
            reasons.append('history_not_preserved')
        rewards = [m.get('reward') for m in members]
        if any(not isinstance(r, (int, float)) or isinstance(r, bool) or not math.isfinite(r) for r in rewards):
            reasons.append('reward_not_finite')
        if gate.get('complete_prompt_groups') != 1:
            reasons.append('original_gate_not_complete')
        if (set(group.get('native_output_record_indices') or []) != set(range(8))
                or {m.get('member_index') for m in members} != set(range(8))
                or gate.get('original_native_helpers_used') is not True
                or any(m.get('uid') != group.get('uid') for m in members)):
            reasons.append('native_output_records_not_8')
        if reasons:
            rejected.append({'uid': group.get('uid'), 'reasons': reasons})
        else:
            valid.append({
                'uid': group.get('uid'),
                'completion_seconds': group.get('completion_seconds'),
                'effective_training_tokens': sum(m.get('effective_training_tokens', 0) for m in members),
            })
            if gate.get('nonzero_training_advantage'):
                nonzero_signal_groups += 1

    valid_elapsed = sorted(float(g['completion_seconds']) for g in valid if isinstance(g.get('completion_seconds'), (int, float)))
    valid_tokens = sum(g['effective_training_tokens'] for g in valid)

    window_cost_gpu_hours = None
    if (isinstance(allocated_seconds, (int, float)) and allocated_seconds > 0 and
            len(gpu_uuids) == len(set(gpu_uuids)) and gpu_uuids):
        window_cost_gpu_hours = allocated_seconds * len(gpu_uuids) / 3600.0

    allocation_cost_gpu_hours = None
    preparation_seconds = 'UNKNOWN'
    if (allocation_start_unix_s is not None and allocation_end_unix_s is not None and
            isinstance(allocation_start_unix_s, (int, float)) and
            isinstance(allocation_end_unix_s, (int, float)) and
            allocation_end_unix_s >= allocation_start_unix_s and
            isinstance(window_elapsed_s, (int, float)) and window_elapsed_s > 0):
        allocation_elapsed = allocation_end_unix_s - allocation_start_unix_s
        prep = allocation_elapsed - window_elapsed_s
        preparation_seconds = prep if prep >= 0 else 'UNKNOWN'
        if prep >= 0:
            allocation_cost_gpu_hours = allocation_elapsed * len(gpu_uuids) / 3600.0

    obs, counters = _stage_observations(segments)
    stage_results = {}
    for stage, data in obs.items():
        values = sorted(data['elapsed_s'])
        stage_results[stage] = {
            'observed_elapsed_s': values if values else None,
            'sample_count': data['sample_count'],
            'elapsed_sample_count': len(values),
            'p50_s': _percentile(values, 0.50),
            'p95_s': _percentile(values, 0.95),
            'pids': sorted(str(x) for x in data['pids']),
            'boot_ids': sorted(str(x) for x in data['boot_ids']),
        }

    return {
        'arm': row.get('arm'),
        'accounting_scope': 'cost_and_observation_only',
        'valid_group_count': len(valid),
        'rejected_group_count': len(rejected),
        'rejected_groups': rejected,
        'valid_group_rate': (len(valid) / len(groups)) if groups else None,
        'nonzero_signal_group_count': nonzero_signal_groups,
        'group_completion_seconds': {
            'p50_s': _percentile(valid_elapsed, 0.50),
            'p95_s': _percentile(valid_elapsed, 0.95),
            'per_group': {str(g['uid']): g['completion_seconds'] for g in valid},
        },
        'token_accounting': {
            'valid_effective_training_tokens': valid_tokens,
            'driver_actual_new_model_tokens': physical_new_tokens if isinstance(physical_new_tokens, int) else None,
            'physical_member_new_tokens': None,
            'physical_recomputed_tokens_native_counter': counters.get('physical_recomputed_tokens'),
            'token_normalization': 'only_integer_actual_tokens_no_estimates',
            'note': 'driver total is the unchanged window counter; physical-member generated tokens have no producer here and remain null',
        },
        'gpu_cost': {
            'unique_gpu_count': len(set(gpu_uuids)),
            'window_cost_gpu_hours': window_cost_gpu_hours,
            'allocation_cost_gpu_hours': allocation_cost_gpu_hours,
            'preparation_seconds': preparation_seconds,
            'basis': 'all_window_allocated_gpus_including_preparation_and_idle',
        },
        'stage_observations': stage_results,
        'benefit_or_performance': 'NOT_RUN',
        'zero_variance_policy': 'zero_variance_is_not_an_automatic_reward_change',
    }
