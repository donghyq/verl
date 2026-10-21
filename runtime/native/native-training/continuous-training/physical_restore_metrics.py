"""Consume recovery marker receipts already loaded by the existing driver."""


def physical_restore_summary(receipts):
    unique, unavailable = {}, []
    for row in receipts:
        value = row.get('physical_restore_input_compute') or {}
        rid = value.get('target_rid')
        count = value.get('physical_restore_actual_recomputed_tokens')
        if (value.get('status') != 'OBSERVED_PHYSICAL_IMPORT_ONLY' or not rid
                or type(count) is not int or count < 0):
            unavailable.append(row.get('trajectory_id'))
            continue
        if rid in unique and unique[rid] != value:
            unavailable.append(rid)
        unique[rid] = value
    return {'physical_restore_actual_recomputed_tokens':
                sum(row['physical_restore_actual_recomputed_tokens'] for row in unique.values())
                if unique and not unavailable else None,
            'physical_restore_compute_scope': 'only observed physical target requests, deduplicated by actual target_rid; TP ranks agree and are not summed',
            'physical_restore_compute_receipts': list(unique.values()),
            'physical_restore_compute_unavailable': unavailable,
            'all_path_actual_recomputed_tokens': None,
            'ray_fallback_compute_coverage': 'NOT_OBSERVED'}
