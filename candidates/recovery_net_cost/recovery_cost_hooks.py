"""Observations around the existing partial-rollout calls; no extra engine call."""
import contextlib
import functools
import json
import os
import time
from pathlib import Path


def emit(path, stage, status, **fields):
    try:
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        boot_id=None
    row = dict(stage=stage, status=status, pid=os.getpid(),boot_id=boot_id, **fields)
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
    except (OSError, ValueError, TypeError) as error:
        # Observation failure must not turn a completed original call into a retry.
        print('RECOVERY_COST_OBSERVATION_MISSING:'+type(error).__name__, flush=True)


@contextlib.asynccontextmanager
async def span(path, stage, **fields):
    started = time.monotonic()
    try:
        yield
    except BaseException as error:
        emit(path, stage, 'failed', elapsed_s=time.monotonic()-started,
             error_type=type(error).__name__, **fields)
        raise
    else:
        emit(path, stage, 'success', elapsed_s=time.monotonic()-started, **fields)


def observe_backend(backend, path, *, source):
    """Bind wrappers to the existing instance; internal self.control uses this hook."""
    if source:
        original = backend.control

        @functools.wraps(original)
        async def control(route, payload):
            operation = payload.get('operation') if isinstance(payload, dict) else None
            stage = ('source_pause' if route == '/pause_generation' else
                     'source_export' if operation == 'export' else
                     'source_release' if operation == 'release' else 'source_control')
            async with span(path, stage, operation=operation, route=route):
                return await original(route, payload)
        backend.control = control
    else:
        original = backend.generate_from_snapshot

        @functools.wraps(original)
        async def restore(*args, **kwargs):
            # Server-side tensor read/import is combined by this original RPC.
            async with span(path, 'target_restore_and_continuation_combined'):
                return await original(*args, **kwargs)
        backend.generate_from_snapshot = restore
    return backend


def record_native_recovery(path, recovery):
    value = recovery.get('physical_restore_input_compute') or {}
    count = value.get('physical_restore_actual_recomputed_tokens')
    rid = value.get('target_rid')
    if (value.get('status') == 'OBSERVED_PHYSICAL_IMPORT_ONLY' and isinstance(rid, str)
            and type(count) is int and count >= 0):
        emit(path, 'physical_target', 'success', request_id=rid,
             actual_recomputed_tokens=count,
             counter_scope='native_completed_forward_post_import_only_TP_not_summed')
    else:
        emit(path, 'physical_target_counter', 'unavailable',
             reason='Explicit native post-import successful-forward counter absent')


def summarize(root, row, *, allocation=None):
    from net_cost_accounting import summarize_window
    from benefit_metrics import window_metrics

    records = []
    missing = []
    for name in ('source', 'target'):
        path = Path(root)/name/'recovery-cost-segments.jsonl'
        if path.is_file():
            records += [json.loads(line) for line in path.read_text().splitlines() if line]
        else:
            missing.append(str(path))
    adaptation = dict(row, window_elapsed_s=row.get('raw_owned_window_seconds'))
    result = summarize_window(adaptation, records,
        allocation_start_unix_s=(allocation or {}).get('actual_assigned_start_unix_s'),
        allocation_end_unix_s=(allocation or {}).get('actual_released_unix_s'))
    # Preserve the original complete-group definition and nearest-rank P95.
    # Zero reward variance is separate from valid original scored completion.
    try:
        original = window_metrics(row)
    except ValueError as error:
        original = {'status': 'NOT_ACCEPTED', 'reason': str(error)}
    result['original_window_metrics'] = original
    result['observation_missing_paths'] = missing
    result['nested_spans_must_not_be_added_to_window_elapsed'] = True
    result['read_only_cost_segments_not_observed_separately'] = {
        'physical_tensor_read_s': None, 'physical_tensor_import_s': None,
        'reason': 'Original generate_from_snapshot returns combined restore/continuation; no separate native duration producer'}
    result['recovery_vs_routing'] = 'Recovery comparison only; no routing benefit acceptance'
    return result
