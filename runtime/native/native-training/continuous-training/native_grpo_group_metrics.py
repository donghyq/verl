"""Read-only first-dispatch GRPO timings from the native ReplayBuffer snapshot."""
import json
import math
import os
from pathlib import Path
import time


class NativeGRPOGroupMetrics:
    def __init__(self, driver, *, clock=time.monotonic):
        self.driver, self.clock = driver, clock
        self.pid = os.getpid()
        self.started = None
        self.groups = {}
        self.dispatches = []
        self.polls = 0
        self.errors = []
        self.finalized_at = None
        result_path = driver.options.result_path
        self.path = Path(result_path).with_name("native-grpo-group-metrics.json") if result_path else None

    def dispatch(self, prompts):
        now = self.clock()
        uids = sorted(self.driver._prompt_uids(prompts))
        self.dispatches.append({"dispatch_index": len(self.dispatches), "uids": uids,
                                "offset_s": 0.0 if self.started is None else now - self.started})
        # A single native admission is the declared eight-group sampling window.
        # Retain later dispatch UIDs explicitly; never mix refill windows silently.
        if self.started is None:
            self.started = now
            self.groups = {uid: {"dispatch_monotonic": now, "completed_monotonic": None,
                                 "reason": "NO_COMPLETE_REPLAY_METADATA_OBSERVED", "failed_seen": False}
                           for uid in uids}
        self.persist()

    def observe(self, replay):
        self.polls += 1
        now = self.clock()
        if os.getpid() != self.pid:
            raise ValueError("GRPO timing must remain in the driver process")
        # Actual ReplayBuffer fields populated by _sync_metadata_from_transfer_queue:
        # partitions['train'] is {trajectory_key: tag}; finished/failure_keys hold UIDs.
        partition = replay.partitions.get("train", {})
        finished = replay.finished_keys.get("train", set())
        failed = replay.failure_keys.get("train", set())
        for uid, row in self.groups.items():
            if row["completed_monotonic"] is not None:
                continue
            if uid in failed:
                row.update(failed_seen=True, reason="NATIVE_PROMPT_FAILURE")
                row.setdefault("failure_monotonic", now)
            if row["failed_seen"] or uid not in finished:
                continue
            expected = {f"{uid}_{index}_0" for index in range(8)}
            # Same UID interpretation as native ReplayBuffer._materialize_batch.
            members = {key: tag for key, tag in partition.items() if key.split("_")[0] == uid}
            if set(members) != expected:
                row["reason"] = "MEMBER_KEYS_NOT_EXACT_NATIVE_EIGHT"
                continue
            legal = True
            for tag in members.values():
                lengths = [tag.get(name) for name in ("prompt_len", "response_len", "seq_len")]
                versions = [tag.get(name) for name in ("global_steps", "min_global_steps", "max_global_steps")]
                if (tag.get("status") != "success" or tag.get("is_prompt", False)
                        or any(type(value) is not int or value < 0 for value in lengths + versions)
                        or lengths[0] + lengths[1] != lengths[2] or lengths[2] <= 0
                        or versions[1] > versions[2]):
                    legal = False
                    break
            if not legal:
                row["reason"] = "NATIVE_SUCCESS_LENGTH_OR_VERSION_TAGS_INVALID"
                continue
            row.update(completed_monotonic=now, reason=None, member_keys=sorted(expected))
        self.persist()

    def report(self):
        stop = self.finalized_at if self.finalized_at is not None else self.clock()
        completed = [row for row in self.groups.values() if row["completed_monotonic"] is not None]
        durations = sorted(row["completed_monotonic"] - row["dispatch_monotonic"] for row in completed)
        all_complete = bool(self.groups) and len(completed) == len(self.groups)
        all_settled = bool(self.groups) and all(row["completed_monotonic"] is not None or row["failed_seen"]
                                               for row in self.groups.values())
        end = (max(row["completed_monotonic"] if row["completed_monotonic"] is not None
                   else row["failure_monotonic"] for row in self.groups.values()) if all_settled else stop)
        elapsed = end - self.started if self.started is not None else None
        valid_window = self.started is not None and len(self.groups) == 8 and elapsed is not None and elapsed > 0
        rate = len(completed) / (8 * elapsed / 3600) if valid_window and completed else None
        missing_reason = ("NO_NATIVE_DISPATCH_OBSERVED" if self.started is None else
                          "INITIAL_DISPATCH_IS_NOT_EIGHT_GROUPS" if len(self.groups) != 8 else
                          "NO_VALID_COMPLETE_GROUPS_OBSERVED" if not completed else
                          "WINDOW_DURATION_NOT_POSITIVE" if not valid_window else None)
        return {"schema": "native-first-dispatch-grpo-group-metrics-v1",
            "status": "FINALIZED" if self.finalized_at is not None else "OBSERVING",
            "driver_pid": self.pid, "clock": "same_process_monotonic",
            "scope": "first_native_generate_sequences_dispatch; later_dispatches_listed_separately",
            "duration_kind": "dispatch_to_first_complete_ReplayBuffer_metadata_observation",
            "completion_poll_delay_included": True,
            "expected_prompt_groups": 8, "registered_prompt_groups": len(self.groups),
            "legal_complete_groups": len(completed), "distribution_sample_count": len(durations),
            "p50_seconds": durations[math.ceil(.50 * len(durations)) - 1] if durations else None,
            "p95_seconds": durations[math.ceil(.95 * len(durations)) - 1] if durations else None,
            "quantile_method": "nearest_rank", "window_seconds": elapsed if valid_window else None,
            "window_end": "last_complete_or_failed_group_observation" if all_settled else "current_or_finalize_observation",
            "allocated_gpu_count": 8, "gpu_count_basis": "fixed native 4-trainer plus 4-rollout allocation; includes idle allocated GPUs",
            "window_gpu_hours": 8 * elapsed / 3600 if valid_window else None,
            "complete_groups_per_gpu_hour": rate, "missing_metric_reason": missing_reason,
            "all_eight_complete": all_complete and len(self.groups) == 8,
            "failed_groups": sum(row["failed_seen"] for row in self.groups.values()),
            "unsettled_groups": sum(row["completed_monotonic"] is None and not row["failed_seen"]
                                    for row in self.groups.values()),
            "groups": [{"uid": uid, "completion_seconds": row["completed_monotonic"] - row["dispatch_monotonic"]
                        if row["completed_monotonic"] is not None else None,
                        "dispatch_offset_s": row["dispatch_monotonic"] - self.started,
                        "member_keys": row.get("member_keys"), "missing_reason": row["reason"]}
                       for uid, row in sorted(self.groups.items())],
            "dispatches": self.dispatches, "native_metadata_sync_polls": self.polls,
            "observation_errors": self.errors, "token_recompute_count": None,
            "token_recompute_missing_reason": "NO_RELIABLE_PER_REQUEST_RECOMPUTE_COUNTER; cache hits are not substituted",
            "learning_signal_or_reward_success_implied": False,
            "ab_comparison_or_significance_claim": False}

    def persist(self):
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".partial")
            temporary.write_text(json.dumps(self.report(), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            temporary.replace(self.path)
        except Exception as error:
            self.errors.append({"stage": "persist", "error_type": type(error).__name__})

    def finalize(self):
        self.finalized_at = self.clock()
        self.persist()
        return self.report()


def install_group_metrics(driver):
    observer = NativeGRPOGroupMetrics(driver)
    original_generate = driver.manager.generate_sequences
    def generate_with_observation(prompts):
        try:
            observer.dispatch(prompts)
        except Exception as error:
            observer.errors.append({"stage": "dispatch_observation", "error_type": type(error).__name__})
        try:
            return original_generate(prompts)
        except BaseException as error:
            observer.errors.append({"stage": "native_generate_sequences", "error_type": type(error).__name__})
            observer.finalize()
            raise
    driver.manager.generate_sequences = generate_with_observation
    replay = driver.trainer.replay_buffer
    original_sync = replay._sync_metadata_from_transfer_queue
    def sync_with_observation(*args, **kwargs):
        try:
            result = original_sync(*args, **kwargs)
        except BaseException as error:
            observer.errors.append({"stage": "native_metadata_sync", "error_type": type(error).__name__})
            observer.finalize()
            raise
        try:
            observer.observe(replay)
        except Exception as error:
            observer.errors.append({"stage": "complete_group_observation", "error_type": type(error).__name__})
            observer.persist()
        return result
    replay._sync_metadata_from_transfer_queue = sync_with_observation
    observer.persist()
    return observer


"""Read-only publication windows and explicitly sourced token counters.

Cache affinity is not evidence of engine KV validity or reduced compute work.
"""
import threading
import time


class RoutingVersionMetrics:
    def __init__(self, driver, *, clock=time.monotonic):
        self.driver, self.clock = driver, clock
        self.started = clock()
        self.windows, self.requests, self.errors = [], {}, []
        self.lock = threading.Lock()

    def begin(self, version, gateway_enabled):
        with self.lock:
            index = len(self.windows)
            self.windows.append({'index': index, 'requested_policy_version': version,
                'gateway_control_enabled': gateway_enabled, 'started_offset_s': self.clock()-self.started,
                'ended_offset_s': None, 'outcome': 'IN_PROGRESS', 'routing_prefix_purge_verified': None})
            return index

    def end(self, index, outcome, control, error=None):
        with self.lock:
            row = self.windows[index]
            row.update(ended_offset_s=self.clock()-self.started, outcome=outcome)
            row['elapsed_s'] = row['ended_offset_s']-row['started_offset_s']
            # These are existing lifecycle receipts, not proof of SMG's tree contents.
            row['registration_observations'] = list(getattr(control, 'registration_observations', [])) if control else []
            row['removal_observations'] = list(getattr(control, 'removal_observations', [])) if control else []
            if error:
                row['error_type'] = type(error).__name__

    def engine_result(self, *, request_id, execution_generation, native_meta_info,
                      source, scope, actual_generated_tokens=None, actual_recomputed_tokens=None):
        """Call with a terminal native engine receipt, never with estimated hits.

        The caller must supply actual compute counters only when the producer
        defines and emits them. Missing counters remain missing, including zero.
        """
        if not request_id or not source or scope not in {'business', 'health', 'unknown'}:
            raise ValueError('request/source and an explicit workload scope are required')
        if type(execution_generation) is not int or execution_generation < 0 or not isinstance(native_meta_info, dict):
            raise ValueError('actual execution identity and raw native metadata are required')
        if native_meta_info.get('id') != request_id:
            raise ValueError('native metadata request identity differs')
        for value in (actual_generated_tokens, actual_recomputed_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('actual counters must be nonnegative integers or absent')
        for name, value in [('actual_generated_tokens', actual_generated_tokens), ('actual_recomputed_tokens', actual_recomputed_tokens)]:
            if value is not None and (type(native_meta_info.get(name)) is not int or native_meta_info[name] != value):
                raise ValueError('supplied compute counter has no matching explicit native metadata field')
        key = (request_id, execution_generation)
        row = {'request_id': request_id, 'execution_generation': execution_generation,
            'scope': scope, 'source': source, 'native_meta_info': dict(native_meta_info),
            'actual_generated_tokens': actual_generated_tokens,
            'actual_recomputed_tokens': actual_recomputed_tokens,
            'generated_tokens_missing_reason': None if actual_generated_tokens is not None else 'NO_EXPLICIT_NATIVE_COMPUTE_COUNTER',
            'recomputed_tokens_missing_reason': None if actual_recomputed_tokens is not None else 'NO_EXPLICIT_NATIVE_COMPUTE_COUNTER'}
        with self.lock:
            if key in self.requests and self.requests[key] != row:
                raise ValueError('one terminal execution receipt changed; refusing double counting')
            self.requests[key] = row

    def report(self):
        with self.lock:
            rows = list(self.requests.values())
            windows = [dict(row) for row in self.windows]
        business = [row for row in rows if row['scope']=='business']
        def counter(name):
            values = [row[name] for row in business]
            return sum(values) if values and all(v is not None for v in values) else None
        groups = getattr(self.driver, '_group_metrics', None)
        group_report = groups.report() if groups else None
        return {'schema': 'native-routing-version-window-metrics-v1',
            'publication_windows': windows, 'engine_execution_receipts': rows,
            'actual_generated_tokens': counter('actual_generated_tokens'),
            'actual_recomputed_tokens': counter('actual_recomputed_tokens'),
            'counter_scope': 'only supplied business execution receipts; health excluded',
            'counter_coverage': 'OBSERVED_EXECUTIONS_ONLY; not all business work unless driver proves coverage',
            'token_counter_missing_reason': None if business and all(row['actual_generated_tokens'] is not None and row['actual_recomputed_tokens'] is not None for row in business) else 'NO_COMPLETE_EXPLICIT_NATIVE_COUNTER_COVERAGE',
            'original_group_metrics': group_report,
            'group_metrics_preserved_without_reclassification': True,
            'cache_affinity_or_hit_implies_compute_saving': False,
            'smg_prefix_tree_reset_runtime_verified': None,
            'observation_errors': list(self.errors)}


def install_routing_window_metrics(driver):
    """Observe original CheckpointEngineManager._publish_weights exactly once.

    This adds no network request and changes no route, scheduling or cache flag.
    """
    manager = driver.trainer.checkpoint_manager
    if getattr(manager, '_routing_version_metrics_observer', None) is not None:
        raise ValueError('routing version metrics already installed')
    observer = RoutingVersionMetrics(driver)
    original = manager._publish_weights
    async def observed(*args, **kwargs):
        version = kwargs.get('global_steps', args[0] if args else None)
        control = getattr(manager, 'gateway_control', None)
        index = None
        try:
            index = observer.begin(version, control is not None)
        except Exception as error:
            observer.errors.append({'stage': 'publication_begin_observation', 'error_type': type(error).__name__})
        def finish(outcome, cause=None):
            try:
                if index is not None:
                    observer.end(index, outcome, control, cause)
            except Exception as error:
                observer.errors.append({'stage': 'publication_end_observation', 'error_type': type(error).__name__})
        try:
            result = await original(*args, **kwargs)
        except BaseException as error:
            finish('FAILED_NOT_A_COMPLETED_PUBLICATION_WINDOW', error)
            raise
        finish('ORIGINAL_PUBLICATION_RETURNED')
        return result
    manager._publish_weights = observed
    manager._routing_version_metrics_observer = observer
    return observer
