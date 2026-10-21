"""A3: Lease-Aware Partial Restore Recovery — pure decision logic + patched bodies.

Loaded by the combined sitecustomize (v2 tracing + A3 glue) when A3_ENABLED=1.
Imported standalone by deterministic tests (no sglang dependency at import time).

Frozen-r3 anchors (all verified on node1 /tmp/frozen-r3):
  managers/cache_controller.py:1026 _page_get_zero_copy   (replaced, frozen body + A3)
  managers/cache_controller.py:1068 _page_transfer        (replaced, frozen body + bookkeeping)
  managers/cache_controller.py:1007 terminate_prefetch    (chained wrapper, outcome record)
  unified_radix_cache.py:2318  "HiCache prefetch success" (rewritten via logger proxy)

Contracts:
  - TP MIN all_reduce (_reduce_prefetch_ack) is NOT touched. Retry never adds
    or removes PrefetchAck events; ack count and per-batch positions are
    identical to frozen r3.
  - Outcome vocabulary: FULL / PARTIAL / FAILED, computed from
    requested_tokens vs completed_tokens. completed_tokens values themselves
    are never faked: 8192 stays 8192 — only the *status* becomes explicit.
  - Raw mooncake error codes: batch_get_into returns per-object ints
    (>0 = bytes read, negative = error). The pybind layer of
    mooncake-transfer-engine 0.3.12.post1 does NOT expose the C++ error-code
    enum, so codes are reported verbatim and classification is
    *behavior-based*: a fresh re-Query (batch_exists) decides permanent vs
    transient. Known codes from upstream tests / logs (-200
    NO_AVAILABLE_HANDLE family, -706 OBJECT_HAS_LEASE) are annotated but no
    invented enum mapping is used. This is a version-coupling boundary: if a
    future mooncake exposes typed errors, classify_failure() should switch
    to them.
  - Bounded retry: max_retry = 1 per PrefetchOperation (hardcoded — no tuning
    knob), LEASE-class transient only, and only after the cost gate:
        remaining_remote_cost_estimate
          = elapsed_restore + retry_cost_estimate + remaining_restore_cost_estimate
        remaining_recompute_cost_estimate
          = (remaining_tokens / requested_tokens) * recompute_estimate_s
    Retry re-walks the Query path: an explicit batch_exists re-check (permanent
    gate + fresh visibility) followed by batch_get_v1, whose internal
    mooncake Query re-acquires the lease before transfer. The retry re-reads
    only the failed page suffix of the failed batch (failed-subset retry;
    pages before the first failure are already fully written).
  - Permanent failure (re-Query says key missing) never retries: Safe Miss +
    token re-prefill happen naturally because the frozen _page_transfer
    all_success mechanism stops issuing IO for later batches.
  - Everything is fail-open: any exception inside A3 logic falls back to
    frozen-equivalent behavior.
"""
import os
import sys
import time
from queue import Empty

# ---------------------------------------------------------------------------
# configuration (read at glue time, but defaults resolved lazily for tests)
# ---------------------------------------------------------------------------

MAX_RETRY = 1  # hardcoded per A3 brief; do not tune

_A3_STATE = {
    "emit": None,            # wired by glue to sitecustomize._emit
    "logger": None,          # wired by glue to sglang cache_controller logger
    "outcome_registry": {},  # request_id -> last outcome record (pre-log)
    "recompute_estimate_s": None,
    "applied": {},           # patch status for exit report
}

STATE = None  # optional preloaded state for tests


def _emit(kind, **fields):
    fn = _A3_STATE["emit"]
    if fn is not None:
        try:
            fn(kind, **fields)
        except Exception:
            pass


def _recompute_estimate_s():
    if _A3_STATE["recompute_estimate_s"] is None:
        try:
            _A3_STATE["recompute_estimate_s"] = float(
                os.environ.get("A3_RECOMPUTE_ESTIMATE_S", "3.15")
            )
        except Exception:
            _A3_STATE["recompute_estimate_s"] = 3.15
    return _A3_STATE["recompute_estimate_s"]


def _logger():
    lg = _A3_STATE["logger"]
    if lg is not None:
        return lg
    mod = sys.modules.get("sglang.srt.managers.cache_controller")
    lg = getattr(mod, "logger", None)
    if lg is None:
        import logging

        lg = logging.getLogger("a3_lease_recovery")
    _A3_STATE["logger"] = lg
    return lg


# ---------------------------------------------------------------------------
# pure logic (unit-testable)
# ---------------------------------------------------------------------------


def outcome_status(requested_tokens, completed_tokens):
    """FULL / PARTIAL / FAILED per the A3.1 definition."""
    if requested_tokens is None or requested_tokens <= 0:
        return "FULL"  # no storage IO was requested (pure cache hit path)
    if completed_tokens >= requested_tokens:
        return "FULL"
    if completed_tokens <= 0:
        return "FAILED"
    return "PARTIAL"


def classify_failure(raw_results, key_multiplier, requery_result):
    """Behavior-based classification of a failed batch_get.

    raw_results: per-object ints from mooncake batch_get_into (may be None).
    key_multiplier: objects per page (2 for MHA k/v, 1 for MLA).
    requery_result: result of a fresh batch_exists over the failed page
      suffix: int = number of pages that exist from the front, or None if
      the re-Query itself errored.

    Returns dict(reason, retryable_class, raw_codes).
    """
    codes = []
    if raw_results is not None:
        try:
            codes = sorted({int(c) for c in raw_results if isinstance(c, int) and c < 0})
        except Exception:
            codes = []
    suffix_len_pages = None
    if requery_result is None:
        return {
            "reason": "permanent_requery_error",
            "retryable_class": False,
            "raw_codes": codes,
        }
    # requery_result is the number of existing pages counted from the front
    # of the failed suffix; the caller passes suffix length separately.
    return {
        "reason": "transient_exists_after_requery",
        "retryable_class": True,
        "raw_codes": codes,
    }


def cost_gate(
    elapsed_s,
    requested_tokens,
    tokens_done,
    bytes_per_token,
    retry_bytes,
    observed_bps,
    recompute_estimate_s=None,
):
    """A3.3 lease-aware retry gate.

    remaining_remote_cost_estimate =
        elapsed + retry_cost_estimate + remaining_restore_cost_estimate
      = elapsed + (remaining_bytes + retry_bytes) / observed_bps
    remaining_recompute_cost_estimate =
        (remaining_tokens / requested_tokens) * recompute_estimate_s

    Retry is allowed only when remaining_remote < remaining_recompute.
    observed_bps <= 0 (nothing measurable yet) makes remote estimate
    infinite -> gate rejects (never blind-retry).
    """
    if recompute_estimate_s is None:
        recompute_estimate_s = _recompute_estimate_s()
    requested_tokens = max(1, int(requested_tokens or 0))
    remaining_tokens = max(0, requested_tokens - int(tokens_done or 0))
    remaining_recompute = (remaining_tokens / requested_tokens) * recompute_estimate_s
    if observed_bps and observed_bps > 0 and bytes_per_token > 0:
        remaining_bytes = remaining_tokens * bytes_per_token
        remaining_remote = elapsed_s + (remaining_bytes + max(0, retry_bytes)) / observed_bps
    else:
        remaining_remote = float("inf")
    return {
        "remaining_remote_cost_estimate_s": round(remaining_remote, 6)
        if remaining_remote != float("inf")
        else None,
        "remaining_recompute_cost_estimate_s": round(remaining_recompute, 6),
        "retryable": bool(remaining_remote < remaining_recompute),
        "remaining_tokens": remaining_tokens,
    }


def build_outcome_record(operation, page_size):
    """Assemble the A3.1 record from operation bookkeeping.

    completed_tokens here is operation.completed_tokens — post-MIN-reduce on
    every rank (set by the ack drain), so all TP ranks build identical
    status; local_completed_tokens carries the rank-local pre-reduce value.
    """
    requested = getattr(operation, "_a3_requested_tokens", None)
    if requested is None:
        hashes = getattr(operation, "hash_value", None) or []
        requested = len(hashes) * page_size
    completed = int(getattr(operation, "completed_tokens", 0) or 0)
    local_pages = getattr(operation, "_a3_local_completed_pages", None)
    local_completed = int(local_pages * page_size) if local_pages is not None else None
    failure = getattr(operation, "_a3_failure", None) or {}
    t0 = getattr(operation, "start_time", None)
    elapsed_ms = int((time.monotonic() - t0) * 1000) if t0 is not None else None
    rid = getattr(operation, "request_id", None)
    rec = {
        "request_id": rid,
        "requested_tokens": requested,
        "completed_tokens": completed,
        "local_completed_tokens": local_completed,
        "restore_status": outcome_status(requested, completed),
        "failed_batch_index": failure.get("failed_batch_index"),
        "failure_reason": failure.get("failure_reason"),
        "raw_codes": failure.get("raw_codes"),
        "retryable": failure.get("retryable"),
        "retry_attempted": bool(failure.get("retry_attempted")),
        "retry_count": int(failure.get("retry_count", 0) or 0),
        "retry_result": failure.get("retry_result"),
        "fallback_reason": failure.get("fallback_reason"),
        "elapsed_restore_ms": elapsed_ms,
    }
    return rec


# ---------------------------------------------------------------------------
# frozen-r3 module helpers (resolved lazily; tests pre-inject via _A3_STATE)
# ---------------------------------------------------------------------------


def _cc_mod():
    mod = sys.modules.get("sglang.srt.managers.cache_controller")
    if mod is None:
        raise RuntimeError("cache_controller module not imported yet")
    return mod


def _page_size(self):
    return getattr(self.mem_pool_host, "page_size", 64)


def _bytes_per_token(self, backend):
    bpt = getattr(self, "_a3_bytes_per_token", None)
    if bpt is None:
        gb_per_page = getattr(backend, "gb_per_page", None)
        ps = _page_size(self)
        if gb_per_page and ps:
            bpt = gb_per_page * (1 << 30) / ps
        else:
            bpt = 98304.0  # 16K snapshot fallback: 1.617GB / 16448 tokens
        self._a3_bytes_per_token = bpt
    return bpt


def _key_multiplier(backend):
    if getattr(backend, "is_mla_backend", False):
        return 1
    km = 2
    if getattr(backend, "should_split_heads", False):
        km *= getattr(backend, "split_factor", 1)
    return km


def _contiguous_success(bools):
    inc = 0
    for b in bools:
        if b:
            inc += 1
        else:
            break
    return inc


# ---------------------------------------------------------------------------
# controlled fault injection (validation only; default OFF)
# ---------------------------------------------------------------------------


def _fault_injection_config():
    """Parse A3_FAULT_INJECT. Format (all parts optional, ';'-separated):
        batch_index=N     inject only when op._a3_batch_index == N
        max_requests=K    at most K injected ops per process (default 1)
        code=C            raw code to fabricate (default -706 OBJECT_HAS_LEASE)
    Warmup rids (starting with 'warmup') are never injected. Injection is
    per (request_id, batch_index) so all TP ranks fail the same batch
    symmetrically (MIN semantics stay intact). Default OFF (env unset).
    """
    cfg = os.environ.get("A3_FAULT_INJECT", "").strip()
    if not cfg:
        return None
    out = {"batch_index": None, "max_requests": 1, "code": -706}
    for part in cfg.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        try:
            if k == "batch_index":
                out["batch_index"] = int(v)
            elif k == "max_requests":
                out["max_requests"] = int(v)
            elif k == "code":
                out["code"] = int(v)
        except Exception:
            return None
    return out


def _fault_injection_code(operation):
    """Return the fabricated error code if this call should be injected."""
    try:
        cfg = _fault_injection_config()
        if cfg is None:
            return None
        rid = getattr(operation, "request_id", None) or ""
        if not rid or rid.startswith("warmup"):
            return None
        if cfg["batch_index"] is not None:
            if getattr(operation, "_a3_batch_index", None) != cfg["batch_index"]:
                return None
        key = (rid, cfg["batch_index"])
        seen = _A3_STATE.setdefault("_fault_injected", set())
        if key in seen:
            return None  # never re-inject the same request+batch (no loops)
        # count distinct rids injected so far
        rids = {k[0] for k in seen}
        if len(rids) >= cfg["max_requests"]:
            return None
        seen.add(key)
        return cfg["code"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# A3 replacement: HiCacheController._page_get_zero_copy
# frozen body (cache_controller.py:1026) + failure path
# ---------------------------------------------------------------------------


def a3_page_get_zero_copy(self, operation, hash_values, host_indices, extra_info=None) -> int:
    t0 = time.perf_counter()
    backend = self.storage_backend
    inj_code = _fault_injection_code(operation)
    if inj_code is not None:
        # controlled validation injection (A3_FAULT_INJECT): fabricate a
        # lease-class failure for this batch WITHOUT calling the real GET.
        # The frozen loop consumes the v1-level view (postprocessed BOOLS),
        # so the injected batch is all-False; the raw per-object codes go
        # into the stash exactly as the real impl wrapper would leave them.
        # Everything downstream (re-Query, cost gate, retry, fallback) runs
        # against the real master / real data path.
        n_inj = len(hash_values)
        results = [False] * n_inj
        try:
            backend._a3_last_raw = {
                "results": [inj_code] * n_inj,
                "n": None,
                "synthetic": True,
            }
        except Exception:
            pass
        _emit(
            "a3_fault_injected",
            request_id=getattr(operation, "request_id", None),
            batch_index=getattr(operation, "_a3_batch_index", None),
            pages=n_inj,
            code=inj_code,
        )
    else:
        results = backend.batch_get_v1(hash_values, host_indices, extra_info)
    inc = 0
    for i in range(len(hash_values)):
        if not results[i]:
            try:
                _logger().warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
            except Exception:
                pass
            break
        inc += 1
    inc = _a3_failure_path(self, operation, hash_values, host_indices, extra_info, results, inc)
    try:
        _emit("batch_page_get", dur_s=round(time.perf_counter() - t0, 6), hit_pages=inc)
    except Exception:
        pass
    return inc


def _a3_failure_path(self, operation, hash_values, host_indices, extra_info, results, inc):
    """Runs only when inc < len(hash_values). All exceptions fail-open."""
    n = len(hash_values)
    if inc >= n:
        return inc
    try:
        backend = self.storage_backend
        ps = _page_size(self)
        requested = getattr(operation, "_a3_requested_tokens", None) or n * ps
        batch_index = getattr(operation, "_a3_batch_index", None)
        batch_start = getattr(operation, "_a3_batch_start", 0)
        t0 = getattr(operation, "start_time", None)
        elapsed = (time.monotonic() - t0) if t0 is not None else 0.0

        # raw per-object codes stashed by the _get_batch_zero_copy_impl
        # chain wrapper (most recent call)
        raw = getattr(backend, "_a3_last_raw", None) or {}
        raw_results = raw.get("results")
        km = _key_multiplier(backend)
        suffix = hash_values[inc:]
        suffix_len = len(suffix)

        # fresh re-Query over the failed page suffix (permanent gate)
        requery = None
        try:
            requery = backend.batch_exists(suffix, extra_info=extra_info)
            if not isinstance(requery, int):
                requery = None
        except Exception:
            requery = None

        cls = classify_failure(raw_results, km, requery)
        failure = getattr(operation, "_a3_failure", None) or {}

        if requery is None:
            failure.update(
                {
                    "failure_reason": "permanent_requery_error",
                    "retryable": False,
                    "fallback_reason": "requery_failed_no_blind_retry",
                    "failed_batch_index": batch_index,
                }
            )
            operation._a3_failure = failure
            _emit(
                "a3_restore_failure",
                request_id=getattr(operation, "request_id", None),
                batch_index=batch_index,
                failed_page_index=inc,
                failure_reason="permanent_requery_error",
                retryable=False,
                fallback_reason="requery_failed_no_blind_retry",
                raw_codes=cls["raw_codes"],
            )
            return inc
        if requery < suffix_len:
            # some failed-suffix page is absent at the master -> OBJECT_NOT_FOUND
            # class -> permanent, no lease retry, Safe Miss / re-prefill.
            failure.update(
                {
                    "failure_reason": "permanent_object_missing",
                    "retryable": False,
                    "fallback_reason": "object_not_found_after_requery",
                    "failed_batch_index": batch_index,
                }
            )
            operation._a3_failure = failure
            _emit(
                "a3_restore_failure",
                request_id=getattr(operation, "request_id", None),
                batch_index=batch_index,
                failed_page_index=inc,
                requery_missing_at=requery,
                failure_reason="permanent_object_missing",
                retryable=False,
                fallback_reason="object_not_found_after_requery",
                raw_codes=cls["raw_codes"],
            )
            return inc

        # transient (lease/transfer class): lease-aware cost gate
        tokens_done = batch_start * ps + inc * ps
        bpt = _bytes_per_token(self, backend)
        bytes_moved = tokens_done * bpt
        observed_bps = (bytes_moved / elapsed) if elapsed > 0 else 0.0
        retry_bytes = suffix_len * ps * bpt
        gate = cost_gate(
            elapsed,
            requested,
            tokens_done,
            bpt,
            retry_bytes,
            observed_bps,
        )
        retry_count = int(failure.get("retry_count", 0) or 0)
        can_retry = (
            gate["retryable"]
            and retry_count < MAX_RETRY
            and not bool(failure.get("retry_attempted"))
        )
        failure.update(
            {
                "failure_reason": "transient_exists_after_requery",
                "raw_codes": cls["raw_codes"],
                "retryable": bool(gate["retryable"]),
                "failed_batch_index": batch_index,
            }
        )
        _emit(
            "a3_restore_failure",
            request_id=getattr(operation, "request_id", None),
            batch_index=batch_index,
            failed_page_index=inc,
            failure_reason="transient_exists_after_requery",
            raw_codes=cls["raw_codes"],
            elapsed_s=round(elapsed, 6),
            observed_bps=round(observed_bps, 1),
            remaining_remote_cost_estimate_s=gate["remaining_remote_cost_estimate_s"],
            remaining_recompute_cost_estimate_s=gate["remaining_recompute_cost_estimate_s"],
            retryable=gate["retryable"],
            retry_budget_left=MAX_RETRY - retry_count,
        )
        if not can_retry:
            failure["fallback_reason"] = (
                "retry_budget_exhausted"
                if retry_count >= MAX_RETRY
                else (
                    "retry_budget_exhausted"
                    if failure.get("retry_attempted")
                    else "cost_gate_rejected"
                )
            )
            operation._a3_failure = failure
            return inc

        # ---- bounded retry: failed-page suffix only, one shot --------------
        failure["retry_attempted"] = True
        failure["retry_count"] = retry_count + 1
        operation._a3_failure = failure
        rt = time.perf_counter()
        retry_results = backend.batch_get_v1(
            suffix, host_indices[inc * ps :], extra_info
        )
        inc2 = _contiguous_success(retry_results)
        retry_dur = time.perf_counter() - rt
        total = inc + inc2
        if inc2 >= suffix_len:
            retry_result = "success"
        elif inc2 > 0:
            retry_result = "partial"
        else:
            retry_result = "failed"
        failure["retry_result"] = retry_result
        failure["retry_suffix_pages"] = suffix_len
        failure["retry_hit_pages"] = inc2
        failure["retry_duration_s"] = round(retry_dur, 6)
        if inc2 < suffix_len:
            failure["fallback_reason"] = "retry_did_not_complete"
        operation._a3_failure = failure
        _emit(
            "a3_retry",
            request_id=getattr(operation, "request_id", None),
            batch_index=batch_index,
            retry_suffix_pages=suffix_len,
            retry_hit_pages=inc2,
            retry_result=retry_result,
            retry_duration_s=round(retry_dur, 6),
            retry_bytes=int(retry_bytes),
        )
        return total
    except Exception as exc:
        try:
            _emit("a3_internal_error", where="_a3_failure_path", error=repr(exc))
        except Exception:
            pass
        return inc


# ---------------------------------------------------------------------------
# A3 replacement: HiCacheController._page_transfer
# frozen body (cache_controller.py:1068) + per-batch bookkeeping
# ---------------------------------------------------------------------------


def a3_page_transfer(self, operation) -> int:
    t0 = time.perf_counter()
    try:
        cc = _cc_mod()
        STORAGE_BATCH_SIZE = cc.STORAGE_BATCH_SIZE
        PrefetchAck = cc.PrefetchAck
        PoolName = cc.PoolName
        HiCacheStorageExtraInfo = cc.HiCacheStorageExtraInfo
    except Exception:
        # tests inject state instead
        st = STATE or {}
        STORAGE_BATCH_SIZE = st.get("STORAGE_BATCH_SIZE", 128)
        PrefetchAck = st.get("PrefetchAck")
        PoolName = st.get("PoolName")
        HiCacheStorageExtraInfo = st.get("HiCacheStorageExtraInfo")

    try:
        ps = _page_size(self)
        if getattr(operation, "_a3_initialized", None) is None:
            operation._a3_initialized = True
            operation._a3_requested_tokens = len(operation.hash_value) * ps
        operation._a3_page_size = ps
    except Exception:
        pass

    # Transfer batch by batch
    prefix_keys = operation.prefix_keys
    kv_derived_transfers = [
        transfer
        for transfer in getattr(operation, "pool_transfers", None) or []
        if transfer.indices_from_pool == PoolName.KV
    ]
    all_success = True
    completed_pages = 0
    for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
        try:
            operation._a3_batch_index = i // STORAGE_BATCH_SIZE
            operation._a3_batch_start = i
        except Exception:
            pass
        # When an error is occurred, we should keep looping and produce the same number of
        # PrefetchAck as other ranks do, because prefetch_sync_thread (i.e. consumer of
        # prefetch_sync_queue) perform reduce on the results.  This is so tricky.
        if all_success and operation.is_terminated():
            all_success = False
        if all_success:
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]

            # Get one batch token, and update the completed_tokens if succeed
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)

            hit_pages = self._page_transfer_kv_batch(
                operation,
                batch_hashes,
                batch_host_indices,
                extra_info,
                kv_derived_transfers,
            )
            # Check termination
            if hit_pages != len(batch_hashes):
                all_success = False
            if prefix_keys is not None:
                prefix_keys = prefix_keys + batch_hashes
            completed_pages += hit_pages
        ack = PrefetchAck(
            rid=operation.request_id,
            completed_tokens=completed_pages * self.page_size,
            operation=operation,
        )
        self.prefetch_sync_queue.put(ack)
    try:
        operation._a3_local_completed_pages = completed_pages
    except Exception:
        pass
    r = completed_pages
    try:
        _emit("page_transfer", dur_s=round(time.perf_counter() - t0, 6),
              completed_pages=r)
    except Exception:
        pass
    return r


# ---------------------------------------------------------------------------
# glue helpers used by sitecustomize
# ---------------------------------------------------------------------------


def stash_raw_results(backend, results, key_strs=None):
    """Called by the chained _get_batch_zero_copy_impl wrapper."""
    try:
        backend._a3_last_raw = {
            "results": list(results) if results is not None else None,
            "n": len(key_strs) if key_strs is not None else None,
        }
    except Exception:
        pass


def record_outcome(operation):
    """Build + register + emit the A3.1 outcome record for one operation.

    Production completion path hook: called at the entry of the
    _handle_prefetch_result wrapper (BEFORE the frozen body logs the
    "HiCache prefetch ..." line, which the module-level LoggerProxy then
    rewrites using the registry). Operations that never ran a3_page_transfer
    (no _a3_page_size marker, e.g. pure cache-hit or PP paths) are skipped —
    their lines pass through the proxy verbatim with an a3_log_proxy_miss
    event, keeping the fail-open contract.

    completed_tokens on the operation is post-MIN-reduce on every rank, so
    all TP ranks register an identical restore_status (TP-consistent by
    construction); local_completed_tokens carries the rank-local value.
    """
    try:
        ps = getattr(operation, "_a3_page_size", None)
        if ps is None:
            return None  # not an A3-instrumented storage-IO operation
        rec = build_outcome_record(operation, ps)
        rid = rec.get("request_id") or "?"
        _A3_STATE["outcome_registry"][rid] = rec
        _emit("a3_restore_outcome", **rec)
        return rec
    except Exception as exc:
        try:
            _emit("a3_internal_error", where="record_outcome", error=repr(exc))
        except Exception:
            pass
        return None


def wrap_terminate_prefetch(orig, page_size_getter=None):
    """Chain over the (possibly v2-wrapped) terminate_prefetch.

    The frozen terminate path is NOT on the normal completion route in
    frozen r3 (A2 + A3 run1 traces: 0 terminate calls, completion goes via
    _handle_prefetch_result); this wrapper remains for the timeout /
    early-termination route and registers the same record as record_outcome.
    """

    def wrapper(self, operation):
        r = orig(self, operation)
        try:
            if getattr(operation, "_a3_page_size", None) is None:
                operation._a3_page_size = _page_size(self)
            record_outcome(operation)
        except Exception as exc:
            try:
                _emit("a3_internal_error", where="terminate_prefetch", error=repr(exc))
            except Exception:
                pass
        return r

    return wrapper


_OLD_PREFETCH_FMT = (
    "HiCache prefetch %s req=%s completed=%d matched=%d loaded=%d occupied=%d"
)
_NEW_PREFETCH_FMT = (
    "HiCache prefetch status=%s req=%s completed=%d requested=%d matched=%d "
    "loaded=%d occupied=%d failure_reason=%s retryable=%s retry_attempted=%s "
    "retry_result=%s elapsed_restore_ms=%s"
)


def make_logger_proxy(inner):
    """Rewrite the frozen 'prefetch success' line into a status-aware one.

    Frozen line args: ("success"/"dropped", rid, completed, matched,
    loaded, occupied). The outcome registry (filled by the
    terminate_prefetch wrapper) supplies requested_tokens and the
    failure/retry fields. Registry miss -> line passes through unchanged
    (fail-open) except that 'success' is downgraded to 'full' only when
    requested==completed is *not* provable — i.e. pass through verbatim.
    """

    class _A3LoggerProxy:
        def info(self, fmt, *args, **kwargs):
            if fmt == _OLD_PREFETCH_FMT and len(args) == 6:
                try:
                    marker, rid, completed, matched, loaded, occupied = args
                    rec = _A3_STATE["outcome_registry"].get(rid)
                    if rec is None:
                        # registry miss (e.g. PP path): fail-open verbatim
                        # passthrough, but surface the miss so analysis can
                        # catch any remaining silent-success lines.
                        _emit("a3_log_proxy_miss", rid=rid, completed=completed,
                              matched=matched, marker=marker)
                        inner.info(fmt, *args, **kwargs)
                        return
                    requested = rec.get("requested_tokens")
                    if requested is None:
                        requested = completed  # unknown scope -> cannot
                        # claim FULL; treat completed as scope (FULL) — this
                        # branch only fires on cache-hit paths that never
                        # touched storage IO, where completed IS the scope.
                    status = rec.get("restore_status") or outcome_status(
                        requested, completed
                    )
                    if marker == "dropped":
                        status = status + "/host_dropped"
                    new_args = (
                        status,
                        rid,
                        completed,
                        requested,
                        matched,
                        loaded,
                        occupied,
                        rec.get("failure_reason") or "none",
                        rec.get("retryable"),
                        rec.get("retry_attempted"),
                        rec.get("retry_result") or "none",
                        rec.get("elapsed_restore_ms"),
                    )
                    inner.info(_NEW_PREFETCH_FMT, *new_args)
                    return
                except Exception:
                    pass
            inner.info(fmt, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(inner, name)

    return _A3LoggerProxy()
