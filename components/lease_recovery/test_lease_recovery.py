"""A3 deterministic tests — Cases 1-6 per the A3 brief, no sglang required.

Run: python3 -m unittest test_lease_recovery -v
"""
import os
import queue
import time
import types
import unittest

import lease_recovery as lr

PAGE_SIZE = 64
GB_PER_PAGE = 0.00586  # ~6.29MB/page -> ~98.3KB/token, matches 16K snapshot
RECOMPUTE_ESTIMATE_S = 3.15
REQUESTED_TOKENS = 16448  # 257 pages


def fresh_state():
    lr._A3_STATE["emit"] = None
    lr._A3_STATE["logger"] = None
    lr._A3_STATE["outcome_registry"] = {}
    lr._A3_STATE["recompute_estimate_s"] = RECOMPUTE_ESTIMATE_S
    lr._A3_STATE.pop("_fault_injected", None)
    os.environ.pop("A3_FAULT_INJECT", None)


class Collector:
    def __init__(self):
        self.events = []

    def __call__(self, kind, **fields):
        self.events.append((kind, fields))

    def kinds(self):
        return [k for k, _ in self.events]


class FakeBackend:
    """Scripted mooncake wrapper. scenarios: list of dict per batch_get_v1 call:
    ok_pages  -> bool list length n (True = page retrieved)
    raw       -> per-object raw codes list (len n * km)
    exists_at -> requery result: int (pages existing from front) or 'raise'
    retry_ok_pages -> bool list for the retry call (suffix re-GET)
    """

    def __init__(self, n_pages, scenarios):
        self.scenarios = list(scenarios)
        self.get_calls = []
        self.exists_calls = []
        self.gb_per_page = GB_PER_PAGE
        self.is_mla_backend = False
        self.should_split_heads = False
        self.split_factor = 1
        self._current_scenario = {}
        self._pending_retry = None

    def batch_get_v1(self, keys, host_indices, extra_info=None):
        self.get_calls.append(list(keys))
        if self._pending_retry is not None:
            sc = self._pending_retry
            self._pending_retry = None
            bools = sc["retry_ok_pages"]
            raw = None
        else:
            sc = self.scenarios.pop(0)
            self._current_scenario = sc
            bools = sc["ok_pages"]
            raw = sc.get("raw")
            if "retry_ok_pages" in sc:
                self._pending_retry = sc
        self._a3_last_raw = {"results": raw, "n": len(bools) * 2}
        return list(bools)

    def batch_exists(self, keys, extra_info=None):
        self.exists_calls.append(list(keys))
        sc = self._current_scenario
        if sc.get("raise") == "requery":
            raise RuntimeError("requery boom")
        return sc["exists_at"]


def make_self(backend, page_size=PAGE_SIZE):
    return types.SimpleNamespace(
        storage_backend=backend,
        mem_pool_host=types.SimpleNamespace(page_size=page_size),
    )


def make_op(rid="req-1", n_pages=257, start_age_s=0.0, requested_tokens=None):
    op = types.SimpleNamespace(
        request_id=rid,
        hash_value=["h%04d" % i for i in range(n_pages)],
        start_time=time.monotonic() - start_age_s,
        completed_tokens=0,
    )
    if requested_tokens is not None:
        op._a3_requested_tokens = requested_tokens
    return op


def call_page_get(fake_self, op, n_pages, backend):
    host = list(range(n_pages * PAGE_SIZE))
    return lr.a3_page_get_zero_copy(fake_self, op, op.hash_value[:n_pages], host)


class Case1FullSuccess(unittest.TestCase):
    def test_full_success_no_retry(self):
        fresh_state()
        col = Collector()
        lr._A3_STATE["emit"] = col
        n = 128
        backend = FakeBackend(n, [{"ok_pages": [True] * n}])
        fake_self = make_self(backend)
        op = make_op()
        op._a3_batch_index = 0
        op._a3_batch_start = 0
        inc = call_page_get(fake_self, op, n, backend)
        self.assertEqual(inc, n)
        self.assertNotIn("a3_restore_failure", col.kinds())
        # ack drain would set completed_tokens to the full request
        op.completed_tokens = REQUESTED_TOKENS
        rec = lr.build_outcome_record(op, PAGE_SIZE)
        self.assertEqual(rec["restore_status"], "FULL")
        self.assertEqual(rec["retry_count"], 0)
        self.assertEqual(rec["retry_attempted"], False)
        self.assertEqual(rec["failure_reason"], None)


class Case2LeaseExpiredRetryOnce(unittest.TestCase):
    def test_lease_transient_retry_success(self):
        fresh_state()
        col = Collector()
        lr._A3_STATE["emit"] = col
        n = 128
        # batch 1: first 64 pages ok, page 64 fails with LEASE-class raw
        # codes; requery says all suffix pages exist; retry re-reads the
        # 64-page failed suffix and completes. Gate math (elapsed 0.2s,
        # observed ~2GB/s): remote ~1.0s < recompute ~2.4s -> retryable.
        backend = FakeBackend(
            n,
            [
                {
                    "ok_pages": [True] * 64 + [False] * 64,
                    "raw": [-706] * (2 * 64) + [1] * (2 * 64),
                    "exists_at": 64,
                    "retry_ok_pages": [True] * 64,
                }
            ],
        )
        op = make_op(start_age_s=0.2)
        op._a3_batch_index = 0
        op._a3_batch_start = 0
        op._a3_requested_tokens = REQUESTED_TOKENS
        fake_self = make_self(backend)
        inc = call_page_get(fake_self, op, n, backend)
        # retry re-reads only the failed suffix (pages 64..127 of batch 1)
        self.assertEqual(inc, n)
        self.assertEqual(len(backend.get_calls), 2)
        self.assertEqual(backend.get_calls[1], op.hash_value[64:128])
        self.assertEqual(len(backend.exists_calls), 1)
        self.assertEqual(backend.exists_calls[0], op.hash_value[64:128])
        kinds = col.kinds()
        self.assertIn("a3_restore_failure", kinds)
        self.assertIn("a3_retry", kinds)
        retry_ev = [f for k, f in col.events if k == "a3_retry"][0]
        self.assertEqual(retry_ev["retry_result"], "success")
        self.assertEqual(retry_ev["retry_suffix_pages"], 64)
        failure = op._a3_failure
        self.assertEqual(failure["failure_reason"], "transient_exists_after_requery")
        self.assertTrue(failure["retry_attempted"])
        self.assertEqual(failure["retry_count"], 1)
        self.assertEqual(failure["retry_result"], "success")
        self.assertNotIn("fallback_reason", failure)

    def test_retry_partial_then_later_batches_stop_via_frozen_mechanism(self):
        """Retry that partially completes is reported, not silently success."""
        fresh_state()
        lr._A3_STATE["emit"] = Collector()
        n = 128
        backend = FakeBackend(
            n,
            [
                {
                    "ok_pages": [True] * 64 + [False] * 64,
                    "raw": [-200] * (2 * 64),
                    "exists_at": 64,
                    "retry_ok_pages": [True] * 10 + [False] * 54,
                }
            ],
        )
        op = make_op(start_age_s=0.2)
        op._a3_batch_index = 0
        op._a3_batch_start = 0
        op._a3_requested_tokens = REQUESTED_TOKENS
        fake_self = make_self(backend)
        inc = call_page_get(fake_self, op, n, backend)
        self.assertEqual(inc, 74)
        self.assertEqual(op._a3_failure["retry_result"], "partial")
        self.assertEqual(op._a3_failure["retry_hit_pages"], 10)
        self.assertEqual(op._a3_failure["fallback_reason"], "retry_did_not_complete")


class Case3CostGateSkipsRetry(unittest.TestCase):
    def test_elapsed_over_budget_no_retry(self):
        fresh_state()
        col = Collector()
        lr._A3_STATE["emit"] = col
        n = 128
        backend = FakeBackend(
            n,
            [
                {
                    "ok_pages": [True] * 100 + [False] * 28,
                    "raw": [-706] * (2 * n),
                    "exists_at": n,
                }
            ],
        )
        # elapsed ~5s with only ~629MB moved -> observed ~126MB/s; remaining
        # remote estimate far exceeds remaining recompute estimate.
        op = make_op(start_age_s=5.0)
        op._a3_batch_index = 0
        op._a3_batch_start = 0
        op._a3_requested_tokens = REQUESTED_TOKENS
        fake_self = make_self(backend)
        inc = call_page_get(fake_self, op, n, backend)
        self.assertEqual(inc, 100)
        self.assertEqual(len(backend.get_calls), 1)  # no retry GET
        failure = op._a3_failure
        self.assertEqual(failure["fallback_reason"], "cost_gate_rejected")
        self.assertFalse(failure["retryable"])
        ev = [f for k, f in col.events if k == "a3_restore_failure"][0]
        self.assertFalse(ev["retryable"])


class Case4PermanentNoRetry(unittest.TestCase):
    def test_object_not_found_after_requery(self):
        fresh_state()
        col = Collector()
        lr._A3_STATE["emit"] = col
        n = 128
        backend = FakeBackend(
            n,
            [
                {
                    "ok_pages": [False] * n,
                    "raw": [-2] * (2 * n),
                    "exists_at": 5,  # only 5 of 128 suffix pages still exist
                }
            ],
        )
        op = make_op(start_age_s=0.5)
        op._a3_batch_index = 0
        op._a3_batch_start = 0
        op._a3_requested_tokens = REQUESTED_TOKENS
        fake_self = make_self(backend)
        inc = call_page_get(fake_self, op, n, backend)
        self.assertEqual(inc, 0)
        self.assertEqual(len(backend.get_calls), 1)  # no retry GET
        failure = op._a3_failure
        self.assertEqual(failure["failure_reason"], "permanent_object_missing")
        self.assertEqual(failure["fallback_reason"], "object_not_found_after_requery")
        self.assertFalse(failure["retryable"])
        ev = [f for k, f in col.events if k == "a3_restore_failure"][0]
        self.assertEqual(ev["requery_missing_at"], 5)


class Case5TPMinConsistency(unittest.TestCase):
    def test_min_clamped_status_identical_across_ranks(self):
        fresh_state()
        # rank A completed all 257 pages locally; rank B only 128.
        # MIN all_reduce (untouched, simulated) clamps both to 8192.
        completed_after_min = 128 * PAGE_SIZE  # 8192
        op_a = make_op()
        op_a.completed_tokens = completed_after_min
        op_a._a3_local_completed_pages = 257
        op_b = make_op()
        op_b.completed_tokens = completed_after_min
        op_b._a3_local_completed_pages = 128
        op_b._a3_failure = {
            "failure_reason": "transient_exists_after_requery",
            "retryable": True,
            "retry_attempted": False,
            "retry_count": 0,
            "failed_batch_index": 1,
        }
        rec_a = lr.build_outcome_record(op_a, PAGE_SIZE)
        rec_b = lr.build_outcome_record(op_b, PAGE_SIZE)
        self.assertEqual(rec_a["restore_status"], "PARTIAL")
        self.assertEqual(rec_b["restore_status"], "PARTIAL")
        self.assertEqual(rec_a["completed_tokens"], rec_b["completed_tokens"])
        self.assertEqual(rec_a["restore_status"], rec_b["restore_status"])
        # local asymmetry is preserved for diagnosis
        self.assertEqual(rec_a["local_completed_tokens"], 16448)
        self.assertEqual(rec_b["local_completed_tokens"], 8192)
        # failure marker belongs to the failing rank only
        self.assertIsNone(rec_a["failure_reason"])
        self.assertEqual(rec_b["failure_reason"], "transient_exists_after_requery")
        # boundary cases
        self.assertEqual(lr.outcome_status(16448, 0), "FAILED")
        self.assertEqual(lr.outcome_status(16448, 16448), "FULL")
        self.assertEqual(lr.outcome_status(16448, 8192), "PARTIAL")

    def test_outcome_registry_and_proxy_line(self):
        fresh_state()
        inner = []
        proxy = lr.make_logger_proxy(
            types.SimpleNamespace(info=lambda fmt, *a: inner.append((fmt, a)))
        )
        # registry miss -> verbatim passthrough (fail-open)
        proxy.info(lr._OLD_PREFETCH_FMT, "success", "rx", 8192, 8192, 0, 5)
        self.assertEqual(inner[0][0], lr._OLD_PREFETCH_FMT)  # verbatim passthrough
        # registry hit -> rewritten line, no bare 'success'
        lr._A3_STATE["outcome_registry"]["rx"] = {
            "requested_tokens": 16448,
            "restore_status": "PARTIAL",
            "failure_reason": "transient_exists_after_requery",
            "retryable": True,
            "retry_attempted": False,
            "retry_result": "none",
            "elapsed_restore_ms": 16900,
        }
        proxy.info(lr._OLD_PREFETCH_FMT, "success", "rx", 8192, 8192, 0, 5)
        fmt, args = inner[1]
        self.assertEqual(fmt, lr._NEW_PREFETCH_FMT)
        self.assertEqual(args[0], "PARTIAL")
        self.assertEqual(args[3], 16448)
        self.assertNotIn("success", fmt)
        # non-matching format passes through untouched
        proxy.info("other %s", "x")
        self.assertEqual(inner[2], ("other %s", ("x",)))

    def test_record_outcome_entry_hook_end_to_end(self):
        """Run1 regression: the production completion path calls
        _handle_prefetch_result, NOT terminate_prefetch. record_outcome
        (called at hpr entry) must fill the registry so the proxy rewrites
        the frozen line; ops without the _a3_page_size marker are skipped.
        """
        fresh_state()
        col = Collector()
        lr._A3_STATE["emit"] = col
        # rank A: full restore (257/257 pages), post-MIN completed = full
        op = make_op()
        op._a3_page_size = PAGE_SIZE
        op._a3_local_completed_pages = 257
        op.completed_tokens = REQUESTED_TOKENS
        rec = lr.record_outcome(op)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["restore_status"], "FULL")
        self.assertIn(op.request_id, lr._A3_STATE["outcome_registry"])
        self.assertIn("a3_restore_outcome", col.kinds())
        # rank B same rid, MIN-clamped partial
        op_b = make_op(rid=op.request_id)
        op_b._a3_page_size = PAGE_SIZE
        op_b._a3_local_completed_pages = 128
        op_b.completed_tokens = 8192
        rec_b = lr.record_outcome(op_b)
        self.assertEqual(rec_b["restore_status"], "PARTIAL")
        # proxy now rewrites the frozen line using the registry entry
        inner = []
        proxy = lr.make_logger_proxy(
            types.SimpleNamespace(info=lambda fmt, *a: inner.append((fmt, a)))
        )
        proxy.info(lr._OLD_PREFETCH_FMT, "success", op.request_id, 8192,
                   REQUESTED_TOKENS, 0, 5)
        self.assertEqual(inner[0][0], lr._NEW_PREFETCH_FMT)
        self.assertEqual(inner[0][1][0], "PARTIAL")
        # op WITHOUT the _a3_page_size marker (cache-hit path): skipped,
        # registry untouched, proxy line passes through verbatim (miss)
        op_c = make_op(rid="cache-hit-1")
        op_c.completed_tokens = 1234
        self.assertIsNone(lr.record_outcome(op_c))
        self.assertNotIn("cache-hit-1", lr._A3_STATE["outcome_registry"])
        proxy.info(lr._OLD_PREFETCH_FMT, "success", "cache-hit-1", 1234,
                   1234, 0, 5)
        self.assertEqual(inner[1][0], lr._OLD_PREFETCH_FMT)  # verbatim
        self.assertIn("a3_log_proxy_miss", col.kinds())


class Case7FaultInjection(unittest.TestCase):
    """A3_FAULT_INJECT (validation only, default OFF):"""

    def test_injection_drives_failure_then_retry_path(self):
        fresh_state()
        col = Collector()
        lr._A3_STATE["emit"] = col
        os.environ["A3_FAULT_INJECT"] = "batch_index=1;max_requests=1"
        n = 128
        # injection at batch 1 AFTER a real successful batch 0 (mirrors the
        # run2 validation config): elapsed>0 and tokens_done>0 so the cost
        # gate is measurable and passes; the retry then re-GETs the failed
        # suffix through the real backend path.
        backend = FakeBackend(n, [{"ok_pages": [True] * n},
                                   {"ok_pages": [True] * n}])
        backend._current_scenario = {"exists_at": n}  # re-Query: all exist
        fake_self = make_self(backend)
        op = make_op(rid="pair0-remote-0", start_age_s=0.4,
                     requested_tokens=REQUESTED_TOKENS)  # non-warmup rid
        op._a3_batch_index = 1
        op._a3_batch_start = 128          # batch 0 (128 pages) done
        inc = call_page_get(fake_self, op, n, backend)
        kinds = col.kinds()
        self.assertIn("a3_fault_injected", kinds)
        self.assertIn("a3_restore_failure", kinds)
        # gate passed (measurable observed throughput) -> retry ran
        self.assertIn("a3_retry", kinds)
        retry_ev = [f for k, f in col.events if k == "a3_retry"][0]
        self.assertEqual(retry_ev["retry_result"], "success")
        self.assertEqual(retry_ev["retry_suffix_pages"], n)
        # injected call skipped the real GET; ONLY the retry GET happened
        self.assertEqual(len(backend.get_calls), 1)
        self.assertEqual(len(backend.get_calls[0]), n)  # failed suffix only
        self.assertEqual(len(backend.exists_calls), 1)  # re-Query ran
        # retry restored the whole batch -> completed pages back to n
        self.assertEqual(inc, n)
        # failure bookkeeping on the op
        f = op._a3_failure
        self.assertEqual(f["failure_reason"], "transient_exists_after_requery")
        self.assertEqual(f["retry_result"], "success")
        self.assertTrue(f["retry_attempted"])
        # a second request: injection budget consumed -> real path only
        op2 = make_op(rid="pair1-remote-0", start_age_s=0.4,
                      requested_tokens=REQUESTED_TOKENS)
        op2._a3_batch_index = 1
        op2._a3_batch_start = 128
        inc2 = call_page_get(fake_self, op2, n, backend)
        total_inj = [k for k, _ in col.events].count("a3_fault_injected")
        self.assertEqual(total_inj, 1)
        self.assertEqual(len(backend.get_calls), 2)  # real path used

    def test_injection_off_by_default_and_warmup_exempt(self):
        fresh_state()
        lr._A3_STATE["emit"] = Collector()
        n = 64
        backend = FakeBackend(n, [{"ok_pages": [True] * n},
                                   {"ok_pages": [True] * n}])
        fake_self = make_self(backend)
        op = make_op(rid="warmup-True-0")
        op._a3_batch_index = 0
        op._a3_batch_start = 0
        os.environ["A3_FAULT_INJECT"] = "max_requests=5"
        inc = call_page_get(fake_self, op, n, backend)
        self.assertEqual(inc, n)
        self.assertEqual(len(backend.get_calls), 1)  # real path, no injection
        # default OFF
        fresh_state()
        lr._A3_STATE["emit"] = Collector()
        op = make_op(rid="real-req")
        op._a3_batch_index = 0
        op._a3_batch_start = 0
        inc = call_page_get(fake_self, op, n, backend)
        self.assertEqual(len(backend.get_calls), 2)
        self.assertEqual([k for k, _ in lr._A3_STATE["emit"].events].count(
            "a3_fault_injected"), 0)


class Case6RetryBudgetOneShot(unittest.TestCase):
    def test_no_second_retry_in_same_operation(self):
        fresh_state()
        lr._A3_STATE["emit"] = Collector()
        n = 128
        # batch 1: 64 ok then fail, gate passes, retry FAILS; batch 2 also
        # fails: must NOT retry again (max_retry=1 per operation).
        backend = FakeBackend(
            n,
            [
                {
                    "ok_pages": [True] * 64 + [False] * 64,
                    "raw": [-706] * (2 * 64),
                    "exists_at": 64,
                    "retry_ok_pages": [False] * 64,
                },
                {
                    "ok_pages": [False] * 128,
                    "raw": [-706] * (2 * 128),
                    "exists_at": 128,
                },
            ],
        )
        op = make_op(start_age_s=0.2)
        op._a3_batch_index = 0
        op._a3_batch_start = 0
        op._a3_requested_tokens = REQUESTED_TOKENS
        fake_self = make_self(backend)
        inc = call_page_get(fake_self, op, n, backend)
        self.assertEqual(inc, 64)
        self.assertEqual(op._a3_failure["retry_count"], 1)
        self.assertEqual(op._a3_failure["retry_result"], "failed")
        # second failing batch: retry already spent
        op._a3_batch_index = 1
        op._a3_batch_start = 128
        inc2 = call_page_get(fake_self, op, n, backend)
        self.assertEqual(inc2, 0)
        self.assertEqual(op._a3_failure["retry_count"], 1)  # still 1, no loop
        self.assertEqual(op._a3_failure["fallback_reason"], "retry_budget_exhausted")
        # total GET calls: batch1 + retry + batch2 = 3, never more
        self.assertEqual(len(backend.get_calls), 3)


class FailOpenTests(unittest.TestCase):
    def test_requery_error_is_permanent_no_blind_retry(self):
        fresh_state()
        col = Collector()
        lr._A3_STATE["emit"] = col
        n = 128
        backend = FakeBackend(
            n, [{"ok_pages": [False] * n, "raw": [-706] * (2 * n), "raise": "requery"}]
        )
        op = make_op(start_age_s=0.1)
        op._a3_batch_index = 0
        op._a3_batch_start = 0
        op._a3_requested_tokens = REQUESTED_TOKENS
        fake_self = make_self(backend)
        inc = call_page_get(fake_self, op, n, backend)
        self.assertEqual(inc, 0)
        self.assertEqual(len(backend.get_calls), 1)
        self.assertEqual(op._a3_failure["failure_reason"], "permanent_requery_error")
        self.assertEqual(op._a3_failure["fallback_reason"], "requery_failed_no_blind_retry")

    def test_unmeasurable_throughput_never_blind_retries(self):
        """inc==0 with no bytes moved -> observed_bps=0 -> gate rejects."""
        fresh_state()
        backend = FakeBackend(
            n := 128,
            [{"ok_pages": [False] * n, "raw": [-706] * (2 * n), "exists_at": n}],
        )
        op = make_op(start_age_s=0.001)
        op._a3_batch_index = 0
        op._a3_batch_start = 0
        op._a3_requested_tokens = REQUESTED_TOKENS
        fake_self = make_self(backend)
        inc = call_page_get(fake_self, op, n, backend)
        self.assertEqual(inc, 0)
        self.assertEqual(len(backend.get_calls), 1)
        self.assertEqual(op._a3_failure["fallback_reason"], "cost_gate_rejected")


class PageTransferTests(unittest.TestCase):
    """Case 1 companion: full 257-page transfer keeps frozen ack protocol."""

    class _Ack:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _PoolName:
        KV = "KV"

    def test_happy_path_acks_and_bookkeeping(self):
        fresh_state()
        col = Collector()
        lr._A3_STATE["emit"] = col
        lr.STATE = {
            "STORAGE_BATCH_SIZE": 128,
            "PrefetchAck": self._Ack,
            "PoolName": self._PoolName,
            "HiCacheStorageExtraInfo": lambda **kw: kw,
        }
        try:
            q = queue.Queue()

            def kv_batch(op, batch_hashes, batch_host_indices, extra_info, xf):
                return len(batch_hashes)  # all hit

            fake_self = types.SimpleNamespace(
                mem_pool_host=types.SimpleNamespace(page_size=PAGE_SIZE),
                page_size=PAGE_SIZE,
                prefetch_sync_queue=q,
                _page_transfer_kv_batch=kv_batch,
            )
            op = make_op()
            op.host_indices = list(range(257 * PAGE_SIZE))
            op.prefix_keys = None
            op.pool_transfers = []
            op.is_terminated = lambda: False
            r = lr.a3_page_transfer(fake_self, op)
            self.assertEqual(r, 257)
            self.assertEqual(op._a3_requested_tokens, REQUESTED_TOKENS)
            self.assertEqual(op._a3_local_completed_pages, 257)
            acks = []
            while not q.empty():
                acks.append(q.get_nowait())
            self.assertEqual(len(acks), 3)  # 128+128+1 batch protocol intact
            self.assertEqual([a.completed_tokens for a in acks], [8192, 16384, 16448])
            self.assertEqual(acks[0].rid, "req-1")
            self.assertIn("page_transfer", col.kinds())
        finally:
            lr.STATE = None

    def test_frozen_stop_on_failed_batch(self):
        """After a failed batch, later batches emit acks but no IO."""
        fresh_state()
        lr.STATE = {
            "STORAGE_BATCH_SIZE": 128,
            "PrefetchAck": self._Ack,
            "PoolName": self._PoolName,
            "HiCacheStorageExtraInfo": lambda **kw: kw,
        }
        try:
            q = queue.Queue()
            calls = []

            def kv_batch(op, batch_hashes, batch_host_indices, extra_info, xf):
                calls.append(list(batch_hashes))
                if len(calls) == 1:
                    return len(batch_hashes)
                return 0  # batch 2 fails entirely

            fake_self = types.SimpleNamespace(
                mem_pool_host=types.SimpleNamespace(page_size=PAGE_SIZE),
                page_size=PAGE_SIZE,
                prefetch_sync_queue=q,
                _page_transfer_kv_batch=kv_batch,
            )
            op = make_op()
            op.host_indices = list(range(257 * PAGE_SIZE))
            op.prefix_keys = None
            op.pool_transfers = []
            op.is_terminated = lambda: False
            r = lr.a3_page_transfer(fake_self, op)
            self.assertEqual(r, 128)
            self.assertEqual(len(calls), 2)  # batch 3 skipped (frozen behavior)
            acks = []
            while not q.empty():
                acks.append(q.get_nowait())
            self.assertEqual(len(acks), 3)
            self.assertEqual([a.completed_tokens for a in acks], [8192, 8192, 8192])
        finally:
            lr.STATE = None


class GateMathTests(unittest.TestCase):
    def test_gate_examples(self):
        fresh_state()
        lr._A3_STATE["recompute_estimate_s"] = 3.15
        # healthy case: 8192 tokens done in 0.2s, high observed throughput
        g = lr.cost_gate(0.2, 16448, 8192, 98304.0, 201e6, 800e6)
        self.assertTrue(g["retryable"])
        # remaining = (16448-8192)/16448 * 3.15 = 8256/16448*3.15 = 1.5811
        self.assertAlmostEqual(g["remaining_recompute_cost_estimate_s"], 1.581128, places=3)
        self.assertAlmostEqual(g["remaining_remote_cost_estimate_s"],
                               0.2 + (8256 * 98304.0 + 201e6) / 800e6, places=2)
        # degraded case: nothing measurable
        g2 = lr.cost_gate(0.5, 16448, 0, 98304.0, 201e6, 0.0)
        self.assertFalse(g2["retryable"])
        self.assertIsNone(g2["remaining_remote_cost_estimate_s"])
        # slow-elapsed case from Case 3
        g3 = lr.cost_gate(5.0, 16448, 6400, 98304.0, 176e6, 126e6)
        self.assertFalse(g3["retryable"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
