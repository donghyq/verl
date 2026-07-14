"""Tests for PartialRolloutManager: pause/resume/complete/cancel closed loop."""

import pytest

from verl.experimental.partial_rollout.partial_rollout_manager import (
    PartialRolloutManager,
)
from verl.experimental.partial_rollout.rollout_state import (
    InvalidResumeError,
    RolloutLifecycle,
    new_kv_handle,
)


def make_manager(**kw):
    return PartialRolloutManager(
        model_weight_version=kw.get("model_weight_version", "v0"),
        tokenizer_fingerprint=kw.get("tokenizer_fingerprint", "tok-v0"),
    )


def make_rollout(mgr, rid=None, gen=None, kv=None):
    return mgr.create_rollout(
        prompt_ids=[1, 2, 3],
        sampling_params={"temperature": 0.7, "max_new_tokens": 128},
        generated_ids=gen or [10, 20, 30],
        request_id=rid or "req-1",
        kv_handle=kv,
    )


class TestBasicPauseResumeComplete:
    def test_pause_returns_snapshot(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-1")
        snap = mgr.pause_rollout("r1")
        assert snap.request_id == "r1"
        assert snap.prompt_ids == [1, 2, 3]
        assert snap.generated_ids == [10, 20, 30]
        assert snap.kv_handle == "kv-1"
        assert mgr.get_state("r1").lifecycle == RolloutLifecycle.RESIDENT

    def test_full_closed_loop(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-1")
        snap = mgr.pause_rollout("r1")
        state = mgr.resume_rollout(snap)
        assert state.lifecycle == RolloutLifecycle.RUNNING
        assert state.kv_handle == "kv-1"
        mgr.complete_rollout("r1")
        assert mgr.get_state("r1").lifecycle == RolloutLifecycle.COMPLETED
        assert not mgr.has_resident_kv("r1")
        assert mgr.metrics.complete_count == 1

    def test_pause_non_running_raises(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1")
        mgr.pause_rollout("r1")
        with pytest.raises(InvalidResumeError, match="cannot pause"):
            mgr.pause_rollout("r1")


class TestKVAwareResume:
    def test_kv_aware_resume_zero_recomputed(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-1")
        snap = mgr.pause_rollout("r1")
        state = mgr.resume_rollout(snap)
        assert state.kv_handle == "kv-1"
        assert mgr.metrics.recomputed_tokens == 0
        assert mgr.metrics.kv_reused_count == 1

    def test_kv_handle_preserved_through_snapshot(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-secret")
        snap = mgr.pause_rollout("r1")
        assert snap.kv_handle == "kv-secret"
        state = mgr.resume_rollout(snap)
        assert state.kv_handle == "kv-secret"
        assert mgr.has_resident_kv("r1")


class TestTokenOnlyResume:
    def test_token_only_resume_recomputes_all(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-1")
        snap = mgr.pause_rollout("r1")
        token_only = snap.with_kv_handle(None)
        state = mgr.resume_rollout(token_only)
        assert state.kv_handle is None
        assert mgr.metrics.recomputed_tokens == 3
        assert mgr.metrics.kv_reused_count == 0

    def test_release_kv_then_resume_token_only(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-1")
        snap = mgr.pause_rollout("r1")
        assert mgr.has_resident_kv("r1")
        mgr.release_kv("r1")
        assert not mgr.has_resident_kv("r1")
        assert mgr.get_state("r1").lifecycle == RolloutLifecycle.OFFLOADED
        token_only = snap.with_kv_handle(None)
        state = mgr.resume_rollout(token_only)
        assert state.kv_handle is None
        assert mgr.metrics.recomputed_tokens == 3


class TestVersionMismatch:
    def test_weight_version_mismatch_raises(self):
        mgr = make_manager(model_weight_version="v0")
        make_rollout(mgr, rid="r1", kv="kv-1")
        snap = mgr.pause_rollout("r1")
        mgr.update_model_version("v1")
        with pytest.raises(InvalidResumeError, match="model_weight_version mismatch"):
            mgr.resume_rollout(snap)

    def test_tokenizer_mismatch_raises(self):
        mgr = make_manager(tokenizer_fingerprint="tok-v0")
        make_rollout(mgr, rid="r1", kv="kv-1")
        snap = mgr.pause_rollout("r1")
        mgr.update_tokenizer("tok-v1")
        with pytest.raises(InvalidResumeError, match="tokenizer_fingerprint mismatch"):
            mgr.resume_rollout(snap)

    def test_version_mismatch_invalidates_paused_state(self):
        mgr = make_manager(model_weight_version="v0")
        make_rollout(mgr, rid="r1", kv="kv-1")
        snap = mgr.pause_rollout("r1")
        mgr.update_model_version("v1")
        with pytest.raises(InvalidResumeError):
            mgr.resume_rollout(snap)
        assert mgr.get_state("r1").lifecycle == RolloutLifecycle.INVALIDATED
        assert not mgr.has_resident_kv("r1")
        assert mgr.metrics.invalidated_count == 1


class TestCancelAfterPause:
    def test_cancel_releases_resources(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-1")
        mgr.pause_rollout("r1")
        assert mgr.has_resident_kv("r1")
        state = mgr.cancel_rollout("r1")
        assert state.lifecycle == RolloutLifecycle.CANCELLED
        assert state.stop_reason == "cancelled"
        assert not mgr.has_resident_kv("r1")
        assert state.kv_handle is None
        assert mgr.metrics.cancel_count == 1


class TestDuplicateResume:
    def test_duplicate_resume_raises(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-1")
        snap = mgr.pause_rollout("r1")
        mgr.resume_rollout(snap)
        with pytest.raises(InvalidResumeError, match="already RUNNING"):
            mgr.resume_rollout(snap)


class TestResumeCompletedRollout:
    def test_resume_completed_raises(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-1")
        snap = mgr.pause_rollout("r1")
        mgr.resume_rollout(snap)
        mgr.complete_rollout("r1")
        with pytest.raises(InvalidResumeError, match="terminal lifecycle"):
            mgr.resume_rollout(snap)

    def test_resume_cancelled_raises(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-1")
        snap = mgr.pause_rollout("r1")
        mgr.cancel_rollout("r1")
        with pytest.raises(InvalidResumeError, match="terminal lifecycle"):
            mgr.resume_rollout(snap)


class TestConcurrentPausedRollouts:
    def test_multiple_paused_rollouts_independent(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-a", gen=[10, 20])
        make_rollout(mgr, rid="r2", kv="kv-b", gen=[30, 40, 50])
        make_rollout(mgr, rid="r3", kv="kv-c", gen=[60])
        snap1 = mgr.pause_rollout("r1")
        snap2 = mgr.pause_rollout("r2")
        snap3 = mgr.pause_rollout("r3")
        assert mgr.active_count == 3
        s1 = mgr.resume_rollout(snap1)
        assert s1.kv_handle == "kv-a"
        assert mgr.metrics.recomputed_tokens == 0
        s3 = mgr.resume_rollout(snap3)
        assert s3.kv_handle == "kv-c"
        token_only_2 = snap2.with_kv_handle(None)
        s2 = mgr.resume_rollout(token_only_2)
        assert s2.kv_handle is None
        assert mgr.metrics.recomputed_tokens == 3
        mgr.complete_rollout("r1")
        mgr.complete_rollout("r2")
        mgr.complete_rollout("r3")
        assert mgr.metrics.complete_count == 3
        assert mgr.active_count == 0

    def test_partial_complete_does_not_affect_others(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-a")
        make_rollout(mgr, rid="r2", kv="kv-b")
        snap1 = mgr.pause_rollout("r1")
        mgr.pause_rollout("r2")
        mgr.resume_rollout(snap1)
        mgr.complete_rollout("r1")
        assert mgr.get_state("r1").lifecycle == RolloutLifecycle.COMPLETED
        assert mgr.get_state("r2").lifecycle == RolloutLifecycle.RESIDENT
        assert mgr.has_resident_kv("r2")
        assert not mgr.has_resident_kv("r1")


class TestMetricsTracking:
    def test_metrics_accumulate_across_operations(self):
        mgr = make_manager()
        make_rollout(mgr, rid="r1", kv="kv-1", gen=[10, 20, 30])
        make_rollout(mgr, rid="r2", gen=[40, 50])
        snap1 = mgr.pause_rollout("r1")
        snap2 = mgr.pause_rollout("r2")
        assert mgr.metrics.pause_count == 2
        assert mgr.metrics.save_latency > 0
        mgr.resume_rollout(snap1)
        assert mgr.metrics.resume_count == 1
        assert mgr.metrics.kv_reused_count == 1
        token_only_2 = snap2.with_kv_handle(None)
        mgr.resume_rollout(token_only_2)
        assert mgr.metrics.resume_count == 2
        assert mgr.metrics.recomputed_tokens == 2
        assert mgr.metrics.resume_latency > 0
        mgr.complete_rollout("r1")
        mgr.cancel_rollout("r2")
        assert mgr.metrics.complete_count == 1
        assert mgr.metrics.cancel_count == 1
        d = mgr.metrics.as_dict()
        assert d["pause_count"] == 2
        assert d["recomputed_tokens"] == 2
        assert d["kv_reused_count"] == 1


class TestUnknownRequest:
    def test_pause_unknown_raises(self):
        mgr = make_manager()
        with pytest.raises(KeyError):
            mgr.pause_rollout("nonexistent")

    def test_complete_unknown_raises(self):
        mgr = make_manager()
        with pytest.raises(KeyError):
            mgr.complete_rollout("nonexistent")

    def test_cancel_unknown_raises(self):
        mgr = make_manager()
        with pytest.raises(KeyError):
            mgr.cancel_rollout("nonexistent")
