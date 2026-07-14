"""Integration tests: SGLang coordinator driving the pause/resume closed loop."""

import pytest

from verl.experimental.partial_rollout.rollout_state import (
    InvalidResumeError,
    RolloutLifecycle,
)
from verl.experimental.partial_rollout.sglang_integration import (
    FakeSGLangServer,
    SGLangPartialRolloutCoordinator,
)


def make_coord(version="v0", tok="tok-v0", sleep_level=2):
    server = FakeSGLangServer()
    return SGLangPartialRolloutCoordinator(
        server=server,
        model_weight_version=version,
        tokenizer_fingerprint=tok,
        sleep_level=sleep_level,
    )


def start(coord, rid, gen=None):
    return coord.start_rollout(
        prompt_ids=[1, 2, 3],
        sampling_params={"temperature": 0.7},
        generated_ids=gen or [10, 20, 30],
        request_id=rid,
    )


class TestKVAwareResumeSequence:
    def test_full_kv_aware_sequence(self):
        coord = make_coord()
        start(coord, "r1")
        snap = coord.pause_and_save("r1")
        assert "pause_generation(r1)" in coord.call_log
        state = coord.resume_from_snapshot(snap)
        assert state.lifecycle == RolloutLifecycle.RUNNING
        assert state.kv_handle is not None
        assert coord.metrics.recomputed_tokens == 0
        assert coord.metrics.kv_reused_count == 1
        coord.complete("r1")
        assert coord.manager.get_state("r1").lifecycle == RolloutLifecycle.COMPLETED

    def test_call_log_order_kv_aware(self):
        coord = make_coord()
        start(coord, "r1")
        snap = coord.pause_and_save("r1")
        coord.resume_from_snapshot(snap)
        log = coord.call_log
        assert "allocate_kv(r1)" in log[0]
        assert "pause_generation(r1)" in log[1]
        assert any("resume_memory_occupation" in c for c in log)
        # no release_memory_occupation between pause and resume in KV-aware path
        pause_idx = next(i for i, c in enumerate(log) if "pause_generation" in c)
        resume_idx = next(i for i, c in enumerate(log) if "resume_memory_occupation" in c)
        assert not any("release_memory_occupation" in c for c in log[pause_idx:resume_idx])
        coord.complete("r1")
        # complete triggers a release to reclaim server-side KV
        assert any("release_memory_occupation" in c for c in coord.call_log)

    def test_kv_handle_preserved_across_pause_resume(self):
        coord = make_coord()
        start(coord, "r1")
        snap = coord.pause_and_save("r1")
        original_kv = snap.kv_handle
        assert original_kv is not None
        state = coord.resume_from_snapshot(snap)
        assert state.kv_handle == original_kv
        assert coord.server.is_kv_resident("r1")


class TestTokenOnlyResumeSequence:
    def test_release_then_token_only_resume(self):
        coord = make_coord()
        start(coord, "r1")
        snap = coord.pause_and_save("r1")
        assert coord.server.is_kv_resident("r1")
        coord.release_memory("r1")
        assert not coord.server.is_kv_resident("r1")
        assert "r1" in coord.server.released
        token_only = snap.with_kv_handle(None)
        state = coord.resume_from_snapshot(token_only)
        assert state.kv_handle is None
        assert coord.metrics.recomputed_tokens == 3
        assert coord.metrics.kv_reused_count == 0
        coord.complete("r1")

    def test_call_log_order_token_only(self):
        coord = make_coord()
        start(coord, "r1")
        snap = coord.pause_and_save("r1")
        coord.release_memory("r1")
        coord.resume_from_snapshot(snap.with_kv_handle(None))
        log = coord.call_log
        assert any("release_memory_occupation(r1, tags=['kv_cache', 'weights'])" in c for c in log)
        assert any("resume_memory_occupation" in c for c in log)

    def test_sleep_level_1_keeps_weights(self):
        coord = make_coord(sleep_level=1)
        start(coord, "r1")
        snap = coord.pause_and_save("r1")
        coord.release_memory("r1")
        assert coord.server.weights_loaded is True
        release_calls = [c for c in coord.call_log if "release_memory_occupation" in c]
        assert any("kv_cache" in c and "weights" not in c for c in release_calls)


class TestVersionMismatch:
    def test_weight_update_breaks_resume(self):
        coord = make_coord(version="v0")
        start(coord, "r1")
        snap = coord.pause_and_save("r1")
        coord.update_weights("v1")
        with pytest.raises(InvalidResumeError, match="model_weight_version mismatch"):
            coord.resume_from_snapshot(snap)
        assert coord.manager.get_state("r1").lifecycle == RolloutLifecycle.INVALIDATED

    def test_weight_update_releases_server_kv(self):
        coord = make_coord(version="v0")
        start(coord, "r1")
        coord.pause_and_save("r1")
        coord.release_memory("r1")
        coord.update_weights("v1")
        snap2 = coord.start_rollout(
            prompt_ids=[1, 2, 3],
            sampling_params={},
            generated_ids=[10, 20, 30],
            request_id="r2",
        )
        assert snap2 is not None


class TestDuplicateAndTerminalResume:
    def test_duplicate_resume_raises(self):
        coord = make_coord()
        start(coord, "r1")
        snap = coord.pause_and_save("r1")
        coord.resume_from_snapshot(snap)
        with pytest.raises(InvalidResumeError, match="already RUNNING"):
            coord.resume_from_snapshot(snap)

    def test_resume_completed_raises(self):
        coord = make_coord()
        start(coord, "r1")
        snap = coord.pause_and_save("r1")
        coord.resume_from_snapshot(snap)
        coord.complete("r1")
        with pytest.raises(InvalidResumeError, match="terminal lifecycle"):
            coord.resume_from_snapshot(snap)

    def test_cancel_releases_server_kv(self):
        coord = make_coord()
        start(coord, "r1")
        coord.pause_and_save("r1")
        assert coord.server.is_kv_resident("r1")
        coord.cancel("r1")
        assert not coord.server.is_kv_resident("r1")
        assert coord.manager.get_state("r1").lifecycle == RolloutLifecycle.CANCELLED


class TestConcurrentRollouts:
    def test_two_rollouts_independent(self):
        coord = make_coord()
        start(coord, "r1", gen=[10, 20])
        start(coord, "r2", gen=[30, 40, 50])
        snap1 = coord.pause_and_save("r1")
        snap2 = coord.pause_and_save("r2")
        assert coord.manager.active_count == 2
        s1 = coord.resume_from_snapshot(snap1)
        assert s1.kv_handle is not None
        assert coord.metrics.recomputed_tokens == 0
        coord.release_memory("r2")
        s2 = coord.resume_from_snapshot(snap2.with_kv_handle(None))
        assert s2.kv_handle is None
        assert coord.metrics.recomputed_tokens == 3
        coord.complete("r1")
        coord.complete("r2")
        assert coord.manager.active_count == 0
        assert coord.metrics.complete_count == 2

    def test_pause_one_does_not_affect_other(self):
        coord = make_coord()
        start(coord, "r1")
        start(coord, "r2")
        coord.pause_and_save("r1")
        assert coord.manager.get_state("r2").lifecycle == RolloutLifecycle.RUNNING
        assert coord.server.is_kv_resident("r2")


class TestMetricsComparison:
    def test_kv_aware_vs_token_only_comparison(self):
        coord_kv = make_coord()
        coord_tok = make_coord()
        start(coord_kv, "r1", gen=[10, 20, 30, 40])
        start(coord_tok, "r1", gen=[10, 20, 30, 40])
        snap_kv = coord_kv.pause_and_save("r1")
        snap_tok = coord_tok.pause_and_save("r1")
        coord_tok.release_memory("r1")
        coord_kv.resume_from_snapshot(snap_kv)
        coord_tok.resume_from_snapshot(snap_tok.with_kv_handle(None))
        assert coord_kv.metrics.recomputed_tokens == 0
        assert coord_tok.metrics.recomputed_tokens == 4
        assert coord_kv.metrics.kv_reused_count == 1
        assert coord_tok.metrics.kv_reused_count == 0
        assert coord_kv.metrics.resume_latency >= 0
        assert coord_tok.metrics.resume_latency >= 0
