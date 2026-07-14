"""Tests for the async SGLang adapter bridge (Phase C)."""

import asyncio

import pytest

from verl.experimental.partial_rollout.rollout_state import (
    InvalidResumeError,
    RolloutLifecycle,
)
from verl.experimental.partial_rollout.sglang_adapter_bridge import (
    SGLangAdapterBridge,
)


def make_bridge(version="v0", tok="tok-v0", sleep_level=2):
    return SGLangAdapterBridge(
        model_weight_version=version,
        tokenizer_fingerprint=tok,
        sleep_level=sleep_level,
    )


def start(bridge, rid, gen=None):
    return bridge.start_rollout(
        prompt_ids=[1, 2, 3],
        sampling_params={"temperature": 0.7},
        generated_ids=gen or [10, 20, 30],
        request_id=rid,
    )


class TestAsyncRelease:
    def test_release_frees_kv_and_weights(self):
        bridge = make_bridge(sleep_level=2)
        start(bridge, "r1")
        bridge.pause_and_save("r1")
        tags = asyncio.run(bridge.release())
        assert tags == ["kv_cache", "weights"]
        assert not bridge.server.is_kv_resident("r1")
        assert bridge.server.weights_loaded is False

    def test_release_sleep_level_1_keeps_weights(self):
        bridge = make_bridge(sleep_level=1)
        start(bridge, "r1")
        bridge.pause_and_save("r1")
        tags = asyncio.run(bridge.release())
        assert tags == ["kv_cache"]
        assert not bridge.server.is_kv_resident("r1")
        assert bridge.server.weights_loaded is True

    def test_release_one_single_request(self):
        bridge = make_bridge()
        start(bridge, "r1")
        start(bridge, "r2")
        bridge.pause_and_save("r1")
        bridge.pause_and_save("r2")
        asyncio.run(bridge.release_one("r1"))
        assert not bridge.server.is_kv_resident("r1")
        assert bridge.server.is_kv_resident("r2")


class TestAsyncResume:
    def test_resume_from_snapshot_kv_aware(self):
        bridge = make_bridge()
        start(bridge, "r1")
        snap = bridge.pause_and_save("r1")
        state = asyncio.run(bridge.resume_from_snapshot(snap))
        assert state.lifecycle == RolloutLifecycle.RUNNING
        assert state.kv_handle is not None
        assert bridge.metrics.recomputed_tokens == 0

    def test_resume_after_release_token_only(self):
        bridge = make_bridge()
        start(bridge, "r1")
        snap = bridge.pause_and_save("r1")
        asyncio.run(bridge.release_one("r1"))
        state = asyncio.run(bridge.resume_from_snapshot(snap.with_kv_handle(None)))
        assert state.kv_handle is None
        assert bridge.metrics.recomputed_tokens == 3

    def test_resume_marks_weights_loaded(self):
        bridge = make_bridge()
        start(bridge, "r1")
        snap = bridge.pause_and_save("r1")
        asyncio.run(bridge.release())
        assert bridge.server.weights_loaded is False
        asyncio.run(bridge.resume_from_snapshot(snap.with_kv_handle(None)))
        assert bridge.server.weights_loaded is True


class TestAsyncUpdateWeights:
    def test_update_weights_bumps_version(self):
        bridge = make_bridge(version="v0")
        start(bridge, "r1")
        snap = bridge.pause_and_save("r1")
        new_ver = asyncio.run(bridge.update_weights(global_steps=1))
        assert new_ver == "v1"
        with pytest.raises(InvalidResumeError, match="model_weight_version mismatch"):
            asyncio.run(bridge.resume_from_snapshot(snap))

    def test_update_weights_releases_server_weights(self):
        bridge = make_bridge()
        start(bridge, "r1")
        bridge.pause_and_save("r1")
        asyncio.run(bridge.update_weights(global_steps=5))
        assert bridge.server.weights_loaded is False
        assert bridge.manager.model_weight_version == "v5"


class TestAsyncCompleteCancel:
    def test_complete_releases_server_kv(self):
        bridge = make_bridge()
        start(bridge, "r1")
        snap = bridge.pause_and_save("r1")
        asyncio.run(bridge.resume_from_snapshot(snap))
        assert bridge.server.is_kv_resident("r1")
        asyncio.run(bridge.complete("r1"))
        assert not bridge.server.is_kv_resident("r1")
        assert bridge.manager.get_state("r1").lifecycle == RolloutLifecycle.COMPLETED

    def test_cancel_releases_server_kv(self):
        bridge = make_bridge()
        start(bridge, "r1")
        bridge.pause_and_save("r1")
        asyncio.run(bridge.cancel("r1"))
        assert not bridge.server.is_kv_resident("r1")
        assert bridge.manager.get_state("r1").lifecycle == RolloutLifecycle.CANCELLED


class TestContractParity:
    def test_call_log_matches_adapter_pattern(self):
        bridge = make_bridge()
        start(bridge, "r1")
        snap = bridge.pause_and_save("r1")
        asyncio.run(bridge.resume_from_snapshot(snap))
        asyncio.run(bridge.complete("r1"))
        log = bridge.call_log
        assert any("allocate_kv" in c for c in log)
        assert any("pause_generation" in c for c in log)
        assert any("resume_memory_occupation" in c for c in log)
        assert any("release_memory_occupation" in c for c in log)

    def test_sleep_level_property(self):
        bridge = make_bridge(sleep_level=1)
        assert bridge.sleep_level == 1
        bridge2 = make_bridge(sleep_level=2)
        assert bridge2.sleep_level == 2
