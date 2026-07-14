"""Tests for rollout_state: snapshot save/load, lifecycle transitions, version validation."""

import time

import pytest

from verl.experimental.partial_rollout.rollout_state import (
    InvalidResumeError,
    RolloutLifecycle,
    RolloutState,
    RolloutStateSnapshot,
    new_kv_handle,
    new_request_id,
)


class TestSnapshotSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        snap = RolloutStateSnapshot(
            request_id="req-1",
            prompt_ids=[1, 2, 3],
            generated_ids=[4, 5],
            sampling_params={"temperature": 0.7},
            sequence_position=3,
            stop_reason=None,
            kv_handle="kv-abc",
            model_weight_version="v1",
            tokenizer_fingerprint="tok-v1",
            created_at=1000.0,
            paused_at=1001.0,
        )
        d = snap.to_dict()
        assert d["request_id"] == "req-1"
        assert d["kv_handle"] == "kv-abc"
        restored = RolloutStateSnapshot.from_dict(d)
        assert restored == snap

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown snapshot keys"):
            RolloutStateSnapshot.from_dict(
                {
                    "request_id": "r1",
                    "prompt_ids": [],
                    "generated_ids": [],
                    "sampling_params": {},
                    "sequence_position": 0,
                    "stop_reason": None,
                    "kv_handle": None,
                    "model_weight_version": "v0",
                    "tokenizer_fingerprint": "tok-v0",
                    "created_at": 0.0,
                    "paused_at": 0.0,
                    "extra_key": "bad",
                }
            )

    def test_with_kv_handle_clears_handle(self):
        snap = RolloutStateSnapshot(
            request_id="r1",
            prompt_ids=[1],
            generated_ids=[2],
            sampling_params={},
            sequence_position=1,
            kv_handle="kv-x",
        )
        cleared = snap.with_kv_handle(None)
        assert cleared.kv_handle is None
        assert snap.kv_handle == "kv-x"

    def test_generated_length(self):
        snap = RolloutStateSnapshot(
            request_id="r1",
            prompt_ids=[1, 2],
            generated_ids=[3, 4, 5],
            sampling_params={},
            sequence_position=2,
        )
        assert snap.generated_length == 3


class TestRolloutStateLifecycle:
    def test_default_lifecycle_is_running(self):
        state = RolloutState(
            request_id="r1",
            prompt_ids=[1, 2],
            generated_ids=[],
            sampling_params={},
            sequence_position=2,
        )
        assert state.lifecycle == RolloutLifecycle.RUNNING

    def test_sequence_position_defaults_to_prompt_length(self):
        state = RolloutState(
            request_id="r1",
            prompt_ids=[1, 2, 3],
            generated_ids=[],
            sampling_params={},
            sequence_position=0,
        )
        assert state.sequence_position == 3

    def test_negative_sequence_position_raises(self):
        with pytest.raises(ValueError):
            RolloutState(
                request_id="r1",
                prompt_ids=[1],
                generated_ids=[],
                sampling_params={},
                sequence_position=-1,
            )

    def test_to_snapshot_preserves_fields(self):
        state = RolloutState(
            request_id="r1",
            prompt_ids=[1, 2],
            generated_ids=[3, 4],
            sampling_params={"temperature": 0.8},
            sequence_position=2,
            kv_handle="kv-h",
            model_weight_version="v2",
            tokenizer_fingerprint="tok-v2",
        )
        snap = state.to_snapshot()
        assert snap.request_id == "r1"
        assert snap.prompt_ids == [1, 2]
        assert snap.generated_ids == [3, 4]
        assert snap.kv_handle == "kv-h"
        assert snap.model_weight_version == "v2"
        assert snap.tokenizer_fingerprint == "tok-v2"

    def test_from_snapshot_sets_resuming(self):
        snap = RolloutStateSnapshot(
            request_id="r1",
            prompt_ids=[1],
            generated_ids=[2],
            sampling_params={},
            sequence_position=1,
            kv_handle="kv-h",
        )
        state = RolloutState.from_snapshot(snap)
        assert state.lifecycle == RolloutLifecycle.RESUMING
        assert state.resumed_at is not None
        assert state.kv_handle == "kv-h"

    def test_snapshot_is_independent_of_state(self):
        state = RolloutState(
            request_id="r1",
            prompt_ids=[1, 2],
            generated_ids=[3],
            sampling_params={},
            sequence_position=2,
            kv_handle="kv-h",
        )
        snap = state.to_snapshot()
        state.generated_ids.append(99)
        assert snap.generated_ids == [3]


class TestLifecycleProperties:
    def test_terminal_lifecycles(self):
        for lc in [
            RolloutLifecycle.COMPLETED,
            RolloutLifecycle.CANCELLED,
            RolloutLifecycle.EXPIRED,
            RolloutLifecycle.INVALIDATED,
        ]:
            assert lc.is_terminal
            assert not lc.is_resumable

    def test_resumable_lifecycles(self):
        for lc in [
            RolloutLifecycle.PAUSED,
            RolloutLifecycle.RESIDENT,
            RolloutLifecycle.OFFLOADED,
        ]:
            assert lc.is_resumable
            assert not lc.is_terminal

    def test_running_not_terminal_not_resumable(self):
        assert not RolloutLifecycle.RUNNING.is_terminal
        assert not RolloutLifecycle.RUNNING.is_resumable


class TestValidateResume:
    def test_matching_versions_pass(self):
        state = RolloutState(
            request_id="r1",
            prompt_ids=[1],
            generated_ids=[2],
            sampling_params={},
            sequence_position=1,
            model_weight_version="v1",
            tokenizer_fingerprint="tok-v1",
        )
        state.validate_resume("v1", "tok-v1")

    def test_model_weight_mismatch_raises(self):
        state = RolloutState(
            request_id="r1",
            prompt_ids=[1],
            generated_ids=[2],
            sampling_params={},
            sequence_position=1,
            model_weight_version="v1",
            tokenizer_fingerprint="tok-v1",
        )
        with pytest.raises(InvalidResumeError, match="model_weight_version mismatch"):
            state.validate_resume("v2", "tok-v1")

    def test_tokenizer_mismatch_raises(self):
        state = RolloutState(
            request_id="r1",
            prompt_ids=[1],
            generated_ids=[2],
            sampling_params={},
            sequence_position=1,
            model_weight_version="v1",
            tokenizer_fingerprint="tok-v1",
        )
        with pytest.raises(InvalidResumeError, match="tokenizer_fingerprint mismatch"):
            state.validate_resume("v1", "tok-v2")


class TestIdGenerators:
    def test_new_request_id_unique(self):
        a = new_request_id()
        b = new_request_id()
        assert a != b
        assert a.startswith("req-")

    def test_new_kv_handle_unique(self):
        a = new_kv_handle()
        b = new_kv_handle()
        assert a != b
        assert a.startswith("kv-")
