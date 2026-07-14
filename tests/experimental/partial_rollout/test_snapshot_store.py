"""Tests for SnapshotStore disk persistence and cross-manager resume (Phase D)."""

import json
import os
import tempfile

import pytest

from verl.experimental.partial_rollout.partial_rollout_manager import (
    PartialRolloutManager,
)
from verl.experimental.partial_rollout.rollout_state import (
    InvalidResumeError,
    RolloutState,
    RolloutStateSnapshot,
)
from verl.experimental.partial_rollout.snapshot_store import (
    SnapshotStore,
    SnapshotStoreError,
    resume_from_store,
)


def make_snapshot(rid="r1", kv="kv-1", version="v0", gen=None):
    return RolloutStateSnapshot(
        request_id=rid,
        prompt_ids=[1, 2, 3],
        generated_ids=gen or [10, 20, 30],
        sampling_params={"temperature": 0.7},
        sequence_position=3,
        kv_handle=kv,
        model_weight_version=version,
        tokenizer_fingerprint="tok-v0",
        created_at=1000.0,
        paused_at=1001.0,
    )


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        snap = make_snapshot(rid="r1", kv="kv-x")
        path = store.save(snap)
        assert os.path.exists(path)
        loaded = store.load("r1")
        assert loaded == snap

    def test_load_nonexistent_raises(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        with pytest.raises(SnapshotStoreError, match="no snapshot"):
            store.load("nonexistent")

    def test_save_creates_json_file(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        snap = make_snapshot(rid="r1")
        path = store.save(snap)
        assert path.endswith("r1.json")
        with open(path) as f:
            data = json.load(f)
        assert data["request_id"] == "r1"
        assert data["kv_handle"] == "kv-1"
        assert "_store_saved_at" in data


class TestListAndDelete:
    def test_list_paused_empty(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        assert store.list_paused() == []

    def test_list_paused_multiple(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        store.save(make_snapshot(rid="r1"))
        store.save(make_snapshot(rid="r2"))
        store.save(make_snapshot(rid="r3"))
        assert store.list_paused() == ["r1", "r2", "r3"]

    def test_exists(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        store.save(make_snapshot(rid="r1"))
        assert store.exists("r1")
        assert not store.exists("r2")

    def test_delete(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        store.save(make_snapshot(rid="r1"))
        assert store.delete("r1") is True
        assert not store.exists("r1")
        assert store.delete("r1") is False

    def test_clear(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        store.save(make_snapshot(rid="r1"))
        store.save(make_snapshot(rid="r2"))
        count = store.clear()
        assert count == 2
        assert store.list_paused() == []


class TestCorruptSnapshot:
    def test_corrupt_json_raises(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        path = store._path("r1")
        with open(path, "w") as f:
            f.write("{ invalid json")
        with pytest.raises(SnapshotStoreError, match="corrupt snapshot"):
            store.load("r1")

    def test_unknown_keys_rejected(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        path = store._path("r1")
        with open(path, "w") as f:
            json.dump({
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
                "rogue_field": "bad",
            }, f)
        with pytest.raises(ValueError, match="unknown snapshot keys"):
            store.load("r1")


class TestCrossManagerResume:
    def test_cross_manager_resume_kv_aware(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        mgr_a = PartialRolloutManager(model_weight_version="v0")
        state = mgr_a.create_rollout(
            prompt_ids=[1, 2, 3],
            sampling_params={},
            generated_ids=[10, 20],
            request_id="r1",
            kv_handle="kv-x",
        )
        snap = mgr_a.pause_rollout("r1")
        store.save(snap)

        mgr_b = PartialRolloutManager(model_weight_version="v0")
        mgr_b._resident_kv["r1"] = "kv-x"
        snapshot = resume_from_store(store, mgr_b, "r1")
        assert snapshot.kv_handle == "kv-x"
        assert mgr_b.metrics.recomputed_tokens == 0
        assert mgr_b.get_state("r1").lifecycle.value == "running"

    def test_cross_manager_resume_token_only(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        mgr_a = PartialRolloutManager(model_weight_version="v0")
        mgr_a.create_rollout(
            prompt_ids=[1, 2, 3],
            sampling_params={},
            generated_ids=[10, 20, 30],
            request_id="r1",
            kv_handle="kv-x",
        )
        snap = mgr_a.pause_rollout("r1")
        token_only = snap.with_kv_handle(None)
        store.save(token_only)

        mgr_b = PartialRolloutManager(model_weight_version="v0")
        resume_from_store(store, mgr_b, "r1")
        assert mgr_b.metrics.recomputed_tokens == 3
        assert mgr_b.get_state("r1").kv_handle is None

    def test_cross_manager_version_mismatch_fails(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        mgr_a = PartialRolloutManager(model_weight_version="v0")
        mgr_a.create_rollout(
            prompt_ids=[1, 2],
            sampling_params={},
            generated_ids=[10],
            request_id="r1",
            kv_handle="kv-x",
        )
        snap = mgr_a.pause_rollout("r1")
        store.save(snap)

        mgr_b = PartialRolloutManager(model_weight_version="v1")
        with pytest.raises(InvalidResumeError, match="model_weight_version mismatch"):
            resume_from_store(store, mgr_b, "r1")


class TestPathSafety:
    def test_slash_in_request_id_sanitized(self, tmp_path):
        store = SnapshotStore(str(tmp_path))
        snap = make_snapshot(rid="group/sub-1")
        path = store.save(snap)
        assert "group_sub-1.json" in path
        loaded = store.load("group/sub-1")
        assert loaded.request_id == "group/sub-1"
