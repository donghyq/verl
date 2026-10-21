# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Real file-copy and cold-restore checks without importing training packages."""

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE = "project_b_checkpoint_storage_test_isolated"
parent = ModuleType(PACKAGE)
parent.__path__ = [str(Path(__file__).parents[4] / "verl/trainer/ppo/v1")]
sys.modules[PACKAGE] = parent
state = importlib.import_module(PACKAGE + ".trajectory_state")
checkpoint = importlib.import_module(PACKAGE + ".trajectory_checkpoint")


def publish(coordinator, version):
    identity = {"policy_version": str(version), "model_weights_fingerprint": f"fixture-weights-{version}",
                "tokenizer_template_fingerprint": "fixture-template", "kv_cache_schema_fingerprint": "fixture-kv"}
    ticket = coordinator.begin_policy_publication(identity, "fixture-config", "fixture-checkpoint", ["http://fixture"])
    coordinator.mark_policy_update_started(ticket)
    return coordinator.confirm_policy_publication(ticket, [
        {"engine_url": "http://fixture", "weight_version": str(version), "source": "model_info"}
    ])


def saved(tmp_path):
    coordinator = state.DurableTrajectoryCoordinator(str(tmp_path / "live"), "run")
    publish(coordinator, 0)
    reference = coordinator._transaction(lambda draft: coordinator._put_object(draft, "inflight", b"original tokens"))
    destination = tmp_path / "global_step_1" / "trajectory_state"
    manifest = coordinator.export_checkpoint(str(destination), 1)
    return coordinator, reference, destination, manifest


def test_checkpoint_immutable_copy_survives_live_gc_and_advanced_policy(tmp_path):
    coordinator, reference, source, manifest = saved(tmp_path)
    try:
        key = coordinator._state["objects"][reference]["storage_key"]
        coordinator._transaction(lambda draft: draft["gc_pending"].append(reference))
        assert coordinator.collect_garbage() == 1
        publish(coordinator, 1)
        destination = tmp_path / "restored"
        restored = checkpoint.restore_trajectory_checkpoint(
            str(source), str(destination), "run", 1, expected_model_configuration_fingerprint="fixture-config"
        )
        assert restored == str(destination.resolve())
        reopened = state.DurableTrajectoryCoordinator(restored, "run")
        try:
            assert reopened._objects.get(key) == b"original tokens"
            assert reopened.get_active_published_policy()["identity"]["policy_version"] == "0"
            # Native on_init_end may now publish the loaded checkpoint version.
            publish(reopened, 1)
            assert reopened.summary()["active_policy_version"] == "1"
            assert coordinator.summary()["active_policy_version"] == "1"
        finally:
            reopened.close()
        assert not any(name.endswith("writer.lock") for name in manifest["files"])
    finally:
        coordinator.close()


def test_queued_gc_does_not_need_already_deleted_payload_for_checkpoint(tmp_path):
    coordinator, reference, source, _ = saved(tmp_path)
    try:
        key = coordinator._state["objects"][reference]["storage_key"]
        coordinator._transaction(lambda draft: draft["gc_pending"].append(reference))
        coordinator._objects.delete(key)
        second = tmp_path / "second-checkpoint"
        manifest = coordinator.export_checkpoint(str(second), 1)
        assert manifest["object_keys"] == []
        assert reference in coordinator._state["objects"]  # live retry state is untouched
        assert checkpoint.restore_trajectory_checkpoint(str(second), str(tmp_path / "fresh"), "run", 1)
        assert (source / "manifest.json").exists()
    finally:
        coordinator.close()




@pytest.mark.parametrize("fault", [
    "missing_object", "changed_object", "omitted_inventory", "wrong_step", "wrong_model"
])
def test_bad_checkpoint_fails_before_creating_restore_root(tmp_path, fault):
    coordinator, _, source, manifest = saved(tmp_path)
    coordinator.close()
    objects = [name for name in manifest["files"] if "/objects/" in name]
    step, model = 1, "fixture-config"
    if fault == "missing_object":
        (source / objects[0]).unlink()
    elif fault == "changed_object":
        (source / objects[0]).write_bytes(b"different tokens")
    elif fault == "omitted_inventory":
        manifest["files"].pop(objects[0])
        (source / "manifest.json").write_text(json.dumps(manifest))
    elif fault == "wrong_step":
        step = 2
    else:
        model = "other-model"
    destination = tmp_path / "invalid-restore"
    with pytest.raises(ValueError):
        checkpoint.restore_trajectory_checkpoint(
            str(source), str(destination), "run", step, expected_model_configuration_fingerprint=model
        )
    assert not destination.exists()


def test_checkpoint_rejects_current_publication_or_unreleased_trajectory(tmp_path):
    coordinator = state.DurableTrajectoryCoordinator(str(tmp_path / "live"), "run")
    try:
        publish(coordinator, 1)
        with pytest.raises(ValueError, match="quiesced"):
            coordinator.export_checkpoint(str(tmp_path / "at-published-version"), 1)
        coordinator.register_producer("source", "incarnation")
        coordinator.open_trajectory("uid/member-0", "source", 600)
        with pytest.raises(ValueError, match="quiesced"):
            coordinator.export_checkpoint(str(tmp_path / "unfinished"), 2)
    finally:
        coordinator.close()


def test_restore_never_overwrites_live_root_and_actor_names_include_root(tmp_path):
    coordinator, _, source, _ = saved(tmp_path)
    try:
        original = coordinator.get_active_published_policy()
        with pytest.raises(ValueError, match="not fresh"):
            checkpoint.restore_trajectory_checkpoint(str(source), str(tmp_path / "live"), "run", 1)
        assert coordinator.get_active_published_policy() == original
        assert (checkpoint.coordinator_actor_name("run", str(tmp_path / "live"))
                != checkpoint.coordinator_actor_name("run", str(tmp_path / "fresh")))
    finally:
        coordinator.close()
