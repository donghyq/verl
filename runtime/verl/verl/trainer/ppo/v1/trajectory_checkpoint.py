# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Graceful trajectory sidecars for the native save-before-publication boundary."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from uuid import uuid4


def coordinator_actor_name(run_id: str, root_dir: str) -> str:
    root_hash = hashlib.sha256(str(Path(root_dir).resolve()).encode()).hexdigest()[:16]
    return f"trajectory-state-{run_id}-{root_hash}"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quiesced_state(state: dict, checkpoint_step: int) -> bool:
    policy = state["policy_publications"]
    active = policy["active_version"]
    return (active is not None and int(active) < checkpoint_step
            and policy["publication"] is None and policy["barrier"] is None
            and all(row["state"] == "finished" for row in state["handoffs"].values())
            and all(row.get("released") is True for row in state["trajectories"].values()))


def export_trajectory_checkpoint(coordinator, destination: str, checkpoint_step: int) -> dict:
    """Copy verified immutable bytes; never pin only the current mutable path."""
    from .trajectory_state.snapshot import canonical_bytes
    from .trajectory_state.storage import FileMetadataStore, FileObjectStore, _fsync_directory

    coordinator._require_open()
    if type(checkpoint_step) is not int or checkpoint_step <= 0:
        raise ValueError("trajectory checkpoint step must be positive")
    summary = coordinator.summary()
    state = deepcopy(coordinator._state)
    active = state["policy_publications"]["active_version"]
    if (not _quiesced_state(state, checkpoint_step) or summary["active_execution_leases"]
            or summary["unsettled_publications"]):
        raise ValueError("trajectory checkpoint requires quiesced pre-publication state")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("trajectory checkpoint destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".pending-" + uuid4().hex)
    temporary.mkdir()
    # Pending cleanup is not live data. Exclude it from this copy without
    # changing live state or deleting an original object/evidence file.
    for reference in state["gc_pending"]:
        if coordinator._reference_is_pinned(reference):
            raise ValueError("graceful checkpoint cannot discard a pinned cleanup reference")
        state["objects"].pop(reference, None)
    state["gc_pending"] = []
    root = temporary / coordinator.run_id
    objects = FileObjectStore(root / "objects")
    copied_keys = set()
    for descriptor in state["objects"].values():
        key = descriptor["storage_key"]
        if key in copied_keys:
            continue
        payload = coordinator._objects.get(key)
        if len(payload) != descriptor["bytes"] or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
            raise ValueError("trajectory checkpoint object failed integrity validation")
        objects.put(key, payload)
        if objects.get(key) != payload:
            raise ValueError("trajectory checkpoint object copy readback differs")
        copied_keys.add(key)
    metadata = FileMetadataStore(root)
    try:
        metadata.save(state)
        if canonical_bytes(metadata.load()) != canonical_bytes(state):
            raise ValueError("trajectory checkpoint metadata readback differs")
    finally:
        metadata.close()
    files = {str(path.relative_to(temporary)): _digest(path)
             for path in temporary.rglob("*") if path.is_file() and path.name != "writer.lock"}
    manifest = {"schema_version": 1, "run_id": coordinator.run_id, "checkpoint_step": checkpoint_step,
                "last_published_version": active,
                "metadata_state_sha256": hashlib.sha256(canonical_bytes(state)).hexdigest(),
                "files": files, "quiesced": True, "object_keys": sorted(copied_keys)}
    marker = temporary / "manifest.json"
    with marker.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(temporary)
    temporary.rename(destination)
    _fsync_directory(destination.parent)
    return manifest


def restore_trajectory_checkpoint(source: str, destination_root: str, run_id: str, checkpoint_step: int,
                                  *, expected_model_configuration_fingerprint: str | None = None) -> str:
    """Restore into an exclusive root, leaving any advanced live store intact."""
    from .trajectory_state.snapshot import canonical_bytes
    from .trajectory_state.storage import FileMetadataStore, _fsync_directory, _write_new_file

    source, destination_root = Path(source), Path(destination_root)
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("run_id") != run_id
            or manifest.get("checkpoint_step") != checkpoint_step or manifest.get("quiesced") is not True
            or int(manifest["last_published_version"]) >= checkpoint_step):
        raise ValueError("trajectory checkpoint identity or publication frontier differs")
    for name, expected in manifest["files"].items():
        original = source / name
        path = original.resolve()
        if (not path.is_relative_to(source.resolve()) or original.is_symlink()
                or not path.is_file() or _digest(path) != expected):
            raise ValueError("trajectory checkpoint is incomplete or fails integrity verification")
    metadata = FileMetadataStore(source / run_id)
    try:
        state = metadata.load()
    finally:
        metadata.close()
    if (state is None or state["run_id"] != run_id
            or hashlib.sha256(canonical_bytes(state)).hexdigest() != manifest["metadata_state_sha256"]
            or state["policy_publications"]["active_version"] != manifest["last_published_version"]
            or not _quiesced_state(state, checkpoint_step) or state["gc_pending"]):
        raise ValueError("trajectory checkpoint metadata frontier differs")
    record = state["policy_publications"]["records"][manifest["last_published_version"]]
    if (expected_model_configuration_fingerprint is not None
            and record["model_configuration_fingerprint"] != expected_model_configuration_fingerprint):
        raise ValueError("trajectory checkpoint model configuration differs")
    expected_files = {f"{run_id}/coordinator-state.json"}
    for descriptor in state["objects"].values():
        name = f"{run_id}/objects/{descriptor['storage_key']}"
        expected_files.add(name)
        if (name not in manifest["files"] or manifest["files"][name] != descriptor["sha256"]
                or (source / name).stat().st_size != descriptor["bytes"]):
            raise ValueError("trajectory checkpoint object descriptors differ from copied files")
    if set(manifest["files"]) != expected_files:
        raise ValueError("trajectory checkpoint file inventory differs")
    if destination_root.exists():
        raise ValueError("trajectory checkpoint restore root is not fresh")
    # Copy only the verified checkpoint files; do not hydrate execution leases.
    temporary = destination_root.with_name(destination_root.name + ".pending-" + uuid4().hex)
    temporary.mkdir(parents=True)
    for name in manifest["files"]:
        _write_new_file(temporary / name, (source / name).read_bytes())
    _fsync_directory(temporary)
    temporary.rename(destination_root)
    _fsync_directory(destination_root.parent)
    return str(destination_root.resolve())
