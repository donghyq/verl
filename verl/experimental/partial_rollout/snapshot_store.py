"""Phase D: snapshot persistence for cross-process / cross-restart resume.

:class:`SnapshotStore` persists :class:`RolloutStateSnapshot` to disk as JSON,
enabling a manager that paused a rollout in one process to resume it in
another (e.g. after a training step that restarted the serving engine).

The store is intentionally minimal — one JSON file per request_id under a
root directory — to keep the Phase A constraint of no external dependencies.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from verl.experimental.partial_rollout.rollout_state import (
    RolloutState,
    RolloutStateSnapshot,
)
from verl.experimental.partial_rollout.partial_rollout_manager import (
    PartialRolloutManager,
)


class SnapshotStoreError(Exception):
    """Raised on store-level failures (corrupt files, IO errors)."""


class SnapshotStore:
    """Filesystem-backed store for rollout snapshots.

    Each snapshot is written as ``<root>/<request_id>.json``. The store keeps
    no in-memory state; every operation hits disk so it is safe to use across
    processes.
    """

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, request_id: str) -> str:
        safe_name = request_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self.root, f"{safe_name}.json")

    def save(self, snapshot: RolloutStateSnapshot) -> str:
        """Persist a snapshot; returns the file path."""
        path = self._path(snapshot.request_id)
        data = snapshot.to_dict()
        data["_store_saved_at"] = time.time()
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return path

    def load(self, request_id: str) -> RolloutStateSnapshot:
        """Load a snapshot by request_id."""
        path = self._path(request_id)
        if not os.path.exists(path):
            raise SnapshotStoreError(f"no snapshot for {request_id!r} at {path}")
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise SnapshotStoreError(f"corrupt snapshot {path}: {e}") from e
        data.pop("_store_saved_at", None)
        return RolloutStateSnapshot.from_dict(data)

    def exists(self, request_id: str) -> bool:
        return os.path.exists(self._path(request_id))

    def list_paused(self) -> list[str]:
        """Return all stored request_ids (sorted)."""
        if not os.path.isdir(self.root):
            return []
        result = []
        for name in os.listdir(self.root):
            if name.endswith(".json"):
                result.append(name[:-5].replace("_", "/"))
        return sorted(result)

    def delete(self, request_id: str) -> bool:
        """Delete a snapshot; returns True if it existed."""
        path = self._path(request_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def clear(self) -> int:
        """Delete all snapshots; returns count removed."""
        count = 0
        for name in list(os.listdir(self.root)):
            if name.endswith(".json"):
                os.remove(os.path.join(self.root, name))
                count += 1
        return count


def resume_from_store(
    store: SnapshotStore,
    manager: PartialRolloutManager,
    request_id: str,
) -> RolloutStateSnapshot:
    """Load a snapshot from the store and resume it on a (possibly new) manager.

    This is the cross-process resume primitive: manager A paused and saved the
    snapshot; manager B loads and resumes it. Version validation still applies.
    """
    snapshot = store.load(request_id)
    state = RolloutState.from_snapshot(snapshot)
    state.validate_resume(
        manager.model_weight_version,
        manager.tokenizer_fingerprint,
    )
    manager.resume_rollout(snapshot)
    return snapshot
