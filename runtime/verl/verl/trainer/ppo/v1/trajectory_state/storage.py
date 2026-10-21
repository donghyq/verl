# Copyright 2026 Project B contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small single-writer coordination records and immutable trajectory objects.

This is not a training checkpoint or a replacement for TransferQueue. The
configured filesystem must provide cross-client flock, atomic rename and
fsync semantics. Loss of that filesystem or concurrent unfenced controllers
is outside the guarantee. No operation silently substitutes process memory.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Protocol
from uuid import uuid4

from .snapshot import canonical_bytes


class ObjectStore(Protocol):
    """An immutable byte store; successful put must precede pointer visibility."""

    def put(self, key: str, payload: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...


class CoordinationStorageError(RuntimeError):
    pass


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_durable(directory: Path) -> None:
    missing = []
    cursor = directory
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for item in reversed(missing):
        item.mkdir(exist_ok=True)
        _fsync_directory(item.parent)


def _write_new_file(destination: Path, payload: bytes) -> None:
    """Install an immutable file without replacing an existing object."""
    _mkdir_durable(destination.parent)
    temporary = destination.parent / (".object-" + uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise CoordinationStorageError("immutable object already exists with different bytes") from None
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


class FileObjectStore:
    """Durable default for semantic state; KV references remain optional."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        _mkdir_durable(self.root)

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or re.fullmatch(r"(inflight|handoff-member)/[0-9a-f]{64}", key) is None:
            raise ValueError("invalid immutable trajectory object reference")
        category, digest = key.split("/")
        return self.root / category / digest

    def put(self, key: str, payload: bytes) -> None:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("immutable object must be non-empty bytes")
        if hashlib.sha256(payload).hexdigest() != key.rsplit("/", 1)[-1]:
            raise ValueError("immutable object key does not match bytes")
        _write_new_file(self._path(key), payload)

    def get(self, key: str) -> bytes:
        payload = self._path(key).read_bytes()
        if hashlib.sha256(payload).hexdigest() != key.rsplit("/", 1)[-1]:
            raise CoordinationStorageError("immutable trajectory object digest mismatch")
        return payload

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()
            _fsync_directory(path.parent)


class FileMetadataStore:
    """One durable state record under an exclusive, lifetime writer lock.

    A failure after rename has an uncertain outcome for the caller. The
    coordinator must stop serving mutations and reload in a new instance.
    ``fault_hook`` is an injection point for the necessary crash tests only.
    """

    def __init__(self, root: str | Path, *, fault_hook: Callable[[str], None] | None = None):
        self.root = Path(root)
        _mkdir_durable(self.root)
        self.path = self.root / "coordinator-state.json"
        self.fault_hook = fault_hook
        self._lock_fd = os.open(self.root / "writer.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _fsync_directory(self.root)
        except (BlockingIOError, OSError):
            os.close(self._lock_fd)
            self._lock_fd = None
            raise CoordinationStorageError("another coordinator owns this run, or flock is unavailable") from None

    def _inject(self, stage: str) -> None:
        if self.fault_hook is not None:
            self.fault_hook(stage)

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            envelope = json.loads(self.path.read_bytes())
            if not isinstance(envelope, dict) or set(envelope) != {"record_schema", "sha256", "state"}:
                raise ValueError("invalid durable envelope")
            if envelope["record_schema"] != 1:
                raise ValueError("unsupported durable envelope")
            state = envelope["state"]
            if hashlib.sha256(canonical_bytes(state)).hexdigest() != envelope["sha256"]:
                raise ValueError("durable record digest mismatch")
            return state
        except (ValueError, TypeError) as error:
            raise CoordinationStorageError("cannot decode durable trajectory state") from error

    def save(self, state: dict) -> None:
        body = canonical_bytes(state)
        envelope = canonical_bytes({
            "record_schema": 1,
            "sha256": hashlib.sha256(body).hexdigest(),
            "state": state,
        })
        temporary = self.root / (".state-" + uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                stream.write(envelope)
                stream.flush()
                self._inject("before_file_fsync")
                os.fsync(stream.fileno())
            self._inject("before_replace")
            os.replace(temporary, self.path)
            self._inject("after_replace")
            _fsync_directory(self.root)
            self._inject("after_directory_fsync")
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None
