import hashlib
import json

import pytest
from run_native_training import verify_runtime_source


def test_changed_active_source_is_rejected_before_model_initialization(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "trainer.py"
    path.write_text("original = True\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"archive_sha256": "tested", "files": {"trainer.py": hashlib.sha256(path.read_bytes()).hexdigest()}})
    )
    assert verify_runtime_source(repo, manifest, "tested")["verified_source_files"] == 1
    path.write_text("original = False\n")
    with pytest.raises(ValueError, match="Active native source differs"):
        verify_runtime_source(repo, manifest, "tested")


def test_manifest_cannot_select_files_outside_native_checkout(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"archive_sha256": "tested", "files": {"../other.py": "unknown"}}))
    with pytest.raises(ValueError, match="escapes"):
        verify_runtime_source(tmp_path, manifest, "tested")
