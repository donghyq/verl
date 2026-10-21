# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""CPU-only input guard tests; neither tokenizer nor model is loaded."""

import importlib.util
import json
from pathlib import Path

import pytest


DIRECTORY = Path(__file__).resolve().parents[4] / "project-b-final-validation-20260928" / "production-entry-20261008"
SPEC = importlib.util.spec_from_file_location("v1_functional_gate_prepare_tests", DIRECTORY / "prepare_v1_functional_gate.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def test_original_support_is_an_unchanged_file_not_an_old_verl_namespace():
    support = DIRECTORY / "support"
    assert prepare.sha256_file(support / "local_life_functional_support.py") == prepare.SUPPORT_SHA256
    assert prepare.sha256_file(support / "local_life_original_tools.yaml") == prepare.ORIGINAL_TOOLS_SHA256
    assert not (support / "verl").exists()


def test_only_import_namespace_changes_and_original_tool_options_survive():
    original = {"tools": [{"class_name": f"{prepare.ORIGINAL_MODULE}.{name}", "config": {"type": "native"}}
                          for name in prepare.CLASSES]}
    rewritten = prepare.rewrite_tools_namespace(original)
    assert rewritten == {"tools": [{"class_name": f"{prepare.SUPPORT_MODULE}.{name}", "config": {"type": "native"}}
                                   for name in prepare.CLASSES]}
    assert original["tools"][0]["class_name"].startswith(prepare.ORIGINAL_MODULE)
    original["tools"][0]["config"]["added_delay"] = 1000
    with pytest.raises(ValueError):
        prepare.rewrite_tools_namespace(original)


def test_actual_file_digest_changes_if_source_weights_or_template_change(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"actual CPU fixture file; not a real model")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "CPU-fixture"}))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "fixture"}))
    files, weights, tokenizer = prepare.model_file_identity(tmp_path)
    assert len(files) == 3 and len(weights) == 64 and len(tokenizer) == 64
    (tmp_path / "model.safetensors").write_bytes(b"changed actual source bytes")
    _, changed_weights, unchanged_tokenizer = prepare.model_file_identity(tmp_path)
    assert changed_weights != weights and unchanged_tokenizer == tokenizer
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "changed fixture"}))
    assert prepare.model_file_identity(tmp_path)[2] != tokenizer


def test_missing_source_weights_or_tokenizer_does_not_create_a_fingerprint(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError):
        prepare.model_file_identity(tmp_path)
