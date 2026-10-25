# Copyright 2026 Individual Contributor: donghyq
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

"""Opt-in real three-GPU test; no generated counter is substituted for native execution."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


def test_sglang_native_crossweight_requests(tmp_path):
    if os.environ.get("VERL_RUN_SGLANG_PRESERVE_GPU") != "1":
        pytest.skip("Opt-in: requires Linux default SGLang lock and three dedicated GPUs")
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("preserve_gate", root / "examples/sglang_preserve_requests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = json.loads((root / "examples/sglang_preserve_contract.json").read_text())
    module.gate(root, contract)
    from sglang_preserve_runtime import run_gpu

    result = run_gpu(tmp_path / "crossweight", tp=2)
    assert result["status"] == "PASS_TINY_MOE_CROSSWEIGHT_PRESERVATION"
    assert result["rollout_tp"] == 2 and len(result["ranks"]) == 2
