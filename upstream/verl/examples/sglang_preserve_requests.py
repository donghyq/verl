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

"""Opt-in native SGLang cross-weight probe; --help and source checks need only Python."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_sources(native_root, contract):
    evidence = []
    for row in contract["native_sources"]:
        path = native_root / row["relative_path"]
        content = path.read_bytes()
        if sha(content) != row["file_sha256"]:
            raise ValueError(f"Native source differs: {row['relative_path']}")
        text = content.decode()
        tree = ast.parse(text)
        for qualified, expected in row["symbol_source_sha256"].items():
            current = tree
            for name in qualified.split("."):
                current = next(node for node in current.body if getattr(node, "name", None) == name)
            if sha(ast.get_source_segment(text, current).encode()) != expected:
                raise ValueError(f"Native method differs: {qualified}")
        evidence.append({"path": row["relative_path"], "sha256": sha(content)})
    return evidence


def gate(repo_root, contract, native_root=None):
    for row in contract["verl_sources"]:
        if sha((repo_root / row["relative_path"]).read_bytes()) != row["sha256"]:
            raise ValueError(f"VeRL source differs: {row['relative_path']}")
    if sha((repo_root / "uv.lock").read_bytes()) != contract["uv_lock_sha256"]:
        raise ValueError("uv.lock differs from the default SGLang dependency target")
    if native_root is not None:
        return {
            "status": "PASS_SOURCE_PINS_ONLY",
            "native_sources": check_sources(native_root, contract),
            "dependency_import_CUDA_or_GPU": "NOT_RUN",
        }
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("The GPU candidate requires Linux and Python 3.12")
    versions = {name: importlib.metadata.version(name) for name in contract["dependencies"]}
    if versions != contract["dependencies"]:
        raise RuntimeError(f"Dependency versions differ: {versions}")
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Installed SGLang package location is missing")
    native_root = Path(next(iter(spec.submodule_search_locations)))
    return {
        "status": "PASS_INSTALLED_METADATA_AND_SOURCE_PINS",
        "versions": versions,
        "native_sources": check_sources(native_root, contract),
        "CUDA_or_GPU": "NOT_RUN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--observer-dir", type=Path)
    parser.add_argument("--contract", type=Path, default=Path(__file__).with_name("sglang_preserve_contract.json"))
    parser.add_argument("--source-only", type=Path, metavar="NATIVE_SGLANG_ROOT")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument(
        "--run-gpu", action="store_true", help="Explicitly start this probe's own local Ray and tiny model"
    )
    parser.add_argument(
        "--rollout-tp", choices=(1, 2), type=int, default=2, help="TP2 acceptance needs 3 GPUs; TP1 debug needs 2"
    )
    parser.add_argument("--output", type=Path, help="Fresh result directory; existing directories are rejected")
    args = parser.parse_args()
    if args.source_only is not None and args.run_gpu:
        parser.error("--source-only cannot authorize a GPU run")
    if not (args.gate_only or args.source_only is not None or args.run_gpu):
        parser.error("Choose --gate-only, --source-only, or --run-gpu")
    contract = json.loads(args.contract.read_text())
    repo_root = args.repo_root.resolve()
    evidence = gate(repo_root, contract, args.source_only)
    if not args.run_gpu:
        print(json.dumps(evidence))
        return
    if args.output is None:
        parser.error("--run-gpu requires a fresh --output")
    observer_dir = args.observer_dir or Path(__file__).resolve().parents[1] / "tests/special_distributed"
    sys.path[:0] = [str(repo_root), str(observer_dir.resolve())]
    os.chdir(repo_root)
    from sglang_preserve_runtime import run_gpu

    print(json.dumps(run_gpu(args.output.resolve(), tp=args.rollout_tp)))


if __name__ == "__main__":
    main()
