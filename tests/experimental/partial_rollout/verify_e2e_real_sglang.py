#!/usr/bin/env python3
"""End-to-end verification: VeRL Partial Rollout -> real SGLang server.

This script connects VeRL's PartialRolloutManager + SGLangPartialRolloutCoordinator
to a real SGLang server (via RealSGLangServer HTTP client), and drives the full
pause -> save -> resume -> complete closed loop.

Prerequisites:
1. Start SGLang server (with HiCache for KV-aware verification):
   python -m sglang.launch_server \\
       --model-path Qwen/Qwen2.5-0.5B-Instruct \\
       --port 30000 \\
       --enable-hierarchical-cache \\
       --hicache-ratio 2.0 \\
       --context-length 4096 \\
       --mem-fraction-static 0.7

2. Run this script:
   python tests/experimental/partial_rollout/verify_e2e_real_sglang.py --port 30000

Dependencies: pip install requests
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERL_ROOT = os.path.join(_HERE, "..", "..", "..")
_SRC = os.path.join(_VERL_ROOT, "verl", "experimental", "partial_rollout")

import importlib.util
import types

for pkg in ["verl", "verl.experimental", "verl.experimental.partial_rollout"]:
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [os.path.join(_VERL_ROOT, *pkg.split("."))]
        sys.modules[pkg] = m

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

rollout_state = _load("verl.experimental.partial_rollout.rollout_state", os.path.join(_SRC, "rollout_state.py"))
partial_rollout_manager = _load("verl.experimental.partial_rollout.partial_rollout_manager", os.path.join(_SRC, "partial_rollout_manager.py"))
sglang_integration = _load("verl.experimental.partial_rollout.sglang_integration", os.path.join(_SRC, "sglang_integration.py"))
real_sglang_client = _load("verl.experimental.partial_rollout.real_sglang_client", os.path.join(_SRC, "real_sglang_client.py"))

RealSGLangServer = real_sglang_client.RealSGLangServer
SGLangPartialRolloutCoordinator = sglang_integration.SGLangPartialRolloutCoordinator
RolloutLifecycle = rollout_state.RolloutLifecycle
InvalidResumeError = rollout_state.InvalidResumeError


def test_basic_closed_loop(server):
    print("=== Test 1: Basic pause/save/resume/complete (KV-aware) ===")
    coord = SGLangPartialRolloutCoordinator(server=server, model_weight_version="v0", sleep_level=2)
    print("  Generating initial response...")
    resp = server.generate("Explain reinforcement learning briefly", max_tokens=50, temperature=0.0)
    print(f"  Generated {len(resp['choices'][0]['message']['content'])} chars")
    state = coord.start_rollout(
        prompt_ids=[1, 2, 3], sampling_params={"temperature": 0.0, "max_new_tokens": 50},
        generated_ids=[10, 20, 30, 40, 50], request_id="e2e-test-1",
    )
    print(f"  Rollout started: {state.request_id}, kv={state.kv_handle}")
    print("  Pausing (preserve_kv)...")
    snapshot = coord.pause_and_save("e2e-test-1")
    print(f"  Snapshot: kv={snapshot.kv_handle}, gen={snapshot.generated_length} tokens")
    assert snapshot.kv_handle is not None
    print("  Resuming...")
    resumed = coord.resume_from_snapshot(snapshot)
    print(f"  Resumed: lifecycle={resumed.lifecycle.value}, kv={resumed.kv_handle}")
    assert resumed.lifecycle == RolloutLifecycle.RUNNING
    assert coord.metrics.recomputed_tokens == 0
    print(f"  Recomputed: {coord.metrics.recomputed_tokens} (KV-aware OK)")
    coord.complete("e2e-test-1")
    print(f"  Completed: {coord.manager.get_state('e2e-test-1').lifecycle.value}")
    print("  Test 1 PASSED\n")


def test_token_only_resume(server):
    print("=== Test 2: Token-only resume (release KV) ===")
    coord = SGLangPartialRolloutCoordinator(server=server, model_weight_version="v0", sleep_level=2)
    coord.start_rollout(
        prompt_ids=[1, 2, 3], sampling_params={"temperature": 0.0, "max_new_tokens": 50},
        generated_ids=[10, 20, 30, 40, 50], request_id="e2e-test-2",
    )
    snapshot = coord.pause_and_save("e2e-test-2")
    print(f"  Paused: kv={snapshot.kv_handle}")
    coord.release_memory("e2e-test-2")
    print(f"  KV released: resident={server.is_kv_resident('e2e-test-2')}")
    assert not server.is_kv_resident("e2e-test-2")
    token_only = snapshot.with_kv_handle(None)
    resumed = coord.resume_from_snapshot(token_only)
    print(f"  Resumed (token-only): kv={resumed.kv_handle}")
    assert resumed.kv_handle is None
    assert coord.metrics.recomputed_tokens == 5
    print(f"  Recomputed: {coord.metrics.recomputed_tokens} (token-only OK)")
    coord.complete("e2e-test-2")
    print("  Test 2 PASSED\n")


def test_weight_update_invalidation(server):
    print("=== Test 3: Weight version mismatch ===")
    coord = SGLangPartialRolloutCoordinator(server=server, model_weight_version="v0", sleep_level=2)
    coord.start_rollout(
        prompt_ids=[1, 2, 3], sampling_params={"temperature": 0.0, "max_new_tokens": 50},
        generated_ids=[10, 20, 30], request_id="e2e-test-3",
    )
    snapshot = coord.pause_and_save("e2e-test-3")
    print(f"  Paused with version v0")
    coord.manager.update_model_version("v1")
    print(f"  Updated to v1")
    try:
        coord.resume_from_snapshot(snapshot)
        print("  FAIL: Should have raised InvalidResumeError")
        return False
    except InvalidResumeError as e:
        print(f"  Correctly rejected: {e}")
    state = coord.manager.get_state("e2e-test-3")
    print(f"  State after failed resume: {state.lifecycle.value}")
    assert state.lifecycle == RolloutLifecycle.INVALIDATED
    print("  Test 3 PASSED\n")
    return True


def test_ttft_comparison(server, label):
    """Compare TTFT before and after preserve_kv pause."""
    print(f"=== Test 4: TTFT comparison ({label}) ===")
    prompt = "Explain the Transformer architecture in detail"
    ttft1, _, n1 = server.generate_stream(prompt, max_tokens=100, temperature=0.0)
    print(f"  First TTFT:  {ttft1:.3f}s ({n1} tokens)")
    server.pause_generation("ttft-test", mode="preserve_kv")
    time.sleep(0.5)
    server.continue_generation()
    time.sleep(0.5)
    ttft2, _, n2 = server.generate_stream(prompt, max_tokens=100, temperature=0.0)
    print(f"  Resume TTFT: {ttft2:.3f}s ({n2} tokens)")
    ratio = ttft2 / ttft1 if ttft1 > 0 else 0
    hit = "HIT" if ratio < 0.8 else "MISS"
    print(f"  Ratio: {ratio:.2f}x ({hit})")
    print(f"  Test 4 done\n")
    return ttft1, ttft2


def test_concurrent_rollouts(server):
    print("=== Test 5: Concurrent paused rollouts ===")
    coord = SGLangPartialRolloutCoordinator(server=server, model_weight_version="v0", sleep_level=2)
    for i in range(3):
        coord.start_rollout(
            prompt_ids=[1, 2, 3], sampling_params={"temperature": 0.0, "max_new_tokens": 50},
            generated_ids=[10 * i, 20 * i, 30 * i], request_id=f"conc-{i}",
        )
    snaps = {}
    for i in range(3):
        snaps[f"conc-{i}"] = coord.pause_and_save(f"conc-{i}")
    print(f"  Paused {len(snaps)} rollouts, active={coord.manager.active_count}")
    assert coord.manager.active_count == 3
    s0 = coord.resume_from_snapshot(snaps["conc-0"])
    assert s0.kv_handle is not None
    assert coord.metrics.recomputed_tokens == 0
    print(f"  Resumed conc-0 (KV-aware): recomputed={coord.metrics.recomputed_tokens}")
    coord.release_memory("conc-1")
    s1 = coord.resume_from_snapshot(snaps["conc-1"].with_kv_handle(None))
    assert s1.kv_handle is None
    print(f"  Resumed conc-1 (token-only): recomputed={coord.metrics.recomputed_tokens}")
    s2 = coord.resume_from_snapshot(snaps["conc-2"])
    assert s2.kv_handle is not None
    print(f"  Resumed conc-2 (KV-aware)")
    for i in range(3):
        coord.complete(f"conc-{i}")
    print(f"  All completed: active={coord.manager.active_count}")
    assert coord.manager.active_count == 0
    print(f"  Metrics: pause={coord.manager.metrics.pause_count}, "
          f"resume={coord.manager.metrics.resume_count}, "
          f"kv_reused={coord.manager.metrics.kv_reused_count}")
    print("  Test 5 PASSED\n")


def main():
    ap = argparse.ArgumentParser(description="E2E: VeRL Partial Rollout -> real SGLang")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--compare-port", type=int, default=None, help="Second server for comparison")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"SGLang server: {base_url}")

    server = RealSGLangServer(base_url)
    if not server.health_check():
        print(f"ERROR: Cannot connect to {base_url}")
        sys.exit(1)
    print(f"Server healthy\n")

    results = {}
    try:
        test_basic_closed_loop(server)
        results["basic"] = "PASS"
    except Exception as e:
        print(f"  FAIL: {e}\n")
        results["basic"] = f"FAIL: {e}"

    try:
        test_token_only_resume(server)
        results["token_only"] = "PASS"
    except Exception as e:
        print(f"  FAIL: {e}\n")
        results["token_only"] = f"FAIL: {e}"

    try:
        test_weight_update_invalidation(server)
        results["weight_mismatch"] = "PASS"
    except Exception as e:
        print(f"  FAIL: {e}\n")
        results["weight_mismatch"] = f"FAIL: {e}"

    try:
        ttft1, ttft2 = test_ttft_comparison(server, "preserve_kv")
        results["ttft"] = f"{ttft1:.3f}s -> {ttft2:.3f}s ({ttft2/ttft1:.2f}x)"
    except Exception as e:
        print(f"  FAIL: {e}\n")
        results["ttft"] = f"FAIL: {e}"

    try:
        test_concurrent_rollouts(server)
        results["concurrent"] = "PASS"
    except Exception as e:
        print(f"  FAIL: {e}\n")
        results["concurrent"] = f"FAIL: {e}"

    if args.compare_port:
        compare_url = f"http://{args.host}:{args.compare_port}"
        print(f"\nComparison server: {compare_url}")
        server2 = RealSGLangServer(compare_url)
        try:
            ttft1, ttft2 = test_ttft_comparison(server2, "no-hicache")
            results["ttft_nocache"] = f"{ttft1:.3f}s -> {ttft2:.3f}s ({ttft2/ttft1:.2f}x)"
        except Exception as e:
            print(f"  FAIL: {e}\n")
            results["ttft_nocache"] = f"FAIL: {e}"

    print("=" * 60)
    print("E2E Verification Summary")
    print("=" * 60)
    for name, result in results.items():
        status = "OK" if result == "PASS" else "CHECK"
        print(f"  {name:20s}  {result}")
    print("=" * 60)
    all_pass = all(v == "PASS" for k, v in results.items() if k.startswith(("basic", "token", "weight", "conc")))
    if all_pass:
        print("\nAll functional tests PASSED")
    else:
        print("\nSome tests failed - check output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
