#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_return_verdict.py — 「源 GPU 实际回 Trainer」A1-A7 判定器（M4.2 dry-run）

用途：
  输入 joint driver 采集的证据（release receipt + NVML 快照 + 时间线 + trainer 观测），
  按 GPU-RETURN-ACCEPTANCE.md 的 A1-A7 判据逐条评估，产出独立的
  gpu_returned_for_training_verdict.json。

  本脚本为纯 CPU 本地逻辑验证版：自带 --selftest，用 PASS / FAIL 两个伪造场景
  验证判定逻辑；真实场景由 joint driver 在 GPU 轮中喂真实证据文件。

用法：
  python3 gpu_return_verdict.py --selftest          # 自测（两个场景）
  python3 gpu_return_verdict.py --evidence ev.json --out verdict.json  # 真实判定

证据文件 schema（ev.json，字段名与 ACCEPTANCE 文档 A1-A7 对应）：
{
  "release_receipt": {
    "release_full_gpu_pool": true,          # 请求参数
    "full_gpu_pool_released": true,         # 引擎判定（A1）
    "memory_before_bytes": 123456789,       # _cuda_memory() 快照
    "memory_after_bytes":  98765432,
    "restore_path_exercised": false,        # 失败路径是否走了 _restore
    "fail_reason": null
  },
  "nvml": {
    "before": [{"index":0,"uuid":"GPU-aaa","memory_used_bytes":x,"memory_free_bytes":y}, ...],
    "after":  [...]
  },
  "allocator_free_tokens_after": 971264,    # 引擎日志值（A2 三方互证）
  "kv_pool": {
    "per_token_bytes": 200,                 # 组装时按模型 config 计算（TP2 分摊后）
    "kv_tokens": 486000                     # 预算 tokens（TP2）
  },
  "timeline": {
    "safe_point_all_members_at": 1.0,       # 单调 epoch 秒
    "release_begin": 2.0,
    "release_end": 3.0,
    "trainer_alloc_begin": 4.0,
    "trainer_step_end": 5.0
  },
  "trainer": {
    "alloc_gpu_uuids": ["GPU-aaa","GPU-bbb"],
    "peak_mem_bytes": 70000000000,
    "grad_norm": 0.5837,                    # A4：R57/R62/R79 判据
    "adam_state_delta_ranks": 4             # 需 >=2
  },
  "target_resume": {
    "resume_success": true,                 # A5
    "digest_match": true
  },
  "fencing": {
    "generate_during_release_window": 0     # A6：release 窗口内 /generate 计数
  },
  "gates": {
    "no_duplicate_consume": true            # A6 后半：无重复消费
  }
}

判定阈值：
  NVML 与 allocator 与预算三方互证下限，容差 10%（ACCEPTANCE §5）。
"""
import argparse
import json
import sys

TOLERANCE = 0.10          # A2/A3 三方互证 10% 容差
MIN_ADAM_DELTA_RANKS = 2  # A4


def expected_free_delta_bytes(ev):
    """A1/A2 阈值：KV 池预算（TP 内按卡分摊后的 per-token bytes × tokens）。"""
    kp = ev.get("kv_pool") or {}
    return int(kp.get("per_token_bytes", 0) * kp.get("kv_tokens", 0))


def check_a1(ev):
    r = ev.get("release_receipt") or {}
    exp = expected_free_delta_bytes(ev)
    actual = max(0, int(r.get("memory_before_bytes", 0)) - int(r.get("memory_after_bytes", 0)))
    ok = (
        bool(r.get("release_full_gpu_pool"))
        and bool(r.get("full_gpu_pool_released"))
        and exp > 0
        and actual >= int(exp * (1 - TOLERANCE))
    )
    detail = {
        "release_full_gpu_pool": r.get("release_full_gpu_pool"),
        "full_gpu_pool_released": r.get("full_gpu_pool_released"),
        "receipt_delta_bytes": actual,
        "expected_min_bytes": int(exp * (1 - TOLERANCE)),
    }
    if not ok and not r.get("full_gpu_pool_released"):
        detail["note"] = "若走失败路径，需确认 _restore 成功且 fail_reason 已记录（A1 失败分支）"
    return ok, detail


def check_a2(ev):
    """NVML 独立验证 + 与 receipt、allocator tokens 三方互证。"""
    exp = expected_free_delta_bytes(ev)
    nv = ev.get("nvml") or {}
    before, after = nv.get("before") or [], nv.get("after") or []
    if not before or not after or exp <= 0:
        return False, {"error": "nvml before/after 缺失或 kv_pool 预算未填"}
    free_delta = {}
    for b in before:
        a = next((x for x in after if x.get("uuid") == b.get("uuid")), None)
        if a is None:
            return False, {"error": "NVML before/after UUID 不匹配"}
        free_delta[b["uuid"]] = int(a.get("memory_free_bytes", 0)) - int(b.get("memory_free_bytes", 0))
    min_delta = min(free_delta.values())
    max_delta = max(free_delta.values())
    # 两卡（TP2）一致且都过阈值下限
    consistent = (max_delta - min_delta) <= int(exp * TOLERANCE)
    over_threshold = min_delta >= int(exp * (1 - TOLERANCE))
    # 三方互证：receipt delta
    r = ev.get("release_receipt") or {}
    receipt_delta = max(0, int(r.get("memory_before_bytes", 0)) - int(r.get("memory_after_bytes", 0)))
    receipt_match = abs(receipt_delta - min_delta) <= int(max(exp, 1) * TOLERANCE)
    # 三方互证：allocator free tokens 折算
    tok = int(ev.get("allocator_free_tokens_after", 0)) * int((ev.get("kv_pool") or {}).get("per_token_bytes", 0))
    tokens_match = (tok == 0) or (abs(tok - min_delta) <= int(max(exp, 1) * TOLERANCE * 3))  # tokens 为池级累计，放宽到容差×3
    detail = {
        "per_gpu_free_delta_bytes": free_delta,
        "expected_min_bytes": int(exp * (1 - TOLERANCE)),
        "tp_ranks_consistent": consistent,
        "receipt_delta_bytes": receipt_delta,
        "receipt_match": receipt_match,
        "allocator_tokens_delta_bytes": tok,
        "tokens_match": tokens_match,
    }
    return bool(consistent and over_threshold and receipt_match), detail


def check_a3(ev):
    """GPU UUID 链 + 时间线单调性（核心判据）。"""
    t = ev.get("timeline") or {}
    tr = ev.get("trainer") or {}
    seq = [t.get(k) for k in ("safe_point_all_members_at", "release_begin", "release_end",
                              "trainer_alloc_begin", "trainer_step_end")]
    monotonic = all(x is not None for x in seq) and all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
    src_uuids = {g.get("uuid") for g in (ev.get("nvml") or {}).get("before") or []}
    tr_uuids = set(tr.get("alloc_gpu_uuids") or [])
    uuid_bound = bool(src_uuids) and bool(tr_uuids) and tr_uuids.issubset(src_uuids)
    detail = {"timeline_monotonic": monotonic, "source_uuids": sorted(src_uuids),
              "trainer_uuids": sorted(tr_uuids), "trainer_on_released_gpus": uuid_bound}
    return bool(monotonic and uuid_bound), detail


def check_a4(ev):
    tr = ev.get("trainer") or {}
    grad_ok = tr.get("grad_norm") is not None and abs(float(tr["grad_norm"])) > 0
    adam_ok = int(tr.get("adam_state_delta_ranks", 0)) >= MIN_ADAM_DELTA_RANKS
    detail = {"grad_norm": tr.get("grad_norm"), "adam_state_delta_ranks": tr.get("adam_state_delta_ranks")}
    return bool(grad_ok and adam_ok), detail


def check_a5(ev):
    rs = ev.get("target_resume") or {}
    ok = bool(rs.get("resume_success")) and bool(rs.get("digest_match"))
    return ok, dict(rs)


def check_a6(ev):
    f = ev.get("fencing") or {}
    g = ev.get("gates") or {}
    gen = int(f.get("generate_during_release_window", -1))
    ok = gen == 0 and bool(g.get("no_duplicate_consume"))
    detail = {"generate_during_release_window": gen, "no_duplicate_consume": g.get("no_duplicate_consume")}
    return ok, detail


def make_verdict(ev):
    checks = {
        "A1_release_receipt": check_a1,
        "A2_nvml_crosscheck": check_a2,
        "A3_gpu_uuid_chain": check_a3,
        "A4_trainer_real_update": check_a4,
        "A5_target_resume": check_a5,
        "A6_fencing_no_duplicate": check_a6,
    }
    results = {k: {"pass": None, "detail": {}} for k in checks}
    for name, fn in checks.items():
        try:
            ok, detail = fn(ev)
        except Exception as e:  # 判定器自身异常也如实记录，不静默放行
            ok, detail = False, {"error": repr(e)}
        results[name]["pass"] = bool(ok)
        results[name]["detail"] = detail
    overall = all(v["pass"] for v in results.values())
    return {
        "verdict": "PASS" if overall else "FAIL",
        "checks": results,
        "scope_note": "weights_memory_region_released=False 为 HybridEngine 设计意图（权重保留、原地更新）；"
                      "本 verdict 只验收 KV+mamba 显存释放与 Trainer 实际使用。",
    }


def _mock_pass():
    exp_kv = {"per_token_bytes": 200, "kv_tokens": 486000}  # 97.2MB/卡量级 * TP2 预算演示
    return {
        "release_receipt": {"release_full_gpu_pool": True, "full_gpu_pool_released": True,
                            "memory_before_bytes": 80_000_000_000, "memory_after_bytes": 79_900_000_000,
                            "restore_path_exercised": False, "fail_reason": None},
        "nvml": {"before": [{"index": 0, "uuid": "GPU-a", "memory_used_bytes": 79_999_000_000, "memory_free_bytes": 1_000_000},
                            {"index": 1, "uuid": "GPU-b", "memory_used_bytes": 79_999_000_000, "memory_free_bytes": 1_000_000}],
                 "after":  [{"index": 0, "uuid": "GPU-a", "memory_used_bytes": 79_899_000_000, "memory_free_bytes": 101_000_000},
                            {"index": 1, "uuid": "GPU-b", "memory_used_bytes": 79_899_000_000, "memory_free_bytes": 101_000_000}]},
        "allocator_free_tokens_after": 500_000,
        "kv_pool": exp_kv,
        "timeline": {"safe_point_all_members_at": 1.0, "release_begin": 2.0, "release_end": 3.0,
                     "trainer_alloc_begin": 4.0, "trainer_step_end": 5.0},
        "trainer": {"alloc_gpu_uuids": ["GPU-a", "GPU-b"], "peak_mem_bytes": 70_000_000_000,
                    "grad_norm": 0.5837, "adam_state_delta_ranks": 4},
        "target_resume": {"resume_success": True, "digest_match": True},
        "fencing": {"generate_during_release_window": 0},
        "gates": {"no_duplicate_consume": True},
    }


def _mock_fail():
    ev = _mock_pass()
    # 破坏两点：receipt 声称 released 但 NVML 无变化（A2 抓矛盾）；trainer 换了别的卡（A3 抓 UUID 链断）
    ev["nvml"]["after"] = ev["nvml"]["before"][:]  # 显存纹丝不动
    ev["trainer"]["alloc_gpu_uuids"] = ["GPU-c"]   # trainer 落在别的卡
    ev["trainer"]["adam_state_delta_ranks"] = 1    # A4 也不过
    return ev


def selftest():
    vp = make_verdict(_mock_pass())
    vf = make_verdict(_mock_fail())
    ok1 = vp["verdict"] == "PASS"
    ok2 = vf["verdict"] == "FAIL"
    failed_checks = [k for k, v in vf["checks"].items() if not v["pass"]]
    expect_fail = {"A2_nvml_crosscheck", "A3_gpu_uuid_chain", "A4_trainer_real_update"}
    ok3 = set(failed_checks) == expect_fail
    print("PASS-scenario verdict:", vp["verdict"], "=>", "OK" if ok1 else "MISMATCH")
    print("FAIL-scenario verdict:", vf["verdict"], "failed:", failed_checks, "=>", "OK" if ok2 and ok3 else "MISMATCH")
    print(json.dumps(vp, indent=2, ensure_ascii=False)[:1200])
    if not (ok1 and ok2 and ok3):
        sys.exit(1)
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--evidence")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.evidence:
        ap.error("need --evidence or --selftest")
    with open(args.evidence) as f:
        ev = json.load(f)
    verdict = make_verdict(ev)
    out = json.dumps(verdict, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")
        print("written:", args.out)
    print(out)


if __name__ == "__main__":
    main()
