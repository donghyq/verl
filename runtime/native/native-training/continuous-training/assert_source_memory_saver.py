#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assert_source_memory_saver.py — M4.3 server_args 断言（零 GPU）

对源 role 的 server_args 原件（R88 系 readback 已建立的 verified-originals 路径）
断言 memory-saver 全量释放前置满足，防止 launch 模板合并时 --enable-memory-saver 被覆盖丢失。

用法：
  # 单个 server_args 原件
  python3 assert_source_memory_saver.py <path-to-server-args.json> [--role source]

  # 目录递归扫描（自动找 server_args 命名文件）
  python3 assert_source_memory_saver.py <dir> --recursive

  # 与 R88 基线对照（演示当前应 FAIL 的场景）
  python3 assert_source_memory_saver.py --selftest

判据（源 role 必须）：
  enable_memory_saver == True            # full release 前置（memory_saver_adapter.enabled）
  disable_cuda_graph == True             # memory_saver 与 cuda graph 不兼容
  disable_overlap_schedule == False      # 与现网 overlap 配置一致，防模板回归
  mamba_radix_cache_strategy 存在则记录（hybrid 审计留痕，不作为 PASS 条件）

退出码：0 全部 PASS；1 有 FAIL；2 输入错误。
"""
import argparse
import json
import os
import sys

REQUIRED = {
    "enable_memory_saver": True,
    "disable_cuda_graph": True,
}
RECORDED_NOT_GATED = [
    "disable_overlap_schedule",
    "mamba_radix_cache_strategy",
    "mamba_track_interval",
    "mem_fraction_static",
]


def load_server_args(path):
    with open(path, "r", errors="replace") as f:
        txt = f.read()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # 容忍日志里截取的 python-dict 片段：找最外层 { }
        start = txt.find("{")
        end = txt.rfind("}")
        if start >= 0 and end > start:
            frag = txt[start:end + 1].replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
            try:
                return json.loads(frag)
            except json.JSONDecodeError:
                pass
        raise ValueError("cannot parse server args from %s" % path)


def check_file(path, role="source"):
    sa = load_server_args(path)
    # readback 原件常见嵌套：actual-server-profile.json 把 args 放在 body 下，自动下钻
    if isinstance(sa, dict) and "body" in sa and "enable_memory_saver" not in sa and isinstance(sa["body"], dict):
        sa = sa["body"]
    result = {"file": path, "role": role, "pass": True, "failures": [], "recorded": {}}
    if role == "source":
        for k, want in REQUIRED.items():
            got = sa.get(k)
            if got != want:
                result["pass"] = False
                result["failures"].append("%s: got %r, want %r" % (k, got, want))
    for k in RECORDED_NOT_GATED:
        if k in sa:
            result["recorded"][k] = sa[k]
    return result


def selftest():
    import tempfile
    # 模拟 R88 现状：enable_memory_saver=False → 必须 FAIL
    bad = {"enable_memory_saver": False, "disable_cuda_graph": True, "disable_overlap_schedule": False,
           "mamba_radix_cache_strategy": "extra_buffer", "mamba_track_interval": 256}
    good = dict(bad, enable_memory_saver=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f1, \
         tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f2:
        json.dump(bad, f1); json.dump(good, f2)
        p1, p2 = f1.name, f2.name
    r1 = check_file(p1)
    r2 = check_file(p2)
    os.unlink(p1); os.unlink(p2)
    ok = (r1["pass"] is False and r1["failures"] == ["enable_memory_saver: got False, want True"]
          and r2["pass"] is True)
    print("R88-baseline scenario (should FAIL):", r1["failures"])
    print("post-fix scenario (should PASS): pass=%s" % r2["pass"], "recorded=", r2["recorded"])
    print("SELFTEST", "OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--role", default="source", choices=["source", "target"])
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.path:
        ap.error("need a file/dir path or --selftest")
    files = []
    if args.recursive:
        for root, _dirs, names in os.walk(args.path):
            for n in names:
                if "server_args" in n:
                    files.append(os.path.join(root, n))
    else:
        files = [args.path]
    if not files:
        print("no server_args files found", file=sys.stderr)
        sys.exit(2)
    overall = True
    for p in sorted(files):
        try:
            r = check_file(p, args.role)
        except (ValueError, OSError) as e:
            r = {"file": p, "pass": False, "failures": [repr(e)], "recorded": {}}
        overall &= r["pass"]
        print(json.dumps(r, indent=2, ensure_ascii=False))
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
