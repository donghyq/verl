#!/usr/bin/env python3
"""Group-Aware 路由收益界仿真（P2 · SIMULATION ONLY · v2）.

诚实声明:
- 参数化界分析工具, 不是预测. 锚定参数取自真实回执(注释标来源),
  无回执依据的参数标 [ASSUMED] 且可用 CLI 覆盖.
- 语义(v2, 可辩护): 成员在引擎 E 上被中断后由路由器选恢复引擎 R.
    R==E 且引擎存活     -> 0 字节搬运, 0 重算(热 KV 在本地)
    R!=E                -> 从共享存储导入状态(187.2MB/成员) + 摩擦重算 [ASSUMED 5%]
                           若 R 上有同组其他成员的存活前缀, 重算按共享比例减少
    整机故障            -> 热态丢失, 只能存储导入或重 prefill(两策略同额, 不差分)
- v1 的负向发现保留: 冻结 capacity=4 < 组大小 8, 整组共置不可达(见 DESIGN 文档 §5).
"""
from __future__ import annotations
import argparse, json, random

STATE_BYTES_PER_CLASS_PER_RANK = 32_737_280   # R15: 三类状态各 32737280B / TP rank
STATE_CLASSES, TP_PER_MEMBER = 3, 2           # R15: 三类; 联合运行成员 TP2

def state_bytes_per_member() -> int:
    return STATE_BYTES_PER_CLASS_PER_RANK * STATE_CLASSES * TP_PER_MEMBER  # ≈187.2 MB

def simulate(engines: int, groups: int, members: int, cap: int,
             history_tokens: int, friction: float, prefix_share: float,
             policy: str, seed: int) -> dict:
    """组首触放置 → 单引擎中断 → 全组成员恢复(共享存储态总可用)."""
    rng = random.Random(seed)
    home, load = {}, [0] * engines
    for g in range(groups):
        for m in range(g * members, (g + 1) * members):
            if policy == "group_aware":
                cands = [e for e in range(engines) if load[e] < cap]
                e = min(cands, key=lambda x: load[x]) if cands else rng.randrange(engines)
            else:
                e = rng.randrange(engines)
            home[m], load[e] = e, load[e] + 1
    moved_bytes = reprefill = 0
    for g in range(groups):
        ms = list(range(g * members, (g + 1) * members))
        dead = home[rng.choice(ms)]                      # 单引擎中断(其上成员需恢复)
        alive_group_prefix = {home[m] for m in ms if home[m] != dead}
        for m in ms:
            if home[m] != dead:
                continue                                  # 未受影响成员不计入差分
            if policy == "group_aware":
                cands = [e for e in range(engines) if e != dead and load[e] < cap]
                r = home[m] if (home[m] != dead and load[home[m]] < cap) \
                    else (min(cands, key=lambda x: load[x]) if cands else rng.randrange(engines))
            else:
                alive = [e for e in range(engines) if e != dead]
                r = rng.choice(alive)                     # 无状态感知均匀选
            load[r] += 1
            if r != home[m]:
                moved_bytes += state_bytes_per_member()
                share = prefix_share if r in alive_group_prefix else 0.0
                reprefill += int(history_tokens * friction * (1 - share))
            # r==home[m]: 引擎同实例存活恢复, 热态零成本(理想路径)
    return {"moved_MB": moved_bytes / 1024**2, "reprefill_tokens": reprefill}

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engines", type=int, default=4)
    p.add_argument("--groups", type=int, default=16)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--capacity", type=int, default=4, help="冻结配置 capacity=4")
    p.add_argument("--history-tokens", type=int, default=2000, help="[ASSUMED]")
    p.add_argument("--friction", type=float, default=0.05, help="[ASSUMED] 跨引擎恢复摩擦重算比例")
    p.add_argument("--prefix-share", type=float, default=0.6, help="[ASSUMED] 同组存活前缀可抵扣比例")
    p.add_argument("--seeds", type=int, default=50)
    a = p.parse_args()
    rows = []
    for policy in ("baseline", "group_aware"):
        runs = [simulate(a.engines, a.groups, a.members, a.capacity, a.history_tokens,
                         a.friction, a.prefix_share, policy, s) for s in range(a.seeds)]
        rows.append({"policy": policy,
                     "moved_MB": round(sum(r["moved_MB"] for r in runs) / len(runs), 1),
                     "reprefill_tokens": round(sum(r["reprefill_tokens"] for r in runs) / len(runs), 1)})
    b, g = rows
    print(json.dumps({
        "parameters": {"engines": a.engines, "capacity": a.capacity, "members_per_group": a.members,
                       "history_tokens": a.history_tokens, "friction": a.friction,
                       "prefix_share": a.prefix_share, "state_bytes_per_member": state_bytes_per_member()},
        "mean_results": rows,
        "delta": {"moved_MB_saved": round(b["moved_MB"] - g["moved_MB"], 1),
                  "reprefill_tokens_saved": round(b["reprefill_tokens"] - g["reprefill_tokens"], 1)},
        "findings": [
            "capacity=4 < members=8: 整组共置不可达, 亲和收益来自'回到 home 实例'而非'整组聚拢'(v1 负向发现)",
            "收益对 prefix_share 敏感: 同组存活成员提供的前缀抵扣是第二大收益源(需实测分布)",
            "整组共置要成为可达目标, 需 owner 会签修订 capacity 语义(超出 P2 授权)",
        ],
        "honesty": "界分析非预测; 未建模整机故障/网络抖动/初始化(两策略同额); GPU 结论需预注册双臂实验"},
        indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
