# VeRL Partial Rollout 验证指南

## 1. 当前验证边界

本目录包含两类验证，结论不能混用：

1. **纯 Python 控制面验证**：覆盖 snapshot、生命周期、版本失效、token-only/KV-handle 分支和回收，不需要 GPU。
2. **真实 SGLang 数据面验证**：必须在同一个 `rid`、同一个流式请求中执行中途 pause/resume，才能证明断点恢复。

`verify_e2e_real_sglang.py` 当前主要验证 VeRL manager、HTTP client 和 SGLang API 的编排。脚本中的 `kv_handle` 和 `recomputed_tokens` 是 VeRL 控制面模型值，并非 SGLang 的真实 block handle/runtime counter；其中 TTFT 对比只能证明 prefix cache 趋势，不能单独证明同请求 KV-aware resume。

真正的同请求验证以 SGLang 仓库的 `test/registered/rl/verify_partial_rollout.py` 为准。

## 2. 分支与资源

| 仓库 | 分支 | 用途 |
|---|---|---|
| `donghyq/verl` | `feat/partial-rollout` | 状态机、snapshot、HTTP bridge、控制面测试 |
| `donghyq/sglang` | `feat/partial-rollout-kv-resume` | scheduler paused queue、KV preservation、同 stream 验证 |

资源分层：

| 层级 | 验证目标 | 最低资源 |
|---|---|---|
| L1 | 状态机、save/load、版本失效、并发与回收 | CPU |
| L2 | 同 `rid`、同 stream pause/resume | 12GB 单卡 + 0.5B/1.5B 模型 |
| L3 | 真实 KV block、Host load-back、泄漏和 break-even | 目标单卡/多卡环境 |
| L4 | VeRL 训练 step 关键路径 | 代表性训练集群 |

12GB 显存足以完成 L1、L2 和部分 L3，但不能用于发布 235B 性能结论。

## 3. CPU 验证

在 VeRL 仓库运行：

```bash
cd ~/Projects/verl
git checkout feat/partial-rollout

python3 -m py_compile \
  verl/experimental/partial_rollout/rollout_state.py \
  verl/experimental/partial_rollout/partial_rollout_manager.py \
  verl/experimental/partial_rollout/snapshot_store.py \
  tests/experimental/partial_rollout/conftest.py

PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-project --with pytest pytest -q \
  tests/experimental/partial_rollout/test_snapshot_store.py \
  tests/experimental/partial_rollout/test_rollout_state.py \
  tests/experimental/partial_rollout/test_partial_rollout_manager.py
```

预期：以上三组测试全部通过。`conftest.py` 会隔离 `verl/__init__.py` 的 Ray/Torch 等仓库级依赖，因此这组测试只需 pytest。

如需运行整个实验目录：

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache \
uv run --no-project --with pytest pytest -q \
  tests/experimental/partial_rollout
```

## 4. 启动真实 SGLang Server

在远端 GPU 机器安装并切换 fork 分支：

```bash
git clone https://github.com/donghyq/sglang.git
cd sglang
git checkout feat/partial-rollout-kv-resume
pip install -e "python[all]"
```

使用小模型启动 HiCache server；具体参数以当前分支 `python -m sglang.launch_server --help` 为准：

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --port 30000 \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-write-policy write_through \
  --context-length 4096 \
  --mem-fraction-static 0.7
```

等待 server ready，并先检查：

```bash
curl -f http://127.0.0.1:30000/health
```

## 5. 同请求、同 stream 验证

在 SGLang 仓库运行：

```bash
python test/registered/rl/verify_partial_rollout.py \
  --host 127.0.0.1 \
  --port 30000
```

脚本必须完成以下真实序列：

```text
Request A（固定 rid=test-req-001）开始流式生成
-> 至少收到 5 个 chunk
-> pause Request A
-> 观察 1 秒，chunk 数不增长
-> resume 同一个 Request A
-> 原 HTTP stream 继续输出并完成
```

通过条件：

- pause 前请求尚未完成；
- pause 窗口没有新 chunk；
- resume 后 chunk 数继续增长；
- HTTP stream 和 `rid` 都没有更换；
- server 不崩溃。

还应补跑以下异常路径并保存日志：

- cancel paused request；
- duplicate resume；
- resume completed request；
- 多个 paused request 独立恢复；
- complete/cancel 后 scheduler queue 和 KV blocks 回到基线。

## 6. VeRL HTTP 编排 Smoke Test

SGLang server 运行后，在 VeRL 仓库执行：

```bash
cd ~/Projects/verl
pip install requests
python tests/experimental/partial_rollout/verify_e2e_real_sglang.py \
  --host 127.0.0.1 \
  --port 30000
```

该脚本覆盖：

- pause/save/resume/complete 控制面闭环；
- token-only 模型分支；
- model weight version mismatch；
- 多 rollout 状态隔离；
- HTTP pause/resume API 可达性。

不要把以下输出当作真实 KV 数据面证明：

- client 生成的 opaque `kv_handle`；
- client 计算的 `recomputed_tokens`；
- 完成请求后重发相同 prompt 得到的 TTFT 下降。

它们只能作为控制面或普通 prefix-cache 的辅助证据。

## 7. 必采 runtime 数据

真实 L3 实验每次保存：

| 分类 | 字段 |
|---|---|
| 环境 | 两仓 commit、GPU、CUDA、模型、dtype、TP/PP、完整启动参数 |
| 请求 | `rid`、weight version、prompt/generated length、pause position/count |
| 正确性 | sequence position、stream continuity、固定采样基线输出、fallback reason |
| 时间 | pause/save/restore/requeue/re-prefill latency、E2E P50/P95/P99 |
| KV | recomputed tokens、matched device/host tokens、saved/restored/evicted bytes |
| 容量 | free/used GPU blocks、Host blocks、request block table |
| 回收 | complete/cancel/expire 前后 queue/block 快照 |
| 训练 | step wall time、generation GPU-seconds、throughput、pipeline bubble |

`nvidia-smi` 只能观察进程级显存，不能证明内部 KV page 已释放；SGLang 通常预分配 KV pool。TTFT 下降也可能来自普通 prefix cache，不能替代同 stream 和 block counter。

## 8. Token-only / Resident / Host 对照

使用完全相同的请求和 pause trace 比较：

1. **Token-only**：丢弃 KV，恢复时重新 prefill；
2. **GPU resident**：停止 decode，但 KV page 保留在 GPU；
3. **Host offload**：保存到 Host，恢复时 load-back。

报告：

\[
NetSaving=C_{re-prefill}-(C_{save}+C_{restore}+C_{scheduler})
\]

同时报告 resident 的 GPU 容量机会成本、Host quota、eviction 和其他请求吞吐。跨 `model_weight_version` 默认只运行 token-only，不复用旧 KV。

## 9. 远端结果回传模板

```text
## Environment
- VeRL commit:
- SGLang commit:
- GPU / CUDA:
- Model / dtype / TP / PP:
- Server command:

## Same-stream correctness
- rid:
- chunks before pause:
- chunks after 1s pause:
- chunks after resume:
- sequence position before/after:
- baseline output match:

## Runtime counters
- recomputed tokens:
- matched device/host tokens:
- saved/restored/evicted bytes:
- free GPU blocks: before / paused / resumed / completed:
- used Host blocks: before / paused / resumed / completed:

## Latency
- pause/save/restore/requeue/re-prefill:
- P50/P95/P99:

## Reclaim
- complete result:
- cancel result:
- expire result:
- leak observed:

## VeRL（如果运行主流程）
- pause count/position/duration distribution:
- resume/cancel/cross-version ratio:
- step wall time:
- generation GPU-seconds:
```

请回传原始 JSON/CSV/W&B run 和 server 日志，不只提供截图。只有同请求正确性、真实 counters 和回收证据齐全后，才进入 VeRL 主流程 token-only 接入；KV-aware 仅作为同权重条件分支。
