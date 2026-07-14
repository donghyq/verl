# VeRL Partial Rollout — 验证结果记录

> 首次验证日期：2026-07-15
> 验证环境：WSL Ubuntu 26.04 + RTX 4070 SUPER (12GB)

---

## 验证环境

| 项目 | 详情 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12282 MiB |
| 模型 | Qwen2.5-7B-Instruct-GPTQ-Int4 |
| SGLang 分支 | `feat/partial-rollout-kv-resume` (sglang-fork) |
| VeRL 分支 | `feat/partial-rollout` (verl) |
| HiCache | 启用, ratio=0.3, write-policy=write_through |
| Python | 3.12 (conda env: sglang-env) |
| PyTorch | 2.9.1+cu128 |

---

## 验证结果汇总

| # | 测试项 | 结果 | 关键数据 |
|---|---|---|---|
| 1 | HiCache 启用确认 | ✅ PASS | `enable_hierarchical_cache=True` |
| 2 | 全局 pause_generation (preserve_kv) | ✅ PASS | 暂停→恢复后正常生成 |
| 3 | Per-request pause_request | ❌ FAIL | Server 崩溃（见 Bug #1） |
| 4 | GPU 显存变化 | ⚠️ 不明显 | 变化仅 ~45MB（KV pool 预分配） |
| 5 | 前缀缓存命中（HiCache） | ✅ PASS | TTFT 加速 **15x** (0.438s → 0.029s) |
| 6 | preserve_kv TTFT 比值 | ✅ PASS | **0.66x**（阈值 < 0.8） |
| 7 | VeRL E2E 功能测试 | ❌ FAIL | pause 导致 server 崩溃（同 Bug #1） |
| 8 | VeRL health_check | ✅ 已修复 | 原为 JSON 解析错误（见 Bug #2） |

### 最关键指标

**TTFT 比值 = 0.66x** ✅

> preserve_kv 暂停后，相同 prompt 的 TTFT 从首次的 0.0465s 降至 0.0306s，比值 0.658 < 0.8。
> 说明 KV cache 被正确保留在 Host 层，恢复时通过 HiCache 的 match_prefix 命中，无需重新 prefill。

---

## 发现的 Bug

### Bug #1: HiRadixCache + per-request pause 断言失败

**严重程度**：高（导致 server 崩溃）

**现象**：
调用 `/pause_request` 或 VeRL E2E 测试中的 pause 操作时，server 进程崩溃，报错：

```
AssertionError: Only MambaRadixCache allow freeing before alloc
```

**位置**：
- SGLang: `python/sglang/srt/mem_cache/common.py` 第 637 行
- 调用路径：`schedule_batch.py:release_req()` → `req.radix_cache.release_kv_cache()`

**根因**：
`release_kv_cache` 方法中有一个断言：

```python
assert self.supports_mamba(), (
    "Only MambaRadixCache allow freeing before alloc"
)
```

该断言仅在 `req_pool_idx is None` 时触发。per-request pause 流程中，`write_backup` 完成后调用 `release_req`，此时 `req.req_pool_idx` 为 None（per-request 不走 req_pool），于是进入了 mamba-only 分支。但 `HiRadixCache` 没有实现 `supports_mamba()` 方法（或返回 False），断言失败。

**影响范围**：
- per-request pause（`/pause_request` 端点）— 完全不可用
- VeRL E2E 测试 — 因依赖 per-request pause 而失败
- 全局 pause（`/pause_generation`）— 不受影响，可以正常工作

**修复方向**：
- 在 `HiRadixCache` 中实现正确的 `release_kv_cache` 逻辑，或
- 修改 `release_req` 中 preserve_kv 路径的释放条件，避免触发 mamba-only 断言

---

### Bug #2: VeRL health_check JSON 解析错误

**严重程度**：低（已修复）

**现象**：
`RealSGLangServer.health_check()` 永远返回 False，即使 server 正常运行。

**位置**：
- VeRL: `verl/experimental/partial_rollout/real_sglang_client.py` 第 197 行

**根因**：
`health_check` 调用了 `self._get("/health")`，而 `_get` 方法内部会执行 `resp.json()`。但 SGLang 的 `/health` 端点返回的是非 JSON 响应（纯文本 "ok"），导致 JSONDecodeError 被 except 吞掉，返回 False。

**修复**：
```python
def health_check(self) -> bool:
    try:
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        return resp.status_code == 200
    except Exception:
        return False
```

**状态**：已在本地修复

---

## 其他发现

### 1. KV pool 预分配导致显存变化不明显

`verify_gpu_memory.py` 中观察到暂停前后显存变化仅 ~45MB，远小于预期。原因是 SGLang 的 KV cache pool 是预分配的（`mem_fraction_static`），pause 释放的是 pool 内的 slot 而非实际显存。验证显存收益需要换一种方式：

- 比较 pause 前后可同时服务的并发请求数
- 或直接测量 GPU memory pool 中 free/total 的变化

### 2. sglang-kernel 版本兼容性

`sglang-kernel 0.4.1.post1` 与 `torch 2.9.1` 搭配使用时，需要设置：
```bash
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
```
否则版本检查会失败（要求 torch >= 2.11）。

### 3. flashinfer 版本兼容性

`flashinfer-python 0.6.8.post1` 与 sglang fork 分支搭配使用时，需要设置：
```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

---

## 未来建议

### 短期（P0 — 提交前必须完成）

1. **修复 Bug #1** — per-request pause 的断言失败是核心阻塞问题，不修则 VeRL 集成无法工作
2. **修复 VeRL E2E 测试中的 pause 调用** — 确认使用 `/pause_generation` 还是 `/pause_request`，与 SGLang 侧能力对齐
3. **补充 per-request pause 的单元测试** — SGLang 侧需要有针对 HiRadixCache 的测试覆盖

### 中期（P1 — 功能完善）

4. **与真实 rollout 流程联动**
   - 修改 `verl/workers/rollout/sglang_rollout/sglang_rollout.py` 中的 adapter，支持 partial rollout
   - 修改 `verl/trainer/ray_trainer.py` 的训练循环，在 training step 前暂停 rollout，完成后恢复
   - 真正展示端到端的训练吞吐收益

5. **错误重试与容错**
   - `real_sglang_client.py` 中 HTTP 调用缺少 retry 机制
   - pause/resume 失败时的降级策略（如退化为 token-only resume）

6. **性能基准测试**
   - 测量不同前缀长度下的 TTFT 收益曲线
   - 测量 pause/resume 的 overhead
   - 与训练 step 时间对比，评估整体吞吐提升

### 长期（P2 — 生产化）

7. **多 GPU / 多节点支持**
   - 当前设计假设单 GPU KV cache，需要验证 tensor parallel 场景
   - KV snapshot 的分布式存储

8. **与 reward model 流水线集成**
   - 在 reward 计算阶段释放 KV cache
   - reward 完成后恢复生成

9. **可观测性**
   - 暴露 pause/resume 的 metrics（次数、成功率、overhead）
   - KV cache 命中率监控

10. **snapshot_store 修复**
    - `list_paused` 中 `replace("_", "/")` 的双向映射 bug
    - 支持 request_id 中包含下划线的场景

---

## 复现命令

### 启动 SGLang Server

```bash
# 环境变量（绕过版本检查）
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# 启动
python -m sglang.launch_server \
    --model-path ~/models/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --port 30000 \
    --quantization gptq_marlin \
    --enable-hierarchical-cache \
    --hicache-ratio 0.3 \
    --hicache-write-policy write_through \
    --context-length 4096 \
    --mem-fraction-static 0.7 \
    --log-level info
```

### 运行验证

```bash
# SGLang 侧 — 基础功能（全局 pause 通过，per-request 崩溃）
python test/registered/rl/verify_partial_rollout.py --port 30000

# SGLang 侧 — TTFT 对比（全部通过，比值 0.66x）
python test/registered/rl/verify_kv_reuse.py --hicache-url http://localhost:30000 --skip-nocache

# SGLang 侧 — 显存验证（变化不明显）
python test/registered/rl/verify_gpu_memory.py --port 30000

# VeRL 侧 — E2E（因 Bug #1 失败）
python tests/experimental/partial_rollout/verify_e2e_real_sglang.py --port 30000
```
