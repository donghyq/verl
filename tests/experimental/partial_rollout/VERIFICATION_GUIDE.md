# VeRL Partial Rollout — 验证指南

## 目标

验证 VeRL 的 `PartialRolloutManager` + `SGLangPartialRolloutCoordinator` 能驱动真实 SGLang server 完成完整的 pause → save → resume → complete 闭环，并收集关键性能数据。

## 架构

```
VeRL (this repo)                        SGLang (feat/partial-rollout-kv-resume)
┌─────────────────────────┐             ┌──────────────────────────┐
│ PartialRolloutManager   │             │  Scheduler               │
│   - RolloutState        │             │    - pause_generation    │
│   - RolloutStateSnapshot│             │    - pause_request       │
│   - validate_resume()   │             │    - resume_request      │
│                         │  HTTP       │                         │
│ SGLangPartialRollout    │◄───────────►│  HiRadixCache           │
│   Coordinator           │  /pause     │    - write_backup (GPU→Host)
│                         │  /resume    │    - load_back (Host→GPU) │
│ RealSGLangServer        │  /generate  │    - match_prefix         │
│   (HTTP client)         │             │                         │
└─────────────────────────┘             └──────────────────────────┘
```

## 前提条件

### 两个仓库

| 仓库 | 分支 | 作用 |
|---|---|---|
| `~/Projects/sglang` | `feat/partial-rollout-kv-resume` | SGLang server（pause/resume + KV preservation） |
| `~/Projects/verl` | `feat/partial-rollout` | VeRL manager + E2E 验证脚本 |

### 安装

```bash
# SGLang
cd ~/Projects/sglang
git checkout feat/partial-rollout-kv-resume
pip install -e "python[all]"

# VeRL（不需要完整安装，E2E 脚本直接加载模块）
cd ~/Projects/verl
git checkout feat/partial-rollout
pip install requests

# 模型
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/models/Qwen2.5-0.5B-Instruct
```

---

## 验证步骤

### 第一步：启动 SGLang Server

```bash
cd ~/Projects/sglang

python -m sglang.launch_server \
    --model-path ~/models/Qwen2.5-0.5B-Instruct \
    --port 30000 \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_through \
    --context-length 4096 \
    --mem-fraction-static 0.7
```

等待 `The server is fired up and ready to roll!`。

### 第二步：运行 VeRL E2E 验证

```bash
cd ~/Projects/verl
python tests/experimental/partial_rollout/verify_e2e_real_sglang.py --port 30000
```

### 第三步（可选）：带对照 server 运行

```bash
# 另一个终端启动不带 HiCache 的 server
python -m sglang.launch_server \
    --model-path ~/models/Qwen2.5-0.5B-Instruct \
    --port 30001 \
    --context-length 4096 \
    --mem-fraction-static 0.7

# 带对照运行
python tests/experimental/partial_rollout/verify_e2e_real_sglang.py \
    --port 30000 --compare-port 30001
```

---

## E2E 脚本测试内容

`verify_e2e_real_sglang.py` 跑 5 个测试：

### Test 1: 基本闭环（KV-aware）

验证完整的 pause → save → resume → complete 流程，KV handle 保留：

```
生成请求 → start_rollout(kv_handle) → pause_and_save → resume_from_snapshot → complete
```

预期：
- `recomputed_tokens = 0`（KV 复用，无需重算）
- `kv_handle` 保留
- lifecycle 最终为 `completed`

### Test 2: Token-only resume

验证释放 KV 后的 token-only resume：

```
start_rollout → pause_and_save → release_memory → resume_from_snapshot(kv_handle=None) → complete
```

预期：
- `recomputed_tokens = 5`（等于 generated_ids 长度）
- `kv_handle = None`
- server 侧 KV 被释放

### Test 3: 权重版本失效

验证权重更新后旧 snapshot 无法 resume：

```
start_rollout(v0) → pause_and_save → update_model_version(v1) → resume_from_snapshot → 应抛 InvalidResumeError
```

预期：
- 抛出 `InvalidResumeError`（model_weight_version mismatch）
- 旧状态被标记为 `INVALIDATED`

### Test 4: TTFT 对比（关键性能指标）

验证 preserve_kv 暂停后重新发送相同 prompt，TTFT 是否低于首次：

```
generate_stream(prompt) → 记录 TTFT1
pause_generation(preserve_kv) → continue_generation
generate_stream(同prompt) → 记录 TTFT2
比较 TTFT2 / TTFT1
```

预期：
- 比值 < 0.8 → KV 复用生效（Host 层命中）
- 比值 >= 0.8 → KV 未被保留或 match_prefix 未命中

### Test 5: 并发 rollout

验证多个 rollout 同时暂停、混合 resume（KV-aware + token-only）互不干扰：

```
start 3 rollouts → pause all → resume #0 (KV-aware) → release+resume #1 (token-only) → resume #2 (KV-aware) → complete all
```

预期：
- 3 个 rollout 独立完成
- KV-aware 的 recomputed_tokens = 0
- token-only 的 recomputed_tokens = len(generated_ids)
- 最终 active_count = 0

---

## 需要收集的结果

跑完后把以下结果贴回来：

### 结果模板

```
## 验证环境
- GPU: [型号, 显存]
- 模型: Qwen2.5-0.5B-Instruct
- SGLang: feat/partial-rollout-kv-resume @ [commit hash]
- VeRL: feat/partial-rollout @ [commit hash]
- HiCache: [启用/未启用, hicache-ratio=X]

## E2E Verification Summary
[贴 verify_e2e_real_sglang.py 的 Summary 输出]

## Test 4 TTFT 数据
- 首次 TTFT: 0.XXX s
- 恢复后 TTFT: 0.XXX s
- 比值: 0.XXx
- 判定: [命中/未命中]

## Server 日志（可选）
[贴含 preserve_kv / write_backup / pause_request 的行]

## 模拟层测试（本地已验证）
- 88 tests: ALL PASSED
```

### 关键判据

| 数据点 | 通过标准 | 不通过怎么办 |
|---|---|---|
| Test 1 recomputed_tokens | = 0 | 检查 write_backup 是否被调用 |
| Test 2 recomputed_tokens | = 5 | 检查 release_memory 是否生效 |
| Test 3 InvalidResumeError | 抛出 | 检查 validate_resume 逻辑 |
| Test 4 TTFT 比值 | < 0.8 | **最关键** — 见下方排查 |
| Test 5 并发 | 全部完成 | 检查 manager 状态隔离 |
| 模拟层 88 tests | 全通过 | 本地 `python3 -c "..."` 重跑 |

**Test 4 是 go/no-go 判据**：如果 TTFT 没降，说明 `write_backup` + `match_prefix` 链路没打通，需要先修 SGLang 侧。

---

## 本地模拟层验证（不需要 GPU）

在跑真实 server 之前，先验证 VeRL 侧的逻辑正确性：

```bash
cd ~/Projects/verl

# py_compile
python3 -m py_compile \
  verl/experimental/partial_rollout/*.py \
  tests/experimental/partial_rollout/*.py

# 模拟层测试（用 FakeSGLangServer，不需要 GPU/SGLang）
python3 -c "
import sys, types, importlib, importlib.util, traceback, os, tempfile, inspect
from pathlib import Path

ROOT = os.getcwd()

pytest_mod = types.ModuleType('pytest')
class _RaisesCtx:
    def __init__(self, exc, match=None):
        self.exc = exc; self.match = match; self.value = None
    def __enter__(self): return self
    def __exit__(self, et, ev, eb):
        if et is None: raise AssertionError(f'Did not raise {self.exc.__name__}')
        if not issubclass(et, self.exc): return False
        self.value = ev
        if self.match:
            import re
            if not re.search(self.match, str(ev)):
                raise AssertionError(f'{ev!r} does not match {self.match!r}')
        return True
def raises(exc, match=None): return _RaisesCtx(exc, match)
class _Mark:
    def __getattr__(self, n):
        def d(f): return f
        return d
pytest_mod.raises = raises; pytest_mod.mark = _Mark()
sys.modules['pytest'] = pytest_mod

def load(fp, mn):
    spec = importlib.util.spec_from_file_location(mn, fp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mn] = mod
    spec.loader.exec_module(mod)
    return mod

for pkg in ['verl', 'verl.experimental', 'verl.experimental.partial_rollout']:
    m = types.ModuleType(pkg)
    m.__path__ = [os.path.join(ROOT, pkg.replace('.', '/'))]
    sys.modules[pkg] = m

for fp, mn in [
    ('verl/experimental/partial_rollout/rollout_state.py', 'verl.experimental.partial_rollout.rollout_state'),
    ('verl/experimental/partial_rollout/partial_rollout_manager.py', 'verl.experimental.partial_rollout.partial_rollout_manager'),
    ('verl/experimental/partial_rollout/sglang_integration.py', 'verl.experimental.partial_rollout.sglang_integration'),
    ('verl/experimental/partial_rollout/sglang_adapter_bridge.py', 'verl.experimental.partial_rollout.sglang_adapter_bridge'),
    ('verl/experimental/partial_rollout/snapshot_store.py', 'verl.experimental.partial_rollout.snapshot_store'),
    ('verl/experimental/partial_rollout/e2e_simulation.py', 'verl.experimental.partial_rollout.e2e_simulation'),
]:
    load(fp, mn)

class R:
    passed = 0; failed = 0; errors = []

def inject(fn):
    sig = inspect.signature(fn)
    kwargs = {}
    for p in sig.parameters:
        if p == 'tmp_path':
            kwargs[p] = Path(tempfile.mkdtemp(prefix='ptest_'))
    return fn(**kwargs)

for fp, mn in [
    ('tests/experimental/partial_rollout/test_rollout_state.py', 't1'),
    ('tests/experimental/partial_rollout/test_partial_rollout_manager.py', 't2'),
    ('tests/experimental/partial_rollout/test_sglang_integration.py', 't3'),
    ('tests/experimental/partial_rollout/test_sglang_adapter_bridge.py', 't4'),
    ('tests/experimental/partial_rollout/test_snapshot_store.py', 't5'),
    ('tests/experimental/partial_rollout/test_e2e_simulation.py', 't6'),
]:
    try: mod = load(fp, mn)
    except Exception as e:
        R.errors.append(f'IMPORT {mn}: {e}'); traceback.print_exc(); continue
    for name in sorted(dir(mod)):
        obj = getattr(mod, name)
        if isinstance(obj, type) and name.startswith('Test'):
            inst = obj()
            for m2 in sorted(dir(inst)):
                if m2.startswith('test_') and callable(getattr(inst, m2)):
                    try:
                        inject(getattr(inst, m2)); R.passed += 1
                    except Exception as e:
                        R.failed += 1; R.errors.append(f'{name}.{m2}'); traceback.print_exc()

print(f'\n=== {R.passed} passed, {R.failed} failed ===')
if R.errors:
    for e in R.errors: print(f'  FAIL: {e}')
    sys.exit(1)
else: print('ALL TESTS PASSED')
"
```

预期：`88 passed, 0 failed`

---

## 脚本说明

| 脚本 | 位置 | 需要 GPU | 用途 |
|---|---|---|---|
| `verify_e2e_real_sglang.py` | `tests/experimental/partial_rollout/` | 是 | VeRL + 真实 SGLang 端到端 |
| `real_sglang_client.py` | `verl/experimental/partial_rollout/` | - | HTTP client（被 E2E 脚本调用） |
| 模拟层测试（6 个文件） | `tests/experimental/partial_rollout/` | 否 | 纯 Python 逻辑验证 |
| `verify_partial_rollout.py` | SGLang 仓库 `test/registered/rl/` | 是 | SGLang 侧基础功能 |
| `verify_kv_reuse.py` | SGLang 仓库 `test/registered/rl/` | 是 | SGLang 侧 TTFT 对比 |
| `verify_gpu_memory.py` | SGLang 仓库 `test/registered/rl/` | 是 | SGLang 侧显存验证 |

---

## 故障排查

### E2E 脚本连接失败

```bash
# 确认 server 在运行
curl http://localhost:30000/health

# 确认端口
python tests/experimental/partial_rollout/verify_e2e_real_sglang.py --port <正确端口>
```

### Test 4 TTFT 没有下降

这是最可能遇到的问题。排查步骤：

1. **确认 preserve_kv 分支被执行**
   ```bash
   grep "preserve_kv" /path/to/sglang/server.log
   ```
   如果没有日志，确认 SGLang 分支正确。

2. **确认 write_backup 被调用**
   ```bash
   grep "write_backup" /path/to/sglang/server.log
   ```
   如果没有，可能是 HiCache 未启用或 `req.last_node` 为 None。

3. **确认 HiCache 启用**
   ```bash
   grep -i "hierarchical" /path/to/sglang/server.log
   ```

4. **增大 HiCache 比例**
   ```bash
   --hicache-ratio 4.0
   ```

5. **如果 write_backup 失败**
   日志会有 `write_backup failed`，说明 HiCache Host pool 未初始化或已满。

### ImportError

```bash
# E2E 脚本不需要完整 verl 环境，但需要 requests
pip install requests

# 在 verl 仓库根目录运行
cd ~/Projects/verl
python tests/experimental/partial_rollout/verify_e2e_real_sglang.py --port 30000
```

### 模拟层测试失败

模拟层测试不需要 GPU，如果失败说明 VeRL 代码有 bug。先修模拟层再跑 E2E。
