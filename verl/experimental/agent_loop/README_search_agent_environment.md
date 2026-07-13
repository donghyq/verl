# Search Agent Environment Prototype

## 项目定位

这是 VeRL `experimental/agent_loop` 下的第一阶段搜索环境原型，目标是给后续 Agentic RL、Reward Infrastructure、Partial Rollout 和 Retrieval-KV 研究提供**真实、可回放、可测试**的环境层基础。

当前阶段只覆盖 M0/M1：

- 真实接口核对
- Search Environment Contract
- mock / snapshot 搜索环境
- 最小多轮 Search Agent Loop
- trajectory metadata
- deterministic reward baseline

当前状态：`environment_prototype`

## 与普通 Search-R1 复现的区别

这里的重点不是单纯让模型会输出 `<search>...</search>`，而是：

- 版本化 SearchAction / Observation / Snapshot 协议
- 可回放环境结果
- 稳定 response hash / content hash
- timeout / failure / invalid-action 的结构化 fallback
- 与 VeRL 现有 `response_mask` 语义兼容

## Search Environment Contract

位置：`verl.experimental.agent_loop.search_environment`

核心结构：

- `SearchAction`
- `SearchDocument`
- `SearchCost`
- `SearchObservation`
- `SearchTrace`
- `SearchSnapshotRecord`

约束：

- `query` 非空
- `query` 长度上限
- `top_k` 上下界
- `recall_profile` allowlist
- 未知字段 fail-closed
- 非法动作不抛出批量级异常，而是转成结构化 observation

## mock / snapshot 模式

已实现两类后端：

1. `InMemorySearchAdapter`
   - 轻量 lexical / explicit-result mock
   - 确定性输出
   - 支持 empty / degraded / timeout / failure 注入

2. `SnapshotReplaySearchAdapter`
   - 从 JSON / JSONL 回放
   - 按稳定 action identity 查找 observation
   - 校验 schema version / index version / response hash

M2 最小新增：

3. `RecordingSearchAdapter`
   - 包装底层 adapter
   - 自动记录 action -> observation
   - 产出 `SearchSnapshotRecord`

4. `SearchSnapshotStore`
   - 内存态 snapshot 收集
   - 支持 dump / load 轻量工作流

此外，`SearchAgentLoop` 现在支持通过配置直接启用 record mode，例如：

```yaml
actor_rollout_ref:
  rollout:
    custom:
      search_agent:
        record_mode: true
        record_snapshot_id: search-agent-snapshot
        record_snapshot_path: /tmp/search_agent_snapshot.json
```

## Agent Loop 数据流

位置：`verl.experimental.agent_loop.search_agent_loop`

数据流：

```text
raw prompt
-> apply_chat_template
-> LLM token-in token-out generate
-> parse <search> / <answer>
-> search adapter
-> render observation text
-> append environment tokens
-> continue or stop
```

## mask 语义

- `response_mask = 1`：LLM 生成的 action token
- `response_mask = 0`：搜索环境返回的 observation token

准确语义是：environment token 不是 policy action，不参与 actor loss。

## trajectory metadata

当前通过 `AgentLoopOutput.extra_fields` 记录：

- `trajectory_id` / `request_id`
- `search_traces`
- `search_trace_summary`
- `num_searches`
- `stop_reason`
- `renderer_version`
- `environment_token_ranges`
- `action_token_ranges`
- query / top_k / recall_profile
- doc ids / content hashes / index version / response hash
- latency / degraded / error / SearchCost

## baseline reward components

位置：`verl.experimental.agent_loop.search_reward`

当前 deterministic baseline 组件：

- `answer_correctness`
- `groundedness`
- `retrieval_utility`
- `search_latency_cost`
- `search_resource_cost`
- `invalid_action_penalty`
- `failure_penalty`
- `format_penalty`

说明：这只是 Reward Infrastructure 的第一阶段数据契约，不是 PPO / GRPO 算法实现。

M2 最小新增：

- `evaluate_search_reward_batch(...)`
- `summarize_search_reward_batch(...)`

用于对一组 trajectory metadata 做 deterministic batch 评估与聚合摘要，仍然不修改训练主流程。

## 运行测试

```bash
python3 -m pytest tests/experimental/agent_loop/test_search_environment_on_cpu.py
python3 -m pytest tests/experimental/agent_loop/test_search_agent_loop_on_cpu.py
python3 -m pytest tests/experimental/agent_loop/test_search_reward_on_cpu.py
```

## 测试 TODO / 当前阻断

在将该原型对外表述为“已完成 M1 验证”之前，仍需完成以下真实验证：

1. 建立包含下列依赖的本地测试环境：
   - `pytest`
   - `pytest-asyncio`
   - `ray`
   - `torch`
   - `tensordict`
   - `transformers`
   - `codetiming`

2. 真实执行并记录结果：
   - `tests/experimental/agent_loop/test_search_environment_on_cpu.py`
   - `tests/experimental/agent_loop/test_search_agent_loop_on_cpu.py`
   - `tests/experimental/agent_loop/test_search_reward_on_cpu.py`

3. 真实执行 dry-run harness：

```bash
PYTHONPATH=/path/to/verl python3 examples/tutorial/agent_loop_get_started/search_agent_dry_run.py
```

4. 依赖环境就绪后，补跑至少一条相关现有 agent-loop 回归测试。

5. 对本轮修改文件执行 repo 标准 formatter / linter / static check，并记录真实结果。

6. 执行并记录：
   - `git diff --check`
   - `git status --short --branch`
   - `git diff --stat`

在这些步骤完成前，本原型应被准确描述为：

> 已完成 M0/M1 范围内代码实现，但测试验证仍在进行中。

## 运行最小示例

```bash
PYTHONPATH=/path/to/verl python3 examples/tutorial/agent_loop_get_started/search_agent_dry_run.py
```

该示例不依赖内部网络、GPU 或真实搜索服务。

## 当前限制

- 未接入真实搜索服务
- 未实现 PPO/GRPO 搜索训练闭环
- 未实现 Partial Rollout KV Resume
- 未实现 Retrieval KV Runtime
- 未改动 SGLang 内核或 rollout 核心逻辑

## 后续 M2～M6

- M2：snapshot record + richer reward infrastructure
- M3：小规模 Search RL closed loop
- M4：搜索等待期间的 KV 生命周期管理
- M5：Retrieval-KV identity / reuse 联合实验
- M6：高保真或真实远端搜索后端验证
