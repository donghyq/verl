# 连续参数发布的身份接线

本模块把原 Checkpoint 成功发布的参数版本与轨迹片段绑定。它不传输权重，
不改变 ReplayBuffer 的 prompt 出生步、固定 `drop/wait` 阈值或 PPO 算法。
当前独立工作树只验证模块与模拟回读；根入口和真实多轮训练尚未因此通过。

## 根入口只在原生命周期中增加连接点

```python
hooks = PolicyPublicationDriver(trajectory_coordinator_client)
ticket = await hooks.prepare(
    identity=canonical_identity,  # 原运行时 builder 的完整四字段身份
    model_configuration_fingerprint=actual_model_configuration_digest,
    publisher_id=existing_checkpoint_owner_id,
    engine_urls=owned_engine_urls,
)
committed = False
try:
    await gateway_control.stop_admission()       # 现有 remove/确认，不再包一层
    await hooks.mark_update_started(ticket)
    metrics = await self._update_weights(step)   # 原传输、ACK、finalize、版本发布
    record = await hooks.confirm_before_admission(ticket, observe_owned_engines)
    committed = True                            # 已原子持久化
    await gateway_control.start_admission(step)  # 原 register/ready；失败由原 owner 清理
except BaseException as error:
    if not committed:
        await hooks.fail(ticket, type(error).__name__)
    raise                                       # 不在 finally 中开放流量
```

`prepare()` 必须早于任何参数修改。`confirm_before_admission()` 必须晚于原
传输和所有实际 ACK，早于任何 Gateway 注册。根入口不得在外面调用完整
`update_weights()` 后再补身份记录，因为旧实现可能已开放流量。

`observe_owned_engines()` 是根入口必须补的只读连接点：使用原 owned server
handles 的实际 `tokenizer_manager.config_value("weight_version")`，或直读仅
owned URLs 的 `/model_info`，返回每个实际引擎的一条记录：

```python
[
    {"engine_url": "http://owned-a:30000", "weight_version": "2", "source": "model_info"},
    {"engine_url": "http://owned-b:30000", "weight_version": "2", "source": "model_info"},
]
```

这里的 `source` 表示原生实际模型版本，不是 startup `server_info`、配置期望值
或列表规格。已有 Gateway `get_active_weight_version()` 要求 live registry，
不能为了回读先 register。注册部分成功后的 owned cleanup 仍归原 lifecycle。

原 builder 的 `model_weights_fingerprint` 是模型配置和已提交版本的身份摘要，
不是训练后 tensor bytes hash。固定模型配置摘要必须来自实际配置；不能用
更新步数伪造该摘要，也不能改变既有四字段身份含义。

## 模块如何拒绝不确定版本

一个 `PublishedPolicyRecord` 保存 run、完整身份、固定模型配置摘要、实际引擎
回读和提交时间。内层身份、引擎观察和外层记录均为不可变 dataclass。

- 所有 configured 引擎必须出现且只出现一次，实际版本必须一致。
- 同一 run 的参数版本只能递增；已有版本不能重复执行权重更新或改写身份。
- 不允许在同一 run 内换模型配置、tokenizer/template 或 KV 表示。
- 更新/回读失败、返回结果不明及协调 Actor 重启后的未完成更新保持 BLOCKED。
- `get_policy_coordination_state()` 可核对实际未完成记录；它不重放任何写操作。
- 已成功记录的旧版本仍能验证历史片段，但不能冒充当前正在更新的版本。

## source 在有限交接窗口中固定身份

```python
runtime = PromptGroupHandoffRuntime(
    coordinator_client,
    owner_id=owner, incarnation_id=incarnation,
    policy_identity=initial_actual_identity,
    environment_version=environment_version, tool_version=tool_version,
    published_policy_client=coordinator_client,  # 必须同一协调器
)
```

`request_preemption(uid)` 停止尚未开始的成员，随后使用当前实际执行租约取得
一个 `HandoffPublicationBarrier`。参数发布与该窗口互斥。原 Tool adapter 从
`commit_snapshot.get_snapshot_policy_identity()` 取得窗口身份，继续提交完整
工具批次后的 Snapshot；历史片段仍使用各自真实版本。

Runtime 形成的 Manifest 和每份 Snapshot 仍要求源身份完全相等。只有
Manifest COMMITTED 且实际 source 权限已释放后，才释放窗口并返回 handoff ID。
截止时间只限制 Snapshot/Manifest 提交；到期不会自动放行并发参数写入。
失败清理也必须先撤销原实际租约。协调器重启不复用旧 monotonic deadline，
保留窗口为 BLOCKED；旧执行权失效后才可显式清理。

该约束不承诺 arbitrary mid-token checkpoint，也不改变原 generation fencing。

## target 新版参数必须显式按历史 token 恢复

```python
result = await runtime.run_target(handoff_id, resume_member, run_pending, token_only=True)
```

默认仍要求完整 source/target 身份相同。显式 `token_only=True` 时，Runtime
先验证 Manifest 的源身份对应原成功发布记录，再选择当前真实成功发布的目标
身份。目标 tokenizer/template 与原环境/工具仍必须匹配。

callback 的 `check_owner` 提供以下绑定，原 Tool adapter 实际使用它们：

- `resolve_published_policy(version)`：真实已提交记录。
- `token_only_resume=True`：由本次显式调用授权。
- `target_policy_identity`：本次目标的真实身份。

根 `V1TrajectoryEntry._resume_member()` 应把
`snapshot_policy_identity=snapshot.policy_identity` 传给 adapter，保留确切源身份，
不能仍传构造时的固定 runtime 身份。原 kwargs、prompt `global_steps` 与历史
tokens/mask/logprob/工具结果必须保持。pending callback 仍首次执行原输入。

adapter 不渲染旧 messages 获取历史 tokens，不读取/导入旧 `kv_ref`。新版生成
使用原 Gateway 的实际版本命名空间和原权重更新后的缓存清理。跨版本续跑不能
被解释为跨版本 KV 复用。

## 实际生成片段的校验

连续版本路径只接受真实 `native_gateway_attempts`：每个 attempt 自身必须是
单一原生版本、区间完整覆盖真实 token 与行为 logprob；多个 attempt 可以分别
来自不同已发布版本。未知版本和缺少原生回读都拒绝。已有
`trajectory_segments` 同时保存每个片段版本与完整身份，不给过去的 tokens
套上最后一个版本。

没有 publication client 时保留原固定窗口行为。该 opt-in 不改变社区
ReplayBuffer 的固定新旧程度规则，也不把版本指标变成新的采样接受算法。

## 验证边界

独立 CPU 测试验证提交/开放顺序、失败关闭、不可改写记录、Actor 进程模拟重开、
窗口超时不放行、n=8 原组保存、新版显式接管，以及实际 Tool adapter 的历史
tokens/logprob/工具不重放。回读是明确的 fixture，不构成集群版本发布证据。
不启动模型、GPU 或共享 Ray 服务。真实持续多轮训练仍须根入口接线后验证。

### 恢复与剩余责任

| 情形 | 模块行为 | driver 责任与级别 |
|---|---|---|
| 身份记录已提交，但返回 ACK 丢失 | 重开后保留该真实记录，拒绝重复参数传输 | 回读记录与当前 owned 引擎，再恢复原 admission；接线是 P0 |
| handoff 窗口已保存，协调器重开 | 旧 lease/monotonic deadline 失效，窗口保留 BLOCKED | 查询真实记录；所有原 source 权利已失效后显式清理，保留已提交历史；该校验已实现 |
| 原参数更新仍未决，协调器重开 | 保存操作 ID、原阶段与 BLOCKED，不接受原 ticket 的重放确认 | 必须取得原 Checkpoint 调用完成或被实际终止的证明；只看当前版本不能证明不存在旧 writer。这是跨参数发布中断后继续训练的 P0 接口缺口 |
| publisher 进程也丢失 | 不推断权重写入成功，不自动重传 | 广义 controller/publisher 恢复记 P1/FUTURE，本轮不实现新调度器、复制协议或 Checkpoint engine |

### 本轮实际 CPU 结果

CodeLab 617728841 的独立 `policy-publication-cpu/20261008-r2` 目录：

```text
CUDA_VISIBLE_DEVICES=''
python -m pytest -q -k 'not real_ray' \
  tests/trainer/ppo/v1/test_trajectory_state_on_cpu.py \
  tests/trainer/ppo/v1/test_trajectory_state_runtime_on_cpu.py \
  tests/trainer/ppo/v1/test_trajectory_state_tool_agent_on_cpu.py \
  tests/trainer/ppo/v1/test_trajectory_state_policy_publication_on_cpu.py
88 passed, 1 deselected, 8 warnings in 37.10s
```

首轮选择失误包含旧 `test_real_ray_actor_restart...`，结果为 87 PASS / 1 FAIL；
失败是旧 readiness 轮询读到 kill 前 Actor，不是新发布模块。该测试使用自己
的临时 CPU Ray instance 并在 finally shutdown，没有连接共享实例。r2 明确
排除该项，不启动任何 Ray 服务。两轮原始日志均保留，不能把首轮记为全绿。
