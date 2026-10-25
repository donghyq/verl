# Project B：异步 RL Rollout 存储与数据传输

这个分支把分散在不同工作区的**当前功能代码、历史存储组件、测试和诊断代码**集中到一个审阅入口。它是源码目录，不是把不同版本强行合并后的运行版本。实际部署归档、训练数据、模型、内部服务配置和日志不在这里。

## 先看这些代码

| 要审阅的内容 | 入口 | 当前边界 |
| --- | --- | --- |
| VeRL：SGLang 请求在权重更新期间保留与恢复 | [干净的功能分支](https://github.com/donghyq/verl/tree/codex/sglang-preserve-weight-update)、[仅本功能的差异](https://github.com/donghyq/verl/compare/432eb05fcc2bc4a0ce1bff13a3a5c3b35d1cb77b...82759fc32de300eeefd145f6ef531c7702c4d21b)、[本分支源码副本](upstream/verl/) | 基于 #8273 公共接口；当前提交的真实 SGD / TP2 原请求继续生成、历史零重算与 CLI 退出/清理通过，见[验收记录](validation/cross-weight-sglang/README.md)；[fork 审阅 Draft #1](https://github.com/donghyq/verl/pull/1) 已创建。当前默认锁定环境 CI 与人审待完成 |
| SGLang：活跃请求的同版本跨进程恢复 | [已有 Draft PR #45572](https://github.com/sgl-project/sglang/pull/45572)、[已推送版本的源码副本](upstream/sglang/) | 现有 PR 保留。历史固定版本的 GPU 恢复结果与当前公开 PR 版本分别记录，不能混用 |
| 原生异步训练使用的轨迹、缓存、完整样本组交付与版本隔离 | [轨迹状态模块](runtime/verl/verl/trainer/ppo/v1/trajectory_state/)、[训练队列连接](runtime/verl/verl/trainer/ppo/v1/agent_loop_tq.py)、[训练入口](runtime/verl/verl/trainer/ppo/v1/trainer_separate_async.py) | 保留完整源码快照以便追踪调用；原 35B 统一功能链已据原件独立审计，见[详细证据](validation/r22/README.md) |
| 可靠产物存储、Mooncake 载荷、处理租约及传输成本 | [历史存储与数据面组件](components/legacy_storage_data_plane/)、[独立租约恢复组件](components/lease_recovery/) | 与当前原生训练快照分开。不能把 Mooncake SDK/组件验证说成当前 35B 数据面已使用 Mooncake |
| 物理缓存接管、训练更新和源 GPU 复用的检查 | [原生训练与观测代码](runtime/native/native-training/)、[生产 SGLang 恢复扩展](runtime/sglang/) | 功能路径和观测路径并列；检查器存在不代表其验收条件已经通过 |
| 完整训练 Checkpoint 与恢复净成本 | [保存与加载候选](candidates/checkpoint/)、[成本统计候选](candidates/recovery_net_cost/) | 保存模型、优化器和训练进度的完整 3→4 恢复待验证；没有有效、工作量等价的收益结论 |
| 最近一次问题定位、路由探索 | [诊断代码](diagnostics/)、[隔离的 CPU 路由仿真](experiments/group_routing/) | 诊断不进入训练；仿真不算生产性能结果，也不扩入当前 P0 |

## 按请求流阅读

1. 轨迹和缓存说明先写入可校验的持久化产物；只发布成员齐全、身份与策略版本一致的样本组。
2. 恢复端核对请求身份、缓存布局和全部 TP 分片，导入真实缓存后继续生成。
3. 完整样本组经现有数据面和训练队列交给 Trainer；更新成功后才标记为已处理。
4. 用真实梯度、Adam 状态、参数变化、新版本生成，以及实际 GPU UUID 的使用记录分别验收。

源代码目录中包含上游原有代码作为依赖上下文，不能把整个快照计为本项目新增代码。优先从上表所列模块和功能分支差异开始看。

## 版本与审阅规则

- [REVIEW_STATUS.md](REVIEW_STATUS.md)：已验证结果、未完成项和版本边界。
- [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)：每个文件的来源、原始 SHA256、公开副本 SHA256及路径/服务地址替换记录。运行快照以归档 SHA 作为来源，不伪造一个无法证明的上游 Git 基线。
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)：上游代码、AI 辅助与人工审阅声明。
- 部分辅助脚本的机器路径和内部服务地址替换成了示例值；生产目录中的原文件没有改动。这些公开副本用于 review，不能直接通过原部署包的 SHA 门，也不承诺在公共环境一键运行。
- 真正准备向上游合入的 VeRL 改动位于独立功能分支；这个汇总分支不作为一个包含所有历史实验的上游 PR。

在本分支根目录执行 `python3 tools/validate_review_bundle.py`，可检查文件完整性、Python 语法和公开副本中禁止出现的内部地址。该命令不加载模型、不访问服务，也不替代功能测试。
