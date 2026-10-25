# 当前结果与未完成项

目标是让多轮 Agent 强化学习已经完成的轨迹、工具结果、完整样本组和 KV 状态可靠保存，在资源交接后继续生成并交给训练端。当前统一功能链已有真实训练证据，完整训练恢复和公平净收益仍在推进。

| 工作 | 已验证结果 | 剩余任务 |
| --- | --- | --- |
| SGLang 同版本跨进程恢复 | 原固定版本在同两张 A100、TP2 上将缓存池容量从 219 改为 281；源进程退出后新进程导入新槽。128 个 token 和 logprob 与不中断 greedy 参考逐项相等；两 TP 各 53 个历史位置重算为 0，完成 118 次原生 decode | [现有上游 Draft #45572](https://github.com/sgl-project/sglang/pull/45572) 保留；公开基线迁移后的 GPU 复验与旧结果分开，不扩称跨主机或跨权重恢复 |
| VeRL SGLang 跨权重更新 | 当前提交 `82759fc32` 已通过真实 4×A100、TP2 加训练发送端的原生集成验证：一次 SGD 后两层 router 参数内容摘要改变；保留 10 个输出 token 及 logprob 后续写 502 个，双 TP 历史输入重算均为 0；token/logprob/routing 原对象复核一致。原生 CLI 退出码 0，活跃自有进程与外层阶段清理通过，见[当前验收记录](validation/cross-weight-sglang/README.md) | [fork 审阅 Draft #1](https://github.com/donghyq/verl/pull/1) 基于 #8273 公共接口，共 20 文件；当前 head `82759fc32`。上游接口协调、人类逐行审阅及亲跑测试待完成；本次是小型 MoE 原地更新，不是 35B 检查点恢复 |
| VeRL 复现与环境 | 当前提交在真实 Linux 环境执行 4 个公开 CPU 模块：57 个测试、31 个子测试通过，无跳过、错误或 CUDA 初始化；同一 GPU 任务再次通过这些检查。真实 GPU CLI 与 10 个外层阶段清理均通过；引擎无活跃自有子进程，另保留 4 个 zombie 的观测记录 | 使用受控旧 URL 锁，依赖审计含 3 项预先声明的覆盖；仓库原默认锁定环境与完整 CI 仍待验证，不把本次结果写成默认环境兼容 |
| 原 35B、8×H20 接管到训练 | 一次实际运行成功结束。12 份原始产物逐文件核对 SHA 后，独立审计确认：活跃请求双 TP 物理导入、原 8 个成员进入实际 64 条训练批次、4 张源 GPU 实际用于 Trainer、4 rank 梯度和 Adam 推进、权重 0→1 后原生生成 4 个新 token | 原运行报告受两个观察器错误影响，仍保留 `NOT_PASS_MISSING_EVIDENCE`；修正后的离线独立审计通过，见[详细证据](validation/r22/README.md)。这不是长期稳定性或生产收益结论 |
| 完整训练 checkpoint | 已在真实 35B、8×H20 运行中保存四个 rank 的模型分片；Adam 保存未完成，未形成完整 checkpoint3。后续小范围 I/O 检查显示目标路径约 5 MiB/s，16 MiB 缓冲未改善，见[负结果](validation/checkpoint-io/README.md) | 完整 model/Adam/scheduler/RNG/cursor 保存、原进程退出、新进程加载并更新到 4 仍未通过；继续解决保存链路，不把旧 trajectory-only step3 或统一功能链 step1 作为替代 |
| 恢复净成本、pooling/offload | 保留现有统计、准入与后端复用代码；物理恢复减少历史重算的机制证据成立 | 工作量等价的 PartialRollout A/B 尚未完成；不写收益百分比。pooling/offload 单独整理既有能力，暂停新增路由性能与存储系统扩展 |

## 按版本审阅代码

- [当前 VeRL 功能 Draft](https://github.com/donghyq/verl/pull/1)：head `82759fc32de300eeefd145f6ef531c7702c4d21b`，base 为 #8273 的 `432eb05fcc2bc4a0ce1bff13a3a5c3b35d1cb77b`。这是 fork 内的审阅 PR，尚未向上游 VeRL 提交。
- `upstream/verl/` 是早期公开源码副本；当前功能以 Draft head 为准。`upstream/sglang/` 对应其独立来源，不与训练归档混用。
- `runtime/` 是归档训练功能基线；`validation/r22/` 增加本次独立审计的两个修正函数及证据说明，没有覆盖旧运行报告或更改已运行包。
- `candidates/checkpoint/`、`candidates/recovery_net_cost/` 及 `components/legacy_storage_data_plane/` 各有来源；不能直接互相覆盖运行。

实现、测试与整理使用了 AI 辅助。提交者逐行理解并审阅变更、亲跑相关测试仍是向上游提出前的工作；自动测试和 mccodex 结果分别记录。
