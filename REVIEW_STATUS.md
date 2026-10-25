# 当前结果与未完成项

目标是让多轮 Agent 强化学习已经完成的轨迹、工具结果、完整样本组和 KV 状态可靠保存，在资源交接后继续生成并交给训练端。当前统一功能链已有真实训练证据，完整训练恢复和公平净收益仍在推进。

| 工作 | 已验证结果 | 剩余任务 |
| --- | --- | --- |
| SGLang 同版本跨进程恢复 | 原固定版本在同两张 A100、TP2 上将缓存池容量从 219 改为 281；源进程退出后新进程导入新槽。128 个 token 和 logprob 与不中断 greedy 参考逐项相等；两 TP 各 53 个历史位置重算为 0，完成 118 次原生 decode | [现有上游 Draft #45572](https://github.com/sgl-project/sglang/pull/45572) 保留；公开基线迁移后的 GPU 复验与旧结果分开，不扩称跨主机或跨权重恢复 |
| VeRL SGLang 跨权重更新 | 固定提交 `3fdcd5226` 的 Linux 4×A100 实验完成一次真实 SGD：原请求 10 个输出及 logprob 保留，新增 502 个输出；双 TP 各 501 个输出使用更新后的 router 参数，历史输入重算为 0。重叠调度结束后另完成一次无输出计算，仍计入计算量 | [fork 审阅 Draft #1](https://github.com/donghyq/verl/pull/1) 为 #8273 公共接口上的 SGLang 差异，共 19 文件；当前 head 为 `f3a8c3b4`。上游接口协调、人类逐行审阅及亲跑测试待完成 |
| VeRL 复现与环境 | 当时冻结 Linux 环境的 84 项 CPU 检查通过，无跳过、错误或 CUDA 初始化。功能 GPU 证据已写入后，原命令因即时子进程检查返回非零；后续清理修改的 4 个真实 CPU 进程用例通过 | GPU 环境含 3 项依赖审计例外；当前默认锁定环境和完整 CI 未验证，清理修改后未重复运行 GPU 命令 |
| 原 35B、8×H20 接管到训练 | 一次实际运行成功结束。12 份原始产物逐文件核对 SHA 后，独立审计确认：活跃请求双 TP 物理导入、原 8 个成员进入实际 64 条训练批次、4 张源 GPU 实际用于 Trainer、4 rank 梯度和 Adam 推进、权重 0→1 后原生生成 4 个新 token | 原运行报告受两个观察器错误影响，仍保留 `NOT_PASS_MISSING_EVIDENCE`；修正后的离线独立审计通过，见[详细证据](validation/r22/README.md)。这不是长期稳定性或生产收益结论 |
| 完整训练 checkpoint | 保存模型、Adam、scheduler、RNG 和数据游标的原生候选已绑定同一份已验证 35B 模型；唯一 8×H20 作业已接收，预算上限 8h / 64 GPUh | 实际保存完整 checkpoint3、原进程退出、新进程加载并更新到 4 尚待本次运行结果；旧 trajectory-only step3 和统一功能链的 step1 均不替代它 |
| 恢复净成本、pooling/offload | 保留现有统计、准入与后端复用代码；物理恢复减少历史重算的机制证据成立 | 工作量等价的 PartialRollout A/B 尚未完成；不写收益百分比。pooling/offload 单独整理既有能力，暂停新增路由性能与存储系统扩展 |

## 按版本审阅代码

- [当前 VeRL 功能 Draft](https://github.com/donghyq/verl/pull/1)：head `f3a8c3b4c607938ee7fff829d519643742909a01`，base 为 #8273 的 `432eb05fcc2bc4a0ce1bff13a3a5c3b35d1cb77b`。这是 fork 内的审阅 PR，尚未向上游 VeRL 提交。
- `upstream/verl/` 是早期公开源码副本；当前功能以 Draft head 为准。`upstream/sglang/` 对应其独立来源，不与训练归档混用。
- `runtime/` 是归档训练功能基线；`validation/r22/` 增加本次独立审计的两个修正函数及证据说明，没有覆盖旧运行报告或更改已运行包。
- `candidates/checkpoint/`、`candidates/recovery_net_cost/` 及 `components/legacy_storage_data_plane/` 各有来源；不能直接互相覆盖运行。

实现、测试与整理使用了 AI 辅助。提交者逐行理解并审阅变更、亲跑相关测试仍是向上游提出前的工作；自动测试和 mccodex 结果分别记录。
