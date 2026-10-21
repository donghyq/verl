# 来源、许可证和贡献说明

- VeRL 源码及派生代码遵循 Apache-2.0；各文件中已有的公司、个人和上游作者署名保留。完整运行快照包含大量未修改的上游代码，不代表本项目拥有其原创归属。
- SGLang 源码及派生代码遵循其上游 Apache-2.0 许可。`runtime/sglang` 基于 `71de97b264b04dcd514cf904003028aefe9775c8` 的生产扩展；`upstream/sglang` 是现有 Draft PR 的公开版本。两者没有被混称为同一个测试版本。
- VeRL 权重更新功能分支基于 [#8273](https://github.com/verl-project/verl/pull/8273) 的 `432eb05fcc2bc4a0ce1bff13a3a5c3b35d1cb77b`。共享公共接口和前序提交保持原作者历史；本项目新增 SGLang 后端实现，不把前序 vLLM 工作据为己有。
- generic hybrid cache 存取不是本项目新发明。SGLang Draft 的差异范围是活跃请求、已提交的 native 解码边界、token/logprob，以及新进程继续生成；仍需与相关上游工作确认重叠。
- 使用了 OpenAI Codex 和 mccodex/GLM 的 AI 辅助。仅对实际有最终产物的任务记为交付；超时或额度拒绝不算实现成功。本次目录、公开材料和少量发布检查修正也有 AI 辅助。
- 这是人类审阅用的 fork 分支，不代表已经完成人工逐行审查、原生 GPU 验证或上游提交。不得隐瞒 AI 辅助、伪造人工贡献、替换原作者，或把实验结果改写成生产收益。
