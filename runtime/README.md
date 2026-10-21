# 原生训练的功能源码基线

`verl/` 保留 846 文件的源码归档上下文；它是训练管线、轨迹持久化、完整样本组交付、恢复和版本隔离代码的实际功能基线，不是仅提取类名的示意代码。来源与每文件 SHA 见根清单。

`native/` 放原生训练入口、物理接管检查、训练更新观测、GPU 复用验收及工具调用适配。只保留源代码与公开依赖 pin；内部服务实现、原始提示词、工具目录、奖励配置及模型数据省略。机器路径和服务地址经过替换，原部署散列门不会对这些 review 副本直接通过。

`sglang/` 是固定旧版的恢复扩展源码，不是整套 SGLang 安装包。它和 `upstream/sglang/` 的新公开 Draft 版本分别保留。

阅读顺序：`verl/verl/trainer/ppo/v1/trajectory_state/storage.py` → `coordinator.py` → `token_kv_source.py` / `token_kv_resume.py` → `agent_loop_tq.py` → `native/native-training/continuous-training/native_joint_handoff_driver.py`。检查更新和资源交还时，再读 `joint_gpu_return_observer.py` 与 `update-audit/`。

当前 V1 轨迹存储接口中的具体后端是文件对象存储。Mooncake 产物载荷接口位于历史组件目录；`verl/checkpoint_engine` 的 Mooncake 用于权重同步，两者不要混淆。
