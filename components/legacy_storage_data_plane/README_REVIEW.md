# 历史存储与数据面组件

从既有提交导出代码和测试，保留路径以便沿导入关系审阅。这组代码与当前原生 V1 训练实现不是同一个运行版本，也没有声称能把该目录单独安装后跑完训练。

- `rollout_artifact.py`：Ray 元数据与不可变载荷分离；包括 Mooncake 载荷存取、发布前校验和独占处理租约。
- `trajectory_recovery.py` / `trajectory_snapshot.py`：保存与恢复轨迹，拒绝身份、布局和策略版本不一致的缓存。
- `transfer_admission.py` / `restore_cost_model.py`：限制在途数据，比较搬运和重算成本；短前缀可以选择重算，不把命中存储直接算成收益。
- `handoff_source.py` / `handoff_executor.py` / `preemptible_handoff.py`：执行权交接和恢复适配。
- `tests/experimental/fully_async_policy/`：与上述组件对应的历史测试。

历史 TCP SDK 和组件测试证据不等于当前 35B 原生训练已由 Mooncake 交付数据，也不是 RDMA、HA 或生产收益验收。
