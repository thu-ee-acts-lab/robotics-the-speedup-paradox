# 工作路线

想参与开发，可以从下表选一个方向，再用 [任务模板](../templates/task.md) 在 Issue 中说明负责人、准备修改的模块和完成标准。

模型实验从 [静态](../../benchmarks/static/README.md) 或 [动态](../../benchmarks/dynamic/README.md) 页面选择 case；多个 case 共用的工具标记为 `shared`。

| 归属 | 方向 | 前置条件与主要交付 |
| --- | --- | --- |
| 静态 | π0.5＋LIBERO | 已有object全量同步/论文异步；下一步为完整推理测量与经过质量验证的轻量化 |
| 静态 | Cosmos-Policy＋LIBERO/RoboCasa | 已有独立单环境入口与GPU smoke；扩大任务覆盖，补齐完整环境安装与推理测量 |
| 静态 | LingBot-VA＋RoboTwin | 已有sync/paper_async单任务GPU smoke及KV/VAE历史可见性协议；继续扩展任务覆盖 |
| 动态 | DynamicVLA＋DOM | 单场景时钟、历史观测、延迟与动作生效、终止/reset |
| 动态 | [Kinetix原生策略与rollout](../../benchmarks/dynamic/kinetix/README.md) | 仓内flow/env/physics及fine/coarse协议；扩展配对评估，时间片研究单独开展 |
| shared | 数据契约与CPU事件语义 | 按真实需求扩展契约；可手算的时间线与错误边界 |
| shared | Trace viewer与分析工具 | 动作来源、时间映射、按case比较指标 |
| 对应case / shared | 端侧优化 | 固定模型任务和设备profile；数值与完整入口证据，共享部分再抽取 |

协议和 CPU 工具可以独立开发。涉及性能或模型质量时，在对应设备和任务上测试；接入新模拟器时先跑通控制循环和基线。修改公共代码时，列出需要检查的 case。
