# 动态任务实验入口

[环境配置](../../docs/environment_setup.md) · [分步实验](../../docs/hands-on.md#dynamic) · [§4.5预测器](../../docs/kinetix-latency-study.md)

先选择要研究的动态任务。下面两个入口各自管理模型、模拟器和延迟处理方式：

| Case | 当前状态 | 执行路径 |
| --- | --- | --- |
| [KINETIX](kinetix/README.md) | 已有仓内源码和可执行入口 | RTC JAX flow policy＋KINETIX/Jax2D；fine/coarse延迟、flow步数/延迟配对矩阵、完整视频与统计 |
| [DynamicVLA＋DOM](dynamicvla_dom/README.md) | 原生单环境入口 | 仓内模型/DOM源码、独立Isaac环境、原生non-streaming/streaming |

KINETIX 的模型、关卡和 Jax2D 物理代码都在仓内，适合独立改变采样步数和虚拟延迟。DOM 使用 DynamicVLA checkpoint 和 Isaac 场景，适合观察物体持续运动时的流式执行。先按对应 README 准备资源；时间定义和实现区别见 [仿真协议](../../docs/protocols/simulation.md)。

运行结果放在 `runs/dynamic/<case>/`。查看结果时分别读取虚拟延迟、宿主运行时间、任务和 seed 列表、成功率及步数预算。硬件加速比需要另做相同计时范围的测量。历史实现说明见 [参考整理](../../docs/kinetix-reference-review.md)，静态任务见 [static](../static/README.md)。