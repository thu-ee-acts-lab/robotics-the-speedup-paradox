# 动态任务实验入口

动态任务按case维护自己的模型、环境与时间协议，共享统计和产物约定。

| Case | 当前状态 | 执行路径 |
| --- | --- | --- |
| [KINETIX](kinetix/README.md) | 已有仓内源码和可执行入口 | RTC JAX flow policy＋KINETIX/Jax2D；fine/coarse延迟、flow步数/延迟配对矩阵、完整视频与统计 |
| [DynamicVLA＋DOM](dynamicvla_dom/README.md) | 原生单环境入口 | 仓内模型/DOM源码、独立Isaac环境、原生non-streaming/streaming |

KINETIX的实际模型、环境、物理核心和关卡代码位于 `src/robotics_bench/kinetix/`，没有外部源码运行依赖。框架环境和checkpoint路径通过case模板配置。DOM使用自己的DynamicVLA checkpoint、Isaac环境与场景资源，按独立case配置。

KINETIX默认保持原生物理网格，通过执行器指令混合近似分数时间槽内的动作切换；旧控制继续推进世界。它在虚拟时间轴上重放动作可用性；静态 `paper_async` 则按控制步回取历史观测。详见 [时间协议](../../docs/protocols/simulation.md)、[参考代码整理](../../docs/kinetix-reference-review.md) 和 [case命令](kinetix/README.md)。

运行产物放 `runs/dynamic/<case>/`。报告分别记录虚拟延迟、host调用时间、实际任务/种子覆盖、质量与预算。真实硬件加速比使用独立计时的同范围baseline。静态入口见 [static](../static/README.md)。
