# 静态任务实验入口

[环境配置](../../docs/environment_setup.md) · [分步实验](../../docs/hands-on.md#static)

这个目录提供静态任务的运行入口，用来比较同步执行、历史观测异步和不同模型精度。每个 case 使用自己的 checkpoint、任务和模拟器；从下表选择组合，再按对应 README 配置和运行。

| case入口 | 绑定范围 | 运行配置 |
| --- | --- | --- |
| [pi05_libero](pi05_libero/README.md) | 原生 LeRobot π0.5＋LIBERO任务 | 已完成object同步及论文异步各500回合；使用任务匹配权重及同源processor，不使用VLASH微调权重 |
| [cosmos_libero](cosmos_libero/README.md) | Cosmos-Policy＋LIBERO spatial/object/goal/10 | H=16、n默认16；sync/paper_async；BF16与显式DiT整数优化 |
| [cosmos_robocasa](cosmos_robocasa/README.md) | RoboCasa专用Cosmos checkpoint＋指定fork和控制器 | H=32、n默认16；sync/paper_async；三相机与配对初态核对 |
| [lingbot_robotwin](lingbot_robotwin/README.md) | LingBot-VA＋RoboTwin aloha-agilex双臂任务 | sync/paper_async；完整KV/VAE观测历史延迟；独立模型worker与仿真环境 |

如果要研究 LingBot 的 KV 剪枝，先运行 [CPU 参考选择器](lingbot_robotwin/KV_READ_WINDOW.md)，了解它保留哪些 token。把选择器接入模型 worker 并测量 GPU 性能，需要完成该页列出的后续步骤。

## 文件归属

- `benchmarks/static/<case>/`：CLI、预检与运行生命周期；π0.5保留外部evaluator桥接。
- `src/robotics_bench/engines/`、`models/`、`simulators/`：引擎、仓内模型执行与环境适配；权重从外部路径加载。
- `src/robotics_bench/protocols/static_runner.py`：Cosmos静态调度；LingBot由专用runner维护有状态缓存。产物写入 `runs/static/<case>/`。

π0.5 与 Cosmos 的 `paper_async` 会让模型看到较早的观测：在控制步 t 使用 t−n′ 的图像和 state，历史不足时使用当前观测。图像和状态始终来自同一快照，宿主按顺序执行推理和控制。n 是每次执行的动作数，`0 ≤ n′ ≤ n`；π0.5 默认 n=5，Cosmos 默认 n=16。前者使用外部 `eval.async_delay`，后者由仓内 runner 选择观测。

## 接入与验收

添加新模型或优化时，先用固定输入比较动作输出，再跑一个同步回合，最后比较量化或异步配置。各组使用相同任务、初态、checkpoint、processor、设备、控制周期和预算。运行后检查实际后端、观测来源、成功率与请求数；详细要求见 [组合实验](../../docs/protocols/composable-experiments.md) 和 [模型接入](../../docs/protocols/model-adapters.md)。

计算 [论文周期](../../docs/protocols/speedup-metrics.md) 时，Tinf 使用同配置的推理测量，Tact 使用声明的动作时间：π0.5 默认 `1000/30 ms`，Cosmos 默认 `50 ms`。这组参数用于估算，不改变模拟器步长。没有 Tinf 时，周期和加速比留空；实际评估墙钟时间记录在 `eval_s`。需要检查动作平滑度时，使用 [轨迹工具](../../tools/embodied/README.md) 分析位置记录和 jerk。

提交开发任务时，用 `static/<case>` 标记对应入口，共享改动用 `shared`。[动态实验](../dynamic/README.md) 有自己的运行方式和对照基线。