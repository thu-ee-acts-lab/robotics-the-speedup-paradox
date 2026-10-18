# 静态任务实验入口

本目录组织静态任务的case、运行入口和实验配置。[π0.5＋LIBERO](pi05_libero/README.md) 已有原精度同步评估及论文静态抽象 `paper_async` 入口；Cosmos的 [LIBERO](cosmos_libero/README.md) 和 [RoboCasa](cosmos_robocasa/README.md) case通过独立engine/simulator及单环境runner执行。具体实验设置与验证范围分别见case说明。

| case入口 | 绑定范围 | 运行配置 |
| --- | --- | --- |
| [pi05_libero](pi05_libero/README.md) | 原生 LeRobot π0.5＋LIBERO任务 | 已完成object同步及论文异步各500回合；使用任务匹配权重及同源processor，不使用VLASH微调权重 |
| [cosmos_libero](cosmos_libero/README.md) | Cosmos-Policy＋LIBERO spatial/object/goal/10 | H=16、n默认16；sync/paper_async；BF16与显式DiT整数优化 |
| [cosmos_robocasa](cosmos_robocasa/README.md) | RoboCasa专用Cosmos checkpoint＋指定fork和控制器 | H=32、n默认16；sync/paper_async；三相机与配对初态核对 |
| [lingbot_robotwin](lingbot_robotwin/README.md) | LingBot-VA＋RoboTwin aloha-agilex双臂任务 | sync/paper_async；完整KV/VAE观测历史延迟；独立模型worker与仿真环境 |

每个case的指南包含运行命令与验证记录。比较配置时，固定任务/初态清单、checkpoint与processor hash、环境、硬件、控制周期、预算和baseline；LingBot的有状态缓存由专用协议维护。

LingBot论文剪枝的操作点、CPU选择规则和后续GPU验收边界见 [KV read-window设计](lingbot_robotwin/KV_READ_WINDOW.md)。当前提供默认关闭的CPU选择器契约；接入worker及GPU计时按该设计中的验收步骤进行。

## 文件归属

- 每个case的启动代码与配置放 `benchmarks/static/<case>/`；`pi05_libero/` 保留独立评测桥接，`cosmos_libero/` 使用 `config.py` 预检与 `run.py` 执行生命周期。
- Cosmos模型适配放 `src/robotics_bench/engines/`；原生LIBERO适配放 `src/robotics_bench/simulators/`；单环境静态控制协议位于 `src/robotics_bench/protocols/static_runner.py`。仓内模型执行源码位于 `src/robotics_bench/models/`，权重通过外部路径绑定。
- baseline、量化、同步/异步变体属于同一个case。配置引用固定版本的模型/环境与优化方案，运行产物放仓库已忽略的 `runs/static/<case>/`。

静态任务可以异步运行，也可以发生正常物理演化。任务分类、调度策略和等待期间世界推进分别声明；本路径不依赖DOM的时延补偿或过期动作规则。

π0.5和Cosmos的论文异步使用 `schedule=paper_async`，`overlap_actions=n′` 在0到实际执行长度n之间；π0.5默认n为5，Cosmos默认n为16。π0.5入口映射到外部 `eval.async_delay`，Cosmos由本仓runner回取 `t−n′` 观测；两者均使用同一历史快照的图像/state，历史不足时使用当前观测。这两条路径通过串行runner评估历史观测下的闭环质量。

## 接入与验收

先用固定输入与噪声对齐原生动作，再验证单任务同步闭环，之后比较同case支持的量化/异步变体。每个阶段保留独立的输入、配置与结果记录。

每组记录实际backend、观测来源、成功率及各时间域的任务耗时；有同范围推理时间时，按该case及硬件的baseline计算加速比。`paper_model` 的 `Tact` 按case声明：π0.5默认 `1000/30 ms`，Cosmos默认 `50 ms`，都不改变模拟器physics dt。未提供 `Tinf` 时不生成周期或加速比，宿主 `eval_s` 单列。轨迹/jerk分析复用 [现有工具](../../tools/embodied/README.md)。配置和验收遵循 [组合实验](../../docs/protocols/composable-experiments.md)、[模型接入](../../docs/protocols/model-adapters.md) 与 [加速比协议](../../docs/protocols/speedup-metrics.md)。

开发任务在Issue中标记 `static/<case>`；共享底层改动标记 `shared`。动态任务有 [独立入口](../dynamic/README.md)，两条路径不共用任务baseline。
