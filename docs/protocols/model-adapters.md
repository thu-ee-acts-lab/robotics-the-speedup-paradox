# 模型与模拟器接入协议

接入新模型或模拟器时，本页帮助你划分模型、runner 和环境各自负责的工作，并确定需要的测试。运行已有 case 请看 [操作指南](../hands-on.md)。静态 `paper_async` 使用历史观测；真实并发有独立的调度约定。

每个 case 指定模型、checkpoint、processor、任务和模拟器。接入时同时检查权重、观测/动作含义、控制周期和所需协议；静态与动态 case 各自维护入口和 baseline，见 [架构指南](../architecture.md)。

## 所有 adapter 共享的要求

公共约定是输入来源、状态生命周期、动作语义和时间域。VLA 每次推理消费带 observation/episode ID 的不可变快照，输出带 request/observation 来源、单位、执行间隔和有效区间的 action chunk。Kinetix 等原生 RL 路径保留参数/carry/PRNG 与观测 PyTree，输出原生动作分布或所选动作，关联相同含义的来源事件；无需伪造文本输入、机械臂状态或固定长度 chunk。

模型可以同步调用，也可以放在编译后的 rollout 中。“动作就绪”表示调用方已经能读取输出，因此设备 enqueue 和完成要分别记录。批量执行可先在设备数组中记录事件，再导出实际时钟域和发生时刻。

adapter 提供 reset 行为，声明历史状态、cache、随机流和输出 buffer 生命周期。reset 的完成确认、跨任务 cache scope 和加载模式遵循 [后端状态协议](backend-state.md)。VLA 的 action horizon、实际执行动作数、denoise/refinement steps、cache chunk 数分别记录；原生 RL 按自身动作和状态定义字段，不编造不适用的去噪/chunk 参数。配置序列化为 resolved manifest；公开选项必须真正进入调用路径，新增元数据先遵循契约演进流程。

建议先对齐固定输入的动作输出，再跑通单回合闭环，最后比较优化配置。配置、shape、错误路径和 trace 格式可在 CPU 上检查；模型计时和任务质量在对应 GPU 环境中评估。

## 首版方向

| 方向 | policy adapter 应保留 | protocol / simulator 负责 | 第一份验收证据 |
| --- | --- | --- | --- |
| Cosmos-Policy 静态 | vision/DiT/action 输出、层名、候选量化站点、dtype/pack manifest | 静态执行周期、动作块、成功与失败统计 | 固定 observation/noise 下 reference action 对齐；实际 backend 映射 |
| LingBot-VA 静态 | visual/action KV 写入及读取窗口、跨轮状态、action 解码 | 环境重置、chunk 执行、时间线 | reset 不泄漏旧 KV；可读窗口与存储/写入量分开报告 |
| π0.5 静态 | vision/language prefix、flow action expert、预处理/normalization | 同步基线、`paper_async`历史快照与周期模型 | 固定输入/噪声动作对齐；n′映射和state/warmup规则明确；无内部 env.step |
| VLA动态仿真 | 模型侧只消费快照并返回动作 | 世界在等待中推进、延迟采样、旧动作/hold规则、terminal | CPU scripted policy验证运动和延迟；之后接DOM等匹配环境 |
| Kinetix原生RL | 任务匹配Flax网络、参数PyTree、动作分布、可选carry与PRNG | 原生JAX环境/rollout、延迟状态与按环境终止 | 固定关卡/参数动作对齐、真实policy动作被环境消费、reset与时间线，见 [专用入口](../../benchmarks/dynamic/kinetix/README.md) |

Cosmos-Policy 和 LingBot-VA 机器人适配器由本仓维护；新增来源按模块记录许可与移植范围。

## 量化与融合接口

同一模型/任务比较量化、异步及其组合时，使用 [组合实验协议](composable-experiments.md)：精度配置作用于 policy 构建，调度配置作用于 runner，保持共同实验案例与统一记录格式。

静态 `paper_async` 的 `overlap_actions=n′` 满足 `0≤n′≤n_action_steps`，π0.5 入口映射到外部 `eval.async_delay`。按论文回取 `t−n′` 观测；论文未规定 state 和历史不足时的处理，本 case 显式选择 `same_snapshot` 与 `current_until_available`。该标签属于独立 case manifest，通用 v1 schema 的 `async` 仍表示实际并发。

量化配置独立于 policy 与 kernel。逐层记录路径、weight/activation/accumulator dtype、scale/group/zero-point、calibration/pack 版本、backend 及 fallback。fake quant 用于质量分析，低比特时延由实际 backend 测量。

保留 reference、量化 GEMM、量化加外围优化三个运行点。分别报告算子误差、固定输入多步动作误差、闭环 SR/任务时间及失败代价；缺少的验收层级标为未验证。

backend 共用 capability、输入输出和计时约定；图、KV 格式与量化 kernel 按π0.5、Cosmos 和 LingBot 各自实现。

## 仿真接入接口

LIBERO、RoboCasa、RoboTwin、Kinetix、DOM 采用各自 adapter。公共接口定义实验语义；运行环境、批量/JIT 或独立进程方式按 [后端协议](simulator-backends.md) 保留，不要求统一逐步 Python 调用或底层状态格式。

模型与环境按 case 匹配。静态控制循环与动态时钟/历史/过期动作协议分别实现；DOM 和 Kinetix 各有 runner。任务类型与同步/异步调度分别配置。

`observe()` 返回真实采样时刻的快照；`step(action)` 只推进已声明的 control/physics tick；任务显式给出等待期间 robot control 和世界动力学规则。`render()` 是观测读操作，不能改变状态或阻塞控制。

delay profile 是参数化外部输入，至少标识硬件、模型/配置、计时 scope、原始样本与采样方式。`zero`、虚拟重放、真实计时、additive、target-total 分开；对 target-total 记录 overshoot，对虚拟重放避免叠加宿主计算耗时。完整定义见 [仿真协议](simulation.md)。

论文周期估计单列为 `paper_model`，以声明的 `Tinf`、`Tact` 和 n′计算。π0.5 默认 `Tact=1000/30 ms` 不改变环境步长；缺少 `Tinf` 时保留质量实验与观测偏移设置，不填造周期/加速比，也不用宿主 `eval_s` 代替。

## 可视化消费者接口

viewer 设计从保存的 trace 读取帧引用、inference、queue、action 和 terminal，支持按动作追踪来源观测；用合成样例验证交互与状态映射。

实时 viewer 扩展采用慢消费者丢显示帧并计数、关键事件独立持久化的策略。benchmark 单列 tracing/录像开销；截图、时间线和统计各自验收。
