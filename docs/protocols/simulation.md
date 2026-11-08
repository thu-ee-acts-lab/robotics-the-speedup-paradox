# 仿真时间、异步推理与实验记录协议

版本：设计规范 v0.1。实现或修改推理调度时，用本页明确环境什么时候推进、动作什么时候可以执行，以及日志中的时间表示什么。通用时钟和队列部分是扩展设计；运行现有 case 请看 [操作指南](../hands-on.md)，记录格式见 [数据契约](artifacts.md)。

实验分别记录 **物理环境时间、程序墙钟、动作等待和观测年龄**。固定 policy、seed、delay profile 和协议后，这些记录应能解释并回放完整执行过程。

## 两条独立的配置轴

调度协议与延迟计量分别配置，避免一个 `async=True` 同时改变几个实验因素。

| schedule | 定义 | 用途 |
| --- | --- | --- |
| `sync` | 采样、推理就绪、执行配置数量的动作，再发下一请求 | 同步基线 |
| `paper_async`（case manifest） | 按论文回取 `t−n′` 的历史观测，并单列解析周期模型 | 静态论文异步实验；不要求host推理与控制并发 |
| `history_observation` | 同步host控制流，提供过去n′个动作对应的观测 | `paper_async`的实现方式，也可单独作为历史观测消融 |
| `async` | 当前动作执行中发起后续请求；结果就绪后按队列规则接管 | 验证真实 overlap 和状态一致性 |

`paper_async`使用独立 case manifest；通用 v1 的 schedule 枚举保持原定义。π0.5＋LIBERO 将 `overlap_actions=n′` 映射为外部 `eval.async_delay`，范围 `0≤n′≤n_action_steps`，默认 n=5。论文定义历史观测偏移；本 case 另行定义 `same_snapshot`（图像/state 同源）与 `current_until_available`（历史不足取当前观测）。[论文§3.1 与 B.1](https://arxiv.org/html/2606.28529v2)

[LingBot＋RoboTwin](../../benchmarks/static/lingbot_robotwin/README.md) 的有状态扩展同时延迟 KV/VAE 观测流：名义关键帧 k 取 max(0,k−n′)，负索引重复初始快照，保持观测时间单调；推理边界 t 的最新缓存观测为 t−n′。原生首块执行 16 条、后续 32 条，n′限制在 0..16。预测动作缓存条件沿用原始动作块，与测量 state 明确区分。该 case 仍串行执行，指令的物理时长可变，当前不据此生成固定 Tact 的论文加速比；具体约定随 manifest 和 `paper-async.json` 记录。

论文周期为 `C=Tinf+n*Tact−min(Tinf,n′*Tact)`，记录为 `paper_model`。π0.5 默认 `Tact=1000/30 ms`，独立于 control/physics dt；`Tinf`需声明来源，缺失时周期/加速比为空，质量实验仍可运行。`eval_s`单列宿主耗时。

| clock / delay mode | 推理时环境如何推进 | 延迟来源 |
| --- | --- | --- |
| `virtual` + `zero` | 推理视为无仿真时间成本 | 质量消融；实际计算仍消耗 host 时间 |
| `virtual` + `profile_replay` | 按固定 dt 推进到请求的模拟就绪时刻 | 同模型/形状/配置/设备的实测 profile 或显式合成分布 |
| `realtime` + `measured` | 环境按墙钟 deadline 继续推进，推理 worker 独立运行 | 本次实际 pipeline 延迟 |
| `realtime` + `additive` | 在实际就绪后增加额外等待，世界继续推进 | 实际延迟 + 注入延迟 |
| `realtime` + `target_total` | 仅补足目标总延迟的剩余部分 | `max(0, target - measured_so_far)`；超过目标必须记录 overshoot |

在 `virtual` 模式下，sim time 按目标延迟推进，真实 CPU/GPU 耗时单独记录。如果宿主算得更慢，就在首次需要结果时等待，增加的是 host elapsed。`zero`表示零仿真延迟，报告中保留这一标签。`paper_model`记录解析周期，n′表示观测偏移；实际并发区间仍从事件测量。

## 时钟及事件字段

最小 envelope 为 `schema_version, run_id, episode_id, event_seq, event_type, clock_domain, timestamp_ns`。进程另记 `process_id, monotonic_origin`；跨机器时间差需时钟对齐方法及误差，否则仅计算各域内部 duration。

关键关联字段：`observation_id, request_id, chunk_id, action_index, sim_tick`。配置和 artifact 用独立 manifest 存储并通过 hash 关联，事件中无需重复塞入完整配置。

| 事件 | 时刻的定义 | 最少附加数据 |
| --- | --- | --- |
| `observation_sampled` | 输入实际采样完成 | observation ID、采样 sim tick、输入引用 |
| `inference_submitted` | runner 把 immutable snapshot 交给 policy | request ID、来源 observation ID |
| `inference_started` | 请求实际开始处理 | backend、worker/stream ID |
| `inference_completed` | 动作输出可被消费，含必要同步和后处理 | output/chunk ID、完成状态 |
| `result_released` | 实际或注入的等待已结束，允许入队 | delay mode、sample、profile ID、overshoot |
| `action_started` / `action_finished` | 每个 primitive action 真正开始/完成 | 来源 chunk、action index、来源 observation ID |
| `queue_underrun` / `result_dropped` | 无可执行动作 / 按协议弃结果 | 原因、队列长度、epoch |
| `episode_finished` | 首次 success、timeout 或 task failure | terminal reason、成功标记、budget 类型 |

kernel 计时用 `duration_ns` 与设备时钟域单列，host trace 使用自身 epoch。异步完成时刻以动作真正可读为准，设备 enqueue 另记。

## 三个常被混淆的量

```text
policy_pipeline = output_ready - request_started
observation_age(action) = action_started - observation_sampled
boundary_wait = max(0, next_action_started - previous_chunk_boundary)
release_wait = max(0, result_released - previous_chunk_boundary)
```

同一时间域内计算差值；每条动作保留来源 observation，以记录块内持续增长的 age。

手算例子：t=0 ms 采样新观测并开始推理，旧 chunk 还会执行到 t=120 ms，新结果在 t=80 ms 就绪，首个新动作在 t=120 ms 开始。此时 `boundary_wait=0`，`observation_age=120 ms`，`policy_pipeline=80 ms`。完全隐藏推理等待也不能消除观测陈旧。若新结果 t=180 ms 才就绪，则等待 60 ms、观测年龄 180 ms。该例应成为规范测试。

剩余等待代理量单独命名；observation age 由动作与观测时间戳计算。历史观测实验与真实并发分别报告。

## 环境推进与队列规则

静态质量基线可冻结推理期间的环境；动态延迟由环境 step 推进世界。task adapter 声明等待时采用末次指令、位置保持或 idle controller；零动作须按控制器语义验证后使用。

`world_progression=frozen_during_inference` / `advance_during_delay`作为独立配置扩展，加入 v1 前先更新契约。状态恢复、动作积分 replay 与 policy 闭环分别标识；策略任务成功率来自真实 policy 闭环。

新的虚拟协议用整数 tick 或时间累加器推进，使长期累计的控制时刻与目标时间一致；不要每次延迟独立四舍五入造成系统偏差。历史 Kinetix 复现可单独使用 `legacy_round` 映射以保留原论文 round(tau/33.33ms) 语义，禁止把新映射结果混作历史曲线。小于 physics dt 的延迟也必须保留余量。控制 dt、physics dt、实际积分 substeps 和剩余时间都写入配置。

以下规则用于实际并发 `async` 扩展：每个 policy 一个在途请求、队列有界；观察线程写 snapshot/mailbox，worker 隔离于可变 env。触发点由剩余动作或指定时钟决定，请求与重叠区间通过队列/完成事件记录。静态 `paper_async`沿用历史观测协议。

默认完成旧 chunk 再交接。`replace_pending`和按 valid-time 丢弃前缀作为独立协议，声明交接位置、空队列、late-result 及队列上限；重复或截断动作均需明确规则和记录。

episode reset 增加 epoch；旧 epoch 的 Future 即使完成也必须丢弃。episode 在推理/注入等待期间达到 terminal，应立即逻辑结束并拒绝动作；在途 CPU/GPU 请求隔离并等待完成或可靠取消后，才能释放其输入输出和 workspace，结果稍后返回不能继续执行。必须覆盖推理异常、worker 退出、取消/超时、输出非法值、队列耗尽与 late completion。

每次开始下一条动作前检查预算，确保 K 步上限不会执行到 K+1。terminal 记录和视频末帧应对应同一次终止决定，不能等渲染完成才结束回合。

实时 runner 按 deadline 推进环境，不在等待 `Future.result()` 时停住。用 trace 检查 GIL、CUDA 同步和渲染是否阻塞控制循环，必要时拆分进程。记录 deadline miss 和 real-time factor，把运行超期与注入延迟分开。

## 可复现实验 manifest

每个 run 保存：代码 SHA/dirty patch hash、配置 hash、模型 checkpoint hash、adapter/backend 版本、量化 layer manifest/calibration/pack hash、软件环境、GPU/CPU/驱动/功耗/频率/热状态、warmup 与测量次数、任务与 reset seed 清单、控制/物理 dt、chunk 长度、执行长度、协议版本、delay source 与采样 seed、队列/过期规则、预算及成功判据。

delay profile 附带 scope（仅 forward 或完整 perception→action pipeline）、硬件、输入形状、精度、denoise step、预热方法、原始样本。均值重放、IID 重采样、保序 trace 重放应作为不同实验；发生时间相关抖动时只保存均值无法重现尾部风险。远程网络延迟单列。设备 label 不足以唯一标识 profile。

实验分为零延迟质量、固定 policy 延迟响应、硬件 profile 联合比较三组，操作见 [预测器指南](../kinetix-latency-study.md)。跨方法同时保留原始配置与实测 inference speedup，按相近延迟或质量约束比较。

## 任务级指标及失败

每组报告 trials、successes、SR 及区间、成功条件 task time/chunk count、全部终止原因/耗时和 timeout budget。预算声明 sim/wall 域、起点及独立 host watchdog，按同域比较。零成功时成功条件时间为 null 并排除排序。

保留两套明确命名的失败处理：

1. `paper_legacy`：为需要兼容的历史实验，按它的动作/chunk 上限和配置相关 `Tmax=Nmax*Tchunk` 计算 penalty，并显示其 budget。
2. `fixed_budget`：新的跨配置主比较固定任务秒数预算 B，`penalized_time=(sum(success completion time)+failures*B)/trials`。同时报告真实终止耗时，避免将 penalty 误称为实测耗时。

失败包含 task failure 与 timeout，基础设施错误单独计数。成功 seed 集合不同的配置，同时报告共同成功 seed 的配对结果与全体 SR/penalty。区间以独立 episode 计算；多任务先分项、再按声明权重聚合。

静态 `N*Tcycle`为 paper_model 估计；真实任务时间从事件测量，包含首尾不完整 chunk、变长推理和 fill/drain。paper_model、simulated task time、wall elapsed 与 CUDA duration 分列。

TISED 的分量拟合、选点与验证使用分开的配置/seed 或 held-out 硬件。明确 O(n) 表述依赖分量可校准和可迁移假设，计入校准成本，并检查非分离的交互项。N 或质量曲线不强行单调拟合；模型不确定性要传递到候选 sweet spot。

## 最小验收场景

| 场景 | 应观察到的结果 |
| --- | --- |
| `paper_async`，n=5、n′=2 | 外部历史偏移为2；历史足够时取 `t−2` 同一快照，不要求后台并发 |
| `paper_async`，未提供Tinf | 保留历史观测实验设置，周期/加速比缺失，不从host耗时填造 |
| 同步：2 次推理各 80 ms、每次执行 2×50 ms | 虚拟串行总时间 360 ms，无漏计/重复计延迟 |
| t=0 采样、80 ms 就绪、120 ms 接管 | age=120 ms，boundary wait=0 |
| t=0 采样、180 ms 就绪、120 ms 旧块结束 | age=180 ms，wait=60 ms，欠载事件 |
| 100 ms 注入、世界目标速度 1 m/s | 无其他作用时目标推进 0.1 m；冻结基线为 0 |
| 连续多次非整数 dt 延迟 | 累积推进与总延迟吻合，误差有明确界限 |
| 虚拟 profile=100 ms、host 计算=25 ms | 模拟只计 100 ms；host 25 ms 单列 |
| 虚拟 profile=100 ms、host 计算=250 ms | host 等待但 sim 就绪仍为 100 ms；不宣称满足实时 deadline |
| 墙钟 target=100 ms、实际=120 ms | 不注入负延迟，overshoot=20 ms |
| reset 后旧请求完成 | 没有跨 episode 动作；有 dropped reason |
| 等待中 success/timeout | 恰好一个 terminal，后续动作不执行 |
| 0 个成功 episode | SR=0，successful time=null，penalty=预算 |
| 可视化消费者卡住 | 控制逻辑继续，显示丢帧可计数 |

前两项用于静态论文异步；其余用于通用实际时钟/队列扩展。字段/事件由契约校验器检查，调度行为可先用 CPU fake clock/policy/simulator 验收，再按 case 补充模型证据。

## KINETIX原生时间实现

[KINETIX case](../../benchmarks/dynamic/kinetix/README.md) 使用专用 `kinetix-native-delay-run-v1`。默认 fine（native-blend）保持物理 dt、frame_skip、约束参数及控制边界终止；分数延迟用 process 后的执行器指令时间平均近似，槽内状态仍按原生网格求解。旧队列从零动作开始，无预取；宿主耗时独立于虚拟 release。每条控制记录观测来源与物理槽权重。

coarse（legacy-round）保留控制周期取整对照，使用本 case 的 direct 启动规则。历史 RTC 预取协议与物理时间片细分按独立配置研究。
