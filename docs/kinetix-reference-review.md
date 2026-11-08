# KINETIX参考代码与接入边界

修改 KINETIX 代码时，可以用本页查找参考实现和仓内模块的对应关系。模型与环境的实际执行代码在 `robotics_bench.kinetix`；下表列出的外部文件用于追踪方法来源。

## 各部分解决的问题

| 参考部分 | 机制 | 仓内对应 |
| --- | --- | --- |
| `rtc_kinetix.py`、`upstream_runners.py` | 关卡/参数绑定、延迟映射、flow步数和horizon扫描；将实验配置与上游实现分开 | 接入固定关卡、成对随机种子、独立质量/时间参数和可审计输出 |
| RTC `src/model.py` | Flax flow policy；MLP-Mixer生成8步motor/thruster控制，Euler采样步数改变网络调用次数 | 迁入 `flow_model.py`，当前直接使用原checkpoint，不训练或蒸馏 |
| `run_sensitivity_eval.py` | 控制步取整与细粒度物理子步延迟，区分噪声采样节奏并保留物理horizon | 仅保留原生步长下的取整对照；不把物理细分作为默认延迟实现 |
| `run_torch_task_accuracy_eval.py` | 将等价Torch策略置于JAX环境中，支持direct/旧RTC队列、原生eager或CUDA Graph | 整理接口与对照规则；首版实际执行保持JAX/Flax |
| `rtc_torch_policy.py`及低精度runtime | conditioning复用、融合MLP、CUDA Graph、W4/W2等独立执行路径 | 后端研究参考；当前runner使用JAX/Flax |
| `rtc_student*.py` | 结构化小模型与训练/蒸馏路径 | 当前不迁入训练；结构变化需要独立model配置与checkpoint验收 |
| native-delay协议说明与后期 `protocol.json` | 固定原生物理网格，按执行器指令进行分数槽混合；区分硬件profile与模拟时间 | 经用户确认，在本仓重建native-blend，保留近似边界说明 |
| 后期quality–latency拟合报告 | 分开拟合零延迟质量、固定质量下的延迟响应、硬件完整action-generation时延 | 质量、延迟与硬件profile分开采集；现有[预测器](kinetix-latency-study.md)独立拟合和验证 |

参考 native-blend 实现位于 `kinetix-native-delay-clock` 工作区的 `run_torch_task_accuracy_eval.py`。本仓按协议重建执行器混合，源码及变更由仓内来源记录追踪。

该工作区的 N=1～5 硬件数据见[2026-08-11档案](kinetix-hardware-latencies.md)：同架构随机权重、PyTorch eager FP16、独立进程重复及完整调用计时。当前 JAX checkpoint 的测量另行记录。

## 两类时间不能混淆

flow 采样变量的积分步长为 1/N；物理时间步为 dt，控制周期为 dt×frame_skip。减少 N 可以改变动作质量和完整推理工作量，但不应改变环境动力学。

早期细分协议用更小 dt 和更大 frame_skip 保留控制周期。例如从 `1/60×2` 改到 `1/2400×80`，名义控制周期仍为 1/30 秒，但每模拟秒的接触/关节求解次数提高 40 倍。碰撞 Baumgarte 参数的单项缩放不能自动保持 warm-start、位置修正、关节限制、摩擦与终止采样等语义。

walker/half-cheetah 的零延迟质量对上述物理细分敏感。比较轻量化时固定原生网格；对轨迹差异先检查数值与协议变化。

本仓的 fine 模式（native-blend）保持原来的物理步长：先用旧指令，在切换所在的时间槽混合新旧指令，之后使用新指令。两组动作先转换为执行器指令，保留原生裁剪和自动电机行为；碰撞仍在原物理网格上求解。

## latency–quality实验的拆分

建议保持相同 checkpoint、物理配置、噪声、执行 horizon 及 seed 列表，分别运行：

1. 固定延迟为 0，改变 flow 步数，得到本协议下的内在质量代理。环境仍按原生物理推进。
2. 固定 flow 步数，改变注入延迟，得到任务对延迟的响应。
3. 使用目标硬件完整 action-generation 测量，为每个 flow 步数映射延迟，观察组合结果。

参考分解为 `A(t,H,N)≈G_t(N)×H_t(D_H(N))`：零延迟质量、延迟响应和硬件时延分别校准，再验证组合预测。按[预测器指南](kinetix-latency-study.md)冻结模型、报告拟合误差及全部选定条件；后端不同的外部 profile 标记为延迟重放场景。

比较时固定调度与计时范围，分别标注 fake/dequantized 误差、真实低精度 GEMM、CUDA Graph 及 flow 步数设置；端到端收益使用完整调用和任务记录。

## 源码归属

实际模型、环境、物理核心、关卡与渲染素材迁入 `robotics_bench.kinetix`，内部使用私有包名。JAX、Flax、通用环境接口和渲染库仍是正常 Python 依赖。原始 checkpoint 是独立资源。

迁入的运行依赖闭包保留 MIT 版权与原文件 hash；省略重复定义和训练/云端保存函数。原生端点等价检查验证移植后的物理计算。

CLI 和产物见[KINETIX case](../benchmarks/dynamic/kinetix/README.md)，分步实验见[实操指南](hands-on.md#dynamic)。小时间片研究保持独立分支与独立协议。
