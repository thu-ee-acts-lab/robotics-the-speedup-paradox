# 多模拟器后端与运行环境

选择或接入模拟器时，先确定它怎样接收动作、推进物理世界和结束回合。本页对比 LIBERO、RoboCasa、RoboTwin、KINETIX 与 DynamicVLA DOM 的接口和环境要求；安装与运行命令见 [操作指南](../hands-on.md)。

[静态](../../benchmarks/static/README.md)与 [动态](../../benchmarks/dynamic/README.md)case 分别固定模型权重、任务和环境版本，并使用各自的控制循环。下表列出各后端需要准备的依赖和检查项。

## 后端矩阵

| 后端 | 上游层次与修改归属 | 需要固定和验收的差异 |
| --- | --- | --- |
| LIBERO | task/wrapper 在 LIBERO，控制及物理步进部分位于 robosuite/MuJoCo | init state、控制器、成功/超时、camera/control/physics dt |
| RoboCasa | 厨房任务/场景/对象在 RoboCasa；控制与底层积分在 robosuite/MuJoCo | 原版或365版本、场景/资产、horizon、controller 和依赖组合 |
| RoboTwin | 采用选定上游版本的任务与环境 adapter | 场景/机器人资产、动作/坐标约定、reset 与该版本实际依赖；不因接口相似假定与上述栈通用 |
| Kinetix | 任务/观测/奖励在 Kinetix，物理引擎为 Jax2D，运行于 JAX | state/PRNG/static params、frame_skip、JIT、batch shape、auto-reset、渲染与device同步 |
| DOM | DynamicVLA 仓库中的仿真配置/评估代码，基于 Isaac Lab/Isaac Sim | USD场景/对象、physics/control/camera节拍、动作接收与保持、并行env索引及episode边界 |

RoboCasa 需固定原版/365、任务预算和依赖版本。[官方版本说明](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/README.md#L18-L48)、[依赖定义](https://github.com/robocasa/robocasa/blob/4f8a2980def75a55dff96b990745b83540425f09/setup.py#L17-L44)

Kinetix 显式传入随机 key、state 与参数；step 包含 JIT、多物理帧和 reset 逻辑。control step、physics step 与 host/GPU 同步分别记录。[Kinetix step](https://github.com/FLAIROx/Kinetix/blob/67f89f45e32cfd7899bc63e252a6b044eacafb83/kinetix/environment/env.py#L65-L163)、[官方使用示例](https://github.com/FLAIROx/Kinetix#-basic-usage)

DOM 官方推荐分别准备模型和 Isaac Lab 仿真环境，评估入口在 `simulations/evaluate.py`；其场景/对象资产单独下载。因此应将 DOM benchmark adapter、Isaac 运行环境和模型服务分别管理。[DynamicVLA 安装与评估](https://github.com/hzxie/DynamicVLA/blob/bb702465fe3976ac853bc1fbed600668045909f2/README.md)

## 公共接口统一语义，后端声明能力

每个 adapter 对外说明 reset、动作应用、观测采样、推进时间、terminal 和帧引用的含义，输出相同版本的事件。能力声明至少回答：

- 谁拥有步进时钟：runner 手动推进、后端自身循环，还是编译后的 rollout？
- 推理等待期间世界如何演化，机器人使用什么控制？
- 是否支持批量环境、状态保存/恢复、独立采帧和外部事件注入？
- 动作/观测是否在设备上异步就绪，谁负责同步与输入输出生命周期？
- 是否自动 reset，怎样保留 terminal observation 并产生新的 episode ID？

adapter 应说明已经实现并检查过的能力，对不支持的请求明确报错。MuJoCo、JAX PyTree 和 Isaac 保留各自的完整状态；公共层记录状态版本、文件引用和恢复条件。RGB 图像用于展示，状态恢复还需要后端的完整快照。

接口可以用本地函数或进程间消息实现。批量和编译执行仍需记录实际步数、终止时刻与观测采样时刻。

## 三类执行方式

**LIBERO/RoboCasa：** 通过 adapter 与支持的参数步进。task/scene 和 physics/control 改动分别提交；在固定软件/资产版本下比较同 seed/action 结果。

**Kinetix：** 保留函数式 state、PRNG 与 shape，将延迟作为 tick 或编译循环状态。事件可在设备数组中记录原始时间后批量导出；compile/warmup、设备执行、同步、像素观测与展示分别计时。关闭 auto-reset 或保存 reset 前 terminal state，保持 episode 归属。

模型侧保留原生 JAX/Flax、动作空间及 recurrent carry；policy、环境与 rollout 形成独立运行单元，见 [Kinetix 入口](../../benchmarks/dynamic/kinetix/README.md)。真实并发通过模型/环境重叠事件测量。

**DOM：** 非阻塞读取最新动作；无新动作时保持上次动作或初始姿态并继续步进。保留观测与 episode 来源检查；等待/丢弃策略变化使用独立协议。记录控制超期和实际推进时间。[DOM 评估循环](https://github.com/hzxie/DynamicVLA/blob/bb702465fe3976ac853bc1fbed600668045909f2/simulations/evaluate.py#L270-L371)

## 独立环境与目录

以下为扩展目录设计，不是当前文件清单；现有实现位置见 [架构指南](../architecture.md)：

```text
src/robotics_bench/simulators/
    libero/  robocasa/  robotwin/  kinetix/  dom/
configs/simulators/               # 每个后端的版本化配置
environments/
    libero/  robocasa/  kinetix/  dom/   # 各自依赖锁定/启动配方
third_party/
    libero/  robocasa/  kinetix/  dynamicvla/  # 仅必要的源码依赖
tests/integration/simulators/     # 按后端分开的验收
```

LIBERO 与 RoboCasa 分别锁定 robosuite/MuJoCo 版本，兼容验证后再共享环境。Kinetix 使用独立 JAX 栈；DOM 按 Isaac SDK 要求启动，模型服务另设 Python 环境。CPU 协议工具的 Python 3.11/3.12 独立安装。

源码修改与分发遵循 [fork/submodule 协议](simulator-dependencies.md)；引擎、SDK 插件和资产采用上游分发方式。DynamicVLA 使用含非商业条件的 S-Lab License 1.0，源码与资产分别记录条款。[上游许可证](https://github.com/hzxie/DynamicVLA/blob/bb702465fe3976ac853bc1fbed600668045909f2/LICENSE)

推理与仿真/渲染共享 GPU 时，分别记录 policy-only、simulator-only 与共部署计时，并保存实际资源竞争条件。

## 统一验收与后端专项验收

公共要求是明确单位/时间域、来源 ID、延迟期间动力学、终止与重置、失败预算以及媒体时间。后端专项测试分别覆盖：RoboCasa 资产/任务与 controller；Kinetix frame_skip/JIT/PRNG/auto-reset；DOM 动作保持、通信、camera 更新和独立运行进程。普通 CPU 协议 CI 不安装它们。

每个后端/协议/渲染组合分别标记 planned、implemented-unverified 或 verified。

trace v1 按串行 episodes 组织数据，Kinetix/DOM 的批量环境分别导出 run。需要联合时间线时，再扩展 env/agent ID 和时钟字段；后端能力、环境版本与媒体字段也按 [格式变更流程](artifacts.md)加入。

## DOM原生单环境实现

[DynamicVLA＋DOM 入口](../../benchmarks/dynamic/dynamicvla_dom/README.md)已迁入模型与 DOM 评估所需源码，
使用独立模型/Isaac 环境、原生动作保持、0.04s 物理步长和 Franka 单环境，
支持 native-non-streaming 及 native-streaming。模型生成 chunk、
实际应用 chunk 及保持旧动作的控制步数分别记录。物体掉落等原生失败与真正预算超时
按触发的 termination term 区分；异常和断连不记作策略失败。
