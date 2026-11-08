# 模拟器源码修改、版本与协作

需要修改模拟器内部代码时，本页说明改动放在哪里、如何固定版本，以及怎样随主仓发布。KINETIX/Jax2D 已按 MIT 许可维护仓内副本，DOM 执行源码也在本仓；Isaac SDK 和资产独立安装。来源见各模块 PROVENANCE 及 [第三方记录](../../THIRD_PARTY_NOTICES.md)。下文 fork/submodule 目录是扩展设计，现有环境见 [后端指南](simulator-backends.md)。

先检查公开 API 或 adapter 能否满足需求。确实需要持续修改内部源码且许可允许时，再维护 fork 并用 submodule 固定提交。SDK、引擎和资产按上游方式安装；调度、实验与展示代码留在主仓。

## 先判断改哪一层

| 改动 | 默认位置 | 何时需要改第三方源码 |
| --- | --- | --- |
| 同步/异步推理、请求触发、动作队列、延迟采样 | `src/robotics_bench/protocols/` | 现有 API 无法在等待中正确推进世界时，增加最小环境钩子 |
| 输入输出封装、状态快照、单位、trace | `src/robotics_bench/simulators/libero/` 等 adapter | 缺少必要状态/事件接口时，在对应依赖中增加接口 |
| 控制频率、相机、horizon 等已有参数 | 主仓 simulator config，通过 API 传入 | 上游没有所需配置能力时 |
| 新场景、对象、任务定义或物体运动 | 主仓任务/资产扩展目录，使用上游注册机制 | 注册机制不能表达时才修改 simulator；标明新任务版本 |
| physics 子步、controller 更新、观测刷新、底层终止逻辑 | 真正拥有这段实现的依赖 fork | 可能属于 LIBERO、robosuite 或更低层，不能预设都在 LIBERO |
| 可视化、录像、时间轴、结果聚合 | 主仓 viewer/trace/experiments | 通常只需公开采帧或状态接口，不把渲染面板写进 simulator |

LIBERO 的 wrapper 传入 control_freq，并将 step 委托内部环境；其 BDDL 环境调用父类 step 后又处理任务成功状态。控制/物理子步的部分实现位于 robosuite。直接绕过 environment 调 `sim.step()`，可能遗漏 controller、观测刷新、reward 或 terminal 更新，须逐项核对。[LIBERO wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py#L27-L88)、[BDDL step](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/bddl_base_domain.py#L800-L819)、[robosuite step](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/environments/base.py#L171-L398)

例如“推理期间目标继续运动”：runner 决定等待多久及期间采用什么控制；adapter 将持续时间映射到 simulator 支持的步进接口；模拟器负责动力学积分、观测和终止一致性。只有现有接口不能做到时，才在 fork 中新增清晰的步进能力。模型名称、GPU profiler 和 latency profile 不进入该底层接口。

## 引入方式

| 方式 | 适用情况 | 维护代价 |
| --- | --- | --- |
| 原版固定版本 + adapter/扩展 | 公开 API 已足够 | 首选；不承担下游源码维护 |
| **独立 fork + 固定 SHA submodule** | 持续修改内部接口、多人协作、计划向上游贡献 | 两仓关联 PR；由维护者负责版本联动 |
| 固定 upstream + 补丁目录 | 少量、稳定且容易重放的修改 | 检查应用顺序与基础 SHA；补丁频繁冲突时转 fork |
| 直接 vendor/subtree | 明确要求单仓离线分发，且团队愿意维护来源同步 | 主仓有大量第三方 diff；需记录上游基线和更新过程 |

fork 与主仓保留独立历史，方便核查差异和回收上游修复。submodule 在主仓保存指定提交，普通 update 按这个提交检出；环境安装/实验运行不使用 `--remote` 自动追踪最新分支。[Git submodule 文档](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

正式实验语义使用可追踪的源码提交或明确补丁链；禁止叠加未记录的 submodule 修改、site-packages 替换或全局 monkey patch。

## 建议目录与版本来源

```text
robotics/
├── third_party/
│   ├── libero/                       # 需要源码修改时加入的 submodule
│   └── simulators.lock.json           # 上游基线、依赖及环境元数据
├── src/robotics_bench/
│   ├── simulators/libero/             # 我们维护的 adapter/扩展
│   └── protocols/                     # 推理与动作调度
├── configs/simulators/                # 原版/扩展协议分别命名
├── tests/integration/simulators/      # 接口、步进、reset、终止验收
└── docs/third_party/libero.md          # 改动目录、原因、上游/下游 PR
```

fork 提交以 gitlink 为准；lock 记录 upstream URL/base SHA、依赖版本、环境约束和资产 hash。运行产物保存 gitlink/dirty 状态、import 来源及依赖版本；发布内容必须来自已提交版本。

通用 v1 的 simulator provenance 扩展按 [契约演进](artifacts.md)增加字段与校验。

开源 `.gitmodules` 使用所有读者可访问的公开 HTTPS 地址，开发者可在本机另设推送凭据。普通 CPU 契约任务仍不需要初始化模拟器；只有 simulator 任务运行相应依赖初始化/安装。大型数据和权重不作为 submodule 内容引入。

## 一次模拟器修改如何提交

1. 在主项目任务卡说明触发场景、为什么公开 API 不够、拟改层及验收；关联模拟器 fork 的任务/PR。
2. 在 fork 建 feature 分支，提交最小环境修改和对应测试。主仓可同时准备 Draft PR，提前 review adapter 与配置变更。
3. 模拟器 PR 在团队 fork 审查并合并后，确定新的可获取 SHA；主仓 PR 再更新 gitlink、adapter、环境锁定信息和集成测试。
4. 主仓 reviewer 必须查看模拟器 diff 或关联 PR，不能只批准一个 opaque SHA。两边验收通过后合并主仓。
5. 通用改进可另向原上游提 PR；主仓无需等待上游接受，但应保留后续去掉下游修改的路径。

主仓引用的提交必须已推送到公开 fork，并由长期保留的分支或 tag 指向。修改 submodule 前先建分支，提交完成后再更新主仓引用。

主仓 CODEOWNERS 覆盖 adapter/gitlink，fork 规则覆盖模拟器源码。维护者负责跨仓集成与发布，贡献者按模块提交。

## 验收与实验标识

为修改前的受支持路径保留兼容测试：固定 seed/state/action 序列，比较 step 数、观测、终止及数值容差。新 hook 默认关闭时应保持原行为；新的动态协议单独测试时间推进、控制保持、末次动作/终止及 reset。

观测钩子或可选参数关闭时保持基线；对象运动、成功判据、控制频率或物理参数变化时，使用新 task/protocol ID（如 `libero_latency_v1`），并记录原任务来源。

模拟器测试使用对应渲染/物理环境，可由轻量 policy 驱动。CPU 契约 CI 独立运行；gitlink/adapter 变更的集成证据由指定 job/设备提供，未执行的组合标为未验证。

## 发布时一起交付什么

发布固定主仓/fork commit、上游基线、依赖/资产与测试配置，并提供 submodule 克隆说明。主仓 ZIP 与含 submodule 的完整离线源码包分别构建；离线包检查全部来源与许可。

保留实际引入文件的版权、许可证和修改记录；数据/资产另核对各自条款。发布说明提供模拟器改动摘要与关联 PR，使用户能选择原版基线或扩展协议。相关来源统一登记 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
