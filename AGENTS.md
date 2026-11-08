# AI 协作约定

模型、runner和适配器位于 `src/robotics_bench/`，算子位于 `src/robotics_kernels/`。仓内维护π0.5、Cosmos、DynamicVLA热路径及KINETIX模型/物理；框架、部分模拟器和模型资产按case配置。静态异步使用 `paper_async`。报告验证范围时区分契约、合成样例、单回合和任务集；Blackwell FP4/FP8源码需目标硬件验证。

Cosmos＋RoboCasa使用三相机、H=32及独立robosuite1.5.1环境，reset后读取本回合语言；LIBERO使用H=16。`--reference-run` 核对初态，训练数据与厨房资产分别准备。TurnOffMicrowave的sync/paper_async接入记录见case指南。

`tools/prepare_resources.py` 下载或复用资源：模型/数据默认 `~/.cache/robotics/hub`，LIBERO资产为 `~/.cache/libero/assets`。环境安装独立完成；评估离线运行，公共CPU CI不下载模型。

## 开始工作

遵循 [贡献与PR制度](CONTRIBUTING.md)：主题分支开发，PR进入main，禁止直接提交/推送main。依赖未合并功能时使用前置分支为base并注明PR。标题采用 `type: 中文摘要 / English summary`，正文保留模板的双语四节，范围与实际diff一致。审核后默认merge commit；普通开发不自动合并。用户已明确授权某PR合并且规则满足时继续执行，不重复询问。

1. 阅读 `README.md`、`CONTRIBUTING.md`、当前 Issue/PR 的任务范围及相关 `docs/protocols/` 文档。
2. 确认 `git status --short`、分支和基线 commit，保留其他人的修改。任务范围以 Issue/PR 和用户最新指令为准。
3. 先跑相关最小例子，再改变一个可观察行为。缺少设备、资产或已知实验参数时明确记录，不自动编造。
4. 确认任务归属为 `static/<case>`、`dynamic/<case>` 或 `shared`；case入口在 `benchmarks/static/`、`benchmarks/dynamic/`。模型、权重、任务与环境按case绑定；新增adapter不意味着支持其他组合。静态/动态控制实现分别演进，共享底层工具。

## 轻量化方法参考

优化工作优先参考外部 `$edge-model-lightweight`（[来源](https://github.com/Ther-nullptr/edge-model-lightweight-skill)），按需阅读计时、profile、算子/精度和质量评估部分。仓内 [edge-inference-optimization](.agents/skills/edge-inference-optimization/SKILL.md) 补充case约束。外部skill独立安装，CPU工具和CI保持自包含。

- 保留静态/动态case绑定、Kinetix原生JAX路径和各自控制协议；方法参考不要求统一模型输入、动作空间或执行器。
- 原始实现、优化全精度和轻量化实现可分别比较；量化专项增量收益与主表相对固定case baseline的总收益分别命名，遵循 [加速比协议](docs/protocols/speedup-metrics.md)。
- skill内的估算、示例和外部实测摘要只作方法参考，不作为本仓性能证据；实际结果仍需本case、目标设备及声明时间域的验证。实验过程记录继续按本仓版本管理边界留在本地或Issue/PR。

## 修改边界

- 一个任务一个 owner；只改约定文件。公共接口、时间语义或数值门槛变化须说明影响，并更新对应文档和案例。
- 默认CPU检查和公共文档链接须在干净克隆中自包含。模型试跑可通过显式参数使用case声明的外部源码、checkpoint和资产；不得硬编码、自动扫描或发布父目录私有资料、凭据、本机路径和外部软链接。
- 未来通用核心不依赖机器人专用 adapter；Torch、CUDA 和模拟器不进入当前 CPU 契约工具的默认依赖。
- 区分 wall time、sim time、device duration、result release、boundary wait 和 observation age。
- 静态 `paper_async` 按论文历史观测 `t−n′` 与周期模型验收，`history_observation` 是其实现方式；真实并发是可选扩展。`paper_async` 仅为独立case manifest标签，不改变通用v1 schema中 `async` 的既有语义。state/warmup规则由case显式声明；论文模型的 `Tact` 不修改physics dt，缺少 `Tinf` 时不生成周期/加速比，host `eval_s` 不充当论文周期。
- 保留真实 backend、dtype 和 fallback 证据。fake quant、GPU enqueue 返回和单个 GEMM 数字不代表真实端到端加速。
- 不在公共PR CI中调用实验室GPU runner或下载模型权重。模型/GPU试跑使用独立case环境并记录来源、设备和验证范围；CPU检查不依赖这些外部资源。
- 使用已有风格，避免无关重构和全仓格式化。新增测试针对行为和失败边界；文字修订无需新测试。

## 验证与交接

按改动选择验证。纯文档/模板变更核对链接、命令、图稿和 `git diff --check`，不新增测试或运行GPU。Python行为或数据契约变更先做相关检查，提交前执行CPU检查：

```bash
python -m ruff check tools tests benchmarks src
python -m ruff format --check tools tests benchmarks src
python tools/validate_contracts.py --examples
python -m pytest -q
python -m compileall -q tools benchmarks src
```

依赖见 `CONTRIBUTING.md`。Ruff负责格式，`compileall`负责语法。按实际命令报告结果；实时性、数值与任务质量分别提供对应证据。

交接时检查diff，在Issue/PR记录事实、证据和下一步。代码、稳定协议、使用说明和小样例进入Git；过程记录、综述和生成图表放 `.local/`、`docs/handoffs/` 或研究目录。公共文档链接须可共享；合并与发布由维护者负责。

## 算子优化与可视化

推理优化入口见 `benchmarks/inference/README.md`。每项优化保留独立开关；默认不开启。每轮以实际无优化BF16锚点直接计算完整policy时延比；低精度另外对照共享优化相同的BF16。先验证实际加载的源码、算子命中及动作差异，再报告性能；单个输入的逐值一致不等于任务集成功率已验证。

每轮通过外部 `profile-visualizer` skill生成历史图和可审阅记录，包含变慢/数值失败的候选。skill只渲染实测数据，不启动GPU、不估造缺失比较。实验ledger、trace、输入和生成图留在 `runs/`；稳定代码、来源记录和命令说明进入Git。

INT实现参照本仓独立FP副本的loader/Linear/pack/prepare/forward_packed分层；不导入父目录VLM代码。格式、scale、zero-point、pack版本和实际硬件后端显式区分，不能把INT4称为FP4。

Cosmos固定/渐进整数与共享VAE配方见 [量化指南](docs/cosmos-quantization.md)。新调优按明确任务执行，不自动追加压测；全量任务质量另行验收。π0.5优先text/LLM，diffusion/action expert暂缓。INT/CUDA及固定CUTLASS头文件从robotics包加载，不使用外部VLM、QuaRot或Mini QServe目录。先测固定INT4/INT8完整调用，再扩展渐进覆盖；INT4可保留较大动作偏差继续测量，须记录偏差、覆盖与任务结果，零成功时不计算成功条件任务加速比。

## LingBot＋RoboTwin

LingBot优先适配RoboTwin，入口为 `benchmarks/static/lingbot_robotwin/`。模型和仿真使用独立本地环境；支持sync和paper_async；禁止把WebSocket async命名当成论文抽象。paper_async的n′范围0..16，每个名义关键帧k使用max(0,k−n′)的完整快照，负索引重复初始帧；原始预测动作缓存条件保持原位。不得只延迟推理请求而让KV/VAE读取较新的观测。首次执行16条指令，后续完整块32条，每4条采样实际观测更新KV/VAE；终止/预算截断不更新未执行动作。`primitive_steps` 是RoboTwin接受的take_action指令数，不是250Hz物理tick。此入口默认复用显式提供的本地checkpoint、模拟器资产和环境；执行时不自动下载，输出写入调用者目录，不修改共享资源。

## KINETIX动态case

`benchmarks/dynamic/kinetix` 是可运行入口，实际flow model、KINETIX环境和Jax2D源码在 `src/robotics_bench/kinetix/` 维护；不得重新导入参考仓库或已安装的顶层kinetix/jax2d包。外部只绑定checkpoint与Python框架环境。默认fine（原名native-blend），原生dt/frame_skip/solver/碰撞参数固定；按processed actuator command混合，控制噪声对新旧候选同源，无初始prefetch，旧队列保留预测尾部。coarse（原名legacy-round）只改变延迟映射，不能修改物理网格。zero/full-old/constant控制保留原生编译边界并通过逐值对齐；缩小物理时间片的研究在独立分支，不改主线默认参数。每个质量/延迟单元按相同seed列表评估，异常不计为策略失败；统计复用安装包内 `robotics_bench.statistics`，失败按预算惩罚。视频只补齐编码边缘，不改变模拟器。

历史硬件推理延迟统一查阅 [KINETIX 硬件延迟档案](docs/kinetix-hardware-latencies.md) 及其 CSV。按用户约定仅维护 `local_ada`、`rtx3090`、`agx15`、`agx30` 四档；AGX Orin 50W 不纳入该档案、后续汇总或默认延迟映射。档案是历史 PyTorch eager FP16、同架构随机初始化权重的完整同步调用计时，不得作为当前已训练 JAX FP32 policy 的实测，也不得据此直接归因量化收益。查询已有数据先读档案，不自动启动新压测。

硬件延迟研究使用 [配对实验工作流](docs/kinetix-latency-study.md)。主任务按既有零延迟扫描的总体上升趋势选择：catapult、trampoline、mjc_half_cheetah、mjc_walker；允许高步数段小幅回落，不强制单调，也不为重现某种曲线选择任务。每单元512个配对seed。校准后冻结预测，再运行映射验证；复用数据的来源须显式记录，校准重叠点不计为留出验证。动态延迟仍由虚拟时钟注入，不重新启动缩小物理时间片的研究。
