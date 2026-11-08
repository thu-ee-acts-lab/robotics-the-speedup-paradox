# KINETIX：迭代步数与任务成功率

这个实验比较采样次数对任务成功率的影响：固定 RTC 权重和仿真设置，只改变 flow 采样次数。模型与环境代码已放在仓内，运行前配置 checkpoint 和 Python 环境即可。要研究延迟或预测最佳配置，继续看[动态实操](hands-on.md#dynamic)和[预测器指南](kinetix-latency-study.md)。

## 实验设置

| 项目 | 固定设置 |
| --- | --- |
| 任务 | 12个仓内KINETIX关卡，每关绑定自己的原RTC BC checkpoint |
| 模型执行 | 仓内JAX/Flax flow policy，FP32，8条预测动作 |
| 迭代次数N | 1、2、3、4、5；只改变flow ODE的迭代次数 |
| 重规划 | 每执行4条控制指令重新推理，无初始prefetch |
| 物理 | dt=1/60秒，frame_skip=2，原生solver与碰撞参数 |
| 控制 | 原生30 Hz，最多256条控制；首次原生终止即结束 |
| 噪声 | 每控制步动作噪声标准差0.1 |
| 时延 | 注入时延为0，host运行耗时不推进仿真 |
| 配对 | 每格相同seed 0..127，共12×5×128=7680回合 |

每个 seed 从任务初态开始，到首次成功、失败或预算耗尽时结束，成功标记取原生 `GoalR`／`returned_episode_solved`。环境、动作噪声和 policy 使用各自的随机流。总体步数中，失败回合按 256 步计入，成功回合用实际步数；运行异常单独报告。

## 参考来源与差异

- [VLASH](https://github.com/mit-han-lab/vlash/tree/fa1e65c32b35109904de66e2404880a8bc1ed52b/benchmarks)固定[vlash-kinetix aab1c8e](https://github.com/Sakits/vlash-kinetix/tree/aab1c8e0f6b3ab5f7498d030dafd99707ace4d7f)。`num_flow_steps` 默认 5，评估默认 2048 样本并支持多 GPU 分片，统计首次回合；`step` 选择 checkpoint。
- 本实验用原 RTC BC 权重、horizon=4、零延迟、噪声 0.1，扫描 N=1..5 及 128 个配对 seed；async 训练 checkpoint 是另一配置。
- 仓内 JAX FP32 结果与历史 Torch FP16、不同 RNG/批量组织的结果分别归档，质量曲线保留实际形状。

## 运行

按[环境说明](../benchmarks/dynamic/kinetix/README.md)准备本地路径变量。下载不是运行入口的一部分。

```bash
source .local/kinetix.env

# Preflight only; no simulation is started.
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/flow_quality.py \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" --gpus 1,2,3 \
  --output-dir runs/dynamic/kinetix/flow-quality-001 --dry-run

# Use currently idle GPUs. One persistent process evaluates all N/seeds of a task.
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/flow_quality.py \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" --gpus 1,2,3 \
  --output-dir runs/dynamic/kinetix/flow-quality-001
```

首次尝试可以用 `--levels`、`--flow-steps`、`--episodes` 和 `--start-seed` 缩小规模。每次选择新输出目录。调度器为每个任务启动一个常驻进程，在其中运行所有 N 和 seed，以复用模型和 JIT 编译；主扫描默认不录视频。

## 产物

- `sweep-manifest.json`：预先声明的任务矩阵、完整命令、GPU 分配、完成状态及总 wall time。
- `logs/<task>.log`：每关运行进度与异常信息。
- `<task>/case-manifest.json`：代码、checkpoint、环境参数和运行依赖身份。
- `<task>/episodes.jsonl`：逐回合成功、实际控制步数、预算、seed 与推理次数。
- `<task>/requests.jsonl`：动作块及控制事件记录。
- `<task>/summary.{csv,json,md}`：每个 N 的成功率和失败预算惩罚步数。

任务非零退出或覆盖不完整时，sweep 标为失败并保留部分结果。数据、日志、图和 checkpoint 留在忽略目录。

## 分析与绘图

```bash
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/analyze_flow_quality.py \
  --input runs/dynamic/kinetix/flow-quality-001
```

输出默认在实验目录的`analysis/`中：

| 文件 | 内容 |
| --- | --- |
| `analysis.json`、`report.md` | 完整性、统计设置、运行身份与英文报告 |
| `cell_metrics.csv` | 逐关卡、逐N的成功率、Wilson 95%区间和步数统计 |
| `macro_metrics.csv` | 固定任务集等权平均、区间及相对最高N的配对差异 |
| `paired_vs_reference.csv` | 按关卡比较各N与最高N，包含胜/负配对及检验结果 |
| `task_success_vs_flow_steps.png/.pdf` | 各关卡成功率曲线 |
| `macro_success_vs_flow_steps.png/.pdf` | 等权任务平均成功率曲线 |

分析需要 NumPy，绘图还需要 Matplotlib；可加 `--no-plots` 只生成表格。实验未完成时，加 `--allow-partial` 查看已有回合；完成后再生成总体和配对比较。分析器会检查重复回合、初态、延迟、物理参数以及代码和依赖版本。

报告会生成成功率曲线和配置比较表。查看曲线时，结合每个配置的回合数和统计区间判断结果是否稳定。

## 已完成验证

原 RTC BC `24` checkpoint、JAX/Flax FP32、上述固定设置下，完成 12 关×5 个 N×128 个 seed 的 7680 回合。每个 N 均为 1536 回合，任务等权与按回合汇总在这次平衡设计中相同。

| N | 成功数 / 回合数 | 平均成功率 | 95% seed块bootstrap区间 | 全体预算惩罚平均步数 | 仅成功平均步数 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 1205 / 1536 | 78.45% | 77.08%–79.75% | 111.816 | 72.211 |
| 2 | 1317 / 1536 | 85.74% | 84.05%–87.37% | 103.357 | 77.974 |
| 3 | 1372 / 1536 | 89.32% | 87.89%–90.69% | 99.953 | 81.300 |
| 4 | 1382 / 1536 | 89.97% | 88.61%–91.28% | 98.029 | 80.425 |
| 5 | 1365 / 1536 | 88.87% | 87.37%–90.43% | 100.838 | 81.400 |

N=3、4 相对 N=5 的配对差值 95%区间为−1.50 至+2.35、−0.72 至+2.86 个百分点，均含 0，尚不能确定最优 N 或等价性。增益主要来自 walker/trampoline；简单任务 N=1 接近饱和。48 项 Holm 校正比较中，walker N=1 和 trampoline N=1、2 显著低于 N=5。

总体均值以全部回合为分母，失败计 256 步；成功均值按各 N 的成功子集计算。硬件最优点还需组合延迟响应，见[预测器](kinetix-latency-study.md)。

## 评估加速的边界

评估计时分冷启动、编译/预热和稳态完整 rollout。对照保持 dt、frame_skip、模型/噪声 RNG、动作顺序、预算及逐控制步终止/非有限检查。

`flow_quality.py` 按任务并行并复用编译；三 GPU 总 wall time 约 99.6 分钟，未配套单 GPU 全矩阵基线。历史多 worker 曾提高吞吐却增加单回合耗时，两种指标分别测量。

要检查评估本身能否更快，使用 `benchmark_eval.py` 对照 batch=1 的原路径和动作预处理 JIT。后者编译 motor/thruster 绑定、clip 和 select，物理内核与每步检查保持相同。这个选项只在 benchmark 中使用，`run.py` 和主质量扫描仍走原路径。

```bash
# Check that the selected GPU is idle before timing.
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv

"$ROBOTICS_KINETIX_PYTHON" -B -u benchmarks/dynamic/kinetix/benchmark_eval.py \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" --gpu 0 \
  --level car_launch --flow-steps 5 --seeds 0,1,2,3 --repeats 3 \
  --compare-preprocess --output-dir runs/dynamic/kinetix/eval-speed-001
```

`benchmark.json` 保存源码/关卡/checkpoint SHA256、Git 状态、运行栈、GPU/驱动、初始化、预热、回合时间与轨迹。`reference.pstats`／`reference_profile.txt` 为计时后独立 cProfile。省略 `--compare-preprocess` 测原路径，`--preprocess-jit` 测候选；加速比需要配对数据。

每轮计时交替运行原实现和候选实现，然后在计时之外按 seed 记录完整状态、动作 hash 和语义事件。只有逐控制步轨迹一致、计时回合结果也一致时，才输出 `validated_speedup = reference rollout wall / candidate rollout wall`，否则填 null。

计时包含 reset、完整回合和每步检查，排除冷启动/JIT、文件日志、录像、trace 与 cProfile。比较语义事件时，仅忽略 host policy 耗时。

RTX 6000 Ada、JAX 0.4.35/jaxlib 0.4.34 上的首次单任务正式配对测试：

| 指标 | Reference | Action preprocessing JIT |
| --- | ---: | ---: |
| 12次稳态回合总wall time | 29.563 s | 26.424 s |
| 平均每回合 | 2.464 s | 2.202 s |
| 回合/秒 | 0.406 | 0.454 |
| 环境步进累计host span | 28.476 s | 25.295 s |
| 模型调用累计host span | 0.974 s | 1.003 s |

加速比 1.119×，三轮为 1.097×、1.118×、1.141×，耗时减少约 10.6%。12 对计时结果一致；4 对完整轨迹覆盖每种实现 392 个状态、98 次模型输出及全部语义事件。前一轮 1.132×缺少完整身份记录，采用补齐身份后的测量。

环境 host span 约占原始 rollout 的 96%，含 GPU 等待；`device_get`等待与物理 kernel 计算分别解释。该增益来自环境调用路径，覆盖 Car Launch、N=5、4 个 seed；多任务结果见下节。

旧 jaxlib 的循环/条件 command-buffer 扩展在编译参数阶段失败，未产生计时。设备端回合或跨环境批量化需验证 RNG、首次终止/finite、编译时间和显存；当前保留原物理网格与 runner。GPU 实验独立于公共 CPU CI。

## 多任务加速验证与重复性边界

同一测试覆盖了 12 关、N=5、seed 0..3，每个 seed 做三轮交替计时，共 288 次计时回合和 96 次完整轨迹回合。每个任务的 A/B 在同一 GPU、同一进程运行；三张 RTX 6000 Ada 分担独立任务。任务耗时总和与并行调度 wall time 分开记录。4 个任务因 GPU 忙而延后启动，已完成的测量保持原样。

| 任务 | 已通过逐步检查的加速比 | 逐位轨迹检查 |
| --- | ---: | --- |
| grasp_easy | — | 未通过 |
| catapult | — | 未通过 |
| cartpole_thrust | 1.223× | 通过 |
| hard_lunar_lander | 1.134× | 通过 |
| mjc_half_cheetah | — | 未通过 |
| mjc_swimmer | 1.125× | 通过 |
| mjc_walker | 1.127× | 通过 |
| h17_unicycle | 1.246× | 通过 |
| chain_lander | 1.214× | 通过 |
| catcher_v3 | 1.246× | 通过 |
| trampoline | 1.240× | 通过 |
| car_launch | 1.121× | 通过 |

144 对计时结果都与对应 trace 匹配，检查覆盖每种实现的 4914 个状态和 1243 次模型输出。9 关通过全部检查；另 3 关虽然终态和步数相同，中间状态或动作 hash 仍有差异，所以保留原始耗时，并将有效加速比和整 12 关加速比填为 null。这组 4-seed 性能检查与 128-seed 质量实验分别报告。

grasp_easy、seed0 诊断中，29 步的原始/JIT 预处理命令逐位相同，但两条路径各自重复运行也出现不同状态 hash；终态均成功、29 步。需将原生 GPU 重复性与优化差异分开定位。

Jax2D 关节/碰撞包含重复索引 scatter 累加；[JAX 0.4.35说明](https://github.com/jax-ml/jax/blob/jax-v0.4.35/jax/_src/numpy/array_methods.py)指出并发更新顺序可能不确定。这是候选原因，首个差异算子尚待定位；现有检查阈值、solver、RNG 及物理参数保持不变。

多任务结果可用标准库分析工具重新审查；绘图额外使用 Matplotlib：

```bash
# Each task directory contains the output of benchmark_eval.py --level TASK.
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/analyze_eval_benchmarks.py \
  --input runs/dynamic/kinetix/eval-speed-multitask-001
```

先打开 `analysis/report.md` 查看结果，再看 `task_metrics.csv` 和 `evaluation_speedup.png/.pdf`；完整数据在 `analysis/analysis.json`。分析器会从原始计时和 trace 重新核对来源、配置及一致性。

缺少任务时用 `--allow-partial` 查看部分统计。所有任务完整且通过检查后，会同时给出 rollout 总时长之比和等任务权重几何平均。图中的线段表示三轮测量范围。

复现单任务重复性诊断：

```bash
"$ROBOTICS_KINETIX_PYTHON" -B benchmarks/dynamic/kinetix/diagnose_eval_preprocess.py \
  --policy-dir "$ROBOTICS_KINETIX_POLICY_DIR" --gpu 0 \
  --level grasp_easy --flow-steps 5 --seed 0 --repeatability-trials 10 \
  --output-dir runs/dynamic/kinetix/preprocess-diagnostic-001
```

工具先比较相同输入的预处理，再分别重复原始/候选回合，保存 hash、语义事件与源码身份。额外同步用于诊断，数据与性能样本分开。

12 关原路径中，模型调用 host span 约 3.30%、环境约 96.29%；保持其余工作不变，消除全部模型调用的理想上界为 1.034×。后续重点是 host 调度/同步与批量执行，保留原生物理、独立随机流和每步首次终止/finite 检查，并先区分原生数值波动与优化差异。
