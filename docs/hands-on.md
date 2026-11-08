# 实验实操：从推理速度到任务结果

[English](hands-on.en.md) · [环境配置](environment_setup.md) · [§4.5 预测器实操](kinetix-latency-study.md) · [项目首页](../README.zh-CN.md)

这份指南带你跑一组实验：先准备模型和环境，再比较不同配置，最后查看结果。按下面的问题选择一条路线即可，不用从头运行所有实验。

## 1. 选择要回答的问题

章节编号对应 [arXiv v3](https://arxiv.org/html/2606.28529v3)：§4.5为 **Predictive TISED预测器**，实机实验位于§4.6。

| 问题 | 论文对应内容 | 操作入口 |
| --- | --- | --- |
| 精度降低后，完成任务是否也更快？ | 静态任务，§4.2 | 第3.1节：Cosmos量化pilot |
| 使用更早的观测会怎样改变任务执行？ | 执行感知优化，§4.2 | 第3.2节：π0.5 `paper_async` |
| 动作质量和反应延迟如何共同影响成功率？ | 动态任务，§4.3 | 第4.1节：KINETIX采样步数×延迟矩阵 |
| 换硬件后，最佳配置是否改变？ | 跨硬件实验，§4.4 | 第5节：同口径推理计时与延迟重放 |
| 能否先校准，再预测最佳采样步数？ | Predictive TISED，§4.5 | [独立的分步case study](kinetix-latency-study.md) |
| 如何检查剪枝和运动质量？ | 附录B.1／D.2 | 第3.4／6节：LingBot CPU契约与轨迹分析 |
| 如何分析实机任务记录？ | 实机实验，§4.6 | 第6节：控制器导出的回合与轨迹数据 |

示例使用任务子集，便于先跑通流程。扩大规模时，对所有配置保持checkpoint、任务/初态列表、seed、动作长度和运行环境一致。

## 2. 准备一次工作会话

先完成[环境配置](environment_setup.md)中对应路线：所有实验需要CPU/分析工具；第3.1节需要Cosmos＋LIBERO；第3.2节需要π0.5＋LIBERO；第4–5节及预测器需要KINETIX。模型和资产在准备阶段下载，评估读取本地资源。

以下命令在仓库根目录的Bash中执行。选定空闲GPU，并建立新的结果根目录：

```bash
cd /path/to/robotics-the-speedup-paradox
export ROBOTICS_REPO="$PWD"
export ROBOTICS_CPU_PYTHON="${ROBOTICS_CPU_PYTHON:-$ROBOTICS_REPO/.venv/bin/python}"
export KINETIX_ANALYSIS_PYTHON="${KINETIX_ANALYSIS_PYTHON:-$ROBOTICS_REPO/.local/envs/analysis/bin/python}"
export ROBOTICS_GPU=0
export ROBOTICS_RUN="$ROBOTICS_REPO/runs/hands-on/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ROBOTICS_RUN"
nvidia-smi
```

时间戳在本次会话开始时生成一次。执行同一条路线时沿用 `ROBOTICS_RUN`；各运行入口自行建立子目录。修改配置或重复已完成实验时，使用新的根目录。

先检查CPU工具：

```bash
"$ROBOTICS_CPU_PYTHON" tools/validate_contracts.py --examples
"$ROBOTICS_CPU_PYTHON" tools/compare_speedups.py \
  --input examples/speedups/synthetic.json --markdown
```

应看到契约检查通过，以及标有 `synthetic=true` 的加速比表。这一步用于确认安装和统计口径；下面的运行生成实际测量。

<a id="static"></a>

## 3. 静态任务

### 3.1 Cosmos：渐进量化、推理计时与任务完成

**目标：** 在相同初态上比较原始BF16、优化BF16和INT8→INT4档位。pilot在同一进程中按相同配方完成推理计时与闭环评估。

**第一步：绑定资源并预检。** 环境指南会创建 `.local/cosmos-libero.env`。检查源码、checkpoint、统计量、T5、VAE和LIBERO配置：

```bash
source .local/cosmos-libero.env
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 \
  --seed 195 --env-seed 0 --schedule sync \
  --output-dir "$ROBOTICS_RUN/cosmos-preflight" --dry-run
```

预检通过后，终端会打印运行计划，但不会创建结果目录。下一步才会加载模型并运行模拟器。

**第二步：运行配对pilot。** 五种配置各评估两个初态，共10回合。T0的280个候选Linear均为W8A8；T5有46个W4A4；T10的280个候选均为W4A4。统一使用5个采样步，每个chunk执行16条动作。

```bash
export STATIC_PILOT="$ROBOTICS_RUN/cosmos-tiers"
"$ROBOTICS_COSMOS_PYTHON" benchmarks/inference/evaluate_cosmos.py \
  --gpu "$ROBOTICS_GPU" --suite libero_object --task-id 0 \
  --initial-states 0 1 --seed 195 --env-seed 0 \
  --steps 5 --action-steps 16 --action-time-ms 50 \
  --tiers 0 5 10 --integer-tactic 1 --repeats 9 \
  --output-dir "$STATIC_PILOT"
cat "$STATIC_PILOT/summary.md"
```

比较项为 `original`、`optimized-bf16`、`w8a8`、`w4-t5`、`w4-t10`。共享优化为modulation、gated-residual融合和CUDA Graph；整数配置还开启共享激活准备、激活融合和pack复用。计时与任务结果始终绑定这套配方。

**第三步：读取结果。** `summary.md` 包含成功数、推理毫秒数、推理加速比、全体预算惩罚步数、成功平均步数，以及论文时间模型下的成功任务加速比。`runs.json` 和每个配置的 `episodes.jsonl` 保留原始记录。任务时间估计为逐次实测推理时间之和，加上执行动作数×声明的50 ms动作时间；模拟器宿主耗时单独记录。

成功回合平均chunk直接从实际推理事件计数：

```bash
"$ROBOTICS_CPU_PYTHON" - <<'PYTHON'
import json, os
from pathlib import Path
from statistics import mean
for path in sorted(Path(os.environ["STATIC_PILOT"]).glob("*/episodes.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    successes = [row for row in rows if row["success"]]
    print(json.dumps({
        "configuration": path.parent.name,
        "episodes": len(rows),
        "successes": len(successes),
        "success_rate": len(successes) / len(rows),
        "successful_mean_chunks": mean(row["inference_calls"] for row in successes)
            if successes else None,
    }))
PYTHON
```

零成功配置的成功均值为空。将成功率与加速比一起看：单次推理变快后，任务可能需要更多chunk，也可能有更少回合成功。根据测量结果解释权衡，无需预设曲线形状。

**第四步：扩大规模。** 增加 `--initial-states` 的初态编号、更换 `--task-id`，或扫描0–10之间更多 `--tiers`；每组使用新目录。厨房任务使用 [RoboCasa入口](../benchmarks/static/cosmos_robocasa/README.md)：它有独立资源、H=32和 `--reference-run` 初态对照。上面的pilot属于LIBERO入口。

### 3.2 π0.5：同步基线与paper_async

**目标：** 固定模型、动作执行长度和采样步数，改变观测年龄。每块执行5条动作时，n′=0、2、5均为合法设置；下面在n′=0时显式选择同步基线。

```bash
source .local/pi05-libero.env
for nprime in 0 2 5; do
  schedule=paper_async
  if [ "$nprime" -eq 0 ]; then schedule=sync; fi
  bash benchmarks/static/pi05_libero/run.sh \
    --suite libero_object --episodes 10 --batch-size 1 --seed 42 \
    --n-action-steps 5 --num-inference-steps 10 --quant none \
    --schedule "$schedule" --overlap-actions "$nprime" \
    --gpu "$ROBOTICS_GPU" --record-video \
    --output-dir "$ROBOTICS_RUN/pi05-nprime-$nprime"
done
"$ROBOTICS_CPU_PYTHON" tools/summarize_experiment.py \
  --input "$ROBOTICS_RUN/pi05-nprime-0" \
          "$ROBOTICS_RUN/pi05-nprime-2" \
          "$ROBOTICS_RUN/pi05-nprime-5" --per-task
```

三种配置共30回合。评测器将每组10回合预算分配到suite，实际任务和初态覆盖见 `coverage.json`。沿用已有完整 `libero_object` 设置时，每组改为 `--episodes 500 --batch-size 10`。

在 `paper-async.json` 核对n′和历史观测规则，在 `coverage.json` 核对初态一致性，在 `episode-summary.md` 读取成功率与步数。视频保存完整执行回合。`paper_async` 通过串行评估中的历史图像/state快照测量质量，周期估计另用同配置推理profile。先报告质量与步数；补齐 `Tinf` 后，按[加速比协议](protocols/speedup-metrics.md)计算时间收益。

### 3.3 组合量化与异步观测

Cosmos可在同一命令中启用两类控制。下面另开一组配对实验，固定T5量化，对照同步与n′=2：

```bash
source .local/cosmos-libero.env
for nprime in 0 2; do
  schedule=paper_async
  if [ "$nprime" -eq 0 ]; then schedule=sync; fi
  bash benchmarks/static/cosmos_libero/run.sh \
    --suite libero_object --task-ids 0 --episodes 2 \
    --seed 195 --env-seed 0 --n-action-steps 16 --num-inference-steps 5 \
    --precision int8 --quant-scope dit --quant-tier 5 --integer-tactic 1 \
    --schedule "$schedule" --overlap-actions "$nprime" \
    --gpu "$ROBOTICS_GPU" --record-video \
    --output-dir "$ROBOTICS_RUN/cosmos-t5-nprime-$nprime"
done
```

这组使用不带额外融合开关的T5，计时配方与第3.1节的融合pilot分开。第5.1节给出匹配的固定输入计时。要构成完整四组对照，再用相同其他参数运行BF16的n′=0和n′=2，并分别命名输出目录。

### 3.4 LingBot：剪枝契约与evaluator结果

已合入的LingBot工具包括CPU KV读取窗口选择器和报告评分器。先运行仓内样例：

```bash
"$ROBOTICS_CPU_PYTHON" benchmarks/static/lingbot_robotwin/kv_read_window.py \
  --input benchmarks/static/lingbot_robotwin/kv_read_window.synthetic.json
"$ROBOTICS_CPU_PYTHON" benchmarks/static/lingbot_robotwin/score_reports.py \
  --input benchmarks/static/lingbot_robotwin/scoring.synthetic.json \
  --output "$ROBOTICS_RUN/lingbot-scoring-example"
```

选择器输出保留/丢弃slot，以及独立的视觉与动作预算。`base`、`p1`–`p4`定义见 [KV_READ_WINDOW.md](../benchmarks/static/lingbot_robotwin/KV_READ_WINDOW.md)，当前和未知来源slot保持可读。评分器消费作者evaluator的 `*_analysis.json`，生成逐回合及汇总表；此处输入为合成样例。

实际模型执行按 [LingBot＋RoboTwin指南](../benchmarks/static/lingbot_robotwin/README.md)，使用 `sync`／`paper_async` 入口及完整KV/VAE观测历史规则。Read-window pruning当前提供CPU契约；对应指南列出worker接入与GPU验证步骤，将它与已可运行的控制调度分开使用。

<a id="dynamic"></a>

## 4. 动态任务

### 4.1 KINETIX：分开控制质量与延迟

**目标：** 在可控的采样步数和延迟网格上观察成功率。三个采样步数×三个延迟，各使用相同16个seed，共144回合。

```bash
source .local/kinetix.env
bash benchmarks/dynamic/kinetix/run.sh \
  --levels catapult --flow-steps 1,3,5 --latencies-ms 0,25,50 \
  --mapping fine --episodes 16 --start-seed 0 \
  --output-dir "$ROBOTICS_RUN/kinetix-grid" --dry-run

bash benchmarks/dynamic/kinetix/run.sh \
  --levels catapult --flow-steps 1,3,5 --latencies-ms 0,25,50 \
  --mapping fine --episodes 16 --start-seed 0 \
  --gpu "$ROBOTICS_GPU" --validate-native --record-video \
  --output-dir "$ROBOTICS_RUN/kinetix-grid"

"$ROBOTICS_CPU_PYTHON" benchmarks/dynamic/kinetix/summarize.py \
  --input "$ROBOTICS_RUN/kinetix-grid"
```

读取 `summary.md/.json/.csv`、`episodes.jsonl` 和 `requests.jsonl`。固定N观察延迟响应，固定延迟观察采样质量。`fine` 在原生物理网格的边界槽混合执行器指令；模型预测8条动作，执行4条后刷新，默认控制周期33.333 ms、预算256条指令。延迟参数单位为毫秒，不是再叠加到宿主运行上的sleep时长。

要对照映射方式，在新目录用 `--mapping coarse` 重跑。Coarse取整到控制步网格，同时报告requested/effective延迟；两组保持物理参数、噪声和seed列表一致。

论文§4.5请接着阅读[预测器case study](kinetix-latency-study.md)：完整准备零延迟N=1…5基线，校准固定策略延迟响应，冻结预测，再验证硬件映射条件，并逐项解释报告。

### 4.2 DynamicVLA＋DOM：原生流式推理

DOM通过独立模型与Isaac进程评估持续推进的运动物体场景。在 `.local/dynamicvla-dom.env` 绑定checkpoint、场景、物体、测试JSON及Franka USD，然后比较两种原生调度：

```bash
source .local/dynamicvla-dom.env
# Select two idle GPUs for the model and simulator.
bash benchmarks/dynamic/dynamicvla_dom/run.sh \
  --model-gpu 0 --sim-gpu 1 --episodes 2 --seed 42 --num-steps 10 \
  --output-dir "$ROBOTICS_RUN/dom-native" --dry-run

bash benchmarks/dynamic/dynamicvla_dom/run.sh \
  --model-gpu 0 --sim-gpu 1 --episodes 2 --seed 42 --num-steps 10 \
  --output-dir "$ROBOTICS_RUN/dom-native"

bash benchmarks/dynamic/dynamicvla_dom/run.sh \
  --model-gpu 0 --sim-gpu 1 --episodes 2 --seed 42 --num-steps 10 --streaming \
  --output-dir "$ROBOTICS_RUN/dom-streaming"
```

`server.log`／`client.log` 记录启动和执行；`effective-config.json` 记录时间参数；`episodes.jsonl` 记录结果；`requests.jsonl` 和 `episode-*-controls.jsonl` 记录chunk生成及消费。MP4默认开启。Streaming可能只执行预测块尾部或丢弃过期预测，因此生成chunk、应用chunk与控制步分别计数。

研究采样步数权衡时，保持streaming开启，在相同场景和seed下改变 `--num-steps`，使用新目录。独立控制延迟的网格实验使用第4.1节的KINETIX。完整运行参数见 [DOM指南](../benchmarks/dynamic/dynamicvla_dom/README.md)。

<a id="hardware"></a>

## 5. 硬件计时与预测选点

### 5.1 测量完整policy调用

为第3.3节的普通T5配置捕获真实初始观测，读取对应的实际任务指令，然后在Cosmos模型环境中计时：

```bash
source .local/cosmos-libero.env
export STATIC_CAPTURE="$ROBOTICS_RUN/cosmos-inputs"
"$ROBOTICS_COSMOS_PYTHON" benchmarks/inference/capture_libero.py \
  --case cosmos_libero --suite libero_object --task-id 0 \
  --initial-states 0 --env-seed 0 --gpu "$ROBOTICS_GPU" \
  --output-dir "$STATIC_CAPTURE"
TASK_TEXT=$("$ROBOTICS_CPU_PYTHON" - <<'PYTHON'
import json, os
from pathlib import Path
print(json.loads((Path(os.environ["STATIC_CAPTURE"]) / "observation-000.json").read_text())["task"])
PYTHON
)
"$ROBOTICS_COSMOS_PYTHON" benchmarks/inference/run.py \
  --case cosmos_libero --input "$STATIC_CAPTURE/observation-000.npz" \
  --task "$TASK_TEXT" --gpu "$ROBOTICS_GPU" --steps 5 --seed 195 \
  --precision int8 --quant-scope dit --quant-tier 5 --integer-tactic 1 \
  --warmup 5 --repeats 13 --output-dir "$ROBOTICS_RUN/cosmos-t5-timing"
```

`measurements.json` 包含原始BF16锚点、整数候选、重复采样、中位时延和实际量化覆盖；模型加载、编译、权重准备与预热单独记录。换GPU时，使用相同checkpoint、输入、采样步数、精度、tactic及开关，在新目录执行，manifest记录目标设备。

固定输入测量提供推理耗时；配对闭环记录提供成功率和chunk数。[指标工具](../tools/compare_speedups.py)根据显式输入计算baseline/variant比值，[输入格式](../examples/speedups/README.md)说明必填字段和空值规则。

### 5.2 在KINETIX重放硬件时延

[硬件档案](kinetix-hardware-latencies.md)保存四档历史profile。选定其中一档，导出单case入口接受的两列CSV：

```bash
"$ROBOTICS_CPU_PYTHON" - <<'PYTHON'
import csv, os
from pathlib import Path
source = Path("docs/data/kinetix-hardware-latencies-20260811.csv")
with source.open() as handle:
    rows = [row for row in csv.DictReader(handle) if row["profile_id"] == "agx30"]
assert sorted(int(row["flow_steps"]) for row in rows) == [1, 2, 3, 4, 5]
target = Path(os.environ["ROBOTICS_RUN"]) / "agx30.csv"
with target.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["flow_steps", "latency_ms"])
    writer.writeheader()
    writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
print(target)
PYTHON
source .local/kinetix.env
bash benchmarks/dynamic/kinetix/run.sh \
  --levels catapult --flow-steps 1,2,3,4,5 \
  --latency-profile "$ROBOTICS_RUN/agx30.csv" --mapping fine \
  --episodes 16 --start-seed 0 --gpu "$ROBOTICS_GPU" \
  --output-dir "$ROBOTICS_RUN/kinetix-agx30"
```

`agx30` 将其记录的延迟重放到当前宿主GPU。档案对应历史Torch FP16、随机初始化权重的计时；runner执行训练后的JAX FP32策略，因此结果按“硬件延迟场景”解读。测量实际部署收益时，采集对应policy/backend的profile并保留来源。

§4.5工作流使用另一种四profile CSV格式，并验证完整N=1…5。按[预测器实操](kinetix-latency-study.md)拟合质量/延迟因子，在验证前读出冻结的推荐N，再与实测结果对照；其中也说明了为何校准重叠点不计入留出误差。

<a id="analysis"></a>

## 6. 回合记录、轨迹与实机实验

使用三类互补记录：每回合结果、推理/chunk事件，以及机器人实测轨迹。实机实验（§4.6）由控制器导出开始/成功/超时事件、执行指令和末端实测位置；本仓的共享工具分析这些数据，机器人驱动由对应平台控制器负责。

先用样例检查轨迹分析入口：

```bash
"$ROBOTICS_CPU_PYTHON" tools/embodied/trajectory_metrics.py \
  --input tools/embodied/examples/smooth.csv --frame synthetic \
  --include-series --output "$ROBOTICS_RUN/trajectory-example.json"
"$KINETIX_ANALYSIS_PYTHON" tools/embodied/plot_trajectory.py \
  --input tools/embodied/examples/smooth.csv tools/embodied/examples/perturbed.csv \
  --labels smooth perturbed --frame synthetic \
  --output "$ROBOTICS_RUN/trajectory-example.png"
```

再替换为每回合一份的实测CSV，包含 `t,x,y,z`，单位为秒、米，或通过参数显式声明单位。比较时保持坐标系、采样和滤波一致。速度使用实际相邻时间差；加速度和jerk要求均匀采样，否则输出null并在 `missing_reasons` 中说明原因。成功任务耗时使用开始到终止时间，失败回合在总体步数统计中使用声明预算。

静态case输出用 `tools/summarize_experiment.py --input RUN --per-task`；LingBot作者evaluator报告用 `score_reports.py`，其归一化动作稳定性评分与笛卡尔SI单位jerk分别报告。[轨迹指南](../tools/embodied/README.md)说明输入列、单位、缺失值和绘图。

## 7. 找到结果并继续实验

| 路线 | 首先查看 | 明细／可视化 |
| --- | --- | --- |
| Cosmos量化pilot | `summary.md` | `runs.json`，各配置的 `episodes.jsonl` |
| 静态case入口 | `episode-summary.md` | 入口支持的 `run.log`、`coverage.json`、可选视频 |
| KINETIX矩阵 | `summary.md` | `requests.jsonl`、`episodes.jsonl`、`videos/` |
| DOM | `summary.md` | client/server日志、控制事件、MP4 |
| Predictive TISED | `calibration/report.md`，之后看 `analysis/report.md` | 冻结模型、预测、验证CSV与四联图 |

普通case运行使用新输出目录。预测器有独立的 `status`、分阶段恢复和 `finish` 命令，按其恢复章节操作，无需重建已有计划。导出结果时保留原始记录与manifest，各变体使用独立目录。输出放在忽略的 `runs/`，命令与协议随版本管理。
