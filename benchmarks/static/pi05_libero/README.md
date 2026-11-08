# π0.5＋LIBERO 静态 case

[环境配置](../../../docs/environment_setup.md) · [配对实验步骤](../../../docs/hands-on.md#32-π05同步基线与paper_async)

这个入口用 π0.5 在 LIBERO 上比较同步执行和历史观测异步。[run.sh](run.sh) 调用 [run.py](run.py)，再连接兼容的 VLASH sim 评测器；模型使用原生 LeRobot π0.5 checkpoint 和配套的图像、状态处理器。量化与融合参数在本页末尾说明。

## 配置一次本机路径

先按环境指南安装模型、模拟器和兼容评测器。缺少模型或 tokenizer 时，运行 `python tools/prepare_resources.py --case pi05_libero`，再加载生成的 `~/.cache/robotics/env/pi05_libero.env`。模型默认缓存到 `~/.cache/robotics/hub/`，LIBERO 资产放在 `~/.cache/libero/assets/`。评估无需训练轨迹；需要训练或校准数据时再加 `--with-dataset`。更多下载、复用选项见 [资源准备](../../../tools/RESOURCE_PREPARATION.md)。

已有资源可填写 [路径模板](paths.env.example)：

```bash
mkdir -p .local
cp -n benchmarks/static/pi05_libero/paths.env.example .local/pi05-libero.env
# 编辑 .local/pi05-libero.env，填写自己的Python和资源路径
source .local/pi05-libero.env
```

| 变量 | 内容 |
| --- | --- |
| `ROBOTICS_PI05_PYTHON` | 已安装模型/仿真依赖的Python解释器 |
| `ROBOTICS_SIM_SOURCE` | 兼容VLASH sim checkout |
| `ROBOTICS_CHECKPOINT` | 任务匹配的原生LeRobot π0.5 LIBERO checkpoint |
| `ROBOTICS_TOKENIZER` | 与checkpoint匹配的tokenizer目录 |
| `ROBOTICS_LIBERO_CONFIG_DIR` | 含 `config.yaml` 的LIBERO配置目录 |
| `ROBOTICS_QUANT_SOURCE` | 量化适配源码根目录，量化实验需要 |
| `ROBOTICS_KERNEL_SOURCE` | 兼容量化kernel源码根目录，量化实验需要 |

命令行路径会覆盖环境变量。`run.sh` 使用 `ROBOTICS_PI05_PYTHON` 指定的解释器，也可以用该解释器直接运行 `run.py`。每次打开新终端都要加载自己的 env 文件；这些本机路径文件留在 Git 之外。

## 两种运行方式

下面用 GPU 3 举例，请换成本机空闲设备的编号或完整 UUID。第一条命令先做 dry-run，去掉该选项后才实际运行；第二条直接运行异步实验。每次真实运行都使用新的输出目录。

```bash
# 同步基线：先预览，去掉 --dry-run 即执行
bash benchmarks/static/pi05_libero/run.sh \
  --schedule sync --quant none \
  --gpu 3 --output-dir runs/static/pi05_libero/sync-001 --dry-run

# 论文异步：重叠动作数 n′=2
bash benchmarks/static/pi05_libero/run.sh \
  --schedule paper_async --overlap-actions 2 --quant none \
  --gpu 3 --output-dir runs/static/pi05_libero/paper-async-001

```

`--dry-run` 会检查文件并打印运行计划。实际运行时才加载模型、启动模拟器并创建结果目录。新环境建议先跑一个回合，确认模型能加载、机器人能执行动作，再增加回合数。

## 录制机器人完整执行视频

添加 `--record-video` 录制全部实际评估episode，保留主相机的初始帧与成功、失败或超时终止帧：

```bash
bash benchmarks/static/pi05_libero/run.sh \
  --schedule paper_async --overlap-actions 2 --quant none \
  --episodes 1 --record-video \
  --gpu 3 --output-dir runs/static/pi05_libero/video-001
```

视频默认关闭。`--record-video` 录制全部回合，`--no-record-video` 关闭录制，`--video-episodes-per-task N` 录制每个任务前 N 个回合，三者只能选一个。例如 N=1 时，每个任务保留1段视频。`--video-fps 30` 设置播放帧率，不改变控制周期。

视频保存到 `<output-dir>/videos/<task>/init_<初态编号>_<success|fail>.mp4`。任务、初态、帧数与 FPS 会写入 `eval_results.json` 的 `video_paths` / `video_metadata` 和 manifest。录制依赖 imageio/FFmpeg，会增加运行耗时；编码失败时程序报错。已有两组500回合结果没有录视频；录像功能的 CPU 逻辑与入口检查已通过，实际画面需在开启录制的闭环运行中确认。

## 常用参数

直接追加CLI参数，例如 `--suite libero_object --episodes 10 --batch-size 1 --seed 42`。`--episodes` 是整个suite的预算，实际任务/初态分配见 `coverage.json`。

| 参数 | 默认值 / 含义 |
| --- | --- |
| `--suite` | `libero_object` |
| `--episodes` / `--batch-size` | `1` / `1` |
| `--seed` | `42` |
| `--n-action-steps` | `5`，每块实际执行动作数 |
| `--num-inference-steps` | `10`，flow推理步数 |
| `--compile-model` / `--no-compile-model` | 默认不编译；量化smoke要求不编译 |
| `--schedule` | `sync` 或 `paper_async` |
| `--overlap-actions` | 默认0，范围0到 `n-action-steps` |
| `--quant` | 仅 `none`，没有内置量化模型预设 |
| `--record-video` / `--no-record-video` | 录制所有实际评估episode / 关闭，默认关闭 |
| `--video-episodes-per-task` | 可选非负整数，每任务最多录前N条完整episode，0关闭 |
| `--video-fps` | 默认30，仅设置视频播放帧率 |
| `--paper-action-time-ms` | `1000/30`，仅用于论文周期估计 |
| `--paper-inference-time-ms` | 默认不提供，需同配置的推理profile |
| `--paper-inference-time-source` | 提供推理时间时必填，说明其来源与口径 |

参数优先级为 **命令行 > `--config` JSON > 默认值**，完整列表可运行 `run.sh --help` 查看。从异步切回同步时，同时指定 `--schedule sync --overlap-actions 0`；同步模式与正 n′ 一起使用会报错。

## 论文异步与时间口径

`--overlap-actions` 设置 n′，范围为 $0\le n'\le n$。n 是每次实际执行的 `n_action_steps`，与模型预测的50步 horizon 分开。需要新动作块时，评测器通过 `async_delay` 取出 t−n′ 的历史图像和 state；历史不足时使用当前快照。`action_quant=1` 让这里的一步对应一条原始控制指令。这是 [论文§3.1/B.1](https://arxiv.org/html/2606.28529v2) 使用的串行历史观测实验。

[paper_async.py](paper_async.py) 与 [加速比工具](../../../tools/compare_speedups.py) 使用：

$$
C(n')=T_{\mathrm{inf}}+nT_{\mathrm{act}}-\min(T_{\mathrm{inf}},n'T_{\mathrm{act}}).
$$

Tact 默认取 `1000/30 ms`，对应论文中的30 Hz设定。它用于 `paper_model` 估算，不改变 LIBERO 物理步长或视频帧率。要得到周期估算，同时提供 `--paper-inference-time-ms` 和非空的 `--paper-inference-time-source`，说明相同模型、精度、设备和输入下的测量来源；入口保存这段说明，不读取验证来源文件。没有 Tinf 也可以评估质量，此时周期与等待时间为 `null`，状态是 `requires_inference_profile`。

`eval_s` 为串行评估墙钟时间，观测年龄由快照来源确定；任务收益使用同配置profile、实际请求和闭环统计，定义见 [加速比协议](../../../docs/protocols/speedup-metrics.md)。

## 已验证范围与外部依赖

运行记录使用 `environment_protocol=explicit_initial_state_v1`。[适配器](libero_adapter.py) 先 reset，再设置初态并执行 settling。每次显式 reset 才切换到下一个初态；回合终止后，环境保持终止状态。[审计器](evaluation_audit.py) 从环境读取实际任务和初态 ID，检查是否漏跑、重复或与汇总不符。

已有两组原精度 `libero_object` 评估，各覆盖10任务×50初态，使用相同 checkpoint、seed42、batch10、n=5 和10步 flow，关闭编译。同步成功494/500（98.8%），n′=2异步成功464/500（92.8%）；每个初态各运行一次，加载与覆盖检查通过。由于没有采集对应 Tinf，这组记录报告成功率和步数。用 `--episodes 500 --batch-size 10` 可运行相同规模。

- 兼容VLASH sim须提供 `runtime_stack=lerobot`、历史观测和入口参数。已跑通来源为含本地改动的 `d618e497eb6279fc9b95a78e1fa5635a51626a26` 工作树，每次记录source/dirty身份。
- checkpoint、processor、normalization和tokenizer须同源且匹配LIBERO任务；环境包含LeRobot、PyTorch、CUDA、LIBERO及资产。
- 外部 `--quant-ladder` 使用调用者的源码/kernel；本仓 `--precision` 使用随仓算子和CUDA开发工具链。两条路径独立配置。

## 汇总实验控制步数

正常结束自动写出 `episode-summary.md/.json`。总体步数对失败使用声明预算，成功使用实际步数；本case失败运行至预算上限，预算来自 `coverage.json.max_primitive_steps`。

```bash
# 新结果自动读取预算；显示各任务明细
python tools/summarize_experiment.py --input runs/static/pi05_libero/sync-001 --per-task

# 旧libero_object结果缺少预算字段时，显式提供280步
python tools/summarize_experiment.py \
  --input runs/static/pi05_libero/sync-001 runs/static/pi05_libero/paper-async-001 \
  --max-steps 280 --per-task --output results/pi05-step-comparison.csv
```

`--input` 接受运行目录或 `episodes.jsonl`，多个运行分别汇总。默认Markdown；`--format json|csv` 或输出扩展名选择格式，目标文件须新建。JSON包含逐任务信息，Markdown/CSV用 `--per-task` 展开。`--task-max-steps TASK=LIMIT` 可重复覆盖通用预算；缺少失败预算时惩罚项为null。

总体平均步数按回合数加权，成功回合平均步数按成功数加权；没有成功回合时，成功均值为 null。详细计算见 [统计说明](../../../docs/protocols/speedup-metrics.md#8-控制步数统计与实验汇总)。分析未完成的运行时加 `--allow-partial`，输出标为 `partial`；只有裸 ledger 时标为 `unverified`。工具会结合 manifest、coverage 和 ledger 判断运行是否完整。

## JSON各自做什么

CLI用于日常运行，JSON保存预设和自动产物：

| 类别 | 文件 | 用途 |
| --- | --- | --- |
| 输入默认值 / 预设 | [case.json](case.json)、[论文异步预设](paper-async.case.json) | 保留可复用实验设置；高级用法可传 `--config PATH`，命令行继续覆盖 |
| 自动输出 | manifest、加载审计、评测结果等 | 运行后自动生成，无需手工创建或修改 |

外部量化用 `--quant-ladder`、`--quant-selected-profile`、`--quant-profile` 及源码/kernel路径；入口校验profile字段、重名和实际包装层。[load_guard.py](load_guard.py) 检查tied-weight加载覆盖，后端、processor、动作与任务结果各自记录。

| 输出文件 | 内容 |
| --- | --- |
| `case-manifest.json` | 解析后的配置与指纹、外部source身份、checkpoint/processor/tokenizer身份、运行环境 |
| `paper-async.json` | 同步/论文异步约定、n′、历史/state规则、推理时延来源及周期估计；manifest中同时保存 |
| `checkpoint-load.json` | 加载覆盖、tied-weight别名与失败原因 |
| `eval_results.json` | 外部evaluator的评测结果 |
| `episodes.jsonl` | 每个实际任务/初态的成功标记、首次终止步数与环境seed |
| `coverage.json` | 已完成/预期episode数、各任务初态覆盖、控制步数预算 `max_primitive_steps` 及审计状态 |
| `episode-summary.md` / `episode-summary.json` | 正常结束时生成的控制步数统计；区分全体、成功、失败与失败预算惩罚 |
| `videos/<task>/*.mp4` | 启用录制时的完整episode视频；路径与帧信息同时进入评测结果 |
| `failure.json` | 实际执行失败时的错误与traceback |

本入口使用 `pi05-libero-smoke-v1` manifest 和独立的回合记录，由 case 检查器校验。通用 `schemas/` 的使用方式见 [产物说明](../../../docs/protocols/artifacts.md)。如果加载前就失败，可能还没有生成 `checkpoint-load.json`。结果文件留在本地，需要共享的结论写入 Issue/PR；模型接入方法见 [模型协议](../../../docs/protocols/model-adapters.md)。

## 本仓推理实现与优化开关

`--model-runtime owned` 使用仓内PI0.5/PaliGemma/Gemma/SigLIP执行代码，VLASH提供evaluator，LeRobot提供配置/processor；`native` 选择原始参考路径。

优化默认关闭：`flow_loop`、`mask_cache`、`rope`、`gated_residual`、`gelu_mul`、`norm` 独立选择；`cuda_graph` 依赖 `flow_loop`，`empty_image_cache` 依赖 `cuda_graph` 并缓存缺省相机占位编码。`projection_fusion` 可能改变舍入。

低精度示例为 `--precision int8 --quant-scope text`，可加 `shared_quant`、`activation_quant_fusion`；`--integer-tactic 0..7` 选择kernel配置。外部 `--quant-ladder` 与仓内优化互斥，完整计时和数值检查见 [inference benchmark](../../inference/README.md)。
