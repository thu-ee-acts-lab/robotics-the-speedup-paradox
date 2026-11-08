# Cosmos-Policy＋LIBERO 静态 case

[环境配置](../../../docs/environment_setup.md) · [量化与异步实操](../../../docs/hands-on.md#static)

这个入口在单个 LIBERO 环境中运行 Cosmos Policy。你可以通过 [run.sh](run.sh) / [run.py](run.py) 选择同步、历史观测异步和模型精度；内部由 [CosmosEngine](../../../src/robotics_bench/engines/cosmos.py) 推理、[LIBERO adapter](../../../src/robotics_bench/simulators/libero.py) 管理环境、[静态 runner](../../../src/robotics_bench/protocols/static_runner.py) 执行动作。默认使用 BF16，预测 H=16 个七维动作，每次执行前 n 个（默认16，范围1..16），直到成功、环境终止或预算耗尽。

## 配置路径与运行环境

填写 [路径模板](paths.env.example)，显式加载已安装环境与资源：

```bash
mkdir -p .local
cp -n benchmarks/static/cosmos_libero/paths.env.example .local/cosmos-libero.env
# 编辑 .local/cosmos-libero.env，将 PATH/TO 替换为实际路径
source .local/cosmos-libero.env
```

| 环境变量 | 对应 CLI / 内容 |
| --- | --- |
| `ROBOTICS_COSMOS_PYTHON` | `run.sh` 使用的 Python 解释器，须已有兼容模型与仿真依赖 |
| `ROBOTICS_COSMOS_SOURCE` | `--cosmos-source`，包含 `cosmos_policy/` 的外部源码目录 |
| `ROBOTICS_COSMOS_CHECKPOINT` | `--checkpoint`，`Cosmos-Policy-LIBERO-Predict2-2B.pt` 文件 |
| `ROBOTICS_COSMOS_DATASET_STATS` | `--dataset-stats`，配套 `libero_dataset_statistics.json` |
| `ROBOTICS_COSMOS_TEXT_EMBEDDINGS` | `--text-embeddings`，配套 `libero_t5_embeddings.pkl` |
| `ROBOTICS_COSMOS_VAE_CHECKPOINT` | `--vae-checkpoint`，配套视频 tokenizer 的 `tokenizer.pth` |
| `ROBOTICS_COSMOS_LIBERO_CONFIG_DIR` | `--libero-config-dir`，包含 `config.yaml` 的目录 |

命令行路径会覆盖环境变量，`run.sh` 使用指定解释器；也可以直接运行 `run.py`。检查 `config.yaml` 中的 BDDL、初态和资产路径是否正确，入口会设置 `LIBERO_CONFIG_PATH`。

已有实验使用 Python 3.10.12、PyTorch 2.7.0+cu128、MuJoCo 3.2.6、LIBERO 0.1.1 和 robosuite 1.4.0。请与 RoboCasa 的1.5.x环境分开安装。GPU 运行还需要 Cosmos 依赖和 EGL，录像需要 imageio/FFmpeg。安装步骤与检查命令见 [环境指南](../../../docs/environment_setup.md)。

缺少资源时运行 `python tools/prepare_resources.py --case cosmos_libero`，再加载生成的 `~/.cache/robotics/env/cosmos_libero.env`。默认模型在 `~/.cache/robotics/hub/`、资产在 `~/.cache/libero/assets/`，复用方式见 [资源说明](../../../tools/RESOURCE_PREPARATION.md)。加载器将已知Video2World基础引用绑定到显式Policy checkpoint，并核对最终路径。

## 预检与小规模运行

先运行 CPU 预检，检查文件、统计量和 `get_action` 接口。这个步骤会读取文件计算 SHA-256，但不会加载 checkpoint 张量、使用 GPU 或启动模拟器：

```bash
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 \
  --schedule sync --quant none \
  --output-dir runs/static/cosmos_libero/sync-001 --dry-run
```

真实运行指定空闲GPU索引或完整UUID，并使用新输出目录：

```bash
# 同步单 episode，保存完整执行视频
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 \
  --schedule sync --quant none --record-video \
  --gpu 0 --output-dir runs/static/cosmos_libero/sync-video-001

# 论文历史观测抽象：n′=2，仍由单环境串行执行器推进
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 \
  --schedule paper_async --overlap-actions 2 --quant none \
  --gpu 0 --output-dir runs/static/cosmos_libero/paper-async-001
```

控制台输出会自动保存到 `run.log`。运行前关闭环境变量 `COSMOS_SMOKE`，让模型正常加载 checkpoint；命令行的 `--run-kind smoke` 只是运行标签，不会跳过加载。

## 任务选择与总预算

`--episodes` 设置本次运行的总回合数，默认1。`--task-ids` 可使用 `all`，或 `0,2,9` 这样的逗号列表；四个 suite 的任务 ID 都是0..9。程序在所选任务间分配回合，从初态0开始且不重复使用，超过可用初态总数时会报错。运行后在 `coverage.json` 中查看实际跑过的任务和初态。

LIBERO-object含10任务×50初态，完整500回合命令如下；与异步比较时固定资源、任务/初态、seed、n和采样步数：

```bash
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids all --episodes 500 \
  --schedule sync --quant none \
  --gpu 0 --output-dir runs/static/cosmos_libero/object-all-sync-001
```

| 参数 | 默认值 / 约束 |
| --- | --- |
| `--suite` | `libero_object`；另支持 `libero_spatial`、`libero_goal`、`libero_10` |
| `--task-ids` / `--episodes` | `all` / 总预算 `1` |
| `--seed` / `--env-seed` | 推理采样 `195` / 环境 `0` |
| `--n-action-steps` | `16`，范围 1..16；模型预测 horizon 固定 16 |
| `--num-inference-steps` | `5`，正整数 |
| `--max-steps` | 按 suite：spatial 220、object 280、goal 300、10 为 520；可覆盖为正整数 |
| `--schedule` / `--overlap-actions` | 默认 `sync` / `0`；n′ 范围 0..n |
| `--model-config` | `cosmos_predict2_2b_480p_libero__inference_only` |
| `--quant` | 仅 `none`，未接入的量化选项会被拒绝 |
| `--run-kind` | 默认单 episode 为 `smoke`，多 episode 为 `evaluation`；标签不证明验证范围 |

每个回合先 reset、设置指定初态，再执行10步 settling，随后开始策略控制计数。回合终止后，环境等待下一次显式 reset。

## 论文异步与计时

`sync` 在需要新动作块时读取当前观测。`paper_async` 在控制步 t 读取 t−n′ 的主相机、腕相机和 proprio 快照，历史不足时使用当前观测，每个回合重新建立历史。省略 `--schedule` 时，正 n′ 会自动选择异步；显式同步与正 n′ 一起使用会报错。

推理和环境按顺序执行。运行后可在 `requests.jsonl` 中检查 `control_step`、`observation_step`、`history_offset_steps` 和 `inference_index`，确认每次推理实际用了哪一步观测。[paper_async.py](../pi05_libero/paper_async.py) 按下式估算动作块周期：

$$
C(n')=T_{\mathrm{inf}}+nT_{\mathrm{act}}-\min(T_{\mathrm{inf}},n'T_{\mathrm{act}}).
$$

`--paper-action-time-ms` 默认50 ms，用于 `paper_model` 估算，模拟器物理步长保持原样。要计算周期，需要提供 `--paper-inference-time-ms` 和非空的 `--paper-inference-time-source`；入口保存来源说明，不读取来源文件。没有 Tinf 时，周期为 null，状态为 `requires_inference_profile`。实际评估墙钟时间另存为 `eval_s`。

## 视频与自动产物

视频默认关闭。`--record-video` 录制全部回合，`--video-episodes-per-task N` 录每个任务前 N 回合，`--no-record-video` 关闭录制，三者只能选一个。视频从 settling 后的初始画面录到终止画面；`--video-fps` 默认30，只控制播放速度。编码会增加墙钟耗时，写入失败时程序报错。

| 输出 | 内容 |
| --- | --- |
| `case-manifest.json` | 解析选项、计划任务/初态、资源与代码身份、case 指纹、运行依赖和完成状态 |
| `run.log` | 入口自动保存的 Python 控制台日志 |
| `checkpoint-load.json` | 实际加载覆盖审计；审计未通过则不开始 rollout |
| `paper-async.json` | 调度、历史/state 规则及可选论文周期估算 |
| `episodes.jsonl` | 逐 episode 任务/初态、成功、控制步数、推理调用数、终止原因和控制 dt |
| `requests.jsonl` | 逐请求的任务/初态及观测来源控制步 |
| `coverage.json` | 预期/完成 episode 数、初态覆盖、步数预算和运行状态 |
| `eval_results.json` | 正常结束时的总体/逐任务成功统计与 `eval_s`；启用视频时含路径及元数据 |
| `episode-summary.md` / `episode-summary.json` | 正常结束时的控制步数汇总，保留成功统计与失败预算惩罚口径 |
| `videos/<task>/*.mp4` | 启用录制时的完整 episode 视频 |
| `failure.json` | 实际执行异常及 traceback；失败发生前尚未生成的产物可能缺失 |

运行结束后，先查看 `episode-summary.md` 和 `coverage.json`。也可以用 `python tools/summarize_experiment.py --input runs/static/cosmos_libero/sync-video-001 --per-task` 重新统计；分析未完成的运行时需加 `--allow-partial`。本入口使用 `cosmos-libero-run-v1` manifest 和独立的请求记录，实验结果与本机配置留在 Git 之外。

## 验证与版本边界

已有586项 CPU 检查，以及 RTX 6000 Ada 上的固定输入和单回合验证。固定 `sample_libero_10_observation.pkl`、seed195 时，动作与原生接口逐值相同，形状为16×7、dtype为float64、最大绝对误差为0。在 object 任务0、初态0、环境 seed0、H=n=16、5步采样下，同步在137步成功，n′=2异步在154步成功。加载与覆盖检查通过；异步首轮用当前观测，此后回看2步。对应录像为256×256、30 FPS、155帧。

这些记录覆盖固定输入和指定初态闭环；完整推理计时见 [量化指南](../../../docs/cosmos-quantization.md)，RoboCasa见 [对应case](../cosmos_robocasa/README.md)。每次case指纹绑定资源、执行代码与选项：源码有Git信息时记录commit/dirty，否则保存Python源码树SHA-256和 `git_available=false`。

来源与许可见 [第三方说明](../../../THIRD_PARTY_NOTICES.md)，接口、仿真与计时分别遵循 [模型接入](../../../docs/protocols/model-adapters.md)、[仿真协议](../../../docs/protocols/simulation.md) 和 [加速比定义](../../../docs/protocols/speedup-metrics.md)。

## 本仓执行代码与优化

`--model-runtime owned` 加载并审计权重后使用仓内policy/sampler/DiT；loader、VAE和attention库来自兼容依赖。`native` 选择原始参考路径。

优化默认关闭，`modulation`、`gated_residual`、`cuda_graph` 独立选择。整数用 `--precision int8|int4 --quant-scope dit`，旧协议保持 `--quant none`；可加 `shared_quant`、`activation_quant_fusion`。`modulation_quant` 需整数精度与modulation，融合调制/量化并保留原生LayerNorm。`--precision int8 --quant-scope dit --quant-tier N` 选择 [渐进INT4覆盖](../../../docs/protocols/progressive-quantization.md)。

`vae_norm_fusion`、`vae_silu_fusion`、`vae_condition_prefix` 与RoboCasa共享实现。LIBERO使用原生16帧编码窗口，执行33帧中的前17帧，保留9帧latent/噪声/DiT形状；动作输出优化省略辅助clean-latent后缀。命令、数值检查和计时见 [inference benchmark](../../inference/README.md)。
