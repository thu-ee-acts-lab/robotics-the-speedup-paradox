# Cosmos Policy＋RoboCasa 单环境实验

[环境配置](../../../docs/environment_setup.md) · [配对实验步骤](../../../docs/hands-on.md#static)

使用 [run.sh](run.sh) 在一个固定任务和厨房布局中运行 Cosmos Policy，可选择同步或历史观测异步。[CosmosEngine](../../../src/robotics_bench/engines/cosmos.py) 加载 `suite=robocasa` 模型，[adapter](../../../src/robotics_bench/simulators/robocasa.py) 管理场景，[static_runner](../../../src/robotics_bench/protocols/static_runner.py) 执行动作。

模型读取左、右、腕部三路 RGB 图像和9维 proprio，预测 H=32 个7维动作，默认执行前 n=16 个。默认精度为 BF16；选择整数精度时使用 `--precision int8|int4 --quant-scope dit`，旧的 `--quant` 参数保持 `none`。

## 环境和资源

按 [官方说明](https://github.com/NVlabs/cosmos-policy/blob/main/ROBOCASA.md) 安装 [RoboCasa fork](https://github.com/moojink/robocasa-cosmos-policy)。已验证的组合是 [edd9a328b3ec98050f42d194c1419307a79c4d87](https://github.com/moojink/robocasa-cosmos-policy/commit/edd9a328b3ec98050f42d194c1419307a79c4d87)、robosuite 1.5.1、MuJoCo 3.2.6、Python 3.10 和 PyTorch 2.7.0+cu128。请与 LIBERO 所需的 robosuite 1.4.0 环境分开安装，并让一个进程只运行一种 Cosmos case。

[路径模板](paths.env.example) 绑定RoboCasa Policy `.pt`、`robocasa_dataset_statistics.json`、`robocasa_t5_embeddings.pkl`、VAE和Cosmos提供的 `robocasa_controller_configs.pkl`：

```bash
python -m pip install -r tools/requirements-download.txt
python tools/prepare_resources.py --case cosmos_robocasa \
  --source /path/to/cosmos-policy \
  --robocasa-source /path/to/robocasa-cosmos-policy \
  --python /path/to/robocasa-env/bin/python
source ~/.cache/robotics/env/cosmos_robocasa.env
```

checkpoint/VAE默认写入 `~/.cache/robotics/hub/`，`--root` 改缓存位置，`--reuse policy=DIR --reuse vae=DIR` 复用资源。`--with-dataset` 可下载训练/校准轨迹 `nvidia/RoboCasa-Cosmos-Policy`。厨房资产在已安装、可写的fork中单独准备：

```bash
"$ROBOTICS_COSMOS_ROBOCASA_PYTHON" \
  "$ROBOTICS_ROBOCASA_SOURCE/robocasa/scripts/download_kitchen_assets.py"
```

厨房资产保存在 fork 的 `robocasa/models/assets/`。命令行路径会覆盖 env 配置。预检先检查场景、设施、纹理和 Objaverse 目录，首次初始化再确认网格和纹理能正常加载。

## 启动

从仓库根目录执行，每次使用新输出目录：

```bash
# CPU preflight: no model, simulator, GPU or output directory is created.
bash benchmarks/static/cosmos_robocasa/run.sh \
  --task TurnOffMicrowave --layout-id 1 --style-id 1 \
  --episodes 1 --schedule sync \
  --output-dir runs/static/cosmos_robocasa/sync-001 --dry-run

# Synchronous baseline, with complete three-view video.
bash benchmarks/static/cosmos_robocasa/run.sh \
  --task TurnOffMicrowave --layout-id 1 --style-id 1 --env-seed 0 \
  --episodes 1 --schedule sync --record-video --gpu 3 \
  --output-dir runs/static/cosmos_robocasa/sync-001

# Paper async n-prime=2; verify that initialization matches the baseline.
bash benchmarks/static/cosmos_robocasa/run.sh \
  --task TurnOffMicrowave --layout-id 1 --style-id 1 --env-seed 0 \
  --episodes 1 --schedule paper_async --overlap-actions 2 --record-video --gpu 3 \
  --reference-run runs/static/cosmos_robocasa/sync-001 \
  --output-dir runs/static/cosmos_robocasa/async-001
```

`--task` 接受 checkpoint 对应的24个原始任务名。一次命令选择一个任务和一个 layout/style 组合，`--episodes N` 依次使用 `env_seed + episode_index` 作为环境种子。默认模型 seed 为195，每块执行16个动作（可选1..32），去噪步数为5。`--max-steps` 默认沿用任务预算，例如 `TurnOffMicrowave` 为500步。

## 初始化、异步与动作

每个回合创建带指定 seed 的环境，在 reset 后读取 `get_ep_meta()["lang"]`，再执行10步 settling。策略控制从第0步开始计数。模型与 CPU 上的 T5 缓存会在回合间复用，当前指令必须能在缓存中找到对应 embedding。

适配器先核对 `HYBRID_MOBILE_BASE` 控制器和关节索引，再给7维模型动作追加 `[0,0,0,0,-1]`，形成模拟器需要的12维动作。成功由原生 `_check_success()` 判断，每次完成的 `env.step` 计为一个控制步。

`paper_async` 在控制步 t 选择 t−n′ 的三相机和 proprio 快照，历史不足时使用当前观测，范围是 `0 ≤ n′ ≤ n`。推理和环境仍按顺序执行。`--paper-action-time-ms` 默认50，用于估算论文周期；环境实际控制步长单独记录。计算加速比还需要同配置的 `--paper-inference-time-ms` 及其来源说明。

做配对实验时，将 `--reference-run` 指向已经完成的基线，保持任务、布局和 seed 列表相同。程序在第一次推理前比较物理状态、XML、图像和指令指纹；发现差异会报初始化错误。保存的状态供排查使用，目前不用于跨版本恢复环境。

## 输出与验证范围

| 文件 | 内容 |
| --- | --- |
| `run.log` | 自动保存控制台输出 |
| `case-manifest.json` / `checkpoint-load.json` | 源码、资源身份、参数及真实模型加载审计 |
| `initializations/000000/` | 本回合的 `state.npy`、`model.xml`、场景元数据与指纹 |
| `requests.jsonl` | 请求时的控制步、观测步、实际历史偏移 |
| `episodes.jsonl` / `coverage.json` | 成功、控制步数、模型调用数、预算和覆盖记录 |
| `episode-summary.md` / `.json` | 成功率、全体失败预算惩罚口径和仅成功步数统计 |
| `videos/TASK/` | 可选三视角横排 MP4，包含初始和终止帧 |
| `failure.json` | 初始化、模型或其他运行异常；不伪装成有效失败 episode |

`--record-video` 开启三视角672×224 MP4，`--no-record-video` 关闭（默认）。视频保留初始/终止帧，默认播放30 FPS，仿真频率单独记录。

已有 RTX 6000 Ada 单回合对照使用相同任务、layout/style=1/1、环境 seed0 和模型 seed195；预测 H=32、执行 n=16、采样5步。同步在277步成功，调用18次；n′=2异步在292步成功，调用19次。加载、覆盖和初态检查通过，异步首轮偏移为0，其后为2；视频分别含278帧和293帧。

优化参数包括 `--model-runtime owned|native`、`--enable`、`--precision` 和 `--integer-tactic`，量化范围固定为 `--quant-scope dit`，默认关闭优化。先用 `capture_robocasa.py` 保存三相机观测，再在 [推理 benchmark](../../inference/README.md) 中选择 `--case cosmos_robocasa`，测量 BF16、INT8/INT4 或融合开关。任务收益则使用同配置的闭环结果。

VAE前缀优化与LIBERO共享实现：RoboCasa原始41帧、latent 11帧、条件5帧，编码执行前17帧。它适用于动作输出，省略辅助clean-latent后缀；归一化/SiLU融合可独立开关。配方与计时见 [Cosmos量化指南](../../../docs/cosmos-quantization.md)。
