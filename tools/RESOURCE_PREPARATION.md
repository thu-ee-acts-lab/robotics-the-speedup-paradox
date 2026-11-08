# 模型和数据资源准备

[prepare_resources.py](prepare_resources.py) 为 `cosmos_libero`、`pi05_libero`、`cosmos_robocasa` 下载或复用资源并生成env配置，供离线评估使用。完整安装与首次运行见 [环境指南](../docs/environment_setup.md)。

## 默认下载位置

| 资源 | 默认目录 |
| --- | --- |
| 模型、VAE、文本 tokenizer | `~/.cache/robotics/hub/models--ORG--REPO/snapshots/COMMIT/` |
| 可选训练/校准轨迹 | `~/.cache/robotics/hub/datasets--ORG--REPO/snapshots/COMMIT/` |
| LIBERO mesh、texture、场景资产 | `~/.cache/libero/assets/` |
| RoboCasa 厨房资产（单独准备） | 兼容fork源码下的 `robocasa/models/assets/` |
| 生成的路径配置 | `~/.cache/robotics/env/CASE.env` |
| 下载来源记录 | 与 env 同目录的 `CASE.env.resources.json` |
| 生成的 LIBERO 配置 | `~/.cache/robotics/configs/CASE/config.yaml` |

`--root /data/robotics` 优先于环境变量 `ROBOTICS_RESOURCE_ROOT`，两者都未设置时使用 `~/.cache/robotics`。本工具显式传递 Hugging Face 的 `cache_dir`，所以 `HF_HOME` 不改变上表模型和数据集的下载位置。具体文件路径由下载结果返回并写入 env；[Hub 缓存与 local_dir 行为](https://huggingface.co/docs/huggingface_hub/guides/download) 由 `huggingface_hub` 管理。

**LIBERO assets固定为 `~/.cache/libero/assets`。** 已验证wheel优先查包内资产，缺失时查该缓存并忽略YAML资产路径。自定义fork按case手工绑定布局。

模型、下载缓存、数据和生成配置均不进入 Git。放在仓库内时，使用已忽略的 `.local/`、`artifacts/` 等目录。

RoboCasa使用 `--case cosmos_robocasa` 准备专用Policy、统计量、T5和VAE，变量前缀为 `ROBOTICS_COSMOS_ROBOCASA_*`。`--source` 指向Cosmos，`--robocasa-source` 指向模拟器fork，`--python` 选择解释器。厨房资产按 [case指南](../benchmarks/static/cosmos_robocasa/README.md) 单独下载；可选 `--with-dataset` 获取 `nvidia/RoboCasa-Cosmos-Policy`。

## 新机器准备命令

从仓库根目录执行。`--dry-run` 只需 Python 标准库，不联网、不创建文件，也不检查占位路径是否存在：

```bash
python tools/prepare_resources.py --case cosmos_libero --dry-run
python tools/prepare_resources.py --case pi05_libero --dry-run

# Install only the optional download client.
python -m pip install -r tools/requirements-download.txt

# Prepare the default Cosmos inference resources.
python tools/prepare_resources.py --case cosmos_libero
source ~/.cache/robotics/env/cosmos_libero.env
```

上面的简短命令只配置模型资源。若同时提供已安装的运行环境与源码位置，工具会生成完整的启动路径配置：

```bash
python tools/prepare_resources.py --case cosmos_libero \
  --root /data/robotics \
  --source /path/to/cosmos-policy \
  --python /path/to/cosmos-env/bin/python \
  --libero-root /path/to/cosmos-env/lib/python3.10/site-packages/libero/libero

source /data/robotics/env/cosmos_libero.env
bash benchmarks/static/cosmos_libero/run.sh \
  --task-ids 0 --episodes 1 --schedule sync --quant none \
  --output-dir runs/static/cosmos_libero/demo-001 --dry-run
# After preflight, remove --dry-run and add --gpu with your selected GPU.
```

π0.5 使用同一工具，提供它自己的运行环境和兼容 sim 源码：

```bash
python tools/prepare_resources.py --case pi05_libero \
  --source /path/to/compatible-vlash-sim \
  --python /path/to/pi05-env/bin/python \
  --libero-root /path/to/pi05-env/lib/python3.10/site-packages/libero/libero

source ~/.cache/robotics/env/pi05_libero.env
bash benchmarks/static/pi05_libero/run.sh \
  --schedule paper_async --overlap-actions 2 --quant none \
  --output-dir runs/static/pi05_libero/demo-001 --dry-run
```

`--libero-root` 指向含 `bddl_files/` 和 `init_files/` 的已安装包目录；这些任务定义与初态随兼容 LIBERO 包提供。工具检查四个 suite 的 40 个 BDDL/初态文件是否成对存在，生成配置以避免首次 import 的交互提示。不提供该参数时保留现有 LIBERO 配置变量，终端会提示尚未生成配置。

先按 [环境指南](../docs/environment_setup.md) 安装CUDA、Torch、模型和模拟器依赖，再用 `--source` 与 `--python` 绑定已有位置。工具完成后写入 `resources_prepared`，表示资源文件准备就绪；随后执行case预检与单回合GPU运行。

## 下载内容与已有资源复用

| case / 名称 | 来源与内容 |
| --- | --- |
| Cosmos `policy` | [NVIDIA Cosmos Policy LIBERO](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B)：Policy `.pt`、数据统计 JSON、预计算 T5 embedding |
| Cosmos `vae` | [NVIDIA Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)：仅 `tokenizer/tokenizer.pth` |
| π0.5 `policy` | [LeRobot π0.5 LIBERO](https://huggingface.co/lerobot/pi05_libero_finetuned_v044)：模型、配置及配套 processor 状态；不使用 VLASH 微调权重 |
| π0.5 `tokenizer` | [Google PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224)：仅 tokenizer/processor 配置和词表，不下载 PaliGemma 模型 |
| 两者 `libero-assets` | 当前 LIBERO wheel 使用的 [libero-assets](https://huggingface.co/jadechoghari/libero-assets)：场景、网格与纹理 |
| Cosmos `dataset`，可选 | [LIBERO-Cosmos-Policy](https://huggingface.co/datasets/nvidia/LIBERO-Cosmos-Policy)：包含成功与失败轨迹的原生 HDF5 数据 |
| π0.5 `dataset`，可选 | [LeRobot LIBERO](https://huggingface.co/datasets/lerobot/libero)：LeRobot 格式轨迹；不是 Cosmos HDF5 格式 |

只有 `--with-dataset` 才下载最后两项。LIBERO 全任务闭环评估使用任务初态和模拟器，不读取训练轨迹；预计算 T5 和归一化统计属于推理必需文件，始终准备。数据集路径导出为 `ROBOTICS_COSMOS_DATASET` 或 `ROBOTICS_DATASET`，供后续校准工具使用，当前评测器不消费这两个变量，也不自动转换数据格式。

```bash
# Dataset downloads are opt-in. Use a new env filename for a changed plan.
python tools/prepare_resources.py --case cosmos_libero --with-dataset \
  --env-file ~/.cache/robotics/env/cosmos_libero_with_data.env

# Reuse existing local directories without copying or downloading them.
python tools/prepare_resources.py --case cosmos_libero \
  --reuse policy=/path/to/Cosmos-Policy-LIBERO-Predict2-2B \
  --reuse vae=/path/to/Cosmos-Predict2-2B-Video2World \
  --reuse libero-assets="$HOME/.cache/libero/assets" \
  --env-file .local/cosmos-resources.env
```

重复 `--reuse NAME=PATH` 复用资源；全部复用可离线准备。VAE目录须含 `tokenizer/tokenizer.pth`；工具检查必需文件、非空/LFS状态及π0.5 processor引用，记录 `origin=local`。远端版本和完整哈希由来源证据确认。

checkpoint/VAE使用固定默认版本，可用 `--revision NAME=COMMIT` 覆盖；其余资源下载前解析为commit。记录版本、文件数和字节数，拒绝缺失、空文件、LFS pointer和大小不匹配。失败后重跑同一命令复用Hub缓存。

若 Hub 要求权限，例如 PaliGemma，先在模型页面接受相应条款，再按 Hugging Face 官方流程登录。凭据由 Hub 客户端读取，不写入 env 或来源记录。离线模式变量 `HF_HUB_OFFLINE=1` 须在在线准备前取消。

生成的 env、来源记录和 LIBERO 配置只接受新文件或完全相同内容，拒绝覆盖已有不同配置。修改资源计划时选择新的 `--env-file`；如需更换 LIBERO 包位置，同时使用新的 `--root`。下载完成后显式 `source`，CLI 路径参数仍优先于环境变量。

## Cosmos 的额外基础权重引用

加载时将已知Video2World基础引用绑定到显式Policy文件，退出时恢复解析器。核对最终 `checkpoint.load_path` 与VAE路径，并写入 `checkpoint-load.json`；无需额外下载该基础引用的权重。
