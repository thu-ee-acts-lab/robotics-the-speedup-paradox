# 环境配置与首次运行

[English](environment_setup.en.md) · [实验上手指南](hands-on.md) · [§4.5 预测器](kinetix-latency-study.md) · [项目首页](../README.zh-CN.md)

先选择实验，再安装GPU依赖。**KINETIX** 是动态质量—延迟实验和§4.5预测器的最短上手路径；**Cosmos＋LIBERO** 支持静态量化与 `paper_async`，RoboCasa使用独立模拟器环境。π0.5、LingBot和DOM的环境绑定见后文。

准备Linux、NVIDIA驱动、所选checkpoint/资产的磁盘空间和空闲GPU，先用 `nvidia-smi` 检查驱动与占用。下文命令在执行时会安装依赖或下载资源；已有资源直接填写绝对路径复用。每个新环境先完成资源预检和单回合运行，再启动实验矩阵。

<a id="1-cpu工具环境"></a>
## 1. CPU工具与分析环境

从仓库根目录创建通用CPU环境，以及预测器的拟合/绘图环境。以下统一使用Python 3.11，切换目录时保留指向仓库绝对路径的 `ROBOTICS_REPO`。

```bash
# Run from the repository root.
export ROBOTICS_REPO="$(pwd)"
export ROBOTICS_GPU=0
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
export ROBOTICS_CPU_PYTHON="$ROBOTICS_REPO/.venv/bin/python"
.venv/bin/python tools/validate_contracts.py --examples
.venv/bin/python tools/compare_speedups.py \
  --input examples/speedups/synthetic.json --markdown

python3.11 -m venv .local/envs/analysis
.local/envs/analysis/bin/python -m pip install \
  -r benchmarks/dynamic/kinetix/requirements-analysis.txt
export KINETIX_ANALYSIS_PYTHON="$ROBOTICS_REPO/.local/envs/analysis/bin/python"
```

**检查点：** 契约检查打印 `VALID`，加速比示例输出合成对照表。这两个环境提供NumPy/SciPy和CPU工具，分析环境额外安装Matplotlib；模型运行栈分别配置。

<a id="2-为每个case配置独立gpu环境"></a>
## 2. 选择运行环境

| Case | 模型／模拟器环境 | 资源绑定 |
| --- | --- | --- |
| KINETIX | Python 3.11、JAX/Flax；源码和关卡随仓提供 | `kinetix.env` |
| Cosmos＋LIBERO | Cosmos CUDA 12.8栈、robosuite 1.4.0 | `cosmos-libero.env` |
| Cosmos＋RoboCasa | 独立Cosmos栈、robosuite 1.5.1与RoboCasa fork | `cosmos-robocasa.env` |
| π0.5＋LIBERO | 兼容VLASH evaluator、LeRobot与LIBERO | `pi05-libero.env` |
| LingBot＋RoboTwin | 独立模型与SAPIEN/cuRobo环境 | `lingbot-robotwin.env` |
| DynamicVLA＋DOM | 独立PyTorch与Isaac环境 | `dynamicvla-dom.env` |

下表为已有静态实验使用的组合；安装时结合上游配方，并检查当前机器实际解析出的版本。

| 依赖 | π0.5＋LIBERO | Cosmos＋LIBERO | Cosmos＋RoboCasa |
| --- | --- | --- | --- |
| Python | 3.10 | 3.10 | 3.10 |
| PyTorch | 2.7.1+cu126 | 2.7.0+cu128 | 2.7.0+cu128 |
| NumPy | 1.26.4 | 2.2.6 | 2.2.6 |
| MuJoCo | 3.8.0 | 3.2.6 | 3.2.6 |
| robosuite | 1.4.0 | 1.4.0 | 1.5.1 |
| 其他依赖 | LeRobot 0.4.1、Transformers 4.53.3 | LIBERO 0.1.1、Transformer Engine、NATTEN | RoboCasa fork、Transformer Engine、NATTEN |

已有Cosmos环境在RTX 6000 Ada上使用Transformer Engine `2.2+cu128.torch27` 和NATTEN `0.21.0+cu128.torch27`。编译扩展需匹配Torch/CUDA；整数算子另需CUDA开发工具链，目标架构配置见 [算子说明](../src/robotics_kernels/README.md)。LIBERO与RoboCasa分别安装。

### 复用已经跑通的环境

如果本机已经能运行模型和模拟器，保留原解释器与资源文件。复制所选case的 `paths.env.example`，将全部占位内容改为绝对路径后加载，例如：

```bash
# Existing compatible environments: copy the selected template and fill paths.
mkdir -p .local
cp -n benchmarks/static/cosmos_libero/paths.env.example .local/cosmos-libero.env
${EDITOR:-vi} .local/cosmos-libero.env
source .local/cosmos-libero.env
"$ROBOTICS_COSMOS_PYTHON" -c 'import torch, mujoco, robosuite; print(torch.__version__, mujoco.__version__, robosuite.__version__)'
```

其他case使用 `benchmarks/static/<case>/` 或 `benchmarks/dynamic/<case>/` 下的对应模板。保留虚拟环境的 `bin/python` 路径，而非解析软链接后的基础解释器。然后按下文运行case的 `--dry-run` 和一个回合；缺少环境或资源时再执行安装／下载小节。

<a id="kinetix动态实验环境"></a>
## 3. KINETIX：安装、绑定与验证

### 安装JAX运行栈

[运行依赖](../benchmarks/dynamic/kinetix/requirements-runtime.txt) 指定JAX 0.4.35、jaxlib 0.4.34、Flax 0.10.2、NumPy 1.26.4及CUDA 12插件。共享机器上先选择目标GPU，再执行导入检查。

```bash
cd "$ROBOTICS_REPO"
python3.11 -m venv .local/envs/kinetix
.local/envs/kinetix/bin/python -m pip install \
  -r benchmarks/dynamic/kinetix/requirements-runtime.txt
export ROBOTICS_KINETIX_PYTHON="$ROBOTICS_REPO/.local/envs/kinetix/bin/python"
CUDA_VISIBLE_DEVICES="$ROBOTICS_GPU" "$ROBOTICS_KINETIX_PYTHON" -c 'import jax; print(jax.__version__); print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices())'
```

**检查点：** 导入检查列出GPU设备。如果仅出现CPU，先解决驱动/CUDA插件配置，再开始GPU实验。

### 准备checkpoint

已有资产时，将 `ROBOTICS_KINETIX_POLICY_DIR` 设为其绝对目录。每个选定任务需要 `worlds_l_<task>.pkl`。[上游RTC项目](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) 的模型位于 `gs://rtc-assets/bc/24/policies/`。新机器安装Google Cloud CLI后可执行：

```bash
# Download only when the required checkpoint is not already available.
export ROBOTICS_KINETIX_POLICY_DIR="$HOME/.cache/robotics/rtc-policies"
mkdir -p "$ROBOTICS_KINETIX_POLICY_DIR"
for task in mjc_walker catapult trampoline mjc_half_cheetah; do
  gcloud storage cp \
    "gs://rtc-assets/bc/24/policies/worlds_l_${task}.pkl" \
    "$ROBOTICS_KINETIX_POLICY_DIR/"
done
```

Pickle checkpoint从显式选择的可信来源加载，评估无需训练轨迹。单任务预测器教程只需下载 `catapult`；上面的循环为四个研究任务准备模型。

### 保存路径、预检并运行一个回合

环境文件保存绝对路径，可在新终端或其他worktree复用。`.local/` 与实验输出目录均在Git之外。

```bash
cd "$ROBOTICS_REPO"
mkdir -p .local
cat > .local/kinetix.env <<EOF
export ROBOTICS_KINETIX_PYTHON="$ROBOTICS_KINETIX_PYTHON"
export ROBOTICS_KINETIX_POLICY_DIR="$ROBOTICS_KINETIX_POLICY_DIR"
export KINETIX_ANALYSIS_PYTHON="$KINETIX_ANALYSIS_PYTHON"
EOF
source .local/kinetix.env
bash benchmarks/dynamic/kinetix/run.sh \
  --levels catapult --flow-steps 5 --latencies-ms 0 \
  --episodes 1 --start-seed 0 \
  --output-dir runs/dynamic/kinetix/setup-001 --dry-run
bash benchmarks/dynamic/kinetix/run.sh \
  --levels catapult --flow-steps 5 --latencies-ms 0 \
  --episodes 1 --start-seed 0 --gpu "$ROBOTICS_GPU" --validate-native \
  --output-dir runs/dynamic/kinetix/setup-001
```

**检查点：** dry-run打印计划且不创建 `setup-001`；实际运行生成 `case-manifest.json`、`episodes.jsonl`、`requests.jsonl` 和 `summary.md`。`--validate-native` 检查原生端点一致性。随后进入 [§4.5预测器教程](kinetix-latency-study.md)，为质量基线选择新的输出目录。

## 4. Cosmos：模型栈与模拟器资产

### 构建上游容器

下面采用NVIDIA的 [Docker配置](https://github.com/NVlabs/cosmos-policy/blob/main/SETUP.md)、[LIBERO安装](https://github.com/NVlabs/cosmos-policy/blob/main/LIBERO.md) 与 [RoboCasa安装](https://github.com/NVlabs/cosmos-policy/blob/main/ROBOCASA.md)。宿主机先安装Docker和NVIDIA Container Toolkit；镜像包含CUDA开发工具、EGL库、FFmpeg和 `uv`。

```bash
# Host terminal; choose a GPU that is available for your run.
cd "$ROBOTICS_REPO"
export ROBOTICS_HOST_GPU=0
mkdir -p .local/external
# Reuse an existing compatible checkout instead if available.
git clone https://github.com/NVlabs/cosmos-policy.git .local/external/cosmos-policy
docker build -t robotics-cosmos .local/external/cosmos-policy/docker
docker run --rm -it --gpus "device=${ROBOTICS_HOST_GPU}" --ipc=host \
  -v "$ROBOTICS_REPO:/workspace/robotics" \
  -v "$HOME/.cache:/root/.cache" \
  -w /workspace/robotics --entrypoint bash robotics-cosmos
```

选中的宿主GPU在容器内编号为 `0`。仓库和缓存已挂载，退出容器后产物仍保留。后续Cosmos命令均在**该容器内部**执行。已有兼容的原生环境也可直接用case模板绑定解释器和源码路径。

### 为一个模拟器安装独立依赖组

```bash
# Inside the container; select ONE case per environment.
export ROBOTICS_REPO=/workspace/robotics
export ROBOTICS_GPU=0
export ROBOTICS_CPU_PYTHON="$ROBOTICS_REPO/.local/envs/cpu-container/bin/python"
if [ ! -x "$ROBOTICS_CPU_PYTHON" ]; then
  uv venv --python 3.11 "$ROBOTICS_REPO/.local/envs/cpu-container"
fi
uv pip install --python "$ROBOTICS_CPU_PYTHON" \
  -r "$ROBOTICS_REPO/requirements-dev.txt" \
  -r "$ROBOTICS_REPO/benchmarks/dynamic/kinetix/requirements-analysis.txt"
export KINETIX_ANALYSIS_PYTHON="$ROBOTICS_CPU_PYTHON"
export ROBOTICS_COSMOS_SOURCE="$ROBOTICS_REPO/.local/external/cosmos-policy"
export COSMOS_CASE=libero
# For RoboCasa, use: export COSMOS_CASE=robocasa
export UV_PROJECT_ENVIRONMENT="$ROBOTICS_REPO/.local/envs/cosmos-${COSMOS_CASE}"
cd "$ROBOTICS_COSMOS_SOURCE"
uv sync --extra cu128 --group "$COSMOS_CASE" --python 3.10
export COSMOS_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
uv pip install --python "$COSMOS_PYTHON" \
  -r "$ROBOTICS_REPO/tools/requirements-download.txt"
"$COSMOS_PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
cd "$ROBOTICS_REPO"
```

**检查点：** Torch报告CUDA可用。下一节LIBERO使用 `COSMOS_CASE=libero`；RoboCasa重新执行上述安装步骤并选择 `COSMOS_CASE=robocasa`，两者的 `UV_PROJECT_ENVIRONMENT` 目录分开。仓内模型使用上游loader和框架依赖，预检会核对所选源码接口。

### LIBERO资源与首次运行

使用下面的LIBERO/robosuite兼容组合，在不触发LIBERO首次导入交互的情况下找到任务/初态目录，再生成资源配置：

```bash
# Inside the LIBERO container/environment created above.
export ROBOTICS_COSMOS_PYTHON="$COSMOS_PYTHON"
uv pip install --python "$ROBOTICS_COSMOS_PYTHON" \
  'robosuite==1.4.0' 'mujoco==3.2.6'
export LIBERO_ROOT="$("$ROBOTICS_COSMOS_PYTHON" -c 'from importlib.util import find_spec; from pathlib import Path; roots=[Path(p) for p in find_spec("libero").submodule_search_locations]; print(next(q for p in roots for q in (p,p/"libero") if (q/"bddl_files").is_dir()))')"
"$ROBOTICS_COSMOS_PYTHON" tools/prepare_resources.py --case cosmos_libero \
  --source "$ROBOTICS_COSMOS_SOURCE" --python "$ROBOTICS_COSMOS_PYTHON" \
  --libero-root "$LIBERO_ROOT" --env-file .local/cosmos-libero.env
source .local/cosmos-libero.env
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 --schedule sync \
  --output-dir runs/static/cosmos_libero/setup-001 --dry-run
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 --schedule sync --gpu 0 \
  --output-dir runs/static/cosmos_libero/setup-001
```

**检查点：** 单回合写出完成manifest和汇总。资源工具准备policy、归一化统计、文本embedding、VAE和模拟器资产，并将选定的解释器/源码路径写入 `.local/cosmos-libero.env`。静态精度与异步对照见 [实验上手指南](hands-on.md)。

### RoboCasa资源与首次运行

选择RoboCasa环境后，安装兼容fork、厨房资产与macros。下面使用仓库记录的接入版本：

```bash
# Inside the RoboCasa container/environment created above.
export ROBOTICS_COSMOS_ROBOCASA_PYTHON="$COSMOS_PYTHON"
git clone https://github.com/moojink/robocasa-cosmos-policy.git \
  .local/external/robocasa-cosmos-policy
export ROBOTICS_ROBOCASA_SOURCE="$ROBOTICS_REPO/.local/external/robocasa-cosmos-policy"
git -C "$ROBOTICS_ROBOCASA_SOURCE" checkout edd9a328b3ec98050f42d194c1419307a79c4d87
uv pip install --python "$ROBOTICS_COSMOS_ROBOCASA_PYTHON" -e "$ROBOTICS_ROBOCASA_SOURCE"
"$ROBOTICS_COSMOS_ROBOCASA_PYTHON" "$ROBOTICS_ROBOCASA_SOURCE/robocasa/scripts/download_kitchen_assets.py"
"$ROBOTICS_COSMOS_ROBOCASA_PYTHON" "$ROBOTICS_ROBOCASA_SOURCE/robocasa/scripts/setup_macros.py"
"$ROBOTICS_COSMOS_ROBOCASA_PYTHON" tools/prepare_resources.py --case cosmos_robocasa \
  --source "$ROBOTICS_COSMOS_SOURCE" --python "$ROBOTICS_COSMOS_ROBOCASA_PYTHON" \
  --robocasa-source "$ROBOTICS_ROBOCASA_SOURCE" --env-file .local/cosmos-robocasa.env
source .local/cosmos-robocasa.env
bash benchmarks/static/cosmos_robocasa/run.sh \
  --task TurnOffMicrowave --episodes 1 --schedule sync \
  --output-dir runs/static/cosmos_robocasa/setup-001 --dry-run
bash benchmarks/static/cosmos_robocasa/run.sh \
  --task TurnOffMicrowave --episodes 1 --schedule sync --gpu 0 \
  --output-dir runs/static/cosmos_robocasa/setup-001
```

**检查点：** 场景初始化成功，`initializations/` 含状态/XML信息，并生成回合汇总。控制器配置来自Cosmos源码。一个进程运行一个case，checkpoint、三相机输入和归一化统计按case绑定。

<a id="4-绑定运行路径"></a>
## 5. 绑定其他模型／模拟器栈

### π0.5＋LIBERO

先按 [case兼容性要求](../benchmarks/static/pi05_libero/README.md#已验证范围与外部依赖) 准备 [VLASH模拟评测器](https://github.com/mit-han-lab/vlash/tree/sim/libero) 及其LIBERO/LeRobot环境。该入口使用原生LeRobot π0.5 LIBERO checkpoint；evaluator需要提供 `runtime_stack=lerobot`、历史观测延迟及预检检查的参数，迁移机器时保留对应兼容改动。

```bash
# After installing a compatible evaluator and its model/simulator environment.
export ROBOTICS_PI05_PYTHON=/absolute/path/to/pi05-env/bin/python
export ROBOTICS_SIM_SOURCE=/absolute/path/to/compatible-vlash-sim
export PI05_LIBERO_ROOT=/absolute/path/to/libero/libero
.venv/bin/python -m pip install -r tools/requirements-download.txt
.venv/bin/python tools/prepare_resources.py --case pi05_libero \
  --source "$ROBOTICS_SIM_SOURCE" --python "$ROBOTICS_PI05_PYTHON" \
  --libero-root "$PI05_LIBERO_ROOT" --env-file .local/pi05-libero.env
source .local/pi05-libero.env
bash benchmarks/static/pi05_libero/run.sh \
  --suite libero_object --episodes 1 --batch-size 1 --schedule sync --quant none \
  --output-dir runs/static/pi05_libero/setup-001 --dry-run
```

**检查点：** dry-run接受evaluator接口、LIBERO任务文件及模型/processor配置。去掉 `--dry-run`、添加 `--gpu 0` 运行单回合。已有checkpoint/tokenizer时直接填写 [paths.env.example](../benchmarks/static/pi05_libero/paths.env.example) 即可。

### LingBot＋RoboTwin

按 [LingBot模型安装](https://github.com/Robbyant/lingbot-va#installation) 和 [RoboTwin安装](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) 配置两个环境。已检查的模型栈为Python 3.10.16、Torch 2.9.0+cu126、Diffusers 0.36.0、Transformers 5.0.0和FlashAttention 2.8.3；仿真栈为SAPIEN 3.0.0b1、mplib 0.2.1、cuRobo与Vulkan。准备 [RoboTwin posttrain checkpoint](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin) 的 `transformer/`、`text_encoder/`、`vae/`、`tokenizer/`，以及RoboTwin的objects、aloha-agilex和background-texture资产。

```bash
# After upstream model and RoboTwin setup, fill all five absolute paths.
cp -n benchmarks/static/lingbot_robotwin/paths.env.example .local/lingbot-robotwin.env
${EDITOR:-vi} .local/lingbot-robotwin.env
source .local/lingbot-robotwin.env
"$ROBOTICS_LINGBOT_PYTHON" -c 'import torch, diffusers, transformers, flash_attn, websockets; print(torch.__version__)'
"$ROBOTICS_ROBOTWIN_PYTHON" -c 'import sapien, mplib, curobo; print(sapien.__version__)'
bash benchmarks/static/lingbot_robotwin/run.sh \
  --task adjust_bottle --episodes 1 --start-seed 10000 --model-seed 0 \
  --output-dir runs/static/lingbot_robotwin/setup-001 --dry-run
```

**检查点：** 两个导入检查通过，dry-run解析出任务与资源文件。去掉 `--dry-run`、添加 `--gpu 0` 运行单回合。[case指南](../benchmarks/static/lingbot_robotwin/README.md) 说明模型worker、缓存协议和任务种子筛选。

<a id="dynamicvladom"></a>
### DynamicVLA＋DOM

依据 [仓内DOM接入版本](../benchmarks/dynamic/dynamicvla_dom/README.md#environments-and-resources)，按 [Isaac Sim安装指南](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/index.html) 安装兼容Isaac Sim 4.5的环境。通过 [上游资源列表](https://github.com/hzxie/DynamicVLA) 准备 [DynamicVLA DOM checkpoint](https://huggingface.co/hzxie/dynamic-vla-DOM)、tokenizer/backbone缓存、DOM测试JSON、场景、物体与Franka USD依赖。主线adapter采用已记录的4.5运行栈；上游安装说明升级时，按本仓adapter选择SDK版本。

```bash
cd "$ROBOTICS_REPO"
python3.11 -m venv .local/envs/dynamicvla
.local/envs/dynamicvla/bin/python -m pip install \
  -r benchmarks/dynamic/dynamicvla_dom/requirements-model.txt
cp -n benchmarks/dynamic/dynamicvla_dom/paths.env.example .local/dynamicvla-dom.env
${EDITOR:-vi} .local/dynamicvla-dom.env
source .local/dynamicvla-dom.env
# Here ROBOTICS_DOM_PYTHON is Isaac's python.sh, as in the template.
"$ROBOTICS_DOM_PYTHON" -m pip install \
  -r benchmarks/dynamic/dynamicvla_dom/requirements-simulator.txt
bash benchmarks/dynamic/dynamicvla_dom/run.sh \
  --model-gpu 0 --sim-gpu 1 --episodes 1 --seed 42 --streaming \
  --output-dir runs/dynamic/dynamicvla_dom/setup-001 --dry-run
```

将 `ROBOTICS_DYNAMICVLA_PYTHON` 设为新模型环境 `bin/python` 的绝对路径，并填写其余六项。**检查点：** 预检打印两个进程的命令；去掉 `--dry-run` 后在指定空闲GPU启动模型与模拟器，写出 `client.log`、`server.log`、`summary.md` 和MP4。用 `--no-record-video` 关闭视频。

<a id="3-准备模型任务与资产"></a>
## 6. 资源位置与输出文件

| 内容 | 位置／准备方式 |
| --- | --- |
| π0.5/Cosmos policy、tokenizer、VAE、统计量与T5 | 资源工具，默认 `~/.cache/robotics/hub/` |
| 可选训练／校准轨迹 | 资源工具添加 `--with-dataset` |
| LIBERO任务定义／初态 | 兼容安装包，通过 `--libero-root` 指定 |
| LIBERO网格与纹理 | `~/.cache/libero/assets/` |
| RoboCasa厨房资产 | 兼容fork的 `robocasa/models/assets/` |
| 生成的路径配置 | 工具默认 `~/.cache/robotics/env/<case>.env`，或显式 `--env-file` |
| 实验结果 | 显式的新 `--output-dir`，通常放在 `runs/` 下 |

`--root`／`ROBOTICS_RESOURCE_ROOT` 修改工具的模型/数据缓存根目录；LIBERO资产沿用原生位置。`--reuse policy=DIR --reuse vae=DIR` 复用本地资源，其余资源名见 [资源准备说明](../tools/RESOURCE_PREPARATION.md)。全部复用即可离线准备，实验入口读取本地文件。改变计划时使用新的env文件，CLI路径优先于环境变量。每个新终端都要 `source .local/<case>.env`，并保持生成配置时的宿主／容器路径空间一致。

<a id="6-日志视频与结果"></a>
### 日志、视频与汇总

```bash
# Use the case's setup command without --dry-run and with an idle --gpu 0.
# Example: record all Cosmos + LIBERO episode frames and save launcher output.
mkdir -p runs/logs
set -o pipefail
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 --schedule sync --gpu 0 \
  --record-video --output-dir runs/static/cosmos_libero/video-001 \
  2>&1 | tee runs/logs/cosmos-libero-video-001.log
"$ROBOTICS_CPU_PYTHON" tools/summarize_experiment.py \
  --input runs/static/cosmos_libero/video-001 --per-task
```

Cosmos保存 `run.log`；π0.5可用同样的 `tee` 方式记录控制台。静态case与KINETIX默认关闭视频，DOM默认开启。静态结果使用 `episode-summary.md/.json`；KINETIX与DOM使用 `summary.md/.json`，并保留各自的请求／控制日志。完整实验命令、指标与预期产物见 [实验上手指南](hands-on.md)。

<a id="5-预检并启动"></a>
## 7. 检查点与常见问题

1. **导入：** 选定解释器能加载框架和模拟器依赖。
2. **预检：** `--dry-run` 解析模型、源码、任务和资产路径；大文件计算hash可能需要时间。
3. **单回合：** 同一命令去掉 `--dry-run`，选择空闲GPU运行。检查完成状态、初始观测、请求与汇总后再扩大规模。

| 现象 | 下一步 |
| --- | --- |
| JAX只列出CPU | 检查KINETIX环境中的NVIDIA驱动和CUDA 12 JAX插件 |
| `SingleArmEnv` 导入错误 | 使用LIBERO的robosuite 1.4.0环境，而非RoboCasa环境 |
| checkpoint／T5／tokenizer缺失 | 准备或显式复用与case匹配的资源 |
| 新机器拒绝evaluator／源码接口 | 安装兼容版本及改动，再执行预检 |
| RoboCasa资产缺失或采样失败 | 检查所选fork的厨房设施、纹理及Objaverse解压内容 |
| reference初态不同 | 核对任务、layout/style、seed和资产，查看 `initializations/` |
| CUDA扩展编译失败 | 匹配CUDA工具链、Torch与目标GPU架构，参照算子指南 |
| EGL／视频失败 | 检查case环境的设备选择、EGL库、imageio和FFmpeg |
| `XML_SetReparseDeferralEnabled` 导入错误 | 在分析环境安装匹配的Python/libexpat |
| 输出目录已存在 | 选新目录；预测器托管实验按study命令恢复 |
