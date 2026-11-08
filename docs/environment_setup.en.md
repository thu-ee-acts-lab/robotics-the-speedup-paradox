# Environment setup and first run

[中文](environment_setup.md) · [Hands-on experiments](hands-on.en.md) · [§4.5 predictor](kinetix-latency-study.en.md) · [Project overview](../README.md)

Choose the experiment before installing GPU dependencies. **KINETIX** is the shortest path to dynamic quality–latency experiments and the §4.5 predictor. **Cosmos + LIBERO** provides static quantization and `paper_async`; RoboCasa uses a separate simulator environment. PI0.5, LingBot and DOM have their own runtime bindings below.

Use Linux with an NVIDIA driver, space for the selected checkpoint/assets, and an available GPU. Run `nvidia-smi` to check the driver and device occupancy. Commands below install dependencies or download resources when executed; reuse existing resources by supplying their absolute paths. Complete resource preflight and one episode on each new environment before starting a matrix.

<a id="1-cpu-environment"></a>
## 1. CPU tools and analysis environment

From the repository root, create the general CPU environment and the predictor's plotting/fitting environment. Python 3.11 is used throughout these examples. Keep `ROBOTICS_REPO` as the absolute repository root when changing directories.

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

**You should see:** contract validation prints `VALID`; the speedup example prints a synthetic comparison table. These environments contain NumPy/SciPy and CPU tooling. The analysis environment adds Matplotlib; model runtimes are configured separately.

<a id="2-separate-gpu-environments"></a>
## 2. Choose a runtime

| Case | Model / simulator environment | Resource binding |
| --- | --- | --- |
| KINETIX | Python 3.11, JAX/Flax; source and levels packaged here | `kinetix.env` |
| Cosmos + LIBERO | Cosmos CUDA 12.8 stack, robosuite 1.4.0 | `cosmos-libero.env` |
| Cosmos + RoboCasa | Separate Cosmos stack, robosuite 1.5.1 and RoboCasa fork | `cosmos-robocasa.env` |
| PI0.5 + LIBERO | Compatible VLASH evaluator, LeRobot and LIBERO | `pi05-libero.env` |
| LingBot + RoboTwin | Separate model and SAPIEN/cuRobo environments | `lingbot-robotwin.env` |
| DynamicVLA + DOM | Separate PyTorch and Isaac environments | `dynamicvla-dom.env` |

The following combinations were used for existing static runs; use the upstream install recipe and verify its resolved versions on your machine.

| Dependency | PI0.5 + LIBERO | Cosmos + LIBERO | Cosmos + RoboCasa |
| --- | --- | --- | --- |
| Python | 3.10 | 3.10 | 3.10 |
| PyTorch | 2.7.1+cu126 | 2.7.0+cu128 | 2.7.0+cu128 |
| NumPy | 1.26.4 | 2.2.6 | 2.2.6 |
| MuJoCo | 3.8.0 | 3.2.6 | 3.2.6 |
| robosuite | 1.4.0 | 1.4.0 | 1.5.1 |
| Additional dependencies | LeRobot 0.4.1, Transformers 4.53.3 | LIBERO 0.1.1, Transformer Engine, NATTEN | RoboCasa fork, Transformer Engine, NATTEN |

The validated Cosmos extensions were Transformer Engine `2.2+cu128.torch27` and NATTEN `0.21.0+cu128.torch27` on RTX 6000 Ada. Match compiled extensions to Torch/CUDA. Integer operators additionally need a CUDA development toolkit; [operator setup](../src/robotics_kernels/README.md) describes architecture selection. Keep LIBERO and RoboCasa environments separate.

### Reuse an existing validated environment

If the model and simulator already run on this machine, keep their interpreters and resource files. Copy the chosen `paths.env.example`, replace every placeholder with an absolute path and source it. For example:

```bash
# Existing compatible environments: copy the selected template and fill paths.
mkdir -p .local
cp -n benchmarks/static/cosmos_libero/paths.env.example .local/cosmos-libero.env
${EDITOR:-vi} .local/cosmos-libero.env
source .local/cosmos-libero.env
"$ROBOTICS_COSMOS_PYTHON" -c 'import torch, mujoco, robosuite; print(torch.__version__, mujoco.__version__, robosuite.__version__)'
```

Use the corresponding template under `benchmarks/static/<case>/` or `benchmarks/dynamic/<case>/` for the other cases. Preserve a virtual environment's `bin/python` path rather than resolving its symlink to the base interpreter. Run the case's `--dry-run`, then one episode as shown below. Installation/download subsections are for missing environments or resources.

<a id="kinetix-dynamic-case"></a>
## 3. KINETIX: install, bind and validate

### Install the JAX runtime

The [runtime requirements](../benchmarks/dynamic/kinetix/requirements-runtime.txt) specify JAX 0.4.35, jaxlib 0.4.34, Flax 0.10.2, NumPy 1.26.4 and the CUDA 12 plugin. Choose the target GPU before the import check if sharing a machine.

```bash
cd "$ROBOTICS_REPO"
python3.11 -m venv .local/envs/kinetix
.local/envs/kinetix/bin/python -m pip install \
  -r benchmarks/dynamic/kinetix/requirements-runtime.txt
export ROBOTICS_KINETIX_PYTHON="$ROBOTICS_REPO/.local/envs/kinetix/bin/python"
CUDA_VISIBLE_DEVICES="$ROBOTICS_GPU" "$ROBOTICS_KINETIX_PYTHON" -c 'import jax; print(jax.__version__); print(jax.devices()); assert any(d.platform == "gpu" for d in jax.devices())'
```

**You should see:** the import check lists a GPU device. A CPU-only device list points to the driver/CUDA plugin setup; resolve that before running GPU studies.

### Prepare checkpoints

With existing assets, set `ROBOTICS_KINETIX_POLICY_DIR` to their absolute directory. Each selected task needs `worlds_l_<task>.pkl`. The [upstream RTC project](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) publishes the policy files under `gs://rtc-assets/bc/24/policies/`. For a new machine with the Google Cloud CLI installed:

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

Pickle checkpoints are loaded from the explicitly selected trusted source. Training trajectories are optional for evaluation. Download just `catapult` for the single-task predictor walkthrough; the loop prepares all four study tasks.

### Save paths, preflight, then run one episode

The environment file stores absolute paths so it also works from another terminal or worktree. Its parent `.local/` and all experiment output directories stay outside Git.

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

**You should see:** dry-run prints the plan without creating `setup-001`; the real run creates `case-manifest.json`, `episodes.jsonl`, `requests.jsonl` and `summary.md`. `--validate-native` checks native endpoint equivalence. Continue with the [§4.5 predictor walkthrough](kinetix-latency-study.en.md), using a new output directory for its quality baseline.

## 4. Cosmos: model stack and simulator assets

### Build the upstream container

The commands follow NVIDIA's [Docker setup](https://github.com/NVlabs/cosmos-policy/blob/main/SETUP.md), [LIBERO setup](https://github.com/NVlabs/cosmos-policy/blob/main/LIBERO.md) and [RoboCasa setup](https://github.com/NVlabs/cosmos-policy/blob/main/ROBOCASA.md). Install Docker and the NVIDIA Container Toolkit on the host first. The image includes CUDA development tools, EGL libraries, FFmpeg and `uv`.

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

The selected host GPU is exposed as device `0` in this container. The repository and cache are mounted; output persists after the container exits. Run the remaining Cosmos commands **inside this container**. Existing compatible native environments can instead bind their own interpreter/source paths using the case templates.

### Install one simulator group into its own environment

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

**You should see:** Torch reports CUDA available. Use `COSMOS_CASE=libero` for the next subsection, or repeat this setup with `COSMOS_CASE=robocasa` for RoboCasa. Their `UV_PROJECT_ENVIRONMENT` directories remain separate. The owned model code uses the upstream loader and framework dependencies; preflight checks the selected source interface.

### LIBERO resources and first run

Use the compatible LIBERO/robosuite combination below. Locate its task/initial-state directory without invoking LIBERO's interactive first-import setup, then generate the resource configuration:

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

**You should see:** the single episode produces a completion manifest and summary. The resource tool downloads the policy, normalization statistics, text embeddings, VAE and simulator assets; it writes the selected interpreter/source paths to `.local/cosmos-libero.env`. Run static precision or async comparisons from [Hands-on experiments](hands-on.en.md).

### RoboCasa resources and first run

With the RoboCasa environment selected, install its compatible fork, kitchen assets and macros. These commands use the integration revision recorded in this repository:

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

**You should see:** scene initialization succeeds, `initializations/` contains state/XML metadata, and the episode summary is written. The controller configuration comes from the Cosmos source. Run one case per process; checkpoints, three-camera inputs and normalization remain case-specific.

<a id="4-bind-runtime-paths"></a>
## 5. Bind the other model/simulator stacks

### PI0.5 + LIBERO

Prepare the [VLASH simulation evaluator](https://github.com/mit-han-lab/vlash/tree/sim/libero) and its LIBERO/LeRobot environment first, using the compatibility requirements in the [case guide](../benchmarks/static/pi05_libero/README.md#已验证范围与外部依赖). The integration uses the native LeRobot PI0.5 LIBERO checkpoint. Its evaluator must expose `runtime_stack=lerobot`, historical-observation delay and the arguments checked by preflight; retain the matching compatibility changes when moving machines.

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

**You should see:** dry-run accepts the evaluator interface, LIBERO task files and model/processor configuration. Remove `--dry-run` and add `--gpu 0` to run the episode. For an existing checkpoint/tokenizer, fill [paths.env.example](../benchmarks/static/pi05_libero/paths.env.example) instead of downloading.

### LingBot + RoboTwin

Use the [LingBot model installation](https://github.com/Robbyant/lingbot-va#installation) and [RoboTwin installation](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) in two environments. The checked model stack uses Python 3.10.16, Torch 2.9.0+cu126, Diffusers 0.36.0, Transformers 5.0.0 and FlashAttention 2.8.3; the simulator uses SAPIEN 3.0.0b1, mplib 0.2.1, cuRobo and Vulkan. Prepare the [RoboTwin posttrain checkpoint](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin), including `transformer/`, `text_encoder/`, `vae/` and `tokenizer/`, and RoboTwin's objects, aloha-agilex and background-texture assets.

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

**You should see:** both import checks pass and dry-run resolves the task and resource files. Remove `--dry-run` and add `--gpu 0` for a single episode. The [case guide](../benchmarks/static/lingbot_robotwin/README.md) covers its model worker, cache protocol and task-seed selection.

<a id="dynamicvla--dom"></a>
### DynamicVLA + DOM

Install an Isaac Sim 4.5-compatible runtime matching the [owned DOM integration](../benchmarks/dynamic/dynamicvla_dom/README.md#environments-and-resources), following the [Isaac Sim installation guide](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/index.html). Prepare the [DynamicVLA DOM checkpoint](https://huggingface.co/hzxie/dynamic-vla-DOM), tokenizer/backbone cache, DOM test JSON, scenes, objects and Franka USD dependencies using the [upstream resource list](https://github.com/hzxie/DynamicVLA). The main branch's adapter uses its recorded 4.5 runtime; choose SDK versions against that adapter when upstream installation instructions change.

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

Set `ROBOTICS_DYNAMICVLA_PYTHON` to the new model environment's absolute `bin/python`, and fill the remaining six paths. **You should see:** preflight prints two process commands. Remove `--dry-run` to start model and simulator on the selected idle GPUs; the run writes `client.log`, `server.log`, `summary.md` and MP4. Use `--no-record-video` to disable video.

<a id="3-models-tasks-and-simulator-assets"></a>
## 6. Resource locations and output files

| Content | Location / preparation |
| --- | --- |
| PI0.5/Cosmos policy, tokenizer, VAE, statistics and T5 | Resource helper; default `~/.cache/robotics/hub/` |
| Optional training/calibration trajectories | Add `--with-dataset` to the helper |
| LIBERO task definitions / initial states | Installed compatible package, selected by `--libero-root` |
| LIBERO meshes and textures | `~/.cache/libero/assets/` |
| RoboCasa kitchen assets | Compatible fork's `robocasa/models/assets/` |
| Generated paths | Helper default `~/.cache/robotics/env/<case>.env`, or explicit `--env-file` |
| Experiment results | Explicit new `--output-dir`, normally under `runs/` |

`--root` / `ROBOTICS_RESOURCE_ROOT` relocates the helper's model/data cache; LIBERO assets keep their native location. `--reuse policy=DIR --reuse vae=DIR` reuses local resources, with other resource names documented in [resource preparation](../tools/RESOURCE_PREPARATION.md). Reuse every required resource for offline preparation. Experiments read local files. Changing the plan requires a new env file; CLI paths override environment variables. Source `.local/<case>.env` in every new terminal, using the same host/container path namespace as when it was written.

<a id="6-logs-and-results"></a>
### Logs, video and summaries

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

Cosmos saves `run.log`; PI0.5 console output can use the same `tee` pattern. Static and KINETIX video default off; DOM video defaults on. Static summaries use `episode-summary.md/.json`; KINETIX and DOM use `summary.md/.json` with their own request/control logs. Full experiment commands, metrics and expected outputs are in [Hands-on experiments](hands-on.en.md).

<a id="5-preflight-and-launch"></a>
## 7. Readiness checks and troubleshooting

1. **Imports:** the selected interpreter imports its framework and simulator dependencies.
2. **Preflight:** `--dry-run` resolves model, source, task and asset paths. It may hash large files.
3. **One episode:** run the same command without `--dry-run`, selecting idle GPUs. Inspect completion, first observation, requests and summary before a larger batch.

| Symptom | Next action |
| --- | --- |
| JAX only lists CPU devices | Check the NVIDIA driver and CUDA 12 JAX plugin in the KINETIX environment |
| `SingleArmEnv` import error | Use LIBERO's robosuite 1.4.0 environment rather than the RoboCasa environment |
| Missing checkpoint/T5/tokenizer | Prepare or explicitly reuse the matching case resources |
| New machine rejects evaluator/source interface | Install the compatible source revision/changes and repeat preflight |
| RoboCasa assets missing or sampling fails | Check kitchen fixtures, textures and Objaverse extraction in the selected fork |
| Reference initialization differs | Compare task, layout/style, seed and assets; inspect `initializations/` |
| CUDA extension fails to build | Match CUDA toolkit, Torch and target GPU architecture; see the operator guide |
| EGL/video failure | Check selected device, EGL libraries, imageio and FFmpeg in the case environment |
| `XML_SetReparseDeferralEnabled` import failure | Install matching Python/libexpat in the analysis environment |
| Output directory already exists | Choose a new directory; resume managed predictor studies with their study commands |
