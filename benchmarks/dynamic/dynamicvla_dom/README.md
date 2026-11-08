# Native DynamicVLA + DOM

[Environment setup](../../../docs/environment_setup.en.md) · [Streaming walkthrough](../../../docs/hands-on.en.md#42-dynamicvla--dom-native-streaming)

Use this entry point to run DynamicVLA on a single Franka DOM scene. The model, preprocessing, streaming queue and scene/control code are maintained in this repository. Before starting, provide local paths to the PyTorch environment, Isaac SDK, checkpoint and scene assets.

Choose between two native schedules. With `native-non-streaming` (the default), the simulator keeps the previous control target while the model generates the next chunk. With `--streaming`, a worker generates chunks and the client drops expired predictions or merges their remaining actions into the queue. In both modes, the model uses its native precision and the scene keeps advancing during inference.

## Environments and resources

Create separate model and simulator environments. At integration, the upstream setup used Python 3.10, PyTorch 2.7.1, Isaac Sim 4.5.0 and Isaac Lab 2.2.1; the checked model environment used Python 3.11, PyTorch 2.7.1 and Transformers 5.2.0. Install the [model requirements](requirements-model.txt) in the model environment and the [simulator requirements](requirements-simulator.txt) in the compatible Isaac environment. Then run preflight and one episode before a larger batch.

Prepare:

- DOM checkpoint `config.json` and `model.safetensors`, plus its local Hugging Face tokenizer/backbone-config cache.
- DOM test JSON, referenced USD scenes/textures, object assets and local Franka `panda_instanceable.usd` with material dependencies.
- [paths.env.example](paths.env.example) with absolute paths. `ROBOTICS_DOM_PYTHON` accepts Isaac's `python.sh`, a Python executable or `isaaclab.sh` (adds `-p`). Preserve virtual-environment interpreter symlinks.

Workers enforce `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` and use explicit Franka assets. Isaac launchers remove an unrelated parent Conda prefix.

## Launch

From the repository root, source the completed env file and select idle GPUs:

```bash
source .local/dynamicvla-dom.env
bash benchmarks/dynamic/dynamicvla_dom/run.sh \
  --model-gpu 0 --sim-gpu 1 --episodes 1 --seed 42 \
  --output-dir runs/dynamic/dynamicvla_dom/native-001 --dry-run

bash benchmarks/dynamic/dynamicvla_dom/run.sh \
  --model-gpu 0 --sim-gpu 1 --episodes 1 --seed 42 \
  --output-dir runs/dynamic/dynamicvla_dom/native-001

bash benchmarks/dynamic/dynamicvla_dom/run.sh \
  --model-gpu 0 --sim-gpu 1 --episodes 1 --seed 42 --streaming \
  --output-dir runs/dynamic/dynamicvla_dom/streaming-001
```

Dry-run checks local paths and configuration without starting CUDA/Isaac or creating the output directory. The real run starts separate model and simulator processes on localhost. The model loads before the simulation handshake. Use a new output directory for each run. If startup or execution fails, check the client/server logs; the launcher stops its own child processes.

`--rotation euler` specifies how to interpret the rotational action values; `rotvec` and `quat` are explicit alternatives. Use `--num-steps` to change refinement steps while keeping the checkpoint's cameras, normalization and action horizon. The checked DOM checkpoint uses two observations, 20 action steps and 10 refinement steps. Video is on by default; add `--no-record-video` to disable it. Set `--img-port` and `--act-port` when running independent sessions.

## Native execution and time

The physics timestep is 0.04 s and control decimation is 1. Camera cadence, success checks and timeout come from the DOM test configuration. The native wall pacing and model `dt_scale` compensation remain enabled. Both simulation and wall timestamps are recorded, so you can inspect how the scene progresses while inference is running.

Observations carry episode/index/instruction/clocks; actions echo their source. Terminal/ACK handshakes prevent pending results crossing a reset. Worker errors or missing actions are execution errors. Episode seeds are `seed + episode_id`; model sampling uses a continuous seeded run RNG.

Non-streaming skips its first three observations; streaming adds startup warmup. Warmup is excluded from generation counts. A completed chunk may be discarded or used only at its tail, so generation, applied chunks, sent actions and control steps are separate counts.

## Output

| File | Meaning |
| --- | --- |
| `case-manifest.json` | CLI/resource identity, source hashes, checkpoint hash, GPU preflight and completion status |
| `server.log`, `client.log` | Separate simulator/model logs |
| `effective-config.json` | Effective physics/control/camera cadence, task configuration and runtime versions |
| `engine.json` | Actual model configuration, input/output contract and worker details |
| `observations/` | First consumed raw observation of each episode and its metadata |
| `episodes.jsonl` | Native success, control steps/budget, held actions and applied chunks |
| `requests.jsonl` | Actual chunk generation, readiness and expiration events |
| `client-actions.jsonl` | Action selection/sending and source chunk identity |
| `episode-*-controls.jsonl` | Every simulator control step, applied/held command source and time |
| `episode-*.mp4` | Native pre-action camera frames; playback 24fps, distinct from25Hz control |
| `summary.json`, `summary.md`, `coverage.json` | Completed-episode statistics and coverage |

Start with `summary.md` for success rate, control steps and chunk counts, then inspect the episode and request logs for individual runs. Success rate includes all valid task episodes. Overall step statistics charge failures their declared budget; the successful mean is null when no episode succeeds. Generated chunks are counted from model events, while applied chunks count unique predictions actually used by the simulator, including partial tails.

`model_host_duration_s` measures the host-side model call. Client action selection also includes queue/cache access and pacing. For a GPU latency comparison, measure a complete policy call in a separate synchronized benchmark; use simulation timestamps for task duration.

## Source and distribution

[Model](../../../src/robotics_bench/models/dynamicvla/PROVENANCE.json) and [DOM](../../../src/robotics_bench/dynamicvla_dom/native/PROVENANCE.json) provenance retain revisions, file hashes and local compatibility changes. Derived source retains **S-Lab non-commercial terms**; Franka configuration retains the Isaac Lab BSD notice. SDK, models and assets are external resources with separate terms; see [third-party notices](../../../THIRD_PARTY_NOTICES.md).

## Validation scope

On RTX 6000 Ada, fixed-input non-streaming actions matched the local source exactly. One non-streaming and two consecutive streaming episodes completed with MP4, controls and generation/applied-chunk checks. These three episodes cover integration; evaluate a declared task/seed set for task quality or performance comparisons. Wheel source/licenses and CPU contract/lifecycle checks were also verified.

The checked simulator used Python 3.10.15, Isaac Lab distribution 0.45.9, Torch 2.7.0+cu128 and Isaac Sim 4.5. `effective-config.json` records detected versions, with unavailable metadata set to null.
