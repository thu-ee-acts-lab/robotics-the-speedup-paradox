# Native DynamicVLA + DOM

Single-environment Franka DOM evaluation with repository-owned DynamicVLA model,
preprocessing, streaming queue and DOM scene/control source. Isaac Sim/Isaac Lab,
PyTorch/LeRobot/Transformers, checkpoints and assets remain external dependencies.
Execution uses the model and DOM source packaged in this repository.

Choose `native-non-streaming` (default) or `native-streaming` (`--streaming`).
Both run beside a continuously evolving simulator. Non-streaming retains the
previous control target while generating the next chunk. Streaming generates
in a worker, skips expired predicted actions and merges indexed action queues.
This entry uses native model precision and the DOM timing protocol.

## Environments and resources

Use separate model and simulator environments. The upstream setup describes
Python 3.10, PyTorch 2.7.1, Isaac Sim 4.5.0 and Isaac Lab 2.2.1. The local model
environment used during integration is Python 3.11 with PyTorch 2.7.1 and
Transformers 5.2.0. On a new machine, install the dependencies below and verify
resource loading and a single episode before a larger evaluation.

- Model dependencies: [requirements-model.txt](requirements-model.txt).
- Simulator additions: [requirements-simulator.txt](requirements-simulator.txt);
  install Isaac through its supported distribution first.
- Checkpoint: local `config.json` and `model.safetensors` from the DOM policy.
- Hugging Face cache: tokenizer and backbone configuration referenced by the
  checkpoint must already exist. Execution sets `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1`; it never downloads missing resources.
- DOM test JSON, referenced USD scenes/textures and object assets are external.
- Franka: an explicit local `panda_instanceable.usd` with its referenced USD and
  material files. No automatic Nucleus robot asset fallback.

Copy [paths.env.example](paths.env.example) to an ignored local file, fill paths,
and source it. `ROBOTICS_DOM_PYTHON` accepts Isaac's `python.sh`, an appropriate
Python executable, or `isaaclab.sh` (the runner adds `-p`). Interpreter symlinks
are preserved so virtual environments keep their identity. An unrelated parent
Conda prefix is removed for Isaac shell launchers.

## Launch

From the repository root:

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

The dry run validates explicit local paths/configuration and prints commands; it
neither starts CUDA/Isaac nor creates the output directory. Full asset dependency
and environment compatibility are verified by real execution. Normal runs require
clean idle selected GPUs, bind communication to localhost, preserve existing
outputs and terminate their own child processes on errors. Use new output paths
for retries. The model is loaded before the simulation handshake.

`--rotation euler` is explicit by default: seven action coordinates do not identify
whether their three rotational values are Euler angles or rotation vectors.
`--num-steps` optionally overrides checkpoint refinement steps; chunk length,
normalization and camera/state contract remain checkpoint-defined. The local DOM
checkpoint uses two observations, 20 action steps and 10 refinement steps.
Video recording defaults on; use `--no-record-video` to disable it. Ports can be
changed using `--img-port` and `--act-port` for simultaneous independent runs.

## Native execution and time

The initial adapter fixes native physics dt to 0.04 s and control decimation to1.
Camera periods and the native task success/timeout terms come from the DOM test
configuration. Waiting for model results never pauses the scene. Simulator wall
pacing and model `dt_scale` compensation remain enabled; their actual values are
recorded alongside separate simulation and wall clocks.

Each observation includes its episode, index, instruction and simulation/wall
clocks. Actions echo their source identity. Terminal/ACK messages are retained;
a reset cannot consume an earlier episode's pending inference. Worker errors or
missing actions are execution errors, not policy failures. Episode seeds are
`seed + episode_id`; model sampling uses a seeded continuous per-run RNG.

As upstream, non-streaming skips its first three observations before inference;
streaming additionally performs a worker startup warmup. Warmup chunks are excluded
from episode generation counts. Native streaming may discard a completed chunk
or execute only its tail: this is why control steps, sent actions and generated
chunks are counted separately.

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

Success rate includes successful and failed task episodes; overall control-step
statistics penalize failures at the declared budget. Mean generated chunks is
conditioned on success and undefined at zero successes. Counts come from model
generation events. Applied chunks count unique generated chunks actually used
by the simulator, including tails.

`model_host_duration_s` records the host-side model call span without explicit
CUDA synchronization. Client action selection includes cache pops, queue access
and pacing. Use these fields to inspect execution; complete-policy GPU latency
requires a synchronized benchmark, and task duration uses the simulation clock.

## Source and distribution

[Model provenance](../../../src/robotics_bench/models/dynamicvla/PROVENANCE.json)
and [DOM provenance](../../../src/robotics_bench/dynamicvla_dom/native/PROVENANCE.json)
record the selected local DynamicVLA revision, original file hashes and migration
changes, including the local compatibility fixes in this source snapshot.
Derived files retain S-Lab's non-commercial license; the Franka configuration
retains its Isaac Lab BSD notice.
See [third-party notices](../../../THIRD_PARTY_NOTICES.md). Model and asset terms
remain separate and their files are not distributed in this repository.

## Validation scope

The integration was checked on RTX 6000 Ada GPUs with the local DOM checkpoint:
fixed-input non-streaming output matched the local source implementation exactly;
one native non-streaming episode and two consecutive native streaming episodes
completed with MP4, control ledgers and generation/applied-chunk cross-checks.
This evidence covers fixed-input equivalence and three integration episodes.
Task-suite quality and performance comparisons require their own declared task
and seed sets. Built wheels were checked for owned source and retained licenses;
public CI runs CPU contract/lifecycle tests with local synthetic fixtures.

The validated local simulator uses Python 3.10.15, Isaac Lab distribution version
0.45.9 and Torch 2.7.0+cu128 with the installed Isaac Sim4.5 runtime. This differs
from the upstream recommended environment; `effective-config.json` records the
actually detected versions, leaving unavailable distribution metadata null.
