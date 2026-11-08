# Hands-on experiments: from inference speed to task outcomes

[中文](hands-on.md) · [Environment setup](environment_setup.en.md) · [§4.5 predictor walkthrough](kinetix-latency-study.en.md) · [Project](../README.md)

This guide turns the project's existing experiment workflows into a sequence you can run: prepare one case, compare matched configurations, inspect their records, then scale the study. Start with the experiment you need; the rows below are independent routes.

## 1. Choose the question

The section numbers refer to [arXiv v3](https://arxiv.org/html/2606.28529v3), where §4.5 is **Predictive TISED** and §4.6 contains real-world experiments.

| Question | Paper connection | Runnable route |
| --- | --- | --- |
| Does lower precision reduce task completion time? | Static tasks, §4.2 | Cosmos quantization pilot, Section 3.1 |
| What changes when observations arrive from earlier control steps? | Execution-aware optimization, §4.2 | π0.5 `paper_async`, Section 3.2 |
| How do quality and reaction delay interact? | Dynamic tasks, §4.3 | KINETIX sampling-step × latency matrix, Section 4.1 |
| How does hardware change the preferred configuration? | Cross-hardware study, §4.4 | Matched inference timing and latency replay, Section 5 |
| Can calibration predict the preferred sampling step before validation? | Predictive TISED, §4.5 | [Dedicated step-by-step case study](kinetix-latency-study.en.md) |
| How are pruning and motion quality inspected? | Appendix B.1 / D.2 | LingBot CPU contracts and trajectory analysis, Sections 3.4 / 6 |
| How are physical task records analyzed? | Real-world study, §4.6 | Controller-exported episode/trajectory data, Section 6 |

The examples use task subsets to make the workflow readable. Larger studies preserve the same checkpoints, task/initial-state lists, seeds, action horizons and runtime settings across configurations.

## 2. Prepare a working session

Complete the relevant track in [Environment setup](environment_setup.en.md): CPU/analysis tools for all routes, Cosmos + LIBERO for Section 3.1, π0.5 + LIBERO for Section 3.2, or KINETIX for Sections 4–5 and the predictor. Model and asset downloads are preparation steps; evaluation loads local resources.

Run commands from the repository root in Bash. Choose an idle GPU and a fresh output root:

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

The timestamp is created once per session. Keep the same `ROBOTICS_RUN` while following a route. Individual launchers create their child directories; use a new root when changing configuration or repeating a completed run.

First check the CPU tools:

```bash
"$ROBOTICS_CPU_PYTHON" tools/validate_contracts.py --examples
"$ROBOTICS_CPU_PYTHON" tools/compare_speedups.py \
  --input examples/speedups/synthetic.json --markdown
```

The examples print a valid contract and a speedup table labeled `synthetic=true`. This checks installation and metric interpretation. Real measurements come from the following runs.

<a id="static"></a>

## 3. Static tasks

### 3.1 Cosmos: progressive precision, inference timing and task completion

**Goal:** compare original BF16, optimized BF16 and selected INT8→INT4 tiers on identical initial states. The pilot performs timing and rollouts with the same recipe in one process.

**Step 1 — bind resources and preflight.** The environment guide creates `.local/cosmos-libero.env`. Verify the source, checkpoint, statistics, T5, VAE and LIBERO configuration:

```bash
source .local/cosmos-libero.env
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 \
  --seed 195 --env-seed 0 --schedule sync \
  --output-dir "$ROBOTICS_RUN/cosmos-preflight" --dry-run
```

Expected checkpoint: the command prints a validated plan and creates no run directory. The next command loads the model and runs the simulator.

**Step 2 — run the paired pilot.** This evaluates five configurations on two initial states: 10 episodes in total. T0 uses W8A8 at all 280 candidate Linear sites; T5 uses W4A4 at 46 sites; T10 uses W4A4 at all 280. All use five sampling steps and 16 executed actions per chunk.

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

The pilot compares `original`, `optimized-bf16`, `w8a8`, `w4-t5` and `w4-t10`. Its shared recipe is modulation, gated-residual fusion and CUDA Graph; integer candidates also use shared activation preparation, activation fusion and pack reuse. Keep this recipe attached to its timing and task results.

**Step 3 — read the result.** `summary.md` shows success counts, inference milliseconds, inference speedup, overall failure-budget steps, successful-episode steps and successful task speedup under the paper time model. `runs.json` and each configuration's `episodes.jsonl` retain the original records. The task estimate sums measured request times plus executed actions × the declared 50 ms action time; simulator host time has its own field.

Read successful-episode chunk counts directly from the inference events:

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

A configuration with zero successes has a null successful mean. Compare success rate beside speedup: a faster inference call can require more chunks or fewer episodes may succeed. Inspect the measured trade-off, rather than expecting a particular curve.

**Step 4 — expand.** Add initial-state IDs to `--initial-states`, choose another `--task-id`, or include more `--tiers` from 0–10 in a new output directory. Use the [RoboCasa entry](../benchmarks/static/cosmos_robocasa/README.md) for kitchen tasks; that case uses H=32, its own resources and `--reference-run` for matched initialization. The pilot above is the LIBERO entry.

### 3.2 π0.5: synchronous baseline and paper_async

**Goal:** vary observation age while retaining the model, action execution length and sampling steps. With five actions executed per chunk, n′=0,2,5 are valid settings. At n′=0 the example selects the explicit synchronous baseline.

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

This runs 30 episodes across three configurations. The evaluator distributes each ten-episode budget across the suite; `coverage.json` records the actual task and initial-state coverage. To run the existing full `libero_object` setting, use `--episodes 500 --batch-size 10` for each configuration.

Check `paper-async.json` for n′ and history rules, `coverage.json` for matched initial states, and `episode-summary.md` for success and step counts. Videos contain the complete executed episode. `paper_async` uses historical image/state snapshots in a serial evaluation loop; its cycle estimate uses a separately supplied inference profile. Report the measured quality and steps first, then use [speedup metrics](protocols/speedup-metrics.md) when matched `Tinf` is available.

### 3.3 Combine quantization and asynchronous observation

Cosmos accepts both controls in one command. This separate paired study holds T5 quantization fixed and compares synchronous execution with n′=2:

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

The pair uses plain T5 without extra fusion switches. Its timing recipe is therefore separate from the fused pilot in Section 3.1. Section 5.1 shows a matching fixed-input measurement. For a full four-way comparison, also run BF16 with n′=0 and n′=2, keeping every other setting fixed and naming each output separately.

### 3.4 LingBot: pruning contracts and evaluator reports

The merged LingBot contributions provide a CPU KV read-window selector and a report scorer. Run their bundled examples first:

```bash
"$ROBOTICS_CPU_PYTHON" benchmarks/static/lingbot_robotwin/kv_read_window.py \
  --input benchmarks/static/lingbot_robotwin/kv_read_window.synthetic.json
"$ROBOTICS_CPU_PYTHON" benchmarks/static/lingbot_robotwin/score_reports.py \
  --input benchmarks/static/lingbot_robotwin/scoring.synthetic.json \
  --output "$ROBOTICS_RUN/lingbot-scoring-example"
```

The selector reports retained/dropped slots and independent visual/action budgets. Its `base`, `p1`–`p4` presets are defined in [KV_READ_WINDOW.md](../benchmarks/static/lingbot_robotwin/KV_READ_WINDOW.md); current/unknown slots are retained. The scorer consumes evaluator `*_analysis.json` reports and produces per-episode plus aggregate tables. Its example input is synthetic.

For model execution, follow the [LingBot + RoboTwin guide](../benchmarks/static/lingbot_robotwin/README.md): use its `sync` / `paper_async` launcher and complete KV/VAE observation-history rules. Read-window pruning is currently an explicit CPU contract; the guide lists the worker integration and GPU validation steps. Keep that contract separate from already executable control schedules.

<a id="dynamic"></a>

## 4. Dynamic tasks

### 4.1 KINETIX: separate quality from delay

**Goal:** observe success under a controlled sampling-step and delay grid. Use three sampling counts × three delays with the same 16 seeds: 144 episodes.

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

Inspect `summary.md/.json/.csv`, `episodes.jsonl` and `requests.jsonl`. Hold N fixed to compare delay responses; hold latency fixed to compare sampling quality. The `fine` mode mixes actuator commands at the boundary slot on the native physics grid. The model emits eight actions and refreshes after four; default control period is 33.333 ms and budget is 256 commands. The requested delay is in milliseconds, not a sleep duration to add to host runtime.

For mapping sensitivity, repeat with `--mapping coarse` in a new directory. Coarse rounds to control-step bins; report requested and effective latency together. Preserve physics parameters, noise settings and seed lists in both modes.

For §4.5, go to the [predictor case study](kinetix-latency-study.en.md). It prepares a complete zero-delay N=1…5 baseline, calibrates fixed-policy delay response, freezes predictions, validates hardware-mapped conditions and explains each report field.

### 4.2 DynamicVLA + DOM: native streaming

DOM exercises a continuously advancing moving-object scene with separate model and Isaac processes. Configure `.local/dynamicvla-dom.env` with its checkpoint, scenes, objects, test JSON and Franka USD, then compare the two native schedules:

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

Read `server.log` / `client.log` for startup and execution, `effective-config.json` for timing, `episodes.jsonl` for outcomes, and `requests.jsonl` / `episode-*-controls.jsonl` for chunk generation and consumption. MP4 recording defaults on. Streaming may execute a chunk tail or discard an expired prediction, so generated chunks, applied chunks and control steps are distinct counts.

To study the sampling-step trade-off, hold streaming enabled and repeat with another `--num-steps` value, the same scene/seeds and a new directory. Use KINETIX for the independently controlled latency grid in Section 4.1. See the [DOM guide](../benchmarks/dynamic/dynamicvla_dom/README.md) for complete runtime settings.

<a id="hardware"></a>

## 5. Hardware timing and predictive configuration selection

### 5.1 Measure a complete policy call

For the plain T5 configuration in Section 3.3, capture a real initial observation, read its actual task instruction and measure in the Cosmos model environment:

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

`measurements.json` contains the original BF16 anchor, the integer candidate, repeated samples, median latency and actual quantization coverage. Loading, compilation, weight preparation and warmup are recorded separately. For another GPU, repeat with the same checkpoint, input, sampling steps, precision, tactic and switches in a new directory; record the target device in the manifest.

This saved-input measurement provides inference latency. Use the paired rollout records for success and chunk counts. The [metric tool](../tools/compare_speedups.py) computes baseline/variant ratios from explicit inputs; its [input format](../examples/speedups/README.md) specifies required fields and null statistics.

### 5.2 Replay a hardware latency profile in KINETIX

The [hardware archive](kinetix-hardware-latencies.md) contains four historical profiles. Select one profile and export the two columns accepted by the single-case runner:

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

`agx30` replays its recorded delay on the chosen host GPU. The archive describes historical Torch FP16 calls with random-initialized weights; the runner uses the trained JAX FP32 policy. Interpret the output as a declared hardware-latency scenario. For a measured deployment comparison, collect a profile with the deployed policy/backend and retain its provenance.

The §4.5 study takes a different, four-profile CSV format and validates all N=1…5. Follow the [predictor walkthrough](kinetix-latency-study.en.md) to fit the quality/delay factors, inspect frozen recommended N values, and compare them with later observations. It also explains why calibration-overlap cells are excluded from held-out prediction error.

<a id="analysis"></a>

## 6. Task records, trajectories and physical experiments

Use three complementary records: outcome per episode, inference/chunk events, and measured robot trajectories. For physical experiments (§4.6), the robot controller supplies start/success/timeout events, executed commands and measured end-effector positions. This repository's shared tools analyze those exports; robot actuation remains in the platform controller.

Before analyzing robot data, run the trajectory example:

```bash
"$ROBOTICS_CPU_PYTHON" tools/embodied/trajectory_metrics.py \
  --input tools/embodied/examples/smooth.csv --frame synthetic \
  --include-series --output "$ROBOTICS_RUN/trajectory-example.json"
"$KINETIX_ANALYSIS_PYTHON" tools/embodied/plot_trajectory.py \
  --input tools/embodied/examples/smooth.csv tools/embodied/examples/perturbed.csv \
  --labels smooth perturbed --frame synthetic \
  --output "$ROBOTICS_RUN/trajectory-example.png"
```

Replace the synthetic CSVs with one measured trajectory per episode, containing `t,x,y,z` in seconds and meters (or declare units explicitly). Use a fixed coordinate frame and matched sampling/filtering for comparisons. Velocity uses the actual adjacent timestamp differences. Acceleration and jerk require uniformly sampled positions; otherwise the tool emits null values with `missing_reasons`. Successful-only task time uses start-to-terminal timestamps, while failures contribute their declared budget to overall step summaries.

For static case outputs, use `tools/summarize_experiment.py --input RUN --per-task`. For LingBot evaluator reports use `score_reports.py`; its normalized action-stability score has a different definition from Cartesian SI-unit jerk. The [trajectory guide](../tools/embodied/README.md) describes columns, units, missing values and plots.

## 7. Locate results and continue an experiment

| Route | First file to inspect | Detail / visualization |
| --- | --- | --- |
| Cosmos quantization pilot | `summary.md` | `runs.json`, per-configuration `episodes.jsonl` |
| Static case launchers | `episode-summary.md` | `run.log` where provided, `coverage.json`, optional videos |
| KINETIX grid | `summary.md` | `requests.jsonl`, `episodes.jsonl`, `videos/` |
| DOM | `summary.md` | client/server logs, control events, MP4 |
| Predictive TISED | `calibration/report.md`, then `analysis/report.md` | frozen model, predictions, validation CSV and four-panel plots |

A plain case run uses a fresh output directory. The predictor has its own `status`, phase-resume and `finish` commands; follow its recovery section rather than recreating an existing plan. Keep raw files and manifests with any exported result, and give every variant its own directory. Outputs stay under ignored `runs/`; the commands and protocols remain versioned.
