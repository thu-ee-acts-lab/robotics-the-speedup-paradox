# Cosmos quantization and VAE optimization

Use this guide to run Cosmos with W8A8, W4A4 or mixed INT8/INT4 layers, together with VAE optimizations for LIBERO and RoboCasa. Start with the [static hands-on pilot](hands-on.en.md#static), then use the recipe below to choose switches. The default is unoptimized BF16.

<a id="supported-boundary"></a>

## Model, format and execution configuration

| Item | Configuration | Application conditions |
| --- | --- | --- |
| Models | Cosmos-Policy-LIBERO-Predict2-2B and Cosmos-Policy-RoboCasa-Predict2-2B | Each case requires its own checkpoint, statistics and text cache |
| Integer scope | 280 Linear sites in 28 DiT blocks: self/cross-attention Q/K/V/out and both MLP projections | VAE and sensitive noncandidate operations retain native precision |
| Formats | W8A8 and W4A4, symmetric signed integers; per-output-channel weight scales and dynamic per-row activation scales | No calibration, rotation, smoothing or fine-tuning is applied |
| GEMM | Integer Tensor Core accumulation in INT32; fused scale/bias conversion to BF16 | Activation preparation and layout costs remain part of complete-call timing |
| Hardware | RTX 6000 Ada, SM89; PyTorch 2.7.0+cu128 and CUDA 12.8 development toolkit | Ampere requires its own device validation; preserved Blackwell FP4/FP8 source is not an SM89 result |
| Execution | One serialized model instance, CPU observation to CPU action chunk | No concurrent requests, planning or future-image decoding in the optimized action boundary |

Integer kernels and pinned CUTLASS headers are packaged here; [kernel setup](../src/robotics_kernels/README.md) and [source notices](../THIRD_PARTY_NOTICES.md) describe build/provenance. Loaders, VAE/attention libraries, simulators and assets are external dependencies.

## Starting recipe for performance experiments

The recipe keeps native DiT normalization and attention. Check a fusion against the unchanged model at the same precision, and compare quantization error against BF16. Then use paired rollouts to measure the effect on task success.

| Switch | Purpose |
| --- | --- |
| `modulation`, `gated_residual`, `cuda_graph` | Shared pointwise fusion and DiT graph replay |
| `shared_quant` | Reuse prepared activations across compatible projection consumers |
| `activation_quant_fusion` | Fuse the MLP activation with integer preparation |
| `integer_pack_reuse` | Reuse row data during INT4 quantization and nibble packing |
| `modulation_quant` | Fuse BF16 modulation with integer preparation, keeping native LayerNorm |
| `vae_norm_fusion`, `vae_silu_fusion` | Keep native VAE channel reduction and fuse subsequent pointwise work |
| `vae_condition_prefix` | Skip complete encoder chunks outside the required conditioning prefix |

Start with tactic `1` on the measured Ada setup (`TORCH_CUDA_ARCH_LIST=8.9`); benchmark tactics on the target device.

Activate the case environment from the [setup guide](environment_setup.en.md). Capture inputs with [capture_libero.py](../benchmarks/inference/capture_libero.py) or [capture_robocasa.py](../benchmarks/inference/capture_robocasa.py), using the actual reset instruction saved in its manifest.

```bash
python benchmarks/inference/run.py \
  --case cosmos_libero --gpu 0 \
  --input /path/to/observation.npz --task 'the matching captured instruction' \
  --steps 5 --seed 195 --warmup 5 --repeats 13 \
  --precision int4 --quant-scope dit --integer-tactic 1 \
  --enable modulation --enable gated_residual --enable cuda_graph \
  --enable shared_quant --enable activation_quant_fusion \
  --enable integer_pack_reuse --enable modulation_quant \
  --enable vae_norm_fusion --enable vae_silu_fusion --enable vae_condition_prefix \
  --output-dir runs/optimization/cosmos-libero/int4-001 --profile
```

Use `--precision int8` for W8A8. RoboCasa requires a separate process, `--case cosmos_robocasa`, its resources and three-camera observation. The case selects H16 or H32.

The command measures three configurations: original BF16, BF16 with the shared optimizations, and the integer candidate. To compare VAE switches on/off in the same loaded model, use `--variants` and `--ablate-shared`; the benchmark adds a BF16 reference for each shared configuration. More input, seed and rendering options are in the [benchmark guide](../benchmarks/inference/README.md).

Case launchers accept the same runtime, precision, scope, tactic and `--enable` options. Keep `--quant none`; `--precision` selects arithmetic. Rollout commands are in the [LIBERO](../benchmarks/static/cosmos_libero/README.md) and [RoboCasa](../benchmarks/static/cosmos_robocasa/README.md) guides.

## Shared VAE implementation

[cosmos_prefix.py](../src/robotics_bench/optimizations/cosmos_prefix.py) shares the encoder gate and request lifecycle across these contracts:

| Case | Raw / latent frames | Conditioning frames | Encoder frames executed | Action horizon |
| --- | --- | --- | --- | --- |
| LIBERO | 33 / 9 | 4 | 17, with the native 16-frame window | 16 |
| RoboCasa | 41 / 11 | 5 | 17, with the native 16-frame window | 32 |

The adapter preserves raw input, full latent projection, normalization, noise/DiT dimensions, RNG and per-request cache clearing; each denoise checks the conditioning mask. Both cases also share the [pointwise adapter](../src/robotics_bench/optimizations/cosmos_pointwise.py) and [kernel](../src/robotics_kernels/common/vae.py).

Prefix cropping serves action output and omits the auxiliary `orig_clean_latent_frames` suffix. Leave the prefix switch off for planning/future-image consumers. Direct VAE calls outside engine action requests retain native encoding; rollout video is independent.

## Measured snapshot

These measurements used the Ada setup above on 2026-09-22, with 5 sampling steps, 5 warmups and 13 timed repetitions. The table reports the median complete-policy latency in milliseconds. Loading, packing, compilation and warmup are excluded; GPU kernel sums are recorded separately for diagnosis.

| Case / cohort | Original BF16 | Shared optimized BF16 | W8A8 | W4A4 |
| --- | ---: | ---: | ---: | ---: |
| LIBERO / `round8-shared-vae` | 436.52 | 381.94 | 214.36 | 167.12 |
| RoboCasa / `round12-precision-final` | 495.08 | 411.99 | 246.94 | 186.98 |

All optimized columns include the VAE recipe. LIBERO's VAE bundle reduced INT4 latency from 222.88 to 167.12 ms (1.334x); reversed-order repeat: 222.38 vs 167.29 ms. Original BF16→INT4 was 2.612x. Convolution-family GPU work fell 46.85→24.33 ms; other elementwise work fell 34.84→8.78 ms. The incremental measurement isolates the bundle, not its individual switches.

RoboCasa `round11-shared-vae` measured prefix-off/on INT4 at 253.28/188.57 ms with pointwise fusion on both sides. It rechecks the shared adapter's existing prefix optimization. Scenario: TurnOffMicrowave, layout/style 1/1; instruction: "press the stop button on the microwave".

Validation used seeds 0/42/195: one LIBERO observation and three RoboCasa observations. LIBERO round8/RoboCasa round11 VAE-off/on actions matched exactly in all 3/9 same-precision checks. RoboCasa round12 confirmed all 280 integer sites: INT8 was 2.005x faster than original BF16 and 1.668x faster than shared-optimized BF16. Across 9 checks, maximum differences from original BF16 were 0.418804 (INT8) and 1.107105 (INT4); optimized BF16 matched exactly. Round12 has no isolated INT8 VAE-off/on comparison. LIBERO used "put both the alphabet soup and the tomato sauce in the basket". These records cover fixed-input timing/actions; paired case runs supply task success rates.

Each output directory retains source/input hashes, switches, backend coverage and action comparisons. Raw traces, inputs and generated figures remain outside Git; the cohort names above identify their evidence.

## Experimental alternatives and rollback

| Option | Current conclusion |
| --- | --- |
| `norm_modulation_quant`, `residual_norm_modulation_quant` | Small measured gains in some cohorts; altered LayerNorm reduction order changed actions. Explicit numerical candidates, excluded from the starting recipe |
| `cross_kv_cache` | Correct fixed-input results, but request refresh/check/copy overhead outweighed projection savings in the measured RoboCasa cohort |
| Alternative attention backends | No measured improvement over the native backend in the audited RoboCasa workload; some changed numerical results |
| `conv_cutlass_*`, `vae_spatial_padding` | Owned experiments retained; microkernel savings did not establish a complete-policy advantage over the selected native convolution path |
| Projection grouping / additional tactics | Optional shape-dependent candidates; not automatically enabled by the precision choice |

To disable an optimization, remove its switch. For original owned BF16, use `--precision bf16` and omit `--enable`; add `--model-runtime native` to use the external reference. After changing weights or precision, recreate the optimization context before making further serialized calls.

[Progressive quantization](protocols/progressive-quantization.md) defines tiers 0..10 and the paired pilot's explicit recipe. [Speedup metrics](protocols/speedup-metrics.md) covers task/cycle estimates, failure budgets and null successful-only values. Retain checkpoint, input, sampling and hardware identity for each result; PI0.5 uses its separate text/LLM scope.

Run full paired commands from the [hands-on guide](hands-on.en.md#static).
