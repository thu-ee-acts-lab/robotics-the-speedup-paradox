# Inference optimization experiments

Use [run.py](run.py) to time a complete policy call, from a prepared CPU observation to a CPU action chunk. Loading, weight packing, compilation, graph capture and warmup are measured separately. The case runners record simulator and control time. Output labels are in English.

Start with the [hands-on timing example](../../docs/hands-on.en.md#hardware) or [Cosmos recipe](../../docs/cosmos-quantization.md). This page defines input, switch and measurement interfaces.

The model forward implementations live in `src/robotics_bench/models/`: PaliGemma/Gemma/SigLIP for PI0.5, and policy/sampler/DiT for Cosmos. Install their loader, framework and VAE/attention dependencies in the model environment. The source directories include their original hashes and licenses.

Activate the matching model environment. Copy `benchmarks/static/<case>/paths.env.example` to `.local/`, fill in your paths, and source that file. Prepare missing weights with `tools/prepare_resources.py`; the benchmark loads local resources.

```bash
python benchmarks/inference/run.py \
  --case pi05_libero \
  --input /path/to/prepared-observation.npz \
  --task 'pick up the alphabet soup and place it in the basket' \
  --gpu 0 --output-dir runs/optimization/pi05/trial-001 \
  --enable flow_loop --enable mask_cache --enable rope \
  --enable gated_residual --enable gelu_mul --enable norm --enable cuda_graph \
  --repeats 10 --profile \
  --profile-skill /path/to/profiler-visualizer \
  --render-python /path/to/render-env/bin/python
```

For PI0.5, supply an NPZ with prepared LeRobot CPU observation keys. For Cosmos, use the raw image/proprio keys of the selected case. Add `--validation-input` and `--validation-seed` to compare more observations and random seeds. Use the task instruction saved with the input.

Each run first measures original BF16. Integer runs also measure BF16 with the same shared optimizations, so you can separate the benefit of quantization from the other switches:

```bash
export CUDA_HOME=/path/to/compatible-cuda
# Add to the PI0.5 command above:
# --precision int8 --quant-scope text --enable shared_quant --enable activation_quant_fusion
```

Integer formats use signed INT8/packed INT4, dynamic per-row activation scale, per-output-channel weight scale, S32 accumulation and BF16 output. Preparation, packing and conversion are timed. Tactics 0–7 select CUTLASS tiles through `--integer-tactic`; headers come from `src/robotics_kernels/ampere_ada/third_party/cutlass/`. FP4/FP8 have separate encodings and sources under `blackwell/`, with `tools/build_blackwell.py` for target-device validation. Record action differences for every precision.

Cosmos uses `--case cosmos_libero` or `cosmos_robocasa`, matching resource variables and `--quant-scope dit`. Shared switches include `modulation`, `gated_residual`, `cuda_graph`; integer preparation adds `shared_quant`/`activation_quant_fusion`.

`norm_modulation_quant` requires `modulation`, `modulation_quant`, INT4/INT8 and `dit`. It fuses non-affine LayerNorm, AdaLN and packing where all consumers are integer projections; other boundaries retain native normalization. BF16 intermediate rounding is retained, but FP32 reduction order can change actions. The shared LIBERO/RoboCasa adapter records same-precision drift. Remove this default-off switch to restore native LayerNorm.

`residual_norm_modulation_quant` requires `norm_modulation_quant` and `gated_residual`. It adds the preceding residual at cross-attention/MLP input boundaries, writes BF16 for the skip path and normalizes inside the kernel. Mixed-format/unquantized boundaries and the final MLP residual stay separate. It shares the normalization rounding tradeoff and defaults off.

Add the following to an existing Cosmos INT4/INT8 command that already enables
`modulation` and `gated_residual` to evaluate the complete fusion:

```bash
--enable modulation_quant --enable norm_modulation_quant --enable residual_norm_modulation_quant
```

Each output includes a manifest, repeated timings, numerical checks, checkpoint audit and `ledger.json`. `--profile` adds separate traces/tables; `--profile-skill` renders timestamped PNG/SVG/Markdown, HTML and a manifest under `figures/`. CairoSVG/Cairo belongs in the rendering environment.

Report same-precision equality, quantization drift and paired task success separately. Compute speedups directly from the same cohort's BF16 anchor.

You can apply a selected configuration to a rollout with the case launcher's `--model-runtime owned`, `--enable`, `--precision`, `--quant-scope` and `--integer-tactic` options. Switches default off; PI0.5 `cuda_graph` also needs `flow_loop`.

Use one model instance for serialized calls and keep its weights fixed while a graph or cache is active. Reinstall the optimization context after changing weights. Select `native` for the reference runtime; it rejects local optimization switches.

Capture several real initial observations using the matching case environment:

```bash
python benchmarks/inference/capture_libero.py \
  --case pi05_libero --suite libero_object --task-id 0 \
  --initial-states 0,1,2 --gpu 0 \
  --output-dir runs/optimization/inputs/pi05-object
```

Use `observation-000.npz` as `--input`, others as repeatable `--validation-input`. Capture records actual task text and initial states. It runs the simulator; timing replays only the saved observations.

PI0.5 `empty_image_cache` requires `cuda_graph` and caches fixed missing-camera placeholder encodings. Real images are re-encoded. Camera-role changes invalidate the graph; embedding storage lasts as long as the graph.

## Integer fusion and tuning

The implementation is under `src/robotics_kernels/ampere_ada/`; portable fusion
and graph helpers are under `common/`, while Blackwell keeps its own sources.

Add independent switches to the complete policy benchmark or case launcher:

```bash
# INT8 or INT4 uses actual integer Tensor Core GEMM:
# --precision int4 --integer-tactic 1 --quant-scope text
# --enable shared_quant --enable activation_quant_fusion
# --enable integer_pack_reuse
# Optional ablations:
# --enable integer_qkv
# --enable integer_gate_up       # PI0.5 gated MLP only
# --enable integer_group_views   # requires projection grouping
```

`integer_grouped` enables both QKV and Gate-Up grouping, partitioned by compatible integer format. `integer_group_views` removes explicit slice copies; consumers determine further layout work. Buffers belong to one serialized model call.

To compare several configurations with one loaded model, pass `--variants FILE`. The JSON list accepts `id`, `precision`, `switches`, `scopes` and `tactic`; omitted values come from the CLI. Give each entry a unique filename-safe ID. Shared switches must match the CLI unless you add `--ablate-shared`; quantization switches may differ. The output includes action arrays and comparisons with the first row at the same precision. Timing continues for finite outputs, with the observed action differences recorded.

Measure individual shapes separately from complete policy latency:

```bash
python benchmarks/inference/bench_integer.py --case both --gpu 0 \
  --pack-reuse --output-dir runs/optimization/integer/round-001
python benchmarks/inference/render_integer.py \
  --input runs/optimization/integer/round-001 \
  --skill /path/to/profiler-visualizer \
  --render-python /path/to/render-env/bin/python
```

Shapes come from recorded PI0.5 text/Cosmos DiT calls and projection concatenations. `prepared_gemm` excludes activation preparation; `complete_linear` includes it. Both use CUDA events around repeated graph nodes. JSON retains every tactic/sample; CSV selects the fastest observed complete Linear per shape/precision. Apply a tactic explicitly and validate it in the complete policy on that GPU.

Charts separate matrix, reduction, copy/cast, normalization, quantization and pointwise work. GPU sums and synchronized policy medians remain distinct; more than four profiles are paginated with the original anchor repeated.

`integer_biasless` explicitly selects a specialized no-bias epilogue; biased modules retain their normal epilogue.

PI0.5 `condition_projection_cache` requires `condition_cache` and `flow_loop`. It caches projections of registered immutable timestep embeddings, while observation-dependent states are recomputed. Keep weights/timestep settings fixed or reinstall the context; graphs retain their projection buffers.

`--ablate-shared` permits shared-switch changes and inserts matched BF16 measurements for each distinct set.

[Progressive quantization](../../docs/protocols/progressive-quantization.md) defines Cosmos `--progressive-sweep`, `--quant-tier` and the matched `evaluate_cosmos.py` pilot, using this repository's measured BF16 reference.

Use Cosmos DiT or PI0.5 `--quant-scope text` for the documented recipes; action-expert/diffusion scope measurements remain exploratory.

Cosmos `modulation_quant` requires integer precision and `modulation`. It keeps native LayerNorm, fuses BF16 modulation with dynamic quantization and prepares only the formats each mixed-tier consumer needs. Packed data stays inside the model; CPU action output is unchanged. Check same-format actions and complete-policy latency.

To inspect the actual per-layer formats recorded by a completed tier sweep:

```bash
# Uses the optional Matplotlib CPU environment.
python benchmarks/inference/render_progressive.py \
  --input runs/optimization/cosmos/tier-sweep-001
```

## RoboCasa and remaining Cosmos hotspots

RoboCasa uses its own Python environment, `ROBOTICS_COSMOS_ROBOCASA_*` resources, three cameras and H=32; LIBERO uses H=16.

Capture a real reset observation and read its actual instruction from the
generated manifest before starting replay:

```bash
python benchmarks/inference/capture_robocasa.py \
  --task TurnOffMicrowave --layout-id 1 --style-id 1 --env-seeds 0 1 2 \
  --gpu 0 --output-dir runs/optimization/inputs/robocasa-001

python benchmarks/inference/run.py --case cosmos_robocasa \
  --input runs/optimization/inputs/robocasa-001/observation-000.npz \
  --task 'press the stop button on the microwave' --gpu 0 --steps 5 \
  --precision int4 --quant-scope dit --integer-tactic 1 \
  --enable modulation --enable gated_residual --enable cuda_graph \
  --enable shared_quant --enable activation_quant_fusion --enable integer_pack_reuse \
  --output-dir runs/optimization/robocasa/int4-001 --profile
```

Capture selects RoboCasa EGL with a physical numeric GPU index and saves post-settling images, actual language, scene state and fingerprints. Use that language for replay, whose timing excludes simulation.

These shared switches default off; compare quantized variants with matched BF16 controls:

| Switch | Scope and boundary |
| --- | --- |
| `vae_norm_fusion` | Wan VAE native channel-norm reduction, followed by fused division, scale, affine and bias. BF16 rounding boundaries are retained. |
| `vae_silu_fusion` | Requires `vae_norm_fusion`; also fuses adjacent SiLU using a table generated by the installed Torch implementation. |
| `vae_spatial_padding` | Encoder-only experiment inspired by worldmodel/Wan: retain explicit causal time padding, move symmetric spatial padding into the convolution descriptor, and skip zero-padding copies. No cache survives between requests. |
| `vae_condition_prefix` | Shared LIBERO/RoboCasa action-only path: execute only complete causal encoder chunks needed by each case's conditioning prefix. Retain the full raw input, final latent projection, normalization and diffusion/noise shapes. Auxiliary clean-latent suffixes are not preserved. |
| `cross_kv_cache` | Cache native text K/V projections and norms within each policy request. Refresh fixed-address buffers outside the DiT graph; retain their storage in the graph and invalidate on layout changes. Query-dependent attention still runs. |
| `attention_sdpa_flash` / `attention_sdpa_efficient` / `attention_sdpa_math` / `attention_flash_attn` | Select exactly one dense DiT attention backend. QKV, RoPE and output projections remain unchanged; unsupported kernels raise rather than falling back. VAE attention keeps its native backend. |
| `conv_cutlass_c96_0` ... `conv_cutlass_c96_7` | Explicit VAE 96-to-96, 3x3x3, stride-one family ablation on channels-last 3D inputs. Other modules/layouts remain native with recorded reasons. `conv_cutlass_c96` is an alias for tactic 0. |
| `conv_cutlass_0` ... `conv_cutlass_7` | Experimental broader dispatch for supported BF16 Conv2d/3d modules. These are explicit tactics, not automatic fastest-kernel selection. |

CUTLASS preserves causal padding/cache handling; complete-call timing includes activation/output layout conversion, while weight packing is initialization work. Convolution/attention reduction changes can affect actions, so compare both output and full-policy latency.

Set `--cudnn-benchmark` before loading the model in a fresh process; changing it later preserves cached plans. The manifest records benchmark/benchmark_limit, deterministic and TF32 settings. Keep each cuDNN mode in its own cohort.

[Convolution notes](../../src/robotics_kernels/ampere_ada/CONVOLUTION.md) explain the pinned iterator, layout, tiles and pipeline/resource tradeoffs.

```bash
python benchmarks/inference/bench_convolution.py \
  --trace runs/optimization/robocasa/int4-001/candidate-trace.json \
  --gpu 0 --limit 12 --warmup-ms 500 \
  --output-dir runs/optimization/robocasa/conv-tactics-001

# CPU-only selected breakdown; use names actually present in the cohort.
python benchmarks/inference/render_breakdown.py \
  --input runs/optimization/robocasa/int4-001 \
  --profiles original optimized-bf16 candidate \
  --skill /path/to/profiler-visualizer \
  --render-python /path/to/render-env/bin/python
```

The PNG adapter converts the skill's CSS HSL colors to equivalent RGB for
CairoSVG compatibility. The original SVG and raw durations are retained.

Integer GEMM fuses scale, bias and BF16 conversion in its epilogue. Activation quantization/packing still writes GPU buffers. QKV grouping is an independent switch; CUDA Graph replays kernels rather than fusing GEMM, residual and row-quantization boundaries.

`src/robotics_bench/optimizations/cosmos_vae_memory.py` owns spatial padding; its method reference is worldmodel revision `51529dc48047c61c17bd25fc088c8569c005ddd1`, `wan/modules/vae2_1.py`. The context changes symmetric spatial padding before convolution planning and restores it/methods on exit; asymmetric downsample padding stays native. Check descriptor-induced rounding and complete-policy latency.

Cosmos action inference encodes VAE inputs with future-image decoding disabled. Preserve its temporal window, tail handling and request cache lifecycle. Decoder batching, upsample folding, FP32 encoder kernels and SM110-specific C96 epilogues require separate applicability checks.

`vae_condition_prefix` uses one adapter and encoder gate for the supported cases:

| Case | Raw / latent frames | Fixed conditioning frames | Action horizon | Camera keys |
| --- | --- | --- | --- | --- |
| LIBERO | 33 / 9 | 4 | 16 | `primary_image`, `wrist_image` |
| RoboCasa | 41 / 11 | 5 | 32 | `primary_image`, `secondary_image`, `wrist_image` |

With the 16-frame encoder window, both cases execute the first 1+16 frames. LIBERO skips 16 trailing frames; RoboCasa skips 16+8. Finite feature tensors fill omitted chunks before the unchanged full projection (9/11 latent frames). The model geometry selects the case; normalization/SiLU share an implementation. Enable independently with `--enable vae_norm_fusion --enable vae_silu_fusion --enable vae_condition_prefix`. Native RNG seeding/cache clearing remain, and every denoise validates the unconditioned suffix. This action-only path omits `orig_clean_latent_frames`; consumers of that suffix must leave `vae_condition_prefix` off. Direct VAE calls outside engine action requests retain native encoding.

`cross_kv_cache` prepares K/V once per request, even when the prompt repeats. Queries and attention outputs are recomputed. If the context changes during a request, the call fails; tensors without version counters are checked against a saved copy. Complete-policy timing includes these checks, copies and refreshes.

Use the cache for serialized inference with identity text projection and fixed weights/precision. Image cross-attention, training and context parallelism are unsupported. Give concurrent requests separate model instances.
