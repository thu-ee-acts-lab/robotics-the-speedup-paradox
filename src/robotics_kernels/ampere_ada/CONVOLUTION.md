# CUTLASS convolution candidates

The convolution path has three parts: [convolution.py](convolution.py) prepares weights and layouts, [convolution_fprop.cu](csrc/convolution_fprop.cu) dispatches CUDA kernels, and [the VAE context](../../robotics_bench/optimizations/cosmos_convolution.py) installs and restores selected modules.

The causal wrapper keeps cache concatenation and asymmetric temporal padding. `forward_packed` takes NDHWC activations and KTRSC filters. Complete calls include layout copies and return the input memory format. The epilogue adds bias to FP32 accumulators before converting the output to BF16.

This is an approximate replacement: [PyTorch cuDNN dispatch](https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/native/Convolution.cpp) adds bias after BF16 output rounding, whereas this epilogue adds it before the final conversion. The fused bias pass and reduction order can change outputs. Operator tests use explicit tolerances; policy benchmarks record action drift.

Headers are the unmodified NVIDIA/CUTLASS revision `982748aa7356fa838c2ea4994ddcb0b2a4b4cefa`, vendored with [BSD-3-Clause and file-specific notices](third_party/cutlass/LICENSE.txt). [PROVENANCE.json](third_party/cutlass/PROVENANCE.json) records revision and hashes; build/runtime use these local sources.

## What the implementation takes from CUTLASS

Implicit GEMM constructs activation tiles during loads rather than materializing
an `im2col` tensor. Here the logical GEMM dimensions are `M=batch*Z*P*Q`,
`N=output_channels`, and `K=input_channels*T*R*S`.
See the [official convolution guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html).

The pinned [optimized activation iterator](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/conv/threadblock/conv3d_fprop_activation_tile_access_iterator_optimized.h)
uses initial predicates and fast-divmod indexing. Its
[parameter object](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/conv/threadblock/conv3d_params.h)
precomputes pointer increments for the next S, R, T, and C position.
The [multistage mainloop](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/conv/threadblock/implicit_gemm_multistage.h)
pipelines asynchronous global-to-shared copies with Tensor Core work.
The [kernel composition](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/conv/kernel/default_conv3d_fprop.h)
exposes threadblock shape, warp shape, and pipeline depth independently.

## Explicit tactics

Select a tactic ID explicitly when benchmarking. The `c96` VAE policy applies it to 96-to-96, 3x3x3, stride-one modules with channels-last 3D input. Other modules keep their native implementation, and the report explains why each one was skipped. Generic kernels also support other aligned channel counts and validated strides.

| ID | Threadblock M,N,K | Warp M,N,K | Stages | Warps/CTA | A+B staging estimate, KiB | Accumulator FP32 words/thread |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | 128,128,32 | 64,64,32 | 3 | 4 | 48 | 128 |
| 1 | 128,64,32 | 64,32,32 | 3 | 4 | 36 | 64 |
| 2 | 64,64,64 | 32,32,64 | 3 | 4 | 48 | 32 |
| 3 | 256,64,32 | 64,32,32 | 3 | 8 | 60 | 64 |
| 4 | 128,32,32 | 64,32,32 | 3 | 2 | 30 | 64 |
| 5 | 128,32,64 | 64,32,64 | 3 | 2 | 60 | 64 |
| 6 | 64,32,64 | 32,32,64 | 3 | 2 | 36 | 32 |
| 7 | 128,32,32 | 64,32,32 | 4 | 2 | 40 | 64 |

Staging is estimated as `2*(TileM+TileN)*TileK*Stages` bytes, before layout padding and epilogue storage. Accumulator words exclude pointers, predicates, operand fragments and temporaries. Use compiled resource usage for occupancy; larger tiles/stage counts trade reuse and latency hiding against CTA residency.

At `input_channels=output_channels=96`, N=32 avoids the output-channel tail; N=64/128 allocates 128 columns. Narrow N also increases CTAs and activation reloads. The pinned [3D loop count](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/conv/conv3d_problem_size.h) gives `27*ceil(96/TileK)` iterations: 81 for K=32 and 54 for K=64. K=64 masks 32 of 128 channels. Tactics 4/5 compare loop/padding cost, 5/6 compare M/accumulator size, and 4/7 compare pipeline depth. Measure these tradeoffs on the target shapes.

K=128 failed compilation: the pinned BF16 `DefaultMmaCore` sets shared-memory crosswise extent to TileK, while its [layout](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/layout/tensor_op_multiplicand_sm75.h) allows a 128-byte line, or 64 BF16 values. Supporting K=128 requires another layout/iterator composition.

## Measurement boundary

Run [bench_convolution.py](../../../benchmarks/inference/bench_convolution.py) on a real trace with explicit `--tactics 0 1 2 3 4 5 6 7`. It preserves strides, records unsupported coverage and numerical error, and separates device time, synchronized eager wall time and one-time weight packing. Warmup defaults to 500 ms per scope; raw samples and unlocked GPU clocks/power are retained. Check temporal drift and repeat comparisons before selecting a policy tactic.

To compare cuDNN default with `--cudnn-benchmark`, start a fresh process for each mode and set the flag before the first convolution. Restart after changing compiled tactics as well: the loader rejects a stale extension. Once you select a kernel, compare complete-policy actions and latency, then run the closed-loop task. See the [hands-on timing workflow](../../../docs/hands-on.en.md#hardware).
