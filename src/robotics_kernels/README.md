# Robotics GPU operators

这里维护项目使用的 GPU 算子，包括整数矩阵乘、逐元素融合、卷积和 CUDA Graph 辅助工具。模型通过优化适配层调用它们；只运行 CPU 工具时不会加载 GPU 依赖。首次使用建议先看[推理实操](../../docs/hands-on.md#hardware)。

源码按架构组织，架构目录内部再放对应的 CUDA/C++ 源码：

```text
robotics_kernels/
  ampere_ada/     SM80兼容的整数与BF16卷积包装、准备和调优配置
    csrc/        INT4/INT8 GEMM与BF16 CUTLASS卷积实现
    third_party/cutlass/ 固定版本的构建头文件与许可证
  blackwell/     独立FP4/FP8及其CUDA源码
  common/        跨架构Triton融合、CUDA Graph和浮点投影工具
  integer.py     兼容导入入口；实现位于ampere_ada/
  fused.py       兼容导入入口；实现位于common/
  graph.py       兼容导入入口；实现位于common/
  projections.py 兼容导入入口；实现位于common/
```

Ampere 和 Ada 使用同一套整数实现，通过 tactic 选择分块配置。已有数值和性能测试使用 SM89；在 Ampere 或 Blackwell 上运行时，需要完成对应设备的检查。

| 入口 | 格式与职责 | 验证边界 |
| --- | --- | --- |
| `fused.py` | RoPE、gated residual、GELU×up、norm affine、AdaLN modulation；保留低精度中间舍入 | RTX 6000 Ada；GPU 数值与当前 stream 测试 |
| `ampere_ada/integer.py` / `ampere_ada/csrc/integer_gemm.cu` | signed INT8/W8A8、signed INT4/W4A4；S32 Tensor Core 累加，融合 scale/bias，BF16 输出 | SM89 实机验证；Ampere 源码路径仍需目标设备验证 |
| `projections.py` | 加载时 QKV/Gate-Up 权重合并与输出切片 | BF16 合并可能改变 GEMM 舍入；不列入已验证无损配置 |
| `graph.py` | 输入刷新、输出复制、目标设备 capture stream、失败恢复和常量所有权 | 每个实例串行调用；权重在上下文内不可变 |
| `blackwell/` | 保留 FP4、tensor FP8、MXFP8、相关 mixed-input 及 CuTe 实现 | 独立来源记录；Blackwell 编译/模型性能尚待实机验证 |
| `ampere_ada/convolution.py` | 仓内CUTLASS 2D/3D卷积，8个显式tactic，权重预打包、布局处理与bias epilogue | Ada实测；近似数值，不是无损替换；不支持的模块保持native并记录 |
| `common/vae.py` | 保留原生通道归约，融合后续除法、scale、affine和可选SiLU | BF16逐值GPU检查及LIBERO/RoboCasa固定输入检查；不代表全任务集质量 |

整数 Linear 的使用顺序是 `from_linear → pack_input → forward_packed/forward`：先准备权重，再打包输入，最后调用矩阵乘。`forward_gelu` 可以把 GELU 或 gated GELU 与输入打包合并。

激活按行计算动态 float32 scale，权重按输出通道保存静态 scale。INT4 每字节的低 nibble 放前一个元素。计算时 K 补齐到 128、N 补齐到 8，补齐部分填零，返回前裁剪到原尺寸。

在 `ampere_ada/tactics.py` 选择 8 个 tactic：0/1 是原配置，2..7 提供窄 N、大 N、大 K 和不同流水深度。使用实际输入形状在目标 GPU 上比较。完整 Linear 计时包括激活准备、打包和布局转换，离线权重打包单独计时。

用 `integer_qkv` 或 `integer_gate_up` 合并同输入、同格式的投影；`integer_grouped` 同时开启两者。权重和 scale 在初始化时拼接，运行时共用一次 GEMM。

`integer_group_views` 让后续算子直接使用输出切片，支持末维连续、行间带 stride 的输入，其他布局会报错。`integer_pack_reuse` 在 INT4 量化和 nibble 打包之间复用行数据。逐项测试这些开关后，再比较完整 policy 的耗时。

Cosmos 的调制和量化融合位于 `ampere_ada/modulation.py`。它保留 BF16 中间舍入，按后续投影需要提供 INT4 或 INT8 数据，engine 对外仍返回 CPU 动作。

`norm_modulation_quant` 进一步融合 LayerNorm，`residual_norm_modulation_quant` 再融合前置门控残差。它们会改变 FP32 归约顺序，需要检查动作误差。`integer_biasless` 则针对无 bias 的 Linear；有 bias 的层仍按正常路径计算。

模型实验使用 Cosmos DiT 与π0.5 text/LLM scope，记录后端、动作偏差和任务质量；配方与回退见[Cosmos指南](../../docs/cosmos-quantization.md)。

卷积 tactic 与数值说明见[CONVOLUTION.md](ampere_ada/CONVOLUTION.md)。卷积及 attention 替换均通过独立开关启用。

模型侧的调用代码在 `robotics_bench/optimizations/`。修改算子后，依次检查算子数值、完整 policy 的动作与耗时、闭环任务结果。

INT 后端固定 CUTLASS revision `982748aa7356fa838c2ea4994ddcb0b2a4b4cefa`，头文件、许可证和逐文件 hash 位于 `ampere_ada/third_party/cutlass/`。构建使用仓内头文件及本机 CUDA 工具链：

```bash
export CUDA_HOME=/path/to/cuda
export TORCH_CUDA_ARCH_LIST=8.9
```

编译结果保存到 Torch 的 CUDA extension 缓存。Torch 管理临时 workspace；CUDA 调用使用输入所在设备的当前 stream。

Blackwell 构建入口：

```bash
python tools/build_blackwell.py --backend fp8 \
  --cutlass-root /path/to/compatible-cutlass --arch 11.0a
```

`--arch` 匹配目标 GPU、CUDA 和 CUTLASS。FP4/NVFP4/MXFP4、tensor FP8/MXFP8 与 INT4/INT8 分别使用各自编码、scale 和 packed buffer；FP 来源和条款见 `blackwell/PROVENANCE.json` 与许可证。

GPU 测试在模型环境中显式启用，公共 CPU CI 默认跳过：

```bash
ROBOTICS_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/gpu
```

完整调用计时和可视化见[推理实验说明](../../benchmarks/inference/README.md)。
