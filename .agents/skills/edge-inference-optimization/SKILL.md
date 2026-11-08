---
name: edge-inference-optimization
description: Use when optimizing edge policy, VLM, or streaming inference, especially when quantized GEMM speedups are lost to activation preparation, fusion gaps, memory copies, cache handling, or runtime overhead.
---

# Edge Inference Optimization

按用户指定的模型、设备与任务优化完整推理和闭环指标。GEMM占比用于诊断，目标是降低完整调用代价。

## 建立一个可比较的案例

固定baseline commit、checkpoint、输入/seed、warmup、计时范围和质量门槛。记录形状、精度、实际backend、软件栈、设备功耗/频率及host/GPU/sim时间域。分阶段记录prefill/decode或vision/prefix/flow/action-head。

先检查共享GPU占用，性能测量使用隔离设备；受干扰数据单列，不用于性能结论。GPU计时需等待实际计算完成。

## 找到真正执行的热点

同时测kernel、完整推理和任务层。检查矩阵计算、attention、norm、激活量化、布局/拷贝、cache、分配和host dispatch；并发stream按关键路径分析。

为主要热点写明“证据 → 修改 → 预期收益 → 验证/停止条件”，优先处理证据支持的高收益路径。

## 按执行路径选择优化

- **量化：** 记录fake/真实backend、位宽、scale/group/zero-point、累加dtype、校准及pack格式；fallback写明原因和占比。可复用的pack/concat在加载时完成，区分tile内unpack与完整权重反量化。
- **融合：** 检查QKV/gate-up共享准备、norm+quant、bias/residual与布局转换。先评估编译器或现有后端，再手写kernel；核对register pressure与端到端收益。
- **缓存与图：** 在shape和生命周期允许时复用workspace、I/O buffer及KV；并发请求隔离可写buffer，图捕获失败或shape不支持时提供明确fallback。

按workload检查：自回归分别测prefill与q_len=1 decode，以及GEMV/KV append/reset；diffusion/flow核对vision/prefix复用与denoise mask/position/cache等价性，减少step按有损方案评估；Jetson核对架构、统一内存、CPU线程、功耗和热状态，在目标设备选择kernel。

<a id="保留两种质量证据"></a>

## 质量检查

| 层级 | 证据 |
| --- | --- |
| 算子 | 参考误差、非对齐shape、dtype/layout、NaN/Inf、累加精度 |
| 模型 | 固定observation/noise/seed的action、必要hidden state和多步误差 |
| 闭环 | 相同episode seeds下的SR、任务时间、失败预算与成功条件时间 |

数值等价融合与量化、剪枝、减步数分别记录。LLM首token/exact match用于生成任务；连续动作模型使用动作与闭环指标。

## 完成和迭代

在同设备、同范围完整入口复核收益。达到质量门槛且收益超出噪声后保留优化；否则回退或保留为独立实验路径。交付实际backend、前后数据、质量证据、复现命令和适用条件。公共规则来自可重复证据，单模型技巧留在对应case。
