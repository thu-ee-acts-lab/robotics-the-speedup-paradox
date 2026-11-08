# 渐进式整数精度协议

渐进量化让你逐档增加 Cosmos 中使用 INT4 的层，而不是一次替换全部层。本页列出
[论文](https://arxiv.org/abs/2606.28529)附录 B.1、Table5 的层选择规则，以及每档的计时和任务评估方法。
本仓使用 BF16 基线独立测量收益；π0.5 的 text/LLM 量化使用自己的 scope。

在 Cosmos 命令中添加 `--precision int8 --quant-scope dit --quant-tier N`，即可选择 T0–T10 档位，
逐步将候选 Linear 从 W8A8 替换为 W4A4。每档记录替换了哪些层、完整推理耗时和闭环任务结果。
固定精度与共享 VAE 优化的运行配方见[Cosmos 量化指南](../cosmos-quantization.md)。

## 固定的格式与候选集合

候选为 28 个 DiT block 中的 280 个 Linear：self-attention 和 cross-attention 的 Q/K/V/out，
以及 MLP 的 layer1/layer2。early 为 B0–B9，mid 为 B10–B19，late 为 B20–B27。
程序会在执行前核对块数、模块名和初始 BF16 dtype；任何一项不匹配都会报错。

W8A8 和 W4A4 分别使用 signed 整数权重与激活、逐输出通道权重 scale、逐行动态激活 scale、
INT32 累加和 BF16 输出。没有额外 smoothing、rotation 或微调。FP32 敏感模块、VAE 和
其他非候选模块保留原始精度。整数源码、打包和固定 CUTLASS 头文件均在本仓维护。

| 档位 | 本档新增W4A4位置；之前的位置保留 | W4A4 / 候选Linear |
| --- | --- | ---: |
| `w8a8` / tier0 | 无，全部候选使用W8A8 | 0 / 280 |
| `w4-t1` | ca.out，late | 8 / 280 |
| `w4-t2` | mlp.layer1，early | 18 / 280 |
| `w4-t3` | ca.out，early | 28 / 280 |
| `w4-t4` | mlp.layer1，late | 36 / 280 |
| `w4-t5` | mlp.layer1，mid | 46 / 280 |
| `w4-t6` | ca.out，mid | 56 / 280 |
| `w4-t7` | mlp.layer2，all | 84 / 280 |
| `w4-t8` | sa.q/k/out，all | 168 / 280 |
| `w4-t9` | sa.v、ca.q，all | 224 / 280 |
| `w4-t10` | ca.k/v，all | 280 / 280 |

实际运行记录每个模块命中的 backend；同一个 QKV 组出现不同位宽时分别准备输入，
合并投影限定于同格式、同输入子组；INT8 与 INT4 使用各自的 packed activation。

## 完整推理调用对照

从无优化 BF16、共享优化相同的 BF16 开始，再测量化档位。固定 checkpoint、输入、
seed、采样步数、设备和计时边界；加载、编译、打包权重、CUDA Graph 捕获与 warmup 单列。
每行相对本轮实测 baseline 计算；每种配置至少 3 次 warmup，重复计时取中位数。
有限动作的偏差随性能结果报告；非有限动作或执行错误按基础设施异常处理。

在已有 Cosmos 环境和资源变量下运行，`--input`来自本仓观测捕获工具或明确声明的 fixture：

```bash
python benchmarks/inference/run.py --case cosmos_libero \
  --input /path/to/observation.npz --task 'the matching task instruction' \
  --gpu 0 --steps 5 --integer-tactic 1 --progressive-sweep \
  --enable modulation --enable gated_residual --enable cuda_graph \
  --output-dir runs/optimization/cosmos/tier-sweep-001 \
  --profile --profile-skill /path/to/profiler-visualizer \
  --render-python /path/to/render-env/bin/python
```

`--progressive-tiers 0 1 3 7 10`可选择子集；单次 case 启动使用
`--precision int8 --quant-scope dit --quant-tier N`。位宽越低不保证时延单调下降。
Table5 采用 LIBERO 五步推理计时，附录静态 Cosmos 设置为一步。比较时统一采样步数：
本仓默认五步，一步实验的全部对照统一传 `--steps 1`。

## 闭环与任务时间

小规模的同步 LIBERO 对照入口同时支持固定精度和选定档位：

```bash
# 固定BF16 / 优化BF16 / INT8 / INT4
python benchmarks/inference/evaluate_cosmos.py --gpu 0 \
  --suite libero_object --task-id 0 --initial-states 0 1 --steps 5 \
  --output-dir runs/optimization/cosmos/pilot-fixed-001

# 在相同初态上比较渐进档位
python benchmarks/inference/evaluate_cosmos.py --gpu 0 \
  --suite libero_object --task-id 0 --initial-states 0 1 --steps 5 \
  --tiers 0 1 3 7 10 --output-dir runs/optimization/cosmos/pilot-tiers-001
```

该 pilot 评估命令指定的任务与初态子集。每档先 warmup 与计时，再重置相同初态，
统计从策略开始到成功或预算耗尽的控制步数。reset/settling 和 warmup 不计入任务时长。
各档独立保存 episode ledger 并分别汇总。

输出分别包含：成功率、全体失败按预算计入的步数、仅成功步数、实测 policy 延迟、
仿真器宿主墙钟任务时间，以及论文模型下的任务时间估计。同步的有限回合估计为：

$$\widehat T_e=\sum_{r\in e} I_r+b_eT_{act}.$$

其中`I_r`是该回合逐请求实测的完整推理时间，`b_e`为实际执行动作数，默认声明
`Tact=50ms`。该有限回合估计包含末尾不完整动作块，与宿主耗时和模拟器 physics dt 分别记录。

成功条件任务加速比按各档成功样本的均值之比计算；任一侧没有成功样本时为 null。
总体预算惩罚时间另列：保留已发生的推理和动作，仅对提前终止后缺失的动作/请求按
声明预算与本档代表推理时延补齐；已跑满预算的失败回合不替换其已测耗时。
完整定义见 [统一加速比口径](speedup-metrics.md)。

保留原始采样、逐层精度、动作对照、profile 及任务统计；profile-visualizer 从实测 ledger 绘图，
产物置于忽略的 `runs/`。零成功配置的成功条件加速比为 null。完整操作见 [hands-on 指南](../hands-on.md)。
