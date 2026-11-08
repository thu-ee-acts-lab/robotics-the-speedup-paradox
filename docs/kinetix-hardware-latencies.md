# KINETIX 历史硬件推理延迟

这里保存 `dynamic-experiments` 在 **2026-08-11** 测得的四档硬件延迟。可以直接查表，也可以下载[完整精度 CSV](data/kinetix-hardware-latencies-20260811.csv)，用于 KINETIX 延迟重放。

档案覆盖 RTX 6000 Ada、RTX 3090、AGX Orin 15W／30W；AGX Orin 50W 不进入本档案、汇总或默认映射。

## 实测数据

单位为 **毫秒／次完整 `policy.action` 调用**。每次生成 8 条动作，N 是 flow 采样迭代次数。表中保留三位小数；CSV 保留原始精度以及三个独立进程的平均值。

| 硬件／功耗档 | Profile ID | N=1 | N=2 | N=3 | N=4 | N=5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 6000 Ada | `local_ada` | 1.373 | 2.605 | 4.041 | 5.127 | 6.898 |
| RTX 3090 | `rtx3090` | 1.853 | 3.538 | 5.232 | 7.074 | 8.806 |
| AGX Orin 15W | `agx15` | 11.349 | 22.744 | 32.682 | 46.159 | 57.634 |
| AGX Orin 30W | `agx30` | 7.465 | 14.343 | 21.938 | 29.532 | 36.738 |

本表保存测量点；采用旧项目的线性拟合曲线时需另行标注。

## 模型与计时口径

| 项目 | 此批次设置 |
| --- | --- |
| 模型 | RTC MLP-Mixer，`teacher-1s` 架构；4 个 Mixer block，channel=256，channel hidden=512，token hidden=64 |
| 输入／输出 | 679 维观测、6 维动作、预测 horizon=8、batch=1 |
| 权重 | **同架构随机初始化权重**，用于架构和运行时性能测量 |
| 后端／精度 | PyTorch eager，FP16，`method=naive`，禁用梯度 |
| 优化开关 | 无量化、无 CUDA Graph、无 conditioning 预计算、无 native-eager runner |
| 每个 N 的重复 | 3 个独立进程；每进程预热 50 次、稳态计时 200 次 |
| 汇总 | 先分别求每进程 200 次调用的均值，再取 3 个均值的中位数 |

测量前先把观测放到 GPU。每次调用的顺序是：CUDA 同步 → 启动 `perf_counter` → 执行完整 `policy.action` → CUDA 同步后停止计时。结果包含 Python 调度和 GPU 等待，输出仍是 GPU 张量。模型初始化、预热、仿真、录像、文件输出和动作回传另行计时。

统计公式为：

```text
latency_ms = median(process_mean_0_ms, process_mean_1_ms, process_mean_2_ms)
```

CSV 保留三个进程均值，可按上式复算；600 次调用的合并均值或中位数是另一口径。

## 软件栈与来源

| Profile ID | 原记录的 PyTorch 版本 | 原记录的时钟／功耗配置 | 快照文件名 |
| --- | --- | --- | --- |
| `local_ada` | `2.9.1+cu128` | stock dynamic clocks | `local_ada-flow-step-latency-snapshot.json` |
| `rtx3090` | `2.5.0+cu124` | stock dynamic clocks | `rtx3090-flow-step-latency-snapshot.json` |
| `agx15` | `2.3.0` | `MODE_15W`、`jetson_clocks` | `agx15-flow-step-latency-snapshot.json` |
| `agx30` | `2.3.0` | `MODE_30W`、`jetson_clocks` | `agx30-flow-step-latency-snapshot.json` |

来源为 `kinetix-native-delay-clock` 工作区的 **4 份快照、60 份原始计时 JSON**，覆盖 N=1～5 及对应 `flowN-repeat0/1/2.json`。归档核对精度、batch、预热次数、开关和随机权重，并重算进程均值的中位数。

CSV 中的 `snapshot_file` 和 `snapshot_sha256` 可以定位原快照，`process_mean_*_ms` 用于复算表中数值。原始日志和 checkpoint 保存在本地。

## 使用约定

- 按“历史 PyTorch eager FP16 架构／运行时 profile”引用；它与当前训练后 JAX FP32 策略的实测分别记录。
- 软件版本随设备变化，差异表示对应软硬件组合的表现。
- 单 case 重放先筛选一个 `profile_id`，导出 `flow_steps,latency_ms`；[实操指南](hands-on.md#hardware)提供命令。
- 预测器使用完整四档输入，操作见[校准与验证流程](kinetix-latency-study.md)。部署计时需匹配目标模型、精度、后端和时间范围。
- 查询历史数据直接读取本档案；新压测另行安排。

## English summary

Four profiles from 2026-08-11 cover RTX 6000 Ada, RTX 3090 and AGX Orin 15W/30W. Each value is the median of three process means, with 50 warmups and 200 timed synchronous calls per process. The workload uses a randomly initialized RTC network, PyTorch eager FP16 and batch one. The CSV retains full precision, versions and provenance. Use these values as declared latency-replay scenarios; deployment measurements require the matching trained policy and runtime.
