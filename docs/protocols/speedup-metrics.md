# 相对统一 baseline 的加速比

用本页选择加速比的计算方式，并把推理耗时与闭环任务结果放在一起比较。公式参考 [The Speedup Paradox，arXiv:2606.28529v2](https://arxiv.org/pdf/2606.28529v2) 的 Eq.1–5 和成功条件统计 Eq.31，统一采用 **baseline 时间 / 当前配置时间**：大于 1 表示更快，小于 1 表示变慢，baseline 为 1。

同一个 case 中的量化、采样步数、cache 和异步配置使用共同 baseline。比较结果时，分别查看推理、动作块周期和完整任务的加速比。主表展示 `speedup_chunk_estimated`、`speedup_task_success`和 SR，模型内部收益看 `speedup_inference`。完整操作见 [实验指南](../hands-on.md)。

## 1. baseline 按实验案例固定

一个比较案例使用同一基础模型、任务/初态集合、硬件、环境版本、world progression、控制周期、执行长度、计时范围和统计规则。baseline 通常选择未采用待测轻量化方案的全精度同步配置。不同硬件各自建立 baseline，不能用某张 GPU 的 FP16 时间归一化所有设备。

F-S、Q-S、F-A、Q-A 都引用同一个 baseline ID。若还要比较某两项优化之间的增量收益，另设字段报告，见 [组合实验说明](composable-experiments.md)。

## 2. 论文的周期时间与统一写法

记：

- `I0`：baseline 的一次完整 policy 推理时间，即论文的 `Tinf`。
- `Ic`：配置 c 对应的推理时间；量化配置可写作 `α I0`。
- `n`：每个完整动作块实际执行的原子动作数；预测 horizon 另记。
- `a`：一个原子动作的执行时长，即论文的 `Tact`。
- `m`：可以与下一次推理重叠的动作数，即论文的 `n′`，范围 0 到 n。

baseline 动作块周期为：

$$
C_0=I_0+na.
$$

论文量化等模型内优化对应 `Cc=α I0+na`；执行感知优化对应 `Cc=I0+na−min(I0,ma)`。[论文 pp.3–4，Eq.1–3](https://arxiv.org/pdf/2606.28529v2#page=3)

为了同时表示量化、异步和组合，采用以下统一写法：

$$
\boxed{C_c=I_c+na-\min(I_c,ma)=na+\max(0,I_c-ma)}.
$$

这是本仓合并论文两类周期模型的工程推导，假设规则重叠、固定动作长度和代表性推理成本。排队、通信与资源竞争通过测量或 trace 另行记录。

静态 `paper_async` 按论文附录 B.1 回取 `t−n′` 观测，串行评估质量，上述周期单列为 `paper_model`。π0.5 默认 n=5、`Tact=1000/30 ms`，仿真步长独立设置。缺少 `Tinf` 时周期/加速比为空，质量实验仍可运行；`eval_s`单列宿主耗时。[论文§3.1 与 B.1](https://arxiv.org/html/2606.28529v2)

| 配置 | `Ic` | `m` |
| --- | --- | --- |
| baseline | `I0` | 0 |
| 量化＋同步 | 量化变体的推理时间 | 0 |
| 原精度＋异步 | 理想情况下为`I0`；实际以测量为准 | 指定重叠深度 |
| 量化＋异步 | 组合条件下的推理时间 | 指定重叠深度 |

复杂调度使用同范围实测周期的 baseline/variant 比值，标记为 measured，与解析估计分列。

## 3. 三层加速比分别计算

### 单次推理

$$
S_{\mathrm{inf},c}=\frac{I_0}{I_c},\qquad
\alpha_c=\frac{I_c}{I_0}=\frac1{S_{\mathrm{inf},c}}.
$$

`α`为时延比例。理想纯异步保持推理加速比 1，收益来自周期与任务层；实际资源竞争可能改变 Ic，计算时使用实测值。

### 每个动作块周期

$$
\boxed{S_{\mathrm{chunk},c}=\frac{C_0}{C_c}}
=\frac{I_0+na}{I_c+na-\min(I_c,ma)}.
$$

`ηchunk`提供量化与异步的共同速度坐标。残余等待 `max(0,Ic−ma)`与完整推理时延分别记录；隐藏等待时推理计算仍存在。

### 整项任务

论文用动作块数量与周期分解任务时间：

$$
\widehat T_{\mathrm{task},c}=N_c C_c,
\qquad
\boxed{\widehat S_{\mathrm{task},c}
=\frac{N_0}{N_c}S_{\mathrm{chunk},c}}.
$$

帽号表示稳态模型估计，Nc 取自对应配置的任务统计。更多动作块可能抵消周期收益，形成论文中的 speedup paradox。[论文 p.4，Eq.4–5](https://arxiv.org/pdf/2606.28529v2#page=4)

## 4. 成功条件均值与直接任务计时

为对应论文附录 D.1，任务时间与 N 的默认统计范围是成功试验：[论文 p.23，Eq.31](https://arxiv.org/pdf/2606.28529v2#page=23)。直接任务加速比定义为：

$$
S_{\mathrm{task,succ},c}
=\frac{\operatorname{mean}_{e\in\mathcal S_0} T_{0,e}}
       {\operatorname{mean}_{e\in\mathcal S_c} T_{c,e}}.
$$

先按每个配置的成功样本计算均值，再求比值。它不同于“逐试验加速比的平均”：例如 baseline 耗时 `[1,9]`，候选 `[1,3]`，均值之比为 2.5，而逐试验比值平均为 2。

同时报告 trials、successes、SR 及其百分点差值。任一侧零成功时，成功条件加速比为 null 并标原因；成功集合不同时可补充共同成功 seed 配对结果，保留全体 SR。

实际任务时间应按声明的起点到 terminal 记录，包含首轮、末轮和提前终止。即使理想条件下完整执行 N 个等长 chunk，首轮没有旧动作可重叠时，也可能出现：

$$
T_{\mathrm{finite},c}=I_c+na+(N-1)C_c
=NC_c+\min(I_c,ma).
$$

该有限回合推导单列，calculator 默认仍使用稳态模型。抖动、变长 chunk 或复杂队列的任务时间从完整 trace 统计。

## 5. 四组手算例子

设 baseline 推理 100 ms，量化后 50 ms；每块执行 10 个动作，每动作 20 ms；异步重叠 3 个动作，即 60 ms。以下均为合成演示数据。

| 配置 | 推理/ms | 周期估计/ms | 推理加速比 | 周期加速比 |
| --- | ---: | ---: | ---: | ---: |
| FP16＋同步 | 100 | 300 | 1.00× | 1.00× |
| 量化＋同步 | 50 | 250 | 2.00× | 1.20× |
| FP16＋异步 | 100 | 240 | 1.00× | 1.25× |
| 量化＋异步 | 50 | 200 | 2.00× | 1.50× |

若 baseline 平均需要 10 个 chunk、组合方案平均需要 16 个 chunk，那么任务时间估计为 3000 与 3200 ms，组合任务加速比仅为 `3000/3200=0.9375×`。周期变快与整项任务变快因此要分开报告。

本例的两个单项周期加速比乘积恰好等于组合值，不能据此一般化。若量化后改为 60 ms，组合周期仍为 200 ms，但单项乘积为 `(300/260)×(300/240)≈1.4423`，并不等于组合的 1.5。

## 6. 计算工具

在仓库根目录：

```bash
python tools/compare_speedups.py --input examples/speedups/synthetic.json
python tools/compare_speedups.py --input examples/speedups/synthetic.json --markdown
```

输入采用 [examples/speedups](../../examples/speedups)中的独立摘要格式，与 run-manifest v1 分开。CPU 工具使用调用方提供的时延与 N，由试验数/成功数计算 SR，再输出加速比。

共同输入声明 case、时间范围、时钟域、n 和动作时长；每行提供推理时间、overlap、试验数/成功数，可选提供成功条件 N 和任务时间均值。baseline 由 ID 显式指定，必须是同步配置。

| 输出字段 | 口径 |
| --- | --- |
| `speedup_inference` | baseline推理时延 / 当前推理时延 |
| `speedup_chunk_estimated` | 解析周期C0/Cc |
| `speedup_task_estimated` | 成功条件N与解析周期形成的比值 |
| `speedup_task_success` | 输入提供的成功条件任务时间均值之比 |
| `latency_ratio_alpha` | 当前推理时间 / baseline推理时间 |

缺少 N 或任务时间均值时，对应指标输出 null 和原因，已经能计算的推理/周期结果仍会保留。`paper_steady_state`保存公式估计，`provided_summary`保存调用方提供的任务均值。

调用方用 `case_id`、计时元数据及原始证据声明硬件/初态/实验可比性。工具校验输入并计算比值；合成示例标记 `synthetic=true`。

## 7. 汇总与可视化规则

标量工具计算固定/代表性时延的周期估计。profile 含 jitter 时先逐样本计算再汇总，或直接使用 trace：推理 20/100 ms 各半、overlap60 ms 时，平均残余等待 20 ms；代入平均推理 60 ms 则得到 0。

周期变化时通常有 `mean(N*C) != mean(N)*mean(C)`，优先报告直接测得的成功任务时间均值。时延比较使用相同范围和统计量。

静态图用周期加速比对照成功条件任务时间/加速比，同时展示 SR。动态图报告 SR 或任务 score、环境速度与观测年龄；score、二项成功率和时间加速比使用各自名称与统计方法。

预算惩罚与成功条件时间分列。`compare_speedups.py`计算给定时间摘要的比值；原始 episode 聚合、置信区间与控制步数由相应统计工具处理。

## 8. 控制步数统计与实验汇总

[summarize_experiment.py](../../tools/summarize_experiment.py) 汇总 `episodes.jsonl` 的控制步数。记录包含 `task`、`init_state_id`、布尔 `success`、非负整数 `primitive_steps`及可选 `env_seed`；重复任务/初态/seed 组合会被拒绝。控制步数与耗时、模型调用及前述 chunk 数 N 分别统计。

以下是本工具的工程统计定义。设本次运行有 N 个 episode、S 个成功 episode，成功集合为 $\mathcal S$，失败集合为 $\mathcal F$；$b_i$ 为实际控制步数，$B_i$ 为该 episode 声明的最大步数预算。

| 统计范围 | 均值 | 含义 |
| --- | --- | --- |
| 全体实际（JSON复核项） | $\sum_i b_i/N$ | 成功和失败均使用实际步数，不在终端、Markdown或CSV中单列 |
| 仅成功 | $\sum_{i\in\mathcal S} b_i/S$ | 仅在成功样本内归一化；S=0时为null |
| 仅失败实际（JSON复核项） | $\sum_{i\in\mathcal F} b_i/(N-S)$ | 失败样本的实际步数；无失败时为null，不单列到终端、Markdown或CSV展示 |
| 失败按预算惩罚 | $(\sum_{i\in\mathcal S}b_i+\sum_{i\in\mathcal F}B_i)/N$ | 失败实际步数被对应预算替换 |
| 成功加权分量（JSON复核项） | $\sum_{i\in\mathcal S} b_i/N$ | 等于SR乘成功条件均值；不在终端、Markdown或CSV中单列，不能标成“仅成功均值” |

报告展示样本数、成功数、SR，以及总体预算惩罚和成功回合的总步数/均值。零成功时成功总步数为 0、均值为 null；实际步数与加权分量保留在 JSON 用于复核。π0.5＋LIBERO 失败运行至预算，其他环境即使提前失败，也按其声明预算计入总体代价。

跨任务汇总时，总体预算均值按 $n_t/N$ 加权，成功均值按 $s_t/S$ 加权；$n_t$、$s_t$分别为任务 t 的样本数和成功数。无成功任务的成功总和为 0、均值为 null。多个 run 分别汇总。

预算按任务覆盖或统一提供，优先读取 `coverage.json` 的 `max_primitive_steps`；旧结果需显式指定。失败样本缺少预算时惩罚总数/均值为 null，并列出任务，实际统计保留。全成功样本无需失败预算；声明预算须覆盖该任务全部已记录实际步数。

从仓库根目录运行：

```bash
# 新运行：读取目录中的ledger与完成/预算元数据，展开任务明细
python tools/summarize_experiment.py --input runs/static/pi05_libero/sync-001 --per-task

# 旧LIBERO object结果：显式声明280步预算；两次运行分别汇总
python tools/summarize_experiment.py \
  --input runs/static/pi05_libero/sync-001 runs/static/pi05_libero/paper-async-001 \
  --max-steps 280 --per-task --output results/episode-comparison.csv

# 独立ledger：任务名须与记录一致；可重复传入任务预算
python tools/summarize_experiment.py --input PATH/TO/episodes.jsonl \
  --max-steps 280 --task-max-steps TASK_NAME=320 --format json
```

`--input` 接受一个或多个运行目录或 JSONL 文件。默认向 stdout 输出 Markdown；`--format markdown|json|csv` 显式选择格式，未指定时可由 `--output` 的扩展名推断。输出文件必须是新文件。JSON 始终包含各任务统计；Markdown 和 CSV 使用 `--per-task` 展开任务行。缺失值在 JSON 中为 null，Markdown 显示 N/A。

CPU 工具会读取相邻的 manifest 和 coverage，核对运行状态、样本数和任务覆盖。汇总未完成或失败的 run 时加 `--allow-partial`，结果标为 `partial`；只有 ledger 或缺少完成记录时标为 `unverified`。
