# 统一加速比：CPU 手算工具

从仓库根目录运行，无模型、GPU 或第三方运行依赖：

```bash
python tools/compare_speedups.py --input examples/speedups/synthetic.json
python tools/compare_speedups.py --input examples/speedups/synthetic.json --markdown
```

默认stdout输出JSON，可显式加 `--json`；`--markdown` 输出表格。成功退出0，错误写stderr并退出2。样例均为synthetic，工具独立于manifest/trace v1。

## 一份输入代表一个可比较案例

每份case由输入作者声明同场景、硬件、控制设置和统计范围，工具校验字段并计算。用 `case_id` 关联实验条件，`timing_scope` 和 `clock_domain` 指定计时范围/时间域；样本与成功seed一致性在原始记录中核对。

`inference_time_ms` 为完整推理范围的代表性时延标量。抖动跨越overlap阈值时，`min(mean(I), L)` 与 `mean(min(I, L))` 可不同，输出按标量周期估计报告。

| 字段 | 约束 |
| --- | --- |
| `format_version` | 字符串 `1.0` |
| `synthetic` | 必需 boolean；合成数据必须为 true |
| `case_id`, `timing_scope`, `clock_domain`, `baseline_id` | 非空字符串 |
| `n_actions` | 每周期执行动作数，正整数；所有行相同 |
| `action_time_ms` | 每个动作的固定执行时间，正有限数 |
| `runs` | 非空列表，每行 ID 唯一；baseline 必须存在且 overlap 为 0 |
| 每行 `id`, `inference_time_ms`, `overlap_actions` | ID 非空；推理时间正有限；overlap 为 0 到 n 的整数 |
| 每行 `trials`, `successes` | trials 正整数，successes 为 0 到 trials 的整数 |
| 每行 `successful_chunk_count_mean` | 可缺省/null；否则为成功样本的正有限 chunk 均值 |
| 每行 `successful_task_time_mean_ms` | 可缺省/null；否则为提供的成功任务时间正有限均值 |

bool 不作为数字接受；NaN/Infinity、零分母、非法 overlap、重复 ID 均拒绝。successes 为 0 时两个成功条件均值只能缺省或 null。计算中超出有限浮点表示范围也报错，不输出 0/Infinity 伪比值。

## 四类加速比分开报告

令 baseline 推理时间为 I0，第 i 行为 Ii，动作数为 n，每动作时间为 A，overlap 动作数为 n′。

```text
C0 = I0 + n*A
Ci = Ii + n*A - min(Ii, n′*A)
speedup_inference       = I0 / Ii
speedup_chunk_estimated = C0 / Ci
task_time_estimated_i  = Ni * Ci
speedup_task_estimated = (N0*C0) / (Ni*Ci)
speedup_task_success   = T0_success / Ti_success
latency_ratio_alpha   = Ii / I0
```

`estimate_model=paper_steady_state` 使用规则、稳态、等长动作块。组合周期是串行/重叠公式的工程推广；首尾pipeline、提前终止、有限回合与调度抖动需由实际trace分析。

`paper_steady_state` 保存周期、隐藏/剩余推理时间及N×C估计；`provided_summary` 保存提供的成功均值。两种task指标分别是chunk模型估计与成功任务时间的均值之比；逐seed比值平均、含失败效率另行定义。

任一侧没有成功样本或缺少所需均值，对应 task ratio 为 null，并在 `missing_reasons` 写明原因。不从缺失任务时间反推数据。SR 和相对 baseline 的百分点变化始终并报；即使两行 SR 相同，成功子集仍可能不同。

## 样例的独立手算结果

| 配置 | I/ms | n′ | 周期估计/ms | 成功 chunk 均值 N | N×C/ms | 任务估计加速比 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 100 | 0 | 300 | 10 | 3000 | 1 |
| quantization | 50 | 0 | 250 | 12 | 3000 | 1 |
| async | 100 | 3 | 240 | 12 | 2880 | 1.041667 |
| combined | 50 | 3 | 200 | 16 | 3200 | 0.9375 |

组合方案的推理快 2 倍、周期估计快 1.5 倍，但 N 增加后任务估计变慢。样例没有提供成功任务时间，因此 `speedup_task_success` 为 null，不能把 N×C 填进这个字段。

也不能一般性地相乘独立优化收益：若把量化推理时间改为 60 ms，独立周期加速比的乘积为 `(300/260)*(300/240)≈1.442308`，组合周期却仍为 200 ms，对应 1.5 倍。

验证命令：`python -m pytest -q tests/test_speedups.py`，覆盖独立手算、输入负例和CLI。
