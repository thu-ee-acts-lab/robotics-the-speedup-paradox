# KINETIX：原生时间下的 latency–quality 实验

[环境配置](../../../docs/environment_setup.md) · [质量/延迟实操](../../../docs/hands-on.md#dynamic) · [§4.5预测器](../../../docs/kinetix-latency-study.md)

这个入口用来分别改变策略的采样步数和动作到达延迟，观察任务成功率如何变化。RTC flow policy、KINETIX/Jax2D、12个关卡和渲染素材都由仓内维护。模型使用 JAX/Flax，读取 symbolic 观测，输出6维 motor/thruster 动作；每次预测8条，默认执行4条后再推理。

## 环境与资源

已有运行使用 Python 3.11、JAX 0.4.35、jaxlib 0.4.34、Flax 0.10.2、NumPy 1.26.4 和 CUDA 12插件。按 [requirements-runtime.txt](requirements-runtime.txt) 安装对应依赖：

```bash
python3.11 -m venv .local/envs/kinetix
.local/envs/kinetix/bin/python -m pip install -r benchmarks/dynamic/kinetix/requirements-runtime.txt
```

为每个关卡准备对应的 RTC checkpoint，例如 `worlds_l_car_launch.pkl` 或 `worlds_l_mjc_walker.pkl`，来源是 [RTC项目](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) 的 `gs://rtc-assets/bc/`。入口只加载指定的可信本地 pickle，评估不需要训练轨迹。填写 [路径模板](paths.env.example) 后再加载：

```bash
mkdir -p .local
cp -n benchmarks/dynamic/kinetix/paths.env.example .local/kinetix.env
# Fill in the two existing local paths, then source the file.
source .local/kinetix.env
```

## 启动与矩阵评估

从仓库根目录预检。它检查仓内关卡/checkpoint身份，实际运行再导入JAX并创建目录：

```bash
bash benchmarks/dynamic/kinetix/run.sh \
  --levels car_launch,mjc_walker --flow-steps 1,3,5 \
  --latencies-ms 0,7.465,21.938 --episodes 2 --start-seed 0 \
  --output-dir runs/dynamic/kinetix/matrix-001 --dry-run
```

选择空闲GPU并运行；初次接入或修改物理实现时启用原生对齐：

```bash
bash benchmarks/dynamic/kinetix/run.sh \
  --levels car_launch,mjc_walker --flow-steps 1,5 \
  --latencies-ms 0,21.938 --episodes 1 --start-seed 0 \
  --gpu 3 --validate-native --record-video \
  --output-dir runs/dynamic/kinetix/matrix-001
```

`--episodes` 指每个“关卡 × flow步数 × 延迟”组合的回合数，上面的实际运行命令共8回合。所有组合使用相同 seed 列表，环境、动作噪声和 policy 各有独立随机流。`--levels all` 运行12关，每关使用自己的 checkpoint；`--cpu` 可代替 GPU，`--max-steps` 可缩小原生预算。每次选择新的输出目录，程序会逐回合运行一个环境。

每个回合从初态开始，在首次成功、失败接触或预算耗尽时结束。运行异常、非有限状态和非法动作会单独报错。要做零延迟质量扫描，可参考 [flow_quality.py指南](../../../docs/kinetix-flow-quality.md)，默认运行12关、N=1..5、每格128个配对 seed。固定 N 的延迟校准、四档硬件映射及512-seed流程见 [预测器教程](../../../docs/kinetix-latency-study.md)。

## fine：细粒度延迟

默认的 `--mapping fine` 用新旧执行器指令的混合比例表示分数延迟；`coarse` 则把延迟取整到完整控制步。旧名称 `native-blend` 和 `legacy-round` 仍可使用，运行记录会保留你传入的名称。

原生物理步长是 `dt=1/60 s`，`frame_skip=2`，因此一条控制指令持续33.333 ms。默认回合预算是256条，每执行4条后更新一次 policy。flow 模型内部的 ODE 步长 `1/N` 与物理步长分开。第 k 个物理槽使用以下混合方式：

```text
old_weight[k] = clip(L / physics_dt - k, 0, 1)
command[k] = old_weight[k] * process(old_action)
           + (1 - old_weight[k]) * process(new_action)
```

L 与 physics_dt 使用同一时间单位。新旧动作先按原生规则完成 motor/thruster 绑定、裁剪和自动电机处理，再进行混合；同一个控制步共享标准差0.1的噪声。例如21.938 ms相当于1.31628个物理槽：第一个槽用旧指令，第二个槽用31.6%旧指令与68.4%新指令，随后使用新指令。每条控制仍执行两次原生物理求解。

等待新动作时，旧指令取自上一预测块未执行的尾部。初始队列是经过原生转换的零动作，不做预先推理。延迟不能超过当前执行窗口或旧尾部长度；H=8、每次执行4条时，最大延迟为133.333 ms。零延迟、完整旧动作和恒定指令直接使用原生步进，`--validate-native` 会逐值比较状态、观测、奖励与终止结果。

这里的分数槽表示指令在时间上的平均。碰撞检测和状态求解仍在原生物理网格上运行，虚拟延迟只决定新旧指令各占多少；宿主推理耗时单独记录。

## 旧映射与硬件profile

`coarse` 使用 nearest-even 规则，将延迟取整到控制周期；小于半个周期的延迟可能变为0，输入值和实际生效值都会保存。它保持零初始队列和原生物理参数。此前约0.417 ms的物理细分会改变约束求解频率，属于独立研究，不是这个入口的延迟设置。

单硬件profile使用两列CSV，以下为格式示例：

```text
flow_steps,latency_ms
1,7.465
3,21.938
5,36.738
```

使用 `--flow-steps 1,3,5 --latency-profile profile.csv` 加载这类文件，并省略 `--latencies-ms`。manifest 会记录文件 hash 和外部 profile 来源。使用之前核对目标硬件、dtype 和后端：历史 Torch/FP16 或低精度计时可以作为声明的延迟场景重放，当前 JAX/float32 模型的真实推理时间需另行测量。

[2026-08-11历史档案](../../../docs/kinetix-hardware-latencies.md) 与 [CSV](../../../docs/data/kinetix-hardware-latencies-20260811.csv) 保留RTX 6000 Ada、RTX 3090、AGX Orin 15W／30W各N=1～5，同架构随机权重、PyTorch eager FP16。先筛选一个 `profile_id`，再导出上述两列。

## 输出与统计

| 文件 | 内容 |
| --- | --- |
| `case-manifest.json` | 仓内源码/关卡/checkpoint身份、物理配置、模型结构、设备及状态 |
| `run.log` | 加载、编译/运行进度和异常 |
| `requests.jsonl` | 每次观测/推理、虚拟release、新旧动作来源、物理槽权重及控制步 |
| `episodes.jsonl` | 每回合成功、实际控制/物理步数、预算、return、模拟时间和推理次数 |
| `summary.md/.json/.csv` | 分关卡、flow步数、延迟单元汇总SR、全体预算惩罚步数和仅成功步数 |
| `videos/` | 可选完整回合录像，初始帧加每条实际控制后的帧 |
| `failure.json` | 基础设施异常；已完成部分保持partial状态 |

视频默认关闭。开启后，程序将原生125×125画面补边到126×126编码，保存初始帧和每条控制后的画面，播放帧率按控制周期设置。统计时，失败按声明预算惩罚，成功使用实际步数；成功平均步数以成功回合数为分母。失败接触可能早于预算，实际步数仍保存在 ledger，每个实验组合分别汇总。

```bash
python benchmarks/dynamic/kinetix/summarize.py --input runs/dynamic/kinetix/matrix-001
```

分析未完成的运行时加 `--allow-partial`。请求中的 host 时间可能包含首次 JIT 编译；要比较稳态推理性能，应先预热，再单独计时，并使用相同测量范围的 baseline。

## 源码与范围

`src/robotics_bench/kinetix/` 中，`flow_model.py` / `policy.py` 负责模型和参数/PRNG，`native/` 隔离KINETIX/Jax2D源码，`environment.py` 实现步进/指令混合，`protocol.py`、`runner.py`、`results.py` 分别管理延迟、回合与统计。执行源码和小型资产随仓维护，框架与checkpoint外部绑定。

来源/改动hash见 [PROVENANCE.json](../../../src/robotics_bench/kinetix/PROVENANCE.json)，许可见 [第三方说明](../../../THIRD_PARTY_NOTICES.md)，历史机制见 [参考整理](../../../docs/kinetix-reference-review.md)。DOM使用 [独立入口](../dynamicvla_dom/README.md)。

## 当前验证

RTX 6000 Ada接入检查覆盖 `car_launch` / `mjc_walker`、N∈{1,5}、L∈{0,21.938 ms}、seed0，共8回合。car_launch四格成功，步数41、43、44、46；walker为成功167步、失败、成功149步、成功207步，失败按256条计入总体。

两关通过zero-delay、full-old、constant-command逐值对齐，四配置初始hash一致；八段126×126视频可解码，帧数=实际步数+1，wheel包含关卡、纹理、许可与CPU统计。

零延迟质量矩阵覆盖12关、N=1..5、每格128配对seed，共7680回合，平均成功率依次78.45%、85.74%、89.32%、89.97%、88.87%。区间与绘图见 [质量评估](../../../docs/kinetix-flow-quality.md)。独立 `benchmark_eval.py` 在12关、N=5、4seed比较预处理JIT：9关逐步检查通过、稳态加速约1.12～1.25×；3关未通过逐位检查，grasp_easy原始GPU路径自身也有逐位波动。未认证全任务加速比，选项不进入默认runner；汇总与重复性检查用 `analyze_eval_benchmarks.py` / `diagnose_eval_preprocess.py`。

## English summary

The owned JAX policy and KINETIX/Jax2D runtime replay quality/latency cells on the native physics grid. Fine uses time-averaged processed commands; coarse rounds to control bins. Supply trusted checkpoints, compare paired seeds and inspect each cell's success, budget and request records. The [English predictor tutorial](../../../docs/kinetix-latency-study.en.md) extends this workflow to frozen predictions and hardware-latency validation.
