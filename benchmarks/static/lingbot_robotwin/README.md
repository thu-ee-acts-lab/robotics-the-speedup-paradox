# LingBot-VA＋RoboTwin 静态实验

[环境配置](../../../docs/environment_setup.md) · [评估与剪枝实操](../../../docs/hands-on.md#34-lingbot剪枝契约与evaluator结果)

这个入口在 RoboTwin 双臂任务上运行 LingBot-VA。模型放在独立 Python worker 中，模拟器通过本机 WebSocket 请求动作，保留模型原有的条件帧和 KV/VAE 更新方式。可以选择原精度的 `sync` 或 `paper_async`，默认任务是 `adjust_bottle`。

## 本地资源与环境

准备模型目录中的 `transformer/`、`text_encoder/`、`vae/` 和 `tokenizer/`，以及 RoboTwin 已解压的 `assets/objects`、`assets/embodiments/aloha-agilex` 和 `assets/background_texture`。预检会检查分片索引与文件 SHA-256，模型随后离线加载。源码和 checkpoint 以只读方式使用，结果写入新的 `--output-dir`；Torch/cuRobo 的编译缓存仍由运行环境管理，闭环评估无需 LeRobot 训练轨迹。

| 内容 | 模型环境 | 仿真环境 |
| --- | --- | --- |
| Python | 3.10.16 | 3.10.19 |
| Torch | 2.9.0+cu126 | 2.9.0+cu126 |
| NumPy | 2.2.6 | 1.26.4 |
| 模型依赖 | Diffusers 0.36.0、Transformers 5.0.0、FlashAttention 2.8.3 | 无需安装模型运行栈 |
| 仿真依赖 | 无需安装SAPIEN | SAPIEN 3.0.0b1、mplib 0.2.1、cuRobo及Vulkan |

已有实验使用 [LingBot 源码](https://github.com/Robbyant/lingbot-va) `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`、RoboTwin posttrain 权重，以及本地 RoboTwin 版本 `125db35a88cb89d859da7691a8909d25cd9cbacf`。每次运行会在 manifest 中记录实际文件，便于检查环境是否一致。

Transformers 5为UMT5额外初始化 `encoder.embed_tokens` 时，worker根据checkpoint的 `shared.weight` 恢复共享embedding并审计覆盖，记录于 `checkpoint-load.json`。源码需FlashAttention依赖，本case实际attention为Torch SDPA。

## 启动

从仓库根目录填写路径并预检，再选择空闲GPU：

```bash
mkdir -p .local
cp -n benchmarks/static/lingbot_robotwin/paths.env.example .local/lingbot-robotwin.env
# Replace PATH/TO values with existing local resources.
source .local/lingbot-robotwin.env

# Standard-library preflight; no model or simulator is imported.
bash benchmarks/static/lingbot_robotwin/run.sh \
  --task adjust_bottle --episodes 1 --start-seed 10000 \
  --output-dir runs/static/lingbot_robotwin/trial-001 --dry-run

# Actual evaluation: starts and stops its own local model worker.
bash benchmarks/static/lingbot_robotwin/run.sh \
  --task adjust_bottle --episodes 1 --start-seed 10000 --model-seed 0 \
  --gpu 3 --record-video \
  --output-dir runs/static/lingbot_robotwin/trial-001

# Paper-style observation delay, including the model's KV/VAE history.
bash benchmarks/static/lingbot_robotwin/run.sh \
  --task adjust_bottle --episodes 1 --start-seed 10000 --model-seed 0 \
  --schedule paper_async --overlap-actions 2 --gpu 3 --record-video \
  --output-dir runs/static/lingbot_robotwin/paper-async-001
```

需要临时更换路径时，使用 `--model-python`、`--lingbot-source`、`--robotwin-source` 或 `--checkpoint` 覆盖 env 文件。仿真客户端由 `ROBOTICS_ROBOTWIN_PYTHON` 启动，两个进程使用同一张指定 GPU，通信端口自动选择。默认 CPU offload 把 VAE 和文本编码器留在 CPU；`--no-cpu-offload` 可以关闭它。默认启动等待时间为600秒。

## 控制与缓存协议

模型输出形状为 `[16, 2, 16]`：16个双臂末端/夹爪通道、2个帧块，每帧含16条控制指令。首块的第一帧作为条件，不执行，因此首次实际执行16条；后续完整块执行32条。

```text
选择原生expert能够完成的场景种子
→ 重建该场景并获取指令、相机和初始末端位姿
→ reset模型缓存与本回合模型随机种子
→ 推理动作块
→ 逐条执行双臂末端动作，检查成功和预算
→ 完整块每4条指令选一组真实历史观测，更新KV/VAE缓存
→ 下一块；成功或预算耗尽立即结束
```

首次更新缓存使用4组真实观测，后续完整块使用8组。缓存中的动作条件保留模型原始预测及首块条件帧，末端位姿由模拟器适配器转换后执行。如果中途成功或用完预算，就停止更新剩余未执行部分。

`primitive_steps` 统计原生 `take_action` 接受的控制指令。每条指令可能推进多个250 Hz物理步，实际时长由运动规划决定。`adjust_bottle` 的预算是400条，`--max-steps` 可以缩小预算；模型、仿真或初始化异常单独报错。

初始化沿用 RoboTwin 的 expert 筛选，并记录尝试过和接受的 seed。筛选通过后，模型失败的回合仍正常计入结果。`--episodes` 是总评估回合数；下一回合从上一个接受的 seed 之后继续搜索，最多尝试 `--max-initialization-attempts` 个候选，默认32。

## 论文异步与缓存可见性

使用 `--schedule paper_async --overlap-actions n′` 开启 [历史观测实验](../../../docs/protocols/simulation.md)。n′ 按控制指令计数，范围是0..16，推理和控制仍顺序执行。同步要求 n′=0，异步的 n′=0 也会走相同的观测和动作路径。

在推理边界 t，模型使用 t−n′ 的完整观测。对于 KV/VAE 的历史关键帧 k，则读取 `max(0,k−n′)`；索引为负时重复第0步图像。这样，三相机与测量 state 会一起延迟，而首次4帧、后续8帧的缓存位置保持不变。缓存 RPC 中的 `state` 字段仍表示原始预测动作，包括条件帧；它与实际测量的 state 分别处理。

| n′=2 | 名义关键帧 | 实际观测来源 | 下一次推理边界 |
| --- | --- | --- | --- |
| 首块缓存 | 4、8、12、16 | 2、6、10、14 | t=16，最新观测14 |
| 第二块缓存 | 20、24、…、48 | 18、22、…、46 | t=48，最新观测46 |

运行后可以用 `requests.jsonl` 核对延迟。推理记录包含 `observation_step`、`history_offset_steps` 和 `cache_observation_max_step`，缓存记录包含 `nominal_keyframe_steps` 和 `observation_steps`。这些字段应表明模型最多看到 t−n′ 的观测。录像仍记录机器人当前执行的动作，回合终止后停止更新缓存。

所选规则会写入 manifest 和 `paper-async.json`。因为每条控制指令的实际时长可变，这个入口先记录指令数和 RPC 时间；估算论文周期还需要标定 Tact 与 Tinf。250 Hz物理频率和15 FPS视频帧率单独记录。

## 汇总作者 evaluator 报告

用 [score_reports.py](score_reports.py) 在 CPU 上汇总作者 evaluator 生成的 `*_analysis.json`。它读取的格式与当前 case 的 `episodes.jsonl` 不同。先运行合成示例检查安装，再按配置指定真实报告目录和共同 baseline：

```bash
python benchmarks/static/lingbot_robotwin/score_reports.py \
  --input benchmarks/static/lingbot_robotwin/scoring.synthetic.json \
  --output /tmp/lingbot-score-example
```

```bash
python benchmarks/static/lingbot_robotwin/score_reports.py \
  --group base=/path/to/base/episodes \
  --group candidate=/path/to/candidate/episodes \
  --baseline base --output /tmp/lingbot-score
```

工具输出逐回合 CSV，以及 CSV/JSON/Markdown 汇总。成功率使用全部回合，chunk 时延按成功 chunk 加权，端点时间与稳定性按成功回合加权。端点时间是 action/KV 请求往返时间之和；每个 chunk 都提供 `execute_actions_equivalent_s` 时，才额外报告包含控制等效时间的结果。Stability 组合归一化 jerk、符号翻转、高频和尾段波动，与按 SI 单位计算的轨迹 jerk 分开。合成示例用于检查聚合，也覆盖“chunk 更快、任务计算更慢”的情况。

## 输出与已验证结果

| 文件 | 内容 |
| --- | --- |
| `case-manifest.json` | 配置、资源及源码身份、状态 |
| `paper-async.json` | 延迟深度、缓存观测/动作条件约定及计时限制 |
| `run.log` / `server.log` | 仿真/调度日志与独立模型worker日志 |
| `checkpoint-load.json` | Transformer参数覆盖、文本embedding共享关系和模型运行信息 |
| `initialization-000000.json` | 原生种子筛选、指令、初始末端位姿、观测指纹 |
| `requests.jsonl` | 推理/缓存使用的观测步号、历史偏移及各自RPC墙钟耗时 |
| `episodes.jsonl` / `coverage.json` | 成功、实际控制指令数、预算、请求和缓存更新次数 |
| `episode-summary.md` / `.json` | 成功率、失败预算惩罚的总体步数、仅成功步数 |
| `videos/<task>/` | 可选完整控制指令录像，三视角横排 |

视频默认关闭。开启 `--record-video` 后，会保存初始画面、每条控制后的画面和终止画面，按15 FPS播放。结束时程序会清理自己的 worker，所以 `server.log` 中的 SIGTERM 可能只是正常退出；请结合 manifest 判断运行是否完成。

下面是 RTX 6000 Ada 上的两次单回合结果。两次都运行 `adjust_bottle/demo_clean`，使用环境 seed10000、模型 seed0 和400条指令预算；初始图像、末端位姿、指令与 seed 一致，Transformer 和文本编码器的加载覆盖检查通过。

| 配置 | 结果 | 控制指令 | 推理 / cache更新 | 视频帧数 |
| --- | --- | ---: | --- | ---: |
| 已有sync记录 | 成功 | 115 | 5 / 4 | 116 |
| paper_async n′=2 | 成功 | 120 | 5 / 4 | 121 |

异步推理边界0、16、48、80、112对应观测0、14、46、78、110，全部来源已核对；视频960×240、15FPS。CPU检查641 passed、3个可选GPU模块跳过，覆盖零延迟、非4倍数偏移、初始填充、缓冲隔离及终止/预算截断。上述结果报告指令数；时间加速比另需匹配的推理/控制测量。

## English summary

Configure the two local environments, run preflight, then select an idle GPU. The native loop executes 16 commands initially and 32 thereafter. `paper_async` shifts complete KV/VAE observation history while preserving predicted action conditioning; the table above reports matched single-scene results. The separate CPU scorer aggregates evaluator reports and distinguishes endpoint timing, control-equivalent timing and normalized action stability. See the [English hands-on guide](../../../docs/hands-on.en.md#34-lingbot-pruning-contracts-and-evaluator-reports) for the workflow.
