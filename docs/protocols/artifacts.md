# 数据契约与演进

为工具新增实验记录格式，或接入新的日志生产者时，从本页开始。通用 manifest/trace 使用 **schema_version 1.0.0**；各 case 的运行命令和专用输出见 [操作指南](../hands-on.md)。

可用字段和枚举由 [manifest schema](../../schemas/run-manifest.schema.json)与 [event schema](../../schemas/trace-event.schema.json)定义，[字段说明](../../schemas/README.md)提供阅读索引。新增字段时，先更新 schema 和样例，再让程序写入。

## 先验证一个例子

从仓库根目录执行：

```bash
python tools/validate_contracts.py --examples
python tools/validate_contracts.py --manifest examples/contracts/synthetic.manifest.json --trace examples/contracts/synthetic.trace.jsonl
```

两条命令均输出 `VALID synthetic-run: 8 events`，验证合成的观测—请求—完成—释放—动作—终止序列。验证自己的 run 时替换第二条命令的文件路径。非法数据退出 1，用法错误退出 2；工具只读数据，manifest 中的路径和命令不执行。

## v1 已检查的内容

manifest 声明 run、synthetic 标记、调度与时钟/延迟两个配置轴、clock domain、episode seeds 和预算；可选 summary 不能在零成功时给出非空成功条件均值。未知字段拒绝，防止拼错参数被无声忽略。

trace 的 observation、request、chunk 和 action 必须关联同一 episode；时间在同一域中不倒退；动作在结果完成并释放后才可执行。episode 结束后拒绝新观测、请求、释放和动作，允许旧请求完成/丢弃的诊断事件。每个声明 episode 都有唯一 terminal，空或未完成 trace 不作为完整 artifact 通过。

一个 run 包含串行 episodes，同域时间跨 reset 连续。多机器人或并行 episodes 分别导出 run；联合时间线扩展需定义 agent/session ID、时钟、cache ownership 及跨 run 关联。

多个进程的事件需要先汇总、排序，再写入 JSONL。`event_seq`记录文件中的事件顺序；计算时间差时使用同一个 clock domain。

<a id="校验没有证明什么"></a>

## 校验与执行证据

校验器负责检查字段和事件关系。物理推进、推理、队列、延迟、deadline、资源清理与预算执行由运行程序的测试检查。硬件来源、summary 计算、动作年龄和性能结果还需要各自的原始记录。

checkpoint、backend、量化层 map、硬件状态、profile 来源及视频帧引用属于后续扩展。新增字段需提供兼容设计、schema 和样例，继续保持严格校验。

[仿真协议](simulation.md)的手算案例用于 runtime 验收；`tests/test_contracts.py`验证校验器的数据接受/拒绝规则。

## 如何修改契约

修改格式的 PR 需要说明：为什么增加字段、读取方如何处理、旧样例是否仍有效，以及新增的正常和错误样例。根 schema 拒绝未知字段，因此新增可选字段时也要检查旧读取器。

改变字段含义、单位、clock/ID 规则或必填条件时，升级版本并提供迁移方法。纯文案修改保留 schema_version；需要支持多版本时，再增加相应读取逻辑。

持久化 artifact 使用可定位的 run ID 和相对引用；不提交私有路径、凭据、模型权重或大型原始结果。正式实验结果应另存版本、hash 和可访问的 artifact URI，仓内只保留小规模可公开例子。
