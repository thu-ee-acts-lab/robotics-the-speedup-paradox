# 后端状态、重置与能力声明

接入有状态模型时，需要明确哪些数据可以跨回合复用、哪些必须在 reset 时清理。本页给出模型加载、缓存和重置的约定。现有 case 的实现见 [架构指南](../architecture.md)；通用 schema 记录 episode/request 生命周期，新增 RPC 或缓存字段时遵循 [格式变更流程](artifacts.md)。

## 会话与 episode 的边界

一个连接（session）可以运行多个实验回合（episode）。每次 reset 都要指定结束哪个回合、开始哪个回合及其 seed，并在对应模型状态清理完成后确认。

同步 reset 的顺序是：停止接收新请求，结束旧 episode，等待在途计算完成并清理状态，然后接受下一回合的请求。异步 reset 返回完成句柄或 ack；旧 epoch 的晚到回调只用于诊断。GPU 工作完成或可靠取消后，才能复用 buffer。

## 状态分区

| 状态 | 通常的有效范围 | 失效条件必须由实现声明 |
| --- | --- | --- |
| 不变权重与 packed weights | backend/model 实例 | checkpoint、量化格式、LoRA/模型权重变更 |
| workspace 与 graph 实例 | device/stream/shape/layout | shape、输入地址、graph 约束或并发 ownership 改变 |
| prefix/text/visual KV | 明确的任务或 episode | prompt、输入、模型版本、mask/position、任务切换及 reset |
| action/temporal state | episode | 已执行 offset、时间窗口、episode reset |
| 请求输入输出 | request 至消费者用完 | completion、队列消费、drop/cancel 后可靠 drain |

每个 adapter 需要说明缓存如何索引、谁可以共享或修改，以及何时失效。读取窗口、存储容量和写入频率分别影响算量、内存和状态一致性，需要分开记录。

跨任务共享要求 checkpoint/adapter/precision、输入前缀、position/mask 等键等价。只读 tensor 引用保留至所有消费者完成；可写 workspace 隔离或串行。缓存布局与验收按模型定义。

## 后端能力与加载模式

backend 声明模型变体、shape/dtype/layout/device、并发方式、graph/cache 及 fallback。超出能力的请求显式失败，或记录已选择的 reference fallback 与实际精度。

缺失指定 checkpoint 时加载报错；显式 synthetic 模式可随机初始化，并在产物中保留 synthetic 标记。通用 manifest 扩展需记录 checkpoint/adapter、预处理、normalization/action 及实际 backend；这些来源字段须先加入 schema。

<a id="后续验收"></a>

## 验收条件

一个连接连续运行两个 episode 与重新建立两个独立 reference 实例的结果应在预定条件下相符；reset 后旧请求晚完成不能污染新状态；修改 prompt、mask、shape 或 LoRA 后按声明失效；同样输入的测试需固定噪声而非依赖偶然重复。

计时声明 normalization、tokenization、vision、prefix、action suffix、排队和 transport 的覆盖范围。跨帧任务按完整 request 记录 start/end，并保留未完成请求状态；请求延迟与每帧统计分开。

通用 v1 run 使用串行 episodes；多任务共享 cache 和多 agent 并发按独立需求扩展并验收。
