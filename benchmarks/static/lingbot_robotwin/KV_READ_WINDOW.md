# LingBot-VA KV read-window 剪枝设计

[分步操作](../../../docs/hands-on.md#34-lingbot剪枝契约与evaluator结果) · [模型入口](README.md)

这里提供 [论文](https://arxiv.org/abs/2606.28529v2) 中 LingBot-VA＋RoboTwin KV 剪枝的配置和 CPU 参考实现。先用它查看不同预算会保留哪些历史 token；它现在处理的是缓存元数据，还没有接入 GPU 模型 worker。实际接入步骤见后文。

## 方法边界

原生配置 `attn_window=72` 最多保留36个视觉 latent chunk 和36个 action chunk。Read-window pruning 保持缓存的分配、写入和替换方式，只在 action-denoise self-attention 读取历史 K/V 前，分别挑选视觉流和动作流中最近的 token：

```text
R_t = Recent_W_lat(valid latent slots) ∪ Recent_W_act(valid action slots)
```

论文Table 5操作点：

| 配置 | 启用read window | 视觉token上限 | 动作token上限 | 相对原生可读KV | 视觉历史chunk | 动作历史chunk |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | 否 | 8640 | 1152 | 100.0% | 36.0 | 36.0 |
| p1 | 是 | 1024 | 256 | 13.1% | 4.3 | 8.0 |
| p2 | 是 | 512 | 128 | 6.5% | 2.1 | 4.0 |
| p3 | 是 | 256 | 64 | 3.3% | 1.1 | 2.0 |
| p4 | 是 | 128 | 32 | 1.6% | 0.5 | 1.0 |

每个视觉 chunk 有240个 token，每个动作 chunk 有32个。表中的比例按 `(视觉上限 + 动作上限) / (8640 + 1152)` 计算，表示允许读取的 K/V 数量；显存占用和 GPU 时延需要在接入模型后实际测量。

## CPU参考契约

[kv_read_window.py](kv_read_window.py) 接收以下slot元数据：

- `slot`：唯一非负物理编号；`source`：`latent`、`action`或兼容值 `unknown`。
- `generation`：单调写入代次，已知来源须非负；`current`：本步必须保留的slot标记。

选择器分别处理视觉和动作预算，优先选取 generation 较新的 slot；generation 相同时，先选编号较小的 slot。预算0表示不设上限。`unknown` 和 `current` 始终保留，因此输出数量可能超过预算。返回的 slot 按编号升序排列，输入和缓存不会被修改；`base` 则保留全部有效 slot。

```bash
python benchmarks/static/lingbot_robotwin/kv_read_window.py \
  --input benchmarks/static/lingbot_robotwin/kv_read_window.synthetic.json
```

可以在输入中设置 `preset` 为 `p1` 到 `p4`，也可以直接填写 `enabled` 和两项预算，两种方式不能同时使用。运行后会打印完整配置、保留和丢弃的 slot，以及各来源的数量。仓内示例是合成输入，输出中的 `synthetic` 字段会标明这一点。

## 运行时接入与验收条件

如果要把它接入 GPU worker，请增加一个默认关闭的开关，并按下面的顺序检查：

1. 写cache时记录source/generation，保留来源未知slot和当前query对应K/V。
2. 在transformer blocks前选择一次indices，各层复用；只限制action-denoise self-attention的历史读取。
3. manifest保存启用状态、两项预算、`attn_window`、dtype/backend与源码身份。
4. 关闭功能时逐元素对齐原生动作；启用后固定observation/noise/seed检查多步误差。
5. 计入选择/gather及完整action-inference耗时，按相同episode seeds比较成功率、成功chunk数、稳定性和任务时间。

当前模型仍从指定的外部 LingBot 源码加载。完成 worker 接入后，再运行固定输入和闭环对照，记录实际的显存、推理时间和任务结果。

## English summary

The CPU selector applies separate recent-token budgets to visual/action KV metadata, preserving current and unknown slots. Cache allocation and writes remain unchanged. Presets `base`, `p1`–`p4` and synthetic checks are available now; model-worker integration follows the opt-in steps above, with numerical, full-inference and paired-rollout validation.
