# 第三方来源与发布状态

本页及各目录 `PROVENANCE.json` 记录推理、算子和模拟器源码及小型资产的来源。TurboVLA、Jetson-PI、Jetson-PI-Edge、OxyGen为研究参考；权重和大型数据集单独获取，遵守各自条款。

## Cosmos＋LIBERO 外部调用边界

[Cosmos case](benchmarks/static/cosmos_libero/README.md) 通过本仓适配器调用 [NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy) 兼容加载/配置依赖及原生LIBERO接口；policy、sampler 与 DiT 执行代码已建立本仓副本。当前检查的Cosmos源码根许可证文件声明Apache-2.0；该声明不覆盖模型权重、数据或所有嵌套依赖。checkpoint、配套统计量/T5和VAE文件的来源与条款须分别核对，不能将源码许可证套用于资产。

Cosmos已验证副本缺少 `.git`，以Python源码树哈希标识；有Git信息时记录commit/dirty状态。已检查的LIBERO环境来自 [lerobot-libero](https://github.com/huggingface/lerobot-libero) 安装包，须连同实际包版本、任务/初态和资产配置记录，不默认等同于其他 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) checkout。

外部运行环境、源码依赖、checkpoint、T5、VAE和模拟器资产由调用者准备，公开模板通过 `PATH/TO` 绑定；仓内移植范围见下文。

## 来源记录与发布

[Cosmos＋RoboCasa case](benchmarks/static/cosmos_robocasa/README.md) 调用 [moojink/robocasa-cosmos-policy](https://github.com/moojink/robocasa-cosmos-policy) fork，已验证commit为 `edd9a328b3ec98050f42d194c1419307a79c4d87`；根源码许可证为MIT。任务预算、原生控制器参数和观测/动作约定参考Cosmos官方RoboCasa评测接口，适配器由本仓库实现。外部fork、其下载的厨房资产、controller pickle、checkpoint及训练数据均不随本仓库分发；各资产仍须独立遵守原条款。

后续引入第三方组件时，在对应 PR 记录：上游 URL、完整 commit SHA、实际文件、引入方式（依赖/子模块/vendor/重写）、原文件许可证与版权、修改说明，以及模型权重或数据的单独条款。保留原始许可文件，不能根据根 LICENSE 推断全部嵌套内容的授权。

自有代码许可证由维护者确定，当前未添加项目级LICENSE；发布前完成选择并更新README。新增资产单独核对条款。

## Repository-owned inference and operator copies

The optional `src/robotics_bench/models/pi05/` runtime includes reviewed LeRobot,
OpenPI-compatible Gemma/PaliGemma, and SigLIP implementations from the validated
local environment. Each original file is identified by SHA-256 in its
`PROVENANCE.json`, including local adaptation.
Original Apache-2.0 notices are retained, with the license text in
`third_party/licenses/Apache-2.0.txt`. Common Transformers and LeRobot package
services remain dependencies; inference forwards are owned here.

`src/robotics_bench/models/cosmos/` similarly owns the reviewed policy, sampler,
action utility, and DiT execution sources from NVlabs/cosmos-policy. Its
`PROVENANCE.json` records the exact source hashes and retained external framework,
VAE and attention dependencies. The native loader constructs the model and loads
weights before the explicit local-runtime binding. Model weights/data have
separate terms and are not included.

`src/robotics_kernels/blackwell/` preserves FP4/FP8 implementations as independent
copies from the workspace VLM project, with the source revision/file hashes and
original MIT license in that directory. Imports, extension namespaces, environment
variables and build locations belong to robotics. Blackwell compilation and device validation remain separate from Ada measurements.

The INT4/INT8 CUTLASS visitor-based GEMM design references the local QuaRot and
Mini QServe prototypes. It independently defines symmetric integer packing,
float32 scales, BF16 output, model-independent tactics, framework-owned workspace
and current-stream execution. Integer build headers are an unmodified CUTLASS
snapshot at `982748aa7356fa838c2ea4994ddcb0b2a4b4cefa`, maintained inside
`src/robotics_kernels/ampere_ada/third_party/cutlass/`. Only `include/`,
`tools/util/include/` and the original `LICENSE.txt` are included, with source
URL and per-file hashes in `PROVENANCE.json`. The runtime does not reuse another
project's source or headers. CUTLASS retains its BSD-3-Clause and file-specific
notices. Integer tests compare the quantized mathematical
reference; this is separate from model quality versus BF16.

## LingBot-VA and RoboTwin external execution

The LingBot RoboTwin adapter calls [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) through a separate local worker, using its native NumPy/messagepack codec and model implementation. The inspected LingBot source is 7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb; root licensing is Apache-2.0. RoboTwin/SAPIEN/cuRobo and model assets remain external dependencies under their own terms. Model sources, weights and assets are supplied externally. The owned adapter records source identity, checks actual transformer/text-encoder parameter coverage, and restores the checkpoint shared-embedding alias when required by Transformers 5.

## Owned RTC / Kinetix / Jax2D execution

`src/robotics_bench/kinetix/flow_model.py` derives from Physical Intelligence's [real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) at `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`, under the [retained RTC MIT license](src/robotics_bench/kinetix/RTC_LICENSE).

The private `native/kinetix` runtime, small textures and bundled level definitions retain their source provenance. Kinetix is pinned to `cf7453ea103fa0b77348af1a39f689c658161613`, copyright Michael Matthews, under the [retained Kinetix MIT license](src/robotics_bench/kinetix/KINETIX_LICENSE). The private Jax2D physics implementation comes from distribution version 1.0.0 under its [retained MIT license](src/robotics_bench/kinetix/JAX2D_LICENSE).

[PROVENANCE.json](src/robotics_bench/kinetix/PROVENANCE.json) lists original file hashes and migration changes. Imports use repository-private names; inactive duplicate definitions and cloud/training serialization helpers were excluded. Model checkpoints remain external resources and are not redistributed. JAX, Flax and general runtime libraries remain separately installed dependencies. Repository-owned code retains its separate licensing status.

## DynamicVLA and DOM owned execution

`src/robotics_bench/models/dynamicvla/` and `src/robotics_bench/dynamicvla_dom/native/`
contain selected source derived from [hzxie/DynamicVLA](https://github.com/hzxie/DynamicVLA),
using the documented local revision and per-file hashes in their PROVENANCE files.
Both retain **S-Lab License 1.0, restricted to non-commercial use**; moving code
into this package does not relicense it or grant unrestricted commercial use.
Derived preprocessing and the engine/client/server adapters preserve the same
source attribution. Franka configuration additionally retains the Isaac Lab
BSD-3-Clause notice and `native/ISAACLAB_LICENSE`. Isaac SDK, checkpoint, scenes,
objects and tokenizer/backbone caches remain separately obtained external resources.
