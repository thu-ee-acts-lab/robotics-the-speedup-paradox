# 架构图源码

这些图用 Mermaid 编写。修改 `.mmd` 后，按下方命令生成 SVG 和本地预览页。当前代码的完整说明见[架构指南](../architecture.md)，运行步骤见[实操指南](../hands-on.md)。

| 图稿 | 内容 |
| --- | --- |
| [01-architecture.mmd](01-architecture.mmd) | 三个已实现静态case、外部依赖、CPU工具与规划中的动态case |
| [02-async-loop.mmd](02-async-loop.mmd) | 当前 `paper_async` 的串行历史观测选择与动作执行 |
| [03-inference-path.mmd](03-inference-path.mmd) | Cosmos加载审计、本仓执行代码、可选算子与动作块交付 |
| [04-contribution-flow.mmd](04-contribution-flow.mmd) | 主题分支、中英文提交/PR、相关验证和维护者合并 |

图中绿色表示已实现模块或协作步骤，灰色表示外部资源，蓝色表示后续设计。实线表示调用或数据传递，虚线表示配置或可选依赖。GitHub 分支保护需要在仓库设置中单独查看。

图 02 展示 Cosmos 串行 runner：保存每步观测，需要动作块时读取 t−n′快照，图像与 proprio 同源延迟。π0.5 由外部 evaluator 实现相同历史观测语义；论文周期和宿主耗时分别记录。

图 01/03 是静态接入参考图。DOM 与 KINETIX 的当前执行路径见[架构指南](../architecture.md)，各 case 验证记录随对应指南维护。

## 本地生成

在独立文档环境安装 Mermaid CLI，使用 [render-config.json](render-config.json)。进入本目录后：

```bash
mkdir -p rendered
for source in ./*.mmd; do
  stem=$(basename "$source" .mmd)
  mmdc -i "$source" -o "rendered/$stem.svg" \
    -I "fig-$stem" -c render-config.json -b white -w 1800
done
python build_gallery.py
```

需要指定浏览器时，向 mmdc 传入 Puppeteer 配置；渲染工具安装在独立文档环境。

[build_gallery.py](build_gallery.py) 生成可缩放／下载的 `architecture-atlas.html`、`gallery.md` 和 `render-manifest.json`。这些产物均被 Git 忽略。
