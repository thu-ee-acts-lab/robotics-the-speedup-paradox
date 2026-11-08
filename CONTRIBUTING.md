# 贡献指南 / Contributing

代码、配置和文档都在主题分支上修改，再通过 PR 合入 `main`。提交说明和 PR 使用中英文；维护者审核后，默认以 merge commit 合并。

Use topic branches and pull requests for code, configuration and documentation changes. Commit titles and PR descriptions use Chinese and English. Maintainers review and merge through PR merge commits.

## 1. 明确范围 / Define the change

一个 PR 解决一个问题，并注明对应的 `static/<case>`、`dynamic/<case>` 或 `shared`。小改动直接在 PR 中说明；跨模块或多人任务使用 [任务表单](.github/ISSUE_TEMPLATE/task.yml)，写明负责人、依赖和完成标准。未确定的负责人或审阅人标记为“待认领”。

Keep each PR focused on one reviewable problem and name its case or shared scope. Small changes can explain scope directly in the PR; larger work can use an Issue. Do not commit a separate planning document for every task.

## 2. 分支与提交 / Branches and commits

从最新 `origin/main` 创建主题分支，名称使用 `feat/`、`fix/`、`docs/`、`refactor/`、`perf/` 或 `chore/` 加简短主题。保留工作区已有改动，不重写他人的提交。多个独立任务使用独立分支或worktree。

```bash
git fetch origin
git switch -c docs/environment-guide origin/main
# Edit only the files in scope.
git diff --check
git add -- README.md docs/environment_setup.md
git commit -m "docs: 完善环境说明 / clarify environment setup"
git push -u origin HEAD
```

提交标题使用 `type: 中文摘要 / English summary`，可加scope，如 `fix(robocasa): 修复初态读取 / fix initial-state loading`。按逻辑变化提交，保留已有历史。

Start from current `origin/main` unless a dependency branch is explicitly required. Use focused commits with bilingual titles. Review the diff before staging; list the intended files rather than staging unrelated work.

**依赖PR：** 以前置分支为base并注明依赖；前置PR合并后调整base到 `main`，复核diff。功能实现与文档更新分别按实际范围命名。

For stacked work, explicitly name the prerequisite branch/PR and use it as the base. After the prerequisite merges, retarget to `main` and review the resulting diff. Existing local feature commits remain separate from a documentation-only review.

## 3. 创建PR / Open a pull request

使用 [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md)，四部分均提供中英文：

| 部分 / Section | 要说明的内容 / Content |
| --- | --- |
| 目的 / Purpose | 具体问题与改动后的行为 / Problem and resulting behavior |
| 范围 / Changes | 关键文件、职责、case及依赖 / Key files, responsibilities, case and dependencies |
| 验证 / Validation | 实际命令、结果、未验证范围 / Commands, observed results and unverified scope |
| 影响与回退 / Impact and rollback | 兼容性、数值或协议影响，以及回退方式 / Compatibility, numerical or protocol impact and rollback |

每节提供中英文，小改动各一句即可。验收期间使用Draft，完成后请求review。正文保留改动事实、验证结果和证据位置。

Use the four bilingual sections and a bilingual PR title. Keep small changes concise. Open a draft while work is incomplete; report checks that were not run explicitly.

可以使用GitHub网页，或在已登录的GitHub CLI中执行：

```bash
mkdir -p .local
cp -n .github/PULL_REQUEST_TEMPLATE.md .local/pr-body.md
# Fill in the actual change and validation results.
gh pr create --draft --base main \
  --title "docs: 完善环境说明 / clarify environment setup" \
  --body-file .local/pr-body.md
```

推送修复后更新验证说明，CI与审阅均对应当前commit。认证受阻时保留分支、base和PR正文，并报告实际提交状态。

### 文档写作 / Documentation writing

文档写给第一次使用项目的人：先说明工具能做什么，再给安装和运行命令，最后解释输出及常见错误。一个概念解释一次，其他页面用链接引用。中英文同步更新，准确保留支持范围、实测结果和许可信息。写作参考 [anti-defensive-writing](https://github.com/Adkid-Zephyr/anti-defensive-writing-Skill)。

Write for someone using the project for the first time: explain what the tool does, give installation and run commands, then explain outputs and common errors. Explain each concept once and link to it elsewhere. Keep both languages current and preserve supported configurations, measurements and licensing. See [anti-defensive-writing](https://github.com/Adkid-Zephyr/anti-defensive-writing-Skill) for writing guidance.

## 4. 与改动相称的验证 / Proportionate validation

| 改动 / Change | 验证 / Validation |
| --- | --- |
| 文档、模板、图稿 | 本地链接、命令参数、Mermaid解析/渲染及 `git diff --check`；不启动GPU / Check links, commands, diagrams and whitespace |
| Python行为或数据契约 | 相关行为和失败边界检查，再执行下方CPU检查 / Focused behavior checks and the CPU suite |
| 模型/控制协议 | 相关CPU检查与实际GPU闭环；记录初态、预算、成功和异常 / CPU checks plus scoped real execution |
| 数值/性能优化 | 匹配的baseline、输入、精度和设备；正确性及完整入口测量 / Matched baseline, correctness and complete-entry measurements |

文字修改检查文档即可；行为测试覆盖实际边界。检查通过后按新改动或失败追加验证，标明合成、单算子或单回合证据范围。

For behavior changes, run from the repository root:

```bash
python -m ruff check tools tests benchmarks src
python -m ruff format --check tools tests benchmarks src
python tools/validate_contracts.py --examples
python -m pytest -q
python -m compileall -q tools benchmarks src
```

开发环境见 [环境指南](docs/environment_setup.md)。公共CI使用托管CPU runner，不下载模型，不接入实验室GPU。视频或性能采集带来的开销必须与测量口径一起说明。

## 5. 审核与合并 / Review and merge

由真实的非作者reviewer核对范围、diff、兼容性和证据；AI自评不代替批准。检查通过且讨论解决后，维护者使用 **Create a merge commit** 合并；其他合并策略在PR注明。

A reviewer other than the author checks scope, behavior, compatibility and evidence. Maintainers merge after relevant checks and discussions are resolved. The default is **Create a merge commit**. Explain any different merge strategy in the PR.

不直接提交或推送到 `main`，不通过本地merge后直接push绕开PR，也不自动启用auto-merge。用户已经明确授权某个PR合并时，代理可在规则满足后执行，不重复请求同一授权；普通开发任务不代表已授权合并。

维护者按 [.github说明](.github/COLLABORATION.md) 核对服务端保护：`main`、`CPU contracts`、审查要求及force push限制。只报告实际确认的设置。

## 6. 文件与依赖边界 / Tracked files and dependencies

Git保留代码、必要输入配置、测试、稳定说明和小型合成样例。checkpoint、数据集、下载缓存、视频、日志、profile输出、个人笔记和临时PR正文留在被忽略的本地目录。公开文档只能引用仓内已跟踪文件或可访问来源，不能依赖某人的家目录。

Track code, required inputs, tests, stable documentation and small synthetic examples. Keep models, datasets, generated outputs, scratch notes and draft PR text local. Do not commit credentials or machine-specific runtime paths.

需要修改模拟器内部代码时，在独立fork提交最小改动，记录其来源和许可；主仓PR固定可获取的版本并说明跨仓依赖。参见 [模拟器依赖协议](docs/protocols/simulator-dependencies.md) 和 [第三方来源](THIRD_PARTY_NOTICES.md)。本项目自有许可证仍待维护者决定，不把第三方许可证自动应用到整个仓库。
