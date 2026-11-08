# GitHub协作配置 / GitHub Collaboration

代码与文档的分支、提交和审核流程见 [CONTRIBUTING](../CONTRIBUTING.md)。

| 文件 | 作用 |
| --- | --- |
| [PULL_REQUEST_TEMPLATE.md](PULL_REQUEST_TEMPLATE.md) | 与VLM项目一致的中英文四部分：目的、范围、验证、影响与回退 |
| [ISSUE_TEMPLATE/task.yml](ISSUE_TEMPLATE/task.yml) | 需要认领或协作的任务范围、依赖与验收 |
| [CODEOWNERS](CODEOWNERS) | 代码owner映射；指定账号须拥有相应仓库权限 |
| [workflows/cpu.yml](workflows/cpu.yml) | CPU检查，稳定job名称为 `CPU contracts` |

## CPU CI

工作流由 `pull_request` 和 `main` 的 `push` 触发，运行Ruff、契约样例、CPU测试与语法检查。使用GitHub托管runner、Python 3.11；token为 `contents: read`，checkout不持久化凭据。CI没有GPU/self-hosted任务或模型下载。

| Action | 固定提交 |
| --- | --- |
| [actions/checkout](https://github.com/actions/checkout) | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| [actions/setup-python](https://github.com/actions/setup-python) | `5fda3b95a4ea91299a34e894583c3862153e4b97` |

更新action版本时核对官方提交与runner兼容性，保持完整SHA固定。相关建议见 [GitHub Actions安全说明](https://docs.github.com/en/actions/reference/security/secure-use)。

## 服务端设置 / Server settings

管理员在GitHub服务端配置并核对以下保护规则：

1. 默认开发目标为 `main`，修改通过PR进入。
2. 相关CI运行正常后，将 `CPU contracts` 设为必需状态检查。
3. 配置真实非作者reviewer，解决讨论后允许合并；唯一owner不能代替自己的非作者审查。
4. 禁止对 `main` force push和删除，按团队权限确定是否限制管理员绕过。
5. 保留merge commit方式；不启用自动合并作为默认路径。

设置方式见 [GitHub分支保护文档](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)；记录实际启用的规则。

Administrators verify PR-only changes to `main`, required `CPU contracts`, non-author review and force-push restrictions. Merge commits are the default; record the settings actually enabled.
