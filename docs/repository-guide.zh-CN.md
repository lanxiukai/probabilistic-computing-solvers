# 仓库指南

[English](repository-guide.md) | 简体中文（译文）

本仓库展示验收后的研究实现、核心数据和复现材料。探索实验、调参、完整原始档案及运行日志由独立的私有研究工作区维护。

从[概率计算与p-bit介绍](probabilistic-computing.zh-CN.md)了解概念、原理和发展背景；
[实验总览](benchmark-overview.zh-CN.md)汇总已验收规模、耗时、并行状态及显存口径。

## 项目边界

使用单仓库和一个锁定的 uv 环境。配置、脚本、测试、数据、基线锁和来源记录按问题分组。
各 Python 问题包独立维护；只有多个项目确实需要时才提取共享代码。共享模块不得反向依赖问题专属研究逻辑。

| 路径 | 内容 |
|---|---|
| `src/<problem_package>/` | 独立求解包；因式分解使用 `sparse_factorization` |
| `configs/<problem>/` | 冻结的公开输入和协议 |
| `scripts/<problem>/`、`tests/<problem>/` | 复现工具和仅本地保留的软件检查 |
| `data/<problem>/` | 验收后的核心数据、证书及标注版本的历史证据 |
| `baselines/<problem>/`、`provenance/<problem>/` | 基线版本和源码来源 |
| `docs/<problem>/` | 报告、算法、复现及源码指南 |

代理约定（`AGENTS.md`）与软件检查（`tests/`）仅在本地保留：
Git 忽略这些路径，已有本地文件继续保留。

仓库包含因式分解、立方自旋玻璃、TSP、MWIS、加权Max-SAT、稀疏／稠密Max-Cut、
LABS、严格二分及LDPC似然搜索。新增七个专题使用独立的
`combinatorial_solvers/<problem>/`子包，共享模型／来源记录工具。
冻结调用配置与证书位于`data/<panel>/rtx5090/`，见[复现指南](reproduction.zh-CN.md)。

修改单一项目时保留其他项目的接口、证据及环境。
新应用应有实际验收结果，空目录不代表成果。

## 文档与引用

英文为主文件：`README.md` 和 `name.md`。中文译文使用 `README.zh-CN.md` 和
`name.zh-CN.md`。每对文档顶部提供语言切换链接，同步更新数值、公式、命令、证据边界和引用标识。
有中文对应页时，中文链接指向译文。论文题名保留原文。

只有Markdown正文使用中英双语。仓库名、路径、目录和文件名（含Markdown文件名）用英文，
代码标识符、注释、docstring、命令帮助/日志/错误信息、配置和数据字段及文本值、图表标注也用英文。
生成报告所需译文存入成对Markdown模板，源码与测试保持英文。此规则作用于公开仓库。

文档只用 Markdown；图表可以使用嵌入 Markdown 的 SVG 或 PNG，不生成文档或图表的 PDF。
各实验报告末尾列出带链接的完整参考文献，涵盖该项目研究实际用到的全部来源，包括背景与排除方向。
论文 PDF、附件、获取记录和详细引用审计留在私有工作区。引用不表示采用或完成复现。

根 README 保持简洁，实现细节放在各项目源码指南。本项目 GPU 结果采用 RTX 5090 实测，
CPU 对照及其硬件另行说明。

## 维护与复现

维护前读取两个仓库的适用约定，检查分支、暂存内容、运行中任务和文件归属。按明确清单导入验收材料，
独立验证后在私有工作区登记来源。私有研究批次完成不自动向此处导入中间结果。

普通软件检查使用本项目锁定环境；耗时复现另行核对设备和预算。计时 GPU 同时只运行一个实验。
网络传输和 CPU 核验可与后续计算重叠处理已封存段，前提是不显著影响采样。

仅当当前分支为 `dev` 且相关检查通过时，授权自动创建本地提交；使用英文 Conventional Commit，
按任务暂存并保留无关修改。远端发布、许可证选择及公开范围变化遵循所有者的相应授权。

## 公开分支与许可证

本项目使用[MIT许可证](../LICENSE)。本地`dev`保留开发历史；公开`main`只包含一次根提交，文件树与验收的`dev`一致。获授权发布时只推送该明确分支，不发送开发分支或标签；发布后保留本地开发分支。

加权集合覆盖的验收源码位于`src/combinatorial_solvers/set_cover/`，
117次自然调用及4次回放核心证据位于`data/set-cover/rtx5090/`。
参见[报告](set-cover/report.zh-CN.md)及[独立复现入口](set-cover/source-guide.zh-CN.md)。

JSSP的72次自然调用及3次回放见[报告](jssp/report.zh-CN.md)，[源码指南](jssp/source-guide.zh-CN.md)提供下载、独立核心见证验证和隔离回放入口。

覆盖数组保留50次自然确认调用及3次回放，另有独立的10次开发调用及1次回放。[报告](covering-array/report.zh-CN.md)和[源码指南](covering-array/source-guide.zh-CN.md)给出模型、独立证书验证和复现步骤。

图着色分别保留最终确认、开发比较和等工作量实现面板。原始DIMACS输入按固定哈希下载，自研TabuCol风格程序作为独立CPU参考，非上游HEAD复现。参见[报告](graph-coloring/experiment-report.zh-CN.md)和[源码指南](graph-coloring/source-guide.zh-CN.md)。

QAP保留72次独立确认、4次实现控制及5次回放；Q1c开发与官方Ro-TS另列台账。[报告](qap/experiment-report.zh-CN.md)和[模型／复现指南](qap/README.zh-CN.md)给出完整参数、见证核验及三个面板的边界。
