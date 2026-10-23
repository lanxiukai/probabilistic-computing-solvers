# 严格平衡图二分

[English](README.md) | 简体中文

实测见[RTX 5090报告](experiment-report.zh-CN.md)。在$\sum_i s_i=0$约束下最小化
$C=\sum_{(i,j)}w_{ij}(1-s_i s_j)/2$，
要求偶数顶点、正整数边权及单位顶点权重，属于严格二分，不是允许不平衡的多路或超图模型。

源码为`src/combinatorial_solvers/partition/`。状态无关的两位／四位块库严格保持基数，
同色块之间没有交叉边，全部交换机会的并图必须连通。
四位块枚举16个模式，条件于当前基数后至多保留6个可行模式。
经典贪心和两位Metropolis用于收益归因。

已测面板包含四张32768顶点图，比较冷启动及共同KaFFPa暖启动，使用八条随机副本和45秒搜索窗口。
每顶点计一个物理更新p-bit，条件模式选择逻辑另计成本；自然孤立顶点仍参与全局平衡。
冷启动质量、继承暖起点和边际改善分别记录。
强经典对照KaHIP设置零不平衡、最终严格平衡修正及strong配置。

独立整数割值与平衡检查覆盖全部保存划分，报告初始化、分块、搜索、认证、完整耗时、
显存及主机内存。CPU调用截止后完成的结果仍保存，但不能改善预算内质量。
小规模精确参考不证明32K割的最优性，也不预设方法优势。

按[通用步骤](../reproduction.zh-CN.md)运行`partition`，合成示例不需要KaHIP。
基准和可执行文件边界见[外部基线](../external-baselines.zh-CN.md)。

## 参考来源

以下保留专题研究中使用的来源入口，包括背景和未实施对照。引用不代表复现，也不代表本次重新阅读全文。基线固定版本见外部基线清单。

- R1: [Restoring Sparsity in Potts Machines via Mean-Field Constraints](https://arxiv.org/html/2602.04200v2)
- R2: [期刊版](https://doi.org/10.1038/s44335-026-00097-x)
- R3: [Think Locally, Act Globally: Perfectly Balanced Graph Partitioning](https://arxiv.org/abs/1210.0477)
- R4: [KaHIP作者仓库](https://github.com/KaHIP/KaHIP)
- R5: [Walshaw Graph Partitioning Archive](https://chriswalshaw.co.uk/partition/)
- R6: [Quantum Annealing for Minimum Bisection Problem: A Machine Learning-based Approach for Penalty Parameter Tuning](https://arxiv.org/abs/2509.19005)
