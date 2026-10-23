# 加权最大独立集

[English](README.md) | 简体中文

实测见[RTX 5090报告](experiment-report.zh-CN.md)。二元选择$x_i$对应
$E=-\sum_i w_i x_i+\lambda\sum_{(i,j)\in E}x_i x_j$。
正权重及$\lambda>\max_i w_i$保证全局最小值可行，不保证局部搜索高效。

每个顶点使用一个二元变量，没有辅助位。实测至32768顶点，包含着色单位热浴／Metropolis、
状态无关相邻两位Gibbs／Metropolis、贪心及单／并发CHILS对照。
联合更新用于检验局部势垒能否被跨越，不注入确定性答案。

源码为`src/combinatorial_solvers/mwis/`。整数证书重算权重与边冲突，原始物理态、可行最好态和修复解分别保存。
CHILS返回启发式参考，不是已证明最优值。四张确认图与两个算法种子只描述有限合成分布；
换设备复用它们不会产生新的图样本。分别报告链数、CPU工作进程、初始化、搜索、认证和完整成本。
这些结果不构成一般p-bit领先结论。

按[通用步骤](../reproduction.zh-CN.md)运行`mwis`示例，代码包含合成生成器和小规模穷举参考，
CHILS见[外部基线](../external-baselines.zh-CN.md)。

## 参考来源

以下保留专题研究中使用的来源入口，包括背景和未实施对照。引用不代表复现，也不代表本次重新阅读全文。基线固定版本见外部基线清单。

- R1: [KaMIS](https://github.com/KarlsruheMIS/KaMIS)
- R2: [CHILS](https://github.com/KarlsruheMIS/CHILS)
- R3: [作者预印本](https://arxiv.org/abs/2412.14198)
- R4: [SEA 2025论文](https://drops.dagstuhl.de/storage/00lipics/lipics-vol338-sea2025/html/LIPIcs.SEA.2025.22/LIPIcs.SEA.2025.22.html)
- R5: [Ebadi等，Quantum Optimization of Maximum Independent Set using Rydberg Atom Arrays](https://arxiv.org/abs/2202.09372)
- R6: [Efthymiou关于独立集的块动力学研究](https://arxiv.org/abs/1304.6666)
