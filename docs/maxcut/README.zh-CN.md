# 稀疏与稠密Max-Cut

[English](README.md) | 简体中文

实测见[RTX 5090报告](experiment-report.zh-CN.md)。对$s_i\in\{-1,1\}$最大化
$C=\sum_{i<j}w_{ij}(1-s_i s_j)/2$，
Ising能量为$\sum_{i<j}w_{ij}s_i s_j=\sum w-2C$；有符号权重必须明确此约定。

源码为`src/combinatorial_solvers/maxcut/`及其`dense/`子包。
稀疏路线包含着色热浴、Metropolis、平行回火、生成的32768变量图、Gset实例和MQLib对照，
自然孤立顶点与目标活跃变量分别记账。

稠密路线保留全部32768原变量，覆盖完全有符号SK图及边密度0.5的正权图，
实现支持精确块Gibbs、停滞同步更新、SCA、Onsager修正和离散模拟分岔dSB。
已验收C3在两张SK图上比较SCA／Onsager与dSB，在两张正权图上比较块Gibbs与dSB；
停滞同步更新不属于额外的C3实测臂。
dSB使用连续位置／动量，是p-bit数为零的经典对照；原生条件采样和同步启发式不能描述成同一平稳过程。

实现包含上三角压缩输入、增量场和独立整数证书。32768维FP32稠密矩阵本身占4 GiB，
还需状态及工作区。报告实测峰值显存、活跃变量、轨迹数和全部计时分项。
同时比较$2C-\sum w$及割值，避免稠密正权图的常数掩盖相对优化差距。

稠密确认使用四张图、两个种子和90秒窗口，保留匹配SCA／Onsager及同GPU dSB对照，
包括不利结果，不预设稳定优势。K2000已知最好值不是最优性证明；作者参考代码不进入本发行版。

按[通用步骤](../reproduction.zh-CN.md)运行`maxcut`或`maxcut-dense`。
修订FESTA及基准许可见[外部基线](../external-baselines.zh-CN.md)。
稠密问题是正式研究路线，稀疏性不是发布前提。

## 参考来源

以下保留专题研究中使用的来源入口，包括背景和未实施对照。引用不代表复现，也不代表本次重新阅读全文。基线固定版本见外部基线清单。

- R1: [原论文](https://arxiv.org/abs/1805.05217)
- R2: [MQLib作者库](https://github.com/MQLib/MQLib)
- R3: [Stanford Gset原始档案](https://web.stanford.edu/~yyye/yyye/Gset/)
- R4: [GPU-pSAv论文](https://www.nature.com/articles/s41598-025-90520-3)
- R5: [作者代码](https://github.com/nonizawa/GPU-pSAv)
- R6: [Programmable Probabilistic Computer with 1,000,000 p-bits](https://arxiv.org/abs/2606.25313v1)
- R7: [Hardware Acceleration of Probabilistic Computing for Max-Cut Problems](https://ieeexplore.ieee.org/document/11668769/)
- R8: [High-performance combinatorial optimization based on classical mechanics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323291/)
- R9: [DOI](https://doi.org/10.1126/sciadv.abe7953)
- R10: [Anechoic预印本v1](https://arxiv.org/abs/2610.13368v1)
- R11: [HTML正文](https://arxiv.org/html/2610.13368v1)
- R12: [作者代码](https://github.com/skethz/anechoic)
- R13: [原始K2000作者数据](https://github.com/hariby/SA-complete-graph/tree/WK2000)
- R14: [Edge-of-chaos enhanced quantum-inspired algorithm for combinatorial optimization](https://arxiv.org/abs/2508.17655)
