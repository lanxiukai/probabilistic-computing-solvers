# 低自相关二元序列

[English](README.md) | 简体中文

实测见[RTX 5090报告](experiment-report.zh-CN.md)。二元符号序列满足
$C_k=\sum_{i=0}^{N-k-1}s_i s_{i+k}$、
$E=\sum_{k=1}^{N-1}C_k^2$及merit factor $F=N^2/(2E)$。
研究对象是不施加偏斜对称等额外序列约束的非周期LABS，不是周期相关或峰值旁瓣最小化。

源码为`src/combinatorial_solvers/labs/`。原生高阶p-bit及Metropolis直接更新原序列，
由CPU整数／XOR和独立整数变换核验证书。FP64 FFT增益计算包含舍入检查及所选翻转的精确检查。
32768变量原生模型没有辅助p-bit，但算术、相关缓存及更新逻辑仍需资源，
不能等同于只有二次耦合的32768位硬件。

比较分开记录随机起点、Legendre构造、附加构造、确定性贪心修整、随机修整和GPU memetic tabu。
后者为项目依据论文实现，不是作者CUDA程序或完整吞吐量复现。
构造和确定性下降的成本及收益明确归属；同一序列长度的多个种子不是独立LABS实例。

认证完整序列、全部非周期时延、能量及merit factor，报告尝试／访问位置、完成轮次或代数、
截止删失、准备、搜索及完整耗时。长构造序列或局部最优不代表全局最优证明或量子优势。

按[通用步骤](../reproduction.zh-CN.md)运行`labs`。
小规模独立lMAts对照带GPLv2边界，本发行版不附带，见[外部基线](../external-baselines.zh-CN.md)。

## 参考来源

以下保留专题研究中使用的来源入口，包括背景和未实施对照。引用不代表复现，也不代表本次重新阅读全文。基线固定版本见外部基线清单。

- R1: [作者档案](https://github.com/borkob/git_labs)
- R2: [CSPLib prob005](https://www.csplib.org/Problems/prob005/)
- R3: [GPU memetic tabu v2](https://arxiv.org/abs/2504.00987v2)
- R4: [算法与结果正文](https://arxiv.org/html/2504.00987v2)
- R5: [原始memetic方法](https://doi.org/10.1016/j.asoc.2009.03.005)
- R6: [2017论文预印本](https://arxiv.org/abs/1406.5301)
- R7: [Advances in the merit factor problem for binary sequences v2](https://arxiv.org/abs/1205.0626v2)
- R8: [定义、定理2.1与极值参数](https://arxiv.org/html/1205.0626)
- R9: [Dual-step optimization](https://doi.org/10.1016/j.dsp.2025.105316)
- R10: [2024预印本正文](https://arxiv.org/html/2409.07222v1)
- R11: [小子系统优化预印本v1](https://arxiv.org/abs/2610.14361v1)
- R12: [模型、分解与改进机制](https://arxiv.org/html/2610.14361v1)
- R13: [QAOA scaling研究v2](https://arxiv.org/abs/2308.02342v2)
- R14: [PCE LABS v2](https://arxiv.org/abs/2506.17391v2)
- R15: [J-Brest/LABS作者数据](https://github.com/J-Brest/LABS)
