# 算术引导的稀疏 Ising 因式分解

[English](README.md) | 简体中文（译文）

将算术引导提案与 Metropolis–Hastings 采样结合的稀疏 Ising 乘法器 GPU 实现。
严格成功要求实际接受态给出正确的非平凡整数因子，且完整与条件模型的 FP64 能量均为零。

最终RTX 5090面板已验收：52/54次严格成功，其中128-bit为7/9，保留两次事件上限未命中及全部控制。GPU模拟不等于实体p-bit实测；受限实例不能证明通用128-bit能力或经典分解速度优势。

- [实验报告与完整参考文献](experiment-report.zh-CN.md)
- [算法说明](method.zh-CN.md)与[冻结协议](protocol.zh-CN.md)
- [复现指南](reproduction.zh-CN.md)与[源码指南](source-guide.zh-CN.md)
- [发布状态](readiness.zh-CN.md)

Python 包名保持 `sparse_factorization`，命令为 `factor-bench`。
多项目结构见[仓库指南](../repository-guide.zh-CN.md)。
