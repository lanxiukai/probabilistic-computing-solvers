# 概率计算求解器

[English](README.md) | 简体中文

> [!NOTE]
> 本仓库绝大部分内容由 GPT-6-Astra 自动完成，尚未经过完整的人工审核。
> 我会不定期推进人工审核，但不承诺具体时间。

**概率计算（probabilistic computing）**把可控随机性用于采样、推断和优化。
**p-bit**是一种输出概率可调的经典随机比特；多个p-bit通过耦合表达问题的相关性与约束。
详见[概率计算与p-bit：原理、发展背景及适用边界](docs/probabilistic-computing.zh-CN.md)。

本仓库以经过审计的p-bit参与流程GPU模拟为主体，经典算法作为参照，混合流程中的经典支撑明确标注，并提供可复核证书。
仓库保留有利和不利结果，区分可行性、最优性、采样与实体硬件结论。

下表所有专题的GPU结果均为 **NVIDIA GeForce RTX 5090** 实测。

| 专题 | 科学与实际价值 | 实测范围与报告 |
|---|---|---|
| 整数因式分解 | 整数算法与算术约束研究 | 最终64–128-bit面板 · [报告](docs/factorization/experiment-report.zh-CN.md) |
| 三维EA自旋玻璃 | 无序物理与低能态搜索 | 16³／24³／32³ · [报告](docs/spin-glass/experiment-report.zh-CN.md) |
| TSP | 路径规划与排列约束 | 200–2103城GPU p-bit流程，最多33,662个二值节点 · [报告](docs/tsp/candidate-report.zh-CN.md) |
| 加权独立集 | 冲突资源选择与调度 | 原生模型至32768变量 · [报告](docs/mwis/experiment-report.zh-CN.md) |
| 加权Max-SAT | 配置、规划与约束推理 | 原生软子句，32768变量 · [报告](docs/maxsat/experiment-report.zh-CN.md) |
| Max-Cut | 二元决策与Ising优化 | 稀疏与完整稠密32768变量模型 · [报告](docs/maxcut/experiment-report.zh-CN.md) |
| LABS | 雷达／通信序列与低自相关 | 原生非周期高阶模型至32768长度 · [报告](docs/labs/experiment-report.zh-CN.md) |
| 平衡图二分 | 并行计算与网络分割 | 严格基数，32768顶点 · [报告](docs/partition/experiment-report.zh-CN.md) |
| LDPC码字搜索 | 通信／存储纠错 | 原生32766及二次32760更新位 · [报告](docs/ldpc/experiment-report.zh-CN.md) |
| 加权集合覆盖／命中集 | 测试集选择、资源分配与覆盖规划 | 1,000–47,311原始变量，128副本 · [报告](docs/set-cover/report.zh-CN.md) |
| 车间作业调度 JSSP | 制造排程、资源冲突与时序约束 | 216–37,500个二元编码条目，64–256副本 · [报告](docs/jssp/report.zh-CN.md) |
| 覆盖数组 | 组合软件测试与交互覆盖设计 | 108–31,500个二元编码条目，128副本 · [报告](docs/covering-array/report.zh-CN.md) |
| 图着色／列表着色 | 寄存器分配、冲突资源安排与组合数学 | 最大46,656存储位、28,026原始交互位，128链 · [报告](docs/graph-coloring/experiment-report.zh-CN.md) |
| 二次指派 QAP | 设施布局、相互作用指派与图对应 | 144–22,500个受约束二元条目，128／512副本 · [报告](docs/qap/experiment-report.zh-CN.md) |

各专题的单模型规模、耗时、并行数、峰值显存与主要限制见[实验总览](docs/benchmark-overview.zh-CN.md)。

此前组合优化面板在RTX 5090上完成**365次自然调用和35次另行工作量／原点回放**，
CPU对照在同一主机实测。已有因式分解和自旋玻璃数据保持原样：
因式分解保留52/54认证成功，其中128-bit为7/9；大规模自旋玻璃基态仍未知。新的TSP候选网络研究另增加28次自然调用与1次回放，七次p-bit调用均未改善前缀。

集合覆盖另外完成117次自然调用与4次回放：27次p-bit调用均改善贪心前缀，15次命中冻结目标；没有优于AC-CFT的调用。

```bash
uv sync --frozen
uv run --frozen python -m combinatorial_solvers.public_api --list
uv run --frozen python scripts/release/verify_core.py --panel mwis
```

从[复现指南](docs/reproduction.zh-CN.md)、[外部参考边界](docs/external-baselines.zh-CN.md)
或[仓库指南](docs/repository-guide.zh-CN.md)开始。
项目自有代码采用[MIT许可证](LICENSE)，保留的第三方声明与外部许可证继续适用。
