# RTX 5090严格平衡图二分

[English](experiment-report.md) | 简体中文

冻结P2包含四张32768顶点图上的**40次自然调用及五次工作量回放**。
约束为两侧顶点数严格各半，采用单位顶点权重、正整数边权。
随机方法使用八个副本及45秒搜索窗口。KaFFPa采用strong配置、
零不平衡和最终严格平衡修正。

NVIDIA GeForce RTX 5090 (32607 MiB); AMD Eng Sample: 100-000001537-04 (quota 17.56 vCPU); Python 3.14.6; PyTorch 2.13.0+cu130; CUDA 13.0; driver 580.178.04.

本轮设备确认复用此前已测的图和种子。

| 图 | 变体 | 调用数 | 平均割值 | runner与已测初始化之和的平均秒 |
|---|---|---:|---:|---:|
| delaunay32k | greedy4-warm | 1 | 343.000 | 3.602 |
| delaunay32k | hb4 | 2 | 5615.500 | 127.956 |
| delaunay32k | hb4-warm | 2 | 343.000 | 48.498 |
| delaunay32k | initial | 1 | 343.000 | 0.642 |
| delaunay32k | kaffpa-restarts | 2 | 323.000 | 46.832 |
| delaunay32k | metro2 | 2 | 8494.000 | 128.606 |
| random32k-a | greedy4-warm | 1 | 109420.000 | 46.264 |
| random32k-a | hb4 | 2 | 111897.000 | 232.763 |
| random32k-a | hb4-warm | 2 | 109420.000 | 91.541 |
| random32k-a | initial | 1 | 109420.000 | 39.160 |
| random32k-a | kaffpa-restarts | 2 | 109420.000 | 107.716 |
| random32k-a | metro2 | 2 | 115939.500 | 254.177 |
| random32k-b | greedy4-warm | 1 | 109685.000 | 35.538 |
| random32k-b | hb4 | 2 | 111783.500 | 250.515 |
| random32k-b | hb4-warm | 2 | 109685.000 | 78.390 |
| random32k-b | initial | 1 | 109685.000 | 28.357 |
| random32k-b | kaffpa-restarts | 2 | 109567.000 | 81.629 |
| random32k-b | metro2 | 2 | 115902.000 | 247.315 |
| rgg32k | greedy4-warm | 1 | 234.000 | 4.727 |
| rgg32k | hb4 | 2 | 6708.500 | 198.576 |
| rgg32k | hb4-warm | 2 | 234.000 | 49.688 |
| rgg32k | initial | 1 | 234.000 | 0.879 |
| rgg32k | kaffpa-restarts | 2 | 182.000 | 46.896 |
| rgg32k | metro2 | 2 | 10340.500 | 186.561 |

四位热浴在4/4张图上的冷启动平均割值低于两位Metropolis，
在0/4张图上低于KaFFPa重启对照。
前一比较同时改变块大小和接受规则，不能单独归因于热浴核。
包括确定性下降在内的四位暖启动改进，在0/12次调用中
降低了共同起点的割值。

原生两位／四位块保持基数，块库不依赖当前状态，同色块无交叉边，交换机会并图连通。
全部保存划分通过独立整数XOR割值及严格平衡检查，RGG孤立顶点仍参与全局平衡。

冷启动优化、继承暖起点质量和随机更新边际收益分别评价。
暖启动表格加上另行测得的共同初始化成本；这是实测分项之和，不是独立测得的通信延迟。
状态输出和认证计入完整成本。本面板不证明32K割全局最优，
也不证明相对多层次、GPU、多路或超图划分方法的一般优势。

![相对KaFFPa重启对照的割值](figures/rtx5090.png)

[方法](README.zh-CN.md) · [核心数据](../../data/partition/rtx5090/calls.json) ·
[核验](../../data/partition/rtx5090/verification.json) · [复现](../reproduction.zh-CN.md)

## 参考来源与软件

下列入口保留研究中使用的背景及未实施对照。引用不表示复现；
方法说明中的版本与获取边界继续适用，本次发布没有重新进行一次全文文献审查。

- R1: [Restoring Sparsity in Potts Machines via Mean-Field Constraints](https://arxiv.org/html/2602.04200v2)
- R2: [https://doi.org/10.1038/s44335-026-00097-x](https://doi.org/10.1038/s44335-026-00097-x)
- R3: [Think Locally, Act Globally: Perfectly Balanced Graph Partitioning](https://arxiv.org/abs/1210.0477)
- R4: [https://github.com/KaHIP/KaHIP](https://github.com/KaHIP/KaHIP)
- R5: [Walshaw Graph Partitioning Archive](https://chriswalshaw.co.uk/partition/)
- R6: [Quantum Annealing for Minimum Bisection Problem: A Machine Learning-based Approach for Penalty Parameter Tuning](https://arxiv.org/abs/2509.19005)
- S1: [Python](https://www.python.org/)
- S2: [NumPy](https://numpy.org/)
- S3: [PyTorch](https://pytorch.org/)
- S4: [Triton](https://github.com/triton-lang/triton)
- S5: [CUDA](https://docs.nvidia.com/cuda/)
- S6: [SciPy](https://scipy.org/)
- S7: [Matplotlib](https://matplotlib.org/)
- S8: [PyYAML](https://pyyaml.org/)
- S9: [uv](https://docs.astral.sh/uv/)
- S10: [GCC](https://gcc.gnu.org/)
- S11: [pytest](https://docs.pytest.org/)
- S12: [Ruff](https://docs.astral.sh/ruff/)
