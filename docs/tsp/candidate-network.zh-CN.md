# 对称TSP的候选弧p-bit网络

[English](candidate-network.md) | 简体中文

本扩展的主体是**p-bit参与的TSP求解流程**：经典初始化、对受限assignment松弛进行GPU模拟的二值p-bit热浴更新、经典读出及tour认证。独立LKH与GA-EAX作为评价该流程的参照求解器，其最优tour属于经典对照结果，不计为p-bit成果。[RTX 5090实测](candidate-report.zh-CN.md)覆盖七个实例，未观察到相对前缀的改进；[旧150城测量](experiment-report.zh-CN.md)保留CPU选择器与GPU局部搜索的原范围。

## 流程与贡献归因

| 阶段 | 实现 | 报告内容 |
|---|---|---|
| 初始化 | 经典LKH前缀及候选弧构建 | 起始tour及其长度 |
| p-bit更新 | GPU二值热浴退火 | 原始位态、能量、度数可行性及采样成本 |
| 读出 | 经典匹配、子环修复及CUDA整数2-opt | 完整候选tour及读出成本 |
| 保留与认证 | 与前缀比较并验证整数tour代价 | 最终质量、相对前缀改进及完整成本 |

公开的方法是混合流程。LKH初始化与经典读出是支撑组件，独立LKH与GA-EAX调用是对照。混合流程的最终结果与实测p-bit阶段贡献分别报告；保留已经最优的前缀不构成p-bit改进。

## 模型与物理节点口径

每城保留16个最近出邻居，以及实际LKH前缀tour的两个方向。每条候选有向弧对应一个二值变量。用$r_i,q_j$表示选中出度／入度，$c_e$为归一化整数距离，罚项为$\lambda$：

$$E(x)=\sum_e c_ex_e+\lambda\sum_i(r_i-1)^2+\lambda\sum_j(q_j-1)^2.$$

常数为$2n\lambda$，线性系数为$c_e-2\lambda$，共享出端或入端的两条弧具有$2\lambda$二次耦合。随机辅助变量为零；单模型节点数等于候选弧数，8个副本另列。每节点映射一个物理p-bit，假定硬件原生支持这些稀疏耦合；控制器算术、权值存储与连线是额外资源。没有测量实体器件、嵌入开销或能耗。

一进一出仍允许子环，候选图未必包含完整图的最优tour，有限罚项也不是可行性证明。因此它不是完整TSP的Ising编码。

## GPU更新与经典读出

二值条件概率为

$$\Delta_e E=c_e+2\lambda(r_i+q_j-2x_e-1),\qquad
P(x_e\leftarrow1)=\sigma(-\beta\Delta_e E).$$

同色弧不共享出端或入端；每个持久Triton block拥有一个副本，颜色间使用屏障。度数是int32，局域场与概率是float32。每个sweep的Philox种子及副本／弧计数器标识随机抽样。退火用于优化，不声明得到TSP平衡样本。

读出对位态偏置代价求稀疏经典匹配，随后拼接子环并执行CUDA整数2-opt。修复可离开候选图。原始状态可行性、读出质量与保留前缀分别观察；匹配、修复、下降、认证和输出都计入成本。对照包括原代价匹配、随机位加相同读出、同前缀ILS、独立LKH与独立GA-EAX。

## 复现输入与工具

[冻结协议](../../data/tsp/candidate-network/protocol.json)列出全部实例、原输入哈希、参考最优值、种子与参数。从其上游链接获取TSPLIB文件，核对解压后的原始字节。此处不再分发基准坐标、原生源码包或可执行文件。LKH与GA-EAX由作者保留权利并供研究使用，本项目MIT许可证不重新许可这些外部程序。

本项目编写的构建工具接受显式取得的固定源码包，保留原件并记录编译器与产物身份。p-bit流程省略`--eax-archive`；仅在构建独立经典参照时提供它。GA-EAX通过`TSP_EAX_SEED`调用作者已有的显式种子函数，搜索与停止规则保持不变。应在目标CPU重建并先记录该构建，再进行测量。

```bash
uv run --frozen python scripts/tsp/prepare_native.py \
  --lkh-archive downloads/LKH-3.0.13.tgz --output /tmp/tsp-native-NEW
uv run --frozen python scripts/tsp/run_candidate.py \
  --plan data/tsp/candidate-network/protocol.json \
  --case pbit-5090-pcb442-pbit-42001 \
  --instances-directory data/external/tsp \
  --baseline-directory /tmp/tsp-native-NEW --output /tmp/tsp-case-NEW
```

第二条命令创建新的隔离运行，不加载已发表的最好tour，也不复用完成目录。已验收核心数据可用`scripts/tsp/verify_candidate.py --core PATH --instances-directory PATH`重新计算整数tour证书及位态能量，无需运行求解器。实测核心目录为`data/tsp/candidate-network/rtx5090`，完整私有操作归档不属于公开核心数据。

## 经典对照证书

以下证书记录经典对照的求解能力，属于参照材料，不是p-bit流程的结果或主要复现目标。[CPU计算所得的u1817证书包](../../data/tsp/candidate-network/u1817-optimum.json)包含长度57,201的实际tour，以及全部四次自然调用和一次回放，包括一次确认未命中。它提供与已发表最优长度一致的可行tour，没有重新计算下界证明。证书不会作为求解器起始tour加载，也不包含基准坐标。

```bash
uv run --frozen python scripts/tsp/verify_candidate.py \
  --certificate data/tsp/candidate-network/u1817-optimum.json \
  --instances-directory data/external/tsp
```

## 参考来源

- [GA-EAX 1.0作者代码，固定提交](https://github.com/nagata-yuichi/GA-EAX/tree/015dfbe9f267230f78787bd244af393ffc018900)，实现Nagata与Kobayashi的2013算法。
- [LKH-3作者分发页](http://webhotel4.ruc.dk/~keld/research/LKH-3/)，固定3.0.13版本。
- [TSPLIB95](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)及[已证明的最优长度](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html)。
- [SciPy稀疏最小权完美二分匹配](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.min_weight_full_bipartite_matching.html)。
- [Triton](https://triton-lang.org/main/index.html)及[PyTorch](https://pytorch.org/docs/stable/index.html)，GPU执行软件。
- [NumPy](https://numpy.org/doc/stable/)及[Matplotlib](https://matplotlib.org/stable/)，用于数组核验和可重绘图表。
- [原TSP来源清单](README.zh-CN.md)，含本扩展未复现的Ising编码、约束温度交换、强经典方法及硬件方向。

引用不代表复现了对应算法或硬件结论。
