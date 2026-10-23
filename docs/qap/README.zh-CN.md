# 用受约束 p-bit 块求解二次指派

[English](README.md) | 简体中文

本专题用GPU模拟二次指派问题（QAP）的概率二元赋值变量。已验收的独立确认、开发证据与官方CPU参照见[实验报告](experiment-report.zh-CN.md)。自然基准端点为`tai150b`：150个指派对象，每模型持久保存22,500个one-hot二元变量，并行副本数量另列。

QAP 描述设施布局、图对应等具有相互作用的指派问题，将严格排列约束和对象间的配对成本结合。给定可行排列后容易计算其目标值；对困难实例证明全局最优是另一项任务。本实验只采用通用整数 QAPLIB 实例，不代表完整物理或工业工作流。[QAPLIB](https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/)

## 模型与转移

对非负整数矩阵 $A,B$，求解

$$C(\pi)=\sum_{i,j}A_{ij}B_{\pi(i),\pi(j)},\qquad \pi\in S_n.$$

持久二元状态满足 $x_{ia}=1$ 当且仅当 $\pi(i)=a$，每行、每列均恰有一个激活位。实现同时维护与完整二元状态等价的排列缓存，并核验二者一致。二元变量数为 $n^2$，可行状态数为 $n!$，这些变量受约束而非相互独立。GPU 存储位和模拟模型变量数不代表已经制造的 p-bit 硬件规模。

加入行、列约束的平方罚项可以得到 QUBO 表述；本次实测求解器直接在块转移中保持约束。因此 $n^2$ 描述所维护的可行二元模型，不代表已经测量了无约束的单比特 QUBO 机器，也不是独立自由度数量。

每步独立于当前状态选择两个或四个不同位置，枚举这些位置当前所占地点的 2 种或 24 种排列。接受模式的概率为

$$P(\sigma\mid x_{\bar S})=\frac{\exp[-\beta(E_\sigma-E_{\min})]}{\sum_\tau\exp[-\beta(E_\tau-E_{\min})]}.$$

这是可行 one-hot 块上的联合条件热浴，直接更新持久二元赋值并保持两类排列约束。整数局部能量包含块外全部相互作用和块内每个有序对，支持非对称项与对角项；无需显式展开 $n^4$ 耦合张量。固定温度转移具有条件 Gibbs 形式，循环温度用于优化。这里不主张已验证平衡分布、混合时间、量子优势或硬件能效。

候选优化实现复用精确块外相互作用表。设选中位置为 $i_a$，原先占据的地点为 $\ell_b$，则

$$T_{ab}=\sum_{k\notin S}\left(A_{i_a k}B_{\ell_b,\pi(k)}+A_{k i_a}B_{\pi(k),\ell_b}\right),$$
$$E_\sigma=\sum_aT_{a,\sigma(a)}+\sum_{a,b}A_{i_a i_b}B_{\ell_{\sigma(a)},\ell_{\sigma(b)}}.$$

对块宽$b$，这一推导把重复块外计算从$O(b!bn)$降为$O(b^2n)$，保留$O(b!b^2)$的块内项。变化仅在能量计算，不改变随机更新规则，原candidate实现继续可选。两组最终完整预热配对的全部20个非计时状态与RNG数组一致；[绑定源码的记录](../../data/qap/rtx5090/implementation-comparisons.json)分别保留搜索、完整runner及launcher比值。搜索仍包含随机数生成、主机观测、传输和恢复写盘，因此时间比不是独立CUDA内核加速比。

各副本从均匀打乱的排列开始。参考解不参与初始化、修复或搜索引导；流程没有经典搜索前缀或末端润色。相同块上的 Metropolis 和随机处理平局的最低能量转移用于辨别更新规则的作用，另有独立实现的单线程属性禁忌搜索作为经典对照。后者受 Taillard 方法启发，不声称严格复现原版鲁棒禁忌程序。[Taillard，1991](https://mistic.iict-heig-vd.ch/taillard/articles.dir/Taillard1991.pdf)

Metropolis 控制与热浴共用完整候选能量计算，目的是控制实现条件以比较转移规则，不是吞吐已经充分优化的 GPU 模拟退火。开发阶段选择热浴参数，控制组复用这些参数，未进行同等预算的独立调参。因此，相关结果需要结合独立禁忌搜索对照阅读，不能据此宣称优于最强经典 GPU 算法。独立实现的禁忌搜索也未获得同等调参预算，且未被验证为先进水平实现。

另有独立外部参照采用 Taillard 官方 2006 年 C 版 Ro-TS，保持公布的禁忌期限 `8*n` 与长期优先阈值 `5*n*n`。测量适配保留上游的移动、期限和增量逻辑，暴露六字 RNG 状态，在完整迭代后保存原始见证，并禁用参考目标提前停止。云软件门禁已在相同工作量下比较其返回见证和后续 64 个 RNG 值，与未改上游函数一致。Q3 已完成 18 次单线程 CPU 试验与三次工作量重放，全部 5,085 个保存见证通过全量审计。其经典结果与 p-bit 成绩分开，也不用于选择原生确认参数。[作者官方源码页](https://mistic.iict-heig-vd.ch/taillard/)

## 核验与边界

测试用含非零对角项的非对称小矩阵，逐步比对完整原始目标枚举；额外的 150 阶案例检查较大的 CUDA 启动布局。保存的当前、历史最佳和时限内最佳排列均由完整矩阵整数计算核验。另检查 one-hot 一致性、周期状态、目标轨迹、源码与输入清单以及固定工作量重放。

实现要求矩阵元素小于 $2^{62}$，并满足保守界 $n^2\max(A)\max(B)<2^{52}$，从而保证向量化完整目标 `int64` 证书精确，并使增量运算不溢出。每个副本组的正式最佳结果还用无界 Python 整数单独核验。批量核验的加速计入核验工作，不写成 p-bit 采样加速。

搜索时间包含随机数生成、同步、观测和周期恢复写盘。GPU准备／预热编译、最终核验与归档成本另列，并包含在完整runner时间中；CPU参照二进制的一次性构建在逐调用计时之外。跨过搜索截止的chunk保留为证据，但不计入时限内质量。首次到达参考值的时间是chunk观测上界。一个完整的种子副本组才是一项独立试验；副本数描述组内搜索多样性，重放不增加试验分母。

规模按未改动的公开基准调整，`tai150b` 使用 22,500 个变量，不人为制造 32K 实例。实测模型规模、访问变量数、可行结果、参考命中和已证明收益分别报告。

## 基准与复现

`configs/qap/benchmarks.json` 记录原始下载地址、SHA-256、参考性质及排列约定。`tai40a` 的参考排列从 0 开始编号，`tai80a` 需要取逆；最终均用原始整数目标核对。Nugent 参考值在目录中标为最优，Taillard 数值只作为历史可行参考，不当作新证明的最优值或已核验的当前纪录。

```bash
uv run --frozen python scripts/qap/fetch_benchmarks.py --output data/qap/qaplib
```

仅在复现自有 CPU 对照调用时构建该程序：

```bash
uv run --frozen python scripts/qap/prepare_reference.py --output data/qap/baselines/attribute-tabu
```

可选的官方 Ro-TS 参照具有独立的**非商业使用与致谢条件**。下载的上游原件、插桩后全文和二进制均不进入本仓库公开源码分发，也不受本仓库 MIT 许可覆盖。阅读条件后，才选择构建这一外部程序：

```bash
uv run --frozen python scripts/qap/prepare_rots.py \
  --output data/qap/baselines/taillard-rots \
  --accept-noncommercial-conditions
```

这里仅分发自有下载／构建和测量适配。原文件固定哈希，完整上游字节及编译器／构建标识保存在使用者的本地实验归档中；使用上游 32 位实现之前核验原始整数边界。这一外部对照没有 p-bit 贡献。

无需启动搜索，即可分别核验三份已验收台账的原始整数见证：

```bash
uv run --frozen python scripts/qap/verify_release.py \
  --repository . --calls data/qap/rtx5090/calls.json
uv run --frozen python scripts/qap/verify_release.py \
  --repository . --calls data/qap/development-q1c/calls.json
uv run --frozen python scripts/qap/verify_release.py \
  --repository . --calls data/qap/external-rots/calls.json
```

此检查需要下载后的基准原始字节和发布核心记录，不需要任一 CPU 可执行程序或外部 Ro-TS 构建。复现单个原生种子副本组时，先准备新的隔离源码／输入快照，再执行并核对封存文件清单：

```bash
uv run --frozen python scripts/qap/experiment.py prepare \
  --config data/qap/rtx5090/configs/q2-nug30-heatbath-s2026102411.json \
  --run-id reproduce-qap-nug30-s2026102411
uv run --frozen python scripts/qap/experiment.py run \
  --run-dir artifacts/runs/qap/reproduce-qap-nug30-s2026102411
uv run --frozen python scripts/qap/experiment.py verify \
  --run-dir artifacts/runs/qap/reproduce-qap-nug30-s2026102411
```

再次执行应使用新的 run ID。墙钟限时的复现可能完成不同工作量，因此相同种子本身不保证最终数组一致。另行发布的 `replay-q2-nug30-heatbath-s2026102411.json` 配置固定原工作量及有效质量边界，可配合另一个新 run ID 在已验证环境中重放。复现和重复计时不增加独立确认试验数。云实例控制协议和私有源码历史不属于公开接口。

三份台账分别保留Q2的81个、Q1c的12个及官方参照的21个记录，独立确认、调参、实现控制和回放分开标注。公开包采用已执行的Q3共享源码版本：六个模块及原生runner分支也与Q2／Q1c一致，runner另增加外部参照分发。九次公开源码回放均匹配原始非计时数组。原始计时与执行版本的绑定见[来源记录](../../provenance/qap/release.json)。CPU构建输入哈希对应原测量产物；编译器、主机和路径可以改变重建产物，但原始源码身份固定。

## 相关工作与软件

以下来源核对于 2026-10-23。其性能主张尚未在本专题复现，不跨硬件、实例面板和计时定义直接排名。

- [Lehigh QAPLIB](https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/)：输入及参考见证。此次核对时 Montreal 主站不可用，发布采用固定哈希的镜像原件。
- É. Taillard，[Robust taboo search for the quadratic assignment problem](https://mistic.iict-heig-vd.ch/taillard/articles.dir/Taillard1991.pdf)，*Parallel Computing* 17，443–455（1991），[DOI](https://doi.org/10.1016/S0167-8191(05)80147-4)：经典方法背景与独立对照实现的设计依据。
- É. Taillard，[官方标准 C 版 Ro-TS 源码](https://mistic.iict-heig-vd.ch/taillard/codes.dir/tabou_qap2.c)，标注 2006-03-14：单独下载的外部实现，采用非均匀期限；其非商业条件与自有适配的许可分开。
- G. Paul，[GPU 模拟退火求解 QAP](https://arxiv.org/abs/1208.2675v1)，2012 年预印本：早期 GPU QAP 退火，本专题不主张首次使用 GPU 求解 QAP。
- C. Novoa、A. Qasem，[GPU 加速的 QAP 并行求解](https://arxiv.org/abs/2307.11248v1)，2023 年预印本：GPU 2-opt 与禁忌搜索实现，尚未在当前设备复现其性能。
- C. Duffee 等，[P-dit Probabilistic Ising Machine for Solving the Quadratic Assignment Problem](https://arxiv.org/abs/2605.24408v1)，2026 年预印本：直接相关的概率排列方法，其副本交互和更新方式与这里的固定大小条件块不同。
- Y. Pan 等，[Learning to Solve the Quadratic Assignment Problem with Warm-Started MCMC Finetuning](https://arxiv.org/abs/2604.20109v1)，2026 年预印本：PLMA 结合学习式排列模型与 MCMC 适应，未来同口径比较需计入训练和适应成本。
- M. Kuramata、R. Katsuki、K. Nakata，[量子退火与比特翻转启发式求解稀疏 QAP](https://arxiv.org/abs/2012.10135v3)，ICIEA 2021：量子退火与经典修复背景，不同约束编码及硬件不能仅凭规模比较性能。
- G. Yan 等，[Rethinking Parity Check Enhanced Symmetry-Preserving Ansatz](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7798f18a1e91aff7ad1df4841d6a01a8-Abstract-Conference.html)，NeurIPS 2024：摘要级变分量子算法硬约束背景，含 QAP，尚未复现。
- [Python](https://www.python.org/)、[NumPy](https://numpy.org/)、[PyTorch](https://pytorch.org/)、[Triton](https://github.com/triton-lang/triton)、[CUDA](https://docs.nvidia.com/cuda/)、[GCC](https://gcc.gnu.org/)、[uv](https://docs.astral.sh/uv/)、[pytest](https://docs.pytest.org/) 和 [Ruff](https://docs.astral.sh/ruff/)：实现、环境和检查工具，具体版本以发布锁文件和实测环境记录为准。
