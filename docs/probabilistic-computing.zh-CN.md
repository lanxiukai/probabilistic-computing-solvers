# 概率计算与p-bit

[English](probabilistic-computing.md) | 简体中文

概率计算（probabilistic computing）把可控的随机性作为计算的一部分。
系统可以探索一组状态形成的概率分布，通过样本估计概率、补全已知条件下的信息，
或者寻找代价较低的解。本文重点介绍由**概率比特p-bit**组成的网络。

这是一份模型与发展背景介绍。本仓库提供GPU模拟和经典对照，具体实测范围与限制见
[实验总览](benchmark-overview.zh-CN.md)。

## 1. p-bit是可调偏置的随机比特

p-bit输出一个二元值，输入可以调节它取不同值的概率。理想p-bit在零偏置下，
两个状态出现的时间比例相同；正、负偏置使其中一个状态更常出现，但不必将它固定。
把多个p-bit连接起来，各自的概率就会依赖其它状态，从而表达相关性和约束。[P3](#p3)

采用双极性状态$m_i\in\{-1,+1\}$时，一种常用的理想更新规则使用每次新产生、彼此独立的均匀随机数：

$$
m_i^{\mathrm{new}}=\operatorname{sign}\!\left[\tanh I_i-r_i\right],
\qquad r_i\sim\mathcal U(-1,1).
$$

因此，

$$
\Pr(m_i^{\mathrm{new}}=+1\mid I_i)
=\frac{1+\tanh I_i}{2}
=\frac{1}{1+\exp(-2I_i)}.
$$

无量纲输入$I_i$控制的是概率，每次输出仍是一个确定的二元值。也可以使用
$x_i=(m_i+1)/2\in\{0,1\}$编码。这是数学参考模型；实际器件需要测量输入响应曲线，
输出还可能存在时间相关性。[P6](#p6)

~~~mermaid
flowchart LR
    R["Random source"] --> P["Tunable p-bit"]
    B["Bias / local field"] --> P
    P --> S["Binary state"]
    S --> C["Coupling / constraint network"]
    C --> B
~~~

图中表示功能关系，不对应某一种具体晶体管电路。软件伪随机数发生器、数字逻辑或
有噪声的物理器件都可以提供随机输入。

## 2. 从局部更新到能量模型

一种有用的模型为联合状态指定能量：

$$
E(\mathbf m)=-\sum_{i<j}J_{ij}m_i m_j-\sum_i h_i m_i,
\qquad J_{ij}=J_{ji},\quad J_{ii}=0.
$$

第$i$位的局部场为

$$
H_i=h_i+\sum_{j\ne i}J_{ij}m_j,\qquad I_i=\beta H_i.
$$

固定其它位时，$m_i=+1$与$m_i=-1$的能量差为$-2H_i$，代入条件Boltzmann概率
便得到前面的sigmoid。在固定、有限的$\beta\ge0$下，采用适当扫描顺序且动力学遍历时，
精确单点热浴更新具有平稳分布

$$
\pi(\mathbf m)=Z^{-1}\exp[-\beta E(\mathbf m)].
$$

常数$Z$用于归一化概率。这把随机二元网络与统计力学、Boltzmann机联系起来。这里的$\beta$是算法的逆温度参数，
不一定等于器件的物理温度所对应的逆温度。[P2](#p2)、[P6](#p6)

**更新顺序是算法的一部分。** 同时用旧局部场更新彼此耦合的位，一般不会得到相同的
Gibbs采样器。条件独立的变量组可以一起更新；有相互作用的块需要正确的联合更新或
有依据的校正。硬件延迟、有限精度、相关随机输入及过期的边界状态，也可能改变分布。
因此要把并行架构和动力学一起评价。[P5](#p5)、[P8](#p8)

本仓库并非所有求解器都使用这一单点规则，也包含Metropolis-Hastings提案、条件块、
温度交换、种群方法和经典修复阶段。实际过程由各专题的方法与协议文档定义。

## 3. 小例子：正向与反向使用一个AND约束

设$x,y,z\in\{0,1\}$，要求$z=x\land y$。作为说明性的构造，罚函数

$$
E_{\mathrm{AND}}(x,y,z)=\lambda(xy-2xz-2yz+3z),\qquad \lambda>0
$$

恰好在合法真值表各行取零，在其余四种赋值上为正：

| $x$ | $y$ | 合法$z$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

固定$x$与$y$后，剩余变量倾向于合法输出；改为固定$z=0$，网络可以探索三个相容输入对；
固定$z=1$则偏向$(x,y)=(1,1)$。这就是**可逆逻辑（invertible logic）**的基本思路：
对约束网络的不同端口施加条件，会改变需要推断的变量。
上面的例子是直接代数构造，不是器件实测。[P3](#p3)

有限温度和有限罚强下仍可能出现非法状态。更大的乘法器或调度模型还会引入大量相互作用的
约束，因此一个小逻辑门的正确性不代表已经获得快速因式分解或通用优化优势。

## 4. 采样、推断与优化回答不同的问题

| 任务 | 期望结果 | 需要检查什么 |
|---|---|---|
| 采样 | 服从指定分布的状态 | 平稳性、混合、自相关和分布误差 |
| 推断 | 给定观测后的条件概率或可能赋值 | 模型、条件、估计量和不确定性 |
| 优化 | 可行且目标较好的赋值 | 原始约束、目标、质量门槛和完整成本 |

优化时，降低有效温度可以增加低能态的概率，但有限时间退火仍可能被困住。温度交换与
联合更新尝试改善探索；强经典方法则可能利用通用能量模型没有充分利用的问题结构。
[P1](#p1)、[P2](#p2)

推断中的最可能完整赋值，与每个变量各自最可能的取值，并不是同一个量。
采样时，找到一个低能态也不能证明分布正确。同一网络可以服务不同任务，但需要分别定义
成功标准。

## 5. 发展背景与实现方式

算法基础早于p-bit这个名称：Monte Carlo方法通过随机转移研究分布，Boltzmann机则把
随机网络与学习联系起来。现代p-bit研究进一步关注：把这些基本操作直接融入架构，
能否降低反复生成随机数、计算局部场和更新状态的成本。能效是需要实测的硬件研究目标，
不能由“概率计算”这一名称推得。[P1](#p1)、[P2](#p2)、[P6](#p6)

| 时期 | 代表性进展 | 意义 |
|---|---|---|
| 1953 | Metropolis等人的Monte Carlo工作 [P1](#p1) | 通过依赖能量的随机转移采样构型 |
| 1985 | Ackley、Hinton与Sejnowski的Boltzmann机学习 [P2](#p2) | 在随机网络中学习并做条件补全 |
| 2017 | 随机p-bit与可逆逻辑 [P3](#p3) | 将可调随机比特抽象与逻辑约束连接 |
| 2019 | 基于随机磁隧道结的因式分解实验 [P4](#p4) | 小型耦合概率系统的实体概念验证 |
| 2022 | 稀疏Ising机架构 [P5](#p5) | 研究并行更新及连接结构与计算的关系 |
| 2025 | 使用电压控制MTJ熵源的集成电路 [P7](#p7) | 将磁性随机源与数字计算结合 |
| 2026 | 报告1,000,000个p-bit的联网FPGA系统 [P8](#p8) | 研究分布式规模及通信引入的精度取舍，仍为预印本 |

这是一条精选阅读路线，不是完整编年史。2026年的条目是预印本，其规模或性能不属于本仓库
结果。比较系统前，需要对应器件数、连接结构、辅助复制变量、副本和目标精度。

常见实现路线包括：

- **GPU／CPU软件：** 数值概率、伪随机采样和显式更新核。本仓库测量的就是这一实现层次。
- **数字FPGA／ASIC：** 可编程状态、随机源、局部场累加和更新控制。一个数字p-bit可能需要
  多个逻辑元件。
- **随机器件及配套电路：** 例如有读出与偏置控制的涨落磁隧道结；也有设计利用电压驱动的
  磁性随机行为，为数字逻辑提供熵源。这些是不同的电路架构，器件数量不能直接互换。
  [P4](#p4)、[P7](#p7)

即使随机元件很小，耦合存储与通信、校准、更新时序和读出仍可能主导系统成本。

## 6. 普通比特、p-bit与量子比特

| 单元 | 操作层面的描述 |
|---|---|
| 数字比特 | 由逻辑存储或操作的经典二元值 |
| p-bit | 概率可控的经典随机二元输出 |
| 量子比特 | 具有振幅与相位、可进行相干操作并以概率方式测量的量子态 |

p-bit网络使用经典概率，其涨落不提供量子相干、干涉或纠缠。经典p-bit方法可以通过适当的
数学映射模拟某些量子模型，但这不使其硬件成为量子计算机。同样，软件模拟p-bit也没有
复现磁性电路的能耗。[P6](#p6)

## 7. 怎样阅读一项结果

有意义的比较是匹配质量与资源后的完整任务，包括实例准备、编码、搜索、传输、修复、
验证以及训练或调参成本。高翻转速率本身不代表能更快找到好解。大型编码网络可能包含
辅助变量，多个小问题副本也不等于一个更大的耦合实例。

本仓库同时保留随机方法有收益和强经典对照更好的情况，不据此声称p-bit普遍领先、
量子优势或通用硬件能效优势。可先读[实验总览](benchmark-overview.zh-CN.md)，
再查看专题报告与[复现说明](reproduction.zh-CN.md)。

## 参考来源与阅读范围

来源核对日期为2026-10-21。前面的公式与AND例子属于解释性推导；历史描述依据下列来源，
出版者摘要和预印本只支持所注明的范围。引用硬件工作不表示在本仓库完成了复现。

<a id="p1"></a>
**P1.** N. Metropolis et al. *Equation of State Calculations by Fast Computing
Machines*. Journal of Chemical Physics 21, 1087–1092 (1953).
[DOI](https://doi.org/10.1063/1.1699114) ·
[论文副本](https://people.umass.edu/bvs/The_1953_paper.pdf)。
用于Monte Carlo采样背景。

<a id="p2"></a>
**P2.** D. H. Ackley, G. E. Hinton, and T. J. Sejnowski.
*A Learning Algorithm for Boltzmann Machines*. Cognitive Science 9, 147–169 (1985).
[作者托管论文](https://www.cs.toronto.edu/~hinton/absps/cogscibm.pdf)。
用于随机网络能量、平衡、条件化与学习。

<a id="p3"></a>
**P3.** K. Y. Camsari, R. Faria, B. M. Sutton, and S. Datta.
*Stochastic p-Bits for Invertible Logic*. Physical Review X 7, 031014 (2017).
[出版者页面](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.7.031014)。
用于p-bit抽象与逻辑网络条件化。

<a id="p4"></a>
**P4.** W. A. Borders et al. *Integer factorization using stochastic magnetic
tunnel junctions*. Nature 573, 390–393 (2019).
[出版者页面](https://www.nature.com/articles/s41586-019-1557-9)。
阅读出版者摘要与扩展数据说明；属于实体概念验证，不用于推出通用速度或能效优势。

<a id="p5"></a>
**P5.** N. A. Aadit et al. *Massively parallel probabilistic computing with
sparse Ising machines*. Nature Electronics (2022).
[DOI](https://doi.org/10.1038/s41928-022-00774-2) ·
[作者预印本v2](https://arxiv.org/abs/2110.02481v2)。
用于架构与并行背景，不直接采纳论文的加速比。

<a id="p6"></a>
**P6.** S. Chowdhury et al. *A full-stack view of probabilistic computing with
p-bits: devices, architectures and algorithms*. IEEE Journal on Exploratory
Solid-State Computational Devices and Circuits (2023).
[DOI](https://doi.org/10.1109/JXCDC.2023.3256981) ·
[作者稿v3](https://arxiv.org/abs/2302.06457v3)。
阅读更新模型与实现路线相关章节。

<a id="p7"></a>
**P7.** *An integrated-circuit-based probabilistic computer that uses
voltage-controlled magnetic tunnel junctions as its entropy source*.
Nature Electronics 8, 784–793 (2025).
[出版者页面](https://www.nature.com/articles/s41928-025-01439-6)。
阅读出版者摘要，用于架构例子，不背书性能比较。

<a id="p8"></a>
**P8.** N. A. Aadit et al. *Programmable Probabilistic Computer with
1,000,000 p-bits*. arXiv:2606.25313v1 (2026)，预印本。
[作者预印本](https://arxiv.org/abs/2606.25313v1)。
摘要级了解联网FPGA及通信／精度取舍；本仓库未复现硬件或独立验证其性能。
