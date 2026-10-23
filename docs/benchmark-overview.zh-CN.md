# 已验收实验总览

[English](benchmark-overview.md) | 简体中文

本页汇总已验收的**RTX 5090实测**，没有新增实验或其它设备的换算值。
十四个专题中，Max-Cut分别展示稀疏与稠密面板。各行的问题规模、目标、方法和停止规则不同，
耗时不能直接用来排列问题难度或求解器优劣。概念背景见
[概率计算与p-bit](probabilistic-computing.zh-CN.md)。

## 范围、资源与结论

表格选取各专题已公开的最大单模型规模。耗时通常是含准备、搜索、输出与认证的完整runner
均值，例外在表中注明。显存取所选方法／规模面板中已记录的单次调用峰值的最大值，
包含其全部并行状态。

**A**表示PyTorch的allocated峰值，**R**表示reserved峰值（含分配器缓存）。
二者是不同的统计量，均不等于整卡占用或理论最小显存需求。

| 专题与所选方法 | 单模型规模 | RTX 5090耗时、并行状态及峰值显存 | 主要结论与边界 |
|---|---|---|---|
| [整数因式分解](factorization/experiment-report.zh-CN.md) | 128-bit乘积；32448总位／32256自由位 | 完整调用中位约34.5分钟；32副本；2.086 GiB R | 128-bit认证成功7/9，失败保留在分母中 |
| [TSP，候选网络p-bit流程](tsp/candidate-report.zh-CN.md) | 2103城；33,662个候选弧变量 | 固定种子单次准备＋进程退出17.047秒；8副本；36.30 MiB A | 前缀改进0/7，终点差距0.2548%；连通性由经典修复处理，没有嵌入模型 |
| [三维EA自旋玻璃](spin-glass/experiment-report.zh-CN.md) | 32768自旋，32³ | 约70–80秒；PT／SA为768状态，PA为2048粒子；396–1547 MiB A | 本面板32³由SA均值领先，大规模基态未知 |
| [MWIS，两位Gibbs](mwis/experiment-report.zh-CN.md) | 32768变量 | 约15.3秒；16轨迹；37.9 MiB A | 平均质量较单点p-bit更新提高0.855%，仍比单线程CHILS低3.489% |
| [加权Max-SAT，p-bit](maxsat/experiment-report.zh-CN.md) | 32768原变量；原生软子句 | 达预定质量目标并返回约8.8秒；16轨迹；169.2 MiB A | 相对实测CPU组合有目标时间收益，未建立热浴核独立优势 |
| [稀疏Max-Cut，热浴退火](maxcut/experiment-report.zh-CN.md) | 32768变量 | 约16秒；16轨迹；36.5 MiB A | 四张32K图的普通热浴均值均未超过BURER2002 |
| [稠密Max-Cut，匹配面板](maxcut/experiment-report.zh-CN.md) | 完整32768变量模型 | 各条件均值约121–132秒；16轨迹；4.546 GiB A | 所测随机路线未超过同GPU dSB，结果依赖图族 |
| [LABS，p-bit](labs/experiment-report.zh-CN.md) | 32768原变量；原生高阶模型 | 约123秒；16轨迹；76.8 MiB A | 0.000750%的能量改善显著贵于便宜下降，未建立实用普遍优势 |
| [平衡二分，四位冷启动热浴](partition/experiment-report.zh-CN.md) | 32768顶点；严格基数 | 各图均值约128–251秒；8副本；173.9 MiB A | 仍落后KaFFPa；完整面板的暖改进为0/12 |
| [LDPC，冷启动似然搜索](ldpc/experiment-report.zh-CN.md) | 原生32766位；二次32760位 | 原生约3.94秒／88.6 MiB A；二次约4.60秒／50.1 MiB A；每次8帧×4副本＝32状态 | 原生冷启动返回部分正确码字，二次冷启动无合法输出；整体仍落后BP／LSD |
| [加权集合覆盖，八位可行热浴](set-cover/report.zh-CN.md) | RAIL516：47,311原始位，无辅助位 | 完整均值38.3254秒；128副本；228.638 MiB A／236 MiB R | 热浴184/185/185，认证最优182；AC-CFT三次均182，完整均值2.2234秒 |
| [JSSP，可行优先序热浴](jssp/report.zh-CN.md) | ta51js：37,500个冗余one-hot条目 | 完整均值50.687秒；256副本；105.880 MiB A／170 MiB R | 2950–2960，高于参考2760；小型四实例12/12次达到最优，CP-SAT在各输入的完整均时更低 |
| [覆盖数组，列内两行热浴](covering-array/report.zh-CN.md) | 35×300×3：31,500个one-hot条目 | 完整均值48.7488秒；128副本；454.903 MiB A／494.000 MiB R | 热浴15/15次零缺失；plateau同样15/15，30行节点命中率更高；不证明最少行数 |
| [图着色，团支撑单点热浴](graph-coloring/experiment-report.zh-CN.md) | inithx.i.1：46,656存储位／28,026原始交互位 | 完整均值7.1736秒；128链；76.399 MiB A／92.000 MiB R | 两次热浴与plateau均128/128零冲突；两个25色图未解，未建立普遍热浴优势 |
| [QAP，四位置条件热浴](qap/experiment-report.zh-CN.md) | tai150b：150²＝22,500个one-hot条目 | 完整均值35.7474秒；512副本；123.073 MiB A／192.000 MiB R | 最好499,088,558，比历史可行参考高0.0384679%；Nugent 9/9次达目录最优，Taillard 0/9次达参考 |

各问题的科学与实际动机见[根目录README](../README.zh-CN.md)。
应用相关性不意味着所测模型已经提供可部署的优势。

## 如何理解这些数字

- **单模型与并行工作量：** 变量及辅助位属于单个编码模型。副本、温度槽、粒子和信道帧是
  单独的资源维度，不作为新增独立试验。这些是软件状态数，不是制造出的p-bit器件数。
- **完整耗时与搜索时间：** 所选MWIS、稀疏Max-Cut、稠密Max-Cut、LABS及划分的搜索窗口分别为
  12、15、90、120、45秒。准备、状态输出、修复和认证可能显著增加完整耗时。
  Max-SAT在声明的质量目标处停止；大码LDPC调用采用固定工作量上限。
- **因式分解：** 34.5分钟中位数包含全部九次128-bit调用，其中两次约49–50分钟后触发事件上限。
  仅成功调用的采样首中中位数约15.8分钟，是另一种统计量。总状态数包含钳制端口。
- **TSP：** 当前候选网络在GPU执行二值热浴，33,662个变量表达受限assignment松弛，另有8个副本；全局连通性需要经典读出／修复。终点17.047秒是一次完整调用，不是首次命中时间或中位数。[旧150城研究](tsp/experiment-report.zh-CN.md)使用CPU选择器与GPU局部搜索，首证书中位4.1秒及0.18 MiB峰值保留为历史记录。
- **自旋玻璃：** PT／SA使用4组×96温度槽×2副本，PA使用4组×128×4粒子。allocated峰值依次为
  490.6、396.2、1547.0 MiB；成功调用的完整均值为70.6、74.1、69.4秒。PA计入一次失败后，
  每个完成逻辑调用摊销79.7秒，保留在表格约70–80秒的范围中。
- **LDPC：** 每帧使用四个副本。二次模型包含16380数据位与16380辅助位，因此原码长与原生模型
  不同。合法码字、发送字及全局最可能码字属于不同结论。

allocated／reserved统计不含完整驱动／上下文及其它进程开销。冷／暖启动、经典CPU对照与
不同方法，在链接报告中分别说明。较早的因式分解／自旋玻璃面板和新增组合优化面板各有
环境记录；相同GPU型号不代表每次测量的整个环境相同。

## 数据与复现

[资源账本](../data/release/benchmark-resources.json)记录所选调用、聚合规则、并行数和显存字段，
汇总此前验收测量，不增加试验数。历史TSP的选择器数量及分配器峰值，是从相同归档调用提取的
补充元数据，保留了来源摘要与调用清单的指纹。

当前TSP行使用单独的[指标](../data/tsp/candidate-network/analysis/metrics.json)及[完整调用](../data/tsp/candidate-network/rtx5090/calls.json)。

多数值可以直接追溯到：

- [因式分解汇总](../data/factorization/release-v3/summary/aggregates.json)，以及报告链接的逐调用记录。
- [自旋玻璃调用](../data/spin-glass/s23/calls.json)。
- 表中报告链接的既有面板调用，包括[TSP](../data/tsp/rtx5090/calls.json)、
  [MWIS](../data/mwis/rtx5090/calls.json)、[Max-SAT](../data/maxsat/rtx5090/calls.json)、
  [稀疏Max-Cut](../data/maxcut-sparse/rtx5090/calls.json)、
  [稠密Max-Cut](../data/maxcut-dense/rtx5090/calls.json)、[LABS](../data/labs/rtx5090/calls.json)、
  [划分](../data/partition/rtx5090/calls.json)及[LDPC](../data/ldpc/rtx5090/calls.json)。

此前组合优化面板仍是**365次自然调用与35次另行回放**；新的TSP研究另增加28次自然调用与1次回放。因式分解和自旋玻璃保留各自原面板。
[复现指南](reproduction.zh-CN.md)说明CPU证书核验、新调用、回放及外部基线要求。
本页全部数值来自这些仓库来源；外部研究引用与阅读限制继续保留在完整专题报告中。

集合覆盖的独立[逐调用记录](../data/set-cover/rtx5090/calls.json)、[完整分母与冻结目标](../data/set-cover/rtx5090/panel.json)和[环境](../data/set-cover/rtx5090/environment.json)绑定本次实测；不与历史面板混算试验数。
