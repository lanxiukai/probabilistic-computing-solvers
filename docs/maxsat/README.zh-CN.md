# 加权Max-SAT

[English](README.md) | 简体中文

实测见[RTX 5090报告](experiment-report.zh-CN.md)。目标为最小化未满足软子句的整数权重和。
原生高阶因子不需要辅助变量，每个原始布尔变量对应一个p-bit。
硬子句和工业部分Max-SAT不属于本次实测面板。

源码为`src/combinatorial_solvers/maxsat/`。着色热浴和Metropolis共享目标及无因子冲突的更新调度。
已确认四个32768变量公式、子句比例8、元数3、权重1–10、两个算法种子和16条GPU轨迹。
CPU对照使用1／4／16工作进程USW-LS；固定目标为60秒内未满足权重不高于26214，
其中保留2秒用于返回。目标未命中和截止删失均保留在分母中。

对原始文字及整数权重认证返回赋值，报告搜索、首次认证目标、完整runner耗时、并行资源和基线插桩。
所选目标不能证明一般扩展性、同GPU求解器优势或硬／工业实例性能，优化启发式也不等于平衡采样。

按[通用步骤](../reproduction.zh-CN.md)运行`maxsat`；小示例不依赖外部代码。
USW-LS上游源码及生成的适配程序依[再分发边界](../external-baselines.zh-CN.md)不导出，
不能静默替换程序或省略强对照。

## 参考来源

以下保留专题研究中使用的来源入口，包括背景和未实施对照。引用不代表复现，也不代表本次重新阅读全文。基线固定版本见外部基线清单。

- R1: [MaxSAT Evaluation 2026官方归档](https://zenodo.org/records/21358014)
- R2: [USW-LS作者代码](https://github.com/shaowei-cai-group/USW-LS)
- R3: [NuWLS作者代码](https://github.com/shaowei-cai-group/NuWLS)
- R4: [TT-Open-WBO-Inc](https://github.com/alexander-nadel-academic/tt-open-wbo-inc)
- R5: [Grimaldi等，p-bit、可逆逻辑与PT的Max-SAT方法](https://arxiv.org/abs/2201.12858v1)
