# 源码指南

[English](source-guide.md) | 简体中文（译文）

Python 包名为 `sparse_factorization`，命令为 `factor-bench`。本仓库独立维护源码、依赖锁和
Git 历史，运行与复现不依赖私有工作区路径。适用范围见[算法](method.zh-CN.md)，命令见[复现](reproduction.zh-CN.md)。

## 目录与职责

| 位置 | 职责 |
|---|---|
| [core/](../../src/sparse_factorization/core/) | 稀疏模型、固定项与能量常数、邻接布局、着色及随机流 |
| [netlist.py](../../src/sparse_factorization/solver/netlist.py)、[gates.py](../../src/sparse_factorization/solver/gates.py) | 乘法器网表与门罚项 |
| [edge_compiler.py](../../src/sparse_factorization/solver/edge_compiler.py) | 不构造完整稠密矩阵的 compact sparse 编译 |
| [circuit_proposal.py](../../src/sparse_factorization/solver/circuit_proposal.py) | root/body 提案、实际有限质量和 MH 接受 |
| [circuit_deferred_radius.py](../../src/sparse_factorization/solver/circuit_deferred_radius.py)、[circuit_deferred_triton.py](../../src/sparse_factorization/solver/circuit_deferred_triton.py) | 坐标预筛、首个通过者与延迟精确开方 |
| [circuit_graph.py](../../src/sparse_factorization/solver/circuit_graph.py) | CUDA Graph 执行与缓冲区生命周期 |
| [publication_runtime.py](../../src/sparse_factorization/solver/publication_runtime.py) | 独立物理起点、引擎准备与 FP64 认证 |
| [runner.py](../../src/sparse_factorization/runner.py)、[repro.py](../../src/sparse_factorization/repro.py) | 有界执行、源码快照与隔离恢复 |
| [stream.py](../../src/sparse_factorization/stream.py)、[replay.py](../../src/sparse_factorization/replay.py) | 封存分段、增量传输/核验、原点回放 |
| [arithmetic.py](../../src/sparse_factorization/arithmetic.py)、[classical.py](../../src/sparse_factorization/classical.py) | 同噪声整数控制与固定 YAFU |
| [protocol.py](../../src/sparse_factorization/protocol.py)、[report.py](../../src/sparse_factorization/report.py) | 冻结面板、完整分母与数据驱动报告 |
| [templates/](../../src/sparse_factorization/templates/) | 成对Markdown报告正文，源码保持英文 |
| [configs/factorization/](../../configs/factorization/) | 实例、seed 角色、对照、资源和停止条件 |
| `tests/factorization/`（仅本地） | 算术、转移等价、证书、分段与恢复检查 |
| [baselines/factorization/](../../baselines/factorization/)、[build_yafu.py](../../scripts/factorization/build_yafu.py) | 基线版本、依赖来源与构建 |
| [data/factorization/](../../data/factorization/) | 标明版本的核心证据 |
| [provenance/factorization/](../../provenance/factorization/) | 提取来源与数值模块检查 |

## 执行与证据

`prepare-run` 冻结输入、协议和源码；`run` 核对导入位置后，从隔离恢复目录执行。
主循环消费全部声明的随机词，生成算术候选与完整 body，执行 MH，再观察实际接受态。
分段保留连续事件状态、RNG、计数和证书。增量核验与原点回放回答不同问题。
完成目录不再执行，恢复使用新目录并保留原始证据。

算法依据包括 [Aadit](experiment-report.zh-CN.md#ref-2022-aadit-sparse-ising-machines)、
[Hastings](experiment-report.zh-CN.md#ref-1970-hastings)、
[Tierney](experiment-report.zh-CN.md#ref-1998-tierney-metropolis-hastings-kernels) 和
[Hittmeir](experiment-report.zh-CN.md#ref-2022-hittmeir-factorization-subset-sum)；
实现参考 [PyTorch](experiment-report.zh-CN.md#ref-pytorch-cuda)、
[Triton](experiment-report.zh-CN.md#ref-triton-docs) 和 [CUDA](experiment-report.zh-CN.md#ref-nvidia-cuda)。

## 已验收来源与维护

[发布来源记录](../../provenance/factorization/release-v3.json)绑定47个与实测来源逐字节一致的数值／运行文件。最终`buffer_sqrt_qr_v1`包含噪声缓冲复用、整数开方和融合剩余检查；参考配置保留用于等价测试。`verify_results.py`在CPU上核对核心证据，`plot_results.py`重建表格和图。源码快照在许可证与双语README存在时一起保存，并包含包／模板、问题工具／测试、配置、基线锁与环境元数据，排除其他问题。

打包／报告变化会改变整体来源指纹，所以另列数值文件身份与原实测哈希。公开计划省略运维历史，保留每个运行的数值配置及seed。新研究须有独立验收范围；全部引用继续保存在报告中。
