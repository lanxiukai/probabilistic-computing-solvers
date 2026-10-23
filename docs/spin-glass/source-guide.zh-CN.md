# 立方自旋玻璃源码与数据指南

[English](source-guide.md) | 简体中文

本包独立于`sparse_factorization`。使用公开代码和数据不需要私有仓库、云账户或下载的论文。

| 路径 | 职责 |
|---|---|
| `src/cubic_spin_glass/cli.py` | 单实例CLI、源码/环境捕获、原始输出及证书导出 |
| 该包中的`model.py`、`core/` | 立方几何、显式耦合与稀疏Ising数据结构 |
| `solver.py` | 热浴/Metropolis更新、PT、ICM、计时与观察次序 |
| `population.py` | 人口温程、重启、multinomial/systematic重采样与诊断 |
| `triton_*.py` | CUDA局部场/更新核及整数能量归约 |
| `certificates.py` | 标准库模型验证和整数证书核验 |
| `scripts/spin_glass/run_cell.py` | 显式单调用复现；列出面板不执行求解 |
| `scripts/spin_glass/analyze_results.py` | 实例级均值、配对bootstrap和补做敏感性分析 |
| `scripts/spin_glass/verify_results.py` | 公开数据完整性、源码绑定、覆盖、统计与证书核验 |

数值运行时保留实测源码字节。公开包装和分析工具用于复现，不改变更新核。
[来源文件](../../provenance/spin-glass/release-v1.json)绑定这些运行文件和冻结面板；
共享依赖锁是其所属环境。

## 公开证据结构

`data/spin-glass/s23/`包含：

- `instances/`：42个随机耦合文件和一个正确性样例。
- `calls.json`：全部272次实际尝试，包含唯一的`execution_order`、`cell`标识和
  `logical_cell`标识。唯一失败及其补做具有不同实际ID、相同逻辑队列。成功行包含
  全部逐组检查点能量、保存数组指纹、RNG、采样/准备/导出/完整调用时间及allocated显存。
- `summary.json`：216个确认调用的统计，以及32³十一实例敏感性面板；先在同耦合内
  平均组和seed。
- `per-call.csv`：216个成功确认调用，包括恢复项；单独使用此便捷表会漏掉失败尝试成本。
- `best-observed.json`、`witnesses/`：108个绑定模型的最低观察态。
- `random-streams.json`：冻结随机角色键；PT/SA匹配键是有意配对。
- `verification-receipt.json`：已记录的全数组审计/回放覆盖及其限制。
- `manifest.json`：公开文件大小与SHA-256。

调用层的失败超时不同于求解器正常到达采样上限。因此，成功质量行可以同时具有
`solver_stop_status: "timeout"`和证据状态`verified`。失败尝试的未知采样进度保留为
JSON `null`，不编造为零。

完整原始轨迹、执行日志和私有归档路径不属于本核心发布。已记录指纹支持一致性检查及
后续固定工作量复现；整数证书支持独立能量核验。两者都不证明全局最优。
来源引用、硬件、分母及全部比较边界见[报告](experiment-report.zh-CN.md)。
