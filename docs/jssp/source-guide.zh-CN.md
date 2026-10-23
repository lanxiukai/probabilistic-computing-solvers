# 作业车间调度源码与复现指南

[English](source-guide.md) · [概览](README.zh-CN.md) · [报告](report.zh-CN.md)

本指南说明已通过软件检查的实现、已验收的独立确认及可移植证据约定。完整原始证据已通过云端和本地审计，组装后的公开版本通过590项软件测试、核心见证检查及三次精确工作量源码回放。

## 源码职责

| 文件 | 职责 |
|---|---|
| `src/combinatorial_solvers/jssp/model.py` | 输入验证、独立 CPU 解码，以及原始工序前后关系和机器容量证书 |
| `src/combinatorial_solvers/jssp/gpu.py` | 显式 one-hot 状态及基于整数 makespan 的精确条件排列更新 |
| `src/combinatorial_solvers/jssp/active_gpu.py` | 两位置块的最早空档插入能量 |
| `src/combinatorial_solvers/jssp/solver.py` | 有界搜索、初始化、历史最好与质量边界、RNG 捕获和恢复检查点 |
| `src/combinatorial_solvers/jssp/baseline.py` | 独立单 worker CP-SAT 调度模型及逐次改善见证 |
| `src/combinatorial_solvers/jssp/runner.py` | 原始约束认证、逐副本首次命中和结果文件 |
| `scripts/jssp/experiment.py` | 隔离的源码/配置/输入准备、执行及清单校验 |
| `scripts/jssp/fetch.py` | 六个固定基准下载及原始字节哈希校验 |
| `scripts/jssp/verify_release.py` | 不运行优化器的可移植见证与公开结论核验 |
| `scripts/jssp/plot_schedule.py` | 同一副本冷初态与时限内最好时间表的示意图 |

问题包使用共享证据与隔离执行框架，不导入其它问题的求解器。公开包前缀为 `combinatorial_solvers`，无需私有仓库副本。

## 表示与精确性

对于 $L=JM$ 道操作，每个状态中每个作业标号恰好出现 $M$ 次。`bits[position, job]` 是显式保存的 one-hot 表示；整数优先序、当前代价、历史最好代价和参与掩码均为针对原始调度问题核验的缓存。

共享元数据字段 `physical_update_pbits_per_model` 统计分配给热浴流程的模型二元项，对经典控制取零。字段名不表示物理硬件，也不表示全部项均参与更新。`visited` 掩码和 `visited_assignment_bits_min` 另行描述更新块实际考虑过的赋值项，后者也不同于位翻转次数。

参与计数覆盖最终返回的执行过程，包括可能保留的截止越界 chunk；时限内解质量对应另一个前缀。时限内参与量的保守下界为最终计数减去 `width**2 * (completed_steps - quality_max_step)`，再与零取最大值，因为每步每副本最多考虑 `width**2` 个赋值项。

`schedule_decoder` 选择数学能量映射。`append` 将操作安排在其机器已经排入的最后操作之后；`active` 将其插入最早可行机器空档。两者均保持作业前序及机器容量约束。最早空档解码属于经典支撑，其冷初态收益必须与后续 p-bit 更新带来的改进分开。当前 active 核只接受两位置块，append 另支持四位置块。

全部最终确认调用均使用 `append`。active 备选实现作为被导入的包组件随源码提供，不附带最终确认的性能主张。

`decoder` 选择精确的寄存器查找实现（`reduce` 或 `gather`），与 `schedule_decoder` 含义不同。开发对照中，`gather` 虽然全部非计时数组相同，却更慢，因此保留 `reduce`。

对于宽度为 $b$ 的块，GPU 核穷举 $b!$ 种排列，用原始整数 makespan 和 FP64 概率采样热浴候选。重复标号具有相同的候选重数。参考最优值只用于报告，不作为能量项、初始化提示或 CP-SAT 提示。`metropolis` 与 `plateau` 是匹配 CUDA 对照；这里的 Metropolis 仍计算完整候选集，因而用于区分更新规则，不代表专门优化过的独立实现。

## 质量、计时与证据

`best_sequence` 和 `best_cost` 包含直到最后一个已完成 chunk 的最好已接受状态。`quality_best_sequence` 和 `quality_best_cost` 只包含声明搜索截止前观察到的状态；固定工作重放还受 `quality_max_step` 限制。越界 chunk 仍保留证据，但不计入质量。runner 分别认证当前、最终返回最好及时限内最好时间表。

每个完成的 chunk 提供一次观察，保存的最好值包含该 chunk 内全部已接受转移，而首次命中时间只有 chunk 分辨率。`states.npz:first_reference_hit_seconds` 是逐副本数组，未命中项为 `NaN`；摘要中的同名字段是整个 ensemble 的最早命中，全部未命中时为 `null`。冷初态命中的时间为零，不能记为 p-bit 发现。

冻结的确认配置设置 `checkpoint_interval_seconds=2.0`：间隔到达后，在下一个已完成 chunk 保存完整当前/历史最好/时限内状态与已提交 RNG 边界。间隔从搜索开始或上次检查点写入完成时起算。此恢复频率独立于粗粒度轨迹快照，所有逐 chunk 曲线仍保留。周期检查点压缩和观察开销仍计入搜索时间；初始捕获归入准备，最终检查点及输出开销保留在完整 runner 耗时中。如果转移在提交前中断，会保存待完成的随机张量，并标记设备状态边界尚不确定。

CP-SAT 从时限内的 incumbent 回调选取质量。`quality_best_schedule` 是对应见证；`best_schedule` 与 `returned_best_cost` 保留最终求解器返回，即使它越过时限。`returned_optimality_proved` 和 `lower_bound` 描述最终求解器响应；`optimality_proved` 还要求证明响应在搜索截止前完成。原始见证不能独立证明机器墙钟计时的准确性。

准备时间包含初始化与预热，搜索时间包含声明的观察/检查点工作。认证和完整 runner 耗时单列；外层源码恢复与进程启动另有成本。GPU 与单 worker CPU 耗时保留各自资源口径。

## 可移植公开证据约定

计划的公开入口是调用列表 `data/jssp/rtx5090/calls.json`。每条记录包含内嵌 `configuration`、完全相同的可执行 `configuration_file`、数值 `metrics`、基准摘要、核心见证摘要及资源/角色字段。配置与基准路径相对于公开仓库根目录；`witness_file` 相对于 `calls.json` 所在目录。

核心见证保留初态/当前/最终返回最好/时限内最好的优先序与代价、原始时间表、one-hot 状态、参与掩码、RNG 端点及逐副本首次命中。公开曲线可只保留任一副本最好值改善点、首末观察与资格边界；`curve_indices` 保存原始序号，`curve_curation` 声明规则与数量。完整原始曲线经过独立审计并另外保留。公开曲线省略的当前状态波动无法从缩减曲线恢复。

核验器重新计算原始时间表与约束、按种子生成的初态、质量资格、逐副本命中、二元/参与计数及公开数值。固定工作重放须匹配非计时数组，不增加独立试验。核验通过说明见证和结论一致，不表示重新执行了随机搜索，也不独立复测耗时。

## 复现入口

使用公开仓库的冻结 uv 环境，将固定基准原始字节下载到新目录，再核验公开核心数据：

```bash
uv run --frozen python scripts/jssp/fetch.py --output data/jssp/instances
uv run --frozen python scripts/jssp/verify_release.py --calls data/jssp/rtx5090/calls.json
```

在新的隔离目录中重跑第一个冻结的 `ft06` 热浴配置：

```bash
uv run --frozen python scripts/jssp/experiment.py prepare \
  --config data/jssp/rtx5090/configs/j4-ft06-heatbath-s1.json \
  --run-id jssp-ft06-reproduction
uv run --frozen python scripts/jssp/experiment.py run \
  --run-dir artifacts/runs/jssp/jssp-ft06-reproduction
uv run --frozen python scripts/jssp/experiment.py verify \
  --run-dir artifacts/runs/jssp/jssp-ft06-reproduction
```

准备阶段保存实际源码和输入，不复用已有完成目录。按墙钟预算重跑可能完成不同数量的更新；如需按固定工作量重放，选择 `data/jssp/rtx5090/configs/replay-j4-ft06-heatbath-s1.json` 并使用新的运行 ID。该配置保留原始完成步数和质量资格边界。验收要求在实测设备和锁定运行时上核对非计时等价性，不代表跨设备计时保证。重复公开种子不增加独立确认试验。

直接从公开见证重画同副本示意图：

```bash
uv run --frozen python scripts/jssp/plot_schedule.py \
  --calls data/jssp/rtx5090/calls.json --id j4-ft06-heatbath-s1 \
  --output artifacts/runs/jssp/ft06-figure
```

绘图器选择时限内最好值最低的副本，同值时取最小副本序号，并展示这个副本在热浴前后的时间表。SVG/PNG 及数值 `figure-data.json` 属于示意，不用于与经典求解器比较。

完整链接参考文献和实测软硬件版本见最终报告。上述复现命令和公开路径已在RTX 5090主机的组装公开版本中验证。
