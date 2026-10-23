# 覆盖数组源码与复现指南

[English](source-guide.md) · [概览](README.zh-CN.md) · [方法](method.zh-CN.md) · [报告](report.zh-CN.md)

## 源码地图

| 源码 | 职责 |
|---|---|
| `src/combinatorial_solvers/covering_array/model.py` | 原始强度二问题、参数读取与整数覆盖证书 |
| `src/combinatorial_solvers/covering_array/gpu.py` | GPU状态分配、行内块更新与分派 |
| `src/combinatorial_solvers/covering_array/column_gpu.py` | 同列两行的精确候选能量与更新规则 |
| `src/combinatorial_solvers/covering_array/solver.py` | 均匀初始化、预热、计时、严格合格质量与恢复边界 |
| `src/combinatorial_solvers/covering_array/baseline.py` | ACTS/IPOG进程及完整数组读出 |
| `src/combinatorial_solvers/covering_array/runner.py` | 配置执行、独立证书与分阶段计时 |
| `scripts/covering_array/experiment.py` | 隔离源码和输入的准备、执行与清单校验 |
| `scripts/covering_array/fetch_acts.py` | 固定上游ACTS下载及既有文件验证 |
| `scripts/covering_array/verify_release.py` | 便携核心见证、目标、调用名册与声明验证 |

公开包名为`combinatorial_solvers`。本问题使用共享隔离执行与证据框架，不导入其它问题求解器。请使用发布配置；求解器默认值并不是所选的列内两行、固定$\beta=8$、128副本路线。

## 恢复参考可执行文件并验证结果

在公开仓库根目录使用其锁定uv环境：

```bash
uv sync --frozen
uv run --frozen python scripts/covering_array/fetch_acts.py \
  --manifest configs/covering-array/external-baselines.json \
  --output data/covering-array/baselines/acts-basic.jar
uv run --frozen python scripts/covering_array/verify_release.py \
  --calls data/covering-array/rtx5090/calls.json \
  --repository .
uv run --frozen python scripts/covering_array/verify_release.py \
  --calls data/covering-array/development-ca2b/calls.json \
  --repository .
```

目录将官方ACTS Basic下载固定为10,803,019字节和SHA-256 `8efe3cdd5847135381b1daba9e8d57d0d013d00b0b2a7c15dd7c4255f0fb1242`。下载内容不符时在写入前拒绝；既有文件不符时保留原文件并拒绝。匹配的既有文件可离线验证。jar从原作者渠道获取，不随本仓库分发。来源目录链接NIST使用声明，并记录实测Java版本OpenJDK 21.0.12.1。详见[引用](report.zh-CN.md#references)。

实例JSON是本项目生成的参数声明，并非下载的解。`rows`、`columns`、`levels`、`strength`完整定义数学输入；初始数组另由求解器种子重建。历史目标来源注记不提供参考数组，也不改变搜索。由于输入与摘要绑定，请保留发布的JSON原始字节。

验证器只使用Python和NumPy，不导入优化器、不运行Java，也不需要GPU；但仍核对ACTS调用所绑定的下载jar字节。通过验证说明原始符号对覆盖与发布记录一致，不是对原始时钟的独立测量，也不回放GPU动力学。

## 发布契约

`data/covering-array/rtx5090/calls.json`将每个调用绑定到内联配置、内容相同的可执行配置、输入摘要和核心见证。`configuration_file`及`benchmark_inputs.path`相对仓库根目录；`witness_file`相对`calls.json`所在目录。经典参考输入路径也相对仓库根目录。输入若包含`recorded_original_sha256`，必须与恢复文件的字节摘要一致。

相邻的`panel.json`为必需文件。`expected_natural_ids`和`expected_replays`保持精确调用成员及回放来源，不仅核对总数。每个`frozen_targets`条目包含`cost: 0`、`status: "fixed-row-feasibility"`、非空来源，以及精确的`rows`、`columns`、`levels`、`strength`。共享实例ID的调用必须绑定相同的原始参数字节。目标能量零表示该行数下没有缺失交互，不表示全局最少行数。

`data/covering-array/development-ca2b/`使用自己的`calls.json`和`panel.json`，完整保留10个自然开发调用及1个工作量回放。其单个开发种子、温度比较与N29未命中，与50个自然确认调用和3个确认回放分开。分别用同一验证器核验两个目录，不能混合计数或种子角色。

原生见证保留初始、当前、返回最好、合格最好数组与能量、one-hot状态、访问条目、改变块次数、GPU随机状态端点及裁剪后的逐块观测。验证器重建均匀初态，枚举原始覆盖，检查one-hot与参与约束，重建截止时间资格，并区分初态命中与后续命中。在这个CPU验证器中，GPU随机状态是不可解释的字节串，通过工作量回放比较其一致性。

在保留全部副本最好值改善、端点及资格边界后，`curve_indices`记录原始序号。验证器检查保留轨迹的契约；证明没有遗漏改善还需要已单独验收的完整轨迹。核心文件无法重建全部中间当前态。

ACTS见证保留完整构造数组。文档定义的`*`单元格具体化为零，所有行均保留，然后核验全部原始交互。`constructed_rows`与`fixed_row_target_met`区分自身行数下覆盖和固定目标。历史字段名`best_feasible_cost`实际指缺失交互数；原生正值代表数组不完整，ACTS的零值必须结合构造行数解读。

## 执行隔离复现

在新的输出目录准备并运行首个冻结的小规模热浴调用：

```bash
uv run --frozen python scripts/covering_array/experiment.py prepare \
  --config data/covering-array/rtx5090/configs/ca3-ca9k4v3-heatbath-s1.json \
  --run-id covering-array-small-reproduction
uv run --frozen python scripts/covering_array/experiment.py run \
  --run-dir artifacts/runs/covering-array/covering-array-small-reproduction
uv run --frozen python scripts/covering_array/experiment.py verify \
  --run-dir artifacts/runs/covering-array/covering-array-small-reproduction
```

准备步骤快照可执行源码、配置与输入。恢复jar并安装Java后，可通过同一入口执行ACTS配置。GPU复现需要兼容CUDA环境；发布的性能仅对应实测RTX 5090系统。同样墙钟上限下的新自然运行可能完成不同工作量；重复发布种子不增加独立确认试验。

发布的`replay-ca3-ca9k4v3-heatbath-s1.json`、`replay-ca3-ca35k300v3-heatbath-s1.json`和`replay-ca3-ca30k300v3-heatbath-s1.json`配置分别保持原调用完成步数及质量资格边界。用新的运行ID选择其中一项可复现固定工作量。该回放的时钟值不构成新的自然试验或通用硬件速度比较。

准备时间含初始化及编译／预热；搜索时间含随机数生成、GPU更新、观测传输、轨迹写入与周期检查点；证书另行计时。`complete_runner_seconds`从运行器入口计至最终数组序列化之后，也包括来源记录及最终状态抓取；最终摘要序列化、外层源码恢复与进程启动不在该计时器内。这些开销意味着各组件计时不一定加和为完整运行器时间。

更新中断且设备状态未提交时，保留的状态仅用于诊断。其原子随机状态／质量边界可支持新的从起点工作量回放，不宣称能直接续跑为完成检查点。维护测试不随公开克隆分发。公开用户可运行见证验证器及限定源码检查：

```bash
uv run --frozen ruff check src/combinatorial_solvers/covering_array scripts/covering_array
uv run --frozen ruff format --check src/combinatorial_solvers/covering_array scripts/covering_array
```

公开组装验证已在RTX 5090主机完成；下载器使用研究客户端标识从NIST恢复固定jar，原始字节和大小均核验。
