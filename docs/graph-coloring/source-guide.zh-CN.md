# 图着色源码与复现指南

[English](source-guide.md) | 简体中文

实现针对原始 DIMACS 图求解固定颜色数下的冲突目标，可选每个顶点具有不同允许颜色列表。[实验报告](experiment-report.zh-CN.md) 解释二元表示、p-bit 参与、经典支撑、实测质量和科学结论边界。

## 代码分工

| 文件 | 职责 |
|---|---|
| `model.py` | 简单图输入、原始边证书、颜色列表和独立核验的 clique 约束 |
| `gpu.py` | 持久二元状态、颜色／计数缓存和精确条件块 |
| `solver.py` | 初始化、温度日程、有界采样、观测和恢复证据 |
| `tempering.py` | 独立组内相邻温度标签交换 |
| `baseline.py` | 独立单线程 TabuCol-style 参照及其证书 |
| `runner.py` | 配置执行和完整结果核验 |
| `scripts/graph_coloring/verify_release.py` | 公开见证、输入哈希、时限及声明计数的纯 CPU 核验 |

专题模块位于 `src/combinatorial_solvers/graph_coloring/`。`verify_release.py` 不导入求解器或 CUDA；它独立解析原始边，检查列表和 clique 证书，再由保存状态复算冲突及规模计数。

## 运行示例

在仓库根目录使用锁定环境运行以下命令。采样命令需要 CUDA；下载输入和构造 clique 属于 CPU 操作。

```bash
uv run --frozen python scripts/graph_coloring/fetch_benchmarks.py --instance fpsol2.i.1
uv run --frozen python scripts/graph_coloring/prepare_clique.py --instance data/graph-coloring/dimacs/fpsol2.i.1.col --palette 65 --output data/graph-coloring/demo/clique.json
uv run --frozen python -m combinatorial_solvers.public_api --problem graph-coloring --config configs/graph-coloring/example.json --run-id graph-coloring-example --output-root artifacts/examples
```

下载器核对 `configs/graph-coloring/benchmarks.json` 记录的精确字节数与 SHA-256。原始基准文件从引用来源下载，不在这里重新分发。clique 脚本由图边生成证书，不读取已知着色；它拒绝覆盖已有证书，重新准备时应选择新输出路径。

示例在完整 `fpsol2.i.1` 图上为每条链实例化 32,240 个 one-hot 位。其中有 227 个孤立顶点，因此只有 17,485 位与冲突因子相连。固定独立核验的 65-clique 后，自由采样位为 28,015，其中与冲突因子相连的自由位为 13,260。clique 构造属于经典支撑；初态已经零冲突时，不会计入 p-bit 新增命中。

示例种子与报告中的试验分开。示例正常完成不会自动增加公开结果，其时间也不必与实测硬件数据一致。

## 配置模型

`problem.colors` 固定颜色数。可选 `problem.allowed_file` 读取 $N\times K$ 二元 NumPy 掩码，每个顶点至少允许一种颜色。clique 对称性文件与列表文件不能同时使用：全局颜色重命名保持普通合法着色，但一般不保持受限列表。

`solver.width` 选择单、双、三或四顶点条件块；三／四点块要求颜色数不超过 8。`method="heatbath"` 选择 p-bit 热浴；`metropolis` 和 `plateau` 单独标为经典对照。有限温度热浴使用 `beta_start`、`beta_end` 和 `cycle_steps`，两端相同即为恒温。

使用温度交换时，偶数 `tempering_ladder_size` 必须整除 `replicas`。每个梯度组是一个独立组。`exchange_every_chunks` 控制交换间隔，因此改变 `chunk_steps` 后，若不相应调整，交换间隔也会变化。改变块形状可能改变随机数布局，不保证轨迹相同。

`cache_boltzmann_weights=true` 缓存有界整数局部能量差对应的 FP64 权重。`reuse_preparation=true` 复用允许颜色候选表，并为预热克隆独立状态，不复用先前运行的解。两者在已审计的实现试测后均默认为 true。`warmup_steps=null` 预热完整的指定块。五对试测在已测输入上保持全部非时间状态数组一致，这一观察不构成所有输入的轨迹或速度保证。

`max_seconds` 限制搜索壁钟，其中包括搜索内的 checkpoint 和曲线写入。准备单独记录，完整 runner 时间还包括最终核验和输出写入。`quality_max_step` 是固定工作量回放使用的额外质量边界。任何完成块如果在任一边界之后才被观测，就保留在证据中但不计入时限内质量。增加副本既改变搜索组数，也增加资源消耗。

`snapshot_every_chunks=null` 关闭重复缓冲的周期颜色快照，同时保留初态／终态、每块观测和独立完整 checkpoint。`checkpoint_seconds` 控制 checkpoint 的壁钟间隔。改变这些选项会改变实测开销，因此须在时间比较前冻结。

## 阅读与核验结果

每次运行保存 `summary.json`、`states.npz` 中的原始颜色见证、初态及 RNG 状态、逐次写入的曲线和恢复 checkpoint。历史字段 `best_feasible_cost` 表示时限内最低的**冲突数**，正值仍含冲突边；`returned_best_cost` 还包括质量边界之后观测到的工作。零冲突证明固定颜色目标的基态；精确图色数需要另外核验的匹配下界。

`visited` 掩码标记被更新访问顶点的全部允许 one-hot 坐标；`visited_*_bits_min` 指标取各链计数的最小值，不是实际位翻转数。`changed` 统计联合赋值发生改变的块更新次数。“相互作用”位数按原始冲突因子的邻接关系计算，不是 clique 固定后剩余自由变量之间的耦合数。

每个公开台账将整理结果绑定到配置、原始输入哈希、见证和指标。`rtx5090` 保存 G3 最终确认，`development-g2b` 保存 13 次开发调用和两次工作量回放，`implementation-g2a` 保存 8 次固定工作量实现调用，三者用途与分母各自独立。将基准输入下载到声明路径后，分别核验各台账：

```bash
uv run --frozen python scripts/graph_coloring/verify_release.py --calls data/graph-coloring/rtx5090/calls.json --repository .
uv run --frozen python scripts/graph_coloring/verify_release.py --calls data/graph-coloring/development-g2b/calls.json --repository .
uv run --frozen python scripts/graph_coloring/verify_release.py --calls data/graph-coloring/implementation-g2a/calls.json --repository .
```

公开曲线保留任一副本最优值改善、首末观测和资格边界。`curve_indices` 记录原始观测索引，核验器检查这些坐标与声明的选择规则；这些点保留相应最优质量观测，不重建被省略的当前状态波动。完整原始曲线、温度交换记录和 checkpoint 留在独立实验归档。可移植见证核验检查提供证据的一致性，不认证实际壁钟执行，也不产生新的求解测量。

独立构建自有 CPU 参照：

```bash
uv run --frozen python scripts/graph_coloring/build_reference.py --output data/graph-coloring/tabu-reference
```

生成的 `reference`、C++ 源副本、`build.json` 和 `build.log` 都是显式基线输入。这个参照不计入 p-bit 热浴结果，也不是外部 HEAD 的复现。DIMACS、COLOR、TabuCol 和温度交换来源见实验报告。

公开源码已通过云端软件检查，四次固定工作回放分别核对全部非计时数组。全部基准由固定哈希抓取，三份核心台账均通过独立证书检查。CPU构建输入中的哈希指原测量的构建产物；重建可随编译器、路径和主机变化，原始C++源码身份固定。
