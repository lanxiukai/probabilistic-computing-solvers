# 集合覆盖源码与复现指南

[English](source-guide.md) · [概览](README.zh-CN.md) · [报告](report.zh-CN.md)

## 源码索引

| 源码 | 职责 |
|---|---|
| [`model.py`](../../src/combinatorial_solvers/set_cover/model.py) | 原始稀疏覆盖约束、SCP／rail 解析及整数证书 |
| [`blocks.py`](../../src/combinatorial_solvers/set_cover/blocks.py) | 仅依赖输入的块库，含已选择的 `cost-minhash` 策略 |
| [`gpu.py`](../../src/combinatorial_solvers/set_cover/gpu.py) | Triton 块更新、行覆盖缓存和活动列选择 |
| [`solver.py`](../../src/combinatorial_solvers/set_cover/solver.py) | 贪心前缀、独立副本状态、温程及期限内有效观测 |
| [`baseline.py`](../../src/combinatorial_solvers/set_cover/baseline.py) | 经典贪心支撑、AC-CFT 适配及 SciPy/HiGHS LP/MILP 参照 |
| [`reference.py`](../../src/combinatorial_solvers/set_cover/reference.py) | 小型参考计算及精确有理对偶检查 |
| [`runner.py`](../../src/combinatorial_solvers/set_cover/runner.py) | 配置化流程、原始状态认证及阶段计时 |
| [`verify_release.py`](../../scripts/set_cover/verify_release.py) | 不导入求解器的独立便携验证 |

验收路线使用 `representation="feasible"`、`block_policy="cost-minhash"`、`selection="active"`、八位块、16 组块库和 128 副本。请使用已发布的逐调用配置；通用求解器默认值不等于 SC8 验收配置。源码中的替代方法不自动构成新增验收面板。

## 恢复输入并核验已保存结果

从公开仓库根目录，在其锁定 uv 环境中执行命令。[基准清单](../../configs/set-cover/benchmarks.json)固定原始所有者提供的九份文件。获取工具恢复其压缩表示并验证原始数值输入，不依赖无关现有文件。

```bash
uv sync --frozen
uv run --frozen python scripts/set_cover/fetch.py \
  --output data/set-cover/or-library-v1
uv run --frozen python scripts/set_cover/verify_release.py \
  --calls data/set-cover/rtx5090/calls.json \
  --repository .
```

验证器本身只需要 Python 与 NumPy，不执行搜索或 GPU 工作。输出报告完整 117 次自然调用、四次回放、43,045 个保留状态见证、27 次热浴自然调用及 15 次热浴目标命中。另核验历史费用 170 见证，后者不在 43,045 的计数内。回放增加零次独立试验。它检查整理后的公开见证，不重跑原实验，也不复现原始完整档案的 1,083,941 个状态计数。

[面板](../../data/set-cover/rtx5090/panel.json)记录精确自然调用及回放标识、冻结目标。[调用账本](../../data/set-cover/rtx5090/calls.json)通过路径与哈希绑定每份配置、基准输入及见证档案。`benchmark_inputs.path` 和 `configuration_file` 相对仓库根目录；`witness_file` 相对账本所在目录。[环境元数据](../../data/set-cover/rtx5090/environment.json)指明实测设备与软件。

原生运行的 `best_bits`/`best_cost` 是期限内质量见证，`final_bits`/`final_cost` 可能包含超时块。从保留的二元数组重新计算每个原始行和整数费用。`observation_indices` 指向原始逐块观测序列：保留全部副本最好值改善、首尾观测和资格变化两侧。被省略的中间当前态波动不会被重建或计为已验证见证。

有理对偶证书使用精确整数不等式，与求解器报告的浮点界／gap 不同。认证最优需要原始可行见证与有理下界相等。单独保留的 [SCPNRG4 目标见证](../../data/set-cover/reference-witnesses/scpnrg4-target170.npz)绑定更早的可行费用 170，不增加任何 SC8 试验或性能测量。

## 执行新的隔离运行

性能求解器需要支持 RTX 5090 的 CUDA 环境。统一公开入口把验收源码、输入及配置快照到新的隔离运行目录，执行并认证原始覆盖约束：

```bash
uv run --frozen python -m combinatorial_solvers.public_api \
  --problem set-cover \
  --config data/set-cover/rtx5090/configs/sc8-scp41-heatbath-s1.json \
  --run-id reproduce-scp41-heatbath-s1
```

每次执行使用新的运行标识。示例是一轮新的 30 s 原生搜索，另有初始化、编译／预热和认证成本；它不覆盖已冻结发布，也不向冻结分母添加试验。比较结果时应重复该方法完整声明的种子集合。GPU 调用有 128 副本，经典参照为一个 CPU 工作线程。归因于 p-bit 搜索时，必须保留完整耗时与原始贪心前缀。

工作回放是另一种操作：固定已完成工作量和原始质量步数边界，其时钟值不是新的性能证据。公开验证器按声明的原调用检查保留的非计时数组。新的自然运行应使用自然调用配置，而非回放配置。

## 恢复强经典参照

AC-CFT 固定到提交 `6cb63d0036223b8be9c4d01f95b296d641209f95`，{fmt} 固定为 10.2.1。[AC-CFT 清单](../../configs/set-cover/cft-source.json)绑定各必需源码文件，[{fmt} 清单](../../configs/set-cover/fmt-source.json)绑定其压缩包。从所有者获取源码后，用可用的 C++ 编译器为当前 CPU 构建：

```bash
uv run --frozen python scripts/set_cover/fetch_baseline.py \
  --output data/set-cover/baselines/cft-v1
uv run --frozen python scripts/set_cover/prepare_baseline.py \
  --source-directory data/set-cover/baselines/cft-v1 \
  --output data/set-cover/baselines/cft-build-v1
uv run --frozen python -m combinatorial_solvers.public_api \
  --problem set-cover \
  --config data/set-cover/rtx5090/configs/sc8-scp41-cft-s1.json \
  --run-id reproduce-scp41-cft-s1
```

获取／构建目标均为新目录。构建保留上游许可证，使用单独固定的 {fmt}，同时产生带观测及未改计时器的适配程序。编译使用 `-march=native`，应在所选 CPU 上重新构建，不应假定二进制可跨机器运行。自然调用时，计时观测器返回未改变的原生时间；只有明确回放时才替换为已记录序列。搜索评分、提议和停止比较保持原样。独立核验公开数学见证不需要该可执行程序或编译器。

AC-CFT 的名义 30 s 是软搜索时限，应报告返回见证的实际时间，包括超时。九次 60 s LP/MILP 参照服务于界与认证，不是等时对照。求解器报告的最优值与有理证书闭合的证明应分开，详见[报告](report.zh-CN.md)。

## 检查与参考资料

```bash
uv run --frozen ruff check src/combinatorial_solvers/set_cover scripts/set_cover
uv run --frozen ruff format --check src/combinatorial_solvers/set_cover scripts/set_cover
```

维护测试仅保留在本地工作树，不进入公开克隆，详见[仓库指南](../repository-guide.md)。云端软件检查已通过，其中包含独立验证器的原始输入、期限、回放、对偶及故意篡改测试；这些验证器测试不执行求解器。新克隆可用上面的验证命令核验已发布证书。软件检查不产生基准证据，也不授权扩大实验面板。报告的[参考资料](report.zh-CN.md#参考资料)涵盖基准所有者、CFT、先前 Ising／高阶表示以及这里使用的数值和运行时软件。
