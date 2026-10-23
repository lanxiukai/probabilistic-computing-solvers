# 立方自旋玻璃结果的复现与核验

[English](reproduction.md) | 简体中文

使用Linux x86-64和本仓库唯一的锁定环境。已验收GPU数据使用Python 3.14.6、
Torch 2.13.0+cu130和RTX 5090；完整硬件与评测范围见[报告](experiment-report.zh-CN.md)。
下列安装与证据检查是CPU软件核验，不是新增基准试验。

## 安装并检查便携入口

```bash
uv sync --frozen
uv run --frozen spin-glass-bench --help
uv run --frozen spin-glass-bench \
  --config configs/spin-glass/example-cpu.json --output artifacts/spin-glass-example
uv run --frozen python src/cubic_spin_glass/certificates.py \
  artifacts/spin-glass-example/instance.json artifacts/spin-glass-example/witnesses.json
```

八自旋铁磁样例用于检查安装；已有输出目录会被拒绝。每次求解保存解析后的输入/配置、
源码、`pyproject.toml`、`uv.lock`、环境版本、RNG、原始数组、观察记录、整数证书和
文件清单。失败运行保留部分文件。

## 不使用GPU核验公开证据

```bash
uv run --frozen python scripts/spin_glass/verify_results.py \
  --output artifacts/spin-glass-verification.json
uv run --frozen python scripts/spin_glass/analyze_results.py \
  --output artifacts/spin-glass-analysis.json
```

核验覆盖运行源码指纹、全部公开数据文件、冻结的271个逻辑调用、包括失败在内的272次
尝试、开发/确认隔离、已记录回放指纹/RNG、重算的配对统计和108份整数证书。
完整原始轨迹未随仓库分发；此命令不表示独立重做了历史原始字节的完整审计。
输出文件必须是新文件。独立证书核验器只需Python标准库；分析与完整公开数据检查使用
锁定的NumPy/运行环境。

核验一份32³导出证书：

```bash
uv run --frozen python src/cubic_spin_glass/certificates.py \
  data/spin-glass/s23/instances/l32-i2.json \
  data/spin-glass/s23/witnesses/l32-i2-sa.json
```

## 输入另一个实例

实例JSON恰好包含`length`、`boundary: "open"`和`bonds: [[i, j, J], ...]`。
编号为`i = x + L*y + L*L*z`。每条无向最近邻边恰好出现一次，整数`J`为−1或+1，
无场，且`2 <= length <= 32`。配置恰好包含`instance_file`（相对于配置文件）和`solver`。
可从CPU样例开始；GPU工作设置`device: "cuda"`、`execution: "lattice-v5"`，
运行前显式确定方法、seed、人口、sweep上限和`max_seconds`。

核验器拒绝缺边、重边、远程边，以及浮点/布尔冒充整数。输入覆盖不意味着对任意
耦合实现的性能或最优性保证。

## 显式GPU重跑

列出冻结面板不会执行求解：

```bash
uv run --frozen python scripts/spin_glass/run_cell.py --list
```

以下命令会执行一次新的60秒采样，准备、输出和认证另占时间。决定执行前，应先安排
可用GPU及适当的运行环境：

```bash
uv run --frozen python scripts/spin_glass/run_cell.py \
  --cell confirm-l32-i2-s1-sa --output artifacts/spin-glass-new-confirmation
```

固定工作量32³回放分别使用`--cell replay-sa`、`replay-pt`、`replay-pa`。
对应sweep上限为38475、14464、14652，采样保护上限220秒。重跑不增加独立耦合实例，
不能悄悄加入公开结果分母。原状态/RNG指纹保存在`data/spin-glass/s23/calls.json`。
同设备固定工作量回放与墙钟截止复现不同；时间戳、完整调用耗时和跨设备轨迹不保证相同。

助手脚本只运行选定调用，不自动执行全体面板、重试失败、配置设备或操作云账户。
助手脚本核验并在独立目录恢复实测源码和输入，以新进程执行原240秒完整调用上限及
八秒终止宽限。失败保留部分输出。`max_seconds`另外限制采样循环，不限制编译/导出。

恢复一次运行时，在独立检出或工作目录恢复其`source/cubic_spin_glass`和环境元数据。
把保存的`config.json`运行到新输出目录，必要时用`--environment-project`指定所属环境目录。
不得覆盖已完成证据目录。

## 软件回归

```bash
CUDA_VISIBLE_DEVICES='' uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

设备被隐藏时CUDA专用检查会跳过。这些检查不替代已记录的RTX 5090实验。
完整研究/软件引用和外部比较边界见实验报告。
