# 组合优化研究复现

[English](reproduction.md) | 简体中文

这七个模块补充已有的[因式分解](factorization/reproduction.zh-CN.md)和
[自旋玻璃](spin-glass/reproduction.zh-CN.md)成果。
新增RTX 5090面板包含365次自然调用和35次另行原点／工作量回放；
稀疏与稠密Max-Cut分别评价，回放不增加独立试验数。

比较资源前先读[实验总览](benchmark-overview.zh-CN.md)，其中说明单模型位数、并行副本、
完整耗时和显存计量的区别。概念与背景见[概率计算与p-bit介绍](probabilistic-computing.zh-CN.md)。

## 环境与小示例

在Linux x86-64上从仓库根目录运行全部命令。`.python-version`指定Python 3.14.6，
`uv.lock`固定依赖，包括CUDA 13.0版PyTorch；小型CPU示例无需CUDA设备。
先安装[uv](https://docs.astral.sh/uv/getting-started/installation/)，再使用本仓库唯一的锁定工程：

```bash
uv sync --frozen
uv run --frozen python -m combinatorial_solvers.public_api --list
uv run --frozen python scripts/release/make_examples.py --output examples/small
uv run --frozen python -m combinatorial_solvers.public_api --problem mwis \
  --config examples/small/mwis.json --run-id example-mwis \
  --output-root artifacts/examples
```

小示例使用CPU和生成输入，检查源码／输入隔离及证书链，不是额外性能测量。
公开运行器将原模型变量数与随机更新p-bit数分开，经典对照的p-bit更新数为零。

## 核验公开核心数据

CPU验证器根据完整保存的最好态重算原目标，包括不可行的LDPC尝试；
检查输入及证书哈希、严格划分平衡、LDPC综合／辅助约束与错误率。
完整轨迹审计及源码恢复凭据继续保存在私有研究档案中。

```bash
uv run --frozen python scripts/release/verify_core.py \
  --panel tsp --panel mwis --panel maxsat --panel maxcut-sparse \
  --panel maxcut-dense --panel labs --panel partition --panel ldpc --fetch
```

`--fetch`从原始来源获取固定TSPLIB、Gset及KaHIP基准，写入被忽略的`external/`目录，
这些基准文件不在此再分发。生成稠密图由冻结种子重建，并检查数学输入身份。
大型稠密／LABS证书核验可能需要数分钟CPU时间。

LDPC的`evaluation/`保存评价用合成信道真值，绝不作为译码器输入；
译码器只接收校验、综合及量化可靠度。WiFi矩阵来源遵循所保留的
[MIT声明](../baselines/ldpc/wifi-LICENSE)。

## 复现一次调用

各面板导出自然调用。TSP还包含明确标记的原点回放；其它面板把固定工作量回放设置附在
对应自然调用上，并在`verification.json`保留回放检查。统计独立试验数时使用
`trial_kind`及核验分母。一次自然调用是一次执行，不一定对应一个不同的问题实例。
完整面板计数如下：

| 面板 | 自然调用 | 已记录回放 |
|---|---:|---:|
| tsp | 16 | 1 |
| mwis | 78 | 4 |
| maxsat | 40 | 5 |
| maxcut-sparse | 110 | 5 |
| maxcut-dense | 20 | 5 |
| labs | 20 | 4 |
| partition | 40 | 5 |
| ldpc | 41 | 6 |
| 合计 | 365 | 35 |

各`data/<panel>/rtx5090/calls.json`包含配置、输入身份、资源账本、核心证书及计时字段。
面板名称为`tsp`、`mwis`、`maxsat`、`maxcut-sparse`、`maxcut-dense`、
`labs`、`partition`、`ldpc`。

```bash
uv run --frozen python scripts/release/run_case.py \
  --panel mwis --case n2048-i0-greedy \
  --output artifacts/reproductions/mwis-greedy

uv run --frozen python scripts/release/run_case.py \
  --panel mwis --case n32768-i0-s0-pair-gibbs --mode replay \
  --output artifacts/reproductions/mwis-origin
```

第一条执行新调用，第二条使用已发布的固定工作量原点配置。
相同种子不保证不同GPU／运行时产生相同轨迹，限时新调用可能完成不同工作量。
输入准备和包装代码也可能改变准备成本，报告图表始终对应原归档实测。

暖启动需要明确选择：先用外部参考程序重建声明的`warm_start_origin`，
再通过`--warm-input`提供输出；或用`--use-recorded-warm-input`选择条件实验。
回放模式复用原始初态输入。复用不会重新测量BP／KaHIP初始化，
工具也不会把历史初始化成本当作新观察。冷启动不读取保存解。
图二分的`--warm-input`接收每顶点一行0／1标签的文本文件；
LDPC接收含`errors`数组、形状为`(frames, original_bits)`的NPZ文件。
提供初态的调用及其成本须另外记录。

外部参考调用需要`--baseline-directory`。默认严格检查输入哈希；
新构建可显式使用`--allow-rebuilt-baseline`记录差异。
固定源码版本和项目适配器见[外部参考](external-baselines.zh-CN.md)。
这一选项不构成源码等价声明，也不授予第三方代码使用许可。

## 计时与统计单位

JSON字段名及各报告表头决定计时边界：

| 字段 | 计入区间 |
|---|---|
| `runner_seconds`, `complete_runner_seconds` | runner准备、搜索、状态输出及认证；不含外层源码恢复与进程启动 |
| `launcher_seconds` | 已记录的外层执行包装时间，包含恢复／进程开销 |
| `common_initial_runner_seconds` | 另测的BP／KaFFPa初始化，在暖启动成本中只加一次 |
| `censored_target_seconds` | Max-SAT包含runner准备的首次认证目标时间；未命中记60秒上限 |
| `prepare_plus_first_seconds` | TSP准备／启动加首次返回并认证的目标tour时间 |

这些区间相互包含，不能把runner与launcher时间相加。安装、外部参考编译及后续
档案核验不计入求解计时。搜索预算不是完整调用预算，例如45秒图二分搜索可能
需要明显更长的准备时间。

七份报告中的均值／范围是所列调用的描述性汇总。重复种子共享图、公式、序列长度
或信道帧；一次调用内的轨迹不增加基准实例。报告不从这些小型复用面板推断
置信区间或一般排名。

## 重建组合优化图表

```bash
uv run --frozen python scripts/release/plot_results.py --topic mwis
```

该命令从保存调用原位重建`docs/mwis/figures/rtx5090.png`及`.svg`，并写出
`docs/mwis/figure-values.json`。省略`--topic`处理全部七个专题，重复该参数可选择多个专题。
它不执行求解器，也不改写报告正文；图表与描述性表格使用相同的保存指标。

## 解释与检查

报告区分初始化、搜索、认证及完整耗时；相同时间窗口不表示硬件、种群或能耗相同。
已知最好／启发式参考不是最优性证明，原生高阶模型也不自动等价于同数量的二次硬件位置。
负结果、未命中及外部参考的有限复现边界均属于发布内容。

```bash
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

无CUDA设备时跳过CUDA检查。公开GPU测量和预检来自RTX 5090，CPU证书核验不需要GPU。

## 加权集合覆盖的独立确认

新面板包含117次自然调用与4次不计新增试验的工作量回放。
先按固定原始哈希下载OR-Library输入，再独立复核已保存证书：

```bash
uv run --frozen python scripts/set_cover/fetch.py --output data/set-cover/or-library-v1
uv run --frozen python scripts/set_cover/verify_release.py --repository .
```

[集合覆盖源码指南](set-cover/source-guide.zh-CN.md)说明隔离运行、完整固定工作量回放、AC-CFT重建及图表再生。该专题使用自己的纯NumPy核验器，不由上面的历史`verify_core.py --panel`入口处理。

JSSP的72次自然调用及3次回放见[报告](jssp/report.zh-CN.md)，[源码指南](jssp/source-guide.zh-CN.md)提供下载、独立核心见证验证和隔离回放入口。

覆盖数组保留50次自然确认调用及3次回放，另有独立的10次开发调用及1次回放。[报告](covering-array/report.zh-CN.md)和[源码指南](covering-array/source-guide.zh-CN.md)给出模型、独立证书验证和复现步骤。

图着色分别保留最终确认、开发比较和等工作量实现面板。原始DIMACS输入按固定哈希下载，自研TabuCol风格程序作为独立CPU参考，非上游HEAD复现。参见[报告](graph-coloring/experiment-report.zh-CN.md)和[源码指南](graph-coloring/source-guide.zh-CN.md)。

QAP保留72次独立确认、4次实现控制及5次回放；Q1c开发与官方Ro-TS另列台账。[报告](qap/experiment-report.zh-CN.md)和[模型／复现指南](qap/README.zh-CN.md)给出完整参数、见证核验及三个面板的边界。
