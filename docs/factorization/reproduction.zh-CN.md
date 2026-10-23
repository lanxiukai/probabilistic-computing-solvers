# 运行、分段回收与复核

[English](reproduction.md) | 简体中文（译文）

## 安装与软件检查

Linux x86-64；Python版本和依赖分别由`.python-version`与`uv.lock`固定。

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen factor-bench --help
```

包与模块职责见[源码指南](source-guide.zh-CN.md)。正式性能采用RTX5090实测，软件正确性检查与
完整分解试验分别计数。工具来源见[uv](experiment-report.zh-CN.md#ref-uv)、[Python](experiment-report.zh-CN.md#ref-python)
及[锁定软件表](experiment-report.zh-CN.md#software-references)。

## 准备与执行

从项目根目录运行所有命令。源码、配置和锁由 `prepare-run` 封存；`run` 从独立恢复目录
启动进程，并检查实际导入路径。一个目录只执行一次。示例输出不纳入Git；维护者将实验输出保存在私有研究工作区。输出位于 `artifacts/runs/factorization/<run-id>/`。

```bash
uv run --frozen factor-bench prepare-run \
  --plan configs/factorization/release-v3.json --name w64-holdout-0-s0 \
  --output artifacts/runs/factorization/w64-holdout-0-s0
uv run --frozen factor-bench run --run-dir artifacts/runs/factorization/w64-holdout-0-s0
```

上述第二条启动正式求解，需要当前会话的设备和预算授权。整数对照使用同一个预注册
`--name` 并在准备时加 `--arm selected_root_integer`，输出另设目录。
经典控制通过 `sparse_factorization.classical.run_yafu` 调用已核验的固定版本二进制，
每个任务独立工作目录、非零uint64 seed、单线程和300秒上限；原stdout与全部临时结果保留。
固定版本的实际解析器使用单个uint64 seed；旧docfile的双32-bit写法已与源码不符。
非交互运行通过独立输入文件传入表达式，避免stdin重定向覆盖命令行输入。

运行经典控制前，先构建固定版本二进制：

```bash
uv run --frozen python scripts/factorization/build_yafu.py \
  --cache .cache/yafu-downloads --output .cache/new-yafu-build --jobs 4
```

构建助手面向Ubuntu 24.04 amd64，需要C/C++工具链、`make`及`dpkg-deb`。
它下载并核对锁定的YAFU源码和依赖包，在本地解包依赖并写出`build.json`，不安装系统包。
构建输出必须使用新目录。二进制及运行库须匹配本次构建记录；
在另一主机重新构建不表示复现历史耗时。该控制仍需相应实验授权。

```bash
uv run --frozen factor-bench classical --plan configs/factorization/release-v3.json \
  --name w128-holdout-0-yafu-s0 --binary .cache/new-yafu-build/yafu \
  --output artifacts/classical/factorization/w128-holdout-0-yafu-s0
```

## 流水回收

每8192个已完成事件或最终尾段产生一个封存段。GPU线程只复制已完成边界的状态和计数；
单个后台线程压缩、校验并原子发布。最多两个排队段，加上下次采集的暂存，内存有界。
未写完的隐藏目录不进入下载；事件数、随机流及累计预算不因分段重置。

示例使用SSH别名`gpu`，并假设仓库位于远端账户主目录下；请替换为自己的别名与路径。
`pull --watch`让网络传输、完整性检查和独立CPU证书线程重叠；
已核验段不重复下载。连接信息由本机SSH配置提供，不写入仓库。

```bash
uv run --frozen factor-bench pull \
  --remote gpu:probabilistic-computing-solvers/artifacts/runs/factorization/w128-holdout-0-s0/result/stream \
  --staging .cache/download-w128-0-0 --output artifacts/downloads/factorization/w128-0-0 \
  --receipts artifacts/audits/factorization/w128-0-0 --watch
```

每段含恢复用状态、RNG、逐事件累计计数、首解记录及必要的转移证据；清单串联上一段。
完整性核验检查大小/hash、事件覆盖、边界计数和状态合法性；证书线程独立重建CPU/FP64模型。
完成标记在最终转移认证后发布并绑定摘要。中断/失败段保留，不能只凭文件存在宣布成功。
整个任务的源码包、环境、控制器日志和总清单仍需随轮次回收；stream不是这些资料的替代品。

## 独立审计与完整回放

```bash
uv run --frozen factor-bench verify --stream artifacts/downloads/factorization/w128-0-0 \
  --receipts artifacts/audits/factorization/w128-0-0-fresh --fresh
uv run --frozen factor-bench replay --stream artifacts/downloads/factorization/w128-0-0 \
  --output artifacts/replays/factorization/w128-0-0 --max-seconds 21600
```

复用核验收据时只处理新增段；`--fresh`使用新收据目录重新检查全部字节。
完整回放从原随机起点开始，逐事件比较计数、逐段比较状态和RNG；分段校验本身不等于原点回放。
`replay --watch`能追赶已回收的连续前缀，但需要独立GPU；单张5090上应在原运行结束后顺序回放。
回放使用对应的RTX 5090执行环境与冻结源码；恢复核验不改写原证据。

报告从冻结方案读取全部分母：

```bash
uv run --frozen factor-bench report --plan configs/factorization/release-v3.json \
  --runs artifacts/runs/factorization --output artifacts/reports/factorization/release-v3
```

没有结果时输出明确的未执行表，不生成虚构性能图。实际结果的图表由CSV确定性生成，
展示所有运行与删失标记，不用少量示例拟合复杂度或TTS99。生成摘要采用英文/中文两份Markdown，图表为PNG/SVG，不生成PDF。

## 软件与基础设施出处

经典控制固定[YAFU](experiment-report.zh-CN.md#ref-project-bbuhrow-yafu)，构建使用
[GMP](experiment-report.zh-CN.md#ref-gmp)、[GMP-ECM](experiment-report.zh-CN.md#ref-ecm)和
[已登记发行包](experiment-report.zh-CN.md#ref-ubuntu-packages)。传输与环境操作参考
[OpenSSH](experiment-report.zh-CN.md#ref-openssh)、[rsync](experiment-report.zh-CN.md#ref-rsync)及
[Vast.ai 官方文档](experiment-report.zh-CN.md#ref-vast-ai)。这些基础设施引用与算法引用均进入报告附录。

## 在CPU上核对随附最终数据

```bash
uv run --frozen python scripts/factorization/verify_results.py --output artifacts/core-verification.json
uv run --frozen python scripts/factorization/plot_results.py --output artifacts/final-figures
```

第一条命令核对核心清单、来源绑定、接受态与整数控制，无需GPU；回放收据只核对记录，不重新执行。第二条从最终核心数据重建全部表格与图。打包／报告变化使公开发行版的总来源哈希不同于实测包；[来源记录](../../provenance/factorization/release-v3.json)列出不变的数值文件及便携公开计划，角色seed和数值设置保留。使用新的输出目录，此检查不需要新增求解实验。
