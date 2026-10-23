# 外部参考与构建边界

[English](external-baselines.md) | 简体中文

根MIT许可证覆盖项目自有代码，不会改变上游软件、基准集合或本机构建组合的许可。
发布内容包含项目适配器及必要声明，不再分发上游源码归档或参考二进制。
[版本清单](../data/release/external-baselines.json)记录实际审阅的固定版本，
不对后续上游版本作许可判断。

| 参考程序 | 固定源码 | 已观察条款／边界 |
|---|---|---|
| CHILS | `515952724cd3dcc6c4365a340ecf0f1da782119a` | MIT；沿用原生CPU构建选项 |
| USW-LS | `1b1d216f91e7da81460a21f5cfffab5eed40d3c9` | 此版本未见LICENSE；按适用条款自行获取和使用 |
| MQLib | `585496274af5abb0849d0d47e135496b4688680b` | MIT；未修改BURER2002及明确披露的FESTA修正 |
| KaHIP | `34e1d0a00afeae2ae2ebfe625826e9c8f8f40ac4` | MIT；未修改KaFFPa |
| quantumgizmos/ldpc | `d3429964cd4ffe1abfc041c6ec8b8425cb174f40` | MIT；保留附带依赖声明 |
| tavildar/LDPC | `05ee7f4af36ed5dacf52861315af68b8a17e71e0` | MIT；WiFi矩阵来源 |
| LKH | `3.0.13` | 作者分发条款；另行取得 |

不分发Anechoic作者代码：稠密动力学依据论文方程实现，作者函数的私有对照继续保留在私有区。
没有执行或复用pSA_LDPC作者代码／矩阵。带GPLv2的lMAts开发对照不属于新增5090 L2面板，也不附带。

## 构建项目适配器

从本仓库运行，将已取得的源码／构建结果放入被忽略的`external/`。
构建器在本地保存源码、编译日志、二进制、输入身份和插桩，要求使用精确版本的干净检出。
重新构建可能改变程序哈希，复现参考程序时须披露；
`--allow-rebuilt-baseline`会记录这些差异。

```bash
git clone https://github.com/KarlsruheMIS/CHILS external/sources/CHILS
git -C external/sources/CHILS checkout --detach 515952724cd3dcc6c4365a340ecf0f1da782119a
uv run --frozen python scripts/mwis/prepare_baseline.py \
  --source external/sources/CHILS --output external/builds/chils

git clone https://github.com/MQLib/MQLib external/sources/MQLib
git -C external/sources/MQLib checkout --detach 585496274af5abb0849d0d47e135496b4688680b
uv run --frozen python scripts/maxcut/prepare_baseline.py \
  --source external/sources/MQLib --output external/builds/mqlib
```

MQLib构建器保存并检查符号／布尔修正`assignments_[i] == 1`；
FESTA变体仍有已披露的访问顺序限制，稀疏主要CPU比较采用BURER2002。

构建后，可将源码成员内容与实测版本比较：

```bash
uv run --frozen python scripts/release/verify_reference.py \
  --reference mqlib --directory external/builds/mqlib
```

该检查也支持`chils`、`usw-ls`、`kahip`和`ldpc`，按源码及适配器内容比较，
不受归档时间戳影响。源码相同不表示不同主机上的二进制或性能相同。

对合法取得且处于上述固定版本的USW-LS检出：

```bash
uv run --frozen python scripts/maxsat/prepare_baseline.py \
  --source external/sources/USW-LS --output external/builds/usw-ls
```

脚本提供项目自有的停止／就绪／证书观察逻辑，在使用者本机生成适配源码。
仓库不再分发生成的补丁、改编上游源码或二进制。取得外部源码仍是复现前提，
适配器不授予额外许可。不能换成未插桩程序，却沿用相同计时口径。

KaHIP和LDPC构建器读取原始GitHub API归档及匹配提交元数据，
入口分别为`https://api.github.com/repos/OWNER/REPO/tarball/COMMIT`
与`https://api.github.com/repos/OWNER/REPO/commits/COMMIT`：

```bash
uv run --frozen python scripts/partition/prepare_baseline.py \
  --archive external/sources/kahip.tar.gz \
  --commit-json external/sources/kahip-commit.json --output external/builds/kahip

uv run --frozen python scripts/ldpc/prepare_references.py \
  --decoder-archive external/sources/decoder.tar.gz \
  --decoder-commit external/sources/decoder-commit.json \
  --wifi-archive external/sources/wifi.tar.gz \
  --wifi-commit external/sources/wifi-commit.json --output external/builds/ldpc
```

使用`KaHIP/KaHIP`、`quantumgizmos/ldpc`、`tavildar/LDPC`及表中版本。
LDPC观察层保留未修改作者头文件，向BP传入正确错误概率，
向OSD传入用于完整log-odds排序的独立评分向量。
这一适配及OSD规模／迭代边界均属于被测方法。

从[LKH作者分发入口](http://webhotel4.ruc.dk/~keld/research/LKH-3/)构建程序，
在提供的基线目录内命名为`LKH`。运行器记录实际二进制身份；
不同编译器或主机不共用一个被假定通用的程序哈希。

## 集合覆盖：AC-CFT与fmt

固定AC-CFT修订`6cb63d0036223b8be9c4d01f95b296d641209f95`采用MIT；fmt 10.2.1保留自身许可证。
[AC-CFT清单](../configs/set-cover/cft-source.json)及[fmt清单](../configs/set-cover/fmt-source.json)锁定独立下载的来源。公开内容仅含项目适配器、抓取与构建脚本，不包含原始基准、上游源码归档或二进制。
计时观察器保留上游搜索逻辑，固定计时轨迹只用于工作量回放，不生成新的性能成绩。
构建后的哈希可以随编译器、宿主与路径改变；原始源文件成员的哈希才是版本比较依据。
具体命令见[集合覆盖源码指南](set-cover/source-guide.zh-CN.md)。

## JSSP

OR-Tools CP-SAT 9.15.6755作为独立单线程CPU参考，采用Apache-2.0许可证。ScheduleOpt基准仅按固定原始哈希从上游下载，不随本仓库分发。[报告](jssp/report.zh-CN.md)记录来源、严格时限及证明范围。

## 覆盖数组

[NIST ACTS Basic/IPOG](https://csrc.nist.gov/projects/automated-combinatorial-testing-for-software/downloadable-tools)作为独立CPU构造参考。jar按固定哈希从官方单独下载，不随MIT源码分发。它自行选择行数，超过固定目标行数的完整覆盖仍计该目标未命中。详见[报告](covering-array/report.zh-CN.md)与[复现指南](covering-array/source-guide.zh-CN.md)。

图着色分别保留最终确认、开发比较和等工作量实现面板。原始DIMACS输入按固定哈希下载，自研TabuCol风格程序作为独立CPU参考，非上游HEAD复现。参见[报告](graph-coloring/experiment-report.zh-CN.md)和[源码指南](graph-coloring/source-guide.zh-CN.md)。

## QAP

自研属性禁忌与官方Taillard Ro-TS分别作为CPU参照。官方2006年源码具有非商业使用和致谢条件，须通过自有适配器显式接受条件后单独下载、构建；上游原件、插桩全文及二进制不随MIT源码分发。[报告](qap/experiment-report.zh-CN.md)保留不同种子、资源、预算与未命中；[复现说明](qap/README.zh-CN.md)提供固定来源和构建步骤。
