# 立方自旋玻璃求解器

[English](README.md) | 简体中文

提供热浴并行回火与等能团簇移动（PT+ICM）、循环Metropolis模拟退火（SA）和人口退火
（PA）的GPU实现，附可独立核验的整数解证书。包名为`cubic_spin_glass`，使用本仓库
唯一的uv环境，遵循[MIT许可证](../../LICENSE)。

已验收的RTX 5090实验覆盖16³、24³、32³完整开放立方晶格、零场、最近邻±1耦合。
最大单个问题包含32,768个逻辑自旋；副本和粒子另计为搜索资源。每个规模12个留出
耦合实例的60秒平均观察能量，分别由PT+ICM、systematic PA和循环SA最低。
这不构成p-bit普遍优势或大规模基态结论。

| 阅读入口 | 内容 |
|---|---|
| [实验报告](experiment-report.zh-CN.md) | 方法、冻结协议、实测、补做记账、比较边界与完整引用 |
| [复现指南](reproduction.zh-CN.md) | 安装样例、公开数据核验、自定义输入与显式GPU重跑 |
| [源码指南](source-guide.zh-CN.md) | 运行模块、数据格式及版本绑定 |

```bash
uv sync --frozen
uv run --frozen spin-glass-bench \
  --config configs/spin-glass/example-cpu.json --output artifacts/spin-glass-example
uv run --frozen python scripts/spin_glass/verify_results.py \
  --output artifacts/spin-glass-verification.json
```

首个求解命令运行八自旋CPU安装样例。核验器读取公开证据，不执行GPU求解。核心数据
包含全部272次尝试、216个成功确认队列、108份便携证书、冻结耦合/配置、数组指纹与RNG记录。
完整原始轨迹和运维档案另行保留；核验器不能从指纹还原未随仓库分发的原始字节。

未知最优能量仍然未知。本发布是GPU模拟，不是p-bit/FPGA硬件实测、平衡采样结果，
也没有复现对外部强基线的优势。尚未在此面板运行外部优化经典实现或作者p-bit代码；
比较边界见报告。
