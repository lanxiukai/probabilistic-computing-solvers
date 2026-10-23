# Weighted set covering

[简体中文](README.zh-CN.md)

GPU-simulated p-bit block heat bath improves a classical greedy cover while preserving every original covering constraint. The accepted RTX 5090 panel includes all 117 natural calls: nine public instances, three fresh seeds for heat bath and three controls, plus nine LP/MILP references. Four work replays add no independent trials.

Heat bath improved its greedy prefix in all 27 calls and reached the frozen target in 15. Against the matched Metropolis and neutral-move descent controls it won/tied/lost 15/12/0 and 19/8/0 times. Against the strong CPU AC-CFT reference it scored 0/16/11; this release establishes an increment within the tested block workflow, not superiority over specialized set-cover solvers.

The largest intact model is `rail516`: 47,311 original binary decisions per replica, 128 replicas, zero auxiliary p-bits. Heat bath returned 184/185/185 against the independently certified optimum 182. Its mean complete runner time was 38.3254 s, with a maximum 228.638 MiB allocated and 236.000 MiB reserved by the CUDA allocator. These are GPU simulation measurements, not a physical p-bit device demonstration.

- [Method, full results and limitations](report.md)
- [Source map and reproduction](source-guide.md)
- [Complete call ledger](../../data/set-cover/rtx5090/calls.json) and [frozen panel](../../data/set-cover/rtx5090/panel.json)
