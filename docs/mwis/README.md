# Weighted maximum independent set

English | [简体中文](README.zh-CN.md)

See the measured [RTX 5090 report](experiment-report.md). For binary selection $x_i$,
minimize $E=-\sum_i w_i x_i+\lambda\sum_{(i,j)\in E}x_i x_j$.
With positive weights and $\lambda>\max_i w_i$, a global minimum is feasible;
this does not guarantee efficient local search.

One graph vertex uses one binary variable, with no auxiliary variables.
The measured evaluation extends to 32768 vertices and includes colored single-site
heat bath/Metropolis, state-independent adjacent-pair Gibbs/Metropolis, greedy,
and single/concurrent CHILS controls. Joint updates test whether local barriers
can be crossed without deterministic answer injection.

Source: `src/combinatorial_solvers/mwis/`. The exact integer certificate recomputes weight
and edge conflicts. Preserve raw physical states, feasible incumbents, and repaired
solutions separately. CHILS results are heuristic references, not proven optima.
Four confirmation graphs and two algorithm seeds describe a finite synthetic
distribution; reusing them on a new device does not create new graph samples.
Report chains, CPU workers, initialization, search, certification, and complete
cost separately. These results do not establish general p-bit superiority.

Run the `mwis` fixture using the [common instructions](../reproduction.md).
The synthetic generator and small exhaustive reference are included;
CHILS is an [external baseline](../external-baselines.md).

## References

These links preserve the research source inventory, including background and unimplemented comparisons. Citation does not imply replication or a new full-text review. Pinned baseline versions are in the baseline inventory.

- R1: [KaMIS](https://github.com/KarlsruheMIS/KaMIS)
- R2: [CHILS](https://github.com/KarlsruheMIS/CHILS)
- R3: [arxiv.org/abs/2412.14198](https://arxiv.org/abs/2412.14198)
- R4: [drops.dagstuhl.de/storage/00lipics/lipics-vol338-sea2025/html/LIPIcs.SEA.2025.22/LIPIcs.SEA.2025.22.html](https://drops.dagstuhl.de/storage/00lipics/lipics-vol338-sea2025/html/LIPIcs.SEA.2025.22/LIPIcs.SEA.2025.22.html)
- R5: [arxiv.org/abs/2202.09372](https://arxiv.org/abs/2202.09372)
- R6: [arxiv.org/abs/1304.6666](https://arxiv.org/abs/1304.6666)
