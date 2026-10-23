# Sparse and dense Max-Cut

English | [简体中文](README.zh-CN.md)

See the measured [RTX 5090 report](experiment-report.md). Maximize
$C=\sum_{i<j}w_{ij}(1-s_i s_j)/2$ for $s_i\in\{-1,1\}$.
The Ising energy is $\sum_{i<j}w_{ij}s_i s_j=\sum w-2C$.
Signed weights require this explicit convention.

Source: `src/combinatorial_solvers/maxcut/` and its `dense/` subpackage.
The sparse route includes colored heat bath, Metropolis, parallel tempering,
generated 32768-variable graphs, Gset instances, and MQLib controls.
Count natural isolates separately from objective-active variables.

The dense route keeps all 32768 original variables. It covers complete signed
SK graphs and positive edge-density 0.5 graphs. The implementation offers exact
block Gibbs, stalled synchronous updates, SCA, Onsager correction, and discrete
simulated bifurcation (dSB). The accepted C3 panel compares SCA/Onsager with dSB
on the two SK graphs and block Gibbs with dSB on the two positive graphs;
stalled synchronous updates are not an additional measured C3 arm.
dSB uses continuous position/momentum and is a classical control with
zero p-bits. Native conditional sampling and synchronous heuristics must not
be described as the same stationary process.

Packed upper-triangle inputs, incremental fields, and independent integer
certificates are included. An FP32 dense matrix at 32768 uses 4 GiB before
state and workspace. Report actual peak memory, active variables, trajectory
count, and all timing components. Compare $2C-\sum w$ as well as the cut:
the dense positive-graph constant can hide large relative optimization gaps.

The dense confirmation uses four graphs, two seeds, and 90-second windows.
Retain matched SCA/Onsager and same-GPU dSB controls, including unfavorable
outcomes; no stable advantage is assumed. K2000 best-known values are not
optimality proofs. Author reference code is excluded from this release.

Run `maxcut` or `maxcut-dense` using the [common instructions](../reproduction.md).
See [external-baseline details](../external-baselines.md), including corrected
FESTA and benchmark licensing. Dense problems remain a first-class research
route; sparsity is not a publication requirement.

## References

These links preserve the research source inventory, including background and unimplemented comparisons. Citation does not imply replication or a new full-text review. Pinned baseline versions are in the baseline inventory.

- R1: [arxiv.org/abs/1805.05217](https://arxiv.org/abs/1805.05217)
- R2: [github.com/MQLib/MQLib](https://github.com/MQLib/MQLib)
- R3: [web.stanford.edu/~yyye/yyye/Gset/](https://web.stanford.edu/~yyye/yyye/Gset/)
- R4: [www.nature.com/articles/s41598-025-90520-3](https://www.nature.com/articles/s41598-025-90520-3)
- R5: [github.com/nonizawa/GPU-pSAv](https://github.com/nonizawa/GPU-pSAv)
- R6: [Programmable Probabilistic Computer with 1,000,000 p-bits](https://arxiv.org/abs/2606.25313v1)
- R7: [Hardware Acceleration of Probabilistic Computing for Max-Cut Problems](https://ieeexplore.ieee.org/document/11668769/)
- R8: [High-performance combinatorial optimization based on classical mechanics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323291/)
- R9: [DOI](https://doi.org/10.1126/sciadv.abe7953)
- R10: [arxiv.org/abs/2610.13368v1](https://arxiv.org/abs/2610.13368v1)
- R11: [arxiv.org/html/2610.13368v1](https://arxiv.org/html/2610.13368v1)
- R12: [github.com/skethz/anechoic](https://github.com/skethz/anechoic)
- R13: [github.com/hariby/SA-complete-graph/tree/WK2000](https://github.com/hariby/SA-complete-graph/tree/WK2000)
- R14: [Edge-of-chaos enhanced quantum-inspired algorithm for combinatorial optimization](https://arxiv.org/abs/2508.17655)
