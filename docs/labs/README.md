# Low-autocorrelation binary sequences

English | [简体中文](README.zh-CN.md)

See the measured [RTX 5090 report](experiment-report.md). For a binary sign sequence,
$C_k=\sum_{i=0}^{N-k-1}s_i s_{i+k}$,
$E=\sum_{k=1}^{N-1}C_k^2$, and merit factor $F=N^2/(2E)$.
This is aperiodic LABS without extra sequence constraints such as skew symmetry,
not periodic correlation or minimum peak sidelobe optimization.

Source: `src/combinatorial_solvers/labs/`. Native high-order p-bit and Metropolis updates
operate directly on the original sequence. CPU integer/XOR and an independent
integer transform validate certificates. FP64 FFT gain calculations have explicit
rounding checks and exact checks on selected moves.
The 32768-variable native model has no auxiliary p-bits, but arithmetic,
correlation caches, and update logic remain resources. It is not equivalent to
32768 sites in a quadratic-only hardware implementation.

The comparison separates random starts, Legendre construction, appended
construction, deterministic greedy polishing, stochastic polishing, and
GPU memetic tabu. The latter is a project implementation of a paper algorithm,
not the authors' CUDA implementation or a complete throughput replication.
Construction and deterministic descent are charged and attributed explicitly.
Multiple seeds at one sequence length are not independent LABS instances.

Certify the complete sequence, every aperiodic lag, energy, and merit factor.
Report attempted/visited sites, completed sweeps or generations, deadline
censoring, setup, search, and complete cost. A long constructed sequence or
local minimum is not a global optimality proof or quantum advantage.

Run `labs` via the [common instructions](../reproduction.md).
The separate small-size lMAts control has a GPLv2 boundary and is excluded;
see [external baselines](../external-baselines.md).

## References

These links preserve the research source inventory, including background and unimplemented comparisons. Citation does not imply replication or a new full-text review. Pinned baseline versions are in the baseline inventory.

- R1: [github.com/borkob/git_labs](https://github.com/borkob/git_labs)
- R2: [CSPLib prob005](https://www.csplib.org/Problems/prob005/)
- R3: [GPU memetic tabu v2](https://arxiv.org/abs/2504.00987v2)
- R4: [arxiv.org/html/2504.00987v2](https://arxiv.org/html/2504.00987v2)
- R5: [doi.org/10.1016/j.asoc.2009.03.005](https://doi.org/10.1016/j.asoc.2009.03.005)
- R6: [arxiv.org/abs/1406.5301](https://arxiv.org/abs/1406.5301)
- R7: [Advances in the merit factor problem for binary sequences v2](https://arxiv.org/abs/1205.0626v2)
- R8: [arxiv.org/html/1205.0626](https://arxiv.org/html/1205.0626)
- R9: [Dual-step optimization](https://doi.org/10.1016/j.dsp.2025.105316)
- R10: [arxiv.org/html/2409.07222v1](https://arxiv.org/html/2409.07222v1)
- R11: [arxiv.org/abs/2610.14361v1](https://arxiv.org/abs/2610.14361v1)
- R12: [arxiv.org/html/2610.14361v1](https://arxiv.org/html/2610.14361v1)
- R13: [arxiv.org/abs/2308.02342v2](https://arxiv.org/abs/2308.02342v2)
- R14: [PCE LABS v2](https://arxiv.org/abs/2506.17391v2)
- R15: [github.com/J-Brest/LABS](https://github.com/J-Brest/LABS)
