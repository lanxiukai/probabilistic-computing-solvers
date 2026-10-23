# Strictly balanced graph bisection

English | [简体中文](README.zh-CN.md)

See the measured [RTX 5090 report](experiment-report.md). Minimize
$C=\sum_{(i,j)}w_{ij}(1-s_i s_j)/2$ subject to $\sum_i s_i=0$,
with even vertex count, positive integer edge weights, and unit vertex weights.
This is exact bisection, not a tolerance-balanced multiway or hypergraph model.

Source: `src/combinatorial_solvers/partition/`. State-independent banks of two-/four-site
blocks preserve cardinality exactly. Same-color blocks have no cross edges;
the union of exchange opportunities must be connected.
A four-site block enumerates all 16 patterns and conditions on its current
cardinality, leaving at most 6 feasible states. Classical greedy and two-site
Metropolis are attribution controls.

The measured panel contains four 32768-vertex graphs with cold and common
KaFFPa warm starts, eight stochastic replicas, and 45-second search windows.
Each vertex is one physical update p-bit; conditional selection logic is a
separate cost. Natural isolates still participate in the global balance.
Cold quality, warm-start inheritance, and marginal improvement are separate.
The strong classical control is KaHIP with zero imbalance, final exact-balance
enforcement, and the strong preset.

Verify every saved partition with an independent integer cut and balance check.
Report initialization, block preparation, search, certification, complete time,
GPU memory, and host memory. Late completed CPU calls remain recorded but cannot
improve within-budget quality. Small exact references do not establish optimal
32K cuts; no advantage is presumed.

Run `partition` via the [common instructions](../reproduction.md).
The synthetic example needs no KaHIP installation. Benchmark and executable
boundaries are in [external baselines](../external-baselines.md).

## References

These links preserve the research source inventory, including background and unimplemented comparisons. Citation does not imply replication or a new full-text review. Pinned baseline versions are in the baseline inventory.

- R1: [Restoring Sparsity in Potts Machines via Mean-Field Constraints](https://arxiv.org/html/2602.04200v2)
- R2: [doi.org/10.1038/s44335-026-00097-x](https://doi.org/10.1038/s44335-026-00097-x)
- R3: [Think Locally, Act Globally: Perfectly Balanced Graph Partitioning](https://arxiv.org/abs/1210.0477)
- R4: [github.com/KaHIP/KaHIP](https://github.com/KaHIP/KaHIP)
- R5: [Walshaw Graph Partitioning Archive](https://chriswalshaw.co.uk/partition/)
- R6: [Quantum Annealing for Minimum Bisection Problem: A Machine Learning-based Approach for Penalty Parameter Tuning](https://arxiv.org/abs/2509.19005)
