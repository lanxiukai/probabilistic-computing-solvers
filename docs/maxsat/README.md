# Weighted Max-SAT

English | [简体中文](README.zh-CN.md)

See the measured [RTX 5090 report](experiment-report.md). Minimize the integer sum of
weights of unsatisfied soft clauses. Native high-order factors avoid auxiliary
variables: one original Boolean variable uses one p-bit.
Hard clauses and industrial partial Max-SAT are outside the measured panel.

Source: `src/combinatorial_solvers/maxsat/`. Colored heat-bath and Metropolis updates
share the objective and conflict-free factor schedule. The measured confirmation
uses four formulas with 32768 variables, clause ratio 8, arity 3, weights 1–10,
two algorithm seeds, and 16 GPU trajectories. CPU controls use USW-LS with
1/4/16 workers. The fixed target is unsatisfied weight at most 26214 within
60 seconds, including a 2-second return reserve. Preserve target misses and
deadline censoring in the denominator.

Certify every returned assignment against the original literals and integer
weights. Report search time, first certified target, complete runner time,
parallel resources, and baseline instrumentation. The selected target cannot
establish general scaling, superiority to same-GPU solvers, or performance on
hard/industrial instances. An optimization heuristic is not an equilibrium sampler.

Run `maxsat` through the [common instructions](../reproduction.md).
The small example needs no external code. USW-LS upstream source and generated
adapted programs are excluded under the [redistribution boundary](../external-baselines.md);
do not silently substitute a different executable or omit the strong control.

## References

These links preserve the research source inventory, including background and unimplemented comparisons. Citation does not imply replication or a new full-text review. Pinned baseline versions are in the baseline inventory.

- R1: [zenodo.org/records/21358014](https://zenodo.org/records/21358014)
- R2: [github.com/shaowei-cai-group/USW-LS](https://github.com/shaowei-cai-group/USW-LS)
- R3: [github.com/shaowei-cai-group/NuWLS](https://github.com/shaowei-cai-group/NuWLS)
- R4: [TT-Open-WBO-Inc](https://github.com/alexander-nadel-academic/tt-open-wbo-inc)
- R5: [arxiv.org/abs/2201.12858v1](https://arxiv.org/abs/2201.12858v1)
