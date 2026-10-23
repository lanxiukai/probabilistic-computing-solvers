# RTX 5090 native high-order LABS

English | [简体中文](experiment-report.zh-CN.md)

The frozen L2 panel has **20 natural calls and four work replays**, covering
construction/polishing attribution and lengths through 32768.
For $s_i\in\{-1,1\}$, $C_k=\sum_{i=0}^{N-k-1}s_i s_{i+k}$ is aperiodic,
$E=\sum_{k=1}^{N-1}C_k^2$, and merit factor is $N^2/(2E)$.

NVIDIA GeForce RTX 5090 (32607 MiB); AMD Eng Sample: 100-000001537-04 (quota 17.56 vCPU); Python 3.14.6; PyTorch 2.13.0+cu130; CUDA 13.0; driver 580.178.04.

This device confirmation reuses previously measured lengths and seeds.

| Variant, length 32768 | Calls | Mean energy | Mean merit factor | Mean complete runner (s) |
|---|---:|---:|---:|---:|
| appended | 1 | 84715248.000 | 6.337359 | 0.241 |
| metro-legendre | 2 | 84226446.000 | 6.374137 | 123.121 |
| mts-legendre | 2 | 84226508.000 | 6.374132 | 123.043 |
| pbit-legendre | 2 | 84226210.000 | 6.374155 | 123.171 |
| polish | 1 | 84226824.000 | 6.374108 | 0.667 |
| quarter | 1 | 89514508.000 | 5.997585 | 0.187 |

The best observed energy among these calls is **84226192**; this is not an
optimality certificate. Greedy polishing reaches 84226824 in
0.667 s complete runner time; the best energy improvement over it is
0.000750%, while the stochastic/memetic calls take about
123.112 s. This small quality improvement does not establish a
general practical time/quality advantage. Construction, greedy descent, stochastic polishing, and
the project GPU memetic-tabu control remain separate. The memetic implementation
is not the authors' CUDA program; incomplete generations remain visible in the data.
Heat bath and Metropolis use 16 trajectories, constructive/greedy controls one
sequence, and the memetic population size is 100.

Native stochastic updates use 32768 original variables and no auxiliary p-bits.
High-order arithmetic/caches are additional resources, so this does not demonstrate
32768 quadratic-only hardware sites. Repeated seeds at the same length are not
different LABS instances. Exact integer/XOR and independent integer-transform
checks validate saved sequences; neither merit factor nor local convergence proves
global optimality or quantum advantage.

![32K merit factor and complete cost](figures/rtx5090.png)

[Method](README.md) · [Core data](../../data/labs/rtx5090/calls.json) ·
[Verification](../../data/labs/rtx5090/verification.json) · [Reproduction](../reproduction.md)

## References and software

The linked inventory includes background and excluded comparisons used during the
research. Citation does not imply replication. Version/access limits in the method
guide remain applicable; this release did not perform a new full-text literature review.

- R1: [https://github.com/borkob/git_labs](https://github.com/borkob/git_labs)
- R2: [CSPLib prob005](https://www.csplib.org/Problems/prob005/)
- R3: [GPU memetic tabu v2](https://arxiv.org/abs/2504.00987v2)
- R4: [https://arxiv.org/html/2504.00987v2](https://arxiv.org/html/2504.00987v2)
- R5: [https://doi.org/10.1016/j.asoc.2009.03.005](https://doi.org/10.1016/j.asoc.2009.03.005)
- R6: [https://arxiv.org/abs/1406.5301](https://arxiv.org/abs/1406.5301)
- R7: [Advances in the merit factor problem for binary sequences v2](https://arxiv.org/abs/1205.0626v2)
- R8: [https://arxiv.org/html/1205.0626](https://arxiv.org/html/1205.0626)
- R9: [Dual-step optimization](https://doi.org/10.1016/j.dsp.2025.105316)
- R10: [https://arxiv.org/html/2409.07222v1](https://arxiv.org/html/2409.07222v1)
- R11: [https://arxiv.org/abs/2610.14361v1](https://arxiv.org/abs/2610.14361v1)
- R12: [https://arxiv.org/html/2610.14361v1](https://arxiv.org/html/2610.14361v1)
- R13: [https://arxiv.org/abs/2308.02342v2](https://arxiv.org/abs/2308.02342v2)
- R14: [PCE LABS v2](https://arxiv.org/abs/2506.17391v2)
- R15: [https://github.com/J-Brest/LABS](https://github.com/J-Brest/LABS)
- S1: [Python](https://www.python.org/)
- S2: [NumPy](https://numpy.org/)
- S3: [PyTorch](https://pytorch.org/)
- S4: [Triton](https://github.com/triton-lang/triton)
- S5: [CUDA](https://docs.nvidia.com/cuda/)
- S6: [SciPy](https://scipy.org/)
- S7: [Matplotlib](https://matplotlib.org/)
- S8: [PyYAML](https://pyyaml.org/)
- S9: [uv](https://docs.astral.sh/uv/)
- S10: [GCC](https://gcc.gnu.org/)
- S11: [pytest](https://docs.pytest.org/)
- S12: [Ruff](https://docs.astral.sh/ruff/)
