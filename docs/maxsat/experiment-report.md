# RTX 5090 weighted Max-SAT target study

English | [简体中文](experiment-report.zh-CN.md)

The frozen X2 panel has **40 natural calls and five work replays**, using four
32768-variable formulas and two seeds. All clauses are soft, arity 3, ratio 8,
weights 1–10. The target is unsatisfied weight at most 26214 within 60 seconds,
including a 2-second return reserve. GPU methods use 16 trajectories; CPU
portfolios use 1/4/16 processes on the same host.

NVIDIA GeForce RTX 5090 (32607 MiB); AMD Eng Sample: 100-000001537-04 (quota 17.56 vCPU); Python 3.14.6; PyTorch 2.13.0+cu130; CUDA 13.0; driver 580.178.04.

This device confirmation reuses previously measured instances and seeds.

| Method | Calls | Target hits | Mean target score (s; misses = 60) | Mean complete runner (s) |
|---|---:|---:|---:|---:|
| metropolis | 8 | 8 | 8.690 | 8.785 |
| pbit | 8 | 8 | 8.683 | 8.777 |
| usw-1 | 8 | 8 | 35.585 | 35.624 |
| usw-16 | 8 | 8 | 32.327 | 32.544 |
| usw-4 | 8 | 8 | 32.225 | 32.355 |

Mean target scores are 8.683 s for p-bit and 8.690 s for
Metropolis; the fastest CPU portfolio's mean score is 3.711 times the p-bit
score. The two GPU values are close. The target score includes runner preparation and certificate
observation; source restoration/launcher time is also retained in the data.
Different stopping and observation mechanisms are charged to their implementations.
Overshoot below the target does not establish superior fixed-budget solution quality.

The evidence supports only this frozen random soft-clause panel and quality target.
It does not establish a universal heat-bath advantage, industrial/hard-clause
performance, hardware energy efficiency, or superiority to the strongest same-GPU
solvers. All original assignments and target censoring remain accounted for.
USW-LS requires independently obtained upstream source; the release provides
project-authored instrumentation, not a license to redistribute that program.

![Certified target score](figures/rtx5090.png)

[Method](README.md) · [Core data](../../data/maxsat/rtx5090/calls.json) ·
[Verification](../../data/maxsat/rtx5090/verification.json) · [Reproduction](../reproduction.md)

## References and software

The linked inventory includes background and excluded comparisons used during the
research. Citation does not imply replication. Version/access limits in the method
guide remain applicable; this release did not perform a new full-text literature review.

- R1: [https://zenodo.org/records/21358014](https://zenodo.org/records/21358014)
- R2: [USW-LS](https://github.com/shaowei-cai-group/USW-LS)
- R3: [https://github.com/shaowei-cai-group/NuWLS](https://github.com/shaowei-cai-group/NuWLS)
- R4: [TT-Open-WBO-Inc](https://github.com/alexander-nadel-academic/tt-open-wbo-inc)
- R5: [https://arxiv.org/abs/2201.12858v1](https://arxiv.org/abs/2201.12858v1)
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
