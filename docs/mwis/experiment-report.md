# RTX 5090 weighted independent-set study

English | [简体中文](experiment-report.zh-CN.md)

The frozen M2 panel has **78 natural calls and four work replays**: two 2048-vertex
development graphs and four 32768-vertex confirmation graphs, with two algorithm
seeds. Graphs have mean degree 8 and weights 1–100. GPU methods use 16 trajectories;
CHILS uses one or four CPU threads, with populations of 1 and 16 respectively.
The 32768-vertex stochastic/CHILS calls use 12-second search limits; greedy
returns once complete. The table contains 52 confirmation calls; the other 26
are development calls. Inputs were not newly held out for this device run.

NVIDIA GeForce RTX 5090 (32607 MiB); AMD Eng Sample: 100-000001537-04 (quota 17.56 vCPU); Python 3.14.6; PyTorch 2.13.0+cu130; CUDA 13.0; driver 580.178.04.

| Method, 32768 vertices | Calls | Mean feasible weight | Mean complete runner (s) |
|---|---:|---:|---:|
| chils-local | 8 | 676851.250 | 12.606 |
| chils-parallel | 8 | 679077.375 | 12.628 |
| greedy | 4 | 624714.750 | 0.599 |
| metropolis | 8 | 648611.625 | 14.958 |
| pair-gibbs | 8 | 653237.875 | 15.318 |
| pair-metropolis | 8 | 634040.750 | 15.356 |
| pbit | 8 | 647702.125 | 15.051 |

Joint Gibbs changes mean weight by 0.855% relative to single-site heat bath
and -3.489% relative to single-thread CHILS. These are finite-panel comparisons of complete implementations,
not a general MWIS ranking. CHILS was rebuilt from the same pinned source using
its original `-march=native` flags on the measured CPU before any trial.

The direct model has 32768 original variables and no auxiliary p-bits.
Single-site and joint two-site updates preserve the declared penalty and raw-state
interpretation. Independent integer weight/conflict checks cover all saved states.
Classical controls have zero stochastic update p-bits; their graph variables remain
counted separately. An incumbent or CHILS output is not an optimality proof.

![Mean independent-set weight](figures/rtx5090.png)

[Method](README.md) · [Core data](../../data/mwis/rtx5090/calls.json) ·
[Verification](../../data/mwis/rtx5090/verification.json) · [Reproduction](../reproduction.md)

## References and software

The linked inventory includes background and excluded comparisons used during the
research. Citation does not imply replication. Version/access limits in the method
guide remain applicable; this release did not perform a new full-text literature review.

- R1: [KaMIS](https://github.com/KarlsruheMIS/KaMIS)
- R2: [CHILS](https://github.com/KarlsruheMIS/CHILS)
- R3: [https://arxiv.org/abs/2412.14198](https://arxiv.org/abs/2412.14198)
- R4: [https://drops.dagstuhl.de/storage/00lipics/lipics-vol338-sea2025/html/LIPIcs.SEA.2025.22/LIPIcs.SEA.2025.22.html](https://drops.dagstuhl.de/storage/00lipics/lipics-vol338-sea2025/html/LIPIcs.SEA.2025.22/LIPIcs.SEA.2025.22.html)
- R5: [https://arxiv.org/abs/2202.09372](https://arxiv.org/abs/2202.09372)
- R6: [https://arxiv.org/abs/1304.6666](https://arxiv.org/abs/1304.6666)
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
