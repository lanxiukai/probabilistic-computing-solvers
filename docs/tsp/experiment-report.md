# RTX 5090 TSP confirmation

English | [简体中文](experiment-report.zh-CN.md)

Four methods, four fixed seeds, one 150-city TSPLIB instance: **16 natural calls
and one separate LKH origin replay**. This is device confirmation on an existing
frozen `kroA150` instance, using TSPLIB EUC_2D integer edge rounding.
The exact integer target is 26524. Seeds are 3200000, 3200010, 3200020, 3200030, with a
30-second runner deadline per call, including its own preparation and
search; outer source preparation/restoration and process launch are separate.
The hybrid p-bit selector runs
on CPU; CUDA accelerates its local search. Full EAX comparison is unexecuted.

NVIDIA GeForce RTX 5090 (32607 MiB); AMD Eng Sample: 100-000001537-04 (quota 17.56 vCPU); Python 3.14.6; PyTorch 2.13.0+cu130; CUDA 13.0; driver 580.178.04.

| Method | Calls | Target hits | Median complete first certificate (s) | Median after runner preparation (s) |
|---|---:|---:|---:|---:|
| cpu_pbit | 4 | 4 | 5.643 | 3.438 |
| cuda_classical | 4 | 4 | 3.686 | 0.406 |
| cuda_pbit | 4 | 4 | 4.054 | 0.758 |
| lkh | 4 | 4 | 2.350 | 0.165 |

The complete first-certificate interval includes source preparation, restoration,
process startup, runner preparation, prefix, search, and certification.
The shorter interval retains prefix/search/certification after runner preparation.
For LKH, first certificate means the first returned complete tour checked by
the adapter, not the unobserved time at which its internal search first found it.
Common wrapper costs can reduce apparent ratios; they do not make algorithmic
search costs equivalent. LKH has the lower median: 2.350 s versus
4.054 s for GPU-assisted p-bit; the median paired time ratio is 1.698.

Every saved tour observation was independently checked; the public core includes
best and first-hit tours for all calls. Paired CPU/CUDA p-bit prefixes and the LKH
origin replay were verified without adding independent trials. Initialization,
repair, and polishing are part of the hybrid; a hit is not automatically a
standalone p-bit success. No fabricated-hardware or general TSP advantage is claimed.

![Complete first-certificate time](figures/rtx5090.png)

[Method](README.md) · [Core data](../../data/tsp/rtx5090/calls.json) ·
[Verification](../../data/tsp/rtx5090/verification.json) · [Reproduction](../reproduction.md)

## References and software

The linked inventory includes background and excluded comparisons used during the
research. Citation does not imply replication. Version/access limits in the method
guide remain applicable; this release did not perform a new full-text literature review.

- R1: [DOI](https://doi.org/10.1287/opre.2.4.393)
- R2: [https://publications.csail.mit.edu/lcs/pubs/pdf/MIT-LCS-TM-209.pdf](https://publications.csail.mit.edu/lcs/pubs/pdf/MIT-LCS-TM-209.pdf)
- R3: [https://doi.org/10.1007/s43069-021-00101-z](https://doi.org/10.1007/s43069-021-00101-z)
- R4: [arXiv v4](https://arxiv.org/abs/1708.04215v4)
- R5: [JACM](https://doi.org/10.1145/3424306)
- R6: [arXiv v6](https://arxiv.org/abs/2007.01409v6)
- R7: [https://doi.org/10.1287/opre.2022.2338](https://doi.org/10.1287/opre.2022.2338)
- R8: [arXiv v1](https://arxiv.org/abs/2212.06296v1)
- R9: [https://www.or.uni-bonn.de/tspbook/book.pdf](https://www.or.uni-bonn.de/tspbook/book.pdf)
- R10: [Cambridge](https://doi.org/10.1017/9781009445436)
- R11: [https://www.math.uwaterloo.ca/~bico/papers/tsp_icm.pdf](https://www.math.uwaterloo.ca/~bico/papers/tsp_icm.pdf)
- R12: [https://www.math.uwaterloo.ca/~bico/papers/clk_ijoc.pdf](https://www.math.uwaterloo.ca/~bico/papers/clk_ijoc.pdf)
- R13: [DOI](https://doi.org/10.1007/s12532-009-0004-6)
- R14: [https://www.math.uwaterloo.ca/tsp/data/art/tours/GA_tsp_art.pdf](https://www.math.uwaterloo.ca/tsp/data/art/tours/GA_tsp_art.pdf)
- R15: [Concorde](https://www.math.uwaterloo.ca/tsp/concorde/)
- R16: [pla85900](https://www.math.uwaterloo.ca/tsp/pla85900/index.html)
- R17: [DOI](https://doi.org/10.1145/2330784.2330978)
- R18: [arXiv v1](https://arxiv.org/abs/2003.11902v1)
- R19: [https://doi.org/10.1016/j.future.2020.01.011](https://doi.org/10.1016/j.future.2020.01.011)
- R20: [arXiv v2](https://arxiv.org/abs/2404.04895v2)
- R21: [https://doi.org/10.1145/3638530.3654394](https://doi.org/10.1145/3638530.3654394)
- R22: [arXiv v1](https://arxiv.org/abs/2608.21233v1)
- R23: [arXiv v3](https://arxiv.org/abs/1803.08475v3)
- R24: [arXiv v3](https://arxiv.org/abs/2010.16011v3)
- R25: [arXiv v1](https://arxiv.org/abs/2110.07983v1)
- R26: [arXiv v2](https://arxiv.org/abs/2207.03876v2)
- R27: [arXiv v2](https://arxiv.org/abs/2302.08224v2)
- R28: [DOI](https://doi.org/10.1126/science.3755256)
- R29: [DOI](https://doi.org/10.3389/fphy.2014.00005)
- R30: [https://doi.org/10.1103/plr8-xys1](https://doi.org/10.1103/plr8-xys1)
- R31: [https://doi.org/10.1038/s41928-023-01065-0](https://doi.org/10.1038/s41928-023-01065-0)
- R32: [https://doi.org/10.1038/s41467-024-47818-z](https://doi.org/10.1038/s41467-024-47818-z)
- R33: [https://doi.org/10.1038/s44335-026-00060-w](https://doi.org/10.1038/s44335-026-00060-w)
- R34: [https://doi.org/10.1038/s41467-025-66864-9](https://doi.org/10.1038/s41467-025-66864-9)
- R35: [https://arxiv.org/abs/2504.13294v1](https://arxiv.org/abs/2504.13294v1)
- R36: [DOI](https://doi.org/10.1109/LES.2023.3298739)
- R37: [arXiv v1](https://arxiv.org/abs/2607.21182v1)
- R38: [https://doi.org/10.1038/s41467-023-41647-2](https://doi.org/10.1038/s41467-023-41647-2)
- R39: [https://doi.org/10.1038/s41467-022-29411-4](https://doi.org/10.1038/s41467-022-29411-4)
- R40: [arXiv v1](https://arxiv.org/abs/2211.15869v1)
- R41: [arXiv v1](https://arxiv.org/abs/2202.08939v1)
- R42: [https://doi.org/10.3390/math10081294](https://doi.org/10.3390/math10081294)
- R43: [arXiv v2](https://arxiv.org/abs/2502.17725v2)
- R44: [arXiv v1](https://arxiv.org/abs/2604.25531v1)
- R45: [arXiv v4](https://arxiv.org/abs/2504.02388v4)
- R46: [https://doi.org/10.1145/3712255.3734313](https://doi.org/10.1145/3712255.3734313)
- R47: [https://doi.org/10.1587/transcom.2020NVI0002](https://doi.org/10.1587/transcom.2020NVI0002)
- R48: [TSPLIB95](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
- R49: [nagata-yuichi/GA-EAX](https://github.com/nagata-yuichi/GA-EAX)
- R50: [RSkinderowicz/GPU-based-MMAS](https://github.com/RSkinderowicz/GPU-based-MMAS)
- R51: [EMI-Group/tensoraco](https://github.com/EMI-Group/tensoraco)
- R52: [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)
- R53: [yd-kwon/POMO](https://github.com/yd-kwon/POMO)
- R54: [liangxinedu/NeuroLKH](https://github.com/liangxinedu/NeuroLKH)
- R55: [JHL-HUST/VSR-LKH-V2](https://github.com/JHL-HUST/VSR-LKH-V2)
- R56: [Edward-Sun/DIFUSCO](https://github.com/Edward-Sun/DIFUSCO)
- R57: [souravsanyal06/Neuro-Ising](https://github.com/souravsanyal06/Neuro-Ising)
- R58: [IOPM02/Prob-Greedy-TSP](https://github.com/IOPM02/Prob-Greedy-TSP)
- R59: [kyshan/QPA](https://github.com/kyshan/QPA)
- R60: [DOI](https://doi.org/10.1137/0110015)
- R61: [DOI](https://doi.org/10.1145/290179.290180)
- R62: [Helsgaun (2000), An effective implementation of the Lin–Kernighan traveling salesman heuristic](https://doi.org/10.1016/S0377-2217%2899%2900284-2)
- R63: [DOI](https://doi.org/10.1287/ijoc.1120.0506)
- R64: [DOI](https://doi.org/10.1109/TCAD.2022.3164330)
- R65: [arXiv v1](https://arxiv.org/abs/2407.03927v1)
- R66: [arXiv v1](https://arxiv.org/abs/2505.00803v1)
- R67: [arXiv v1](https://arxiv.org/abs/2501.04072v1)
- R68: [arXiv v2](https://arxiv.org/abs/2501.14285v2)
- R69: [arXiv v1](https://arxiv.org/abs/2608.12772v1)
- R70: [arXiv v1](https://arxiv.org/abs/2608.09042v1)
- R71: [https://doi.org/10.1145/3712256.3726436](https://doi.org/10.1145/3712256.3726436)
- R72: [https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html)
- R73: [http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz](http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz)
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
