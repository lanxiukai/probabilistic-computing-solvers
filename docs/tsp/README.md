# Symmetric travelling salesperson problem

English | [简体中文](README.zh-CN.md)

The primary released workflow uses **GPU p-bit heat-bath updates on candidate
arcs**, with classical initialization, matching, subtour repair and local search.
See the [method and reproduction](candidate-network.md) and the
[measured RTX 5090 report](candidate-report.md). Seven instances span 200–2103
cities and up to 33,662 binary nodes per model; eight replicas are counted
separately. No call improved its LKH prefix, and the 200-city optimum was already
present in that prefix. These are hybrid outcomes, not evidence of an independent
p-bit benefit or a full-TSP Ising ground state.

Standalone LKH and GA-EAX are classical references. The
[u1817 CPU certificate](../../data/tsp/candidate-network/u1817-optimum.json) belongs
to GA-EAX. New cloud measurements include only the p-bit workflow, its three
attribution controls and a separate origin replay.

Source: `src/combinatorial_solvers/tsp/`. The package also retains permutation,
one-hot and AB-cycle interfaces. The [earlier 150-city report](experiment-report.md)
preserves its original CPU p-bit selector/GPU local-search method and denominator;
its results are separate from this candidate-network study. The synthetic `tsp`
example exercises that earlier certificate pipeline without TSPLIB or a native
solver. The measured candidate workflow uses the explicit method-guide command.

Use the [common reproduction instructions](../reproduction.md) and
[external-reference inventory](../external-baselines.md). Obtain the exact
external inputs and native sources under their authors' terms. Native binaries,
benchmark coordinates and paper files are not redistributed.

## References

These links preserve the research source inventory, including background and unimplemented comparisons. Citation does not imply replication or a new full-text review. Pinned baseline versions are in the baseline inventory.

- R1: [DOI](https://doi.org/10.1287/opre.2.4.393)
- R2: [publications.csail.mit.edu/lcs/pubs/pdf/MIT-LCS-TM-209.pdf](https://publications.csail.mit.edu/lcs/pubs/pdf/MIT-LCS-TM-209.pdf)
- R3: [doi.org/10.1007/s43069-021-00101-z](https://doi.org/10.1007/s43069-021-00101-z)
- R4: [arXiv v4](https://arxiv.org/abs/1708.04215v4)
- R5: [JACM](https://doi.org/10.1145/3424306)
- R6: [arXiv v6](https://arxiv.org/abs/2007.01409v6)
- R7: [doi.org/10.1287/opre.2022.2338](https://doi.org/10.1287/opre.2022.2338)
- R8: [arXiv v1](https://arxiv.org/abs/2212.06296v1)
- R9: [www.or.uni-bonn.de/tspbook/book.pdf](https://www.or.uni-bonn.de/tspbook/book.pdf)
- R10: [Cambridge](https://doi.org/10.1017/9781009445436)
- R11: [www.math.uwaterloo.ca/~bico/papers/tsp_icm.pdf](https://www.math.uwaterloo.ca/~bico/papers/tsp_icm.pdf)
- R12: [www.math.uwaterloo.ca/~bico/papers/clk_ijoc.pdf](https://www.math.uwaterloo.ca/~bico/papers/clk_ijoc.pdf)
- R13: [DOI](https://doi.org/10.1007/s12532-009-0004-6)
- R14: [www.math.uwaterloo.ca/tsp/data/art/tours/GA_tsp_art.pdf](https://www.math.uwaterloo.ca/tsp/data/art/tours/GA_tsp_art.pdf)
- R15: [www.math.uwaterloo.ca/tsp/concorde/](https://www.math.uwaterloo.ca/tsp/concorde/)
- R16: [www.math.uwaterloo.ca/tsp/pla85900/index.html](https://www.math.uwaterloo.ca/tsp/pla85900/index.html)
- R17: [DOI](https://doi.org/10.1145/2330784.2330978)
- R18: [arXiv v1](https://arxiv.org/abs/2003.11902v1)
- R19: [doi.org/10.1016/j.future.2020.01.011](https://doi.org/10.1016/j.future.2020.01.011)
- R20: [arXiv v2](https://arxiv.org/abs/2404.04895v2)
- R21: [doi.org/10.1145/3638530.3654394](https://doi.org/10.1145/3638530.3654394)
- R22: [arXiv v1](https://arxiv.org/abs/2608.21233v1)
- R23: [arXiv v3](https://arxiv.org/abs/1803.08475v3)
- R24: [arXiv v3](https://arxiv.org/abs/2010.16011v3)
- R25: [arXiv v1](https://arxiv.org/abs/2110.07983v1)
- R26: [arXiv v2](https://arxiv.org/abs/2207.03876v2)
- R27: [arXiv v2](https://arxiv.org/abs/2302.08224v2)
- R28: [DOI](https://doi.org/10.1126/science.3755256)
- R29: [DOI](https://doi.org/10.3389/fphy.2014.00005)
- R30: [doi.org/10.1103/plr8-xys1](https://doi.org/10.1103/plr8-xys1)
- R31: [doi.org/10.1038/s41928-023-01065-0](https://doi.org/10.1038/s41928-023-01065-0)
- R32: [doi.org/10.1038/s41467-024-47818-z](https://doi.org/10.1038/s41467-024-47818-z)
- R33: [doi.org/10.1038/s44335-026-00060-w](https://doi.org/10.1038/s44335-026-00060-w)
- R34: [doi.org/10.1038/s41467-025-66864-9](https://doi.org/10.1038/s41467-025-66864-9)
- R35: [arxiv.org/abs/2504.13294v1](https://arxiv.org/abs/2504.13294v1)
- R36: [DOI](https://doi.org/10.1109/LES.2023.3298739)
- R37: [arXiv v1](https://arxiv.org/abs/2607.21182v1)
- R38: [doi.org/10.1038/s41467-023-41647-2](https://doi.org/10.1038/s41467-023-41647-2)
- R39: [doi.org/10.1038/s41467-022-29411-4](https://doi.org/10.1038/s41467-022-29411-4)
- R40: [arXiv v1](https://arxiv.org/abs/2211.15869v1)
- R41: [arXiv v1](https://arxiv.org/abs/2202.08939v1)
- R42: [doi.org/10.3390/math10081294](https://doi.org/10.3390/math10081294)
- R43: [arXiv v2](https://arxiv.org/abs/2502.17725v2)
- R44: [arXiv v1](https://arxiv.org/abs/2604.25531v1)
- R45: [arXiv v4](https://arxiv.org/abs/2504.02388v4)
- R46: [doi.org/10.1145/3712255.3734313](https://doi.org/10.1145/3712255.3734313)
- R47: [doi.org/10.1587/transcom.2020NVI0002](https://doi.org/10.1587/transcom.2020NVI0002)
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
- R71: [doi.org/10.1145/3712256.3726436](https://doi.org/10.1145/3712256.3726436)
- R72: [comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html)
- R73: [webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz](http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz)
