# RTX 5090 sparse and dense Max-Cut

English | [简体中文](experiment-report.zh-CN.md)

The sparse C1 panel has **110 natural calls and five work replays**; the dense C3
panel has **20 natural calls and five work replays**. The latter uses four full
32768-variable graphs, two seeds, 16 trajectories, and 90-second search windows.
All instances and parameters were frozen before this device confirmation.

NVIDIA GeForce RTX 5090 (32607 MiB); AMD Eng Sample: 100-000001537-04 (quota 17.56 vCPU); Python 3.14.6; PyTorch 2.13.0+cu130; CUDA 13.0; driver 580.178.04.

This device confirmation reuses previously measured instances and seeds.

## Sparse panel

| Instance | Method | Calls | Mean cut | Mean complete runner (s) |
|---|---|---:|---:|---:|
| G1 | BURER2002 | 2 | 11624.000 | 10.188 |
| G1 | FESTA2002VNS | 2 | 11604.000 | 10.219 |
| G1 | metropolis-anneal | 2 | 11624.000 | 10.351 |
| G1 | pbit-anneal | 2 | 11624.000 | 10.380 |
| G1 | pbit-pt | 2 | 11622.500 | 10.354 |
| G22 | BURER2002 | 2 | 13346.000 | 10.164 |
| G22 | FESTA2002VNS | 2 | 13216.000 | 10.225 |
| G22 | metropolis-anneal | 2 | 13356.500 | 10.358 |
| G22 | pbit-anneal | 2 | 13353.000 | 10.396 |
| G22 | pbit-pt | 2 | 13319.500 | 10.360 |
| G55 | BURER2002 | 2 | 10244.500 | 15.155 |
| G55 | FESTA2002VNS | 2 | 10028.500 | 15.239 |
| G55 | metropolis-anneal | 2 | 10257.000 | 15.321 |
| G55 | pbit-anneal | 2 | 10242.000 | 15.392 |
| G55 | pbit-pt | 2 | 10256.500 | 15.353 |
| G81 | BURER2002 | 2 | 13754.000 | 15.254 |
| G81 | FESTA2002VNS | 2 | 12903.000 | 15.372 |
| G81 | metropolis-anneal | 2 | 13863.000 | 15.523 |
| G81 | pbit-anneal | 2 | 13791.000 | 15.556 |
| G81 | pbit-pt | 2 | 13949.000 | 15.506 |
| n32768-positive-i0 | BURER2002 | 2 | 566440.500 | 15.519 |
| n32768-positive-i0 | FESTA2002VNS | 2 | 554322.000 | 15.539 |
| n32768-positive-i0 | metropolis-anneal | 2 | 566727.500 | 15.863 |
| n32768-positive-i0 | pbit-anneal | 2 | 565672.000 | 15.898 |
| n32768-positive-i0 | pbit-pt | 2 | 566608.000 | 15.852 |
| n32768-positive-i1 | BURER2002 | 2 | 565657.000 | 15.460 |
| n32768-positive-i1 | FESTA2002VNS | 2 | 553607.000 | 15.705 |
| n32768-positive-i1 | metropolis-anneal | 2 | 565373.500 | 15.937 |
| n32768-positive-i1 | pbit-anneal | 2 | 564360.000 | 15.894 |
| n32768-positive-i1 | pbit-pt | 2 | 565602.500 | 15.869 |
| n32768-signed-i0 | BURER2002 | 2 | 206084.000 | 15.508 |
| n32768-signed-i0 | FESTA2002VNS | 2 | 194749.000 | 15.567 |
| n32768-signed-i0 | metropolis-anneal | 2 | 206524.500 | 15.892 |
| n32768-signed-i0 | pbit-anneal | 2 | 205533.000 | 15.940 |
| n32768-signed-i0 | pbit-pt | 2 | 206642.000 | 15.896 |
| n32768-signed-i1 | BURER2002 | 2 | 206365.500 | 15.494 |
| n32768-signed-i1 | FESTA2002VNS | 2 | 194145.500 | 15.566 |
| n32768-signed-i1 | metropolis-anneal | 2 | 206406.000 | 15.912 |
| n32768-signed-i1 | pbit-anneal | 2 | 205318.500 | 15.963 |
| n32768-signed-i1 | pbit-pt | 2 | 206223.500 | 15.893 |

The table contains the 32K and Gset cases; smaller correctness/development cases
remain in the complete data. Nominal graph size and objective-active vertices
are separate. Sparse GPU methods use 16 trajectories and CPU references one
process, so parallel resources are not identical. BURER2002 is the unmodified primary CPU control; FESTA2002VNS
includes the documented sign/boolean correction and retains its visitation-order
limitation. Classical controls are not p-bit updates.
Single-site heat bath has a strictly higher mean cut than BURER2002 on
0/4 of the 32K graphs; the other variants'
finite per-graph outcomes remain visible above.

## Dense panel

| Instance | Variant | Calls | Mean negative energy | Difference from same-GPU dSB | Mean complete runner (s) |
|---|---|---:|---:|---:|---:|
| er32768-0 | block8x16 | 2 | 1997011.000 | -9.315% | 123.302 |
| er32768-0 | dsb0125 | 2 | 2202141.000 | 0.000% | 121.684 |
| er32768-1 | block8x16 | 2 | 1998270.000 | -9.142% | 121.028 |
| er32768-1 | dsb0125 | 2 | 2199335.000 | 0.000% | 121.683 |
| sk32768-0 | dsb100 | 2 | 4503404.000 | 0.000% | 124.179 |
| sk32768-0 | onsager | 2 | 4501345.000 | -0.046% | 124.884 |
| sk32768-0 | sca-matched | 2 | 4352836.000 | -3.343% | 132.274 |
| sk32768-1 | dsb100 | 2 | 4505052.000 | 0.000% | 122.963 |
| sk32768-1 | onsager | 2 | 4502982.000 | -0.046% | 125.673 |
| sk32768-1 | sca-matched | 2 | 4354314.000 | -3.346% | 124.173 |

Negative energy is $2C-\sum w$. Reporting only raw positive-graph cut would hide
relative optimization differences. dSB is a continuous classical control with
zero p-bits; SCA/Onsager are synchronous heuristics, distinct from exact conditional
block Gibbs. The dense Torch allocation peak was 4.546 GiB.

The predeclared 1% gate requires improvement on both graphs of one family.
Observed family gates: **sk=false, er=false**. No extra 32K CPU comparison was added;
any gate crossing remains a bounded same-GPU observation rather than a claim of
general classical superiority. Large-instance optima remain unknown.

![Dense same-GPU quality comparison](figures/rtx5090.png)

[Method](README.md) · [Sparse data](../../data/maxcut-sparse/rtx5090/calls.json) ·
[Dense data](../../data/maxcut-dense/rtx5090/calls.json) · [Reproduction](../reproduction.md)

## References and software

The linked inventory includes background and excluded comparisons used during the
research. Citation does not imply replication. Version/access limits in the method
guide remain applicable; this release did not perform a new full-text literature review.

- R1: [https://arxiv.org/abs/1805.05217](https://arxiv.org/abs/1805.05217)
- R2: [MQLib](https://github.com/MQLib/MQLib)
- R3: [https://web.stanford.edu/~yyye/yyye/Gset/](https://web.stanford.edu/~yyye/yyye/Gset/)
- R4: [https://www.nature.com/articles/s41598-025-90520-3](https://www.nature.com/articles/s41598-025-90520-3)
- R5: [https://github.com/nonizawa/GPU-pSAv](https://github.com/nonizawa/GPU-pSAv)
- R6: [Programmable Probabilistic Computer with 1,000,000 p-bits](https://arxiv.org/abs/2606.25313v1)
- R7: [Hardware Acceleration of Probabilistic Computing for Max-Cut Problems](https://ieeexplore.ieee.org/document/11668769/)
- R8: [High-performance combinatorial optimization based on classical mechanics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11323291/)
- R9: [DOI](https://doi.org/10.1126/sciadv.abe7953)
- R10: [https://arxiv.org/abs/2610.13368v1](https://arxiv.org/abs/2610.13368v1)
- R11: [https://arxiv.org/html/2610.13368v1](https://arxiv.org/html/2610.13368v1)
- R12: [https://github.com/skethz/anechoic](https://github.com/skethz/anechoic)
- R13: [https://github.com/hariby/SA-complete-graph/tree/WK2000](https://github.com/hariby/SA-complete-graph/tree/WK2000)
- R14: [Edge-of-chaos enhanced quantum-inspired algorithm for combinatorial optimization](https://arxiv.org/abs/2508.17655)
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
