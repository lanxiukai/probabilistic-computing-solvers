# RTX 5090 evaluation of cubic spin-glass optimization

English | [简体中文](experiment-report.zh-CN.md)

Measured on 2026-10-19. This report evaluates implementation speed and finite-time
optimization, not physical dynamics. It supports two bounded findings: the v5
execution backend reduces fixed-work cost relative to v1, and method ranking
changes with lattice size on the stated panel. All three competitors are our
implementations; no external state-of-the-art advantage is established.

## Model and methods

Minimize

$$H(s)=-\sum_{(i,j)\in E}J_{ij}s_i s_j,\qquad s_i,J_{ij}\in\{-1,+1\},$$

on a complete open cubic lattice with zero field and zero constant.
$N=L^3$, $M=3L^2(L-1)$; coordinates use $i=x+Ly+L^2z$. At $L=32$ this is
32,768 logical spins and 95,232 distinct bonds. The published interface accepts
$2\le L\le32$ and rejects defective, periodic, embedded, non-unit-weight and
field-bearing inputs. Internal geometry utilities do not broaden the measured domain.

PT uses two-color heat-bath updates, $P(s_i=+1)=1/(1+e^{-2\beta h_i})$, alternating
neighboring temperature exchanges, and ICM moves on disagreement clusters.
ICM preserves the sum of the paired configurations' energies, not each energy
individually. This fixed geometric-temperature PT is not adaptive PT.
SA uses two-color Metropolis updates and repeats its temperature schedule.
PA adds within-group systematic resampling with weights proportional to
$e^{-\Delta\beta E}$ and independent population restarts between cycles.
Each systematic resampling event uses one uniform offset per group. Lineage
coalescence and ESS diagnostics do not establish equilibrium or independent particles.

The v5 backend fuses local fields, probability lookup and spin writes, with
tiled integer energy reduction. Random draws use FP64. ICM includes CPU work;
the complete sampling interval includes that work and its transfers. Fourteen
runtime files are byte-identical to the evaluated source, as recorded in
[provenance](../../provenance/spin-glass/release-v1.json).

## Hardware and frozen evaluation

One NVIDIA GeForce RTX 5090, 32,607 MiB VRAM, driver 615.71.09, compute capability
12.0; AMD EPYC 9354 host, container CPU quota about 15.36 cores, one Torch CPU
thread. Runtime: Python 3.14.6, Torch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1.
Allocated GPU memory excludes driver/context usage and reserved allocator cache.

The [panel](../../configs/spin-glass/s23-panel.json) contains 42 random coupling
instances: two development and twelve held-out instances at each of $L=16,24,32$,
plus an installation/correctness fixture. Inputs were frozen before tuning,
without screening. Couplings use NumPy PCG64 and seed
`202610190000 + 1000 * length + instance_id`; explicit bond files are authoritative.

| Phase | Planned successful logical calls | Evaluation |
|---|---:|---|
| Preflight | 16 | Fixture, same-device replay and v1/v5 fixed-work comparisons |
| Calibration | 36 | Two development instances, two candidates per method/size, 60 seconds |
| Confirmation | 216 | Twelve held-out instances, two seeds, three frozen methods, 60 seconds |
| Full-origin replay | 3 | One specified 32³ source trajectory per method |

Calibration seed: 2026101901; confirmation seeds: 2026101902 and 2026101903;
fixed-work seeds: 2026101904 and 2026101905; fixture seed: 2026101906.
Development-only selection minimizes mean instance energy, then the worst
instance mean, then candidate order. PT exchange intervals 4/2, SA cycles
16384/8192 and PA cycles 4096/2048 are the complete tuning set. The selected PT
interval is 2 at 16³ and 4 at 24³/32³; ICM interval is 128. Selected SA and PA
cycles are 16384 and 4096. No confirmation-driven retuning occurred.

Each call has four groups. PT/SA use 96 temperatures × 2 copies per group
(768 total states); PA uses 128 × 4 particles per group (2048 total states).
Equal wall sampling time is not equal particle count or update count.
Checkpoints are 1, 5, 15, 30 and 60 seconds. The primary value excludes
improvements first observed after 60 seconds, even if the final sweep finishes later.

The independent statistical unit is the coupling instance: average the two
seeds and four groups within each instance, then weight instances equally.
Each method/size therefore has 24 calls and 96 groups, but twelve independent
coupling instances. Paired percentile bootstrap intervals use 20,000 resamples
and seed 2026101907. Replays, particles and software checks add no independent trials.

## Fixed-work execution improvement

At 32³, one development coupling, two seeds and 1024 sweeps, v1/v5 matched all
recorded states and RNG. Times below are means across those two seeds.

| Method | v1 sampling, s | v5 sampling, s | Sampling reduction | v1 full call, s | v5 full call, s | Full-call reduction |
|---|---:|---:|---:|---:|---:|---:|
| PT+ICM | 13.8850 | 4.3909 | 68.38% | 21.0950 | 12.1197 | 42.55% |
| Cyclic SA | 11.6424 | 1.6046 | 86.22% | 18.1841 | 8.7092 | 52.11% |
| Systematic PA | 31.5422 | 4.4928 | 85.76% | 39.2655 | 12.9466 | 67.03% |

This is an execution comparison of unchanged numerical methods, not a hardware
or algorithmic advantage over the literature. Preflight included 97 passing
software/GPU checks and seven comparisons covering 157 identical arrays plus RNG.
The 1024-sweep PA test does not cover a complete annealing cycle.

## Held-out energy at 60 seconds

Lower energy is better; these are instance-weighted averages, not best-of-panel minima.

| L | Logical spins | PT+ICM | Cyclic SA | Systematic PA |
|---|---:|---:|---:|---:|
| 16 | 4096 | −7065.1667 | −7054.8750 | −7061.3333 |
| 24 | 13824 | −24037.2083 | −24046.1250 | −24057.2917 |
| 32 | 32768 | −57080.4375 | −57227.9792 | −57197.4167 |

| L | Method minus SA | Mean difference | Paired bootstrap 95% interval | Better/tied/worse instances |
|---|---|---:|---|---|
| 16 | PT+ICM | −10.2917 | [−12.6250, −8.2083] | 12/0/0 |
| 16 | PA | −6.4583 | [−7.6458, −5.3125] | 12/0/0 |
| 24 | PT+ICM | +8.9167 | [6.0833, 11.6458] | 0/0/12 |
| 24 | PA | −11.1667 | [−13.3125, −9.2083] | 12/0/0 |
| 32 | PT+ICM | +147.5417 | [142.2286, 152.8125] | 0/0/12 |
| 32 | PA | +30.5625 | [26.6667, 34.1458] | 0/0/12 |

![Coupling-level paired effects](figures/paired-difference.png)

![Observed energy per spin against sampling time](figures/energy-time.png)

Ranking changes with size and time budget; the 16³ PT result does not imply a
general p-bit advantage. Confidence intervals describe this finite coupling
panel, not arbitrary-instance performance or ground-state guarantees.

## Failure, recovery and timing boundaries

There were **272 actual attempts for 271 successful logical calls**. One 32³ PA
call exceeded its 240-second full-call limit and produced no usable state bundle.
Its sampling progress and failure location are unknown. The controller stopped;
one isolated operator retry used the same instance, seed, numerical code,
configuration and 60-second sampling limit, within the original eight-hour cap.
The original failed attempt costs 248.4195 seconds including termination grace.
It remains in the call table; the retry adds no independent coupling or cohort.

The recovered confirmation panel thus contains 217 attempts for 216 logical
calls. It is not a uniformly uninterrupted single-attempt comparison. As a
post-incident sensitivity analysis, omit instance i9 at 32³ from all three
methods. Across the remaining eleven instances, PT minus SA is +147.2500
[141.5000, 152.9545], and PA minus SA is +30.9318 [26.7045, 34.6369]; SA is lower
in 11/11 instances in both comparisons. The full panel remains the main table.

At 32³ the successful-call mean full return times are PT 70.6063 s, SA 74.1169 s,
PA 69.3892 s. These retain slow returns of 128.3985 s for PT and 194.8456 s for SA.
Charging the failed PA attempt across its 24 logical confirmation calls gives
79.7400 s per logical call. GPU allocated peaks are 490.58, 396.19 and 1546.98 MiB.
Sampling, preparation, state export and full child-process time remain separate;
17555.706 s is the sum of all call times, not total elapsed experiment time.
Two 24³ PT groups and one 32³ PT group had late improvements excluded at 60 seconds.

## Evidence and limitations

[Core data](../../data/spin-glass/s23/manifest.json) includes every attempted call,
group checkpoint values, configurations, explicit couplings, RNG records and
6229 array fingerprints. The archived audit checked all raw arrays and ten
required same-device comparisons: 226 identical arrays plus RNG, including
the three full-origin replays; 72 confirmation blocks had matched PT/SA initial
states. Public verification checks those recorded fingerprints and recomputes
the statistics; it does not re-read the separately retained full raw trajectories.

The 108 portable witnesses are one best observed state per method and held-out
coupling, selected across **two successful 60-second calls with four groups each**.
That is 120 seconds of successful method sampling per witness selection, plus
the separately recorded failure cost where applicable. Integer recomputation
checks $H=-M+2U$, with $U$ the number of unsatisfied bonds. It certifies energy and
model identity, not optimality or the truth of an observation timestamp.

All large-instance optima are unknown: no ground-state hit rate or TTS99 is
reported. There is no frozen external target for time-to-target comparison.
No FPGA/p-bit hardware, quantum advantage, cross-GPU bitwise equivalence or
equilibrium-sampling claim follows from these GPU optimization measurements.

## Comparison with prior work

Chowdhury et al. and the pinned OPUSLab APT implementation are direct p-bit
comparison candidates, but their adaptive ladder, four replicas per temperature,
and MATLAB parallel execution differ from this fixed-ladder implementation.
Their author implementation was not run here. Optimized SA (Isakov et al.) and
PA/SA/PT comparisons (Wang et al.) motivate stronger external controls; the
latter's Gaussian panel cannot be merged with these ±1 results.

Multispin GPU work by Fang et al. and Lulli et al. reports execution methods;
spin-flip throughput does not directly establish time to the same optimization
quality. Learning-assisted approaches additionally require accounting for
training and preprocessing. Planar, quasi-two-dimensional, SK and native
annealer graphs are distinct instance domains. In particular, native-annealer
variable counts or energy density are not a matched complete-cubic benchmark.
The references below retain background and excluded directions; citation does
not imply replication, adoption or comparable instances.

## Complete linked bibliography


- [Chowdhury et al. (2025), Pushing the boundary of quantum advantage in hard combinatorial optimization with probabilistic computers](https://doi.org/10.1038/s41467-025-64235-y); [author code and instances](https://github.com/OPUSLab/3DSpinGlassWithPbits). Comparison candidate, not a replicated benchmark.
- [Zhu, Ochoa and Katzgraber (2015), Efficient Cluster Algorithm for Spin Glasses in Any Space Dimension](https://doi.org/10.1103/PhysRevLett.115.077201).
- [Wang, Machta and Katzgraber (2015), Comparing Monte Carlo methods for finding ground states of Ising spin glasses: population annealing, simulated annealing and parallel tempering](https://doi.org/10.1103/PhysRevE.92.013303). The published Gaussian-disorder panel is not our bimodal panel.
- [Isakov et al. (2015), Optimised simulated annealing for Ising spin glasses](https://doi.org/10.1016/j.cpc.2015.02.015). Implementation baseline to evaluate separately.
- [Lulli, Bernaschi and Parisi (2015), Highly optimized simulations on single- and multi-GPU systems of 3D Ising spin glass](https://doi.org/10.1016/j.cpc.2015.06.019). Spin-update throughput is not time to an optimization target.

Runtime dependencies: [Python](https://www.python.org/), [NumPy](https://numpy.org/),
[SciPy](https://scipy.org/), [PyTorch](https://pytorch.org/), [Triton](https://triton-lang.org/);
environment management: [uv](https://docs.astral.sh/uv/).

### Additional research and implementation sources

The list retains background and excluded approaches used during development. Citation does not imply adoption, replication, or comparable problem instances. Version labels identify the archived author manuscript where recorded.

- [Kirkpatrick, 1983. Optimization by Simulated Annealing](https://doi.org/10.1126/science.220.4598.671).
- [Lucas, 2014. Ising formulations of many NP problems](https://doi.org/10.3389/fphy.2014.00005).
- [Barahona, 1982. On the computational complexity of Ising spin glass models](https://doi.org/10.1088/0305-4470/15/10/028).
- [Houdayer, 2001. A Cluster Monte Carlo Algorithm for 2-Dimensional Spin Glasses](https://doi.org/10.1007/PL00011151).
- [Katzgraber, 2006. Feedback-optimized parallel tempering Monte Carlo](https://doi.org/10.1088/1742-5468/2006/03/P03018).
- [Liu, 2015. Universal dynamic scaling in three-dimensional Ising spin glasses](https://doi.org/10.1103/PhysRevE.92.022128).
- [Mohseni, 2021. Nonequilibrium Monte Carlo for unfreezing variables in hard combinatorial optimization](https://doi.org/10.48550/arXiv.2111.13628).
- [King, 2023. Quantum critical dynamics in a 5,000-qubit programmable spin glass](https://doi.org/10.1038/s41586-023-05867-2).
- [Zhang, 2024. Cyclic quantum annealing: searching for deep low-energy states in 5000-qubit spin glass](https://doi.org/10.1038/s41598-024-80761-z).
- [Zhang, 2025. Computational complexity of three-dimensional Ising spin glass: Lessons from D-wave annealer](https://doi.org/10.1103/3bkn-v5rd).
- [Del Bono et al., 2026. Demonstrating Real Advantage of Machine-Learning-Enhanced Monte Carlo for Combinatorial Optimization](https://doi.org/10.1073/pnas.2534768123) — archived author version: `2510.19544v2`.
- [Houdayer, 2001. A hierarchical approach for computing spin glass ground states](https://doi.org/10.1103/PhysRevE.64.056704) — archived author version: `cond-mat/0105617v1`.
- [Boettcher, 2001. Optimization with Extremal Dynamics](https://doi.org/10.1103/PhysRevLett.86.5211) — archived author version: `cond-mat/0010337v2`.
- [Liers, 2003. Ground state of the Bethe-lattice spin glass and running time of an exact optimization algorithm](https://doi.org/10.1103/PhysRevB.68.094406) — archived author version: `cond-mat/0212630v3`.
- [Pardella, 2008. Exact Ground States of Large Two-Dimensional Planar Ising Spin Glasses](https://doi.org/10.1103/PhysRevE.78.056705) — archived author version: `0801.3143v3`.
- [Selby, 2014. Efficient subgraph-based sampling of Ising-type models with frustration](https://doi.org/10.48550/arXiv.1409.3934) — archived author version: `1409.3934v1`.
- [Hen, 2015. Probing for quantum speedup in spin glass problems with planted solutions](https://doi.org/10.1103/PhysRevA.92.042325) — archived author version: `1502.01663v3`.
- [Hamerly, 2019. Experimental investigation of performance differences between Coherent Ising Machines and a quantum annealer](https://doi.org/10.1126/sciadv.aau0823) — archived author version: `1805.05217v3`.
- [Montanari, 2019. Optimization of the Sherrington-Kirkpatrick Hamiltonian](https://doi.org/10.48550/arXiv.1812.10897) — archived author version: `1812.10897v2`.
- [Hibat-Allah et al., 2021. Variational Neural Annealing](https://doi.org/10.1038/s42256-021-00401-3) — archived author version: `2101.10154v1`.
- [Śmierzchalski et al., 2025. SpinGlassPEPS.jl: Tensor-network package for Ising-like optimization on quasi-two-dimensional graphs](https://doi.org/10.1016/j.softx.2025.102257) — archived author version: `2502.02317v1`.
- [Dziubyna, 2025. Limitations of tensor network approaches for optimization and sampling: A comparison to quantum and classical Ising machines](https://doi.org/10.1103/PhysRevApplied.23.054049) — archived author version: `2411.16431v3`.
- [Duxbury, 2026. A Continuous Nonlinear Optimization Perspective on the Spin Glass Problem](https://doi.org/10.1016/j.physa.2026.131356) — archived author version: `2512.05852v1`.
- [Gessert, Janke and Weigel (2023), Resampling schemes in population annealing: Numerical and theoretical results; author manuscript v2 (2024)](https://arxiv.org/abs/2305.19994v2).
- [Fang et al. (2014), Parallel Tempering Simulation of the three-dimensional Edwards-Anderson Model with Compact Asynchronous Multispin Coding on GPU](https://arxiv.org/abs/1311.5582).
- [Hasselgren et al. (2025), Probabilistic Computing Optimization of Complex Spin-Glass Topologies; preprint, abstract-level review only](https://arxiv.org/abs/2510.23419).
- [Goto et al. (2019), Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems; no archived full PDF](https://www.science.org/doi/10.1126/sciadv.aav2372).
- [Zhang and Kamenev, 3D Ising Spin Glass Solutions, v1.0.1; native annealer graph, not a complete cubic panel](https://zenodo.org/records/16547191).
- [Del Bono et al., MLMC_optimization; archived source revision 394d47ab9d0514b6cb1c7f764113c579ab8f3eb6](https://github.com/Laplaxe/MLMC_optimization/tree/394d47ab9d0514b6cb1c7f764113c579ab8f3eb6).
- [OPUSLab, 3DSpinGlassWithPbits, APT.m; reviewed revision e91cf095a2a92a8a90d516a59520e3298270f281](https://github.com/OPUSLab/3DSpinGlassWithPbits/blob/e91cf095a2a92a8a90d516a59520e3298270f281/APT.m).
- [arxiv.org/abs/1412.2104](https://arxiv.org/abs/1412.2104).
- [arxiv.org/abs/1501.05630v2](https://arxiv.org/abs/1501.05630v2).
- [arxiv.org/src/1401.1084v4/anc/an_ms_r1_nf.h](https://arxiv.org/src/1401.1084v4/anc/an_ms_r1_nf.h).
- [docs.nvidia.com/deploy/nvidia-smi/index.html](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
- [PyTorch 2.13: generated/torch.autograd.profiler.record_function.html](https://docs.pytorch.org/docs/2.13/generated/torch.autograd.profiler.record_function.html).
- [PyTorch 2.13: generated/torch.multinomial.html](https://docs.pytorch.org/docs/2.13/generated/torch.multinomial.html).
- [PyTorch 2.13: generated/torch.searchsorted.html](https://docs.pytorch.org/docs/2.13/generated/torch.searchsorted.html).
- [PyTorch 2.13: profiler.html](https://docs.pytorch.org/docs/2.13/profiler.html).
- [docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html).
- [docs.vast.ai/api-reference/instances/destroy-instance](https://docs.vast.ai/api-reference/instances/destroy-instance).
- [filterpy.readthedocs.io/en/latest/_modules/filterpy/monte_carlo/resampling.html](https://filterpy.readthedocs.io/en/latest/_modules/filterpy/monte_carlo/resampling.html).
- [NumPy 2.4: numpy.savez.html](https://numpy.org/doc/2.4/reference/generated/numpy.savez.html).
- [NumPy 2.4: numpy.savez_compressed.html](https://numpy.org/doc/2.4/reference/generated/numpy.savez_compressed.html).
- [Triton: triton.jit.html](https://triton-lang.org/main/python-api/generated/triton.jit.html).
- [Triton: triton.language.load.html](https://triton-lang.org/main/python-api/generated/triton.language.load.html).
- [Triton: triton.language.static_range.html](https://triton-lang.org/main/python-api/generated/triton.language.static_range.html).
- [Triton: triton.language.sum.html](https://triton-lang.org/main/python-api/generated/triton.language.sum.html).
- [www.nature.com/articles/s41467-025-64235-y](https://www.nature.com/articles/s41467-025-64235-y).

- [Matplotlib](https://matplotlib.org/) — published figure rendering.
- [pytest](https://docs.pytest.org/) — software regression checks.
- [Ruff](https://docs.astral.sh/ruff/) — source lint and formatting checks.
