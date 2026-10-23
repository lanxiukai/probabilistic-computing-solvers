# Fixed-row covering arrays on an RTX 5090

[简体中文](report.zh-CN.md) · [Overview](README.md) · [Method](method.md) · [Source guide](source-guide.md)

## Research question and scope

Can a direct pair-coverage factor model with two-row same-column p-bit heat-bath updates construct complete arrays at fixed row counts, and does the thermal rule help compared with controls using the same neighborhood? The application is generation of pairwise test combinations. Coverage establishes a mathematical interaction property, not software defect-detection effectiveness [R1].

Each state has $Nkv$ persistent one-hot binary entries. Energy is the original missing-interaction count. The workflow starts with independent uniform array entries and has no classical optimization prefix. Its complete outcome and its improvement over initialization are reported separately from any measured advantage over matched controls. The [method](method.md) gives the exact energy, joint conditional rule and independent certificate.

## Frozen confirmation

The route was selected before these confirmation seeds: same-column two-row blocks, constant $\beta=8$, 128 replicas, full-chunk warmup and cached scalar parameters. Five homogeneous strength-two inputs use $v=3$:

| Input | $N$ | $k$ | Binary entries per replica | Native search limit | Evaluation role |
|---|---:|---:|---:|---:|---|
| `ca9k4v3` | 9 | 4 | 108 | 1 s | Previously developed dimension, fresh seeds |
| `ca18k20v3` | 18 | 20 | 1,080 | 15 s | Previously developed dimension, fresh seeds |
| `ca35k300v3` | 35 | 300 | 31,500 | 30 s | Previously developed dimension, fresh seeds |
| `ca30k300v3` | 30 | 300 | 27,000 | 45 s | Previously developed dimension, fresh seeds |
| `ca25k100v3` | 25 | 100 | 7,500 | 20 s | Previously untuned dimension |

Seeds are `2026102361`, `2026102362`, `2026102363`. Each input/seed has one heat-bath, one Metropolis and one plateau call; the three share the same initialization. This gives 45 native calls: 15 heat bath and 30 matched controls. Each input also has one ACTS Basic/IPOG constructive reference, giving 50 planned natural calls. The sum of native search limits is 999 s. That sum excludes preparation, certification and other runner overhead.

The three planned fixed-work replays are the first-seed heat-bath calls on `ca9k4v3`, `ca35k300v3` and `ca30k300v3`. They add no independent trials. All declared calls, including misses, failures and unexecuted calls if any, must remain in the final denominator. There is no confirmation-based seed removal, retuning or row-count expansion. Per-seed calls are the units of repeated evaluation; 128 parallel replicas within a call must not be pooled with development settings or replayed seeds as new confirmation runs.

The 35-row, 300-column target comes from NIST's historical IPOG-F construction table [R2]. The 18-row and 30-row targets were set in this study's development. The held-out 25-row, 100-column target was declared before confirmation and is not a literature optimum. No historical solution array was used as a starting point. For $v=3$, the elementary lower bound is $N\ge9$; only the nine-row small input can close it in this panel.

## Independent confirmation results

The complete [call ledger](../../data/covering-array/rtx5090/calls.json) and [panel](../../data/covering-array/rtx5090/panel.json) retain all 50 natural calls and three replays, with no failed or unexecuted calls. Triplets follow the declared seed order; positive $E$ denotes an incomplete array.

| Input | Heat-bath eligible $E$, three seeds | Heat-bath zero hits, each /128 | Metropolis eligible $E$, three seeds | Plateau eligible $E$, three seeds | ACTS rows / fixed target met |
|---|---|---|---|---|---|
| `ca9k4v3` | 0/0/0 | 128/128/128 | 0/0/0 | 0/0/0 | 9 / yes |
| `ca18k20v3` | 0/0/0 | 128/128/128 | 0/0/0 | 0/0/0 | 24 / no |
| `ca35k300v3` | 0/0/0 | 128/128/128 | 0/0/0 | 0/0/0 | 39 / no |
| `ca30k300v3` | 0/0/0 | 107/106/105 | 1/1/0 | 0/0/0 | 39 / no |
| `ca25k100v3` | 0/0/0 | 128/128/128 | 0/0/0 | 0/0/0 | 32 / no |

All 15 heat-bath calls construct a zero-missing array. No replica starts at zero; all 1,920 replicas improve their missing count, with 1,854 newly reaching zero. Paired best-energy comparisons give 2 wins, 13 ties and 0 losses against Metropolis, and 0 wins, 15 ties and 0 losses against plateau. At 30 rows and 300 columns, plateau has 128/128 zero hits in every call, compared with 107/106/105 for heat bath and 0/0/3 for Metropolis. This does not demonstrate a heat-bath advantage over plateau. ACTS covers every interaction at its own returned row count but meets this fixed-row target only on the nine-row small input.

| Input | Cold best $E$, three heat-bath seeds | Eligible best $E$ |
|---|---|---|
| `ca9k4v3` | 12/9/12 | 0/0/0 |
| `ca18k20v3` | 139/145/125 | 0/0/0 |
| `ca35k300v3` | 5220/5571/5475 | 0/0/0 |
| `ca30k300v3` | 9544/10058/10315 | 0/0/0 |
| `ca25k100v3` | 1865/1779/1949 | 0/0/0 |

At 29 rows and 300 columns, the development setting with 256 replicas and a 90 s limit leaves 34/36 missing interactions for heat bath at $\beta=8/12$, 18 for plateau and 75 for Metropolis. None succeeds; this is not an infeasibility proof. At 30 rows, heat bath at $\beta=4/8/12/16$ yields 25/0/0/0, plateau 0 and Metropolis 1. All negative results are retained separately from fresh-seed confirmation.

The separate [development call ledger](../../data/covering-array/development-ca2b/calls.json) and [development panel](../../data/covering-array/development-ca2b/panel.json) retain all 10 natural calls and one fixed-work replay, including the N30 temperature/control comparisons and all N29 residuals. They use one development seed, are not independent fresh-seed confirmation and do not enter the 50-call/three-replay confirmation denominator. Its complete original evidence passed 64,640 state checks.

## Hardware, timing and evidence

GPU work uses an NVIDIA GeForce RTX 5090 with 128 parallel replicas per native call. ACTS is a separately executed CPU reference with one logical processor assigned. Measured Java is OpenJDK 21.0.12.1. The CPU is AMD EPYC 9354; Python 3.14.6, PyTorch 2.13.0+cu130, Triton 3.7.1, NumPy 2.4.4 and CUDA 13.0 are bound by the [environment record](../../data/covering-array/rtx5090/environment.json). These resources are different and support no hardware-normalized speed claim.

Seconds are means of three natural calls (one call for ACTS); A/R denotes maximum allocated/reserved MiB across that group. The ledger retains every individual measurement.

| Input | Method | Preparation s | Search s | Certificate s | Complete s | Overshoot s | A/R MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| `ca9k4v3` | heatbath | 1.3529 | 1.0004 | 0.0136 | 2.6885 | 0.0004 | 2.001/4.000 |
| `ca9k4v3` | metropolis | 1.3358 | 1.0003 | 0.0140 | 2.6760 | 0.0003 | 2.001/4.000 |
| `ca9k4v3` | plateau | 1.3058 | 1.0002 | 0.0137 | 2.6569 | 0.0002 | 2.001/4.000 |
| `ca9k4v3` | acts-ipog | — | 0.2653 | — | 0.5041 | — | — |
| `ca18k20v3` | heatbath | 1.4532 | 15.0005 | 0.0730 | 17.3251 | 0.0005 | 4.235/26.000 |
| `ca18k20v3` | metropolis | 1.4711 | 15.0098 | 0.0723 | 17.4356 | 0.0098 | 4.235/26.000 |
| `ca18k20v3` | plateau | 1.3982 | 15.0003 | 0.0746 | 17.5617 | 0.0003 | 4.235/26.000 |
| `ca18k20v3` | acts-ipog | — | 0.3207 | — | 0.5470 | — | — |
| `ca35k300v3` | heatbath | 4.7126 | 30.0020 | 2.2074 | 48.7488 | 0.0020 | 454.903/494.000 |
| `ca35k300v3` | metropolis | 4.6313 | 30.0030 | 2.1389 | 49.2766 | 0.0030 | 454.903/494.000 |
| `ca35k300v3` | plateau | 4.6112 | 30.0005 | 2.1926 | 49.5245 | 0.0005 | 454.903/494.000 |
| `ca35k300v3` | acts-ipog | — | 3.2790 | — | 3.5654 | — | — |
| `ca30k300v3` | heatbath | 4.4783 | 45.0042 | 2.0275 | 62.7441 | 0.0042 | 446.846/474.000 |
| `ca30k300v3` | metropolis | 4.3691 | 45.0028 | 2.0822 | 63.5378 | 0.0028 | 446.846/474.000 |
| `ca30k300v3` | plateau | 4.8047 | 45.0013 | 2.0169 | 64.1640 | 0.0013 | 446.846/474.000 |
| `ca30k300v3` | acts-ipog | — | 3.3305 | — | 3.6181 | — | — |
| `ca25k100v3` | heatbath | 2.0671 | 20.0010 | 0.4605 | 25.4438 | 0.0010 | 58.059/70.000 |
| `ca25k100v3` | metropolis | 2.0217 | 20.0548 | 0.4666 | 25.9588 | 0.0548 | 58.059/70.000 |
| `ca25k100v3` | plateau | 1.9983 | 20.0002 | 0.4612 | 26.1268 | 0.0002 | 58.059/70.000 |
| `ca25k100v3` | acts-ipog | — | 0.7689 | — | 1.0270 | — | — |

ACTS search includes JVM startup and construction and chooses its own row count. Native search fixes the row count, so raw time ratios are not equal-task, equal-hardware speedups.

Native quality excludes the observed deadline-crossing chunk. Returned-best arrays may therefore improve on the eligible result and must remain separately labeled. `complete_runner_seconds` runs from runner entry through final array serialization; it excludes final summary serialization and outer restoration/process startup. Search includes observation and periodic checkpoint work. Implementation refinements measured on different dimensions or seeds do not establish a strict end-to-end speedup. Fixed-work implementation comparisons are distinct from pure kernel benchmarks.

The release verifier recomputes all original interactions from each retained initial/current/returned-best/eligible-best array, checks cold initialization, one-hot state, participation, strict quality fences and replay equality. It requires the complete frozen ID roster and exact target dimensions. Counted state checks include repeated states and checkpoints and are not trial counts. Public trajectory curation retains all replica-best improvements, endpoints and eligibility boundaries; the accepted full original traces establish that the export did not omit improvements. A CPU witness check does not independently attest the recorded wall clock.

## Comparison boundaries and limitations

ACTS Basic/IPOG is the measured constructive reference [R1, R3]. The corrected readout preserves every row and concretizes documented wildcard cells before certifying coverage. NIST IPOG-F tables and the Forbes et al. paper are historical construction context [R2, R4], not current best-known records or performance measurements on this system. The table's array downloads were unavailable with HTTP 403 during this study; no downloaded reference array supplied search information.

CAgen and its FIPOG family represent an additional engineered constructive approach [R5, R6]. The official CLI guide required a password requested from the authors; none was requested and no CAgen performance measurement was made. Its published historical times are not used as local comparisons. Citation does not imply implementation replication.

This study is limited to five unconstrained homogeneous strength-two dimensions and three fresh seeds per stochastic setting. It does not establish minimal row counts beyond a successfully closed elementary lower bound, current best-known status, superiority over specialized covering-array solvers, mixed-alphabet performance, equilibrium sampling quality or physical p-bit hardware benefit. The largest evaluated and successful deadline-eligible binary model both have 31,500 entries per replica. Complete confirmation evidence passed 329,733 original state checks.

## Reproduction

Follow the [source guide](source-guide.md) to restore the pinned ACTS jar, verify all core certificates, and execute a released configuration in a new isolated directory. The distribution includes accepted source, parameter JSON files, exact call configurations, complete rosters, core arrays and environment records. The assembled public tree passed 662 suite tests, plus two ACTS integration tests after correcting the fixture download path, and three fixed-work replays with 21 non-time arrays matched per replay; both core ledgers passed independent verification. Public Markdown has an English/Chinese pair; no PDF or third-party jar is bundled.

## References

- **R1.** NIST, [Automated Combinatorial Testing for Software: Downloadable Tools](https://csrc.nist.gov/projects/automated-combinatorial-testing-for-software/downloadable-tools). ACTS Basic download, application context and NIST usage statement; recorded access 2026-10-23. The byte identity is fixed by `configs/covering-array/external-baselines.json`.
- **R2.** NIST, [Covering Array Tables: CA(2,k,3), IPOG-F](https://math.nist.gov/coveringarrays/ipof/tables/table.2.3.html). Historical feasible row counts; page dated 2008-02-01. Referenced as upper bounds, not current best-known values.
- **R3.** NIST, [ACTS User Guide: Basic 1.0 and Advanced 3.3](https://csrc.nist.gov/csrc/media/Projects/automated-combinatorial-testing-for-software/documents/acts_user_guide_for_basic1.0_and_advanced3.3_versions_nov23.pdf). Command-line and output semantics. This study runs the pinned Basic jar with IPOG, strength 2, scratch mode, checking enabled and randomized wildcard completion disabled.
- **R4.** M. Forbes, J. Lawrence, Y. Lei, R. Kacker and D. R. Kuhn, [Refining the In-Parameter-Order Strategy for Constructing Covering Arrays](https://doi.org/10.6028/jres.113.022), *Journal of Research of NIST* 113(5), 2008. IPOG refinements and historical construction background.
- **R5.** SBA Research, [CAgen User Guide](https://srd.sba-research.org/tools/cagen/#/help) and [Downloads](https://srd.sba-research.org/tools/cagen/#/downloads). CLI 1.2 access boundary recorded 2026-10-23; not benchmarked.
- **R6.** K. Kleine and D. E. Simos, [An Efficient Design and Implementation of the In-Parameter-Order Algorithm](https://link.springer.com/article/10.1007/s11786-017-0326-0), *Mathematics in Computer Science* 12, 51–67, 2018; first published online 2017-12-08. FIPOG implementation background, without a new measurement here.
- **R7.** [Python](https://www.python.org/), [NumPy](https://numpy.org/), [PyTorch](https://pytorch.org/) and [Triton](https://github.com/triton-lang/triton). Execution, independent integer certificates, GPU state/RNG and compiled update kernels; use the released lock and measured environment for exact versions.
- **R8.** [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) and [OpenJDK](https://openjdk.org/). GPU runtime and Java reference runtime.
- **R9.** Astral [uv](https://docs.astral.sh/uv/) and [Ruff](https://docs.astral.sh/ruff/), and [pytest](https://docs.pytest.org/). Locked environment and software validation; software tests do not add experimental trials.
