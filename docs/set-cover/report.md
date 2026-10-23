# Feasibility-preserving p-bit search for weighted set covering

[简体中文](report.zh-CN.md) · [Overview](README.md) · [Source guide](source-guide.md)

## Accepted result and scope

The frozen SC8 RTX 5090 confirmation panel contains **117 natural calls and four work replays**. Heat bath improved the shared greedy prefix in **27/27** calls and met the frozen quality target in **15/27**. Its original integer costs beat matched Metropolis in 15 calls and neutral-move descent in 19, with no losses to either on this panel. AC-CFT remained stronger: heat bath tied its cost in 16 calls, lost in 11, and never won. These results support a finite-budget improvement inside this particular block search workflow. They do not establish a general speed or quality advantage over specialized classical solvers.

The problem is positive-integer weighted, static set covering:

$$\min_{x\in\{0,1\}^n}c^Tx\quad\text{subject to}\quad Ax\ge\mathbf1.$$

One binary decision represents each original column. All nine input files come from [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html); the benchmark manifest fixes their original numerical bytes and distinguishes row-oriented SCP from column-oriented railway data. The full `rail516` model contains 516 required rows and 47,311 columns, without padding or column reduction. Multiple replicas are reported separately from single-model size. GPU simulation of joint updates does not imply a corresponding fabricated device count or a pairwise hardware embedding.

## Hybrid method and attribution

A deterministic cost-per-newly-covered-row greedy construction, followed by redundant-column deletion, supplies a feasible classical prefix. Each of 128 replicas starts from that same cover. Input-only `cost-minhash` grouping creates 16 banks of width-eight blocks: columns sharing a low-rank covering row are grouped, with cost and fixed random tie-breaking refining their order. Every column occurs once per bank; the first bank provides a random full partition.

For a selected block $B$, the block exterior stays fixed. The kernel enumerates at most $2^8$ local binary patterns and uses original row coverage to exclude infeasible patterns. A subset zeta transform evaluates uncovered rows from row masks. With $L$ total blocks, $D$ banks and $\epsilon=0.0625$, active-column block selection has probability

$$g(B\mid x)=\frac{\epsilon+|x_B|}{\epsilon L+D|x|}.$$

The heat-bath conditional includes the state-dependent selection factor:

$$P(z\mid x_{\bar B},B)\propto\mathbf1[Ax\ge\mathbf1]\exp(-\beta c_B^Tz)g(B\mid x_{\bar B},z).$$

This is a conditional update for the extended distribution $\pi(x)g(B\mid x)$ at fixed temperature. The implementation's finite annealing runs are optimization experiments; no equilibrium sampling or mixing-time claim follows. Temperatures use inverse original cost units. The selected schedules are 0.2→8 for SCP41/SCP42 over 64-sweep cycles, constant 4 for the six 3K/5K/10K SCP instances, and constant 16 for RAIL516.

The matched Metropolis arm uses the same representation, prefix, block library, temperature, replica count and active selection, with the selection-ratio correction in its acceptance probability. The plateau arm samples uniformly among minimum-cost feasible block patterns, permitting neutral moves but no cost increase. Its tie rule differs from the heat-bath weighting, so the comparison is an attribution control rather than a strictly one-parameter causal intervention. Both are classical controls: the public ledger assigns them zero physical update p-bits, even though they use binary arrays and execute on the GPU.

AC-CFT is an independent specialized CPU reference, based on Lagrangian optimization, column selection/fixing and refinement. We build Francesco Cavaliere's [fixed implementation](https://github.com/c4v4/cft/tree/6cb63d0036223b8be9c4d01f95b296d641209f95), not code from the original paper's authors. Its upstream search is preserved; an explicitly documented timer observer enables work replay. The [1999 CFT paper](https://doi.org/10.1287/opre.47.5.730) supplies algorithmic background, not transferred historical speed measurements. LP/MILP calls provide longer-budget bounds and integer witnesses, not an equal-time p-bit competitor.

Native higher-order covering formulations already have precedent. [Lucas (2014), §5.1](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2014.00005/full) gives a QUBO formulation; [Searle et al. (2026), §3.1 and Appendix C.1](https://arxiv.org/html/2605.06037v1) discuss higher-order hitting-set updates. Those are background/excluded alternatives here: neither a squared-slack QUBO performance arm nor projected specialist hardware timings are part of this comparison. The evidence concerns the measured feasible block workflow and its controls, not novelty of set-cover encoding or the heat-bath formula alone.

## Frozen evaluation and timing

The complete denominator is $9\times3\times4+9=117$: nine instances × three seeds × heat bath/Metropolis/plateau/AC-CFT, plus one LP/MILP call per instance. All 117 processes completed; none were silently dropped or left unexecuted. Seeds were **2026102391, 2026102392, 2026102393**. The three native methods use 128 replicas per call; AC-CFT uses one CPU worker. Replicas sharing the same input and prefix are not 128 independent benchmark trials.

SCP41, SCPNRG1 and RAIL516 participated in the current cloud tuning round. The other six confirmation instances were excluded from that round's tuning; earlier research had already encountered these families and instances. Thus the holdout claim is relative to the current tuning round, not a claim of historically unseen data. Parameters were frozen before SC8, and the full three-seed panel is reported without choosing a best seed as the denominator.

Native search receives a 30 s window, a safety cap of 200,000,000 block updates per replica, and one observation per 256 updates. Only observations completed within the search deadline update the reported quality witness. The final overrun chunk remains in the evidence but cannot improve the deadline-qualified score. First-hit times are observation times, not exact transition times. AC-CFT receives a nominal 30 s soft deadline and may return early or overrun; its returned witness is paired with its actual complete elapsed time. LP/MILP receives 60 s, with an LP portion capped at 15 s. This is not a matched total-time or matched hardware-resource tournament.

The measured GPU was **NVIDIA GeForce RTX 5090**, with 33,766,572,032 bytes reported device memory. CPU controls used **AMD EPYC 9354 32-Core Processor**, one requested thread and affinity to one logical CPU. Software was Python 3.14.6, PyTorch 2.13.0+cu130, CUDA build 13.0, Triton 3.7.1, NumPy 2.4.4 and SciPy 1.18.0. The [environment record](../../data/set-cover/rtx5090/environment.json) and locked source distribution define reproduction versions.

`complete_runner_seconds` includes the runner's preparation, greedy initialization, block construction, GPU warmup, sampling, state transfer and final certification. Substage times are retained separately in [calls.json](../../data/set-cover/rtx5090/calls.json). Source restoration and outer process launch are outside this runner timer. Allocated/reserved memory describes the PyTorch CUDA allocator and includes all 128 replicas and its workspaces, not the entire device's driver/system consumption.

## All original integer costs

Each method cell lists the three seed results in the fixed order above. Lower is better. `Target` is the predeclared quality threshold; `LB` is the strongest independently checked rational integer lower bound in this panel. Neither column should automatically be read as a literature global optimum.

| Instance | Columns | Greedy prefix | Target | LB | Heat bath | Metropolis | Plateau | AC-CFT |
|---|---:|---:|---:|---:|---|---|---|---|
| SCP41 | 1,000 | 434 | 429 | 429 | 429/429/429 | 432/430/431 | 433/433/433 | 429/429/429 |
| SCP42 | 1,000 | 529 | 512 | 512 | 512/512/512 | 512/512/512 | 525/525/525 | 512/512/512 |
| SCPA1 | 3,000 | 261 | 253 | 247 | 258/258/258 | 258/258/258 | 260/260/260 | 254/254/253 |
| SCPA4 | 3,000 | 242 | 234 | 232 | 235/235/235 | 236/236/236 | 238/238/238 | 234/234/234 |
| SCPNRE1 | 5,000 | 30 | 29 | 22 | 29/29/29 | 29/29/29 | 29/29/29 | 29/29/29 |
| SCPNRE4 | 5,000 | 32 | 28 | 22 | 28/28/28 | 28/28/28 | 28/28/28 | 28/28/28 |
| SCPNRG1 | 10,000 | 189 | 176 | 160 | 176/176/176 | 181/179/179 | 188/188/188 | 176/176/176 |
| SCPNRG4 | 10,000 | 181 | 170 | 149 | 172/171/172 | 177/176/176 | 180/180/180 | 171/171/171 |
| RAIL516 | 47,311 | 203 | 182 | 182 | 184/185/185 | 187/188/188 | 185/185/185 | 182/182/182 |

SCP41=429, SCP42=512 and RAIL516=182 have feasible original witnesses matching exact rational lower bounds. Heat bath reached the first two optima in every seed, but missed RAIL516's optimum in all three. SCPA1/SCPA4 have solver-reported MILP gap zero at 253/234; their separately verified rational LP bounds remain 247/232. The latter certificates alone do not close those optimality gaps.

SCPNRG4's frozen 170 target comes from an earlier independently verified AC-CFT cover. Its [mathematical witness](../../data/set-cover/reference-witnesses/scpnrg4-target170.npz) is retained without importing old timing into this panel. This panel's AC-CFT returns 171/171/171, and no method reaches 170. The frozen target is neither retrospectively replaced by 171 nor described as a proven literature optimum.

| Comparison over 27 paired natural calls | Heat bath wins | Ties | Losses |
|---|---:|---:|---:|
| Matched Metropolis | 15 | 12 | 0 |
| Plateau descent | 19 | 8 | 0 |
| AC-CFT | 0 | 16 | 11 |

All 27 heat-bath calls improve the greedy prefix; none of its 15 target-hit calls inherits a target already met by initialization. These are call-level observations across nine selected instances, not a calibrated success probability or a statistical significance claim. Strong controls, target misses and tied cases remain in the same ledger.

## Complete time and scale

![Frozen target gaps and complete runtime, three-seed means and ranges](figures/rtx5090.svg)

The figure reports three-seed means and observed ranges, not confidence intervals. Quality gaps use the frozen target, including SCPNRG4's 170, rather than replacing it with this panel's best result. GPU arms use 128 replicas; AC-CFT uses one CPU thread.

| Instance | Mean heat-bath complete time (s) | Mean AC-CFT complete time (s) |
|---|---:|---:|
| SCP41 | 32.7624 | 0.3694 |
| SCP42 | 32.9265 | 0.3086 |
| SCPA1 | 32.7714 | 0.8729 |
| SCPA4 | 32.6987 | 0.6721 |
| SCPNRE1 | 35.4792 | 1.8862 |
| SCPNRE4 | 35.5609 | 1.9870 |
| SCPNRG1 | 35.4675 | 4.7314 |
| SCPNRG4 | 35.4691 | 4.1088 |
| RAIL516 | 38.3254 | 2.2234 |

The largest model uses 47,311 simulated update p-bits per heat-bath replica, zero auxiliary p-bits, and 128 replicas. RAIL516 peaks at **228.638 MiB allocated / 236.000 MiB reserved**. Its best heat-bath cost 184 is two units above the certified optimum 182. This demonstrates execution at that natural model scale, not successful optimum attainment at 47,311 variables. AC-CFT reaches 182 in every seed with substantially lower complete elapsed time.

## Independent verification and reproduction limits

The accepted original audit checked **1,083,941 saved integer states** over all natural calls and replays. The public core contains the original initial/final/quality-best/energy-best witnesses, all returned classical covers, rational dual numerators, and a documented subset of observations. It retains every replica-best improvement, both curve endpoints and eligibility boundaries; intermediate current-state fluctuations and periodic full-state archives are outside the core distribution. Cloud verification of all 121 released records checked **43,045 retained state witnesses**. The historical SCPNRG4 cost-170 witness was also checked, outside that count. The 43,045 public-core checks and 1,083,941 original-archive checks cover different retained scopes and must not be treated as the same count.

The [independent verifier](../../scripts/set_cover/verify_release.py) imports only the standard library and NumPy. It binds inputs and witnesses by SHA-256, parses original incidence text, recomputes integer costs and every row's coverage, verifies the retained curve indices and deadline/fixed-work eligibility, checks exact rational dual inequalities, and rejects incomplete call rosters. It separates initial target hits from new hits after the stochastic stage. For dual row weights $a_i/d$, it checks $a_i\ge0$ and $\sum_{i:A_{ij}=1}a_i\le dc_j$ using integer arithmetic; the lower bound is $\lceil\sum_i a_i/d\rceil$.

Three native work replays and one AC-CFT replay check the original work without adding trials. Native replay freezes the original quality step boundary; its wall-clock observations may differ. AC-CFT's replayed timer is a work-reproduction instrument, not a new performance measurement. A rerun on another device or software stack may produce different finite-time outcomes; the immutable released certificates remain directly checkable. Commands for input restoration, independent checking, isolated fresh runs and rebuilding AC-CFT are in the [source guide](source-guide.md).

## References

1. J. E. Beasley, [OR-Library set-covering data and format specification](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html). Original inputs, dimensions and format distinctions; exact selected files are pinned by the benchmark manifest.
2. A. Caprara, M. Fischetti and P. Toth, [A Heuristic Method for the Set Covering Problem](https://doi.org/10.1287/opre.47.5.730), *Operations Research* 47(5), 730–743 (1999). Publisher abstract and bibliographic record reviewed; not a full-paper replication or historical speed comparison.
3. F. Cavaliere, [AC-CFT implementation, revision 6cb63d0036223b8be9c4d01f95b296d641209f95](https://github.com/c4v4/cft/tree/6cb63d0036223b8be9c4d01f95b296d641209f95). Pinned source, README and MIT license; the implementation is distinct from the original paper's authorship.
4. A. Lucas, [Ising formulations of many NP problems](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2014.00005/full), *Frontiers in Physics* 2:5 (2014), §5.1. QUBO/auxiliary-variable background; no QUBO performance replication here.
5. A. J. Searle et al., [A virtually connected probabilistic computer as a solver for higher-order, densely connected, or reconfigurable combinatorial optimisation problems](https://arxiv.org/html/2605.06037v1), arXiv:2605.06037v1 (7 May 2026), §3.1 and Appendix C.1. Higher-order hitting-set precedent; projected hardware performance is excluded from this GPU comparison.
6. [SciPy](https://scipy.org/) 1.18.0 and its bundled [HiGHS](https://highs.dev/) backend. LP/MILP references; numerical solver reports remain separate from exact rational certificates.
7. [PyTorch](https://pytorch.org/) 2.13.0+cu130 and [Triton](https://triton-lang.org/main/index.html) 3.7.1. GPU state storage, randomness and custom kernels.
8. [NumPy](https://numpy.org/) 2.4.4. Input handling and independent array/certificate checks.
9. [{fmt}](https://github.com/fmtlib/fmt/tree/10.2.1) 10.2.1. Pinned AC-CFT build dependency, fetched separately with its own license.
10. [uv](https://docs.astral.sh/uv/). Locked project environment and reproduction command runner; the repository lock defines package versions.
