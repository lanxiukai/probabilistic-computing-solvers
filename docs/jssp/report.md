# Job-shop scheduling with conditional p-bit updates

[简体中文](report.zh-CN.md)

This report describes GPU simulation of a feasible priority-sequence search. Its p-bit stage samples conditional permutations using the original schedule makespan. A deterministic classical decoder supplies schedules and energies, and OR-Tools CP-SAT supplies an independent reference. No physical p-bit device or advantage over strong classical scheduling methods is claimed.

Independent confirmation comprises six inputs, three new seeds and four methods: 72 original calls plus three fixed-work replays. The heat bath reaches the reference optimum in all three calls on four inputs, with a clear hit-distribution improvement over matched controls on held-out ft10. Both Taillard inputs remain unsolved to their references, and CP-SAT has lower mean complete time on every input. Complete cloud and local original-evidence audits passed.

## Problem and representation

The implementation covers static classical JSSP with positive integer durations, one visit to each machine per job, no preemption and no release dates. It minimizes the makespan while respecting operation precedences and machine capacity.

For $J$ jobs and $M$ machines, a state is a length-$L=JM$ sequence containing each job label $M$ times. The next occurrence of a job selects its next operation. The append decoder starts that operation at the later of the job-ready and machine-ready times, so every valid sequence produces a feasible schedule.

The GPU stores the one-hot assignment $x_{p,j}=1[s_p=j]$, a sequence cache, objective caches and participation counters. There are $J^2M$ explicitly materialized binary entries per replica. This is a redundant representation, not a claim of that many independent degrees of freedom or physical devices. Replica count, the full encoded state and the number of assignment bits actually involved in updates are reported separately. This native decoder energy is not a time-indexed QUBO.

Each row sums to one and each job column sums to $M$. A single-bit flip would leave this constrained state space; permutation blocks preserve both constraints. The participation counter records assignment entries considered in these blocks, not necessarily entries that flipped.

Participation is counted through the returned execution, including any retained deadline-crossing chunk. The quality result uses only the eligible prefix, so these two scopes are reported separately.

## Update and attribution

A step selects two or four distinct sequence positions independently of the current state, enumerates their $b!$ label permutations and chooses a candidate with probability proportional to $\exp[-\beta(E-E_{\min})]$. Duplicate labels give equal multiplicity to each distinct arrangement and therefore do not bias their relative probabilities. The implementation checks integer makespans against an independent CPU decoder and original schedule constraints.

At fixed temperature this is a conditional heat bath on the feasible sequence space. Cyclic temperature schedules are optimization heuristics; this study makes no mixing-time or equilibrium-sampling claim. The deterministic decoder and objective evaluation are classical support. Cold initialization uses uniformly permuted job multisets, and CP-SAT solutions are never injected into it.

Different priority sequences can decode to the same schedule. A fixed-temperature Boltzmann interpretation therefore concerns priority sequences and induces schedule multiplicities; it does not imply a plain Boltzmann distribution over distinct schedules. The measured task here is optimization.

The inverse temperature $\beta$ is measured against the original integer processing-time units, rather than wall-clock seconds. Probabilities are evaluated in FP64. Reference makespans are used for reporting and development selection; they are not passed into the energy kernel, initialization or CP-SAT hints.

Matched-neighborhood Metropolis and random-tie minimum-energy descent (`plateau`) retain the same initialization and position-proposal rules. CP-SAT uses interval variables, job precedences, machine `NoOverlap` constraints and a makespan objective, with one worker. Its primal schedule and proof bound are separate results. Improvement over a cold random prefix does not establish an advantage over these controls.

Metropolis and plateau use the same CUDA implementation as the heat bath. CP-SAT is the independent CPU reference.

The matched Metropolis control evaluates the same complete candidate set as the heat bath, although a dedicated Metropolis solver could evaluate only its proposed candidate. It isolates update-rule effects and is not an optimized standalone Metropolis benchmark.

## Benchmarks and measurement

The six standard inputs come from [ScheduleOpt's benchmark collection at a fixed revision](https://github.com/ScheduleOpt/benchmarks/blob/8a6e36ef31263d36975e56fb481e0ba04e188d6f/jobshop/README.md). Download tooling checks the original byte hashes; it does not imply permission to redistribute upstream data. The reference table lists equal lower and upper bounds for the inputs used here.

The `ft`, `la` and `ta` families originate with Fisher–Thompson, Lawrence and Taillard, respectively. [OR-Library's jobshop index](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html) documents classic inputs and the Taillard generator; the actual bytes in this study are pinned to the ScheduleOpt paths above.

| Instance | Jobs × machines | Operations | Binary entries per model | Reference makespan |
|---|---:|---:|---:|---:|
| ft06 | 6 × 6 | 36 | 216 | 55 |
| ft10 | 10 × 10 | 100 | 1,000 | 930 |
| la01 | 10 × 5 | 50 | 500 | 666 |
| la16 | 10 × 10 | 100 | 1,000 | 945 |
| ta01js | 15 × 15 | 225 | 3,375 | 1,231 |
| ta51js | 50 × 15 | 750 | 37,500 | 2,760 |

The fixed catalog classifies all six as `toy`: at least one reference constraint-programming engine proves optimality within one minute. They establish correctness, measurable optimization and an encoding-scale endpoint; they are not modern frontier JSSP challenges.

Hardware is one NVIDIA GeForce RTX 5090 with approximately 32 GB VRAM; CPU references run on one AMD EPYC 9354 worker. The locked environment uses Python 3.14.6, PyTorch 2.13.0+cu130, Triton 3.7.1 and OR-Tools 9.15.6755. Preparation, search, certification and complete runner time are separate. PyTorch peak allocated/reserved measurements do not include all CUDA driver or JIT memory.

Only accepted states observed before the search deadline count toward quality. A deadline-crossing chunk is retained but excluded. Every-chunk objective curves, RNG states, current/best checkpoints and original schedule certificates are preserved. A fixed-work replay adds no independent trial.

Complete runner time includes preparation, search, certificates and final result output. Outer source restoration and process startup are recorded separately and are not included in that runner timer.

## Frozen confirmation protocol

The policy was frozen before observing the three confirmation seeds, `2026102344`, `2026102345` and `2026102346`. `ft06`, `la16` and `ta51js` were development inputs; `la01`, `ft10` and `ta01js` were held out from tuning and inherited the corresponding policy group. All confirmation calls use append decoding and the `reduce` implementation. Configuration selection is not repeated after these outcomes.

| Group | Inputs and search limits | Replicas per native call | Block width | Inverse-temperature cycle |
|---|---|---:|---:|---|
| Small | ft06: 2 s; la01: 5 s | 64 | 2 | 1 to 10 over 1,024 updates |
| Medium | la16: 20 s; ft10: 20 s; ta01js: 30 s | 256 | 4 | 0.08 to 4 over 8,192 updates |
| Large | ta51js: 45 s | 256 | 2 | Fixed at 1 |

The cycles increase geometrically and restart at their declared update count. Within each input and seed, the heat bath and both CUDA controls share the cold initialization, decoder, block width, replica count, temperature parameters, chunk size, capture cadence and nominal limit. CP-SAT uses one CPU worker and the same nominal search limit. Method order rotates across seeds. These are matched update-rule and resource-labelled comparisons, not equal hardware or optimized-implementation comparisons.

Each completed eight-update chunk is observed. Coarse current/best sequence snapshots occur every 16,384 updates, and a full recovery checkpoint is saved at the next completed chunk after two seconds from search start or the previous checkpoint's completion. This evidence cost remains in search time. The first-hit time has chunk resolution; a deadline-crossing chunk is kept but contributes no solution-quality credit. CP-SAT quality is selected from callbacks within its limit; final returned solutions, final bounds and proof status are separate.

There are three independent calls per input and method. Replica distributions describe a call's ensemble, not extra independent trials. Every planned call and miss remains in the denominator. The three replays reproduce the first-seed heat bath for `ft06`, `la16` and `ta51js` with frozen completed-work and quality-prefix limits; they add no confirmation trials.

## Independent confirmation results

All entries below use only deadline-eligible quality. Triplets follow seed order 1 / 2 / 3; each native value is the best of that call's declared ensemble. The six-by-four-by-three panel completed all 72 calls. Three replays are retained separately.

| Input | Reference | Heat bath | Metropolis | Plateau | CP-SAT, one CPU worker |
|---|---:|---|---|---|---|
| ft06 | 55 | 55 / 55 / 55 | 55 / 55 / 55 | 55 / 55 / 55 | 55 / 55 / 55 |
| la01, held out | 666 | 666 / 666 / 666 | 666 / 666 / 666 | 666 / 666 / 666 | 666 / 666 / 666 |
| la16 | 945 | 945 / 945 / 945 | 946 / 945 / 945 | 945 / 945 / 945 | 945 / 945 / 945 |
| ft10, held out | 930 | 930 / 930 / 930 | 935 / 930 / 935 | 937 / 930 / 937 | 930 / 930 / 930 |
| ta01js, held out | 1231 | 1249 / 1253 / 1256 | 1286 / 1281 / 1277 | 1241 / 1237 / 1248 | 1231 / 1240 / 1231 |
| ta51js | 2760 | 2960 / 2950 / 2952 | 2950 / 2979 / 2944 | 2951 / 2953 / 2953 | 3041 / 3135 / 3073 |

The heat bath reaches the external reference in 12 of 18 calls: all three seeds on each of four inputs. There are no initial reference hits in those 18 calls, so these hits arise after initialization. Neither Taillard input reaches its reference. On the 37,500-entry encoding, the best heat-bath value, 2950, remains 6.88% above 2760.

Reference-hit counts within each native call show more than its ensemble minimum. Each triplet has three separate denominators; the rows must not be pooled into one independent-trial estimate.

| Input | Replicas per call | Heat-bath hits | Metropolis hits | Plateau hits |
|---|---:|---|---|---|
| ft06 | 64 | 64 / 64 / 64 | 64 / 64 / 64 | 58 / 55 / 58 |
| la01 | 64 | 60 / 61 / 61 | 60 / 60 / 60 | 52 / 51 / 56 |
| la16 | 256 | 20 / 30 / 30 | 0 / 2 / 1 | 20 / 22 / 16 |
| ft10 | 256 | 48 / 41 / 38 | 0 / 2 / 0 | 0 / 1 / 0 |
| ta01js | 256 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| ta51js | 256 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

The strongest observed update-rule benefit is on held-out ft10: the heat bath reaches 930 in 48, 41 and 38 of 256 replicas, while the matched controls reach it in at most two replicas per call. Its within-call median is 937 for all three seeds, versus Metropolis 960 / 960 / 958 and plateau 987 / 986 / 983. On la16, the heat-bath medians are 946 / 946 / 946, versus 956 / 956 / 956 and 982 / 982 / 982. These are repeatable differences within this frozen panel, with only three independent seeds.

The negative results are equally material. On held-out ta01js, the heat bath loses to plateau and CP-SAT in best quality for every seed; its median is 1286 throughout, versus plateau 1279.5 / 1286 / 1285. On ta51js, heat-bath medians 3054 / 3052 / 3054.5 improve on Metropolis 3074 / 3077 / 3078 and slightly on plateau 3058 / 3060 / 3057, but ensemble-best comparisons are mixed: the heat bath loses to Metropolis on two seeds and to plateau on one. Its better best values than this one-worker, 45-second CP-SAT reference do not establish a general classical advantage.

CP-SAT proves optimality within the declared search budget in 14 of 18 calls. There are no late incumbent callbacks and no eligible-versus-returned cost differences in this panel. The four unproved calls retain separate final solver bounds: ta01js seed 2 returns cost 1240 and bound 1019, while all three ta51js calls return bound 2760. These returned bounds are not retroactively credited as proofs within the budget; the external catalogue reference for ta01js remains 1231.

## Complete time and participation

The following are arithmetic means over the three complete runner times, in seconds. They include preparation, the measured search, certification and final output. The exact three times and stage timings remain in the linked call data. CP-SAT has one worker, not a replica ensemble.

| Input | Heat bath, GPU | Metropolis, GPU | Plateau, GPU | CP-SAT, CPU |
|---|---:|---:|---:|---:|
| ft06 | 3.698 | 3.685 | 3.663 | 0.400 |
| la01 | 6.861 | 6.886 | 6.802 | 0.430 |
| la16 | 23.514 | 23.051 | 22.781 | 1.512 |
| ft10 | 23.760 | 23.124 | 22.833 | 17.961 |
| ta01js | 34.058 | 33.491 | 33.194 | 24.905 |
| ta51js | 50.687 | 50.539 | 50.943 | 45.443 |

The mean complete heat-bath time exceeds CP-SAT's on every input here. In particular, the stronger hit distribution on ft10 does not imply faster end-to-end solution than CP-SAT. Search budgets and complete times must not be interchanged.

The participation mask records entries considered in conditional blocks through the final returned chunk. To bound participation strictly before the quality cutoff using only public core data, subtract at most $b^2$ entries per excluded update from each replica's final count and clip at zero. This is a conservative lower bound, not the exact eligible count or a count of flips.

| Input | Encoded entries per replica | Returned minimum, seeds 1 / 2 / 3 | Eligible-prefix lower bound, seeds 1 / 2 / 3 |
|---|---:|---|---|
| ft06 | 216 | 216 / 216 / 216 | 184 / 184 / 184 |
| la01 | 500 | 500 / 500 / 500 | 468 / 468 / 468 |
| la16 | 1000 | 1000 / 1000 / 1000 | 872 / 872 / 872 |
| ft10 | 1000 | 1000 / 1000 / 1000 | 872 / 872 / 1000 |
| ta01js | 3375 | 3375 / 3375 / 3375 | 3247 / 3247 / 3247 |
| ta51js | 37500 | 37377 / 37381 / 37380 | 37345 / 37349 / 37380 |

Each minimum is over all replicas in that call. Thus the large experiment does exercise more than 32,768 assignment entries per replica within the eligible prefix, while still missing the scheduling optimum. Peak PyTorch memory across its three heat-bath calls is 105.88 MiB allocated and 170 MiB reserved for 256 replicas. These measurements are neither physical p-bit counts nor whole-device memory or energy measurements.

## Evidence and reproduction

The [complete call index](../../data/jssp/rtx5090/calls.json) preserves all 72 original calls and three replays, inline and executable configurations, input hashes, deadline-scoped metrics and core witness digests. The [source guide](source-guide.md) gives benchmark fetching, isolated reruns, witness verification and figure commands.

Full original-state auditing checks 1,106,542 stored/checkpoint schedule certificates, including repeated captures; this is an audit-coverage count, not an independent-sample count. The supplementary audit verifies the frozen source/settings/input bindings, every recorded quality boundary, original schedules, first-hit arrays, committed checkpoint RNG and participation bounds. Each of the three fixed-work replays matches all 23 non-timing arrays. The portable verifier recomputes claims from the released core without rerunning an optimizer.

The public curve keeps any replica's best improvements, eligibility boundaries and first/last observations. Original observation indices and curation metadata remain; omitted current-state fluctuations cannot be reconstructed from the smaller trace. Full original trajectories remain preserved separately.

The [ft06 illustration](figures/ft06/schedule.svg) and [its numeric data](figures/ft06/figure-data.json) were generated from the released core evidence and independently certified. The renderer selects one replica by lowest eligible cost then replica index and independently certifies its cold-prefix and eligible-best schedules. The illustration shows a same-replica change, not a comparison with a classical solver.

## Limitations

This is an exact native objective over a feasible representation, with classical decoding support. It does not show scheduling speedup over CP-SAT, practical industrial-scale competitiveness, quantum advantage or p-bit device energy savings. The append representation and random position neighborhoods do not exploit the critical-path and critical-block structure used by strong JSSP local search. Larger encoded state alone is insufficient evidence of useful solving scale.

Only three new seeds and six catalogued toy instances are evaluated. Best-of-ensemble quality, per-replica hit fractions and complete time answer different questions; none alone establishes a general advantage. The study does not reproduce critical-block tabu search or the learning methods in the cited benchmark framework. The time-indexed QUBO and quantum-annealing literature uses a different representation and hardware, so no cross-paper speed ratio is inferred. Full-state certificates verify stored schedules and claim consistency, not the accuracy of machine clocks or the internal timing of unobserved transitions.

## References

1. ScheduleOpt. [Jobshop benchmark, fixed revision 8a6e36ef31263d36975e56fb481e0ba04e188d6f](https://github.com/ScheduleOpt/benchmarks/blob/8a6e36ef31263d36975e56fb481e0ba04e188d6f/jobshop/README.md). Sources, bounds and benchmark classifications; accessed 2026-10-23.
2. Google. [The Job Shop Problem — OR-Tools](https://developers.google.com/optimization/scheduling/job_shop). The independent CP-SAT reference uses locked version 9.15.6755.
3. Nowicki E., Smutnicki C. [A Fast Taboo Search Algorithm for the Job Shop Problem](https://doi.org/10.1287/mnsc.42.6.797). *Management Science* 42(6):797–813, 1996. Critical-path/block background; not a reproduced TSAB baseline.
4. Venturelli D., Marchand D. J. J., Rojo G. [Quantum Annealing Implementation of Job-Shop Scheduling](https://arxiv.org/abs/1506.08479). arXiv:1506.08479, 2015/2016. Time-indexed QUBO and quantum-annealing background, not a device or cross-paper speed comparison.
5. Reijnen R., Smit I. G., Zhang H., Wu Y., Bukhsh Z., Zhang Y. [Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods](https://arxiv.org/abs/2308.12794). 2023 preprint, revised 2025; [journal version published 30 May 2026](https://link.springer.com/article/10.1007/s10472-026-10007-3). Background only; broader flexible/dynamic tasks and all listed learning methods were not evaluated.
6. Runtime and rendering software: [PyTorch](https://pytorch.org/), [Triton](https://triton-lang.org/main/index.html), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), and [uv](https://docs.astral.sh/uv/). Exact measured versions are recorded in the locked environment and run metadata; these links identify the software projects, not performance comparisons.
7. Taillard E. [Benchmarks for basic scheduling problems](https://doi.org/10.1016/0377-2217(93)90182-M). EJOR 64(2):278–285, 1993. Original Taillard scheduling benchmark family.
8. Beasley J. E. [OR-Library jobshop index](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html). Canonical benchmark and generator context; not the byte-download location used in this study.


The illustrated replica 0 improves its makespan from 106 to 55; this is one replica, not an ensemble summary.
