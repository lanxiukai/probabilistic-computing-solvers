# QAP: independent RTX 5090 confirmation

[简体中文](experiment-report.zh-CN.md) · [Model and bibliography](README.md)

The conditional p-bit workflow reaches the catalog optimum in all nine fresh-seed Nugent calls. None of the nine Taillard calls reaches its frozen historical reference. The best `tai150b` cost is **499,088,558**, **0.0384679%** above reference 498,896,643. Feasibility is certified on the original integer matrices; a gap to a feasible reference is not an optimality gap.

The workflow starts from uniform random permutations, without a classical search prefix, reference initialization or polishing. Original objective evaluation and constraint-preserving proposals are classical support. Matched CUDA Metropolis/plateau and two separate CPU tabu implementations remain comparison references. Results demonstrate scoped optimization capability, not physical p-bit hardware or general classical superiority.

## Frozen confirmation and benchmarks

Q2 freezes seeds `2026102411/2026102412/2026102413` before observing their outcomes. Six original QAPLIB inputs each receive heat bath, matched Metropolis, plateau and one CPU attribute-tabu call: **72 confirmation calls**. Four additional fixed-work implementation calls and five replays are separate, making 81 retained records. All declared calls complete; none is removed from a denominator.

| Input | Order | Binary entries per model | Reference cost | Reference status |
|---|---:|---:|---:|---|
| `nug12` | 12 | 144 | 578 | `proven_optimum` |
| `nug20` | 20 | 400 | 2,570 | `proven_optimum` |
| `nug30` | 30 | 900 | 6,124 | `proven_optimum` |
| `tai40a` | 40 | 1,600 | 3,139,370 | `historical_feasible_reference` |
| `tai80a` | 80 | 6,400 | 13,499,184 | `historical_feasible_reference` |
| `tai150b` | 150 | 22,500 | 498,896,643 | `historical_feasible_reference` |

Original inputs and reference permutations are fetched from the pinned [QAPLIB mirror](https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/), without redistribution. The catalog preserves reference indexing and inversion conventions. `nug20`, `tai40a` and `tai80a` are held-out inputs. `nug12`, `nug30` and `tai150b` were used during development. The natural scale endpoint is 22,500 constrained one-hot entries, not an artificially padded 32K instance or a count of independent physical devices.

All native calls use width four, four warps, chunks of 512 updates and exact factored energy evaluation. Nugent uses 128 replicas, a 4,096-step cycle and normalized inverse-temperature factors 1–64. Taillard uses 512 replicas, a 65,536-step cycle and factors 1–256. Each factor is divided by `sqrt(n) * mean(A) * mean(B)`, including matrix zeros and diagonals. Search limits are 5 s for `nug12`, 15 s for `nug20/nug30` and 30 s for Taillard. The CPU attribute-tabu comparison uses one worker with the same nominal limits.

Every chunk is observed. Full checkpoints target two search seconds; coarse state snapshots follow the frozen interval derived from development throughput, covering at least 131,072 updates and approximately two development search seconds. Quality includes only completed chunks/iterations observed within the original deadline; retained late work earns no quality credit. Replays preserve the original work and quality fences and add no trials. Controls reuse heat-bath-selected settings and did not receive equal independent tuning.

## Eligible quality and attribution

Triplets follow the declared seed order; each native value is the best of its full replica ensemble.

| Input | Heat bath | Metropolis | Plateau | Attribute tabu, one CPU worker |
|---|---|---|---|---|
| `nug12` | 578/578/578 | 578/578/578 | 578/578/578 | 578/578/578 |
| `nug20` | 2570/2570/2570 | 2570/2570/2570 | 2570/2570/2570 | 2570/2570/2570 |
| `nug30` | 6124/6124/6124 | 6128/6128/6128 | 6142/6128/6128 | 6124/6124/6124 |
| `tai40a` | 3152140/3152596/3156140 | 3168552/3169150/3175790 | 3183628/3172440/3187164 | 3166152/3168976/3164344 |
| `tai80a` | 13740924/13709802/13726626 | 13850062/13846838/13872442 | 13778290/13709460/13748952 | 13733488/13742896/13749400 |
| `tai150b` | 499135234/499088558/499287160 | 501107193/501808246/501391157 | 505197962/503913358/502787623 | 512223237/514310031/514765731 |

| Input | Replicas per native call | Heat-bath reference hits | Metropolis hits | Plateau hits |
|---|---:|---|---|---|
| `nug12` | 128 | 128/128/128 | 128/128/128 | 15/22/13 |
| `nug20` | 128 | 128/128/128 | 128/128/128 | 2/4/6 |
| `nug30` | 128 | 128/128/127 | 0/0/0 | 0/0/0 |
| `tai40a` | 512 | 0/0/0 | 0/0/0 | 0/0/0 |
| `tai80a` | 512 | 0/0/0 | 0/0/0 | 0/0/0 |
| `tai150b` | 512 | 0/0/0 | 0/0/0 | 0/0/0 |

No heat-bath initial ensemble contains a reference hit. All 5,760 heat-bath replicas improve their original cost. These within-call counts are not extra independent seeds. By paired best cost, heat bath has 12 wins/6 ties/0 losses against Metropolis, 11/6/1 against plateau and 8/9/1 against owned attribute tabu. The `tai80a` second seed remains a loss to plateau; its first seed remains a loss to owned tabu.

The clearest matched-control result is `nug30`: heat bath reaches 6124 in every call, with 128/128/127 successful replicas, while both CUDA controls miss the target in all three calls. At `tai150b`, all three heat-bath ensembles have lower eligible best cost than either matched control or the owned CPU implementation, while still missing the reference. These comparisons use different resources for CPU and GPU, and controls share heat-bath-selected parameters. They do not establish superiority over optimized GPU annealing or modern QAP solvers.

## Timing, resources and exact implementation comparison

Measurements use NVIDIA GeForce RTX 5090, AMD EPYC 9354 CPU references, Python 3.14.6, PyTorch 2.13.0+cu130, Triton 3.7.1, NumPy 2.4.4 and CUDA 13.0. The [environment record](../../data/qap/rtx5090/environment.json) binds versions. Complete runner time includes preparation, search, original certificates and final output; outer source restoration/process startup is separate. Search includes random generation, observation, transfers and checkpoints. CPU reference binaries are built once outside the per-call timers.

The next table gives arithmetic means of the three heat-bath calls. A/R is the maximum allocator allocated/reserved MiB, including replicas and warm-up. It excludes full driver/context occupancy.

| Input | Preparation s | Search s | Certificate s | Complete s | A/R MiB |
|---|---:|---:|---:|---:|---:|
| `nug12` | 1.7732 | 5.0011 | 0.0040 | 7.2036 | 6.065/22.000 |
| `nug20` | 1.7322 | 15.0010 | 0.0049 | 17.4676 | 6.147/22.000 |
| `nug30` | 1.7466 | 15.0006 | 0.0059 | 19.4527 | 6.297/22.000 |
| `tai40a` | 1.8438 | 30.0039 | 0.0229 | 33.4329 | 25.912/36.000 |
| `tai80a` | 2.0256 | 30.0015 | 0.0455 | 33.9664 | 36.423/48.000 |
| `tai150b` | 2.2359 | 30.0048 | 0.1238 | 35.7474 | 123.073/192.000 |

| Input | Heat-bath complete s | Metropolis complete s | Plateau complete s | Owned CPU complete s |
|---|---:|---:|---:|---:|
| `nug12` | 7.2036 | 7.1210 | 7.0432 | 5.2658 |
| `nug20` | 17.4676 | 17.4957 | 17.1197 | 15.2910 |
| `nug30` | 19.4527 | 18.4287 | 25.7233 | 15.2968 |
| `tai40a` | 33.4329 | 33.9870 | 41.8312 | 30.2811 |
| `tai80a` | 33.9664 | 34.0622 | 33.6887 | 30.2720 |
| `tai150b` | 35.7474 | 36.0103 | 34.7343 | 30.3285 |

The final implementation pairs use 128 replicas, width four and seed 2026102320 on both layouts, even though Taillard quality confirmation uses 512 replicas. Candidate and factored calls differ only in energy layout. They perform 131,072 updates on `nug30` or 8,192 on `tai150b`, within a 90 s safety cap. All 20 original non-time arrays, including RNG and persistent state, match exactly; all observed search kernel variants were prewarmed.

| Input | Search s, candidate → factored | Complete s, candidate → factored | Search ratio | Complete ratio |
|---|---:|---:|---:|---:|
| `nug30` | 1.062594 → 0.455267 | 3.596703 → 2.627749 | 2.334× | 1.369× |
| `tai150b` | 3.082385 → 0.045714 | 9.310607 → 2.487848 | 67.428× | 3.742× |

These are one paired timing observation per input, not confidence intervals or independent quality trials. The [implementation record](../../data/qap/rtx5090/implementation-comparisons.json) includes unrounded timings, launcher ratios, kernel resources, source/configuration/witness hashes and equality scopes. Host-observed search ratios are **not isolated CUDA-kernel speedups**. Local-memory bytes per thread are not spill-instruction counts.

## Retained development and strong classical reference

The separate [Q1c development ledger](../../data/qap/development-q1c/calls.json) retains nine tuning calls, two fixed-work calls and one replay. It selected the final two family policies before Q2. At 15 s, the two `nug30` chunk choices hit the target in 127/128 and 128/128 replicas. The `tai150b` two-position replica sweep is not monotonic: 512 replicas yield 499,989,110, while 1024 yield 500,290,327. The selected four-position/512-replica setting gives 499,266,749. These development outcomes are separate from confirmation and cannot be pooled as extra fresh-seed trials.

The [official Ro-TS ledger](../../data/qap/external-rots/calls.json) contains 18 independent CPU calls and three fixed-iteration replays. It uses the author's 2006 C source, published `8*n` and `5*n*n` parameters, one CPU thread and 30 s per input/seed, with seeds 2026102401/02/03. Reference-informed early termination is disabled. These are different seeds and resources, so this is not a paired comparison or equal-hardware speed ranking.

| Input | Ro-TS best | Median | Worst | Target-hit calls | Median complete s |
|---|---:|---:|---:|---:|---:|
| `nug12` | 578 | 578 | 578 | 3/3 | 30.2874 |
| `nug20` | 2,570 | 2,570 | 2,570 | 3/3 | 30.2863 |
| `nug30` | 6,124 | 6,124 | 6,124 | 3/3 | 30.3004 |
| `tai40a` | 3,149,548 | 3,149,680 | 3,150,124 | 0/3 | 30.3085 |
| `tai80a` | 13,626,798 | 13,634,674 | 13,653,898 | 0/3 | 30.3164 |
| `tai150b` | 501,493,731 | 501,741,854 | 504,548,477 | 0/3 | 30.4199 |

Official Ro-TS is better on `tai40a/tai80a` in this panel, while the GPU heat-bath ensembles are better on `tai150b`. All Taillard targets remain misses for both. The CPU receives longer Nugent budgets than Q2. Upstream Ro-TS permits non-commercial use with acknowledgement; its original/instrumented source and binaries are fetched separately and excluded from this repository's MIT distribution.

## Source identity, verification and limits

All original archives were recovered and independently audited: Q2 2,020,690 saved states, Q1c 1,186,944 and Q3 5,085. Public core data retain the full call rosters, initial/current/returned/eligible witnesses, one-hot/participation/RNG states and curated improvement/boundary observations. They do not claim to include every private trajectory snapshot. See the [Q2 panel](../../data/qap/rtx5090/panel.json), [calls](../../data/qap/rtx5090/calls.json) and [reproduction instructions](README.md#benchmarks-and-reproduction).

The exported seven-module package matches the executed Q3 shared-source version after namespace replacement and formatting. Six modules and the native runner branch also match Q2/Q1c. The runner additionally dispatches the official external reference and binds its extra source inputs; this difference is explicitly recorded in [provenance](../../provenance/qap/release.json). Timings belong to the original executed panels. Nine public-source replays reproduce the original non-time arrays: 20 for native calls, 13 for owned tabu and 17 for official Ro-TS. The assembled public tree passes 788 software tests with no skips, followed by core-witness and content checks.

This study does not close the Taillard references, establish current best-known records, prove equilibrium sampling, compare hardware energy, or replicate the cited p-dit, learned-MCMC and quantum-annealing systems. Three seeds per setting and one implementation timing pair per input limit generalization. Strong controls, misses and resource differences remain part of the published result.

## References

The following sources were checked on 2026-10-23. Their performance claims have not been reproduced by this module and are not directly compared across different hardware, instance panels or timing definitions.

- [QAPLIB at Lehigh](https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/): input and reference witnesses. The main Montreal endpoint was unavailable during this review; this release pins the retrieved mirror bytes.
- É. Taillard, [Robust taboo search for the quadratic assignment problem](https://mistic.iict-heig-vd.ch/taillard/articles.dir/Taillard1991.pdf), *Parallel Computing* 17, 443–455 (1991), [DOI](https://doi.org/10.1016/S0167-8191(05)80147-4): classical method context and the independent comparison's design inspiration.
- É. Taillard, [official standard C Ro-TS source](https://mistic.iict-heig-vd.ch/taillard/codes.dir/tabou_qap2.c), dated 2006-03-14: the separately fetched external implementation, including nonuniform tenure. Its non-commercial conditions remain separate from the adapters' license.
- G. Paul, [A GPU implementation of the Simulated Annealing Heuristic for the Quadratic Assignment Problem](https://arxiv.org/abs/1208.2675v1), 2012 preprint: earlier GPU QAP annealing; this module makes no first-GPU-QAP claim.
- C. Novoa and A. Qasem, [GPU-accelerated Parallel Solutions to the Quadratic Assignment Problem](https://arxiv.org/abs/2307.11248v1), 2023 preprint: GPU 2-opt and tabu implementations; their reported performance has not been reproduced on the present device.
- C. Duffee et al., [P-dit Probabilistic Ising Machine for Solving the Quadratic Assignment Problem](https://arxiv.org/abs/2605.24408v1), 2026 preprint: directly relevant probabilistic permutation methods. Their replica interaction and update schemes differ from these fixed-size conditional blocks.
- Y. Pan et al., [Learning to Solve the Quadratic Assignment Problem with Warm-Started MCMC Finetuning](https://arxiv.org/abs/2604.20109v1), 2026 preprint: PLMA combines learned permutation models and MCMC adaptation; training and adaptation costs would be necessary in a future matched comparison.
- M. Kuramata, R. Katsuki and K. Nakata, [Larger Sparse Quadratic Assignment Problem Optimization Using Quantum Annealing and a Bit-Flip Heuristic Algorithm](https://arxiv.org/abs/2012.10135v3), ICIEA 2021: quantum-annealing and classical-repair context; different constrained encodings and hardware prevent a size-only performance comparison.
- G. Yan et al., [Rethinking Parity Check Enhanced Symmetry-Preserving Ansatz](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7798f18a1e91aff7ad1df4841d6a01a8-Abstract-Conference.html), NeurIPS 2024: abstract-level background on hard constraints in variational quantum algorithms, including QAP; not reproduced here.
- [Python](https://www.python.org/), [NumPy](https://numpy.org/), [PyTorch](https://pytorch.org/), [Triton](https://github.com/triton-lang/triton), [CUDA](https://docs.nvidia.com/cuda/), [GCC](https://gcc.gnu.org/), [uv](https://docs.astral.sh/uv/), [pytest](https://docs.pytest.org/), and [Ruff](https://docs.astral.sh/ruff/): implementation, environment and checks; exact versions belong to the release lock and measured environment record.
