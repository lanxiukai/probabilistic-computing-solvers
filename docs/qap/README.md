# Quadratic assignment with constrained p-bit blocks

English | [简体中文](README.zh-CN.md)

This module studies a GPU simulation of probabilistic binary assignment variables for the quadratic assignment problem (QAP). Accepted independent confirmation, development evidence and official CPU references are reported in the [experiment report](experiment-report.md). The largest natural benchmark is `tai150b`: 150 assignments, 22,500 persistent one-hot binary variables per model. Replica counts are reported separately.

QAP captures interacting assignments, such as facility layout and graph correspondence. It combines a strict permutation constraint with pairwise costs between assigned objects. A good feasible permutation is easy to evaluate; proving global optimality on difficult instances is a separate task. These experiments use the generic integer QAPLIB instances and do not claim to reproduce a complete physical or industrial workflow. [QAPLIB](https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/)

## Model and transition

For nonnegative integer matrices $A,B$, minimize

$$C(\pi)=\sum_{i,j}A_{ij}B_{\pi(i),\pi(j)},\qquad \pi\in S_n.$$

The persistent binary state is $x_{ia}=1$ exactly when $\pi(i)=a$. Each row and column contains one active bit. The implementation keeps a permutation cache equivalent to this full binary state and verifies their equality. The binary variable count is $n^2$; the state space has $n!$ feasible assignments, so those variables are constrained rather than independent. GPU memory bits and simulated model variables are not a measurement of fabricated p-bit hardware.

Adding squared row/column constraint penalties would produce a QUBO formulation. The evaluated solver instead enforces those constraints in its block transitions. Its $n^2$ count therefore describes the maintained feasible binary model, not a measured unconstrained single-bit QUBO machine or a count of independent degrees of freedom.

At each transition, select two or four distinct positions independently of the current state. Enumerate the 2 or 24 permutations of their currently occupied locations. The accepted pattern has probability

$$P(\sigma\mid x_{\bar S})=\frac{\exp[-\beta(E_\sigma-E_{\min})]}{\sum_\tau\exp[-\beta(E_\tau-E_{\min})]}.$$

This is a joint conditional heat bath on a feasible one-hot block. It directly updates the persistent binary assignment and preserves the two permutation constraints. The integer local energy includes interactions with all outside positions and every ordered pair inside the block, including asymmetric terms and diagonals. It avoids materializing an $n^4$ coupling tensor. A fixed-temperature transition has the conditional Gibbs form; the cyclic temperature schedule is used as an optimizer. No equilibrium, mixing-time, quantum advantage, or hardware energy-efficiency claim is made.

The factored evaluation reuses an exact outside-interaction table. For selected positions $i_a$ and their old locations $\ell_b$,

$$T_{ab}=\sum_{k\notin S}\left(A_{i_a k}B_{\ell_b,\pi(k)}+A_{k i_a}B_{\pi(k),\ell_b}\right),$$
$$E_\sigma=\sum_aT_{a,\sigma(a)}+\sum_{a,b}A_{i_a i_b}B_{\ell_{\sigma(a)},\ell_{\sigma(b)}}.$$

This derivation reduces repeated outside work from $O(b!bn)$ to $O(b^2n)$ for block width $b$, leaving the $O(b!b^2)$ inside terms. It changes energy evaluation rather than the stochastic rule. The candidate implementation remains selectable. Both final fully prewarmed pairs match all 20 non-time state and RNG arrays; their [source-bound records](../../data/qap/rtx5090/implementation-comparisons.json) distinguish search, complete-runner and launcher ratios. Search includes random generation, host observations, transfers and recovery I/O, so its timing ratio is not an isolated CUDA-kernel speedup.

Every replica starts from a uniformly shuffled permutation. Reference solutions never initialize, repair, or guide the search. There is no classical search prefix or polishing stage. Same-block Metropolis and random-tie minimum-energy transitions isolate the update rule; a separately implemented, single-thread attribute-tabu search is an additional classical comparison. The latter is inspired by Taillard and is not advertised as an exact reproduction of the published robust tabu implementation. [Taillard, 1991](https://mistic.iict-heig-vd.ch/taillard/articles.dir/Taillard1991.pdf)

The Metropolis control shares the complete candidate-energy computation with the heat bath. It is a controlled comparison of the transition rule, not a throughput-optimized GPU simulated-annealing implementation. Development selects the heat-bath settings; the controls reuse those settings rather than receive an equal independent tuning campaign. Results against that control must therefore be read alongside the independent tabu comparison; they cannot establish superiority over the best classical GPU algorithms. The independently implemented tabu search also has not received an equal tuning budget or been validated as a state-of-the-art implementation.

A separate external reference uses Taillard's official 2006 C Ro-TS implementation with its published `8*n` tabu-duration and `5*n*n` aspiration parameters. The measurement adapter preserves upstream move, tenure and delta logic, exposes the six-word RNG state, records original witnesses at completed iterations, and disables reference-informed early stopping. A cloud software gate compared its returned witness and next 64 RNG values against the unchanged upstream functions at identical work. The completed Q3 panel has 18 single-thread CPU trials and three work replays; all 5,085 saved witnesses passed full audit. Its classical outcomes remain separate from p-bit results and do not select the native confirmation settings. [Official source page](https://mistic.iict-heig-vd.ch/taillard/)

## Verification and scope

Tests compare every transition against complete original-objective enumeration on asymmetric small matrices, including nonzero diagonals; additional 150-order cases check larger CUDA launch layouts. Original full-matrix integer evaluation validates saved current, best, and eligible-best permutations. One-hot consistency, periodic states, objective histories, source/input manifests and fixed-work replays are audited separately.

The implementation requires matrix entries below $2^{62}$ and the conservative bound $n^2\max(A)\max(B)<2^{52}$. This makes the vectorized full-objective `int64` certificate exact and keeps delta arithmetic within range. Each reported ensemble winner also receives a separate unbounded Python-integer certificate. Faster batch verification is accounted for as verification work, not as faster p-bit sampling.

Search time includes random generation, synchronization, observations and periodic recovery I/O. GPU preparation/warm-up compilation and final certification/archive costs are reported separately and included in complete runner time. CPU reference binaries are built once outside the per-call timers. A chunk finishing after its search deadline is retained as evidence but excluded from eligible quality. First-reference times are upper bounds at chunk observations. A complete seeded ensemble is one independent trial; replicas describe within-ensemble search diversity, and replays add no trials.

The target size is adjusted to the unmodified public benchmark. We use 22,500 variables for `tai150b` rather than manufacture a 32K instance. A measured model size, visited-variable count, feasible result, reference hit and demonstrated advantage are different quantities.

## Benchmarks and reproduction

`configs/qap/benchmarks.json` records original download URLs, SHA-256 values, reference status and permutation conventions. `tai40a` uses a zero-based reference permutation; `tai80a` requires its inverse. The reference is always checked against the exact original objective. Nugent references are catalogued optima; Taillard values are historical feasible references, not newly proved optima or verified current records.

```bash
uv run --frozen python scripts/qap/fetch_benchmarks.py --output data/qap/qaplib
```

Build the self-authored CPU comparison only when reproducing its CPU calls:

```bash
uv run --frozen python scripts/qap/prepare_reference.py --output data/qap/baselines/attribute-tabu
```

The optional official Ro-TS reference has separate **non-commercial-use and acknowledgement conditions**. The downloaded upstream source, instrumented full source and binaries are excluded from this repository's public source distribution and are not covered by its MIT license. Read the conditions before choosing this external build:

```bash
uv run --frozen python scripts/qap/prepare_rots.py \
  --output data/qap/baselines/taillard-rots \
  --accept-noncommercial-conditions
```

Only the self-authored fetch/build and measurement adapters are distributed here. The original file is hash-pinned; full upstream bytes and exact compiler/build identities are retained by the local experiment archive. Original integer bounds are checked before using the upstream 32-bit implementation. This external comparison has no p-bit contribution.

Verify each accepted ledger's original integer witnesses without launching search:

```bash
uv run --frozen python scripts/qap/verify_release.py \
  --repository . --calls data/qap/rtx5090/calls.json
uv run --frozen python scripts/qap/verify_release.py \
  --repository . --calls data/qap/development-q1c/calls.json
uv run --frozen python scripts/qap/verify_release.py \
  --repository . --calls data/qap/external-rots/calls.json
```

This check needs the fetched benchmark bytes and the released core records, but does not need either CPU executable or an external Ro-TS build. To reproduce one native seeded ensemble, prepare a new isolated source/input snapshot, execute it, and verify its sealed file inventory:

```bash
uv run --frozen python scripts/qap/experiment.py prepare \
  --config data/qap/rtx5090/configs/q2-nug30-heatbath-s2026102411.json \
  --run-id reproduce-qap-nug30-s2026102411
uv run --frozen python scripts/qap/experiment.py run \
  --run-dir artifacts/runs/qap/reproduce-qap-nug30-s2026102411
uv run --frozen python scripts/qap/experiment.py verify \
  --run-dir artifacts/runs/qap/reproduce-qap-nug30-s2026102411
```

Use a fresh run ID for another execution. A wall-clock-limited rerun can complete a different amount of work, so the same seed alone does not promise identical final arrays. The separate `replay-q2-nug30-heatbath-s2026102411.json` configuration fixes the recorded work and eligible-quality boundary; use it with another fresh run ID for a work replay in the validated environment. Reproduction and timing remeasurement add no independent confirmation trials. Cloud control protocols and private source histories are not part of the public interface.

The three ledgers retain 81 Q2, 12 Q1c and 21 official-reference records, with independent confirmation, tuning, implementation controls and replays labeled separately. The public package uses the previously executed Q3 shared-source version: six modules and the native runner branch also match Q2/Q1c, while the runner adds external-reference dispatch. Nine public-source work replays match the original non-time arrays. Original timings remain bound to their executed source versions in [provenance](../../provenance/qap/release.json). CPU build-input hashes identify measured artifacts; compiler, host and paths can change rebuilt products, while original source identities remain fixed.

## Related work and software

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
