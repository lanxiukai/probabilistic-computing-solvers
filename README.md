# Probabilistic Computing Solvers

English | [简体中文](README.zh-CN.md)

> [!NOTE]
> The vast majority of this repository's contents were produced automatically by
> GPT-6-Astra and have not undergone a complete human review. I will carry out
> human review from time to time, but cannot commit to a timeline.

**Probabilistic computing** uses controlled randomness for sampling, inference,
and optimization. A **p-bit** is a classical random bit with a tunable output
probability; coupled p-bits express correlations and constraints in a problem.
Read [Probabilistic computing and p-bits](docs/probabilistic-computing.md) for
principles, historical background, and limitations.

This repository centers on audited GPU simulations of workflows with p-bit
participation, with classical algorithms as comparison references and explicit
hybrid support. It provides reproducible certificates. It retains favorable
and unfavorable results; feasibility, optimality, sampling, and physical hardware
measurements are distinct claims.

GPU results for all topics below were measured on an **NVIDIA GeForce RTX 5090**.

| Project | Scientific and practical relevance | Measured scope and report |
|---|---|---|
| Integer factorization | Integer algorithms and arithmetic constraints | Final 64–128-bit panel · [Report](docs/factorization/experiment-report.md) |
| 3D EA spin glass | Disordered physics and low-energy search | 16³/24³/32³ · [Report](docs/spin-glass/experiment-report.md) |
| TSP | Route planning and permutation constraints | 200–2103-city GPU p-bit workflow, up to 33,662 binary nodes · [Report](docs/tsp/candidate-report.md) |
| Weighted independent set | Conflict-aware selection and scheduling | Native model through 32768 variables · [Report](docs/mwis/experiment-report.md) |
| Weighted Max-SAT | Configuration, planning, and constraint reasoning | Native soft clauses, 32768 variables · [Report](docs/maxsat/experiment-report.md) |
| Max-Cut | Binary decisions and Ising optimization | Sparse and full dense 32768-variable models · [Report](docs/maxcut/experiment-report.md) |
| LABS | Radar/communication sequences and low autocorrelation | Native aperiodic high-order model through length 32768 · [Report](docs/labs/experiment-report.md) |
| Balanced graph bisection | Parallel computing and network partitioning | Strict cardinality, 32768 vertices · [Report](docs/partition/experiment-report.md) |
| LDPC codeword search | Communication/storage error correction | Native 32766 and quadratic 32760 update bits · [Report](docs/ldpc/experiment-report.md) |
| Weighted set cover / hitting set | Test selection, resource allocation, and coverage planning | 1,000–47,311 original variables, 128 replicas · [Report](docs/set-cover/report.md) |
| Job-shop scheduling | Manufacturing schedules, resource conflicts and precedence constraints | 216–37,500 binary entries, 64–256 replicas · [Report](docs/jssp/report.md) |
| Covering arrays | Combinatorial software testing and interaction coverage | 108–31,500 binary entries, 128 replicas · [Report](docs/covering-array/report.md) |
| Graph / list coloring | Register allocation, conflict scheduling and combinatorics | Up to 46,656 stored bits / 28,026 original factor-attached bits, 128 chains · [Report](docs/graph-coloring/experiment-report.md) |
| Quadratic assignment | Facility layout, interacting assignments and graph correspondence | 144–22,500 constrained binary entries, 128/512 replicas · [Report](docs/qap/experiment-report.md) |

The [benchmark overview](docs/benchmark-overview.md) summarizes model sizes,
timings, parallel states, peak allocator memory, and the main limitations.

The earlier additional combinatorial campaign contains **365 natural calls and 35 separate
work/origin replays** on RTX 5090, with CPU controls measured on the same host.
The accepted factorization and spin-glass data remain unchanged. Factorization
retains 52/54 certified successes, including 7/9 at 128 bits; large spin-glass
ground states remain unknown. The new TSP candidate-network study adds 28 natural
calls and one separate replay; none of its seven p-bit calls improves the prefix.

The separate set-cover panel adds 117 natural calls and four work replays: all
27 p-bit calls improve their greedy prefix, 15 meet frozen targets, and none
outperforms AC-CFT.

```bash
uv sync --frozen
uv run --frozen python -m combinatorial_solvers.public_api --list
uv run --frozen python scripts/release/verify_core.py --panel mwis
```

Start with the [reproduction guide](docs/reproduction.md), the
[external-reference boundaries](docs/external-baselines.md), or the
[repository guide](docs/repository-guide.md). Project-authored code is
[MIT licensed](LICENSE); retained third-party notices and external licenses
remain applicable.
