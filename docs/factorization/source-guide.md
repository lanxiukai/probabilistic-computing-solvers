# Source guide

English | [简体中文](source-guide.zh-CN.md)

The Python package is `sparse_factorization`; the command is `factor-bench`.
This repository has independent source, locks, and Git history. Running and
reproducing it requires no private-workspace paths. See the [method](method.md)
for scope and [reproduction](reproduction.md) for commands.

## Layout and responsibilities

| Location | Responsibility |
|---|---|
| [core/](../../src/sparse_factorization/core/) | Sparse models, clamps/constants, adjacency, coloring, and RNG |
| [netlist.py](../../src/sparse_factorization/solver/netlist.py), [gates.py](../../src/sparse_factorization/solver/gates.py) | Multiplier netlist and gate penalties |
| [edge_compiler.py](../../src/sparse_factorization/solver/edge_compiler.py) | Compact sparse construction without a full dense matrix |
| [circuit_proposal.py](../../src/sparse_factorization/solver/circuit_proposal.py) | Root/body proposals, actual finite mass, and MH acceptance |
| [circuit_deferred_radius.py](../../src/sparse_factorization/solver/circuit_deferred_radius.py), [circuit_deferred_triton.py](../../src/sparse_factorization/solver/circuit_deferred_triton.py) | Coordinate screening, first survivor, and deferred exact roots |
| [circuit_graph.py](../../src/sparse_factorization/solver/circuit_graph.py) | CUDA Graph execution and buffer lifetime |
| [publication_runtime.py](../../src/sparse_factorization/solver/publication_runtime.py) | Independent physical starts, engine preparation, FP64 certification |
| [runner.py](../../src/sparse_factorization/runner.py), [repro.py](../../src/sparse_factorization/repro.py) | Bounded execution, source snapshots, isolated restoration |
| [stream.py](../../src/sparse_factorization/stream.py), [replay.py](../../src/sparse_factorization/replay.py) | Sealed chunks, incremental transfer/checks, origin replay |
| [arithmetic.py](../../src/sparse_factorization/arithmetic.py), [classical.py](../../src/sparse_factorization/classical.py) | Matched-noise integer control and pinned YAFU |
| [protocol.py](../../src/sparse_factorization/protocol.py), [report.py](../../src/sparse_factorization/report.py) | Frozen panels, complete denominators, data-driven reports |
| [templates/](../../src/sparse_factorization/templates/) | Paired Markdown report text; source code remains English |
| [configs/factorization/](../../configs/factorization/) | Instances, seed roles, controls, resources, and stopping conditions |
| `tests/factorization/` (local only) | Arithmetic, transition equivalence, certificates, streaming, and recovery checks |
| [baselines/factorization/](../../baselines/factorization/), [build_yafu.py](../../scripts/factorization/build_yafu.py) | Baseline versions, dependency sources, and build |
| [data/factorization/](../../data/factorization/) | Versioned core evidence |
| [provenance/factorization/](../../provenance/factorization/) | Extraction and numeric-module checks |

## Execution and evidence

`prepare-run` freezes inputs, protocol, and source; `run` verifies imports and
executes from an isolated restoration. The main loop consumes every declared
random word, generates arithmetic candidates and a complete body, applies MH,
and observes the actual accepted state. Chunks preserve continuous event state,
RNG, counters, and certificates. Incremental verification and origin replay
answer different questions. Completed directories are not executed again;
recovery uses separate destinations and retains originals.

The method references [Aadit](experiment-report.md#ref-2022-aadit-sparse-ising-machines),
[Hastings](experiment-report.md#ref-1970-hastings),
[Tierney](experiment-report.md#ref-1998-tierney-metropolis-hastings-kernels), and
[Hittmeir](experiment-report.md#ref-2022-hittmeir-factorization-subset-sum).
Implementation references include
[PyTorch](experiment-report.md#ref-pytorch-cuda),
[Triton](experiment-report.md#ref-triton-docs), and
[CUDA](experiment-report.md#ref-nvidia-cuda).

## Accepted source and maintenance

[Release provenance](../../provenance/factorization/release-v3.json) binds 47
numerical/runtime files byte-for-byte to the executed source. The selected
`buffer_sqrt_qr_v1` profile combines reusable noise buffers, integer square-root
kernels, and fused residue checks; reference profiles remain for equivalence tests.
`verify_results.py` checks published core evidence on CPU; `plot_results.py`
regenerates tables and figures. Source snapshots include the license and paired
README when present, alongside package/templates, problem tools/tests, configs,
baseline locks, and environment metadata. Other problems are excluded.

Packaging/report changes alter the aggregate source fingerprint, so numerical
file identity and the original executed hash are recorded separately. Public
plan metadata omits operational history while preserving every numerical run
configuration and seed. New research needs a separate accepted scope; all used
sources remain in the report bibliography.
