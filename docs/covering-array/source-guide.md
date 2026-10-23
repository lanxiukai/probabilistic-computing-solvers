# Covering-array source and reproduction guide

[简体中文](source-guide.zh-CN.md) · [Overview](README.md) · [Method](method.md) · [Report](report.md)

## Source map

| Source | Responsibility |
|---|---|
| `src/combinatorial_solvers/covering_array/model.py` | Original strength-two problem, parameter parsing and integer coverage certificates |
| `src/combinatorial_solvers/covering_array/gpu.py` | GPU state allocation, row-block updates and dispatch |
| `src/combinatorial_solvers/covering_array/column_gpu.py` | Exact two-row same-column candidate energies and update rules |
| `src/combinatorial_solvers/covering_array/solver.py` | Uniform initialization, warmup, clocks, strict eligible quality and recovery boundaries |
| `src/combinatorial_solvers/covering_array/baseline.py` | ACTS/IPOG process and complete-array readout |
| `src/combinatorial_solvers/covering_array/runner.py` | Configured execution, independent certificates and stage timing |
| `scripts/covering_array/experiment.py` | Isolated source/input preparation, execution and manifest verification |
| `scripts/covering_array/fetch_acts.py` | Pinned upstream ACTS download and existing-file verification |
| `scripts/covering_array/verify_release.py` | Portable core-witness, target, roster and claim verification |

The public package is `combinatorial_solvers`. This problem uses the shared isolated-execution and evidence framework without importing another problem's solver. Use released configurations: generic solver defaults are not the selected column-pair, constant-$\beta=8$, 128-replica route.

## Restore the reference executable and verify results

Run from the public repository root in its locked uv environment:

```bash
uv sync --frozen
uv run --frozen python scripts/covering_array/fetch_acts.py \
  --manifest configs/covering-array/external-baselines.json \
  --output data/covering-array/baselines/acts-basic.jar
uv run --frozen python scripts/covering_array/verify_release.py \
  --calls data/covering-array/rtx5090/calls.json \
  --repository .
uv run --frozen python scripts/covering_array/verify_release.py \
  --calls data/covering-array/development-ca2b/calls.json \
  --repository .
```

The catalog pins the official ACTS Basic download to 10,803,019 bytes and SHA-256 `8efe3cdd5847135381b1daba9e8d57d0d013d00b0b2a7c15dd7c4255f0fb1242`. A changed download is rejected before writing; an existing mismatched file is preserved and rejected. An existing matching file can be verified offline. The jar is fetched from its owner and is not distributed in this repository. The catalog links the NIST usage statement and records the measured Java version, OpenJDK 21.0.12.1. See [references](report.md#references).

The instance JSON files are project-created parameter declarations, not downloaded solutions. Their `rows`, `columns`, `levels` and `strength` completely specify each mathematical input. The initial arrays are separately reconstructed from the solver seed. Historical target-source annotations do not supply a reference array or alter search. Preserve the released JSON bytes because they are bound by digest.

The verifier uses Python and NumPy, imports no optimizer, executes no Java and requires no GPU. It still checks the downloaded jar's byte binding for ACTS calls. A successful check establishes original pair coverage and consistency of the released records; it does not independently measure the original clock or replay GPU dynamics.

## Release contract

`data/covering-array/rtx5090/calls.json` binds each call to an inline configuration, identical executable configuration, input digests and a core witness. `configuration_file` and `benchmark_inputs.path` are repository-relative. `witness_file` is relative to the directory containing `calls.json`. Baseline input paths are repository-relative as well. When an input includes `recorded_original_sha256`, it must match the restored byte digest.

The adjacent `panel.json` is required. Its `expected_natural_ids` and `expected_replays` preserve exact call membership and replay origins, not only totals. Each `frozen_targets` entry contains `cost: 0`, `status: "fixed-row-feasibility"`, a nonempty source and the exact `rows`, `columns`, `levels`, `strength`. Calls sharing an instance identifier must bind identical original parameter bytes. Zero target cost means no missing interactions at that row count; it does not mean globally minimum row count.

`data/covering-array/development-ca2b/` has its own `calls.json` and `panel.json`, retaining all 10 natural development calls and one work replay. Its single development seed, temperature comparisons and N29 misses are separate from the 50 natural confirmation calls and three confirmation replays. Apply the same verifier independently to both directories; their counts and seed roles must not be pooled.

Native witnesses retain initial/current/returned-best/eligible-best arrays and costs, one-hot state, visited entries, changed-block counts, GPU RNG endpoints and curated per-chunk observations. The verifier regenerates the uniform initial arrays, enumerates original pair coverage, checks one-hot and participation constraints, reconstructs deadline eligibility, and separates cold hits from later hits. RNG bytes are opaque in this CPU verifier and are checked for equality in work replay.

`curve_indices` preserves the original indices after retaining all replica-best improvements, endpoints and eligibility boundaries. The verifier checks the retained trace contract. The separately accepted full trace is needed to establish that no improvement was omitted; the core file cannot reconstruct all intermediate current states.

ACTS witnesses retain the complete constructed array. Documented `*` cells are set to zero without dropping rows, then all original interactions are checked. `constructed_rows` and `fixed_row_target_met` distinguish own-row coverage from the fixed target. The historical name `best_feasible_cost` denotes missing-interaction count; a positive native value is an incomplete array, and ACTS zero must be read together with its constructed row count.

## Execute an isolated reproduction

Prepare and run the first frozen small heat-bath call in a fresh output directory:

```bash
uv run --frozen python scripts/covering_array/experiment.py prepare \
  --config data/covering-array/rtx5090/configs/ca3-ca9k4v3-heatbath-s1.json \
  --run-id covering-array-small-reproduction
uv run --frozen python scripts/covering_array/experiment.py run \
  --run-dir artifacts/runs/covering-array/covering-array-small-reproduction
uv run --frozen python scripts/covering_array/experiment.py verify \
  --run-dir artifacts/runs/covering-array/covering-array-small-reproduction
```

The prepare step snapshots the executable source, configuration and inputs. Run an ACTS configuration through the same entry point after restoring the jar and installing Java. GPU reproduction requires a compatible CUDA environment; published performance remains specific to the measured RTX 5090 system. A natural rerun can complete different work under the same wall-clock limit. Repeating a released seed adds no independent confirmation trial.

The released `replay-ca3-ca9k4v3-heatbath-s1.json`, `replay-ca3-ca35k300v3-heatbath-s1.json` and `replay-ca3-ca30k300v3-heatbath-s1.json` configurations preserve each original completed-step and eligible-quality boundary. Select one with a fresh run ID to reproduce fixed work. Clock values from such a replay are not a new natural trial or a general hardware speed comparison.

Preparation includes initialization and compilation/warmup. Search includes RNG generation, GPU updates, observation transfers, trace writing and periodic checkpoints. Certification is timed separately. `complete_runner_seconds` spans runner entry through final array serialization, also including provenance and final state capture; final summary serialization and outer source restoration/process startup lie outside that timer. These overheads mean the component timers need not sum to the complete runner time.

Interrupted updates with an uncommitted device state remain diagnostic. Their atomic RNG/quality boundary can support a new full-origin work replay; they are not advertised as directly resumable completed checkpoints. Maintainer tests are not included in the public clone. Public users can run the witness verifier and scoped source checks:

```bash
uv run --frozen ruff check src/combinatorial_solvers/covering_array scripts/covering_array
uv run --frozen ruff format --check src/combinatorial_solvers/covering_array scripts/covering_array
```

Assembled-release validation completed on the RTX 5090 host. The fetcher identifies its research client and restores the pinned NIST jar with size and digest verification.
