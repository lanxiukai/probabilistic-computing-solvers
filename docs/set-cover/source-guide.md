# Set-cover source and reproduction guide

[简体中文](source-guide.zh-CN.md) · [Overview](README.md) · [Report](report.md)

## Source map

| Source | Responsibility |
|---|---|
| [`model.py`](../../src/combinatorial_solvers/set_cover/model.py) | Original sparse covering constraints, SCP/rail parsers and integer certificates |
| [`blocks.py`](../../src/combinatorial_solvers/set_cover/blocks.py) | Input-only block banks, including the selected `cost-minhash` policy |
| [`gpu.py`](../../src/combinatorial_solvers/set_cover/gpu.py) | Triton block updates, row-coverage caches and active-column selection |
| [`solver.py`](../../src/combinatorial_solvers/set_cover/solver.py) | Greedy prefix, independent replica state, temperature schedules and deadline-qualified observations |
| [`baseline.py`](../../src/combinatorial_solvers/set_cover/baseline.py) | Classical greedy support, AC-CFT adapter and SciPy/HiGHS LP/MILP references |
| [`reference.py`](../../src/combinatorial_solvers/set_cover/reference.py) | Small reference calculations and exact rational dual checks |
| [`runner.py`](../../src/combinatorial_solvers/set_cover/runner.py) | Configured workflow, original-state certification and stage timing |
| [`verify_release.py`](../../scripts/set_cover/verify_release.py) | Independent portable verification without importing a solver |

The accepted route uses `representation="feasible"`, `block_policy="cost-minhash"`, `selection="active"`, width 8, 16 banks and 128 replicas. Use the released per-call configurations; generic solver defaults are not the accepted SC8 configuration. Alternative methods present in source are not additional accepted result panels.

## Restore inputs and verify saved results

Run commands from the public repository root in its locked uv environment. The [benchmark manifest](../../configs/set-cover/benchmarks.json) pins the nine files from the original owner. The fetcher restores their compressed representation and verifies the original numerical input; existing unrelated files are not required.

```bash
uv sync --frozen
uv run --frozen python scripts/set_cover/fetch.py \
  --output data/set-cover/or-library-v1
uv run --frozen python scripts/set_cover/verify_release.py \
  --calls data/set-cover/rtx5090/calls.json \
  --repository .
```

The verifier itself needs only Python and NumPy and performs no search or GPU work. Its output reports the complete 117 natural calls, four replays, 43,045 retained state witnesses, 27 heat-bath natural calls and 15 heat-bath target-hit calls. It also checks the historical cost-170 witness outside the 43,045 count. Replays contribute zero new independent trials. It checks the curated public witnesses rather than rerunning the original experiment or reproducing the original full-archive count of 1,083,941 states.

The [panel](../../data/set-cover/rtx5090/panel.json) records exact natural-call and replay identifiers plus frozen targets. The [call ledger](../../data/set-cover/rtx5090/calls.json) binds each configuration, benchmark input and witness archive by path and hash. `benchmark_inputs.path` and `configuration_file` are repository-relative; `witness_file` is relative to the ledger's directory. [Environment metadata](../../data/set-cover/rtx5090/environment.json) identifies the measured device and software.

For native runs, `best_bits`/`best_cost` are the deadline-qualified quality witnesses; `final_bits`/`final_cost` can include the overrun chunk. Every original row and integer cost is recomputed from the retained bit arrays. `observation_indices` points into the original every-chunk sequence: all replica-best improvements, first/last observations and both sides of eligibility transitions remain. Missing intermediate current-state fluctuations are not reconstructed or counted as verified witnesses.

Rational dual certificates use exact integer inequalities and are distinct from solver-reported floating bounds/gaps. A certified optimum requires a matching original feasible witness and rational lower bound. The separately retained [SCPNRG4 target witness](../../data/set-cover/reference-witnesses/scpnrg4-target170.npz) binds the earlier feasible cost 170; it supplies no additional SC8 trial or performance measurement.

## Execute a new isolated run

An RTX 5090-compatible CUDA environment is needed for the performance solver. The unified public entry point snapshots the accepted source, inputs and configuration into a new isolated run, executes it and certifies the original covering constraints:

```bash
uv run --frozen python -m combinatorial_solvers.public_api \
  --problem set-cover \
  --config data/set-cover/rtx5090/configs/sc8-scp41-heatbath-s1.json \
  --run-id reproduce-scp41-heatbath-s1
```

Use a fresh run identifier for each execution. The example is a new 30 s native search with its own initialization, compilation/warmup and certification cost; its result does not overwrite or add a trial to the frozen release. Repeat the method's complete declared seed set when comparing outcomes. A GPU call includes 128 replicas, while a classical reference uses one CPU worker. Complete elapsed time and the original greedy prefix must remain visible when attributing any gain to p-bit search.

Work replays are a different operation: they freeze completed work and the original quality step fence. Their clock values are not new performance evidence. The released verifier checks retained non-timing arrays against the declared original call. A fresh natural run should use the natural-call configuration, not a replay configuration.

## Restore the strong classical reference

AC-CFT is pinned to revision `6cb63d0036223b8be9c4d01f95b296d641209f95`; {fmt} is pinned to 10.2.1. The [AC-CFT manifest](../../configs/set-cover/cft-source.json) binds each required source file, and the [{fmt} manifest](../../configs/set-cover/fmt-source.json) binds its archive. Fetch source from its owner, then build for the current CPU with an available C++ compiler:

```bash
uv run --frozen python scripts/set_cover/fetch_baseline.py \
  --output data/set-cover/baselines/cft-v1
uv run --frozen python scripts/set_cover/prepare_baseline.py \
  --source-directory data/set-cover/baselines/cft-v1 \
  --output data/set-cover/baselines/cft-build-v1
uv run --frozen python -m combinatorial_solvers.public_api \
  --problem set-cover \
  --config data/set-cover/rtx5090/configs/sc8-scp41-cft-s1.json \
  --run-id reproduce-scp41-cft-s1
```

The fetch/build destinations are new directories. The build keeps upstream licenses, uses a separately pinned {fmt} dependency and produces both the observed and pristine adapters. It uses `-march=native`; build locally for the chosen CPU rather than assuming a binary is portable. The timer observer returns unchanged native times during natural calls and only substitutes a recorded sequence during explicit replay. Search scoring, proposals and stopping comparisons remain unchanged. Independent verification of the published mathematical witnesses does not require this executable or a compiler.

AC-CFT's nominal 30 s is a soft search limit; report its actual returned-witness time, including any overrun. The nine 60 s LP/MILP references serve bounding/certification and are not equal-time controls. Keep their solver-reported optima separate from rationally closed proofs, as detailed in the [report](report.md).

## Checks and references

```bash
uv run --frozen ruff check src/combinatorial_solvers/set_cover scripts/set_cover
uv run --frozen ruff format --check src/combinatorial_solvers/set_cover scripts/set_cover
```

Maintainer tests are local-only and are not included in the public clone; see the [repository guide](../repository-guide.md). Cloud software checks passed, including the independent verifier's original-input, deadline, replay, dual and deliberate-corruption tests. Those verifier tests execute no solver. A fresh clone can check the released certificates using the verification command above. Software checks do not create benchmark evidence or authorize an enlarged experiment panel. The report's [linked bibliography](report.md#references) covers benchmark ownership, CFT, earlier Ising/higher-order formulations, and all numerical/runtime software used here.
