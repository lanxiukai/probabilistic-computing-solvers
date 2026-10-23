# Reproduce the combinatorial studies

English | [简体中文](reproduction.zh-CN.md)

These seven modules complement the existing
[factorization](factorization/reproduction.md) and
[spin-glass](spin-glass/reproduction.md) releases. The new RTX 5090 campaign
contains 365 natural calls and 35 separate origin/work replays. Sparse and dense
Max-Cut have independent panels. Replays do not add independent trials.

Before comparing resources, read the [benchmark overview](benchmark-overview.md)
for per-model bits, parallel replicas, complete timing, and memory conventions.
Concepts and background are introduced in [probabilistic computing and p-bits](probabilistic-computing.md).

## Environment and small examples

Run all commands from the repository root on Linux x86-64. The checked-in
`.python-version` selects Python 3.14.6, and `uv.lock` pins the dependencies,
including the CUDA 13.0 PyTorch build. Small CPU examples need no CUDA device.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first,
then use this repository's single locked project:

```bash
uv sync --frozen
uv run --frozen python -m combinatorial_solvers.public_api --list
uv run --frozen python scripts/release/make_examples.py --output examples/small
uv run --frozen python -m combinatorial_solvers.public_api --problem mwis \
  --config examples/small/mwis.json --run-id example-mwis \
  --output-root artifacts/examples
```

Small examples use CPU and generated inputs. They test the source/input isolation
and certificate pipeline; they are not additional performance measurements.
The public runners keep original model size separate from stochastic update
p-bits. Classical controls count zero p-bit updates.

## Verify the published core

The CPU verifier reconstructs original objectives from the complete saved best
states, including infeasible LDPC attempts. It checks input and witness hashes,
strict partition balance, LDPC syndrome/auxiliary constraints, and error rates.
The complete trajectory audits and source-recovery receipts remain preserved
in the private research archive.

```bash
uv run --frozen python scripts/release/verify_core.py \
  --panel tsp --panel mwis --panel maxsat --panel maxcut-sparse \
  --panel maxcut-dense --panel labs --panel partition --panel ldpc --fetch
```

`--fetch` obtains the pinned TSPLIB, Gset, and KaHIP benchmark files from their
original sources into ignored `external/` paths. Those benchmark files are not
redistributed here. Generated dense graphs are reconstructed from frozen seeds
and checked against their mathematical input identities. Large dense/LABS
certificate checks can take several CPU minutes.

The LDPC `evaluation/` files contain synthetic channel truth for evaluation.
They are never passed to the decoder, whose input is only checks, syndrome, and
quantized reliability. WiFi matrix provenance is covered by the retained
[MIT notice](../baselines/ldpc/wifi-LICENSE).

## Reproduce a call

Each panel exports its natural calls. TSP additionally includes its explicitly
marked origin replay; other panels attach fixed-work replay settings to the
natural origin and retain replay checks in `verification.json`.
Use `trial_kind` and the verification counts when counting independent trials.
A natural call is one execution, not necessarily a distinct problem instance.
The complete panel accounting is:

| Panel | Natural calls | Recorded replays |
|---|---:|---:|
| tsp | 16 | 1 |
| mwis | 78 | 4 |
| maxsat | 40 | 5 |
| maxcut-sparse | 110 | 5 |
| maxcut-dense | 20 | 5 |
| labs | 20 | 4 |
| partition | 40 | 5 |
| ldpc | 41 | 6 |
| Total | 365 | 35 |

Each `data/<panel>/rtx5090/calls.json` contains the configuration, input identity,
resource ledger, core certificate, and timing fields. Panel names are `tsp`,
`mwis`, `maxsat`, `maxcut-sparse`, `maxcut-dense`, `labs`, `partition`, and `ldpc`.

```bash
uv run --frozen python scripts/release/run_case.py \
  --panel mwis --case n2048-i0-greedy \
  --output artifacts/reproductions/mwis-greedy

uv run --frozen python scripts/release/run_case.py \
  --panel mwis --case n32768-i0-s0-pair-gibbs --mode replay \
  --output artifacts/reproductions/mwis-origin
```

The first command makes a fresh call; the second uses a published fixed-work
origin configuration. Matching seed alone does not ensure a matching trajectory
on another GPU/runtime. Time-limited fresh calls can finish at different work
counts. Input materialization and wrapper packaging can also change preparation
costs; the report figures always refer to the archived measurements.

Warm-start experiments require an explicit choice. Regenerate the stated
`warm_start_origin` with the external reference and pass its output through
`--warm-input`, or select `--use-recorded-warm-input` for a conditional experiment.
Replay mode reuses the original initial input. Such reuse does **not** remeasure
BP/KaHIP initialization, and the tool never reports historical initialization
cost as a new observation. Cold runs do not read saved solutions.
`--warm-input` expects a text file with one 0/1 partition label per vertex for bisection, or an NPZ file with an
`errors` array of shape `(frames, original_bits)` for LDPC. The supplying
initialization call and its cost must be recorded separately.

External reference cases require `--baseline-directory`. Input hashes are strict
by default; `--allow-rebuilt-baseline` explicitly records deviations for a new
build. Use the pinned source versions and project adapters described in
[external references](external-baselines.md). This flag is not a claim of source
equivalence or a license to use third-party code.

## Timing and statistical units

The JSON field name and each report's table heading define the timing boundary:

| Field | Included interval |
|---|---|
| `runner_seconds`, `complete_runner_seconds` | Runner preparation, search, state output and certification; excludes outer source restoration and process launch |
| `launcher_seconds` | Outer execution wrapper, including restoration/process overhead where recorded |
| `common_initial_runner_seconds` | Separately measured BP/KaFFPa initialization, added to warm-run cost once |
| `censored_target_seconds` | Max-SAT first certified target time including runner preparation; a miss receives the 60-second cap |
| `prepare_plus_first_seconds` | TSP preparation/launch plus first returned, certified target tour |

These intervals overlap; do not add runner time to launcher time. Installation,
external-reference compilation and later archive verification are not included
in the solver timings. A search budget is not a complete-call budget.
For example, 45-second bisection search can have much longer preparation.

Means/ranges in the seven reports are descriptive summaries of the listed calls.
Repeated seeds share graphs, formulas, sequence lengths or channel frames;
trajectories within a call are not extra benchmark instances. The reports do not
infer confidence intervals or general rankings from these small reused panels.

## Regenerate combinatorial figures

```bash
uv run --frozen python scripts/release/plot_results.py --topic mwis
```

This rebuilds `docs/mwis/figures/rtx5090.png` and `.svg` in place and writes
`docs/mwis/figure-values.json` from saved calls. Omit `--topic` for all seven
topics, or repeat it to select several. It does not execute a solver or rewrite
report prose. Figures and descriptive tables use the same saved metrics.

## Interpretation and checks

Reports distinguish initialization, search, certification, and complete time;
the same time window does not imply equal hardware, populations, or energy use.
Best-known/heuristic references are not proven optima. Native high-order models
are not automatically realizable with the same number of quadratic hardware sites.
Negative results, misses, and the bounded external-reference limitations are
part of the release.

```bash
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

CUDA checks skip without a CUDA device. Published GPU measurements and preflight
were obtained on RTX 5090; CPU certificate verification does not require a GPU.

## Separate weighted set-cover confirmation

This panel contains 117 natural calls and four work replays that add no trials.
Fetch the original OR-Library inputs by their fixed identities, then independently
verify the released certificates:

```bash
uv run --frozen python scripts/set_cover/fetch.py --output data/set-cover/or-library-v1
uv run --frozen python scripts/set_cover/verify_release.py --repository .
```

The [set-cover source guide](set-cover/source-guide.md) covers isolated runs,
full-work replays, AC-CFT rebuilding, and figure generation. This topic uses its
own NumPy verifier, rather than the historical `verify_core.py --panel` interface.

The JSSP [report](jssp/report.md) retains 72 natural calls and three replays. Its [source guide](jssp/source-guide.md) provides input downloads, independent core-witness checks and isolated work replays.

Covering arrays retain 50 natural confirmation calls and three replays, plus a separate 10-call development panel and one replay. The [report](covering-array/report.md) and [source guide](covering-array/source-guide.md) describe the model, independent certificates and reproduction steps.

Coloring keeps final confirmation, development comparisons and fixed-work implementation panels separate. Original DIMACS inputs are fetched by pinned hashes. The owned TabuCol-style program is an independent CPU reference, not an upstream HEAD reproduction. See the [report](graph-coloring/experiment-report.md) and [source guide](graph-coloring/source-guide.md).

QAP retains 72 independent confirmation calls, four implementation controls and five replays, with separate Q1c development and official Ro-TS ledgers. The [report](qap/experiment-report.md) and [model/reproduction guide](qap/README.md) give complete parameters, witness checks and panel boundaries.
