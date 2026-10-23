# Running and verifying the solver

English | [简体中文](reproduction.zh-CN.md)

## Installation and software checks

Use Linux x86-64. Python and dependencies are fixed by `.python-version` and
`uv.lock`. Run commands from the repository root.

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen factor-bench --help
```

See the [source guide](source-guide.md) for module responsibilities. Official GPU
performance uses actual RTX 5090 measurements; software checks do not add
factorization trials. Tool references include
[uv](experiment-report.md#ref-uv), [Python](experiment-report.md#ref-python),
and the [software bibliography](experiment-report.md#software-references).

## Preparation and execution

`prepare-run` freezes source, configuration, and dependencies. `run` starts an
isolated restored process and verifies its import path. Each output directory
can run once. The examples use a local `artifacts/` directory, which is ignored
by Git; maintainers keep experimental outputs in the private research workspace.

```bash
uv run --frozen factor-bench prepare-run \
  --plan configs/factorization/release-v3.json --name w64-holdout-0-s0 \
  --output artifacts/runs/factorization/w64-holdout-0-s0
uv run --frozen factor-bench run --run-dir artifacts/runs/factorization/w64-holdout-0-s0
```

The second command starts a real experiment and needs the relevant device and
budget allocation. For the integer control, prepare the same preregistered
`--name` with `--arm selected_root_integer` into a distinct directory.

Classical controls use `sparse_factorization.classical.run_yafu` with a verified
fixed binary, separate work directories, nonzero uint64 seeds, one thread, and
a 300-second cap. Preserve stdout and temporary results. The pinned parser takes
one uint64 seed; an older docfile's two-word syntax does not match it.
Use a separate expression input file for noninteractive operation.

Build the pinned binary before running the classical control:

```bash
uv run --frozen python scripts/factorization/build_yafu.py \
  --cache .cache/yafu-downloads --output .cache/new-yafu-build --jobs 4
```

The helper targets Ubuntu 24.04 amd64 and needs a C/C++ toolchain, `make`,
and `dpkg-deb`. It downloads the locked YAFU source and dependency packages,
checks their hashes, extracts dependencies locally, and writes `build.json`;
it does not install system packages. Use a new build output directory.
The binary and libraries must match this build record before execution; a
rebuild on another host is not a reproduction of the historical timing.
This control also requires its own experimental allocation.

```bash
uv run --frozen factor-bench classical --plan configs/factorization/release-v3.json \
  --name w128-holdout-0-yafu-s0 --binary .cache/new-yafu-build/yafu \
  --output artifacts/classical/factorization/w128-holdout-0-yafu-s0
```

## Streaming completed evidence

A sealed chunk is emitted every 8192 completed events and for the final tail.
The GPU thread copies state and counters at a completed boundary; one background
thread compresses, checks, and atomically publishes the chunk. At most two chunks
are queued plus the next capture buffer. Hidden incomplete directories are
excluded; segmentation does not reset event counts, random streams, or budgets.

The example assumes SSH alias `gpu` and a repository beneath the remote
account home directory; replace the alias and path with your own values.
`pull --watch` overlaps transfer, integrity
checks, and independent CPU certification of immutable completed chunks.
Already verified chunks are reused. Keep connection details outside the repository.

```bash
uv run --frozen factor-bench pull \
  --remote gpu:probabilistic-computing-solvers/artifacts/runs/factorization/w128-holdout-0-s0/result/stream \
  --staging .cache/download-w128-0-0 --output artifacts/downloads/factorization/w128-0-0 \
  --receipts artifacts/audits/factorization/w128-0-0 --watch
```

Each chunk includes state, RNG, per-event cumulative counters, first-hit records,
and transition evidence; its manifest links the previous chunk. Checks cover
sizes/hashes, event coverage, boundary counters, and state validity. CPU
certification rebuilds the FP64 model independently. The terminal marker binds
the final certification summary. Preserve interrupted or failed chunks.
Recover source archives, environments, controller logs, and the overall
inventory as well; a stream alone does not replace those materials.

## Independent audit and origin replay

```bash
uv run --frozen factor-bench verify --stream artifacts/downloads/factorization/w128-0-0 \
  --receipts artifacts/audits/factorization/w128-0-0-fresh --fresh
uv run --frozen factor-bench replay --stream artifacts/downloads/factorization/w128-0-0 \
  --output artifacts/replays/factorization/w128-0-0 --max-seconds 21600
```

Existing receipts cover previously verified chunks; `--fresh` requires a new
receipt directory and rechecks all bytes. An origin replay starts from the
original random state, compares per-event counters and per-chunk state/RNG.
Chunk integrity checks alone are not an origin replay. `replay --watch` can
follow a recovered prefix using a separate GPU. On a single RTX 5090, run it
after the original experiment, serially. Use the matching frozen source and
RTX 5090 execution environment; verification does not overwrite original evidence.

Generate reports from all frozen denominators:

```bash
uv run --frozen factor-bench report --plan configs/factorization/release-v3.json \
  --runs artifacts/runs/factorization --output artifacts/reports/factorization/release-v3
```

Without results, the report explicitly lists unexecuted runs and creates no
performance plot. With measurements, CSV data determines PNG/SVG plots including
censoring. Generated summaries are paired English/Chinese Markdown files.
No PDF is generated. A small fixed panel does not support fitted general
complexity or TTS99.

## Software and infrastructure references

The classical control pins [YAFU](experiment-report.md#ref-project-bbuhrow-yafu),
[GMP](experiment-report.md#ref-gmp), [GMP-ECM](experiment-report.md#ref-ecm), and
[distribution packages](experiment-report.md#ref-ubuntu-packages).
Transfer and environment operations reference
[OpenSSH](experiment-report.md#ref-openssh),
[rsync](experiment-report.md#ref-rsync), and
[Vast.ai documentation](experiment-report.md#ref-vast-ai).
All are included in the report's bibliography.

## Check the included final data on CPU

```bash
uv run --frozen python scripts/factorization/verify_results.py --output artifacts/core-verification.json
uv run --frozen python scripts/factorization/plot_results.py --output artifacts/final-figures
```

The first command checks the core inventory, source bindings, accepted states and
integer controls without a GPU. Recorded replay receipts are checked, not re-executed.
The second rebuilds every table and figure from final core data. The stored executed
source hash differs from this distribution because packaging/reporting changed;
[provenance](../../provenance/factorization/release-v3.json) identifies the unchanged
numerical files and portable public plan. Configuration role seeds and numerical
settings are preserved. Use new output directories; no new solver experiments are needed.
