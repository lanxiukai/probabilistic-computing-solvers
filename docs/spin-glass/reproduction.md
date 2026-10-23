# Reproducing and verifying cubic spin-glass results

English | [简体中文](reproduction.zh-CN.md)

Use Linux x86-64 and the repository's single locked environment. The accepted
GPU data used Python 3.14.6 and Torch 2.13.0+cu130 on RTX 5090; see the
[report](experiment-report.md) for the complete hardware and evaluation scope.
Installation and evidence checks below are CPU software checks, not new benchmark trials.

## Install and check the portable entry

```bash
uv sync --frozen
uv run --frozen spin-glass-bench --help
uv run --frozen spin-glass-bench \
  --config configs/spin-glass/example-cpu.json --output artifacts/spin-glass-example
uv run --frozen python src/cubic_spin_glass/certificates.py \
  artifacts/spin-glass-example/instance.json artifacts/spin-glass-example/witnesses.json
```

The eight-spin ferromagnet checks installation. Existing output directories are
rejected. Each solver run preserves resolved inputs/configuration, source,
`pyproject.toml`, `uv.lock`, runtime versions, RNG, raw arrays, observations,
integer witnesses and a file manifest. Failed runs retain partial files.

## Verify published evidence without a GPU

```bash
uv run --frozen python scripts/spin_glass/verify_results.py \
  --output artifacts/spin-glass-verification.json
uv run --frozen python scripts/spin_glass/analyze_results.py \
  --output artifacts/spin-glass-analysis.json
```

Verification checks runtime fingerprints, every published data file, the frozen
271-cell protocol, all 272 attempts including the failed call, tuning/confirmation
separation, recorded replay fingerprints/RNG, recomputed paired statistics and
108 integer witnesses. Full raw trajectories are not distributed; this command
does not claim to independently rerun their complete historical byte audit.
Output files must be new. The standalone certificate checker needs only Python's
standard library; analysis and full public verification use the locked NumPy/runtime environment.

For one exported 32³ witness:

```bash
uv run --frozen python src/cubic_spin_glass/certificates.py \
  data/spin-glass/s23/instances/l32-i2.json \
  data/spin-glass/s23/witnesses/l32-i2-sa.json
```

## Supply another instance

An instance JSON has exactly `length`, `boundary: "open"`, and
`bonds: [[i, j, J], ...]`. Coordinates are `i = x + L*y + L*L*z`.
Every distinct undirected nearest-neighbor edge must occur once, with integer
`J` equal to −1 or +1, zero field, and `2 <= length <= 32`.
A configuration has exactly `instance_file` (relative to that configuration)
and `solver`. Start from the CPU example. For GPU work set `device: "cuda"`
and `execution: "lattice-v5"`, and explicitly choose the method, seed,
population, sweep horizon and `max_seconds` before running.

The checker rejects missing/repeated/long edges and floating-point or Boolean
values masquerading as integers. Input coverage does not imply a performance
or optimality guarantee for an arbitrary coupling realization.

## Explicit GPU reruns

Listing the frozen panel does not execute a solver:

```bash
uv run --frozen python scripts/spin_glass/run_cell.py --list
```

The following command executes one new 60-second sampling call, with preparation,
output and certification in addition to that budget. Allocate an available GPU
and appropriate runtime before choosing to run it:

```bash
uv run --frozen python scripts/spin_glass/run_cell.py \
  --cell confirm-l32-i2-s1-sa --output artifacts/spin-glass-new-confirmation
```

Use `--cell replay-sa` for the fixed-work 32³ SA replay, `replay-pt` for PT+ICM,
or `replay-pa` for PA. Their sweep horizons are 38475, 14464 and 14652,
respectively, with 220-second sampling guards. A rerun is not another independent
coupling instance and must not be silently added to the published denominator.
Recorded state/RNG fingerprints are in `data/spin-glass/s23/calls.json`.
Same-device fixed-work replay is distinct from wall-clock-cutoff reproduction;
timestamps, complete-call time and cross-device trajectories need not match.

The helper launches only the selected cell; it does not automatically rerun the
entire panel, retry failures, provision a device or operate a cloud account.
The helper verifies and restores the measured runtime and input into an isolated
directory, then runs a new process with the original 240-second full-call limit
and eight-second termination grace. Partial output is retained on failure.
`max_seconds` separately bounds sampling, not compilation/export.

To restore a run, recover its saved `source/cubic_spin_glass` package and captured
environment metadata in a separate checkout or working directory. Run the saved
`config.json` into a new output path, using `--environment-project` for the owning
environment directory when necessary. Never overwrite a completed evidence directory.

## Software regression

```bash
CUDA_VISIBLE_DEVICES='' uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

CUDA-specific tests are skipped when the device is hidden. These checks do not
replace the recorded RTX 5090 experiments. The experiment report contains the
complete research and software bibliography and the limits of external comparisons.
