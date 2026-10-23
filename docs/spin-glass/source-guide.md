# Cubic spin-glass source and data guide

English | [简体中文](source-guide.zh-CN.md)

The package is independent of `sparse_factorization`. No private repository,
cloud account or downloaded paper is needed to use the public code and data.

| Path | Responsibility |
|---|---|
| `src/cubic_spin_glass/cli.py` | Single-instance CLI, source/environment capture, raw outputs and witness export |
| `model.py`, `core/` inside that package | Cubic geometry, explicit couplings and sparse Ising data structures |
| `solver.py` | Heat-bath/Metropolis updates, PT, ICM, timing and observation order |
| `population.py` | Population schedules, restarts, multinomial/systematic resampling and diagnostics |
| `triton_*.py` | CUDA field/update kernels and integer energy reductions |
| `certificates.py` | Standard-library model validation and integer witness checking |
| `scripts/spin_glass/run_cell.py` | Explicit single-cell reproduction; listing has no solver side effects |
| `scripts/spin_glass/analyze_results.py` | Instance-level means, paired bootstrap and retry sensitivity |
| `scripts/spin_glass/verify_results.py` | Public-data integrity, source binding, coverage, statistics and witness verification |

The numerical runtime retains the measured source bytes. The public wrapper and
analysis tools were added for reproduction; they do not change the update kernel.
The [provenance file](../../provenance/spin-glass/release-v1.json) binds those runtime
files and the frozen panel. The shared dependency lock is the owning environment.

## Public evidence layout

Under `data/spin-glass/s23/`:

- `instances/`: 42 random coupling files and one correctness fixture.
- `calls.json`: all 272 physical attempts, with a unique `execution_order`, a
  `cell` identifier and a `logical_cell` identifier. The one failed call and its
  retry have different physical IDs but the same logical cohort. Successful
  rows contain all group checkpoint energies, saved-array fingerprints, RNG,
  sampling/preparation/export/full-call time and allocated GPU memory.
- `summary.json`: 216-confirmation statistics and the eleven-instance 32³
  sensitivity panel. Group/seed outcomes are averaged within each coupling.
- `per-call.csv`: the 216 successful confirmation calls, including the recovered
  one; this convenience table alone does not include the failed-attempt cost.
- `best-observed.json`, `witnesses/`: 108 model-bound best observed states.
- `random-streams.json`: frozen role keys; matching PT/SA keys are intentional.
- `verification-receipt.json`: recorded full-array audit/replay coverage and its limits.
- `manifest.json`: public file sizes and SHA-256 values.

The caller's failed full-return timeout is distinct from a solver's normal
sampling stop. A successful quality row may therefore have `solver_stop_status:
"timeout"` while its evidence status is `verified`. Unknown failed sampling
progress remains JSON `null`, never a fabricated zero.

Full raw trajectories, execution logs and private archive paths are not part of
this core release. Recorded fingerprints support consistency checks and future
fixed-work reproduction; integer witnesses support independent energy checks.
Neither establishes global optimality. See the [report](experiment-report.md)
for source references, hardware, denominators and all comparison restrictions.
