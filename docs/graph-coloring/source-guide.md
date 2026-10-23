# Graph-coloring source and reproduction guide

English | [简体中文](source-guide.zh-CN.md)

The implementation solves a fixed-palette conflict objective on original DIMACS graphs, with optional vertex-specific allowed-color lists. The [experiment report](experiment-report.md) explains the binary representation, p-bit participation, classical support, measured quality and scientific limits.

## Code map

| File | Responsibility |
|---|---|
| `model.py` | Simple-graph input, original-edge certificates, color lists and independently checked clique constraints |
| `gpu.py` | Persistent binary states, color/count caches and exact conditional blocks |
| `solver.py` | Initialization, temperature schedules, bounded sampling, observations and recovery evidence |
| `tempering.py` | Adjacent temperature-label exchanges within independent groups |
| `baseline.py` | Independent single-thread TabuCol-style comparison and its certificates |
| `runner.py` | Configuration execution and complete result certification |
| `scripts/graph_coloring/verify_release.py` | CPU-only verification of public witnesses, input hashes, deadlines and declared counts |

The topic modules live under `src/combinatorial_solvers/graph_coloring/`. `verify_release.py` imports neither the solver nor CUDA. It independently parses original edges, checks lists and clique certificates, and recomputes conflicts and resource counts from saved states.

## Run the example

Run these commands from the repository root in its locked environment. The sampling command requires CUDA; fetching inputs and clique construction are CPU operations.

```bash
uv run --frozen python scripts/graph_coloring/fetch_benchmarks.py --instance fpsol2.i.1
uv run --frozen python scripts/graph_coloring/prepare_clique.py --instance data/graph-coloring/dimacs/fpsol2.i.1.col --palette 65 --output data/graph-coloring/demo/clique.json
uv run --frozen python -m combinatorial_solvers.public_api --problem graph-coloring --config configs/graph-coloring/example.json --run-id graph-coloring-example --output-root artifacts/examples
```

The fetcher checks the exact byte count and SHA-256 recorded in `configs/graph-coloring/benchmarks.json`. Original benchmark files are fetched from their cited source rather than redistributed here. The clique script constructs its certificate from graph edges and does not read a known coloring. It refuses to overwrite an existing certificate; choose a new output path for another preparation.

The example materializes 32,240 one-hot bits per chain on the full `fpsol2.i.1` graph. Its 227 isolated vertices mean that only 17,485 of these bits belong to conflict factors. Fixing a verified 65-clique leaves 28,015 freely sampled bits, including 13,260 attached to conflict factors. Clique construction is classical support; a zero-conflict initial state would receive no new p-bit hit.

The example seed is separate from the reported trials. A completed example is not automatically added to the published results, and its timing need not reproduce the measured hardware values.

## Configure the model

`problem.colors` fixes the palette. `problem.allowed_file` optionally loads an $N\times K$ binary NumPy mask with at least one allowed color per vertex. A clique symmetry file and a list file cannot be combined. A global color renaming preserves ordinary proper coloring, but does not generally preserve restricted lists.

`solver.width` chooses a one-, two-, three- or four-vertex conditional block; three/four-vertex blocks require at most eight colors. `method="heatbath"` selects the p-bit heat bath. `metropolis` and `plateau` are separately labeled classical controls. Finite-temperature heat baths use `beta_start`, `beta_end` and `cycle_steps`; equal endpoints give a fixed temperature.

For temperature exchange, an even `tempering_ladder_size` must divide `replicas`. Each ladder is one independent group. `exchange_every_chunks` controls the exchange interval, so changing `chunk_steps` also changes the interval unless adjusted. Changing chunk shape can change the random-number layout; it is not a promise of an identical trajectory.

`cache_boltzmann_weights=true` caches FP64 weights for bounded integer local-energy differences. `reuse_preparation=true` reuses allowed-color choices and clones an independent state for warmup; it does not reuse a previous run's solution. Both default to true after the audited implementation pilot. `warmup_steps=null` warms a full declared chunk. The five pilot pairs preserve all non-time state arrays on their tested inputs; this observation is not a universal trajectory or speed guarantee.

`max_seconds` bounds search wall time, including in-search checkpoints and curve writes. Preparation is recorded separately; complete runner time also includes final certification and output writing. `quality_max_step` is an additional eligibility fence for fixed-work replay. A completed chunk observed after either boundary stays in the evidence and is excluded from deadline-eligible quality. Increasing replicas changes resource consumption as well as the number of search groups.

`snapshot_every_chunks=null` disables redundant buffered periodic color snapshots while retaining initial/final states, every chunk observation and independent full checkpoints. `checkpoint_seconds` controls the checkpoint wall-time interval. Changing these settings changes measured overhead and must be frozen before a timing comparison.

## Read and verify results

Each run saves `summary.json`, original-color witnesses in `states.npz`, initialization and RNG states, flushed observation curves and recovery checkpoints. `best_feasible_cost` is the historical field name for the minimum eligible **conflict count**; positive values still contain conflicting edges. `returned_best_cost` additionally includes work observed after the quality boundary. Zero conflict certifies the fixed-palette ground state; the chromatic number needs a separately checked matching lower bound.

The `visited` mask marks every allowed one-hot coordinate of a vertex considered by an update. The `visited_*_bits_min` metrics take the minimum count across chains; they do not count actual bit flips. `changed` counts block updates that changed their joint assignment. Counts described as interacting use adjacency to original conflict factors and do not count the free–free couplings left after clique fixing.

Each public ledger binds its curated results to configurations, original-input hashes, witnesses and metrics. `rtx5090` contains final G3 confirmation, `development-g2b` contains 13 development calls and two work replays, and `implementation-g2a` contains eight fixed-work implementation calls. Their purposes and denominators remain separate. After fetching the catalog inputs into the declared paths, verify each ledger with:

```bash
uv run --frozen python scripts/graph_coloring/verify_release.py --calls data/graph-coloring/rtx5090/calls.json --repository .
uv run --frozen python scripts/graph_coloring/verify_release.py --calls data/graph-coloring/development-g2b/calls.json --repository .
uv run --frozen python scripts/graph_coloring/verify_release.py --calls data/graph-coloring/implementation-g2a/calls.json --repository .
```

Public curves retain per-replica best improvements, the first/last observations and eligibility boundaries. `curve_indices` records their original observation indices; the verifier checks these coordinates and the declared selection rule. They preserve the corresponding best-quality observations without reconstructing omitted current-state fluctuations. Full raw curves, temperature-exchange records and checkpoints remain in the separate experiment archive. Portable witness verification establishes consistency of the supplied evidence; it does not authenticate wall-clock execution or create new solver measurements.

To build the owned CPU comparison separately:

```bash
uv run --frozen python scripts/graph_coloring/build_reference.py --output data/graph-coloring/tabu-reference
```

The resulting `reference`, copied C++ source, `build.json` and `build.log` are explicit baseline inputs. This comparison does not participate in p-bit heat-bath results and is not an upstream HEAD reproduction. See the report for the DIMACS, COLOR, TabuCol and temperature-exchange sources.

The public source passed cloud software checks and four fixed-work replays comparing every non-time array. Pinned benchmark fetching and all three core ledgers passed independent verification. Baseline input hashes identify the measured build; rebuilding can change compiler/path/host-dependent products while the original C++ source identity remains fixed.
