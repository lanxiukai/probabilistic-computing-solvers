# Job-shop source and reproduction guide

[简体中文](source-guide.zh-CN.md) · [Overview](README.md) · [Report](report.md)

This guide describes the software-gated implementation, accepted independent confirmation and portable evidence contract. Complete original evidence has passed cloud and local audits; the assembled public release passed 590 software tests, core-witness verification and three exact fixed-work source replays.

## Source responsibilities

| File | Responsibility |
|---|---|
| `src/combinatorial_solvers/jssp/model.py` | Input validation, independent CPU decoding, and original precedence/capacity certificates |
| `src/combinatorial_solvers/jssp/gpu.py` | Explicit one-hot states and exact conditional permutation updates using integer makespans |
| `src/combinatorial_solvers/jssp/active_gpu.py` | Earliest-gap insertion energy for two-position blocks |
| `src/combinatorial_solvers/jssp/solver.py` | Bounded search, initialization, incumbent and quality fences, RNG capture and recovery checkpoints |
| `src/combinatorial_solvers/jssp/baseline.py` | Independent single-worker CP-SAT scheduling model and incumbent witnesses |
| `src/combinatorial_solvers/jssp/runner.py` | Original-constraint certification, per-replica first hits and result artifacts |
| `scripts/jssp/experiment.py` | Isolated source/configuration/input preparation, execution and manifest verification |
| `scripts/jssp/fetch.py` | Six pinned benchmark downloads and original-byte hash verification |
| `scripts/jssp/verify_release.py` | Portable witness and published-claim verification without running an optimizer |
| `scripts/jssp/plot_schedule.py` | Same-replica cold-prefix and eligible-best schedule illustration |

The problem package uses the shared evidence and isolated-execution framework. It does not import another problem's solver. The public package prefix is `combinatorial_solvers`; no private repository checkout is required.

## Representation and exactness

For $L=JM$ operations, each state contains exactly $M$ occurrences of each job label. `bits[position, job]` is its explicitly stored one-hot representation. The integer sequence, current cost, best cost and participation mask are caches checked against the original scheduling problem.

The shared metadata field `physical_update_pbits_per_model` counts the modeled binary entries assigned to the heat-bath workflow; it is zero for classical controls. Its name does not indicate physical hardware or full participation. The `visited` mask and `visited_assignment_bits_min` separately describe entries actually considered in update blocks, which is also different from the number of bit flips.

Participation covers the returned execution, including any retained deadline-crossing chunk; deadline-eligible solution quality is a separate prefix. A conservative lower bound on eligible participation is the final count minus `width**2 * (completed_steps - quality_max_step)`, clipped at zero, because one step considers at most `width**2` assignment entries per replica.

`schedule_decoder` selects the mathematical energy map. `append` places an operation after the last already-scheduled operation on its machine. `active` inserts it into the earliest feasible machine gap. Both preserve job precedence and machine capacity. The earliest-gap decoder is a classical support component; its cold-prefix gain must be separated from improvements caused by subsequent p-bit updates. The current active kernel accepts two-position blocks; append also accepts four-position blocks.

All final confirmation calls use `append`. The alternative active implementation is included as an imported package component, without a final-confirmation performance claim.

`decoder` selects an exact register-lookup implementation (`reduce` or `gather`) and is distinct from `schedule_decoder`. The development comparison retained `reduce` because the tested `gather` implementation was slower despite equal non-timing arrays.

For a width-$b$ block, the kernel enumerates $b!$ permutations and samples a heat-bath candidate in FP64 using the original integer makespan. Duplicate labels have equal candidate multiplicity. Reference optima are used for reporting, not as an energy term, initialization hint or CP-SAT hint. `metropolis` and `plateau` are matched CUDA controls. Metropolis still evaluates the full candidate set here, so it is an update-rule control rather than an optimized standalone implementation.

## Quality, clocks and evidence

`best_sequence` and `best_cost` contain the best accepted state through the last completed chunk. `quality_best_sequence` and `quality_best_cost` contain only states observed within the declared search deadline and, for a fixed-work replay, within `quality_max_step`. A late chunk remains in the evidence and is excluded from quality. The runner separately certifies current, returned-best and eligible-best schedules.

Every completed chunk supplies one observation. The saved best value includes all accepted transitions inside that chunk, but the first-hit time has chunk resolution. `states.npz:first_reference_hit_seconds` is a per-replica array with `NaN` for misses. The summary's identically named field is the earliest ensemble hit, or `null` if all replicas miss. A cold-prefix hit has time zero and is not a p-bit discovery.

The frozen confirmation sets `checkpoint_interval_seconds=2.0`: at the next completed chunk after the interval, it saves full current/best/eligible state and the committed RNG boundary. The interval starts at search start or the previous checkpoint's completion. This recovery cadence is separate from coarse trajectory snapshots; all per-chunk curves remain. Periodic checkpoint compression and observation costs remain inside search time. Initial capture belongs to preparation, and final checkpoint/output work remains in complete runner time. A transition interrupted before commitment retains its pending random tensor and marks its device-state boundary as uncertain.

CP-SAT chooses quality from incumbent callbacks at or before its time limit. `quality_best_schedule` is that eligible witness; `best_schedule` and `returned_best_cost` preserve the final solver return even if it is late. `returned_optimality_proved` and `lower_bound` describe the final solver response. `optimality_proved` additionally requires the proof response to finish within the search deadline. The original witnesses do not independently attest machine wall-clock accuracy.

Preparation includes initialization and warmup. Search includes declared observation/checkpoint work. Certification and complete runner time are separate; outer source restoration and process startup add another cost. GPU and single-worker CPU timings remain separate resource measurements.

## Portable release contract

The intended release entry is `data/jssp/rtx5090/calls.json`, a list of calls. Each record contains an inline `configuration`, an identical executable `configuration_file`, numeric `metrics`, benchmark digests, a core witness digest and resource/role fields. Configuration and benchmark paths are relative to the public repository root; `witness_file` is relative to the directory containing `calls.json`.

Core witnesses preserve initial/current/returned-best/eligible-best sequences and costs, original schedules, one-hot state, participation, RNG endpoints and per-replica first hits. The public trace may retain only observations where any replica's best cost improves, plus the first/last observations and eligibility boundaries. `curve_indices` records the original positions and `curve_curation` declares the rule and counts. The full original trace is independently audited and retained separately. Current-state fluctuations omitted by this curation cannot be reconstructed from the public trace.

The verifier recomputes original schedules and constraints, initialization from the recorded seed, quality eligibility, per-replica hits, binary/participation counts and published values. A fixed-work replay must match non-timing arrays and contributes no independent trial. Successful verification establishes witness and claim consistency; it does not rerun the stochastic search or independently measure its elapsed time.

## Reproduction entry points

Use the public repository's frozen uv environment. Download the pinned benchmark bytes into a new destination, then verify the released core data:

```bash
uv run --frozen python scripts/jssp/fetch.py --output data/jssp/instances
uv run --frozen python scripts/jssp/verify_release.py --calls data/jssp/rtx5090/calls.json
```

Rerun the first frozen `ft06` heat-bath configuration in a new isolated directory:

```bash
uv run --frozen python scripts/jssp/experiment.py prepare \
  --config data/jssp/rtx5090/configs/j4-ft06-heatbath-s1.json \
  --run-id jssp-ft06-reproduction
uv run --frozen python scripts/jssp/experiment.py run \
  --run-dir artifacts/runs/jssp/jssp-ft06-reproduction
uv run --frozen python scripts/jssp/experiment.py verify \
  --run-dir artifacts/runs/jssp/jssp-ft06-reproduction
```

Preparation snapshots the actual source and inputs; an existing completed directory is not reused. A wall-clock rerun can complete a different number of updates. To reproduce fixed work, select `data/jssp/rtx5090/configs/replay-j4-ft06-heatbath-s1.json` and a fresh run ID. It preserves the original completed-step and eligible-quality boundaries. Acceptance requires non-timing equality on the measured hardware and locked runtime; it is not a cross-device timing guarantee. Repeating a released seed adds no independent confirmation trial.

Redraw the same-replica illustration directly from the released witness:

```bash
uv run --frozen python scripts/jssp/plot_schedule.py \
  --calls data/jssp/rtx5090/calls.json --id j4-ft06-heatbath-s1 \
  --output artifacts/runs/jssp/ft06-figure
```

The renderer selects the lowest eligible cost, breaking ties by replica index, and shows that same replica before and after the heat bath. The SVG/PNG and numeric `figure-data.json` are an illustration, not a comparison with a classical solver.

The final report contains the complete linked bibliography and measured software/hardware versions. These commands and final public paths were validated in the assembled public release on the RTX 5090 host.
