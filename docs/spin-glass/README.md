# Cubic spin-glass solvers

English | [简体中文](README.zh-CN.md)

GPU implementations of heat-bath parallel tempering with isoenergetic cluster
moves (PT+ICM), cyclic Metropolis simulated annealing (SA), and population
annealing (PA), with independently checkable integer witnesses. The package is
`cubic_spin_glass` in this repository's single uv environment, under [MIT](../../LICENSE).

The accepted RTX 5090 evaluation covers complete open cubic, zero-field,
nearest-neighbor ±1 instances at 16³, 24³ and 32³. The largest problem has
32,768 logical spins; replicas and particles are additional search resources.
Across 12 held-out coupling instances per size, the lowest mean observed energy
at 60 seconds came from PT+ICM, systematic PA and cyclic SA, respectively.
There is no general p-bit advantage or large-instance ground-state claim.

| Read | Contents |
|---|---|
| [Experiment report](experiment-report.md) | Methods, frozen protocol, measured results, retry accounting, comparisons and full bibliography |
| [Reproduction](reproduction.md) | Installation fixture, public-data verification, new-instance input and explicit GPU reruns |
| [Source guide](source-guide.md) | Runtime modules, data schema and version binding |

```bash
uv sync --frozen
uv run --frozen spin-glass-bench \
  --config configs/spin-glass/example-cpu.json --output artifacts/spin-glass-example
uv run --frozen python scripts/spin_glass/verify_results.py \
  --output artifacts/spin-glass-verification.json
```

The `spin-glass-bench` command runs an eight-spin CPU installation fixture. The verifier
checks the published evidence without GPU execution. Public core data includes
all 272 attempts, 216 successful confirmation cohorts, 108 portable witnesses,
frozen couplings/configurations, array fingerprints and RNG records. Full raw
trajectories and operational archives are retained separately; the verifier
does not reconstruct those missing bytes from fingerprints.

Unknown optimal energies remain unknown. The release is GPU simulation, not a
p-bit/FPGA hardware measurement, an equilibrium-sampling result, or a replicated
external-baseline advantage. Optimized external classical and author p-bit code
have not been run on this panel; see the report's comparison boundaries.
