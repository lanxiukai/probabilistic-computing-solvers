"""Run one frozen candidate case using explicitly supplied external inputs."""

import argparse
import json
from pathlib import Path

from candidate_inputs import resolve_case

from combinatorial_solvers.workflows.execution import execute_experiment, prepare_experiment
from combinatorial_solvers.workflows.registry import SolverSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--instances-directory", type=Path, required=True)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    case = next(row for row in plan["cases"] if row["id"] == args.case)
    config = resolve_case(
        case, plan["instances"], args.instances_directory, args.baseline_directory
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "configuration.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    specification = SolverSpec(
        "tsp",
        "combinatorial_solvers.tsp.runner:run_tsp_config",
        "Sparse candidate network and native TSP references",
        input_fields=("problem.instance", "prefix.lkh_executable", "prefix.eax_executable"),
    )
    run = prepare_experiment(
        specification, path, repository=Path.cwd(), run_id=case["id"], output_root=output / "runs"
    )
    raise SystemExit(execute_experiment(run))


if __name__ == "__main__":
    main()
