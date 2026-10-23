"""Run isolated, certified examples from the curated solver source distribution."""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib import import_module
from pathlib import Path

from combinatorial_solvers.workflows.execution import execute_experiment, prepare_experiment
from combinatorial_solvers.workflows.registry import SolverRegistry, SolverSpec


def registry():
    specifications = []
    for name, module in (
        ("mwis", "mwis"),
        ("maxsat", "maxsat"),
        ("maxcut", "maxcut"),
        ("maxcut-dense", "maxcut.dense"),
        ("labs", "labs"),
        ("partition", "partition"),
        ("ldpc", "ldpc"),
        ("set-cover", "set_cover"),
        ("jssp", "jssp"),
        ("covering-array", "covering_array"),
        ("graph-coloring", "graph_coloring"),
        ("qap", "qap"),
    ):
        specification = import_module(f"combinatorial_solvers.{module}.runner").SPECIFICATION
        specifications.append(replace(specification, name=name))
    specifications.append(
        SolverSpec(
            "tsp",
            "combinatorial_solvers.tsp.runner:run_tsp_config",
            "P-bit hybrid workflows with exact integer tour certificates",
            input_fields=(
                "problem.instance",
                "prefix.tours_file",
                "prefix.lkh_executable",
                "prefix.eax_executable",
            ),
        )
    )
    return SolverRegistry(specifications)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--problem")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    selected = registry()
    if args.list:
        print("\n".join(selected.names))
        return
    if not all((args.problem, args.config, args.run_id)):
        parser.error("--problem, --config and --run-id are required")
    directory = prepare_experiment(
        selected.get(args.problem),
        args.config,
        repository=Path.cwd(),
        run_id=args.run_id,
        output_root=args.output_root,
    )
    raise SystemExit(execute_experiment(directory))


if __name__ == "__main__":
    main()
