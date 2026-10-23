"""Reproduce one published case with explicit fresh/replay and warm-input semantics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from verify_core import digest, external_input, load_instance

from combinatorial_solvers.public_api import registry
from combinatorial_solvers.workflows.execution import execute_experiment, prepare_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--panel", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("fresh", "replay"), default="fresh")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--baseline-directory", type=Path)
    parser.add_argument("--allow-rebuilt-baseline", action="store_true")
    parser.add_argument("--warm-input", type=Path)
    parser.add_argument("--use-recorded-warm-input", action="store_true")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    directory = root / "data" / args.panel / "rtx5090"
    records = json.loads((directory / "calls.json").read_text())
    record = next(row for row in records if row["id"] == args.case)
    selected = record["configuration"]
    if args.mode == "replay" and args.panel != "tsp":
        if "origin_replay_configuration" not in record:
            raise ValueError("This case has no published fixed-work origin replay")
        selected = record["origin_replay_configuration"]
    if args.mode == "replay" and args.panel == "tsp" and record["trial_kind"] != "origin-replay":
        raise ValueError("Select the explicitly identified TSP origin-replay case")
    config = copy.deepcopy(selected)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    changes = []
    descriptions = json.loads((directory / "instances.json").read_text())
    if args.panel == "tsp":
        source = external_input(root, descriptions[0]["external"], args.fetch)
        config["problem"]["instance"] = str(source)
        config["prefix"] = config.pop("prefix_parameters")
        if config["solver"]["kind"] == "lkh":
            if args.baseline_directory is None:
                raise ValueError("LKH requires an explicitly supplied external baseline directory")
            binary = args.baseline_directory.resolve() / "LKH"
            actual = digest(binary)
            expected = config["solver"]["executable_sha256"]
            if actual != expected:
                if not args.allow_rebuilt_baseline:
                    raise ValueError("The LKH build differs; explicitly allow a rebuilt reference")
                changes.append({"field": "LKH executable", "recorded": expected, "actual": actual})
            config["prefix"]["lkh_executable"] = str(binary)
            config["solver"]["executable_sha256"] = actual
    else:
        description = next(row for row in descriptions if row["id"] == record["instance"])
        packet = load_instance(root, directory, description, args.fetch)
        if "problem" not in config:
            if args.panel == "maxcut-dense":
                from combinatorial_solvers.maxcut.dense.model import Instance

                model = Instance(int(packet["size"]), str(packet["family"]), packet["packed"])
                target = inputs / "graph.bin"
                model.write_mqlib(target)
                config["problem"] = {"instance_file": str(target)}
            elif args.panel == "maxcut-sparse":
                if "external" in description:
                    target = external_input(root, description["external"], args.fetch)
                else:
                    from combinatorial_solvers.maxcut.model import Instance

                    target = inputs / "graph.mc"
                    Instance(int(packet["size"]), packet["edges"], packet["weights"]).write_mqlib(
                        target
                    )
                config["problem"] = {"instance_file": str(target)}
            elif args.panel == "partition":
                if "external" in description:
                    target = external_input(root, description["external"], args.fetch)
                else:
                    from combinatorial_solvers.partition.model import Graph

                    target = inputs / "graph.graph"
                    Graph(int(packet["size"]), packet["edges"], packet["weights"]).write_metis(
                        target
                    )
                config["problem"] = {"graph": str(target)}
            elif args.panel == "ldpc":
                config["problem"] = {"instance": str(directory / description["file"])}
            else:
                raise ValueError("A generated problem definition is missing")
        if "baseline_parameters" in config:
            if args.baseline_directory is None:
                raise ValueError(
                    "This reference requires a legally obtained external baseline bundle"
                )
            config["baseline"] = config.pop("baseline_parameters")
            for binding in record["baseline_inputs"]:
                path = args.baseline_directory.resolve() / binding["filename"]
                actual = digest(path)
                if actual != binding["sha256"]:
                    if not args.allow_rebuilt_baseline:
                        raise ValueError(
                            "External build identity differs from the recorded reference"
                        )
                    changes.append(
                        {"field": binding["field"], "recorded": binding["sha256"], "actual": actual}
                    )
                config["baseline"][binding["field"].split(".", 1)[1]] = str(path)
        if record["warm_start"]:
            if args.warm_input is not None:
                warm = args.warm_input.resolve()
                warm_kind = "user-supplied-regenerated-input"
            elif args.mode == "replay" or args.use_recorded_warm_input:
                with np.load(
                    directory / record["witness_file"], allow_pickle=False
                ) as packet_state:
                    initial = packet_state["initial_bits"]
                if args.panel == "partition":
                    assert np.all(initial == initial[0])
                    warm = inputs / "recorded-initial-partition.txt"
                    np.savetxt(warm, initial[0], fmt="%d")
                elif args.panel == "ldpc":
                    n = int(packet["size"])
                    assert np.all(initial[..., :n] == initial[:, :1, :n])
                    warm = inputs / "recorded-initial-errors.npz"
                    np.savez_compressed(warm, errors=initial[:, 0, :n])
                else:
                    raise ValueError("Unknown recorded warm-input representation")
                warm_kind = "recorded-original-input-initialization-cost-not-remeasured"
            else:
                raise ValueError(
                    "Regenerate the stated warm-start origin and provide --warm-input, "
                    "or explicitly choose a conditional run with --use-recorded-warm-input"
                )
            key = "partition" if args.panel == "partition" else "errors"
            config["initial"] = {key: str(warm)}
        else:
            warm_kind = None
    config.setdefault("experiment", {})["output_dir"] = "unused"
    configuration = output / "configuration.json"
    configuration.write_text(json.dumps(config, indent=2) + "\n")
    metadata = {
        "panel": args.panel,
        "case": args.case,
        "mode": args.mode,
        "external_build_differences": changes,
        "warm_input": warm_kind if args.panel != "tsp" else None,
        "historical_initialization_cost_reused_as_new_measurement": False,
        "published_measurements_overwritten": False,
        "note": (
            "Wall-clock runs can end at different work counts; "
            "only declared origin replays fix work"
        ),
    }
    (output / "reproduction.json").write_text(json.dumps(metadata, indent=2) + "\n")
    name = "maxcut" if args.panel == "maxcut-sparse" else args.panel
    run = prepare_experiment(
        registry().get(name),
        configuration,
        repository=root,
        run_id="reproduce-" + args.case,
        output_root=output / "runs",
    )
    raise SystemExit(execute_experiment(run))


if __name__ == "__main__":
    main()
