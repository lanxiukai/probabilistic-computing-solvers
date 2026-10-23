"""Render a certified cold-to-best schedule pair from one frozen p-bit replica."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from combinatorial_solvers.core.evidence import verify_manifest
from combinatorial_solvers.jssp.model import read_instance


def contained(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Release paths must be nonempty relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Release paths must stay inside their declared root")
    return path


def checked_file(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise AssertionError("Released file digest mismatch: " + path.name)


def frozen_call(cell):
    verify_manifest(cell)
    prepared = json.loads((cell / "prepared.json").read_text())
    source = next(x for x in prepared["inputs"] if x["field"] == "problem.instance_file")
    instance = read_instance(cell / source["path"])
    summary = json.loads((cell / "result/summary.json").read_text())
    if not summary["pbit_heatbath_participation"]:
        raise ValueError("This figure requires a p-bit heat-bath result")
    with np.load(cell / "result/states.npz", allow_pickle=False) as archive:
        state = {
            key: archive[key]
            for key in (
                "initial_sequence",
                "initial_cost",
                "quality_best_sequence",
                "quality_best_cost",
            )
        }
    return instance, state, summary, prepared["run_id"]


def released_call(calls_file, call_id, repository):
    records = json.loads(calls_file.read_text())
    if not isinstance(records, list):
        raise ValueError("Released calls must be a list")
    selected = [row for row in records if row["id"] == call_id]
    if len(selected) != 1:
        raise ValueError("Choose exactly one existing public call ID")
    record = selected[0]
    config = record["configuration"]
    if record.get("configuration_file"):
        executable = json.loads(contained(repository, record["configuration_file"]).read_text())
        if executable != config:
            raise AssertionError("Executable configuration differs from the recorded configuration")
    source = [row for row in record["benchmark_inputs"] if row["field"] == "problem.instance_file"]
    if len(source) != 1:
        raise ValueError("One original benchmark binding is required")
    source = source[0]
    instance_path = contained(repository, source["path"])
    if contained(repository, config["problem"]["instance_file"]) != instance_path:
        raise AssertionError("Configured benchmark differs from the hashed input")
    checked_file(instance_path, source["sha256"])
    witness = contained(calls_file.parent, record["witness_file"])
    checked_file(witness, record["witness_sha256"])
    if config.get("method", "native") != "native" or config["solver"]["method"] != "heatbath":
        raise ValueError("This figure requires a p-bit heat-bath result")
    summary = {**record["metrics"], "settings": config["solver"]}
    with np.load(witness, allow_pickle=False) as archive:
        state = {
            key: archive[key]
            for key in (
                "initial_sequence",
                "initial_cost",
                "quality_best_sequence",
                "quality_best_cost",
            )
        }
    return read_instance(instance_path), state, summary, call_id


def figure_data(instance, state, summary, call_id):
    if instance.jobs > 20 or instance.size > 300:
        raise ValueError("Choose a readable example with at most 20 jobs and 300 operations")
    if not summary["pbit_heatbath_participation"]:
        raise ValueError("This figure requires a p-bit heat-bath result")
    schedule_decoder = summary["settings"].get("schedule_decoder", "append")
    best_cost = state["quality_best_cost"]
    replica = int(np.argmin(best_cost))
    initial = state["initial_sequence"][replica]
    best = state["quality_best_sequence"][replica]
    initial_cost = int(state["initial_cost"][replica])
    final_cost = int(best_cost[replica])
    schedules = [instance.decode(s, schedule_decoder=schedule_decoder) for s in (initial, best)]
    for schedule, cost in zip(schedules, (initial_cost, final_cost), strict=True):
        assert instance.certificate(schedule) == {"feasible": True, "makespan": cost}
    if final_cost > initial_cost:
        raise AssertionError("Eligible best cost is worse than the same replica's cold prefix")
    if final_cost != summary["best_feasible_cost"]:
        raise AssertionError("Illustrated result differs from the published eligible best cost")
    return {
        "run_id": call_id,
        "replica": replica,
        "schedule_decoder": schedule_decoder,
        "replica_selection": "lowest_eligible_cost_then_lowest_replica_index",
        "initial_makespan": initial_cost,
        "best_makespan": final_cost,
        "initial_starts": schedules[0].tolist(),
        "best_starts": schedules[1].tolist(),
        "scope": "An illustrative same-replica improvement, not a classical-baseline comparison",
    }


def render(instance, data, output):
    matplotlib.use("Agg", force=True)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab20")
    initial_cost, final_cost = data["initial_makespan"], data["best_makespan"]
    schedules = [np.array(data[key]) for key in ("initial_starts", "best_starts")]
    labels = [f"Cold prefix: makespan {initial_cost}", f"After heat bath: makespan {final_cost}"]
    for axis, starts, title in zip(axes, schedules, labels, strict=True):
        for job in range(instance.jobs):
            color = colors(job % 20)
            for operation in range(instance.machine_count):
                machine = int(instance.machines[job, operation])
                duration = int(instance.durations[job, operation])
                start = int(starts[job, operation])
                axis.barh(machine, duration, left=start, color=color, edgecolor="white", height=0.8)
                axis.text(
                    start + duration / 2,
                    machine,
                    str(job + 1),
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        axis.set_yticks(
            range(instance.machine_count), [f"M{i + 1}" for i in range(instance.machine_count)]
        )
        axis.set_title(title, loc="left")
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    axes[-1].set_xlabel("Processing time")
    legend = [Patch(facecolor=colors(j % 20), label=f"Job {j + 1}") for j in range(instance.jobs)]
    axes[0].legend(handles=legend, ncol=min(10, instance.jobs), loc="upper right", fontsize=8)
    figure.suptitle(
        f"Same priority-sequence replica ({data['replica']}); {data['schedule_decoder']} decoder"
    )
    figure.savefig(output / "schedule.svg")
    figure.savefig(output / "schedule.png", dpi=220)
    plt.close(figure)
    (output / "figure-data.json").write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cell", type=Path)
    source.add_argument("--calls", type=Path)
    parser.add_argument("--id")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.calls is not None:
        if args.id is None:
            parser.error("--calls requires --id")
        instance, state, summary, call_id = released_call(
            args.calls.resolve(), args.id, args.repository.resolve()
        )
    else:
        if args.id is not None:
            parser.error("--id applies only to --calls")
        instance, state, summary, call_id = frozen_call(args.cell)
    render(instance, figure_data(instance, state, summary, call_id), args.output)


if __name__ == "__main__":
    main()
