"""Reproduce candidate-network metrics and a figure from the accepted core data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def summarize(core: Path, output: Path) -> dict:
    manifest = json.loads((core / "manifest.json").read_text())
    for name, identity in manifest["files"].items():
        data = (core / name).read_bytes()
        if len(data) != identity["bytes"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
            raise ValueError("A core data member changed")
    data = json.loads((core / "calls.json").read_text())
    natural = [row for row in data["calls"] if row["trial_kind"] == "natural"]
    replays = [row for row in data["calls"] if row["trial_kind"] == "origin-replay"]
    if len(natural) != data["natural_calls"] or len(replays) != data["origin_replays"]:
        raise ValueError("Core denominators differ from the recorded calls")
    output.mkdir(parents=True, exist_ok=False)
    rows, pairs, raw = [], [], []
    for description in data["instances"]:
        group = {row["method"]: row for row in natural if row["instance"] == description["name"]}
        if set(group) != {"pbit", "random", "assignment", "ils"}:
            raise ValueError("Expected the p-bit workflow and all three attribution controls")
        pbit = group["pbit"]
        summary = pbit["summary"]
        events = json.loads((core / pbit["events_file"]).read_text())
        after_prefix = [
            row["length"] for row in events if row["stage"] != "prefix" and "tour" in row
        ]
        rows.append(
            {
                "instance": description["name"],
                "cities": description["dimension"],
                "optimum": description["optimum"],
                "seed": pbit["configuration"]["solver"]["seed"],
                "nodes": summary["model"]["binary_nodes_per_replica"],
                "prefix_length": summary["prefix_length"],
                "best_post_prefix_length": min(after_prefix) if after_prefix else None,
                "best_length": summary["best_length"],
                "gap_percent": 100 * (summary["best_length"] / description["optimum"] - 1),
                "complete_seconds": pbit["prepare_plus_exit_seconds"],
                "first_certificate_seconds": pbit["prepare_plus_first_certificate_seconds"],
                "phase_seconds": summary["phase_seconds"],
                "peak_allocated_bytes": summary["peak_allocated_bytes"],
                "target_hit": summary["observed_target_met"],
                "prefix_improved": summary["best_length"] < summary["prefix_length"],
                "controls": {
                    method: {
                        "best_length": row["summary"]["best_length"],
                        "complete_seconds": row["prepare_plus_exit_seconds"],
                    }
                    for method, row in group.items()
                    if method != "pbit"
                },
            }
        )
        for method in ("random", "assignment", "ils"):
            control = group[method]["summary"]
            if control["prefix_length"] != summary["prefix_length"]:
                raise ValueError("Paired methods did not receive the same prefix quality")
            pairs.append(
                {"method": method, "difference": summary["best_length"] - control["best_length"]}
            )
        for relative in pbit["state_files"]:
            if Path(relative).name == "initial.npz":
                continue
            with np.load(core / relative, allow_pickle=False) as saved:
                feasible = np.all(saved["outgoing"] == 1, axis=1) & np.all(
                    saved["incoming"] == 1, axis=1
                )
                raw.append({"states": len(feasible), "degree_feasible": int(feasible.sum())})
    metrics = {
        "schema": "tsp-candidate-workflow-metrics-v1",
        "natural_calls": len(natural),
        "pbit_natural_calls": len(rows),
        "attribution_control_calls": len(natural) - len(rows),
        "origin_replays": len(replays),
        "pbit_target_hits": sum(row["target_hit"] for row in rows),
        "pbit_prefix_improvements": sum(row["prefix_improved"] for row in rows),
        "maximum_single_model_bits": max(row["nodes"] for row in rows),
        "maximum_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
        "simultaneous_replicas": sorted(
            {row["summary"]["simultaneous_replicas"] for row in natural}
        ),
        "complete_call_seconds_including_replay": sum(
            row["prepare_plus_exit_seconds"] for row in data["calls"]
        ),
        "pbit_complete_seconds": sum(row["complete_seconds"] for row in rows),
        "pbit_phase_seconds": {
            phase: sum(row["phase_seconds"][phase] for row in rows)
            for phase in rows[0]["phase_seconds"]
        },
        "pbit_boundary_states": sum(row["states"] for row in raw),
        "pbit_degree_feasible_boundary_states": sum(row["degree_feasible"] for row in raw),
        "comparisons": {
            method: {
                "pbit_wins": sum(row["difference"] < 0 for row in pairs if row["method"] == method),
                "ties": sum(row["difference"] == 0 for row in pairs if row["method"] == method),
                "pbit_losses": sum(
                    row["difference"] > 0 for row in pairs if row["method"] == method
                ),
            }
            for method in ("random", "assignment", "ils")
        },
        "instances": rows,
        "new_solver_calls": 0,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
    labels = [row["instance"] for row in rows]
    styles = {
        "pbit": ("p-bit workflow", "#1565c0"),
        "random": ("Random + readout", "#ef6c00"),
        "assignment": ("Assignment + repair", "#388e3c"),
        "ils": ("Same-prefix ILS", "#8e24aa"),
    }
    for method, (label, color) in styles.items():
        costs = [
            row["best_length"] if method == "pbit" else row["controls"][method]["best_length"]
            for row in rows
        ]
        seconds = [
            row["complete_seconds"]
            if method == "pbit"
            else row["controls"][method]["complete_seconds"]
            for row in rows
        ]
        gaps = [
            100 * (length / row["optimum"] - 1) for row, length in zip(rows, costs, strict=True)
        ]
        axes[0].plot(labels, gaps, marker="o", label=label, color=color, alpha=0.85)
        axes[1].plot(labels, seconds, marker="o", label=label, color=color, alpha=0.85)
    axes[0].set_ylabel("Retained tour gap (%)")
    axes[1].set_ylabel("Preparation + process exit (s)")
    axes[0].set_title("RTX 5090: one fixed seed per instance; replays excluded")
    axes[0].legend(ncols=2, fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_ylim(bottom=0)
    figure.savefig(output / "quality-and-cost.png", dpi=170)
    plt.close(figure)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = summarize(args.core.resolve(), args.output.resolve())
    print(json.dumps({key: value for key, value in metrics.items() if key != "instances"}))


if __name__ == "__main__":
    main()
