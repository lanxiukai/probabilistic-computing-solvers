"""Recompute published coupling-level statistics from the complete call table."""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np


def summarize_confirmation(rows, definitions, *, bootstrap_seed=2026101907):
    expected = {(r["length"], r["instance_id"], r["seed"], r["family"]) for r in definitions}
    actual = [(r["length"], r["instance_id"], r["seed"], r["family"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Held-out panel is missing calls or includes duplicate/unplanned calls")
    if any(r["phase"] != "confirmation" or r["status"] != "verified" for r in rows):
        raise ValueError("Only independently verified confirmation rows may enter this report")
    frozen = {(r["length"], r["instance_id"], r["seed"], r["family"]): r for r in definitions}
    for row in rows:
        definition = frozen[row["length"], row["instance_id"], row["seed"], row["family"]]
        if (
            row["candidate"] != definition["candidate"]
            or row["groups"] != definition["solver"]["groups"]
        ):
            raise ValueError("Candidate or group count differs from frozen confirmation")
        if any(len(values) != row["groups"] for values in row["checkpoints"].values()):
            raise ValueError("Checkpoint group counts are inconsistent")
    rng = np.random.default_rng(bootstrap_seed)
    lengths = sorted({r["length"] for r in definitions})
    families = ["pt", "sa", "pa"]
    result = {}
    for length in lengths:
        instances = sorted({r["instance_id"] for r in definitions if r["length"] == length})
        bootstrap = rng.integers(0, len(instances), size=(20000, len(instances)))
        panel = {}
        for family in families:
            chosen = [r for r in rows if (r["length"], r["family"]) == (length, family)]
            values = {}
            for instance in instances:
                per_instance = [r for r in chosen if r["instance_id"] == instance]
                values[str(instance)] = {
                    str(t): float(
                        np.mean(
                            [energy for r in per_instance for energy in r["checkpoints"][str(t)]]
                        )
                    )
                    for t in (1, 5, 15, 30, 60)
                }
            panel[family] = {
                "instances": len(instances),
                "calls": len(chosen),
                "groups": sum(r["groups"] for r in chosen),
                "per_instance": values,
                "mean_energy": statistics.mean(values[str(i)]["60"] for i in instances),
                "mean_sweeps": statistics.mean(r["completed_sweeps"] for r in chosen),
                "mean_complete_call_seconds": statistics.mean(
                    r["complete_call_seconds"] for r in chosen
                ),
                "median_complete_call_seconds": statistics.median(
                    r["complete_call_seconds"] for r in chosen
                ),
                "max_complete_call_seconds": max(r["complete_call_seconds"] for r in chosen),
                "max_peak_allocated_mib": max(r["peak_allocated_bytes"] for r in chosen) / 1024**2,
                "late_groups_excluded": sum(r["late_groups_excluded"] for r in chosen),
                "max_sampling_overrun_seconds": max(r["sampling_overrun_seconds"] for r in chosen),
            }
            if family == "pa" and all(r.get("population") for r in chosen):
                ancestors = [
                    value for row in chosen for value in row["population"]["first_cycle_ancestors"]
                ]
                panel[family]["population_diagnostics"] = {
                    "first_cycle_ancestor_counts": ancestors,
                    "restarts_per_call": [row["population"]["restarts"] for row in chosen],
                    "interpretation": (
                        "Lineage diagnostics only; particles are not independent instances "
                        "and these counts do not establish equilibration."
                    ),
                }
        for family in families:
            differences = np.array(
                [
                    panel[family]["per_instance"][str(i)]["60"]
                    - panel["sa"]["per_instance"][str(i)]["60"]
                    for i in instances
                ]
            )
            draws = differences[bootstrap].mean(axis=1)
            panel[family]["paired_delta_to_sa"] = {
                "instance_differences": differences.tolist(),
                "mean": float(differences.mean()),
                "bootstrap_95_percent_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
                "better_tied_worse_instances": [
                    int((differences < 0).sum()),
                    int((differences == 0).sum()),
                    int((differences > 0).sum()),
                ],
                "unit": (
                    "Independent coupling instances; seed/group outcomes averaged within instance."
                ),
            }
        result[str(length)] = panel
    return {
        "protocol": "ea3d-s23-cloud-5090-v1",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": 20000,
        "calls": len(rows),
        "panels": result,
        "optimality_proven": False,
        "comparison_scope": (
            "Three in-house implementations on matched instances and sampling budgets; "
            "no external-baseline or hardware superiority claim."
        ),
    }


def analyze(data, panel):
    rows = json.loads((data / "calls.json").read_text())["calls"]
    rows = sorted(rows, key=lambda row: row["execution_order"])
    accepted = [r for r in rows if r["phase"] == "confirmation" and r["status"] == "verified"]
    definitions = [d for d in panel["cells"] if d["phase"] == "confirmation"]
    result = summarize_confirmation(accepted, definitions, bootstrap_seed=panel["bootstrap_seed"])
    result["sensitivity_l32_without_retried_instance"] = summarize_confirmation(
        [r for r in accepted if (r["length"], r["instance_id"]) != (32, 9)],
        [d for d in definitions if (d["length"], d["instance_id"]) != (32, 9)],
        bootstrap_seed=panel["bootstrap_seed"],
    )["panels"]["32"]
    result["actual_attempts"] = len(rows)
    result["failed_attempts"] = sum(r["status"] == "failed" for r in rows)
    result["all_attempt_complete_call_seconds"] = math.fsum(
        r["complete_call_seconds"] for r in rows
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/spin-glass/s23"))
    parser.add_argument("--panel", type=Path, default=Path("configs/spin-glass/s23-panel.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Analysis output already exists")
    result = analyze(args.data, json.loads(args.panel.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"confirmation_calls": result["calls"], "actual_attempts": result["actual_attempts"]}
        )
    )


if __name__ == "__main__":
    main()
