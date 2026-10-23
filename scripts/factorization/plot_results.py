"""Reproduce final tables and figures from every frozen run and its controls."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from sparse_factorization.report import build_report
from sparse_factorization.solver.publication_io import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/factorization/release-v3"))
    parser.add_argument("--plan", type=Path, default=Path("configs/factorization/release-v3.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "factorization-release-v3"
    build_report(args.plan, args.data / "physical", args.output)
    plan = read_json(args.plan)
    physical = [
        read_json(args.data / "physical" / r["run_id"] / "result/summary.json")
        for r in plan["physical_runs"]
    ]
    integer = [
        read_json(args.data / "integer" / r / "result/summary.json")
        for r in plan["controls"]["selected_root_integer"]["paired_runs"]
    ]
    classical = []
    for config in plan["controls"]["classical"]["runs"]:
        row = read_json(args.data / "classical" / config["run_id"] / "summary.json")
        row["product_bits"] = int(row["target"]).bit_length()
        classical.append(row)
    widths = [64, 80, 96, 112, 120, 128]
    aggregates = []
    for width in widths:
        group = [r for r in physical if r["product_bits"] == width]
        hits = [r["first_hit_seconds"] for r in group if r["success"]]
        cpu = [r["wall_seconds"] for r in classical if r["product_bits"] == width]
        control = [r for r in integer if r["product_bits"] == width]
        aggregates.append(
            {
                "product_bits": width,
                "physical_runs": len(group),
                "strict_successes": len(hits),
                "successful_first_passage_min_seconds": min(hits),
                "successful_first_passage_median_seconds": statistics.median(hits),
                "successful_first_passage_max_seconds": max(hits),
                "all_run_cell_median_seconds": statistics.median(
                    r["cell_wall_seconds"] for r in group
                ),
                "max_reserved_vram_bytes": max(r["peak_reserved_bytes"] for r in group),
                "integer_control_successes": sum(r["success"] for r in control),
                "integer_control_runs": len(control),
                "integer_control_cell_median_seconds": statistics.median(
                    r["cell_wall_seconds"] for r in control
                ),
                "yafu_wall_median_seconds": statistics.median(cpu),
            }
        )
    atomic_json(args.output / "aggregates.json", {"rows": aggregates})
    with (args.output / "controls.csv").open("x", newline="") as stream:
        fields = ["arm", "product_bits", "target", "success", "seconds", "timing_scope"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for kind, rows, field in [
            ("selected_root_integer", integer, "cell_wall_seconds"),
            ("yafu_cpu_noecm", classical, "wall_seconds"),
        ]:
            for row in rows:
                writer.writerow(
                    {
                        "arm": kind,
                        "product_bits": row["product_bits"],
                        "target": str(row["target"]),
                        "success": row["success"],
                        "seconds": row[field],
                        "timing_scope": field,
                    }
                )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True, sharey=True)
    for ax, rows, key, title in zip(
        axes,
        [physical, integer, classical],
        ["cell_wall_seconds", "cell_wall_seconds", "wall_seconds"],
        [
            "Accepted physical state: RTX 5090 (n=54)",
            "Selected-root integer: RTX 5090 (n=18)",
            "YAFU integer: one CPU thread (n=54)",
        ],
        strict=True,
    ):
        for i, row in enumerate(rows):
            ax.scatter(
                row["product_bits"] + (i % 3 - 1) * 0.4,
                row[key],
                marker="o" if row["success"] else "x",
                color="#2878b5" if row["success"] else "#d1495b",
            )
        ax.set(title=title, xlabel="Product bits", yscale="log", xticks=widths)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Measured elapsed seconds (log scale)")
    fig.suptitle(
        "Different success criteria; crosses are event-limited, not successful factorizations"
    )
    fig.savefig(args.output / "controls.png", dpi=180)
    fig.savefig(args.output / "controls.svg", metadata={"Date": None})
    svg = args.output / "controls.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(
        json.dumps(
            {
                "physical_runs": len(physical),
                "integer_controls": len(integer),
                "classical_controls": len(classical),
            }
        )
    )


if __name__ == "__main__":
    main()
