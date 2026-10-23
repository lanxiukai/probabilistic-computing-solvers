"""Small reproducible tables and plots with the frozen denominator preserved."""

from __future__ import annotations

import csv
from importlib.resources import files
from pathlib import Path

from .solver.publication_io import atomic_json, canonical_digest, read_json


def build_report(plan_path: Path, runs: Path, output: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = read_json(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for config in plan["physical_runs"]:
        run = runs / config["run_id"]
        summary = run / "result/summary.json"
        row = {
            "run_id": config["run_id"],
            "instance_id": config["instance_id"],
            "product_bits": config["product_bits"],
            "target": str(config["target"]),
            "status": "not_executed",
            "success": False,
            "first_hit_seconds": None,
            "sampling_seconds": None,
            "cell_wall_seconds": None,
            "events": None,
        }
        if summary.exists():
            actual = read_json(summary)
            frozen = read_json(run / "config.json")
            if frozen.get("plan_sha256") != canonical_digest(plan):
                raise ValueError("Reported result is not bound to this complete frozen plan")
            environment = read_json(run / "environment.json")
            if environment.get("torch_gpu_name") != plan["hardware"]["gpu"]:
                raise ValueError("Primary release tables require measured RTX 5090 results")
            if any(frozen.get(key) != value for key, value in config.items()):
                raise ValueError("Reported run does not match the frozen configuration")
            if actual["target"] != config["target"] or actual["arm"] != "physical":
                raise ValueError("Wrong result target or success criterion")
            row.update(
                {
                    key: actual[key]
                    for key in (
                        "status",
                        "success",
                        "first_hit_seconds",
                        "sampling_seconds",
                        "cell_wall_seconds",
                        "events",
                    )
                }
            )
        elif (run / "started.json").exists():
            row["status"] = "incomplete_or_failed"
        rows.append(row)
    with (output / "results.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "planned_runs": len(rows),
        "reported_runs": sum(r["events"] is not None for r in rows),
        "successes": sum(r["success"] for r in rows),
        "unexecuted": sum(r["status"] == "not_executed" for r in rows),
        "failures_or_partial": sum(r["status"] == "incomplete_or_failed" for r in rows),
        "tts99_estimated": False,
        "rows": rows,
    }
    atomic_json(output / "summary.json", summary)
    measured = [r for r in rows if r["sampling_seconds"] is not None]
    if measured:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for index, row in enumerate(measured):
            x = row["product_bits"] + (index % 3 - 1) * 0.35
            marker, color = ("o", "#2878b5") if row["success"] else ("x", "#d1495b")
            y = row["first_hit_seconds"] if row["success"] else row["sampling_seconds"]
            if y and y > 0:
                axes[0].scatter(x, y, marker=marker, color=color)
            axes[1].scatter(x, row["cell_wall_seconds"], marker=marker, color=color)
        for ax, title in zip(
            axes, ("Sampling first passage / censoring", "End-to-end cell time"), strict=True
        ):
            ax.set(xlabel="Product bits", ylabel="Seconds", yscale="log", title=title)
            ax.set_xticks([64, 80, 96, 112, 120, 128])
            ax.grid(alpha=0.2)
        figure.suptitle("All frozen runs: circles = success; crosses = censored")
        figure.savefig(output / "runtime.png", dpi=180)
        figure.savefig(output / "runtime.svg", metadata={"Date": None})
        svg = output / "runtime.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(figure)
    values = {
        "planned_runs": len(rows),
        "reported_runs": summary["reported_runs"],
        "successes": summary["successes"],
        "unexecuted": summary["unexecuted"],
    }
    templates = files("sparse_factorization").joinpath("templates")
    for name in ("report.md", "report.zh-CN.md"):
        template = templates.joinpath(name).read_text(encoding="utf-8")
        (output / name).write_text(template.format(**values), encoding="utf-8")
    return {key: value for key, value in summary.items() if key != "rows"}
