"""Plot all frozen covering confirmation calls without invoking a solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=Path, default=Path("data/set-cover/rtx5090/calls.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/set-cover/figures"))
    args = parser.parse_args()
    calls = json.loads(args.calls.read_text())
    panel = json.loads(args.calls.with_name("panel.json").read_text())
    targets = panel["frozen_targets"]
    names = list(targets)
    methods = ["heatbath", "metropolis", "plateau", "ac-cft"]
    labels = ["p-bit heat bath", "Block Metropolis", "Block plateau", "AC-CFT (CPU)"]
    colors = ["#0072B2", "#D55E00", "#CC79A7", "#009E73"]
    observations = []
    for name in names:
        for method in methods:
            chosen = [
                row
                for row in calls
                if row["trial_kind"] == "natural"
                and row["instance"] == name
                and row["metrics"]["method"] == method
            ]
            if len(chosen) != 3:
                raise ValueError("Each instance/method must retain all three frozen seeds")
            observations.append(
                {
                    "instance": name,
                    "method": method,
                    "calls": [row["id"] for row in chosen],
                    "costs": [row["metrics"]["best_feasible_cost"] for row in chosen],
                    "gap_percent": [
                        100 * (row["metrics"]["best_feasible_cost"] / targets[name]["cost"] - 1)
                        for row in chosen
                    ],
                    "complete_seconds": [
                        row["metrics"]["complete_runner_seconds"] for row in chosen
                    ],
                }
            )
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.84, hspace=0.15)
    x = np.arange(len(names))
    width = 0.19
    for index, (method, label, color) in enumerate(zip(methods, labels, colors, strict=True)):
        for ax, key in zip(axes, ("gap_percent", "complete_seconds"), strict=True):
            values = np.asarray(
                [
                    next(
                        row[key]
                        for row in observations
                        if row["instance"] == name and row["method"] == method
                    )
                    for name in names
                ]
            )
            means = values.mean(axis=1)
            errors = np.maximum(
                0, np.vstack((means - values.min(axis=1), values.max(axis=1) - means))
            )
            ax.bar(
                x + (index - 1.5) * width,
                means,
                width,
                yerr=errors,
                capsize=2,
                color=color,
                label=label,
            )
    axes[0].set_ylabel("Gap to frozen feasible target (%)")
    axes[1].set_ylabel("Complete runner time (s, log scale)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, names)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    handles, names_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        names_legend,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.53, 0.94),
    )
    fig.suptitle("Weighted set cover: three-seed RTX 5090 confirmation", y=0.985)
    fig.text(
        0.5,
        0.045,
        "Bars: three-seed mean; whiskers: observed range, not confidence intervals.\n"
        "Native arms: 128 GPU replicas; AC-CFT: one EPYC 9354 CPU thread.\n"
        "Targets are feasible references; only independently closed bounds establish optima.",
        ha="center",
        fontsize=9,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "rtx5090.svg", metadata={"Date": None})
    fig.savefig(args.output / "rtx5090.png", dpi=180)
    plt.close(fig)
    (args.output / "values.json").write_text(json.dumps(observations, indent=2) + "\n")


if __name__ == "__main__":
    main()
