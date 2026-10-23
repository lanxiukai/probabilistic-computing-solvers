"""Render paired reports and scientific figures from curated, audited core data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SOFTWARE = (
    ("Python", "https://www.python.org/"),
    ("NumPy", "https://numpy.org/"),
    ("PyTorch", "https://pytorch.org/"),
    ("Triton", "https://github.com/triton-lang/triton"),
    ("CUDA", "https://docs.nvidia.com/cuda/"),
    ("SciPy", "https://scipy.org/"),
    ("Matplotlib", "https://matplotlib.org/"),
    ("PyYAML", "https://pyyaml.org/"),
    ("uv", "https://docs.astral.sh/uv/"),
    ("GCC", "https://gcc.gnu.org/"),
    ("pytest", "https://docs.pytest.org/"),
    ("Ruff", "https://docs.astral.sh/ruff/"),
)


def read(path):
    return json.loads(path.read_text())


def calls(root, panel):
    return read(root / "data" / panel / "rtx5090/calls.json")


def group(rows, *keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row["metrics"][key] for key in keys)].append(row)
    return dict(sorted(result.items()))


def table(rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def avg(rows, key):
    return fmean(float(row["metrics"][key]) for row in rows)


def total(rows):
    return fmean(
        float(row["metrics"]["complete_runner_seconds"])
        + float(row["metrics"].get("common_initial_runner_seconds", 0))
        for row in rows
    )


def number(value, places=3):
    return f"{value:.{places}f}"


def bar_figure(root, topic, labels, values, ylabel, title):
    horizontal = len(labels) > 8
    fig, ax = plt.subplots(
        figsize=(8.5, max(4.5, len(labels) * 0.35 + 1)) if horizontal else (8, 4.5),
        layout="constrained",
    )
    if horizontal:
        ax.barh(np.arange(len(labels)), values, color="#427eae")
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        ax.set_xlabel(ylabel)
        ax.invert_yaxis()
        if topic == "partition":
            ax.set_xscale("log")
            ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
    else:
        ax.bar(np.arange(len(labels)), values, color="#427eae")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    folder = root / "docs" / topic / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"rtx5090.{suffix}", dpi=180)
    plt.close(fig)


def statistics(root, topic):
    values = {}
    if topic == "tsp":
        rows = [r for r in calls(root, topic) if r["trial_kind"] == "natural"]
        groups = group(rows, "method")
        values["RESULT_ROWS"] = table(
            [
                (
                    method,
                    len(items),
                    sum(x["metrics"]["within_budget_hit"] for x in items),
                    number(median(x["metrics"]["prepare_plus_first_seconds"] for x in items)),
                    number(median(x["metrics"]["after_preparation_first_seconds"] for x in items)),
                )
                for (method,), items in groups.items()
            ]
        )
        lkh, pbit = groups[("lkh",)], groups[("cuda_pbit",)]
        reference = {x["metrics"]["seed"]: x["metrics"]["prepare_plus_first_seconds"] for x in lkh}
        values.update(
            LKH_MEDIAN=number(median(reference.values())),
            PBIT_MEDIAN=number(median(x["metrics"]["prepare_plus_first_seconds"] for x in pbit)),
            PBIT_RATIO=number(
                median(
                    x["metrics"]["prepare_plus_first_seconds"] / reference[x["metrics"]["seed"]]
                    for x in pbit
                )
            ),
        )
        bar_figure(
            root,
            topic,
            [k[0] for k in groups],
            [
                median(x["metrics"]["prepare_plus_first_seconds"] for x in g)
                for g in groups.values()
            ],
            "Complete first-certificate seconds",
            "TSP: fixed 150-city confirmation",
        )
    elif topic == "mwis":
        rows = [r for r in calls(root, topic) if r["metrics"]["size"] == 32768]
        groups = group(rows, "method")
        means = {k[0]: avg(v, "weight") for k, v in groups.items()}
        values["RESULT_ROWS"] = table(
            [
                (k[0], len(v), number(means[k[0]]), number(avg(v, "runner_seconds")))
                for k, v in groups.items()
            ]
        )
        values["PAIR_GAIN"] = number(100 * (means["pair-gibbs"] / means["pbit"] - 1))
        values["PAIR_GAP"] = number(100 * (means["pair-gibbs"] / means["chils-local"] - 1))
        bar_figure(
            root,
            topic,
            list(means),
            list(means.values()),
            "Mean feasible weight",
            "MWIS: 32768 vertices",
        )
    elif topic == "maxsat":
        groups = group(calls(root, topic), "method")
        score = {k[0]: avg(v, "censored_target_seconds") for k, v in groups.items()}
        values["RESULT_ROWS"] = table(
            [
                (
                    k[0],
                    len(v),
                    sum(x["metrics"]["target_success"] for x in v),
                    number(score[k[0]]),
                    number(avg(v, "runner_seconds")),
                )
                for k, v in groups.items()
            ]
        )
        values.update(
            PBIT_SECONDS=number(score["pbit"]),
            METRO_SECONDS=number(score["metropolis"]),
            CPU_RATIO=number(
                min(v for k, v in score.items() if k.startswith("usw-")) / score["pbit"]
            ),
        )
        bar_figure(
            root,
            topic,
            list(score),
            list(score.values()),
            "Target score seconds (miss = 60)",
            "Weighted Max-SAT: 32768 variables",
        )
    elif topic == "maxcut":
        sparse = [
            r
            for r in calls(root, "maxcut-sparse")
            if r["metrics"]["size"] == 32768 or r["metrics"]["instance_id"].startswith("G")
        ]
        values["SPARSE_ROWS"] = table(
            [
                (
                    key[0],
                    key[1],
                    len(items),
                    number(avg(items, "cut")),
                    number(avg(items, "runner_seconds")),
                )
                for key, items in group(sparse, "instance_id", "method").items()
            ]
        )
        sparse_groups = group(
            [r for r in sparse if r["metrics"]["size"] == 32768], "instance_id", "method"
        )
        sparse_names = sorted({key[0] for key in sparse_groups})
        values["SPARSE_HB_GRAPHS"] = str(len(sparse_names))
        values["SPARSE_HB_WINS"] = str(
            sum(
                avg(sparse_groups[name, "pbit-anneal"], "cut")
                > avg(sparse_groups[name, "BURER2002"], "cut")
                for name in sparse_names
            )
        )
        dense = calls(root, "maxcut-dense")
        groups = group(dense, "instance_id", "variant")
        base = {
            name: avg(items, "negative_energy")
            for (name, variant), items in groups.items()
            if variant.startswith("dsb")
        }
        differences = {
            key: 100 * (avg(items, "negative_energy") / base[key[0]] - 1)
            for key, items in groups.items()
        }
        values["DENSE_ROWS"] = table(
            [
                (
                    key[0],
                    key[1],
                    len(items),
                    number(avg(items, "negative_energy")),
                    number(differences[key]) + "%",
                    number(avg(items, "runner_seconds")),
                )
                for key, items in groups.items()
            ]
        )
        gates = {}
        for prefix, variant in (("sk", "onsager"), ("er", "block8x16")):
            selected = [
                v
                for (name, kind), v in differences.items()
                if name.startswith(prefix) and kind == variant
            ]
            gates[prefix] = len(selected) == 2 and min(selected) >= 1
        values["DENSE_GATES"] = ", ".join(f"{k}={str(v).lower()}" for k, v in gates.items())
        values["DENSE_GIB"] = number(
            max(r["metrics"]["peak_cuda_allocated_bytes"] for r in dense) / 2**30
        )
        selected = {"/".join(k): v for k, v in differences.items() if not k[1].startswith("dsb")}
        bar_figure(
            root,
            topic,
            list(selected),
            list(selected.values()),
            "Negative-energy difference from dSB (%)",
            "Dense Max-Cut: matched GPU controls",
        )
    elif topic == "labs":
        rows = [r for r in calls(root, topic) if r["metrics"]["size"] == 32768]
        groups = group(rows, "variant")
        values["RESULT_ROWS"] = table(
            [
                (
                    k[0],
                    len(v),
                    number(avg(v, "energy")),
                    number(avg(v, "merit_factor"), 6),
                    number(avg(v, "runner_seconds")),
                )
                for k, v in groups.items()
            ]
        )
        values["BEST_ENERGY"] = str(min(r["metrics"]["energy"] for r in rows))
        greedy = groups[("polish",)]
        greedy_energy = avg(greedy, "energy")
        values["GREEDY_ENERGY"] = number(greedy_energy, 0)
        values["GREEDY_SECONDS"] = number(avg(greedy, "runner_seconds"))
        values["BEST_IMPROVEMENT_PERCENT"] = number(
            100 * (greedy_energy - int(values["BEST_ENERGY"])) / greedy_energy, 6
        )
        values["STOCHASTIC_SECONDS"] = number(
            fmean(
                r["metrics"]["runner_seconds"]
                for r in rows
                if r["metrics"]["variant"] in {"pbit-legendre", "metro-legendre", "mts-legendre"}
            )
        )
        labels = [k[0] for k in groups]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        axes[0].bar(np.arange(len(labels)), [avg(v, "merit_factor") for v in groups.values()])
        axes[1].bar(np.arange(len(labels)), [avg(v, "runner_seconds") for v in groups.values()])
        axes[0].set_ylabel("Mean merit factor")
        axes[1].set_ylabel("Mean complete runner seconds")
        axes[1].set_yscale("log")
        for ax in axes:
            ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("Aperiodic LABS: length 32768")
        folder = root / "docs" / topic / "figures"
        folder.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "svg"):
            fig.savefig(folder / f"rtx5090.{suffix}", dpi=180)
        plt.close(fig)
    elif topic == "partition":
        groups = group(calls(root, topic), "instance", "variant")
        values["RESULT_ROWS"] = table(
            [
                (k[0], k[1], len(v), number(avg(v, "best_cut")), number(total(v)))
                for k, v in groups.items()
            ]
        )
        reference = {
            k[0]: avg(v, "best_cut") for k, v in groups.items() if k[1] == "kaffpa-restarts"
        }
        names = sorted(reference)
        warm = [
            row
            for row in calls(root, topic)
            if row["metrics"]["variant"] in {"hb4-warm", "greedy4-warm"}
        ]
        values.update(
            COLD_METRO_WINS=str(
                sum(
                    avg(groups[name, "hb4"], "best_cut") < avg(groups[name, "metro2"], "best_cut")
                    for name in names
                )
            ),
            COLD_REFERENCE_WINS=str(
                sum(avg(groups[name, "hb4"], "best_cut") < reference[name] for name in names)
            ),
            GRAPHS=str(len(names)),
            WARM_IMPROVEMENTS=str(
                sum(
                    row["metrics"]["best_cut"]
                    < avg(groups[row["metrics"]["instance"], "initial"], "best_cut")
                    for row in warm
                )
            ),
            WARM_CALLS=str(len(warm)),
        )
        selected = {
            "/".join(k): avg(v, "best_cut") / reference[k[0]]
            for k, v in groups.items()
            if k[1] in {"hb4", "hb4-warm", "greedy4-warm"}
        }
        bar_figure(
            root,
            topic,
            list(selected),
            list(selected.values()),
            "Cut / KaFFPa restart cut (lower is better)",
            "Strict bisection: 32768 vertices",
        )
    elif topic == "ldpc":
        rows = calls(root, topic)
        groups = group(rows, "case_id", "variant")

        def extent(items, key):
            numbers = [r["metrics"][key] for r in items]
            return (
                str(min(numbers))
                if min(numbers) == max(numbers)
                else f"{min(numbers)}..{max(numbers)}"
            )

        values["RESULT_ROWS"] = table(
            [
                (
                    k[0],
                    k[1],
                    len(v),
                    extent(v, "valid_frames"),
                    extent(v, "frame_errors"),
                    v[0]["metrics"]["frames"],
                    number(total(v)),
                )
                for k, v in groups.items()
            ]
        )
        warm = [r for r in rows if r["warm_start"] and r["configuration"]["method"] == "native"]
        inherited = 0
        for r in warm:
            with np.load(root / "data/ldpc/rtx5090" / r["witness_file"], allow_pickle=False) as p:
                inherited += np.array_equal(p["initial_bits"], p["best_bits"])
        references = {
            r["metrics"]["case_id"]: r["metrics"] for r in rows if r["metrics"]["variant"] == "lsd"
        }
        lower = 0
        for r in rows:
            if r["configuration"]["method"] != "native":
                continue
            m, ref = r["metrics"], references[r["metrics"]["case_id"]]
            paired = np.asarray(m["frame_model_valid"]) & np.asarray(ref["frame_model_valid"])
            lower += np.count_nonzero(
                paired & (np.asarray(m["frame_costs"]) < np.asarray(ref["frame_costs"]))
            )
        values.update(
            WARM_INHERITED=str(inherited), WARM_CALLS=str(len(warm)), LOWER_THAN_LSD=str(lower)
        )
        wifi = {
            k[1]: avg(v, "frame_errors")
            for k, v in groups.items()
            if k[0] == "wifi648-heldout-esm2"
        }
        bar_figure(
            root,
            topic,
            list(wifi),
            list(wifi.values()),
            "Frame errors / 64 fixed frames",
            "WiFi-648: sampler seeds reuse the same noise",
        )
    else:
        raise ValueError(topic)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--topic", action="append")
    args = parser.parse_args()
    for topic in args.topic or ("tsp", "mwis", "maxsat", "maxcut", "labs", "partition", "ldpc"):
        values = statistics(args.root, topic)
        output = args.root / "docs" / topic / "figure-values.json"
        output.write_text(json.dumps(values, indent=2) + "\n")
        print(json.dumps({"topic": topic, "status": "figures-regenerated"}))


if __name__ == "__main__":
    main()
