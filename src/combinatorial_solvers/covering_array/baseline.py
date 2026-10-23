"""NIST ACTS basic/IPOG constructive reference with complete coverage verification."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

from .model import Instance


def parse_acts_csv(path: Path, columns: int, levels: int):
    """Concretize documented don't-care cells without dropping any generated row."""
    lines = [
        line for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")
    ]
    table = list(csv.reader(lines))
    if not table or table[0] != [f"p{i}" for i in range(columns)]:
        raise ValueError("ACTS CSV parameter order differs from the declared model")
    rows = []
    stars = 0
    for row in table[1:]:
        if len(row) != columns:
            raise ValueError("ACTS CSV row has the wrong number of columns")
        values = []
        for text in row:
            token = text.strip()
            if token == "*":
                values.append(0)
                stars += 1
            elif token.isdigit() and int(token) < levels:
                values.append(int(token))
            else:
                raise ValueError("ACTS CSV contains an unsupported level token")
        rows.append(values)
    if not rows:
        raise ValueError("ACTS CSV contains no generated rows")
    return np.array(rows, dtype=np.int64), stars


def acts_reference(instance, config, output: Path):
    definition = "[System]\nName: homogeneous-pair-coverage\n[Parameter]\n"
    for i in range(instance.columns):
        definition += f"p{i} (int): " + ", ".join(str(x) for x in range(instance.levels)) + "\n"
    path = output / "acts-input.txt"
    path.write_text(definition)
    result = output / "acts-output.csv"
    command = [
        config.get("java", "java"),
        "-Xmx4g",
        "-Djava.awt.headless=true",
        "-Dalgo=ipog",
        "-Ddoi=2",
        "-Doutput=csv",
        "-Dmode=scratch",
        "-Dcheck=on",
        "-Drandstar=off",
        "-jar",
        str(Path(config["jar"]).resolve()),
        str(path.resolve()),
        str(result.resolve()),
    ]
    started = perf_counter()
    with (
        (output / "acts-stdout.log").open("w") as stdout,
        (output / "acts-stderr.log").open("w") as stderr,
    ):
        process = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            timeout=config.get("process_limit_seconds", 120),
            check=False,
        )
    elapsed = perf_counter() - started
    if process.returncode:
        raise RuntimeError("ACTS process failed; retained stdout/stderr contain the diagnostics")
    values, stars = parse_acts_csv(result, instance.columns, instance.levels)
    generated = Instance(len(values), instance.columns, instance.levels)
    certificate = generated.certificate(values)
    if not certificate["feasible"]:
        raise AssertionError("ACTS output failed independent full interaction enumeration")
    return {"constructed_array": values}, {
        "method": "acts-ipog",
        "pbit_heatbath_participation": False,
        "physical_update_pbits_per_model": 0,
        "best_feasible_cost": 0,
        "constructed_rows": len(values),
        "fixed_row_target": instance.rows,
        "fixed_row_target_met": len(values) <= instance.rows,
        "search_seconds": elapsed,
        "certificate": certificate,
        "parallel_workers": 1,
        "dont_care_cells": stars,
        "dont_care_readout": (
            "Every documented '*' is set to zero; all generated rows retained "
            "and independently certified"
        ),
        "comparison_scope": (
            "Constructive row-count reference; not a fixed-row search or equal-work control"
        ),
    }
