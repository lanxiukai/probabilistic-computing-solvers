"""Classical covering controls and independently checkable lower bounds."""

from __future__ import annotations

import heapq
import json
import subprocess
import warnings
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix

from .reference import dual_bound


def fast_greedy(instance):
    masks = [sum(1 << i for i in rows) for rows in instance.columns]
    heap = [
        (Fraction(c, len(rows)), j, len(rows))
        for j, (c, rows) in enumerate(zip(instance.costs, instance.columns, strict=True))
        if rows
    ]
    heapq.heapify(heap)
    remaining = (1 << len(instance.rows)) - 1
    selected = np.zeros(instance.size, dtype=np.int8)
    coverage = np.zeros(len(instance.rows), dtype=np.int32)
    while remaining:
        _, j, previous = heapq.heappop(heap)
        gain = (masks[j] & remaining).bit_count()
        if gain == 0:
            continue
        if gain != previous:
            heapq.heappush(heap, (Fraction(instance.costs[j], gain), j, gain))
            continue
        selected[j] = 1
        coverage[list(instance.columns[j])] += 1
        remaining &= ~masks[j]
    for j in sorted(np.flatnonzero(selected), key=lambda j: (-instance.costs[j], int(j))):
        rows = list(instance.columns[j])
        if np.all(coverage[rows] >= 2):
            selected[j] = 0
            coverage[rows] -= 1
    return selected


def constraint_matrix(instance):
    rows = np.repeat(np.arange(len(instance.rows), dtype=np.int32), list(map(len, instance.rows)))
    cols = np.asarray([j for row in instance.rows for j in row], dtype=np.int32)
    return coo_matrix(
        (np.ones(len(cols)), (rows, cols)), shape=(len(instance.rows), instance.size)
    ).tocsc()


def certify_float_dual(instance, values, scale=1_000_000):
    values = np.asarray(values)
    if values.shape != (len(instance.rows),) or not np.isfinite(values).all():
        raise ValueError("Invalid floating dual proposal")
    numerators = np.floor(np.maximum(values, 0) * scale).astype(np.int64)
    denominator = max(
        scale,
        max(
            (sum(int(numerators[i]) for i in rows) + cost - 1) // cost
            for rows, cost in zip(instance.columns, instance.costs, strict=True)
        ),
    )
    certificate = dual_bound(instance, numerators, denominator)
    return numerators, certificate


def mip_reference(instance, max_seconds=60.0):
    started = perf_counter()
    matrix = constraint_matrix(instance)
    arrays = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lp_start = perf_counter()
        lp = linprog(
            np.asarray(instance.costs, dtype=float),
            A_ub=-matrix,
            b_ub=-np.ones(len(instance.rows)),
            bounds=(0, None),
            method="highs",
            options={"time_limit": min(15.0, max_seconds / 4), "threads": 1},
        )
        lp_seconds = perf_counter() - lp_start
        certified = None
        if lp.success:
            arrays["lp_values"] = lp.x
            arrays["lp_row_marginals"] = lp.ineqlin.marginals
            numerators, certified = certify_float_dual(instance, -lp.ineqlin.marginals)
            arrays["dual_numerators"] = numerators
        remaining = max(0.001, max_seconds - (perf_counter() - started))
        mip = milp(
            np.asarray(instance.costs, dtype=float),
            integrality=np.ones(instance.size),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix, 1, np.inf),
            options={"time_limit": remaining, "mip_rel_gap": 0, "threads": 1},
        )
    certificate = None
    if mip.x is not None:
        arrays["mip_values"] = mip.x
        if np.max(np.abs(mip.x - np.rint(mip.x))) <= 1e-6:
            bits = np.rint(mip.x).astype(np.int8)
            candidate = instance.certificate(bits)
            if candidate["feasible"]:
                certificate = candidate
                arrays["best_bits"] = bits[None, :]
    result = {
        "method": "scipy-highs-milp-and-lp",
        "mip_status": int(mip.status),
        "mip_message": str(mip.message),
        "lp_status": int(lp.status),
        "lp_seconds": lp_seconds,
        "search_seconds": perf_counter() - started,
        "requested_seconds": max_seconds,
        "requested_cpu_threads": 1,
        "solver_reported_bound": float(mip.mip_dual_bound)
        if mip.get("mip_dual_bound") is not None and np.isfinite(mip.mip_dual_bound)
        else None,
        "solver_reported_gap": float(mip.mip_gap)
        if mip.get("mip_gap") is not None and np.isfinite(mip.mip_gap)
        else None,
        "rational_dual_certificate": certified,
        "certificate": certificate,
        "independent_optimality_proven": certificate is not None
        and certified is not None
        and certificate["cost"] == certified["integer_cost_lower_bound"],
        "warnings": [str(w.message) for w in caught],
    }
    return arrays, result


def cft_reference(instance, specification, output: Path):
    input_path, result_path = output / "cft-input.txt", output / "cft-result.json"
    with input_path.open("x") as stream:
        stream.write(f"{len(instance.rows)} {instance.size}\n")
        for cost, rows in zip(instance.costs, instance.columns, strict=True):
            stream.write(" ".join(map(str, (cost, len(rows), *rows))) + "\n")
    trace = output / "cft-clock.bin"
    command = [
        str(Path(specification["executable"]).resolve()),
        str(input_path.resolve()),
        str(result_path.resolve()),
        str(specification["seed"]),
        str(specification["max_seconds"]),
        str(trace.resolve()),
        str(Path(specification["clock_trace"]).resolve())
        if specification.get("clock_trace")
        else "-",
    ]
    started = perf_counter()
    with (output / "cft.log").open("x") as log:
        subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=specification.get("process_limit_seconds", 120),
        )
    result = json.loads(result_path.read_text())
    values = result["columns"]
    if len(set(values)) != len(values) or any(
        not isinstance(j, int) or not 0 <= j < instance.size for j in values
    ):
        raise ValueError("Invalid CFT selected columns")
    bits = np.zeros(instance.size, dtype=np.int8)
    bits[values] = 1
    certificate = instance.certificate(bits)
    if not certificate["feasible"] or certificate["cost"] != result["cost"]:
        raise ValueError("CFT original-objective certificate failed")
    arrays = {
        "best_bits": bits[None, :],
        "cft_dual_multipliers": np.asarray(result["dual_multipliers"]),
    }
    numerators, certified = certify_float_dual(instance, arrays["cft_dual_multipliers"])
    arrays["dual_numerators"] = numerators
    return arrays, {
        "method": "ac-cft",
        "certificate": certificate,
        "rational_dual_certificate": certified,
        "independent_optimality_proven": certificate["cost"]
        == certified["integer_cost_lower_bound"],
        "subprocess_seconds": perf_counter() - started,
        "search_seconds": result["search_seconds"],
        "clock_queries": result["clock_queries"],
        "clock_replay": result["clock_replay"],
        "requested_seconds": specification["max_seconds"],
        "cpu_threads": 1,
        "deadline_overrun_seconds": max(0, result["search_seconds"] - specification["max_seconds"]),
    }
