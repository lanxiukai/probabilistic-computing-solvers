"""Configuration and resource planning for tsp experiments."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from combinatorial_solvers.workflows.runtime import (
    linear_schedule,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_length(problem: Mapping[str, Any], optimum: int | None) -> int:
    explicit = problem.get("target_length")
    if explicit is not None:
        target = int(explicit)
    else:
        if optimum is None:
            raise ValueError("target_length or optimum is required for first-hit statistics")
        relative_gap = float(problem.get("target_relative_gap", 0.0))
        if not math.isfinite(relative_gap) or relative_gap < 0:
            raise ValueError("target_relative_gap must be finite and nonnegative")
        target = math.floor(optimum * (1.0 + relative_gap) + 1e-12)
    if target <= 0:
        raise ValueError("target_length must be positive")
    if optimum is not None and target < optimum:
        raise ValueError("target_length cannot be below the declared exact optimum")
    return target


def _permutation_beta_schedule(
    solver: Mapping[str, Any],
    distance: np.ndarray,
) -> tuple[tuple[float, ...], dict[str, float | str]]:
    """Build an absolute or median-edge-normalized permutation schedule."""

    mode = solver.get("beta_scale_mode")
    if mode is None:
        schedule = linear_schedule(solver)
        return schedule, {
            "beta_scale_mode": "absolute",
            "beta_start_resolved": schedule[0],
            "beta_stop_resolved": schedule[-1],
        }
    if str(mode) != "median_edge_distance":
        raise ValueError("beta_scale_mode must be 'median_edge_distance' when provided")
    if "beta_start" in solver or "beta_stop" in solver:
        raise ValueError("Do not mix absolute and dimensionless beta endpoints")

    matrix = np.asarray(distance)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("distance must be a square matrix with at least two cities")
    edges = matrix[np.triu_indices(matrix.shape[0], k=1)].astype(np.float64, copy=False)
    if edges.size == 0 or not np.all(np.isfinite(edges)) or np.any(edges < 0):
        raise ValueError("Off-diagonal TSP distances must be finite and nonnegative")
    distance_scale = float(np.median(edges))
    if distance_scale <= 0:
        raise ValueError("The median off-diagonal TSP distance must be positive")
    dimensionless_start = float(solver["dimensionless_beta_start"])
    dimensionless_stop = float(solver["dimensionless_beta_stop"])
    if (
        not math.isfinite(dimensionless_start)
        or not math.isfinite(dimensionless_stop)
        or dimensionless_start <= 0
        or dimensionless_stop <= 0
    ):
        raise ValueError("Dimensionless beta endpoints must be finite and positive")

    schedule_config = {
        **solver,
        "beta_start": dimensionless_start / distance_scale,
        "beta_stop": dimensionless_stop / distance_scale,
    }
    schedule = linear_schedule(schedule_config)
    return schedule, {
        "beta_scale_mode": "median_edge_distance",
        "beta_distance_scale": distance_scale,
        "dimensionless_beta_start": dimensionless_start,
        "dimensionless_beta_stop": dimensionless_stop,
        "beta_start_resolved": schedule[0],
        "beta_stop_resolved": schedule[-1],
    }
