"""Certificate and diagnostic artifact writers for tsp."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from combinatorial_solvers.core.model import spins_to_bits
from combinatorial_solvers.tsp import (
    FixedStartOneHotQUBO,
    held_karp_tsp,
)
from combinatorial_solvers.workflows.runtime import (
    dump_json,
)


def _write_one_hot_candidates(
    output: Path,
    encoder: FixedStartOneHotQUBO,
    batches,
    target_length: int,
) -> None:
    fields = [
        "batch_index",
        "chain_index",
        "state_kind",
        "feasible",
        "tour",
        "length",
        "target_met",
        "relative_excess",
    ]
    best_tour_batches: list[np.ndarray] = []
    best_length_batches: list[np.ndarray] = []
    best_feasible_batches: list[np.ndarray] = []
    with (output / "tour_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch_index, batch in enumerate(batches):
            states_by_kind = {
                "best": batch.best_states,
                "final": batch.final_states,
            }
            for state_kind, states in states_by_kind.items():
                certificates = encoder.certify(spins_to_bits(states))
                if state_kind == "best":
                    tours = np.full(
                        (len(certificates), encoder.instance.dimension),
                        -1,
                        dtype=np.int32,
                    )
                    lengths = np.full((len(certificates),), -1, dtype=np.int64)
                    feasible_mask = np.zeros((len(certificates),), dtype=np.bool_)
                    for chain_index, certificate in enumerate(certificates):
                        if certificate is not None and certificate.is_permutation:
                            tours[chain_index] = certificate.tour
                            lengths[chain_index] = int(certificate.length)
                            feasible_mask[chain_index] = True
                    best_tour_batches.append(tours)
                    best_length_batches.append(lengths)
                    best_feasible_batches.append(feasible_mask)
                for chain_index, certificate in enumerate(certificates):
                    feasible = certificate is not None and certificate.is_permutation
                    length = certificate.length if certificate is not None else None
                    writer.writerow(
                        {
                            "batch_index": batch_index,
                            "chain_index": chain_index,
                            "state_kind": state_kind,
                            "feasible": feasible,
                            "tour": ("" if certificate is None else json.dumps(certificate.tour)),
                            "length": "" if length is None else length,
                            "target_met": length is not None and length <= target_length,
                            "relative_excess": (
                                ""
                                if certificate is None or certificate.relative_excess is None
                                else certificate.relative_excess
                            ),
                        }
                    )
    best_tours = np.stack(best_tour_batches)
    best_lengths = np.stack(best_length_batches)
    best_feasible = np.stack(best_feasible_batches)
    np.savez_compressed(
        output / "best_tours.npz",
        best_tours=best_tours,
        best_lengths=best_lengths,
        feasible=best_feasible,
        target_met=best_feasible & (best_lengths <= int(target_length)),
        target_length=np.asarray(target_length, dtype=np.int64),
    )


def _write_one_hot_first_hits(
    output: Path,
    batches,
    first_hit_tours: list[torch.Tensor],
    first_hit_lengths: list[torch.Tensor],
    target_length: int,
) -> None:
    hit_sweeps = torch.stack([batch.first_hit_sweeps for batch in batches]).numpy()
    tours = torch.stack(first_hit_tours).numpy().astype(np.int64, copy=False)
    lengths = torch.stack(first_hit_lengths).numpy().astype(np.int64, copy=False)
    expected = np.arange(tours.shape[2], dtype=np.int64)
    permutation_certificate = np.all(np.sort(tours, axis=2) == expected, axis=2)
    length_certificate = (lengths > 0) & (lengths <= int(target_length))
    certified = permutation_certificate & length_certificate
    if not np.array_equal(certified, hit_sweeps >= 0):
        raise RuntimeError("One-hot first-hit tours do not reproduce first-hit status")
    batch_count, chains = hit_sweeps.shape
    batch_index = np.broadcast_to(
        np.arange(batch_count, dtype=np.int64)[:, None], hit_sweeps.shape
    ).copy()
    chain_index = np.broadcast_to(
        np.arange(chains, dtype=np.int64)[None, :], hit_sweeps.shape
    ).copy()
    np.savez_compressed(
        output / "first_hit_tours.npz",
        first_hit_sweeps=hit_sweeps,
        tours=tours,
        lengths=lengths,
        permutation_certificate=permutation_certificate,
        length_certificate=length_certificate,
        certified=certified,
        batch_index=batch_index,
        chain_index=chain_index,
        target_length=np.asarray(target_length, dtype=np.int64),
    )


def _write_tuning(output: Path, tuning) -> None:
    if tuning is None:
        return
    dump_json(
        output / "replica_tuning.json",
        {
            "selected_replicas": tuning.selected_replicas,
            "trials": [asdict(trial) for trial in tuning.trials],
        },
    )


def _baseline_payload(instance, problem: Mapping[str, Any], population) -> dict[str, Any]:
    best = population[0]
    result: dict[str, Any] = {
        "multistart_nearest_neighbor_two_opt": {
            **asdict(best),
            "method": "multistart_nearest_neighbor_two_opt",
        },
        "multistart_nearest_neighbor_two_opt_population": [
            asdict(candidate) for candidate in population
        ],
    }
    if bool(problem.get("run_held_karp", False)):
        exact = held_karp_tsp(
            instance,
            max_cities=int(problem.get("held_karp_max_cities", 22)),
        )
        result["held_karp"] = asdict(exact)
    return result
