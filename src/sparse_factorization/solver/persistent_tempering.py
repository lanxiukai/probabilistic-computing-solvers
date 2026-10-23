"""Replica exchange around persistent, native-state sparse Gibbs trajectories."""

from __future__ import annotations

import numpy as np
import torch

from sparse_factorization.core.rng import derive_stream_seed

from .persistent_gibbs import PersistentSparseGibbs


def exchange_log_ratio(
    beta_left: np.ndarray,
    beta_right: np.ndarray,
    energy_left: np.ndarray,
    energy_right: np.ndarray,
) -> np.ndarray:
    """Log of the swapped/original product-Boltzmann probability ratio."""
    return (beta_left - beta_right) * (energy_left - energy_right)


def group_first_passage(first: np.ndarray, group_size: int) -> np.ndarray:
    """Count independent replica groups rather than correlated temperature slots."""
    if group_size < 1 or first.ndim != 1 or len(first) % group_size:
        raise ValueError("First-passage slots must form complete groups")
    grouped = first.reshape(-1, group_size)
    sentinel = np.iinfo(np.int64).max
    result = np.where(grouped >= 0, grouped, sentinel).min(axis=1)
    return np.where(result == sentinel, -1, result)


class ReplicaExchange:
    """Disjoint adjacent swaps with a fixed ladder and an independent RNG stream.

    Temperature slots keep their native first-hit and minimum-state observations.
    Only current physical states, energies, and walker labels move. Independent
    inference/optimization units are entire ladders, never individual slots.
    """

    def __init__(self, betas: np.ndarray, groups: int, seed: int) -> None:
        self.betas = np.asarray(betas, dtype=np.float64)
        if (
            self.betas.ndim != 1
            or len(self.betas) < 2
            or groups < 1
            or not np.all(np.isfinite(self.betas))
            or self.betas[0] < 0
            or not np.all(np.diff(self.betas) > 0)
        ):
            raise ValueError("Use a finite strictly increasing nonnegative temperature ladder")
        self.groups = int(groups)
        self.slots = len(self.betas)
        self.seed = derive_stream_seed(seed, "p76_replica_exchange_pcg64_v1")
        self.generator = np.random.Generator(np.random.PCG64(self.seed))
        self.event = 0
        self.draws = 0
        self.attempts = np.zeros(self.slots - 1, dtype=np.int64)
        self.accepted = np.zeros_like(self.attempts)
        self.walkers = np.arange(groups * self.slots).reshape(groups, self.slots)
        self.trip_phase = np.zeros(groups * self.slots, dtype=np.int8)
        self.round_trips = np.zeros(groups * self.slots, dtype=np.int64)
        self._visit_endpoints()

    def _visit_endpoints(self) -> None:
        hot, cold = self.walkers[:, 0], self.walkers[:, -1]
        completed = hot[self.trip_phase[hot] == 2]
        self.round_trips[completed] += 1
        self.trip_phase[hot] = 1
        reached = cold[self.trip_phase[cold] == 1]
        self.trip_phase[reached] = 2

    def step(self, engine: PersistentSparseGibbs) -> None:
        if engine.replicas != self.groups * self.slots:
            raise ValueError("The engine does not match the temperature ladder layout")
        edges = np.arange(self.event % 2, self.slots - 1, 2)
        base = np.arange(self.groups)[:, None] * self.slots
        left = base + edges[None, :]
        right = left + 1
        energies = engine.energies.detach().cpu().double().numpy()
        ratio = exchange_log_ratio(
            self.betas[edges], self.betas[edges + 1], energies[left], energies[right]
        )
        uniforms = self.generator.random(ratio.shape)
        with np.errstate(divide="ignore"):
            accepted = np.log(uniforms) < np.minimum(0.0, ratio)
        permutation = np.arange(engine.replicas)
        chosen_left, chosen_right = left[accepted], right[accepted]
        permutation[chosen_left], permutation[chosen_right] = chosen_right, chosen_left
        index = torch.as_tensor(permutation, dtype=torch.int64, device="cuda")
        engine.states.copy_(engine.states[index])
        engine.energies.copy_(engine.energies[index])
        self.walkers = self.walkers.reshape(-1)[permutation].reshape(self.groups, self.slots)
        self.attempts[edges] += self.groups
        self.accepted[edges] += accepted.sum(axis=0)
        self.draws += uniforms.size
        self.event += 1
        self._visit_endpoints()

    def diagnostics(self) -> dict:
        return {
            "betas": self.betas.tolist(),
            "exchange_events": self.event,
            "rng_seed": self.seed,
            "rng_draws": self.draws,
            "edge_attempts": self.attempts.tolist(),
            "edge_accepted": self.accepted.tolist(),
            "edge_acceptance": np.divide(
                self.accepted,
                self.attempts,
                out=np.zeros_like(self.accepted, dtype=float),
                where=self.attempts > 0,
            ).tolist(),
            "completed_hot_cold_hot_round_trips": int(self.round_trips.sum()),
            "walkers_completing_round_trips": int(np.count_nonzero(self.round_trips)),
            "independent_units": self.groups,
        }
