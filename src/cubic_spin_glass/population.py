"""Fixed-population annealing primitives and resampling evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


def stepped_schedule(beta_levels, cycle_sweeps):
    """Visit a geometric ladder in constant-temperature blocks within each cycle."""
    levels = np.asarray(beta_levels, dtype=np.float64)
    indices = np.arange(cycle_sweeps, dtype=np.int64) * len(levels) // cycle_sweeps
    indices[-1] = len(levels) - 1
    return levels[indices]


def multinomial_resample(spins, energies, beta_delta, generator):
    """Resample independently inside each group using exp(-delta_beta * energy).

    This mutates both states and energies. It does not conserve population energy;
    every returned state is an exact copy of an existing same-group state.
    """
    if not 0 <= beta_delta < math.inf:
        raise ValueError("Population resampling requires a finite nonnegative beta increment")
    groups, size = spins.shape[0], spins.shape[-1]
    states = spins.reshape(groups, -1, size)
    values = energies.reshape(groups, -1)
    relative = values - values.min(dim=1, keepdim=True).values
    probabilities = torch.softmax(-beta_delta * relative, dim=1)
    parents = torch.multinomial(
        probabilities, values.shape[1], replacement=True, generator=generator
    )
    selected = states.gather(1, parents[..., None].expand(-1, -1, size))
    selected_energy = values.gather(1, parents)
    spins.copy_(selected.reshape_as(spins))
    energies.copy_(selected_energy.reshape_as(energies))
    effective_sample_size = 1 / probabilities.square().sum(dim=1)
    return parents, effective_sample_size


def systematic_indices(probabilities, offsets):
    """Map one FP64 offset per group to fixed-population parent indices.

    Inputs are normalized, nonnegative FP64 probabilities and offsets in [0, 1).
    The CDF is made nondecreasing, with an exact endpoint. Equal weights use
    the same linear boundaries as the strata. Each position stays strictly
    below its stratum's upper boundary even when addition rounds upward.
    """
    if (
        probabilities.ndim != 2
        or probabilities.dtype != torch.float64
        or offsets.shape != (probabilities.shape[0], 1)
        or offsets.dtype != torch.float64
        or offsets.device != probabilities.device
        or not probabilities.shape[1]
    ):
        raise ValueError(
            "Systematic indices require matching grouped FP64 probabilities and offsets"
        )
    particles = probabilities.shape[1]
    integer_positions = torch.arange(
        particles + 1, device=probabilities.device, dtype=torch.float64
    )
    boundaries = integer_positions / particles
    cdf = torch.cummax(torch.cumsum(probabilities, dim=1), dim=1).values.clamp(0.0, 1.0)
    equal_weights = (probabilities == probabilities[:, :1]).all(dim=1, keepdim=True)
    cdf = torch.where(equal_weights, boundaries[None, 1:], cdf)
    cdf[:, -1] = 1.0
    positions = (integer_positions[None, :-1] + offsets) / particles
    upper = torch.nextafter(boundaries[1:], torch.zeros_like(boundaries[1:]))
    positions = torch.minimum(positions, upper[None, :]).contiguous()
    parents = torch.searchsorted(cdf.contiguous(), positions, right=True)
    return parents, cdf


def systematic_resample(spins, energies, beta_delta, generator):
    """Resample within each group using one shared offset across its strata."""
    if not 0 <= beta_delta < math.inf:
        raise ValueError("Population resampling requires a finite nonnegative beta increment")
    groups, size = spins.shape[0], spins.shape[-1]
    states = spins.reshape(groups, -1, size)
    values = energies.reshape(groups, -1)
    relative = values - values.min(dim=1, keepdim=True).values
    probabilities = torch.softmax(-beta_delta * relative, dim=1)
    offsets = torch.rand((groups, 1), device=spins.device, dtype=torch.float64, generator=generator)
    parents, cdf = systematic_indices(probabilities, offsets)
    selected = states.gather(1, parents[..., None].expand(-1, -1, size))
    selected_energy = values.gather(1, parents)
    spins.copy_(selected.reshape_as(spins))
    energies.copy_(selected_energy.reshape_as(energies))
    effective_sample_size = 1 / probabilities.square().sum(dim=1)
    return parents, effective_sample_size, offsets, cdf


@dataclass
class PopulationEvidence:
    groups: int
    particles: int
    initial_rng_bytes: int
    restart_sweeps: list = field(default_factory=list)
    restart_rng_states: list = field(default_factory=list)
    restart_energies: list = field(default_factory=list)
    resample_sweeps: list = field(default_factory=list)
    resample_betas: list = field(default_factory=list)
    resample_energies: list = field(default_factory=list)
    resample_parents: list = field(default_factory=list)
    resample_ess: list = field(default_factory=list)
    resampling_scheme: str = "multinomial"
    systematic_offsets: list = field(default_factory=list)
    systematic_cdfs: list = field(default_factory=list)

    def record_restart(self, sweep, before_rng, energies):
        self.restart_sweeps.append(sweep)
        self.restart_rng_states.append(before_rng.cpu().numpy().copy())
        self.restart_energies.append(
            energies.reshape(self.groups, self.particles).cpu().numpy().astype(np.int64)
        )

    def resample(self, sweep, before_beta, beta, spins, energies, generator):
        before = energies.reshape(self.groups, self.particles).cpu().numpy().astype(np.int64)
        if self.resampling_scheme == "systematic":
            parents, ess, offsets, cdf = systematic_resample(
                spins, energies, beta - before_beta, generator
            )
            self.systematic_offsets.append(offsets[:, 0].cpu().numpy().copy())
            self.systematic_cdfs.append(cdf.cpu().numpy().copy())
        else:
            parents, ess = multinomial_resample(spins, energies, beta - before_beta, generator)
        self.resample_sweeps.append(sweep)
        self.resample_betas.append([before_beta, beta])
        self.resample_energies.append(before)
        self.resample_parents.append(parents.cpu().numpy().astype(np.int32))
        self.resample_ess.append(ess.cpu().numpy().copy())

    def arrays(self):
        result = {
            "population_restart_sweeps": np.asarray(self.restart_sweeps, dtype=np.int64),
            "population_restart_rng_states": np.asarray(
                self.restart_rng_states, dtype=np.uint8
            ).reshape(-1, self.initial_rng_bytes),
            "population_restart_energies": np.asarray(
                self.restart_energies, dtype=np.int64
            ).reshape(-1, self.groups, self.particles),
            "population_resample_sweeps": np.asarray(self.resample_sweeps, dtype=np.int64),
            "population_resample_betas": np.asarray(self.resample_betas, dtype=np.float64).reshape(
                -1, 2
            ),
            "population_resample_energies": np.asarray(
                self.resample_energies, dtype=np.int64
            ).reshape(-1, self.groups, self.particles),
            "population_parent_indices": np.asarray(self.resample_parents, dtype=np.int32).reshape(
                -1, self.groups, self.particles
            ),
            "population_effective_sample_size": np.asarray(
                self.resample_ess, dtype=np.float64
            ).reshape(-1, self.groups),
        }
        if self.resampling_scheme == "systematic":
            result["population_systematic_offsets"] = np.asarray(
                self.systematic_offsets, dtype=np.float64
            ).reshape(-1, self.groups)
            result["population_systematic_cdf"] = np.asarray(
                self.systematic_cdfs, dtype=np.float64
            ).reshape(-1, self.groups, self.particles)
        return result
