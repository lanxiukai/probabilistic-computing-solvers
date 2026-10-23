"""Fused color updates with a Torch-computed FP64 probability table."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_fields import CubicFields


def probability_table(beta_values, *, device, metropolis):
    """Evaluate the existing pointwise acceptance formula for every integer field."""
    betas = torch.as_tensor(beta_values, dtype=torch.float64, device=device)
    if betas.ndim != 1 or not betas.numel():
        raise ValueError("A probability table requires a nonempty beta vector")
    fields = torch.arange(-6, 7, dtype=torch.float64, device=device)
    if metropolis:
        return torch.exp(torch.clamp(-2 * betas[:, None] * fields[None, :], max=0))
    return torch.sigmoid(2 * betas[:, None] * fields[None, :])


@triton.jit(
    do_not_specialize=["beta_index"],
    do_not_specialize_on_alignment=["beta_index"],
)
def _updates_kernel(
    states,
    indices,
    neighbors,
    weights,
    uniform,
    probabilities,
    beta_index,
    size: tl.constexpr,
    sites: tl.constexpr,
    degree: tl.constexpr,
    temperatures: tl.constexpr,
    copies: tl.constexpr,
    temperature_ladder: tl.constexpr,
    metropolis: tl.constexpr,
    block: tl.constexpr,
):
    replica = tl.program_id(1)
    site = tl.program_id(0) * block + tl.arange(0, block)
    mask = site < sites
    field = tl.full((block,), 0, tl.int32)
    for neighbor in tl.static_range(degree):
        offset = site * degree + neighbor
        index = tl.load(neighbors + offset, mask, other=0)
        weight = tl.load(weights + offset, mask, other=0)
        # Padding points at site zero, which may belong to the updating color.
        # Mask it out so only opposite-color states are read during mutation.
        spin = tl.load(states + replica * size + index, mask & (weight != 0), other=0).to(tl.int32)
        field += weight * spin
    target = tl.load(indices + site, mask, other=0)
    old = tl.load(states + replica * size + target, mask, other=1).to(tl.int32)
    if temperature_ladder:
        row = (replica // copies) % temperatures
    else:
        row = beta_index
    if metropolis:
        field *= old
    probability = tl.load(probabilities + row * 13 + field + 6, mask, other=0.0)
    random = tl.load(uniform + replica * sites + site, mask, other=0.0)
    accepted = random < probability
    if metropolis:
        updated = tl.where(accepted, -old, old)
    else:
        updated = tl.where(accepted, 1, -1)
    tl.store(states + replica * size + target, updated.to(tl.float64), mask)


class CubicUpdates:
    """Update one independent lattice color with unchanged external random words.

    The solver supplies its canonical cubic coloring and bipolar states. All
    probabilities are computed by Torch in preparation, with the same FP64
    pointwise formulas as the existing path. The kernel uses integer fields,
    a table load, a strict comparison and an exact bipolar state write.
    """

    def __init__(
        self,
        sparse,
        colors,
        beta_values,
        *,
        temperatures,
        copies,
        metropolis,
        temperature_ladder,
    ):
        fields = CubicFields(sparse, colors)
        self.sparse = sparse
        self.cached = [
            (indices.to(torch.int32).contiguous(), *fields.cached[color])
            for color, indices in enumerate(colors)
        ]
        self.probabilities = probability_table(
            beta_values, device=sparse.device, metropolis=metropolis
        ).contiguous()
        self.temperatures = temperatures
        self.copies = copies
        self.metropolis = metropolis
        self.temperature_ladder = temperature_ladder
        if temperature_ladder and self.probabilities.shape[0] != temperatures:
            raise ValueError("A PT probability table must match its temperature ladder")

    def _dispatch(self, spins, color, uniform, beta_index, *, warmup=False):
        sparse = self.sparse
        indices, neighbors, weights = self.cached[color]
        sites, degree = neighbors.shape
        if (
            spins.device != sparse.device
            or spins.dtype != torch.float64
            or spins.shape[-1] != sparse.size
            or not spins.is_contiguous()
            or uniform.device != sparse.device
            or uniform.dtype != torch.float64
            or uniform.shape != (*spins.shape[:-1], sites)
            or not uniform.is_contiguous()
        ):
            raise ValueError("Fused updates require contiguous matching FP64 states and uniforms")
        if not 0 <= beta_index < len(self.probabilities):
            raise ValueError("The probability-table row is out of range")
        grid = (triton.cdiv(sites, 256), spins.numel() // sparse.size)
        return _updates_kernel.run(
            spins,
            indices,
            neighbors,
            weights,
            uniform,
            self.probabilities,
            beta_index,
            size=sparse.size,
            sites=sites,
            degree=degree,
            temperatures=self.temperatures,
            copies=self.copies,
            temperature_ladder=self.temperature_ladder,
            metropolis=self.metropolis,
            block=256,
            num_warps=4,
            grid=grid,
            warmup=warmup,
        )

    def warmup(self, spins):
        """Compile both color shapes without launching or drawing random numbers."""
        for color, (indices, _, _) in enumerate(self.cached):
            uniform = torch.empty(
                (*spins.shape[:-1], indices.numel()), device=spins.device, dtype=spins.dtype
            )
            self._dispatch(spins, color, uniform, 0, warmup=True)

    def __call__(self, spins, color, uniform, beta_index):
        return self._dispatch(spins, color, uniform, beta_index)
