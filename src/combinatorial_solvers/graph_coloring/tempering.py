"""Exact adjacent-temperature exchanges without moving persistent binary states."""

from __future__ import annotations

import numpy as np


def exchange_ranks(costs, ranks, beta_grid, ladder_size, parity, uniforms):
    """Return temperature labels after independent Metropolis exchange proposals.

    The persistent chain index stays fixed. A label exchange is equivalent to a
    state exchange between fixed-temperature slots. Different ladders never mix.
    """
    costs = np.asarray(costs)
    ranks = np.asarray(ranks)
    beta_grid = np.asarray(beta_grid)
    uniforms = np.asarray(uniforms)
    if ladder_size < 2 or ladder_size % 2 or len(costs) % ladder_size:
        raise ValueError("Replica ladders require an even size dividing the chain count")
    if ranks.shape != costs.shape or ranks.ndim != 1 or beta_grid.shape != (ladder_size,):
        raise ValueError("Temperature labels and chain energies have incompatible shapes")
    ladders = len(costs) // ladder_size
    if parity not in (0, 1) or uniforms.shape != (ladders, ladder_size // 2):
        raise ValueError("Exchange parity or uniform draw count is invalid")
    if not np.all((uniforms > 0) & (uniforms < 1)):
        raise ValueError("Exchange uniforms must lie strictly between zero and one")
    result = ranks.copy()
    accepted = np.zeros_like(uniforms, dtype=bool)
    for ladder in range(ladders):
        base = ladder * ladder_size
        labels = ranks[base : base + ladder_size]
        if not np.array_equal(np.sort(labels), np.arange(ladder_size)):
            raise ValueError("Each ladder must contain every temperature exactly once")
        slots = np.argsort(labels)
        for draw, rank in enumerate(range(parity, ladder_size - 1, 2)):
            i, j = base + slots[rank], base + slots[rank + 1]
            log_ratio = (beta_grid[rank] - beta_grid[rank + 1]) * (costs[i] - costs[j])
            if np.log(uniforms[ladder, draw]) < min(0.0, log_ratio):
                result[i], result[j] = result[j], result[i]
                accepted[ladder, draw] = True
    return result, accepted
