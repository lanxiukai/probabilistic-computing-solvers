"""Deterministic graph coloring for exact blocked Gibbs updates."""

from __future__ import annotations

from collections.abc import Sequence
from heapq import heapify, heappop, heappush

import torch

from .edge_model import EdgeIsingModel, edge_terms
from .model import IsingModel


def dsatur_coloring(model: IsingModel, tolerance: float = 0.0) -> tuple[tuple[int, ...], ...]:
    """Color the nonzero coupling graph with deterministic DSATUR tie breaking."""

    adjacency = [set() for _ in range(model.size)]
    if isinstance(model, EdgeIsingModel):
        rows, columns, weights = edge_terms(model)
        selected = weights.abs() > tolerance
        rows, columns = rows[selected], columns[selected]
    else:
        rows, columns = torch.where(torch.triu(model.coupling.abs() > tolerance, diagonal=1))
    for left, right in zip(rows.tolist(), columns.tolist(), strict=True):
        adjacency[left].add(right)
        adjacency[right].add(left)

    colors = [-1] * model.size
    neighbor_colors = [set() for _ in range(model.size)]
    # The original scan orders by saturation, static degree, then node index.
    # Saturations only increase, so stale heap entries can be discarded lazily.
    pending = [(0, -len(adjacency[node]), node) for node in range(model.size)]
    heapify(pending)
    while pending:
        saturation, _, node = heappop(pending)
        if colors[node] >= 0 or -saturation != len(neighbor_colors[node]):
            continue
        unavailable = neighbor_colors[node]
        color = 0
        while color in unavailable:
            color += 1
        colors[node] = color
        for neighbor in adjacency[node]:
            if colors[neighbor] < 0 and color not in neighbor_colors[neighbor]:
                neighbor_colors[neighbor].add(color)
                heappush(
                    pending,
                    (-len(neighbor_colors[neighbor]), -len(adjacency[neighbor]), neighbor),
                )

    blocks: list[list[int]] = [[] for _ in range(max(colors, default=-1) + 1)]
    for node, color in enumerate(colors):
        blocks[color].append(node)
    result = tuple(tuple(block) for block in blocks)
    validate_coloring(model, result, tolerance=tolerance)
    return result


def validate_coloring(
    model: IsingModel,
    colors: Sequence[Sequence[int]],
    tolerance: float = 0.0,
) -> None:
    """Assert that each variable occurs once and no color contains an edge."""

    flattened = [int(node) for block in colors for node in block]
    if sorted(flattened) != list(range(model.size)):
        raise ValueError("Every model variable must occur in exactly one color")
    if isinstance(model, EdgeIsingModel):
        assigned = torch.empty(model.size, dtype=torch.long, device=model.bias.device)
        for color, nodes in enumerate(colors):
            assigned[list(nodes)] = color
        left, right, weights = edge_terms(model)
        if bool(((assigned[left] == assigned[right]) & (weights.abs() > tolerance)).any()):
            raise ValueError("A color contains coupled variables")
        return
    for block in colors:
        if len(block) < 2:
            continue
        index = torch.tensor(block, dtype=torch.long, device=model.coupling.device)
        submatrix = model.coupling.index_select(0, index).index_select(1, index)
        if torch.any(submatrix.abs() > tolerance):
            raise ValueError("A color contains coupled variables")
