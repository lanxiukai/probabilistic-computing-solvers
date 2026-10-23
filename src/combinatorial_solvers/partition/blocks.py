"""State-independent block banks with conflict-free conditional updates."""

from __future__ import annotations

import numpy as np
import torch

from .model import Graph


def make_banks(graph: Graph, width: int, count: int, seed: int):
    if width not in (2, 4) or graph.size % width or count < 2:
        raise ValueError("Two- or four-site banks must divide the graph")
    rng = np.random.default_rng(seed)
    banks = []
    connectivity = [set() for _ in range(graph.size)]
    for _ in range(count):
        blocks = rng.permutation(graph.size).reshape(-1, width)
        owner = np.empty(graph.size, dtype=np.int64)
        owner[blocks] = np.arange(len(blocks))[:, None]
        conflicts = [set() for _ in blocks]
        for i, j in graph.edges:
            a, b = int(owner[i]), int(owner[j])
            if a != b:
                conflicts[a].add(b)
                conflicts[b].add(a)
        colors = np.full(len(blocks), -1, dtype=np.int64)
        order = sorted(range(len(blocks)), key=lambda i: (-len(conflicts[i]), i))
        for i in order:
            unavailable = {int(colors[j]) for j in conflicts[i] if colors[j] >= 0}
            color = 0
            while color in unavailable:
                color += 1
            colors[i] = color
        for block in blocks:
            for i in block:
                connectivity[i].update(int(j) for j in block if j != i)
        banks.append((blocks, colors))
    seen, pending = {0}, [0]
    while pending:
        new = connectivity[pending.pop()] - seen
        seen.update(new)
        pending.extend(new)
    if len(seen) != graph.size:
        raise ValueError("Block-bank union is disconnected and would conserve extra sectors")
    return banks


def validate_banks(graph, banks):
    for blocks, colors in banks:
        if not np.array_equal(np.sort(blocks.ravel()), np.arange(graph.size)):
            raise ValueError("Bank does not cover each vertex exactly once")
        owner = np.empty(graph.size, dtype=np.int64)
        owner[blocks] = np.arange(len(blocks))[:, None]
        a, b = owner[graph.edges[:, 0]], owner[graph.edges[:, 1]]
        if np.any((a != b) & (colors[a] == colors[b])):
            raise ValueError("Same-color conditional blocks interact")


def prepare_groups(graph, banks, device):
    adjacency = graph.adjacency()
    groups = []
    for blocks, colors in banks:
        bank = []
        for color in range(int(colors.max()) + 1):
            nodes = blocks[colors == color]
            width = nodes.shape[1]
            lists = []
            internal = np.zeros((len(nodes), width, width), dtype=np.float64)
            for b, block in enumerate(nodes):
                positions = {int(v): k for k, v in enumerate(block)}
                row = []
                for a, i in enumerate(block):
                    external = []
                    for j, w in adjacency[i]:
                        if j in positions:
                            internal[b, a, positions[j]] = w
                        else:
                            external.append((j, w))
                    row.append(external)
                lists.append(row)
            degree = max(1, max(len(row) for block in lists for row in block))
            neighbors = np.zeros((len(nodes), width, degree), dtype=np.int64)
            weights = np.zeros_like(neighbors, dtype=np.float64)
            for b, block in enumerate(lists):
                for a, row in enumerate(block):
                    for k, (j, w) in enumerate(row):
                        neighbors[b, a, k], weights[b, a, k] = j, w
            # Small local scores are exact integers in FP64. The full cut is int64.
            bank.append(
                tuple(
                    torch.as_tensor(array, device=device)
                    for array in (nodes, neighbors, weights, internal)
                )
            )
        groups.append(bank)
    return groups


def scores(spins, group, patterns):
    nodes, neighbors, weights, internal = group
    current = spins[:, nodes]
    field = (spins[:, neighbors] * weights).sum(dim=-1)
    local = -(field[..., None, :] * patterns[None, None, :, :]).sum(dim=-1) / 2
    inside = -torch.einsum("pk,bkl,pl->bp", patterns, internal, patterns) / 4
    return current, local + inside[None, :, :]


def update(spins, group, patterns, beta, method, uniform):
    nodes = group[0]
    current, energy = scores(spins, group, patterns)
    allowed = patterns.sum(-1)[None, None, :] == current.sum(-1)[..., None]
    if method == "metropolis":
        if nodes.shape[1] != 2:
            raise ValueError("Metropolis control uses exactly two sites")
        # For opposite signs the only nontrivial conditional proposal is their swap.
        index = ((current + 1) / 2).to(torch.int64)
        index = index[..., 0] + 2 * index[..., 1]
        opposite = 3 - index
        old = energy.gather(-1, index[..., None]).squeeze(-1)
        new = energy.gather(-1, opposite[..., None]).squeeze(-1)
        accept = (current.sum(-1) == 0) & (
            uniform < torch.exp(torch.clamp(-beta * (new - old), max=0))
        )
        selected = torch.where(accept, opposite, index)
    elif method == "greedy":
        masked = energy.masked_fill(~allowed, torch.inf)
        selected = masked.argmin(-1)
        old_index = (
            ((current + 1) / 2).to(torch.int64)
            * (2 ** torch.arange(nodes.shape[1], device=spins.device))
        ).sum(-1)
        old = energy.gather(-1, old_index[..., None]).squeeze(-1)
        selected = torch.where(masked.min(-1).values < old, selected, old_index)
    elif method == "heatbath":
        logits = (-beta * energy).masked_fill(~allowed, -torch.inf)
        probabilities = torch.softmax(logits, dim=-1)
        selected = (uniform[..., None] >= probabilities.cumsum(-1)).sum(-1)
        last_allowed = (allowed * torch.arange(len(patterns), device=spins.device)).max(-1).values
        selected = torch.minimum(selected, last_allowed)
    else:
        raise ValueError("Unknown transition")
    spins[:, nodes] = patterns[selected]
