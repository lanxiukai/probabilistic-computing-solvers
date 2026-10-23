"""Static edge-block Gibbs and symmetric joint Metropolis updates for MWIS."""

from __future__ import annotations

import numpy as np
import torch

from combinatorial_solvers.core.coloring import dsatur_coloring
from combinatorial_solvers.core.edge_model import EdgeIsingModel

from .model import Instance


def partition(instance: Instance):
    """Build a deterministic matching, then color the contracted interaction graph."""
    neighbors = instance.adjacency()
    assigned = np.full(instance.size, -1, dtype=np.int64)
    groups = []
    order = sorted(range(instance.size), key=lambda i: (-len(neighbors[i]), i))
    for a in order:
        if assigned[a] >= 0:
            continue
        candidates = [b for b in neighbors[a] if assigned[b] < 0]
        b = min(candidates) if candidates else a
        assigned[a] = len(groups)
        assigned[b] = len(groups)
        groups.append((a, b))
    pairs = np.asarray(groups, dtype=np.int64)
    edges = set()
    for a, b in instance.edges:
        left, right = sorted((int(assigned[a]), int(assigned[b])))
        if left != right:
            edges.add((left, right))
    contracted = np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)
    graph = EdgeIsingModel(
        torch.tensor(contracted[:, 0]),
        torch.tensor(contracted[:, 1]),
        torch.ones(len(contracted), dtype=torch.float64),
        torch.zeros(len(groups), dtype=torch.float64),
    )
    colors = dsatur_coloring(graph)
    block_colors = np.empty(len(groups), dtype=np.int64)
    for color, indices in enumerate(colors):
        block_colors[list(indices)] = color
    for a, b in instance.edges:
        if assigned[a] != assigned[b] and block_colors[assigned[a]] == block_colors[assigned[b]]:
            raise AssertionError("Two simultaneous blocks have a coupling")
    return pairs, assigned, block_colors, colors


def prepare(instance: Instance, device, dtype):
    pairs, assigned, block_colors, colors = partition(instance)
    adjacency = instance.adjacency()
    result = []
    for indices in colors:
        selected = pairs[list(indices)]
        a, b = selected[:, 0], selected[:, 1]
        valid_b = a != b

        def exterior(nodes, other_nodes, active):
            rows = [
                [x for x in adjacency[node] if x != other] if enabled else []
                for node, other, enabled in zip(nodes, other_nodes, active, strict=True)
            ]
            width = max(map(len, rows), default=0)
            neighbors = np.zeros((len(rows), width), dtype=np.int64)
            mask = np.zeros((len(rows), width), dtype=np.int8)
            for row, values in enumerate(rows):
                neighbors[row, : len(values)] = values
                mask[row, : len(values)] = 1
            return torch.tensor(neighbors, device=device), torch.tensor(
                mask, device=device, dtype=dtype
            )

        na, ma = exterior(a, b, np.ones(len(a), dtype=bool))
        nb, mb = exterior(b, a, valid_b)
        result.append(
            {
                "a": torch.tensor(a, device=device),
                "b": torch.tensor(b, device=device),
                "valid_b": torch.tensor(valid_b, device=device),
                "na": na,
                "ma": ma,
                "nb": nb,
                "mb": mb,
                "wa": torch.tensor(instance.weights[a], dtype=dtype, device=device),
                "wb": torch.tensor(instance.weights[b] * valid_b, dtype=dtype, device=device),
            }
        )
    return result, pairs, assigned, block_colors


def gibbs_probabilities(da, db, coupling, beta_over_scale):
    """Marginal P(a=1), and conditional P(b=1|a) from all four energies."""
    zero = torch.zeros_like(db)
    logz0 = torch.logaddexp(zero, -beta_over_scale * db)
    logz1 = -beta_over_scale * da + torch.logaddexp(zero, -beta_over_scale * (db + coupling))
    pa = torch.sigmoid(logz1 - logz0)
    pb0 = torch.sigmoid(-beta_over_scale * db)
    pb1 = torch.sigmoid(-beta_over_scale * (db + coupling))
    return pa, pb0, pb1


def update(bits, block, penalty, beta_over_scale, uniforms, method):
    a, b, valid = block["a"], block["b"], block["valid_b"]
    da = -block["wa"] + penalty * (bits[:, block["na"]] * block["ma"]).sum(-1)
    db = -block["wb"] + penalty * (bits[:, block["nb"]] * block["mb"]).sum(-1)
    coupling = penalty * valid.to(bits.dtype)
    if method == "pair-gibbs":
        pa, pb0, pb1 = gibbs_probabilities(da, db, coupling, beta_over_scale)
        xa = uniforms[..., 0] < pa
        xb = (uniforms[..., 1] < torch.where(xa, pb1, pb0)) & valid
        new_a, new_b = xa.to(bits.dtype), xb.to(bits.dtype)
    elif method == "pair-metropolis":
        new_a = (uniforms[..., 0] < 0.5).to(bits.dtype)
        new_b = ((uniforms[..., 1] < 0.5) & valid).to(bits.dtype)
        old_a, old_b = bits[:, a], bits[:, b] * valid
        delta = (
            da * (new_a - old_a) + db * (new_b - old_b) + coupling * (new_a * new_b - old_a * old_b)
        )
        accepted = uniforms[..., 2] < torch.exp(torch.clamp(-beta_over_scale * delta, max=0))
        new_a = torch.where(accepted, new_a, old_a)
        new_b = torch.where(accepted, new_b, old_b)
    else:
        raise ValueError("Unknown pair update")
    # Singleton blocks use b=a. Preserve the sampled a during the second assignment.
    bits[:, a] = new_a
    bits[:, b] = torch.where(valid, new_b, new_a)
