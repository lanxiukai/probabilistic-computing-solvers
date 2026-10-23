"""Conflict-free factor updates, native parity fields and a response-memory control."""

from __future__ import annotations

import numpy as np
import torch

from .model import SCALE


class Factors:
    def __init__(self, instance, representation, device):
        code = instance.code
        self.code, self.representation, self.device = code, representation, torch.device(device)
        if representation not in ("native", "quadratic"):
            raise ValueError("Unknown parity representation")
        rows = [list(map(int, row)) for row in code.rows]
        coefficients = [[1] * len(row) for row in rows]
        self.size = code.size
        if representation == "quadratic":
            if any(len(row) != 6 for row in rows):
                raise ValueError("The two-slack-bit model requires degree-six checks")
            self.size += 2 * code.checks
            for a, row in enumerate(rows):
                row.extend([code.size + 2 * a, code.size + 2 * a + 1])
                coefficients[a].extend([-2, -4])
        self.rows = rows
        self.coefficients = coefficients
        neighbors = [[] for _ in range(self.size)]
        conflict = [set() for _ in range(self.size)]
        for a, (row, coeff) in enumerate(zip(rows, coefficients, strict=True)):
            for i, c in zip(row, coeff, strict=True):
                neighbors[i].append((a, c))
                conflict[i].update(j for j in row if j != i)
        colors = np.full(self.size, -1, dtype=np.int64)
        for i in sorted(range(self.size), key=lambda v: (-len(conflict[v]), v)):
            occupied = {int(colors[j]) for j in conflict[i] if colors[j] >= 0}
            color = 0
            while color in occupied:
                color += 1
            colors[i] = color
        if any(len(set(colors[row])) != len(row) for row in rows):
            raise AssertionError("Conditional update colors share a factor")
        self.colors = colors
        degree = max(len(row) for row in neighbors)
        adjacency = np.full((self.size, degree), code.checks, dtype=np.int64)
        coeffs = np.zeros_like(adjacency)
        for i, row in enumerate(neighbors):
            for j, (a, c) in enumerate(row):
                adjacency[i, j], coeffs[i, j] = a, c
        self.adjacency = torch.as_tensor(adjacency, device=self.device)
        self.coeffs = torch.as_tensor(coeffs, device=self.device)
        self.groups = []
        for color in range(int(colors.max()) + 1):
            nodes = np.flatnonzero(colors == color)
            check_ids, offsets, edge_coeffs = [], [], []
            for offset, i in enumerate(nodes):
                for a, c in neighbors[i]:
                    check_ids.append(a)
                    offsets.append(offset)
                    edge_coeffs.append(c)
            if len(check_ids) != len(set(check_ids)):
                raise AssertionError("A parallel color has repeated factor writes")
            self.groups.append(
                tuple(
                    torch.as_tensor(a, dtype=torch.int64, device=self.device)
                    for a in (nodes, check_ids, offsets, edge_coeffs)
                )
            )
        width = max(map(len, rows))
        row_nodes = np.full((len(rows), width), self.size, dtype=np.int64)
        row_coeffs = np.zeros_like(row_nodes)
        for a, row in enumerate(rows):
            row_nodes[a, : len(row)] = row
            row_coeffs[a, : len(row)] = coefficients[a]
        self.row_nodes = torch.as_tensor(row_nodes, device=self.device)
        self.row_coeffs = torch.as_tensor(row_coeffs, device=self.device)
        costs = np.zeros((instance.frames, self.size), dtype=np.int64)
        costs[:, : code.size] = instance.costs
        self.costs = torch.as_tensor(costs[:, None, :], device=self.device)
        self.targets = torch.as_tensor(instance.targets[:, None, :], device=self.device)
        self.auxiliary_groups = []
        if representation == "quadratic":
            checks = torch.arange(code.checks, device=self.device)
            for k, coefficient in enumerate((-2, -4)):
                self.auxiliary_groups.append(
                    (
                        code.size + 2 * checks + k,
                        checks,
                        checks,
                        torch.full_like(checks, coefficient),
                    )
                )

    def cache(self, bits):
        zero = torch.zeros((*bits.shape[:-1], 1), dtype=bits.dtype, device=self.device)
        extended = torch.cat([bits, zero], dim=-1)
        local = extended[..., self.row_nodes].to(torch.int64)
        if self.representation == "native":
            cache = (1 - 2 * local).prod(-1) * (1 - 2 * self.targets)
        else:
            cache = (local * self.row_coeffs).sum(-1) - self.targets
        return torch.cat([cache, zero.to(torch.int64)], dim=-1)

    def violations(self, cache):
        if self.representation == "native":
            return (cache[..., :-1] < 0).sum(-1)
        return (cache[..., :-1] != 0).sum(-1)

    def cost(self, bits):
        return (bits * self.costs).sum(-1)

    def flip_delta(self, bits, cache, nodes, penalty):
        direction = 1 - 2 * bits[..., nodes].to(torch.int64)
        local = cache[..., self.adjacency[nodes]]
        if self.representation == "native":
            return direction * self.costs[..., nodes] + penalty * local.sum(-1)
        coeffs = self.coeffs[nodes]
        return direction * self.costs[..., nodes] + penalty * (
            direction * (2 * local * coeffs).sum(-1) + (coeffs * coeffs).sum(-1)
        )

    def colored_update(self, bits, cache, cost, group, beta, penalty, uniform):
        nodes, check_ids, offsets, coeffs = group
        delta = self.flip_delta(bits, cache, nodes, penalty)
        flip = uniform < torch.sigmoid(-beta * delta.to(torch.float64) / SCALE)
        old = bits[..., nodes]
        change = (1 - 2 * old.to(torch.int64)) * flip
        bits[..., nodes] = old ^ flip.to(torch.int8)
        cost += (change * self.costs[..., nodes]).sum(-1)
        if self.representation == "native":
            cache[..., check_ids] *= 1 - 2 * flip[..., offsets].to(torch.int64)
        else:
            cache[..., check_ids] += change[..., offsets] * coeffs
        return flip

    def parallel_update(
        self, bits, cache, cost, memory, beta, penalty, strength, gain_only, uniform, active
    ):
        if self.representation != "native":
            raise ValueError("Response-memory dynamics use native parity factors")
        spins = 1 - 2 * bits.to(torch.int64)
        drive = self.costs + penalty * cache[..., self.adjacency].sum(-1) * spins
        response = torch.tanh(beta * drive.to(torch.float64) / (2 * SCALE))
        decision = (1 + strength) * response if gain_only else response + strength * memory
        proposed = (uniform >= (1 + decision) / 2).to(torch.int8)
        updated = torch.where(active, proposed, bits)
        cost += ((updated.to(torch.int64) - bits) * self.costs).sum(-1)
        changed = updated != bits
        bits.copy_(updated)
        memory.copy_(torch.where(active, response, memory))
        cache.copy_(self.cache(bits))
        return changed

    def auxiliary_pair_update(self, bits, cache, beta, penalty, uniform):
        if self.representation != "quadratic":
            raise ValueError("Joint slack updates require the quadratic representation")
        n = self.code.size
        old = bits[..., n:].clone()
        base = cache[..., :-1] + 2 * old[..., 0::2] + 4 * old[..., 1::2]
        values = torch.arange(4, device=self.device)
        residuals = base[..., None] - 2 * values
        probabilities = torch.softmax(
            -beta * penalty * residuals.to(torch.float64).square() / SCALE, dim=-1
        )
        selected = (uniform[..., None] >= probabilities.cumsum(-1)).sum(-1).clamp(max=3)
        bits[..., n::2] = (selected & 1).to(torch.int8)
        bits[..., n + 1 :: 2] = ((selected >> 1) & 1).to(torch.int8)
        cache[..., :-1] = base - 2 * selected
        return bits[..., n:] != old
