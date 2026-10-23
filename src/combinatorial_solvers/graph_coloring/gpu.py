"""Persistent one-hot graph coloring with exact single/pair conditional Gibbs."""

from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _fill_weights(
    Parameters,
    ReplicaBeta,
    Weights,
    Stride: tl.constexpr,
    Block: tl.constexpr,
    Tempering: tl.constexpr,
):
    replica = tl.program_id(0)
    difference = tl.arange(0, Block)
    if Tempering:
        beta = tl.load(ReplicaBeta + replica)
    else:
        beta = tl.load(Parameters)
    value = libdevice.exp(-beta * difference.to(tl.float64))
    tl.store(Weights + replica * Stride + difference, value, difference < Stride)


@triton.jit
def _advance(
    X,
    Color,
    Best,
    Counts,
    Cost,
    BestCost,
    Adj,
    Degree,
    Allowed,
    Visited,
    Changed,
    Noise,
    Parameters,
    ReplicaBeta,
    Weights,
    Steps,
    Offset,
    Cycle,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    NN: tl.constexpr,
    KK: tl.constexpr,
    PP: tl.constexpr,
    Chunk: tl.constexpr,
    Width: tl.constexpr,
    Method: tl.constexpr,
    Tempering: tl.constexpr,
    CachedWeights: tl.constexpr,
    WeightStride: tl.constexpr,
):
    replica = tl.program_id(0)
    p = tl.arange(0, PP)
    neighbor_lane = tl.arange(0, D)
    vertex_lane = tl.arange(0, NN)
    color_lane = tl.arange(0, KK)
    low, ratio = tl.load(Parameters), tl.load(Parameters + 1)
    cost = tl.load(Cost + replica)
    best_cost = tl.load(BestCost + replica)
    changed = tl.load(Changed + replica)
    for step in range(Steps):
        u = tl.load(Noise + replica * Chunk + step)
        i = tl.where(
            Offset + step < N,
            (Offset + step) % N,
            tl.minimum((u * N).to(tl.int32), N - 1),
        )
        ci = tl.load(Color + replica * N + i)
        a = p % K
        valid = p < K
        degree_i = tl.load(Degree + i)
        ni = tl.load(Adj + i * D + neighbor_lane, neighbor_lane < degree_i, 0)
        if Width == 2:
            v = tl.load(Noise + (R + replica) * Chunk + step)
            slot = tl.minimum(
                (v * tl.maximum(degree_i, 1)).to(tl.int32), tl.maximum(degree_i - 1, 0)
            )
            j = tl.load(Adj + i * D + slot)
            j = tl.where(degree_i > 0, j, (i + 1) % N)
            edge = (degree_i > 0).to(tl.int32)
            cj = tl.load(Color + replica * N + j)
            b = p // K
            valid = p < K * K
            local = tl.load(Counts + (replica * N + i) * K + a, valid, 0) + tl.load(
                Counts + (replica * N + j) * K + b, valid, 0
            )
            local = (
                local
                - edge * (a == cj).to(tl.int32)
                - edge * (b == ci).to(tl.int32)
                + edge * (a == b).to(tl.int32)
            )
            old_local = (
                tl.load(Counts + (replica * N + i) * K + ci)
                + tl.load(Counts + (replica * N + j) * K + cj)
                - edge * (ci == cj).to(tl.int32)
            )
            valid &= tl.load(Allowed + j * K + b, p < K * K, 0)
            old_pattern = ci + K * cj
        else:
            local = tl.load(Counts + (replica * N + i) * K + a, valid, 0)
            old_local = tl.load(Counts + (replica * N + i) * K + ci)
            old_pattern = ci
        valid &= tl.load(Allowed + i * K + a, p < (K if Width == 1 else K * K), 0)
        if Tempering:
            beta = tl.load(ReplicaBeta + replica)
        else:
            beta = low * libdevice.exp(
                ratio * ((Offset + step) % Cycle).to(tl.float64) / tl.maximum(Cycle - 1, 1)
            )
        u_select = tl.load(Noise + (2 * R + replica) * Chunk + step)
        if Method == 1:
            proposed = tl.minimum(
                (u_select * (K if Width == 1 else K * K)).to(tl.int32),
                (K if Width == 1 else K * K) - 1,
            )
            proposal_cost = tl.sum(tl.where(p == proposed, local, 0), 0)
            possible = tl.sum((valid & (p == proposed)).to(tl.int32), 0) > 0
            u_accept = tl.load(Noise + (3 * R + replica) * Chunk + step)
            if CachedWeights:
                acceptance = tl.load(
                    Weights + replica * WeightStride + tl.maximum(proposal_cost - old_local, 0),
                    possible,
                    0.0,
                )
            else:
                acceptance = libdevice.exp(
                    -beta * tl.maximum(proposal_cost - old_local, 0).to(tl.float64)
                )
            selected = tl.where(
                possible & (u_accept < acceptance),
                proposed,
                old_pattern,
            )
        else:
            minimum = tl.min(tl.where(valid, local, 2147483647), 0)
            if Method == 2:
                weights = (valid & (local == minimum)).to(tl.float64)
            else:
                if CachedWeights:
                    weights = tl.load(
                        Weights + replica * WeightStride + local - minimum, valid, 0.0
                    )
                else:
                    weights = tl.where(
                        valid,
                        libdevice.exp(-beta * (local - minimum).to(tl.float64)),
                        0.0,
                    )
            selected = tl.min(
                tl.where(
                    valid & (tl.cumsum(weights, 0) > u_select * tl.sum(weights, 0)),
                    p,
                    PP,
                ),
                0,
            )
        new_i = selected % K
        selected_cost = tl.sum(tl.where(p == selected, local, 0), 0)
        cost += selected_cost - old_local
        changed += selected != old_pattern
        tl.atomic_add(
            Counts + (replica * N + ni) * K + ci,
            -1,
            (neighbor_lane < degree_i) & (ci != new_i),
        )
        tl.atomic_add(
            Counts + (replica * N + ni) * K + new_i,
            1,
            (neighbor_lane < degree_i) & (ci != new_i),
        )
        tl.store(
            X + (replica * N + i) * K + color_lane,
            (color_lane == new_i).to(tl.int8),
            color_lane < K,
        )
        tl.store(
            Visited + (replica * N + i) * K + color_lane,
            tl.load(Allowed + i * K + color_lane, color_lane < K, 0).to(tl.int8),
            color_lane < K,
        )
        tl.store(Color + replica * N + i, new_i)
        if Width == 2:
            new_j = selected // K
            degree_j = tl.load(Degree + j)
            nj = tl.load(Adj + j * D + neighbor_lane, neighbor_lane < degree_j, 0)
            tl.atomic_add(
                Counts + (replica * N + nj) * K + cj,
                -1,
                (neighbor_lane < degree_j) & (cj != new_j),
            )
            tl.atomic_add(
                Counts + (replica * N + nj) * K + new_j,
                1,
                (neighbor_lane < degree_j) & (cj != new_j),
            )
            tl.store(
                X + (replica * N + j) * K + color_lane,
                (color_lane == new_j).to(tl.int8),
                color_lane < K,
            )
            tl.store(
                Visited + (replica * N + j) * K + color_lane,
                tl.load(Allowed + j * K + color_lane, color_lane < K, 0).to(tl.int8),
                color_lane < K,
            )
            tl.store(Color + replica * N + j, new_j)
        tl.debug_barrier()
        if cost < best_cost:
            best_cost = cost
            current = tl.load(Color + replica * N + vertex_lane, vertex_lane < N, 0)
            tl.store(Best + replica * N + vertex_lane, current, vertex_lane < N)
    tl.store(Cost + replica, cost)
    tl.store(BestCost + replica, best_cost)
    tl.store(Changed + replica, changed)


@triton.jit
def _advance_small_palette(
    X,
    Color,
    Best,
    Counts,
    Cost,
    BestCost,
    Adj,
    Degree,
    Allowed,
    Visited,
    Changed,
    Noise,
    Parameters,
    ReplicaBeta,
    Weights,
    Steps,
    Offset,
    Cycle,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    NN: tl.constexpr,
    KK: tl.constexpr,
    PP: tl.constexpr,
    Chunk: tl.constexpr,
    Width: tl.constexpr,
    Method: tl.constexpr,
    Tempering: tl.constexpr,
    CachedWeights: tl.constexpr,
    WeightStride: tl.constexpr,
):
    replica = tl.program_id(0)
    p = tl.arange(0, PP)
    node_lane = tl.arange(0, 4)
    neighbor_lane = tl.arange(0, D)
    vertex_lane = tl.arange(0, NN)
    color_lane = tl.arange(0, KK)
    low, ratio = tl.load(Parameters), tl.load(Parameters + 1)
    cost = tl.load(Cost + replica)
    best_cost = tl.load(BestCost + replica)
    changed = tl.load(Changed + replica)
    for step in range(Steps):
        u = tl.load(Noise + replica * Chunk + step)
        first = tl.where(
            Offset + step < N,
            (Offset + step) % N,
            tl.minimum((u * N).to(tl.int32), N - 1),
        )
        nodes = tl.where(node_lane == 0, first, -1)
        previous = first
        for position in tl.static_range(1, Width):
            degree = tl.load(Degree + previous)
            u_next = tl.load(Noise + (position * R + replica) * Chunk + step)
            slot = tl.minimum(
                (u_next * tl.maximum(degree, 1)).to(tl.int32), tl.maximum(degree - 1, 0)
            )
            candidate = tl.load(Adj + previous * D + slot)
            candidate = tl.where(degree > 0, candidate, (previous + 1) % N)
            for _ in tl.static_range(Width):
                duplicate = (
                    tl.sum(((node_lane < position) & (nodes == candidate)).to(tl.int32), 0) > 0
                )
                candidate = tl.where(duplicate, (candidate + 1) % N, candidate)
            nodes = tl.where(node_lane == position, candidate, nodes)
            previous = candidate
        local = tl.full((PP,), 0, tl.int32)
        old_local = tl.full((), 0, tl.int32)
        old_pattern = tl.full((), 0, tl.int64)
        valid = p < K**Width
        for position in tl.static_range(Width):
            vertex = tl.sum(tl.where(node_lane == position, nodes, 0), 0)
            old_color = tl.load(Color + replica * N + vertex)
            new_color = (p // (K**position)) % K
            local += tl.load(Counts + (replica * N + vertex) * K + new_color, valid, 0)
            old_local += tl.load(Counts + (replica * N + vertex) * K + old_color)
            old_pattern += old_color * (K**position)
            valid &= tl.load(Allowed + vertex * K + new_color, p < K**Width, 0)
            degree = tl.load(Degree + vertex)
            neighbors = tl.load(Adj + vertex * D + neighbor_lane, neighbor_lane < degree, 0)
            for earlier in tl.static_range(position):
                other = tl.sum(tl.where(node_lane == earlier, nodes, 0), 0)
                old_other = tl.load(Color + replica * N + other)
                new_other = (p // (K**earlier)) % K
                edge = tl.sum(((neighbors == other) & (neighbor_lane < degree)).to(tl.int32), 0)
                local += edge * (
                    (new_color == new_other).to(tl.int32)
                    - (new_color == old_other).to(tl.int32)
                    - (new_other == old_color).to(tl.int32)
                )
                old_local -= edge * (old_color == old_other).to(tl.int32)
        if Tempering:
            beta = tl.load(ReplicaBeta + replica)
        else:
            beta = low * libdevice.exp(
                ratio * ((Offset + step) % Cycle).to(tl.float64) / tl.maximum(Cycle - 1, 1)
            )
        u_select = tl.load(Noise + (Width * R + replica) * Chunk + step)
        if Method == 1:
            proposed = tl.minimum((u_select * K**Width).to(tl.int32), K**Width - 1)
            proposal_cost = tl.sum(tl.where(p == proposed, local, 0), 0)
            possible = tl.sum((valid & (p == proposed)).to(tl.int32), 0) > 0
            u_accept = tl.load(Noise + ((Width + 1) * R + replica) * Chunk + step)
            if CachedWeights:
                acceptance = tl.load(
                    Weights + replica * WeightStride + tl.maximum(proposal_cost - old_local, 0),
                    possible,
                    0.0,
                )
            else:
                acceptance = libdevice.exp(
                    -beta * tl.maximum(proposal_cost - old_local, 0).to(tl.float64)
                )
            selected = tl.where(
                possible & (u_accept < acceptance),
                proposed,
                old_pattern,
            )
        else:
            minimum = tl.min(tl.where(valid, local, 2147483647), 0)
            if Method == 2:
                weights = (valid & (local == minimum)).to(tl.float64)
            else:
                if CachedWeights:
                    weights = tl.load(
                        Weights + replica * WeightStride + local - minimum, valid, 0.0
                    )
                else:
                    weights = tl.where(
                        valid,
                        libdevice.exp(-beta * (local - minimum).to(tl.float64)),
                        0.0,
                    )
            selected = tl.min(
                tl.where(
                    valid & (tl.cumsum(weights, 0) > u_select * tl.sum(weights, 0)),
                    p,
                    PP,
                ),
                0,
            )
        cost += tl.sum(tl.where(p == selected, local, 0), 0) - old_local
        changed += selected != old_pattern
        for position in tl.static_range(Width):
            vertex = tl.sum(tl.where(node_lane == position, nodes, 0), 0)
            old_color = tl.load(Color + replica * N + vertex)
            new_color = (selected // (K**position)) % K
            degree = tl.load(Degree + vertex)
            neighbors = tl.load(Adj + vertex * D + neighbor_lane, neighbor_lane < degree, 0)
            tl.atomic_add(
                Counts + (replica * N + neighbors) * K + old_color,
                -1,
                (neighbor_lane < degree) & (old_color != new_color),
            )
            tl.atomic_add(
                Counts + (replica * N + neighbors) * K + new_color,
                1,
                (neighbor_lane < degree) & (old_color != new_color),
            )
            tl.store(
                X + (replica * N + vertex) * K + color_lane,
                (color_lane == new_color).to(tl.int8),
                color_lane < K,
            )
            tl.store(
                Visited + (replica * N + vertex) * K + color_lane,
                tl.load(Allowed + vertex * K + color_lane, color_lane < K, 0).to(tl.int8),
                color_lane < K,
            )
            tl.store(Color + replica * N + vertex, new_color)
        tl.debug_barrier()
        if cost < best_cost:
            best_cost = cost
            current = tl.load(Color + replica * N + vertex_lane, vertex_lane < N, 0)
            tl.store(Best + replica * N + vertex_lane, current, vertex_lane < N)
    tl.store(Cost + replica, cost)
    tl.store(BestCost + replica, best_cost)
    tl.store(Changed + replica, changed)


class CudaColoring:
    def __init__(self, instance, initial, width=1):
        if width not in (1, 2, 3, 4) or width > instance.size:
            raise ValueError("Coloring block width must fit the graph and lie between one and four")
        if width > 2 and instance.colors > 8:
            raise ValueError("Three/four-vertex blocks require at most eight colors")
        initial = np.asarray(initial, dtype=np.int64)
        cert = [instance.certificate(c) for c in initial]
        if not all(c["valid_assignment"] for c in cert):
            raise ValueError("Initial colors violate the declared lists")
        self.n, self.k, self.r, self.width = (
            instance.size,
            instance.colors,
            len(initial),
            width,
        )
        adjacency = instance.adjacency()
        self.d = triton.next_power_of_2(max(1, max(map(len, adjacency))))
        adj = np.zeros((self.n, self.d), dtype=np.int32)
        counts = np.zeros((self.r, self.n, self.k), dtype=np.int32)
        for i, row in enumerate(adjacency):
            adj[i, : len(row)] = row
            for r, c in enumerate(initial):
                counts[r, i] = np.bincount(c[row], minlength=self.k)
        self.color = torch.tensor(initial, device="cuda")
        self.bits = torch.nn.functional.one_hot(self.color, self.k).to(torch.int8)
        self.best = self.color.clone()
        self.counts = torch.tensor(counts, device="cuda")
        self.cost = torch.tensor([c["conflicts"] for c in cert], dtype=torch.int64, device="cuda")
        self.best_cost = self.cost.clone()
        self.adj = torch.tensor(adj, device="cuda")
        self.degree = torch.tensor(list(map(len, adjacency)), dtype=torch.int32, device="cuda")
        self.allowed = torch.tensor(instance.allowed.copy(), device="cuda")
        self.visited = torch.zeros_like(self.bits)
        self.changed = torch.zeros(self.r, dtype=torch.int64, device="cuda")
        self._parameter_key = None
        self._parameters = None
        self._weights = None
        self._weight_key = None
        self.weight_stride = self.width * self.d + 1

    def clone(self):
        """Clone dynamic state while sharing read-only graph and list inputs."""
        result = CudaColoring.__new__(CudaColoring)
        for key in (
            "n",
            "k",
            "r",
            "width",
            "d",
            "weight_stride",
            "_parameter_key",
            "_weight_key",
            "adj",
            "degree",
            "allowed",
        ):
            setattr(result, key, getattr(self, key))
        for key in (
            "color",
            "bits",
            "best",
            "counts",
            "cost",
            "best_cost",
            "visited",
            "changed",
        ):
            setattr(result, key, getattr(self, key).clone())
        for key in ("_parameters", "_weights"):
            value = getattr(self, key)
            setattr(result, key, value.clone() if value is not None else None)
        return result

    def advance(
        self,
        noise,
        *,
        beta_start=0.1,
        beta_end=8.0,
        cycle_steps=4096,
        offset=0,
        method="heatbath",
        steps=None,
        beta_by_replica=None,
        cache_boltzmann_weights=False,
        temperature_epoch=None,
    ):
        """Advance a fixed noise chunk.

        With mutable replica temperatures, omit temperature_epoch to refresh
        cached weights on every call, or increment it whenever the vector changes.
        """
        if method not in {"heatbath", "metropolis", "plateau"}:
            raise ValueError("Unknown transition method")
        noise = torch.as_tensor(noise, dtype=torch.float64, device="cuda").contiguous()
        if noise.shape[:2] != (max(4, self.width + 2), self.r):
            raise ValueError("Random tensor shape differs from the coloring system")
        parameter_key = (float(beta_start), float(beta_end))
        if parameter_key != self._parameter_key:
            self._parameters = torch.tensor(
                [beta_start, math.log(beta_end / beta_start)],
                dtype=torch.float64,
                device="cuda",
            )
            self._parameter_key = parameter_key
        params = self._parameters
        replica_beta = params
        if beta_by_replica is not None:
            replica_beta = torch.as_tensor(
                beta_by_replica, dtype=torch.float64, device="cuda"
            ).contiguous()
            if replica_beta.shape != (self.r,):
                raise ValueError("Each replica requires one inverse temperature")
        cached_weights = (
            cache_boltzmann_weights
            and method != "plateau"
            and (beta_by_replica is not None or beta_start == beta_end)
        )
        weights = params
        if cached_weights:
            if self._weights is None:
                self._weights = torch.empty(
                    (self.r, self.weight_stride), dtype=torch.float64, device="cuda"
                )
            key = (parameter_key, beta_by_replica is not None, temperature_epoch)
            refresh = key != self._weight_key or (
                beta_by_replica is not None and temperature_epoch is None
            )
            if refresh:
                _fill_weights[(self.r,)](
                    params,
                    replica_beta,
                    self._weights,
                    self.weight_stride,
                    triton.next_power_of_2(self.weight_stride),
                    beta_by_replica is not None,
                )
                self._weight_key = key
            weights = self._weights
        chunk = noise.shape[-1]
        kernel = _advance_small_palette if self.width > 2 else _advance
        kernel[(self.r,)](
            self.bits,
            self.color,
            self.best,
            self.counts,
            self.cost,
            self.best_cost,
            self.adj,
            self.degree,
            self.allowed,
            self.visited,
            self.changed,
            noise,
            params,
            replica_beta,
            weights,
            steps or chunk,
            offset,
            cycle_steps,
            self.n,
            self.k,
            self.r,
            self.d,
            triton.next_power_of_2(self.n),
            triton.next_power_of_2(self.k),
            triton.next_power_of_2(self.k**self.width),
            chunk,
            self.width,
            {"heatbath": 0, "metropolis": 1, "plateau": 2}[method],
            beta_by_replica is not None,
            cached_weights,
            self.weight_stride,
            num_warps=4,
        )

    def state(self):
        return {
            key: getattr(self, value).cpu().numpy()
            for key, value in {
                "bits": "bits",
                "colors": "color",
                "best_colors": "best",
                "counts": "counts",
                "conflicts": "cost",
                "best_conflicts": "best_cost",
                "visited": "visited",
                "changed": "changed",
            }.items()
        }
