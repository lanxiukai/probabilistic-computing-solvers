"""Sparse directed assignment QUBO and explicit classical tour readout.

One binary node represents each allowed directed arc. Row and column penalties
are pairwise; connectivity is NOT embedded. A classical sparse assignment and
cycle splice readout is charged separately. Candidate restriction need not
contain the original TSP optimum. This is a hybrid optimization component.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from .instance import TSPInstance, certify_tour


@dataclass(frozen=True)
class EdgeAssignment:
    cities: int
    source: np.ndarray
    destination: np.ndarray
    cost: np.ndarray
    colors: tuple[np.ndarray, ...]
    scale: float
    penalty: float

    @property
    def bits(self) -> int:
        return len(self.source)

    def counts(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        states = np.atleast_2d(states)
        if states.shape[1] != self.bits or not np.isin(states, [0, 1]).all():
            raise ValueError("States must be binary and match the candidate graph")
        outgoing = np.stack(
            [np.bincount(self.source, weights=x, minlength=self.cities) for x in states]
        ).astype(np.int32)
        incoming = np.stack(
            [np.bincount(self.destination, weights=x, minlength=self.cities) for x in states]
        ).astype(np.int32)
        return outgoing, incoming

    def energies(self, states: np.ndarray) -> np.ndarray:
        outgoing, incoming = self.counts(states)
        return np.atleast_2d(states) @ (self.cost / self.scale) + self.penalty * (
            ((outgoing - 1) ** 2).sum(axis=1) + ((incoming - 1) ** 2).sum(axis=1)
        )

    def encode(self, tour) -> np.ndarray:
        successor = np.empty(self.cities, dtype=np.int32)
        successor[np.asarray(tour)] = np.roll(tour, -1)
        state = (successor[self.source] == self.destination).astype(np.int8)
        if int(state.sum()) != self.cities:
            raise ValueError("The candidate graph does not contain the supplied tour")
        return state

    def metadata(self) -> dict:
        rows = np.bincount(self.source, minlength=self.cities)
        cols = np.bincount(self.destination, minlength=self.cities)
        pairs = int((rows * (rows - 1) // 2).sum() + (cols * (cols - 1) // 2).sum())
        return {
            "cities": self.cities,
            "binary_nodes_per_replica": self.bits,
            "auxiliary_binary_nodes": 0,
            "pairwise_couplers": pairs,
            "maximum_coupling_degree": int((rows[self.source] + cols[self.destination] - 2).max()),
            "independent_colors": len(self.colors),
            "distance_scale": self.scale,
            "constraint_penalty": self.penalty,
            "connectivity_embedded": False,
            "full_tsp_encoding": False,
            "physical_devices_measured": 0,
            "mapping_assumption": (
                "One stochastic binary node per candidate arc with native sparse couplings; "
                "digital controller and memory excluded"
            ),
        }


def color_arcs(source, destination, n: int) -> tuple[np.ndarray, ...]:
    """A deterministic proper coloring of the bipartite incidence graph."""
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    groups: list[list[int]] = []
    for index, (left, right) in enumerate(zip(source, destination, strict=True)):
        occupied = row_used[left] | col_used[right]
        color = 0
        while color in occupied:
            color += 1
        if color == len(groups):
            groups.append([])
        groups[color].append(index)
        row_used[left].add(color)
        col_used[right].add(color)
    return tuple(np.asarray(group, dtype=np.int32) for group in groups)


def build_assignment(
    instance: TSPInstance, tours, *, neighbors: int = 16, penalty: float = 2
) -> EdgeAssignment:
    if not 1 <= neighbors < instance.dimension or not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Invalid candidate count or penalty")
    n = instance.dimension
    arcs = set()
    for left in range(n):
        ordered = np.argsort(instance.distance[left], kind="stable")
        arcs.update((left, int(right)) for right in ordered[ordered != left][:neighbors])
    for tour in tours:
        if not certify_tour(instance, tour).is_permutation:
            raise ValueError("Candidate seed must be a complete tour")
        for a, b in zip(tour, np.roll(tour, -1), strict=True):
            arcs.add((int(a), int(b)))
            arcs.add((int(b), int(a)))
    ordered = sorted(arcs)
    source, destination = np.asarray(ordered, dtype=np.int32).T
    costs = instance.distance[source, destination].copy()
    positive = costs[costs > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    return EdgeAssignment(
        n, source, destination, costs, color_arcs(source, destination, n), scale, float(penalty)
    )


def reference_color_update(
    model: EdgeAssignment, states: np.ndarray, color: int, beta, uniform: np.ndarray
) -> np.ndarray:
    """Simultaneously update independent bits using their exact conditional field."""
    result = np.atleast_2d(states).copy()
    outgoing, incoming = model.counts(result)
    ids = model.colors[color]
    before = result[:, ids]
    delta = model.cost[ids] / model.scale + 2 * model.penalty * (
        outgoing[:, model.source[ids]] + incoming[:, model.destination[ids]] - 2 * before - 1
    )
    exponent = np.clip(np.asarray(beta).reshape(-1, 1) * delta, -700, 700)
    result[:, ids] = uniform < 1 / (1 + np.exp(exponent))
    return result


def assignment_readout(model: EdgeAssignment, state=None, *, bias: float = 0) -> np.ndarray:
    """Exact classical assignment of linear readout scores, not the TSP."""
    if bias < 0 or not np.isfinite(bias):
        raise ValueError("Readout bias must be nonnegative and finite")
    costs = model.cost.astype(np.float64) + 1
    if state is not None:
        x = np.asarray(state)
        if x.shape != (model.bits,) or not np.isin(x, [0, 1]).all():
            raise ValueError("Invalid binary readout state")
        costs += bias * model.scale * (1 - x)
    graph = csr_matrix(
        (costs, (model.source, model.destination)), shape=(model.cities, model.cities)
    )
    rows, columns = min_weight_full_bipartite_matching(graph)
    if not np.array_equal(rows, np.arange(model.cities)):
        raise ValueError("The candidate graph has no complete assignment")
    return columns.astype(np.int32)


def successor_cycles(successor: np.ndarray) -> list[list[int]]:
    successor = np.asarray(successor)
    n = len(successor)
    if sorted(successor.tolist()) != list(range(n)) or np.any(successor == np.arange(n)):
        raise ValueError("Readout must be a permutation without self loops")
    seen = np.zeros(n, dtype=bool)
    cycles = []
    for start in range(n):
        if seen[start]:
            continue
        cycle, city = [], start
        while not seen[city]:
            seen[city] = True
            cycle.append(int(city))
            city = successor[city]
        cycles.append(cycle)
    return cycles


def splice_cycles(instance: TSPInstance, successor: np.ndarray) -> tuple[tuple[int, ...], int]:
    """Merge the smallest cycle with its cheapest cross-cycle successor swap.

    New arcs may leave the candidate graph. This is explicit classical repair,
    never a claim about a sampled state's feasibility or its energy.
    """
    successor = successor.copy()
    cycles = successor_cycles(successor)
    initial_cycles = len(cycles)
    members = {i: np.asarray(cycle, dtype=np.int32) for i, cycle in enumerate(cycles)}
    label = np.empty(instance.dimension, dtype=np.int32)
    for key, nodes in members.items():
        label[nodes] = key
    heap = [(len(nodes), key) for key, nodes in members.items()]
    heapq.heapify(heap)
    while len(members) > 1:
        size, key = heapq.heappop(heap)
        if key not in members or len(members[key]) != size:
            continue
        left = members[key]
        right = np.flatnonzero(label != key)
        a, b = left[:, None], right[None, :]
        delta = (
            instance.distance[a, successor[b]]
            + instance.distance[b, successor[a]]
            - instance.distance[a, successor[a]]
            - instance.distance[b, successor[b]]
        )
        i, j = np.unravel_index(int(np.argmin(delta)), delta.shape)
        u, v = int(left[i]), int(right[j])
        successor[u], successor[v] = successor[v], successor[u]
        other = int(label[v])
        label[left] = other
        members[other] = np.concatenate((members[other], left))
        del members[key]
        heapq.heappush(heap, (len(members[other]), other))
    final = successor_cycles(successor)
    if len(final) != 1:
        raise AssertionError("Cycle splicing failed")
    tour = tuple(final[0])
    if not certify_tour(instance, tour).is_permutation:
        raise AssertionError("Invalid repaired tour")
    return tour, initial_cycles
