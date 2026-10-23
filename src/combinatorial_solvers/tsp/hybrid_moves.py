"""Alternating-cycle proposals and explicit subtour repair for symmetric TSP.

Each bit selects a complete AB-cycle from two certified parent tours. Every
selection preserves degree two, but it may create several subtours. Connectivity
is evaluated explicitly; this is not a pairwise QUBO or a full-TSP Gibbs kernel.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from random import Random

import numpy as np

from .instance import TSPInstance, certify_tour

Edge = tuple[int, int]
Tour = tuple[int, ...]


def edge(left: int, right: int) -> Edge:
    return (left, right) if left < right else (right, left)


def canonical_tour(instance: TSPInstance, tour: Sequence[int] | np.ndarray) -> Tour:
    certificate = certify_tour(instance, tour)
    if not certificate.is_permutation:
        raise ValueError("An elite tour must contain every city exactly once")
    ordered = certificate.tour
    pivot = ordered.index(0)
    rotated = ordered[pivot:] + ordered[:pivot]
    return min(rotated, (0, *reversed(rotated[1:])))


def tour_edges(tour: Tour) -> frozenset[Edge]:
    return frozenset(edge(city, tour[(i + 1) % len(tour)]) for i, city in enumerate(tour))


@dataclass(frozen=True)
class ABCycle:
    removed: tuple[Edge, ...]
    added: tuple[Edge, ...]


def alternating_cycles(parent: Tour, donor: Tour, rng: Random) -> tuple[ABCycle, ...]:
    """Pair red/blue incidences at every vertex, then trace edge-disjoint cycles."""
    if len(parent) < 4 or sorted(parent) != list(range(len(parent))):
        raise ValueError(
            "AB-cycle construction requires a certified tour with at least four cities"
        )
    if sorted(donor) != sorted(parent):
        raise ValueError("AB-cycle parents must be permutations of the same cities")
    red = tour_edges(parent) - tour_edges(donor)
    blue = tour_edges(donor) - tour_edges(parent)
    red_at: list[list[Edge]] = [[] for _ in parent]
    blue_at: list[list[Edge]] = [[] for _ in parent]
    for color, incidence in ((red, red_at), (blue, blue_at)):
        for bond in sorted(color):
            for city in bond:
                incidence[city].append(bond)
    paired: dict[tuple[Edge, int], Edge] = {}
    for city, red_edges in enumerate(red_at):
        blue_edges = blue_at[city]
        if len(red_edges) != len(blue_edges):
            raise AssertionError("Parent symmetric difference is not color balanced")
        rng.shuffle(blue_edges)
        for red_edge, blue_edge in zip(red_edges, blue_edges, strict=True):
            paired[red_edge, city] = blue_edge
            paired[blue_edge, city] = red_edge

    unseen = set(red | blue)
    cycles = []
    while unseen:
        start = min(unseen)
        current, city = start, start[0]
        removed, added = [], []
        while True:
            if current not in unseen:
                raise AssertionError("Alternating traversal reused an edge")
            unseen.remove(current)
            (removed if current in red else added).append(current)
            city = current[1] if city == current[0] else current[0]
            following = paired[current, city]
            if following == start:
                if city != start[0]:
                    raise AssertionError(
                        "Alternating cycle closed with an inconsistent orientation"
                    )
                break
            current = following
        if len(removed) != len(added):
            raise AssertionError("An AB-cycle must preserve the edge count")
        cycles.append(ABCycle(tuple(sorted(removed)), tuple(sorted(added))))
    return tuple(cycles)


def edge_frequencies(elites: Sequence[Tour]) -> dict[Edge, float]:
    if not elites:
        raise ValueError("At least one elite tour is required")
    counts = Counter(bond for tour in elites for bond in tour_edges(tour))
    return {bond: count / len(elites) for bond, count in counts.items()}


def nearest_candidate_edges(
    instance: TSPInstance, neighbors: int, check_budget: Callable[[], None]
) -> frozenset[Edge]:
    """Build a deterministic nearest-edge list using the original integer costs."""
    candidates: set[Edge] = set()
    for city in range(instance.dimension):
        check_budget()
        ordered = sorted(
            (other for other in range(instance.dimension) if other != city),
            key=lambda other: (int(instance.distance[city, other]), other),
        )
        candidates.update(edge(city, other) for other in ordered[:neighbors])
    return frozenset(candidates)


def select_uncertain_cycles(
    instance: TSPInstance,
    cycles: Sequence[ABCycle],
    frequencies: dict[Edge, float],
    candidates: frozenset[Edge],
    maximum: int,
    rng: Random,
) -> tuple[ABCycle, ...]:
    """Prefer elite edge disagreement, candidate-supported additions, and lower deltas."""
    ranked = []
    for cycle in cycles:
        bonds = cycle.removed + cycle.added
        uncertainty = sum(
            4.0 * frequencies.get(bond, 0.0) * (1.0 - frequencies.get(bond, 0.0)) for bond in bonds
        ) / len(bonds)
        support = sum(bond in candidates for bond in cycle.added) / len(cycle.added)
        delta = sum(int(instance.distance[a, b]) for a, b in cycle.added) - sum(
            int(instance.distance[a, b]) for a, b in cycle.removed
        )
        ranked.append(((-uncertainty, -support, delta, rng.random()), cycle))
    ranked.sort(key=lambda item: item[0])
    return tuple(cycle for _, cycle in ranked[:maximum])


def selected_edges(base: frozenset[Edge], cycles: Sequence[ABCycle], mask: int) -> set[Edge]:
    if mask < 0 or mask >= 1 << len(cycles):
        raise ValueError("Cycle-selection mask is out of range")
    bonds = set(base)
    for index, cycle in enumerate(cycles):
        if mask & (1 << index):
            if not set(cycle.removed) <= bonds:
                raise AssertionError("Selected AB-cycles do not have disjoint removed edges")
            bonds.difference_update(cycle.removed)
            bonds.update(cycle.added)
    return bonds


def factor_components(cities: int, bonds: set[Edge]) -> tuple[Tour, ...]:
    """Validate a simple two-factor and return its cycles in deterministic order."""
    if len(bonds) != cities:
        raise ValueError("A two-factor must have one edge per city")
    adjacent: list[list[int]] = [[] for _ in range(cities)]
    for left, right in sorted(bonds):
        if not 0 <= left < right < cities:
            raise ValueError("Two-factor edges must have distinct, canonical endpoints")
        adjacent[left].append(right)
        adjacent[right].append(left)
    if any(len(neighbors) != 2 for neighbors in adjacent):
        raise ValueError("A two-factor must have degree two at every city")
    unseen = set(range(cities))
    components = []
    while unseen:
        start = min(unseen)
        previous, city = -1, start
        component = []
        while True:
            if city not in unseen:
                raise ValueError("Two-factor traversal revisited a city before closing")
            unseen.remove(city)
            component.append(city)
            neighbors = adjacent[city]
            following = neighbors[0] if neighbors[0] != previous else neighbors[1]
            if following == start:
                break
            previous, city = city, following
        components.append(tuple(component))
    return tuple(sorted(components, key=lambda tour: (len(tour), tour)))


def repair_subtours(
    instance: TSPInstance,
    bonds: set[Edge],
    candidates: frozenset[Edge],
    check_budget: Callable[[], None],
) -> tuple[Tour, tuple[dict, ...]]:
    """Join distinct cycles by two-edge exchanges; record every explicit repair.

    Candidate-supported connections are preferred. If none exists for a component,
    all complete-graph connections are considered instead of returning an invalid tour.
    """
    bonds = set(bonds)
    components = factor_components(instance.dimension, bonds)
    repairs = []
    while len(components) > 1:
        small = set(components[0])
        inside = sorted(bond for bond in bonds if bond[0] in small)
        outside = sorted(bond for bond in bonds if bond[0] not in small)
        chosen = None
        used_candidates = True
        for require_candidate in (True, False):
            for first in inside:
                check_budget()
                a, b = first
                for second in outside:
                    c, d = second
                    for additions in ((edge(a, c), edge(b, d)), (edge(a, d), edge(b, c))):
                        if require_candidate and not any(bond in candidates for bond in additions):
                            continue
                        delta = (
                            sum(int(instance.distance[u, v]) for u, v in additions)
                            - int(instance.distance[a, b])
                            - int(instance.distance[c, d])
                        )
                        choice = (delta, tuple(sorted(additions)), first, second)
                        if chosen is None or choice < chosen:
                            chosen = choice
            if chosen is not None:
                used_candidates = require_candidate
                break
        if chosen is None:
            raise AssertionError("No reconnection exists for a complete symmetric TSP")
        delta, additions, first, second = chosen
        bonds.difference_update((first, second))
        bonds.update(additions)
        next_components = factor_components(instance.dimension, bonds)
        if len(next_components) != len(components) - 1:
            raise AssertionError("A subtour repair must merge exactly two components")
        repairs.append(
            {
                "removed": (first, second),
                "added": additions,
                "delta": delta,
                "candidate_supported": used_candidates,
            }
        )
        components = next_components
    if len(components) != 1:
        raise AssertionError("Subtour repair must finish with exactly one Hamiltonian cycle")
    return canonical_tour(instance, components[0]), tuple(repairs)
