"""Exact integer, vectorized first-improvement descent for the CPU hybrid."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

import numpy as np

from .hybrid_moves import Tour, canonical_tour
from .instance import TSPInstance, certify_tour


@lru_cache(maxsize=32)
def _two_opt_pairs(cities: int) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.triu_indices(cities, k=2)
    keep = ~((left == 0) & (right == cities - 1))
    left, right = left[keep], right[keep]
    left.setflags(write=False)
    right.setflags(write=False)
    return left, right


@lru_cache(maxsize=96)
def _or_opt_pairs(cities: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    starts, after = np.indices((cities, cities)).reshape(2, -1)
    offset = (after - starts) % cities
    keep = (offset >= size) & (offset != cities - 1)
    starts, after = starts[keep], after[keep]
    starts.setflags(write=False)
    after.setflags(write=False)
    return starts, after


def _or_opt_move(current: np.ndarray, distance: np.ndarray) -> np.ndarray | None:
    """Choose the steepest improving relocation of one to three consecutive cities."""
    n = len(current)
    chosen = None
    for size in range(1, min(3, n - 2) + 1):
        starts, after = _or_opt_pairs(n, size)
        p, a = current[(starts - 1) % n], current[starts]
        b, q = current[(starts + size - 1) % n], current[(starts + size) % n]
        u, v = current[after], current[(after + 1) % n]
        base = distance[p, q] - distance[p, a] - distance[b, q] - distance[u, v]
        for reverse in range(1 if size == 1 else 2):
            delta = base + (
                distance[u, b] + distance[a, v] if reverse else distance[u, a] + distance[b, v]
            )
            index = int(np.argmin(delta))
            choice = (int(delta[index]), size, int(starts[index]), int(after[index]), reverse)
            if choice[0] < 0 and (chosen is None or choice < chosen):
                chosen = choice
    if chosen is None:
        return None
    _, size, start, after, reverse = chosen
    positions = {(start + offset) % n for offset in range(size)}
    chain = [int(current[(start + offset) % n]) for offset in range(size)]
    if reverse:
        chain.reverse()
    rest = [int(city) for index, city in enumerate(current) if index not in positions]
    insertion = rest.index(int(current[after])) + 1
    moved = rest[:insertion] + chain + rest[insertion:]
    anchor = moved.index(0)
    return np.asarray(moved[anchor:] + moved[:anchor], dtype=np.int64)


def first_descent(
    instance: TSPInstance,
    tour: Sequence[int],
    max_moves: int,
    remaining: Callable[[], float],
    *,
    or_opt: bool = False,
) -> Tour:
    """Match scalar first-2-opt ordering, checking time between accepted moves.

    This evaluates the complete neighborhood, with no candidate-graph restriction.
    It bounds accepted moves, not sweeps through a changing tour. All edge deltas
    remain integer; the guard prevents int64 overflow in the four-edge expression.
    """
    if max_moves <= 0:
        raise ValueError("max_moves must be positive")
    certificate = certify_tour(instance, tour)
    if not certificate.is_permutation:
        raise ValueError("Local search requires a complete tour")
    if int(instance.distance.max()) > np.iinfo(np.int64).max // (6 if or_opt else 4):
        raise ValueError("Local-search edge deltas could overflow int64")
    current = np.asarray(certificate.tour, dtype=np.int64)
    left, right = _two_opt_pairs(instance.dimension)
    distance = instance.distance
    for _ in range(max_moves):
        if remaining() <= 0:
            break
        a, b = current[left], current[left + 1]
        c, d = current[right], current[(right + 1) % len(current)]
        delta = distance[a, c] + distance[b, d] - distance[a, b] - distance[c, d]
        improving = np.flatnonzero(delta < 0)
        if not improving.size:
            relocated = _or_opt_move(current, distance) if or_opt else None
            if relocated is None:
                break
            current = relocated
            continue
        chosen = int(improving[0])
        start, stop = int(left[chosen]) + 1, int(right[chosen]) + 1
        current[start:stop] = current[start:stop][::-1]
    return canonical_tour(instance, current)
