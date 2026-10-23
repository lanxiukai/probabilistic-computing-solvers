"""Exact and deterministic classical baselines for symmetric TSP instances."""

from __future__ import annotations

import itertools
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .instance import TSPInstance, certify_tour, tour_length

TSP_BASELINE_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TourResult:
    tour: tuple[int, ...]
    length: int
    method: str


def _result(instance: TSPInstance, tour: Sequence[int], method: str) -> TourResult:
    certificate = certify_tour(instance, tour)
    if not certificate.is_permutation or certificate.length is None:
        raise AssertionError("A TSP baseline produced an invalid tour")
    return TourResult(tour=certificate.tour, length=certificate.length, method=method)


def brute_force_tsp(
    instance: TSPInstance,
    *,
    fixed_city: int = 0,
    max_cities: int = 11,
) -> TourResult:
    """Solve a small symmetric TSP exactly by enumerating unique oriented tours."""

    if not 0 <= fixed_city < instance.dimension:
        raise IndexError("fixed_city is out of range")
    if instance.dimension > max_cities:
        raise ValueError("Instance exceeds the brute-force safety limit")
    remaining = tuple(city for city in range(instance.dimension) if city != fixed_city)
    best_tour: tuple[int, ...] | None = None
    best_length: int | None = None
    for suffix in itertools.permutations(remaining):
        if len(suffix) > 1 and suffix[0] > suffix[-1]:
            continue
        candidate = (fixed_city, *suffix)
        length = tour_length(instance, candidate, validate=False)
        if (
            best_length is None
            or length < best_length
            or (length == best_length and best_tour is not None and candidate < best_tour)
        ):
            best_tour = candidate
            best_length = length
    if best_tour is None or best_length is None:
        raise AssertionError("Brute-force enumeration unexpectedly produced no tour")
    return TourResult(tour=best_tour, length=best_length, method="brute_force")


def held_karp_tsp(
    instance: TSPInstance,
    *,
    fixed_city: int = 0,
    max_cities: int = 22,
) -> TourResult:
    """Solve a small symmetric TSP exactly with Bellman-Held-Karp dynamic programming."""

    if not 0 <= fixed_city < instance.dimension:
        raise IndexError("fixed_city is out of range")
    if instance.dimension > max_cities:
        raise ValueError("Instance exceeds the Held-Karp safety limit")
    cities = tuple(city for city in range(instance.dimension) if city != fixed_city)
    count = len(cities)
    costs: dict[tuple[int, int], int] = {}
    parents: dict[tuple[int, int], int | None] = {}
    for last in range(count):
        mask = 1 << last
        costs[mask, last] = int(instance.distance[fixed_city, cities[last]])
        parents[mask, last] = None

    for subset_size in range(2, count + 1):
        for subset in itertools.combinations(range(count), subset_size):
            mask = sum(1 << city_index for city_index in subset)
            for last in subset:
                previous_mask = mask ^ (1 << last)
                candidates = (
                    (
                        costs[previous_mask, previous]
                        + int(instance.distance[cities[previous], cities[last]]),
                        previous,
                    )
                    for previous in subset
                    if previous != last
                )
                best_cost, best_previous = min(candidates)
                costs[mask, last] = best_cost
                parents[mask, last] = best_previous

    full_mask = (1 << count) - 1
    total_cost, last = min(
        (
            costs[full_mask, city_index] + int(instance.distance[cities[city_index], fixed_city]),
            city_index,
        )
        for city_index in range(count)
    )
    reverse_suffix: list[int] = []
    mask = full_mask
    while True:
        reverse_suffix.append(cities[last])
        previous = parents[mask, last]
        mask ^= 1 << last
        if previous is None:
            break
        last = previous
    tour = (fixed_city, *reversed(reverse_suffix))
    result = _result(instance, tour, "held_karp")
    if result.length != total_cost:
        raise AssertionError("Held-Karp reconstruction does not match its optimum cost")
    return result


def nearest_neighbor_tsp(instance: TSPInstance, *, start_city: int = 0) -> TourResult:
    """Build a deterministic nearest-neighbor tour with city-id tie breaking."""

    if not 0 <= start_city < instance.dimension:
        raise IndexError("start_city is out of range")
    remaining = set(range(instance.dimension))
    remaining.remove(start_city)
    tour = [start_city]
    while remaining:
        current = tour[-1]
        next_city = min(remaining, key=lambda city: (instance.distance[current, city], city))
        tour.append(next_city)
        remaining.remove(next_city)
    return _result(instance, tour, "nearest_neighbor")


def two_opt_tsp(
    instance: TSPInstance,
    initial_tour: Sequence[int],
    *,
    strategy: str = "best",
    max_passes: int | None = None,
) -> TourResult:
    """Apply deterministic symmetric 2-opt while preserving the first city."""

    if strategy not in {"first", "best"}:
        raise ValueError("strategy must be 'first' or 'best'")
    if max_passes is not None and max_passes <= 0:
        raise ValueError("max_passes must be positive when provided")
    certificate = certify_tour(instance, initial_tour)
    if not certificate.is_permutation:
        raise ValueError("initial_tour must be a strict permutation")
    tour = list(certificate.tour)
    passes = 0
    while max_passes is None or passes < max_passes:
        passes += 1
        selected: tuple[int, int, int] | None = None
        for left_edge in range(instance.dimension - 1):
            for right_edge in range(left_edge + 2, instance.dimension):
                if left_edge == 0 and right_edge == instance.dimension - 1:
                    continue
                first = tour[left_edge]
                second = tour[left_edge + 1]
                third = tour[right_edge]
                fourth = tour[(right_edge + 1) % instance.dimension]
                delta = int(
                    instance.distance[first, third]
                    + instance.distance[second, fourth]
                    - instance.distance[first, second]
                    - instance.distance[third, fourth]
                )
                if delta >= 0:
                    continue
                if strategy == "first":
                    selected = (delta, left_edge, right_edge)
                    break
                candidate = (delta, left_edge, right_edge)
                if selected is None or candidate < selected:
                    selected = candidate
            if selected is not None and strategy == "first":
                break
        if selected is None:
            break
        _, left_edge, right_edge = selected
        tour[left_edge + 1 : right_edge + 1] = reversed(tour[left_edge + 1 : right_edge + 1])
    return _result(instance, tour, f"two_opt_{strategy}")


def multistart_nearest_neighbor_two_opt(
    instance: TSPInstance,
    *,
    start_cities: Iterable[int] | None = None,
    strategy: str = "best",
) -> TourResult:
    """Run nearest-neighbor plus 2-opt from multiple deterministic starts."""

    candidates = multistart_nearest_neighbor_two_opt_population(
        instance,
        start_cities=start_cities,
        strategy=strategy,
    )
    best = min(candidates, key=lambda result: (result.length, result.tour))
    return TourResult(
        tour=best.tour,
        length=best.length,
        method="multistart_nearest_neighbor_two_opt",
    )


def multistart_nearest_neighbor_two_opt_population(
    instance: TSPInstance,
    *,
    start_cities: Iterable[int] | None = None,
    strategy: str = "best",
    fixed_city: int = 0,
) -> tuple[TourResult, ...]:
    """Return unique locally optimized tours, rotated to one fixed first city."""

    starts = tuple(range(instance.dimension)) if start_cities is None else tuple(start_cities)
    if not starts:
        raise ValueError("At least one start city is required")
    if not 0 <= fixed_city < instance.dimension:
        raise IndexError("fixed_city is out of range")
    unique: dict[tuple[int, ...], TourResult] = {}
    for start_city in starts:
        initial = nearest_neighbor_tsp(instance, start_city=int(start_city))
        optimized = two_opt_tsp(instance, initial.tour, strategy=strategy)
        pivot = optimized.tour.index(fixed_city)
        rotated = optimized.tour[pivot:] + optimized.tour[:pivot]
        reversed_orientation = (rotated[0], *reversed(rotated[1:]))
        canonical = min(rotated, reversed_orientation)
        candidate = TourResult(
            tour=canonical,
            length=optimized.length,
            method="multistart_nearest_neighbor_two_opt_population",
        )
        unique.setdefault(canonical, candidate)
    return tuple(sorted(unique.values(), key=lambda result: (result.length, result.tour)))


def get_or_build_multistart_population(
    instance: TSPInstance,
    instance_sha256: str,
    cache_directory: str | Path,
) -> tuple[tuple[TourResult, ...], Path, bool]:
    """Load or certify and atomically cache the deterministic multistart population."""

    digest = instance_sha256.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("instance_sha256 must contain 64 hexadecimal characters")
    directory = Path(cache_directory)
    path = directory / f"multistart-two-opt-v{TSP_BASELINE_CACHE_SCHEMA_VERSION}-{digest}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != TSP_BASELINE_CACHE_SCHEMA_VERSION
            or payload.get("instance_sha256") != digest
            or payload.get("dimension") != instance.dimension
        ):
            raise ValueError(f"TSP baseline cache metadata is inconsistent: {path}")
        rows = payload.get("population")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"TSP baseline cache population is invalid: {path}")
        population = tuple(
            TourResult(
                tour=tuple(int(city) for city in row["tour"]),
                length=int(row["length"]),
                method=str(row["method"]),
            )
            for row in rows
        )
        if tuple(sorted(population, key=lambda result: (result.length, result.tour))) != population:
            raise ValueError(f"TSP baseline cache population is not sorted: {path}")
        if len({result.tour for result in population}) != len(population):
            raise ValueError(f"TSP baseline cache contains duplicate tours: {path}")
        for result in population:
            certificate = certify_tour(instance, result.tour)
            if not certificate.is_permutation or certificate.length != result.length:
                raise ValueError(f"TSP baseline cache certificate failed: {path}")
        return population, path, True

    population = multistart_nearest_neighbor_two_opt_population(instance)
    payload = {
        "schema_version": TSP_BASELINE_CACHE_SCHEMA_VERSION,
        "instance_sha256": digest,
        "dimension": instance.dimension,
        "algorithm": "multistart_nearest_neighbor_two_opt_population",
        "strategy": "best",
        "fixed_city": 0,
        "population": [
            {"tour": list(result.tour), "length": result.length, "method": result.method}
            for result in population
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=directory,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(serialized)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    return population, path, False
