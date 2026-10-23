"""Canonical symmetric-TSP instances and independently checked tour certificates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TSPInstance:
    """A symmetric TSP with an immutable, exact int64 distance matrix."""

    name: str
    distance: np.ndarray
    edge_weight_type: str = "EXPLICIT"
    coordinates: np.ndarray | None = None
    optimum: int | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("TSP instance name must not be empty")
        raw_distance = np.asarray(self.distance)
        if raw_distance.ndim != 2 or raw_distance.shape[0] != raw_distance.shape[1]:
            raise ValueError("TSP distance must be a square matrix")
        if raw_distance.shape[0] < 2:
            raise ValueError("A TSP instance must contain at least two cities")
        if not np.issubdtype(raw_distance.dtype, np.integer):
            raise TypeError("Canonical TSP distances must use an integer dtype")
        distance = np.ascontiguousarray(raw_distance, dtype=np.int64)
        if np.any(distance < 0):
            raise ValueError("TSP distances must be nonnegative")
        if np.any(np.diag(distance) != 0):
            raise ValueError("TSP distance diagonal must be zero")
        if not np.array_equal(distance, distance.T):
            raise ValueError("TSPInstance currently supports only symmetric TSPs")
        distance.setflags(write=False)
        object.__setattr__(self, "distance", distance)

        if self.coordinates is not None:
            coordinates = np.asarray(self.coordinates, dtype=np.float64)
            if coordinates.ndim != 2 or coordinates.shape[0] != distance.shape[0]:
                raise ValueError("Coordinates must have one row per city")
            if coordinates.shape[1] not in (2, 3) or not np.all(np.isfinite(coordinates)):
                raise ValueError("Coordinates must contain finite 2D or 3D points")
            coordinates = np.ascontiguousarray(coordinates)
            coordinates.setflags(write=False)
            object.__setattr__(self, "coordinates", coordinates)

        if self.optimum is not None and int(self.optimum) <= 0:
            raise ValueError("A declared optimum must be positive")
        if self.optimum is not None:
            object.__setattr__(self, "optimum", int(self.optimum))
        object.__setattr__(self, "edge_weight_type", self.edge_weight_type.upper())
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def dimension(self) -> int:
        return int(self.distance.shape[0])


@dataclass(frozen=True)
class TourCertificate:
    """An exact tour validation result using the instance's int64 distances."""

    tour: tuple[int, ...]
    is_permutation: bool
    length: int | None
    optimum: int | None
    relative_excess: float | None
    duplicate_cities: tuple[int, ...]
    missing_cities: tuple[int, ...]
    out_of_range_cities: tuple[int, ...]


def _as_integer_tour(tour: Sequence[int] | np.ndarray) -> tuple[int, ...]:
    array = np.asarray(tour)
    if array.ndim != 1:
        raise ValueError("A tour must be one-dimensional")
    if array.size == 0:
        return ()
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("Tour city identifiers must be integers")
    return tuple(int(city) for city in array.tolist())


def tour_length(
    instance: TSPInstance,
    tour: Sequence[int] | np.ndarray,
    *,
    validate: bool = True,
) -> int:
    """Return the closed-tour length, optionally requiring a strict permutation."""

    ordered = _as_integer_tour(tour)
    if validate:
        certificate = certify_tour(instance, ordered)
        if not certificate.is_permutation or certificate.length is None:
            raise ValueError("Tour must contain every city exactly once")
        return certificate.length
    if not ordered:
        raise ValueError("Tour must not be empty")
    if any(city < 0 or city >= instance.dimension for city in ordered):
        raise IndexError("Tour contains an out-of-range city")
    indices = np.asarray(ordered, dtype=np.int64)
    successors = np.roll(indices, -1)
    return int(instance.distance[indices, successors].sum(dtype=np.int64))


def certify_tour(
    instance: TSPInstance,
    tour: Sequence[int] | np.ndarray,
) -> TourCertificate:
    """Validate a zero-based tour without projecting or repairing it."""

    ordered = _as_integer_tour(tour)
    counts = Counter(ordered)
    out_of_range = tuple(sorted(city for city in counts if not 0 <= city < instance.dimension))
    duplicates = tuple(sorted(city for city, count in counts.items() if count > 1))
    present = {city for city in ordered if 0 <= city < instance.dimension}
    missing = tuple(sorted(set(range(instance.dimension)) - present))
    valid = (
        len(ordered) == instance.dimension and not out_of_range and not duplicates and not missing
    )
    length = tour_length(instance, ordered, validate=False) if valid else None
    relative_excess = None
    if length is not None and instance.optimum is not None:
        if length < instance.optimum:
            raise ValueError("Tour is shorter than the declared exact optimum")
        relative_excess = (length - instance.optimum) / instance.optimum
    return TourCertificate(
        tour=ordered,
        is_permutation=valid,
        length=length,
        optimum=instance.optimum,
        relative_excess=relative_excess,
        duplicate_cities=duplicates,
        missing_cities=missing,
        out_of_range_cities=out_of_range,
    )
