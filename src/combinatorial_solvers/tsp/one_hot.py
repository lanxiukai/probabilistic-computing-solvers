"""Fixed-start city-position QUBO construction and strict one-hot decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from combinatorial_solvers.core.model import QUBOModel

from .instance import TourCertificate, TSPInstance, certify_tour


@dataclass(frozen=True)
class OneHotDecodeResult:
    """Strict decoder output; infeasible states have a ``None`` tour."""

    tours: tuple[tuple[int, ...] | None, ...]
    feasible: torch.Tensor
    binary: torch.Tensor
    row_violations: torch.Tensor
    column_violations: torch.Tensor


@dataclass(frozen=True)
class FixedStartOneHotQUBO:
    """Lucas-style one-hot TSP model with one city and position eliminated."""

    instance: TSPInstance
    constraint_weight: float = 1.0
    objective_weight: float = 0.5
    fixed_city: int = 0
    normalize_distances: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.fixed_city < self.instance.dimension:
            raise IndexError("fixed_city is out of range")
        if not math.isfinite(self.constraint_weight) or self.constraint_weight <= 0:
            raise ValueError("constraint_weight must be finite and positive")
        if not math.isfinite(self.objective_weight) or self.objective_weight <= 0:
            raise ValueError("objective_weight must be finite and positive")
        if self.normalize_distances and self.maximum_distance <= 0:
            raise ValueError("Cannot normalize an all-zero distance matrix")

    @property
    def free_cities(self) -> tuple[int, ...]:
        return tuple(city for city in range(self.instance.dimension) if city != self.fixed_city)

    @property
    def order(self) -> int:
        return self.instance.dimension - 1

    @property
    def size(self) -> int:
        return self.order**2

    @property
    def maximum_distance(self) -> int:
        return int(self.instance.distance.max())

    @property
    def distance_scale(self) -> float:
        return float(self.maximum_distance) if self.normalize_distances else 1.0

    @property
    def satisfies_lucas_bound(self) -> bool:
        maximum_scaled = self.maximum_distance / self.distance_scale
        return self.objective_weight * maximum_scaled < self.constraint_weight

    def variable_index(self, free_city_index: int, position_index: int) -> int:
        if not 0 <= free_city_index < self.order or not 0 <= position_index < self.order:
            raise IndexError("One-hot variable coordinate is out of range")
        return free_city_index * self.order + position_index

    def energy(self, bits: torch.Tensor) -> torch.Tensor:
        """Evaluate the structured QUBO without materializing a dense matrix."""

        if bits.shape[-1] == self.size:
            values = bits.reshape(*bits.shape[:-1], self.order, self.order)
        elif bits.ndim >= 2 and bits.shape[-2:] == (self.order, self.order):
            values = bits
        else:
            raise ValueError("One-hot state has an incompatible shape")
        if torch.any((values != 0) & (values != 1)):
            raise ValueError("One-hot state must contain only zero and one")
        dtype = values.dtype if values.is_floating_point() else torch.float64
        values = values.to(dtype=dtype)
        row_counts = values.sum(dim=-1)
        column_counts = values.sum(dim=-2)
        constraint = self.constraint_weight * (
            (1.0 - row_counts).square().sum(dim=-1) + (1.0 - column_counts).square().sum(dim=-1)
        )

        free = torch.tensor(self.free_cities, dtype=torch.long, device=values.device)
        source_distance = torch.tensor(
            np.asarray(self.instance.distance), dtype=dtype, device=values.device
        )
        free_distance = source_distance.index_select(0, free).index_select(1, free)
        fixed_to_free = source_distance[self.fixed_city].index_select(0, free)
        free_to_fixed = source_distance[:, self.fixed_city].index_select(0, free)
        objective = torch.einsum("...i,i->...", values[..., :, 0], fixed_to_free)
        objective = objective + torch.einsum("...i,i->...", values[..., :, -1], free_to_fixed)
        for position in range(self.order - 1):
            objective = objective + torch.einsum(
                "...i,ij,...j->...",
                values[..., :, position],
                free_distance,
                values[..., :, position + 1],
            )
        return constraint + self.objective_weight * objective / self.distance_scale

    def to_qubo(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        max_dense_bytes: int | None = 512 * 1024**2,
    ) -> QUBOModel:
        """Materialize the once-counted symmetric QUBO used by the shared core."""

        if not dtype.is_floating_point:
            raise TypeError("QUBO coefficients require a floating-point dtype")
        element_size = torch.empty((), dtype=dtype).element_size()
        required_bytes = self.size * self.size * element_size
        if max_dense_bytes is not None and required_bytes > max_dense_bytes:
            raise MemoryError(f"Dense QUBO requires {required_bytes} bytes, above max_dense_bytes")
        quadratic = torch.zeros((self.size, self.size), dtype=dtype)
        linear = torch.full((self.size,), -2.0 * self.constraint_weight, dtype=dtype)
        constant = 2.0 * self.order * self.constraint_weight

        def add_pair(left: int, right: int, coefficient: float) -> None:
            if left == right:
                linear[left] += coefficient
                return
            quadratic[left, right] += coefficient
            quadratic[right, left] += coefficient

        for city_index in range(self.order):
            for left_position in range(self.order):
                left = self.variable_index(city_index, left_position)
                for right_position in range(left_position + 1, self.order):
                    right = self.variable_index(city_index, right_position)
                    add_pair(left, right, 2.0 * self.constraint_weight)
        for position in range(self.order):
            for left_city in range(self.order):
                left = self.variable_index(left_city, position)
                for right_city in range(left_city + 1, self.order):
                    right = self.variable_index(right_city, position)
                    add_pair(left, right, 2.0 * self.constraint_weight)

        scaled_weight = self.objective_weight / self.distance_scale
        free_cities = self.free_cities
        for city_index, city in enumerate(free_cities):
            linear[self.variable_index(city_index, 0)] += (
                scaled_weight * self.instance.distance[self.fixed_city, city]
            )
            linear[self.variable_index(city_index, self.order - 1)] += (
                scaled_weight * self.instance.distance[city, self.fixed_city]
            )
        for position in range(self.order - 1):
            for left_city_index, left_city in enumerate(free_cities):
                left = self.variable_index(left_city_index, position)
                for right_city_index, right_city in enumerate(free_cities):
                    coefficient = scaled_weight * self.instance.distance[left_city, right_city]
                    if coefficient == 0:
                        continue
                    right = self.variable_index(right_city_index, position + 1)
                    add_pair(left, right, float(coefficient))

        names = tuple(
            f"city_{city}_position_{position + 1}"
            for city in free_cities
            for position in range(self.order)
        )
        return QUBOModel(
            quadratic=quadratic,
            linear=linear,
            constant=float(constant),
            names=names,
            metadata={
                "problem": "tsp",
                "encoding": "fixed_start_one_hot",
                "instance": self.instance.name,
                "fixed_city": self.fixed_city,
                "constraint_weight": self.constraint_weight,
                "objective_weight": self.objective_weight,
                "distance_scale": self.distance_scale,
            },
        )

    def decode(self, bits: torch.Tensor) -> OneHotDecodeResult:
        """Decode only exact permutation matrices; no projection is performed."""

        if bits.ndim == 1:
            states = bits.unsqueeze(0)
        elif bits.ndim == 2 and bits.shape == (self.order, self.order):
            states = bits.reshape(1, self.size)
        elif bits.ndim == 2:
            states = bits
        elif bits.ndim == 3 and bits.shape[-2:] == (self.order, self.order):
            states = bits.reshape(bits.shape[0], self.size)
        else:
            raise ValueError("bits must have shape [size], [batch, size], or [batch, m, m]")
        if states.shape[1] != self.size:
            raise ValueError("One-hot state has an incompatible size")
        binary = ((states == 0) | (states == 1)).all(dim=1)
        safe = torch.where((states == 0) | (states == 1), states, torch.zeros_like(states))
        matrices = safe.to(torch.int64).reshape(-1, self.order, self.order)
        row_violations = (matrices.sum(dim=2) - 1).abs().sum(dim=1)
        column_violations = (matrices.sum(dim=1) - 1).abs().sum(dim=1)
        feasible = binary & (row_violations == 0) & (column_violations == 0)

        tours: list[tuple[int, ...] | None] = []
        free_cities = self.free_cities
        for matrix, valid in zip(matrices.cpu(), feasible.cpu(), strict=True):
            if not bool(valid):
                tours.append(None)
                continue
            ordered = [self.fixed_city]
            for position in range(self.order):
                city_index = int(torch.nonzero(matrix[:, position], as_tuple=False).item())
                ordered.append(free_cities[city_index])
            tours.append(tuple(ordered))
        return OneHotDecodeResult(
            tours=tuple(tours),
            feasible=feasible.cpu(),
            binary=binary.cpu(),
            row_violations=row_violations.cpu(),
            column_violations=column_violations.cpu(),
        )

    def certify(self, bits: torch.Tensor) -> tuple[TourCertificate | None, ...]:
        decoded = self.decode(bits)
        return tuple(
            None if tour is None else certify_tour(self.instance, tour) for tour in decoded.tours
        )

    def color_blocks(self) -> tuple[tuple[int, ...], ...]:
        """Return a deterministic valid 2m coloring for the fixed-start interaction graph."""

        blocks: list[list[int]] = [[] for _ in range(2 * self.order)]
        for city_index in range(self.order):
            for position in range(self.order):
                color = (position % 2) * self.order
                color += (city_index + position // 2) % self.order
                blocks[color].append(self.variable_index(city_index, position))
        return tuple(tuple(block) for block in blocks if block)


def fixed_start_color_blocks(dimension: int) -> tuple[tuple[int, ...], ...]:
    """Return the analytical coloring without requiring a full TSP instance."""

    if dimension < 2:
        raise ValueError("dimension must be at least two")
    order = dimension - 1
    blocks: list[list[int]] = [[] for _ in range(2 * order)]
    for city_index in range(order):
        for position in range(order):
            color = (position % 2) * order + (city_index + position // 2) % order
            blocks[color].append(city_index * order + position)
    return tuple(tuple(block) for block in blocks if block)
