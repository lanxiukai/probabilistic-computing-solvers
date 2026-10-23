"""Auditable quadratic penalties for invertible logic gates."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from sparse_factorization.core.model import IsingModel, QUBOModel

PenaltyProfile = Literal["canonical", "aadit"]


@dataclass(frozen=True)
class GateAudit:
    """Exhaustive energy audit for one gate template."""

    gate: str
    profile: str
    ground_energy: float
    gap: float
    ground_states: tuple[tuple[int, ...], ...]
    valid_states: tuple[tuple[int, ...], ...]

    @property
    def passed(self) -> bool:
        return self.ground_states == self.valid_states and self.gap > 0.0


@dataclass(frozen=True)
class GateTemplate:
    """A gate represented by a non-negative binary quadratic penalty."""

    name: str
    ports: tuple[str, ...]
    linear: tuple[float, ...]
    quadratic: tuple[tuple[int, int, float], ...]
    constant: float
    valid_states: tuple[tuple[int, ...], ...]

    @property
    def size(self) -> int:
        return len(self.ports)

    def qubo(
        self,
        profile: PenaltyProfile = "canonical",
        *,
        dtype: torch.dtype = torch.float64,
    ) -> QUBOModel:
        """Return the gate penalty in the requested coefficient profile.

        The ``aadit`` profile reproduces the published AND, FA, COPY, and NOT
        couplings, while retaining a constant that puts all valid rows at zero.
        """

        scale = _profile_scale(profile, self.name)
        matrix = torch.zeros((self.size, self.size), dtype=dtype)
        for left, right, value in self.quadratic:
            coefficient = scale * value
            matrix[left, right] += coefficient
            matrix[right, left] += coefficient
        linear = torch.tensor(self.linear, dtype=dtype) * scale
        return QUBOModel(
            quadratic=matrix,
            linear=linear,
            constant=self.constant * scale,
            names=self.ports,
            metadata={
                "gate": self.name,
                "penalty_profile": profile,
                "nonnegative_penalty": True,
            },
        )

    def ising(
        self,
        profile: PenaltyProfile = "canonical",
        *,
        dtype: torch.dtype = torch.float64,
    ) -> IsingModel:
        return self.qubo(profile, dtype=dtype).to_ising()

    def audit(
        self,
        profile: PenaltyProfile = "canonical",
        *,
        dtype: torch.dtype = torch.float64,
    ) -> GateAudit:
        bits = torch.tensor(
            list(itertools.product((0, 1), repeat=self.size)),
            dtype=dtype,
        )
        energies = self.qubo(profile, dtype=dtype).energy(bits)
        ground_energy = float(energies.min().item())
        ground_mask = torch.isclose(
            energies,
            torch.tensor(ground_energy, dtype=dtype),
            atol=1e-10,
            rtol=0.0,
        )
        rows = tuple(tuple(int(value) for value in row) for row in bits[ground_mask].tolist())
        valid = tuple(sorted(self.valid_states))
        ground = tuple(sorted(rows))
        excited = energies[~ground_mask]
        gap = float(excited.min().item() - ground_energy) if excited.numel() else float("inf")
        return GateAudit(
            gate=self.name,
            profile=profile,
            ground_energy=ground_energy,
            gap=gap,
            ground_states=ground,
            valid_states=valid,
        )


def _profile_scale(profile: PenaltyProfile, gate: str) -> float:
    if profile == "canonical":
        return 1.0
    if profile != "aadit":
        raise ValueError(f"Unknown gate penalty profile: {profile}")
    return {
        "copy": 2.0,
        "not": 2.0,
        "and": 4.0,
        "or": 4.0,
        "half_adder": 2.0,
        "full_adder": 2.0,
    }[gate]


def _truth_rows(
    size: int,
    predicate: Callable[[tuple[int, ...]], bool],
) -> tuple[tuple[int, ...], ...]:
    return tuple(row for row in itertools.product((0, 1), repeat=size) if predicate(row))


def _squared_relation(
    name: str,
    ports: tuple[str, ...],
    coefficients: tuple[int, ...],
    valid_states: tuple[tuple[int, ...], ...],
) -> GateTemplate:
    linear = tuple(float(value * value) for value in coefficients)
    quadratic = tuple(
        (left, right, float(2 * coefficients[left] * coefficients[right]))
        for left in range(len(coefficients))
        for right in range(left + 1, len(coefficients))
    )
    return GateTemplate(
        name=name,
        ports=ports,
        linear=linear,
        quadratic=quadratic,
        constant=0.0,
        valid_states=valid_states,
    )


COPY = GateTemplate(
    name="copy",
    ports=("input", "output"),
    linear=(1.0, 1.0),
    quadratic=((0, 1, -2.0),),
    constant=0.0,
    valid_states=((0, 0), (1, 1)),
)

NOT = GateTemplate(
    name="not",
    ports=("input", "output"),
    linear=(-1.0, -1.0),
    quadratic=((0, 1, 2.0),),
    constant=1.0,
    valid_states=((0, 1), (1, 0)),
)

AND = GateTemplate(
    name="and",
    ports=("left", "right", "output"),
    linear=(0.0, 0.0, 3.0),
    quadratic=((0, 1, 1.0), (0, 2, -2.0), (1, 2, -2.0)),
    constant=0.0,
    valid_states=_truth_rows(3, lambda row: row[2] == (row[0] & row[1])),
)

OR = GateTemplate(
    name="or",
    ports=("left", "right", "output"),
    linear=(1.0, 1.0, 1.0),
    quadratic=((0, 1, 1.0), (0, 2, -2.0), (1, 2, -2.0)),
    constant=0.0,
    valid_states=_truth_rows(3, lambda row: row[2] == (row[0] | row[1])),
)

HALF_ADDER = _squared_relation(
    "half_adder",
    ("left", "right", "sum", "carry"),
    (1, 1, -1, -2),
    _truth_rows(4, lambda row: row[0] + row[1] == row[2] + 2 * row[3]),
)

FULL_ADDER = _squared_relation(
    "full_adder",
    ("left", "right", "carry_in", "sum", "carry_out"),
    (1, 1, 1, -1, -2),
    _truth_rows(
        5,
        lambda row: row[0] + row[1] + row[2] == row[3] + 2 * row[4],
    ),
)

GATES: dict[str, GateTemplate] = {
    gate.name: gate for gate in (COPY, NOT, AND, OR, HALF_ADDER, FULL_ADDER)
}


def get_gate(name: str) -> GateTemplate:
    try:
        return GATES[name]
    except KeyError as error:
        raise ValueError(f"Unknown gate: {name}") from error


def audit_gate_library(
    profile: PenaltyProfile = "canonical",
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[GateAudit, ...]:
    """Exhaustively audit every built-in gate."""

    return tuple(gate.audit(profile, dtype=dtype) for gate in GATES.values())
