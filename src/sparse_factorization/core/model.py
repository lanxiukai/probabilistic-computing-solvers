"""Canonical Ising and QUBO representations.

The Ising convention is

    E(s) = -sum_{i<j} J[i, j] s[i] s[j] - sum_i h[i] s[i] + constant.

Both coefficient matrices are symmetric with a zero diagonal. A QUBO off-diagonal
coefficient is stored in both matrix triangles but counted only once in the energy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch


def _default_names(size: int) -> tuple[str, ...]:
    return tuple(f"x_{index}" for index in range(size))


def _validate_square_symmetric(matrix: torch.Tensor, vector: torch.Tensor, label: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{label} must be a square matrix")
    if vector.ndim != 1 or vector.shape[0] != matrix.shape[0]:
        raise ValueError(f"{label} and its linear vector have incompatible shapes")
    if matrix.dtype != vector.dtype or matrix.device != vector.device:
        raise ValueError(f"{label} and its linear vector must share dtype and device")
    if not bool(torch.all(torch.isfinite(matrix))) or not bool(torch.all(torch.isfinite(vector))):
        raise ValueError(f"{label} coefficients must be finite")
    if not torch.allclose(matrix, matrix.mT, atol=1e-7, rtol=1e-7):
        raise ValueError(f"{label} must be symmetric")
    if not torch.allclose(
        torch.diagonal(matrix), torch.zeros_like(torch.diagonal(matrix)), atol=1e-7, rtol=0
    ):
        raise ValueError(f"{label} diagonal must be zero")


@dataclass(frozen=True)
class IsingModel:
    """A pairwise Ising model using the repository-wide energy convention."""

    coupling: torch.Tensor
    bias: torch.Tensor
    constant: float = 0.0
    names: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_square_symmetric(self.coupling, self.bias, "coupling")
        if not self.coupling.is_floating_point() or not self.bias.is_floating_point():
            raise TypeError("Ising coefficients must use a floating-point dtype")
        names = self.names or _default_names(self.bias.numel())
        if len(names) != self.bias.numel() or len(set(names)) != len(names):
            raise ValueError("Ising names must be unique and match the model size")
        object.__setattr__(self, "names", tuple(names))

    @property
    def size(self) -> int:
        return int(self.bias.numel())

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> IsingModel:
        return IsingModel(
            coupling=self.coupling.to(device=device, dtype=dtype),
            bias=self.bias.to(device=device, dtype=dtype),
            constant=self.constant,
            names=self.names,
            metadata=self.metadata,
        )

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        """Return energies for a final dimension containing bipolar spins."""

        if spins.shape[-1] != self.size:
            raise ValueError("Spin shape does not match the model size")
        values = spins.to(dtype=self.coupling.dtype, device=self.coupling.device)
        pair = -0.5 * torch.einsum("...i,ij,...j->...", values, self.coupling, values)
        linear = -torch.einsum("...i,i->...", values, self.bias)
        return pair + linear + self.constant

    def local_fields(
        self, spins: torch.Tensor, indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return h_i + sum_j J_ij s_j for all or selected variables."""

        if spins.shape[-1] != self.size:
            raise ValueError("Spin shape does not match the model size")
        values = spins.to(dtype=self.coupling.dtype, device=self.coupling.device)
        if indices is None:
            return values @ self.coupling + self.bias
        selected = indices.to(device=self.coupling.device, dtype=torch.long)
        return values @ self.coupling[:, selected] + self.bias[selected]

    def clamp(self, fixed: Mapping[int | str, int]) -> ClampResult:
        """Eliminate exactly fixed spins and preserve the energy of reconstructed states."""

        name_to_index = {name: index for index, name in enumerate(self.names)}
        resolved: dict[int, int] = {}
        for key, value in fixed.items():
            index = name_to_index[key] if isinstance(key, str) else int(key)
            if index < 0 or index >= self.size:
                raise IndexError(f"Clamp index is out of range: {index}")
            spin = int(value)
            if spin not in (-1, 1):
                raise ValueError("Fixed Ising values must be -1 or +1")
            if index in resolved and resolved[index] != spin:
                raise ValueError(f"Conflicting clamp for variable {index}")
            resolved[index] = spin

        fixed_indices = tuple(sorted(resolved))
        fixed_set = set(fixed_indices)
        free_indices = tuple(index for index in range(self.size) if index not in fixed_set)
        device = self.coupling.device
        fixed_tensor = torch.tensor(
            [resolved[index] for index in fixed_indices], dtype=self.coupling.dtype, device=device
        )
        fixed_index_tensor = torch.tensor(fixed_indices, dtype=torch.long, device=device)
        free_index_tensor = torch.tensor(free_indices, dtype=torch.long, device=device)

        reduced_coupling = self.coupling.index_select(0, free_index_tensor).index_select(
            1, free_index_tensor
        )
        reduced_bias = self.bias.index_select(0, free_index_tensor).clone()
        reduced_constant = float(self.constant)

        if fixed_indices:
            free_fixed = self.coupling.index_select(0, free_index_tensor).index_select(
                1, fixed_index_tensor
            )
            reduced_bias = reduced_bias + free_fixed @ fixed_tensor
            fixed_bias = self.bias.index_select(0, fixed_index_tensor)
            fixed_coupling = self.coupling.index_select(0, fixed_index_tensor).index_select(
                1, fixed_index_tensor
            )
            constant_delta = -torch.dot(fixed_bias, fixed_tensor)
            constant_delta -= 0.5 * torch.einsum(
                "i,ij,j->", fixed_tensor, fixed_coupling, fixed_tensor
            )
            reduced_constant += float(constant_delta.item())

        reduced = IsingModel(
            coupling=reduced_coupling,
            bias=reduced_bias,
            constant=reduced_constant,
            names=tuple(self.names[index] for index in free_indices),
            metadata={**self.metadata, "clamped_variables": len(fixed_indices)},
        )
        return ClampResult(
            model=reduced,
            original_names=self.names,
            free_indices=free_indices,
            fixed_indices=fixed_indices,
            fixed_spins=tuple(resolved[index] for index in fixed_indices),
        )


@dataclass(frozen=True)
class ClampResult:
    """A reduced Ising model plus an exact reconstruction map."""

    model: IsingModel
    original_names: tuple[str, ...]
    free_indices: tuple[int, ...]
    fixed_indices: tuple[int, ...]
    fixed_spins: tuple[int, ...]

    def reconstruct(self, free_spins: torch.Tensor) -> torch.Tensor:
        if free_spins.shape[-1] != len(self.free_indices):
            raise ValueError("Free spin shape does not match the clamp map")
        result = torch.empty(
            (*free_spins.shape[:-1], len(self.original_names)),
            dtype=free_spins.dtype,
            device=free_spins.device,
        )
        if self.free_indices:
            free_index = torch.tensor(self.free_indices, device=free_spins.device)
            result[..., free_index] = free_spins
        if self.fixed_indices:
            fixed_index = torch.tensor(self.fixed_indices, device=free_spins.device)
            fixed_values = torch.tensor(
                self.fixed_spins, dtype=free_spins.dtype, device=free_spins.device
            )
            result[..., fixed_index] = fixed_values
        return result


@dataclass(frozen=True)
class QUBOModel:
    """A binary quadratic model with symmetric, once-counted off-diagonal terms."""

    quadratic: torch.Tensor
    linear: torch.Tensor
    constant: float = 0.0
    names: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_square_symmetric(self.quadratic, self.linear, "quadratic")
        if not self.quadratic.is_floating_point() or not self.linear.is_floating_point():
            raise TypeError("QUBO coefficients must use a floating-point dtype")
        names = self.names or _default_names(self.linear.numel())
        if len(names) != self.linear.numel() or len(set(names)) != len(names):
            raise ValueError("QUBO names must be unique and match the model size")
        object.__setattr__(self, "names", tuple(names))

    @property
    def size(self) -> int:
        return int(self.linear.numel())

    def energy(self, bits: torch.Tensor) -> torch.Tensor:
        if bits.shape[-1] != self.size:
            raise ValueError("Bit shape does not match the model size")
        values = bits.to(dtype=self.quadratic.dtype, device=self.quadratic.device)
        pair = 0.5 * torch.einsum("...i,ij,...j->...", values, self.quadratic, values)
        linear = torch.einsum("...i,i->...", values, self.linear)
        return pair + linear + self.constant

    def to_ising(self) -> IsingModel:
        coupling = -self.quadratic / 4.0
        bias = -(self.linear / 2.0 + self.quadratic.sum(dim=1) / 4.0)
        constant = (
            float(self.constant)
            + float(self.linear.sum().item()) / 2.0
            + float(self.quadratic.sum().item()) / 8.0
        )
        return IsingModel(
            coupling=coupling,
            bias=bias,
            constant=constant,
            names=self.names,
            metadata={**self.metadata, "converted_from": "qubo"},
        )


def bits_to_spins(bits: torch.Tensor) -> torch.Tensor:
    if torch.any((bits != 0) & (bits != 1)):
        raise ValueError("Bit values must be zero or one")
    return bits.to(torch.int8).mul(2).sub(1)


def spins_to_bits(spins: torch.Tensor) -> torch.Tensor:
    if torch.any((spins != -1) & (spins != 1)):
        raise ValueError("Spin values must be -1 or +1")
    return spins.add(1).div(2).to(torch.int8)


def state_from_mapping(names: Sequence[str], values: Mapping[str, int]) -> torch.Tensor:
    """Build a bipolar state in a declared name order."""

    try:
        ordered = [values[name] for name in names]
    except KeyError as error:
        raise ValueError(f"Missing state value for {error.args[0]}") from error
    result = torch.tensor(ordered, dtype=torch.int8)
    if torch.any((result != -1) & (result != 1)):
        raise ValueError("Mapped spin values must be -1 or +1")
    return result
