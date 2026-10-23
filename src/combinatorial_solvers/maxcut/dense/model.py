"""Packed dense unit-weight graphs and an integer bitset cut certificate."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

MAGIC = b"PCDENSE1"
KINDS = {"sk": 0, "er50": 1}


@dataclass(frozen=True)
class Instance:
    size: int
    family: str
    packed: np.ndarray

    def __post_init__(self):
        if (
            not isinstance(self.size, int)
            or isinstance(self.size, bool)
            or not 2 <= self.size <= 32768
        ):
            raise ValueError("Dense model size must be between 2 and 32768")
        if self.family not in KINDS:
            raise ValueError("Dense family must be sk or er50")
        data = np.asarray(self.packed)
        if data.dtype != np.uint8 or data.shape != (int(self.offsets[-1]),):
            raise ValueError("Packed upper-triangle length or type is invalid")
        for i in range(self.size - 1):
            remaining = (self.size - i - 1) % 8
            if remaining and int(data[self.offsets[i + 1] - 1]) & ((1 << (8 - remaining)) - 1):
                raise ValueError("Unused bits must be zero")
        object.__setattr__(self, "packed", data.copy())

    @cached_property
    def offsets(self):
        return np.concatenate(
            ([0], np.cumsum((np.arange(self.size - 1, 0, -1) + 7) // 8, dtype=np.int64))
        )

    @cached_property
    def positive_edges(self):
        return int(np.bitwise_count(self.packed).sum(dtype=np.int64))

    @property
    def edge_count(self):
        return self.size * (self.size - 1) // 2 if self.family == "sk" else self.positive_edges

    @property
    def total_weight(self):
        return (
            2 * self.positive_edges - self.edge_count
            if self.family == "sk"
            else self.positive_edges
        )

    @cached_property
    def active_variables(self):
        if self.family == "sk":
            return self.size
        active = np.zeros(self.size, dtype=bool)
        for i in range(self.size - 1):
            row = self.row_bits(i).astype(bool)
            active[i] |= row.any()
            active[i + 1 :] |= row
        return int(active.sum())

    def row_bits(self, i):
        return np.unpackbits(
            self.packed[self.offsets[i] : self.offsets[i + 1]], count=self.size - i - 1
        )

    def matrix(self):
        matrix = np.zeros((self.size, self.size), dtype=np.int8)
        for i in range(self.size - 1):
            row = self.row_bits(i).astype(np.int8)
            matrix[i, i + 1 :] = 2 * row - 1 if self.family == "sk" else row
        # Tile the transpose to avoid strided writes across the entire large matrix.
        for a in range(0, self.size, 256):
            end = min(a + 256, self.size)
            square = matrix[a:end, a:end]
            square += square.T.copy()
            for b in range(0, a, 256):
                stop = min(b + 256, a)
                matrix[a:end, b:stop] = matrix[b:stop, a:end].T
        return matrix

    def cut(self, bits):
        values = np.asarray(bits)
        if (
            values.ndim not in (1, 2)
            or values.shape[-1] != self.size
            or not np.isin(values, (0, 1)).all()
        ):
            raise ValueError("Expected binary vectors with the original graph dimension")
        values = values.astype(np.uint8, copy=False)
        packed = np.pad(np.packbits(values, axis=-1), [(0, 0)] * (values.ndim - 1) + [(0, 1)])
        result = np.zeros(values.shape[:-1], dtype=np.int64)
        for i in range(self.size - 1):
            width = self.size - i - 1
            length = (width + 7) // 8
            first, shift = divmod(i + 1, 8)
            right = packed[..., first : first + length].copy()
            if shift:
                right = (right << shift) | (
                    packed[..., first + 1 : first + length + 1] >> (8 - shift)
                )
            different = np.where(values[..., i, None] != 0, np.bitwise_not(right), right)
            if width % 8:
                different[..., -1] &= np.uint8((255 << (8 - width % 8)) & 255)
            positive = self.packed[self.offsets[i] : self.offsets[i + 1]]
            crossing_positive = np.bitwise_count(different & positive).sum(-1, dtype=np.int64)
            if self.family == "sk":
                result += 2 * crossing_positive - np.bitwise_count(different).sum(
                    -1, dtype=np.int64
                )
            else:
                result += crossing_positive
        return result

    def save(self, filename):
        np.savez(filename, size=self.size, family=self.family, packed=self.packed)

    def write_mqlib(self, filename):
        """Write the binary input consumed by the separate dense MQLib driver."""
        with Path(filename).open("xb") as output:
            output.write(MAGIC)
            output.write(
                struct.pack(
                    "<IIQQ", self.size, KINDS[self.family], self.edge_count, len(self.packed)
                )
            )
            output.write(self.packed.tobytes())


def from_binary(filename):
    with Path(filename).open("rb") as source:
        if source.read(8) != MAGIC:
            raise ValueError("Dense input magic mismatch")
        size, kind, count, length = struct.unpack("<IIQQ", source.read(24))
        payload = source.read()
    if kind not in KINDS.values() or len(payload) != length:
        raise ValueError("Invalid dense input header")
    graph = Instance(
        size,
        next(key for key, value in KINDS.items() if value == kind),
        np.frombuffer(payload, dtype=np.uint8),
    )
    if graph.edge_count != count:
        raise ValueError("Declared edge count is incorrect")
    return graph


def instance_from_problem(problem):
    if "instance_file" in problem:
        if set(problem) != {"instance_file"}:
            raise ValueError("Input files and generator parameters cannot be mixed")
        return from_binary(problem["instance_file"])
    if set(problem) != {"size", "family", "instance_seed"}:
        raise ValueError("Dense generator needs size, family, and instance_seed")
    size, family, seed = (problem[k] for k in ("size", "family", "instance_seed"))
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or not 2 <= size <= 32768
        or family not in KINDS
    ):
        raise ValueError("Invalid dense model specification")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Invalid graph seed")
    lengths = (np.arange(size - 1, 0, -1) + 7) // 8
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    rng = np.random.Generator(np.random.PCG64(seed))
    data = rng.integers(0, 256, size=int(offsets[-1]), dtype=np.uint8)
    for i in range(size - 1):
        tail = (size - i - 1) % 8
        if tail:
            data[offsets[i + 1] - 1] &= np.uint8((255 << (8 - tail)) & 255)
    return Instance(size, family, data)


def exact_optimum(instance):
    if instance.size > 20:
        raise ValueError("Exhaustive reference is restricted to 20 variables")
    best, witness = 0, [0] * instance.size
    for start in range(0, 2 ** (instance.size - 1), 1024):
        number = np.arange(start, min(start + 1024, 2 ** (instance.size - 1)), dtype=np.uint64)
        bits = ((number[:, None] >> np.arange(instance.size, dtype=np.uint64)) & 1).astype(np.uint8)
        costs = instance.cut(bits)
        i = int(costs.argmax())
        if costs[i] > best:
            best, witness = int(costs[i]), bits[i].tolist()
    return {"optimum_cut": best, "bits": witness, "method": "integer-bitset-enumeration-v1"}
