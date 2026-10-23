"""Binary parity checks and a finite-precision soft-receiver objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCALE = 256
MAX_LLR = 32


@dataclass
class Code:
    size: int
    pointers: np.ndarray
    columns: np.ndarray

    def __post_init__(self):
        self.pointers = np.asarray(self.pointers, dtype=np.int64)
        self.columns = np.asarray(self.columns, dtype=np.int64)
        if (
            self.size < 2
            or self.pointers.ndim != 1
            or self.pointers.size < 2
            or self.columns.ndim != 1
            or self.pointers[0] != 0
            or self.pointers[-1] != len(self.columns)
            or np.any(np.diff(self.pointers) < 2)
            or np.any(self.columns < 0)
            or np.any(self.columns >= self.size)
        ):
            raise ValueError("Invalid parity-check matrix")
        for row in self.rows:
            if len(np.unique(row)) != len(row):
                raise ValueError("A check contains a duplicate variable")
        if len(np.unique(self.columns)) != self.size:
            raise ValueError("Every data bit must participate in a check")

    @property
    def checks(self):
        return len(self.pointers) - 1

    @property
    def rows(self):
        return [self.columns[self.pointers[i] : self.pointers[i + 1]] for i in range(self.checks)]

    def padded(self):
        result = np.full(
            (self.checks, int(np.diff(self.pointers).max())), self.size, dtype=np.int64
        )
        for i, row in enumerate(self.rows):
            result[i, : len(row)] = row
        return result

    def syndrome(self, bits):
        bits = np.asarray(bits)
        if bits.shape[-1] != self.size or np.any((bits != 0) & (bits != 1)):
            raise ValueError("Expected binary data bits")
        padded = np.concatenate([bits, np.zeros((*bits.shape[:-1], 1), dtype=np.int8)], axis=-1)
        return np.bitwise_xor.reduce(padded[..., self.padded()], axis=-1)

    @classmethod
    def from_rows(cls, size, rows):
        rows = [np.asarray(sorted(row), dtype=np.int64) for row in rows]
        return cls(size, np.r_[0, np.cumsum([len(row) for row in rows])], np.concatenate(rows))

    @classmethod
    def regular(cls, size, seed):
        if size % 6 or size < 12:
            raise ValueError("The (3,6) Gallager construction requires a multiple of six")
        rng = np.random.default_rng(seed)
        rows = np.concatenate([rng.permutation(size).reshape(-1, 6) for _ in range(3)])
        return cls.from_rows(size, rows)

    @classmethod
    def wifi(cls, base, lift):
        base = np.asarray(base, dtype=np.int64)
        if base.ndim != 2 or np.any(base < -1) or np.any(base >= lift):
            raise ValueError("Invalid quasi-cyclic base matrix")
        rows = []
        for block in base:
            for i in range(lift):
                rows.append(
                    [
                        j * lift + (i + int(shift)) % lift
                        for j, shift in enumerate(block)
                        if shift >= 0
                    ]
                )
        return cls.from_rows(base.shape[1] * lift, rows)

    def nullspace(self):
        """RREF over Python integers; used only by the exact small-code reference."""
        rows = [sum(1 << int(j) for j in row) for row in self.rows]
        transforms = [1 << i for i in range(self.checks)]
        pivots, rank = [], 0
        for column in range(self.size):
            pivot = next((i for i in range(rank, len(rows)) if (rows[i] >> column) & 1), None)
            if pivot is None:
                continue
            rows[rank], rows[pivot] = rows[pivot], rows[rank]
            transforms[rank], transforms[pivot] = transforms[pivot], transforms[rank]
            for i in range(len(rows)):
                if i != rank and (rows[i] >> column) & 1:
                    rows[i] ^= rows[rank]
                    transforms[i] ^= transforms[rank]
            pivots.append(column)
            rank += 1
        free = [i for i in range(self.size) if i not in pivots]
        basis = []
        for column in free:
            value = 1 << column
            for row, pivot in zip(rows[:rank], pivots, strict=True):
                if (row >> column) & 1:
                    value |= 1 << pivot
            basis.append(value)
        return rows, transforms, pivots, basis


@dataclass
class Instance:
    code: Code
    costs: np.ndarray
    targets: np.ndarray

    def __post_init__(self):
        self.costs = np.asarray(self.costs, dtype=np.int64)
        self.targets = np.asarray(self.targets, dtype=np.int8)
        if (
            self.costs.ndim != 2
            or self.costs.shape[1] != self.code.size
            or self.targets.shape != (len(self.costs), self.code.checks)
            or np.any(self.costs < 0)
            or np.any(self.costs > SCALE * MAX_LLR)
            or np.any((self.targets != 0) & (self.targets != 1))
        ):
            raise ValueError("Invalid receiver costs or target syndromes")

    @property
    def frames(self):
        return len(self.costs)

    def save(self, path):
        np.savez_compressed(
            path,
            size=self.code.size,
            pointers=self.code.pointers,
            columns=self.code.columns,
            costs=self.costs,
            targets=self.targets,
            scale=SCALE,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as packet:
            if set(packet.files) != {"size", "pointers", "columns", "costs", "targets", "scale"}:
                raise ValueError("Solver inputs may contain only checks, syndrome and reliability")
            if int(packet["scale"]) != SCALE:
                raise ValueError("Unknown receiver quantization")
            return cls(
                Code(int(packet["size"]), packet["pointers"], packet["columns"]),
                packet["costs"],
                packet["targets"],
            )

    def cost(self, bits):
        bits = np.asarray(bits)
        if (
            bits.shape[0] != self.frames
            or bits.shape[-1] != self.code.size
            or np.any((bits != 0) & (bits != 1))
        ):
            raise ValueError("Wrong frame or variable count")
        weights = self.costs.reshape((self.frames,) + (1,) * (bits.ndim - 2) + (self.code.size,))
        return np.sum(bits * weights, axis=-1, dtype=np.int64)

    def violations(self, bits):
        bits = np.asarray(bits)
        target = self.targets.reshape((self.frames,) + (1,) * (bits.ndim - 2) + (self.code.checks,))
        return np.count_nonzero(self.code.syndrome(bits) != target, axis=-1)

    def exact(self):
        if self.code.size > 30:
            raise ValueError("Exact reference is restricted to small codes")
        rows, transforms, pivots, basis = self.code.nullspace()
        if len(basis) > 20:
            raise ValueError("Exact coset enumeration exceeds its dimension limit")
        words = [0]
        for vector in basis:
            words += [w ^ vector for w in words]
        null_bits = np.asarray(
            [[(w >> i) & 1 for i in range(self.code.size)] for w in words], dtype=np.int8
        )
        best_bits, best_cost = [], []
        for costs, target in zip(self.costs, self.targets, strict=True):
            target_word = sum(int(x) << i for i, x in enumerate(target))
            rhs = [(t & target_word).bit_count() % 2 for t in transforms]
            if any(rhs[len(pivots) :]):
                raise ValueError("Inconsistent syndrome")
            particular = np.zeros(self.code.size, dtype=np.int8)
            particular[pivots] = rhs[: len(pivots)]
            candidates = null_bits ^ particular
            values = candidates @ costs
            selected = int(np.argmin(values))
            best_bits.append(candidates[selected])
            best_cost.append(int(values[selected]))
        return np.asarray(best_bits), np.asarray(best_cost), len(pivots)


def channel_batch(code, *, frames, esn0_db, seed):
    """Keep signed observations and transmitted truth outside decoder inputs."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(1 / (2 * 10 ** (esn0_db / 10)))
    noise = rng.standard_normal((frames, code.size))
    received = 1 + sigma * noise
    llr = 2 * received / sigma**2
    hard = (llr < 0).astype(np.int8)
    costs = np.rint(np.minimum(np.abs(llr), MAX_LLR) * SCALE).astype(np.int64)
    instance = Instance(code, costs, code.syndrome(hard))
    return instance, {
        "llr": llr,
        "hard_decision": hard,
        "true_error": hard,
        "received": received,
        "noise": noise,
        "sigma": sigma,
        "esn0_db": esn0_db,
    }


def quadratic_initial(instance, data):
    if np.any(np.diff(instance.code.pointers) != 6):
        raise ValueError("The audited two-slack-bit encoding requires degree-six checks")
    data = np.asarray(data, dtype=np.int8)
    counts = data[..., instance.code.padded()].sum(-1)
    target = instance.targets[:, None, :]
    slack = np.clip((counts - target) // 2, 0, 3)
    auxiliaries = np.stack([slack & 1, (slack >> 1) & 1], axis=-1)
    return np.concatenate([data, auxiliaries.reshape(*data.shape[:-1], -1)], axis=-1)


def quadratic_residual(instance, bits):
    data = bits[..., : instance.code.size]
    aux = bits[..., instance.code.size :].reshape(*bits.shape[:-1], instance.code.checks, 2)
    return (
        data[..., instance.code.padded()].sum(-1)
        - 2 * aux[..., 0]
        - 4 * aux[..., 1]
        - instance.targets[:, None, :]
    )
