"""CRT-sieved full-support Fermat proposals with explicit null candidate draws."""

from __future__ import annotations

import math

import torch

from .circuit_fermat import FilteredFermatRootProposal


def residue_sets(target, interval_size, bases=(64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)):
    """Small public QR tables; never enumerate or test complete factor pairs."""
    modulus = 1
    result = []
    for base in bases:
        squares = {x * x % base for x in range(base)}
        allowed = [x for x in range(base) if (x * x - target) % base in squares]
        result.append((base, allowed))
        modulus *= base
        if modulus >= max(1, interval_size):
            return modulus, result
    raise ValueError("CRT modulus does not cover the public interval")


class FilteredCRTFermatRootProposal(FilteredFermatRootProposal):
    """Bijective mixed-radix draw from modular necessary conditions.

    The 1/16 uniform label is always available. Other labels draw a CRT
    residue uniformly through a power-of-two index envelope. Invalid indices
    or translated residues outside the public interval are null draws, NOT
    uniform fallbacks. This makes the subprobability mass explicit without
    counting the intersection of a large CRT set with the interval.
    """

    maximum_width = 46
    mode_name = "fermat_crt_filter256_v1"
    residue_builder = staticmethod(residue_sets)

    def __init__(self, width, target, device, *, mode):
        if mode != self.mode_name:
            raise ValueError("Unknown CRT Fermat protocol")
        super().__init__(width, target, device, mode=mode)
        self.crt_modulus, tables = self.residue_builder(target, self.x_count)
        self.crt_count = math.prod(len(a) for _, a in tables)
        self.envelope = 1 << (self.crt_count - 1).bit_length()
        if self.envelope > 2**63 or self.crt_modulus >= 2**62:
            raise ValueError("CRT arithmetic exceeds the validated signed-limb range")
        self.crt_tables = []
        self.qr_masks = []
        for base, residues in tables:
            partial = self.crt_modulus // base
            weight = partial * pow(partial, -1, base)
            self.crt_tables.append(
                torch.tensor([(r * weight) % self.crt_modulus for r in residues], device=device)
            )
            self.qr_masks.append((base, sum(1 << r for r in residues)))
        self.uniform_weight = 1 / 16
        tail = self.uniform_weight * 2.0 ** (-2 * width)
        oriented = 15 / (32 * self.envelope)
        self.log_masses = torch.tensor(
            [math.log(tail + k * oriented) for k in (0, 1, 2)],
            dtype=torch.float64,
            device=device,
        )
        self.metadata.update(
            {
                "transformation": "crt_sieved_fermat_bank_or_full_state_identity",
                "candidate_word_layout": {
                    "0": "bit0 orientation; bits1..4 biased label (15 of 16)",
                    "1,2": "low32, high31 CRT envelope index, then mask to D-1",
                    "3,4": "low32, high31 uniform left, then mask to n bits",
                    "5,6": "low32, high31 uniform right, then mask to n bits",
                    "7": "low12 filter coin",
                },
                "base_candidate_distribution": {
                    "x_min": self.x_min,
                    "x_max": self.x_max,
                    "x_count": self.x_count,
                    "crt_modulus": self.crt_modulus,
                    "crt_residue_count": self.crt_count,
                    "crt_tables": [{"modulus": p, "residues": r} for p, r in tables],
                    "power_of_two_envelope": self.envelope,
                    "uniform_label_probability": self.uniform_weight,
                    "root_subprobability_mass": "1/(16*2**(2*n)) + 15/(32*D)*support*(1+diagonal)",
                    "invalid_index_or_interval": "Null candidate, cannot pass the bank filter",
                    "map": (
                        "x=x_min+(CRT(index)-x_min)%M; "
                        "original parity-adjusted sqrt and orientation"
                    ),
                },
                "normalization": "State-independent S_K(Z) includes nulls and cancels",
                "integer_arithmetic": "Exact limbs through 46-bit operands; corrected FP64 sqrt",
            }
        )

    @staticmethod
    def word63(low, high):
        return low | ((high & 0x7FFFFFFF) << 32)

    def index_to_x(self, index):
        rank = index
        value = torch.zeros_like(index)
        for table in self.crt_tables:
            value = (value + table[rank % len(table)]) % self.crt_modulus
            rank = rank // len(table)
        return self.x_min + (value - self.x_min) % self.crt_modulus

    def on_support(self, pairs):
        x = (pairs[:, 0] + pairs[:, 1]) >> 1
        valid = super().on_support(pairs)
        for base, mask in self.qr_masks:
            # The modulus-64 mask can have bit 63 set; view it as signed bits.
            signed_mask = mask if mask < 2**63 else mask - 2**64
            valid = valid & (((signed_mask >> (x % base)) & 1) != 0)
        return valid

    def candidate_pairs(self, words):
        flat = words.reshape(-1, 8)
        index = self.word63(flat[:, 1], flat[:, 2]) & (self.envelope - 1)
        x = self.index_to_x(index)
        uniform_label = ((flat[:, 0] >> 1) & 15) == 0
        biased = ~uniform_label & (index < self.crt_count) & (x <= self.x_max)
        safe_x = torch.where(biased, x, self.x_min)
        y = self.radius(safe_x)
        y = torch.where((flat[:, 0] & 1) == 0, y, -y)
        centered = torch.stack((safe_x - y, safe_x + y), 1)
        uniform = (
            torch.stack(
                (self.word63(flat[:, 3], flat[:, 4]), self.word63(flat[:, 5], flat[:, 6])), 1
            )
            & self.mask
        )
        pairs = torch.where(biased[:, None], centered, uniform)
        matches = self.filter_matches(pairs)
        eligible = (uniform_label | biased) & (matches | ((flat[:, 7] & self.filter_mask) == 0))
        return pairs, biased, matches, eligible
