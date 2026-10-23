"""Four-word CRT Fermat proposals through unsigned 64-bit input registers."""

from __future__ import annotations

import math

import torch

from .circuit_fermat_crt import FilteredCRTFermatRootProposal, residue_sets
from .integer_words import (
    exact_product_match,
    floor_sqrt_bounded_words,
    subtract_words,
    unsigned_square_words,
)


def wide_residue_sets(target, count):
    if target.bit_length() <= 92:
        return residue_sets(target, count)
    # Minimize the actual finite-word envelope, including complete CRT cycles.
    # A modulus much larger than the interval can waste almost one prime factor.
    size = max(1, count)
    best_key = (1 << (size - 1).bit_length(), 1)
    best = (1, [(1, [0])])
    modulus, cardinality, tables = 1, 1, []
    for base in (64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43):
        squares = {x * x % base for x in range(base)}
        allowed = [x for x in range(base) if (x * x - target) % base in squares]
        tables.append((base, allowed))
        modulus *= base
        cardinality *= len(allowed)
        indices = cardinality * ((size + modulus - 1) // modulus)
        key = (1 << (indices - 1).bit_length(), len(tables))
        if key < best_key:
            best_key, best = key, (modulus, list(tables))
        if modulus >= size:
            return best
    raise ValueError("Public interval exceeds the validated CRT modulus range")


def signed64(value):
    return value if value < 2**63 else value - 2**64


class FilteredWideCRTFermatRootProposal(FilteredCRTFermatRootProposal):
    maximum_width = 64
    mode_name = "fermat_crt128_filter256_v1"
    residue_builder = staticmethod(wide_residue_sets)

    def __init__(self, width, target, device, *, mode):
        self.use_triton_sqrt = False
        if width > 46 and target < 3 * ((1 << width) - 1) ** 2 // 4:
            raise ValueError("Wide CRT currently requires public N >= 3*m*m/4")
        super().__init__(width, target, device, mode=mode)
        self.residue_count = self.crt_count
        self.crt_cycles = max(1, (self.x_count + self.crt_modulus - 1) // self.crt_modulus)
        self.crt_count *= self.crt_cycles
        self.envelope = 1 << (self.crt_count - 1).bit_length()
        tail = 1 / (16 * 2 ** (2 * width))
        oriented = 15 / (32 * self.envelope)
        self.log_masses = torch.tensor(
            [math.log(tail + k * oriented) for k in (0, 1, 2)],
            dtype=torch.float64,
            device=device,
        )
        self.target_words = tuple((target >> (32 * i)) & 0xFFFFFFFF for i in range(4))
        self.signed_min, self.signed_max = signed64(self.x_min), signed64(self.x_max)
        self.input_mask = signed64(self.mask)
        self.metadata.update(
            {
                "integer_arithmetic": (
                    "Four little-endian uint32 words; exact-residual Newton "
                    "sqrt correction and exact final comparisons"
                ),
                "unsigned_storage": (
                    "Roots use int64 bit patterns; negative values encode unsigned values >=2**63"
                ),
                "wide_target_boundary": (
                    "For input width >46 require public N >=3*m*m/4; x*x-N <2**124"
                ),
                "random_namespace": "p748_crt128_circuit_cuda_philox_v9",
            }
        )
        layout = self.metadata["candidate_word_layout"]
        layout["3,4"] = "low32, high32 uniform left; mask to n bits, retain bit63"
        layout["5,6"] = "low32, high32 uniform right; mask to n bits, retain bit63"
        self.metadata["base_candidate_distribution"].update(
            crt_residue_count=self.residue_count,
            crt_cycle_count=self.crt_cycles,
            crt_index_count=self.crt_count,
            power_of_two_envelope=self.envelope,
            modulus_selection=(
                "For N>92 bits minimize power-of-two envelope over public prime "
                "prefixes and CRT cycle counts; ties prefer fewer tables"
            ),
            map="x=x_min+(CRT(index%C)-x_min)%M+(index//C)*M; reject invalid index or interval",
        )

    def radius(self, x):
        if self.use_triton_sqrt and x.is_cuda:
            from .wide_execution_triton import radius

            return radius(x, self.target)
        delta = subtract_words(unsigned_square_words(x), self.target_words)
        y = floor_sqrt_bounded_words(delta)
        return y - ((x + y + 1) & 1)

    def index_to_x(self, index):
        rank = index
        value = torch.zeros_like(index)
        for table in self.crt_tables:
            value = (value + table[rank % len(table)]) % self.crt_modulus
            rank = rank // len(table)
        offset = (
            value - (self.x_min % self.crt_modulus)
        ) % self.crt_modulus + rank * self.crt_modulus
        return self.signed_min + offset

    def on_support(self, pairs):
        a, b = pairs.unbind(1)
        half_mask = 0x7FFFFFFFFFFFFFFF
        x = ((a >> 1) & half_mask) + ((b >> 1) & half_mask) + (a & b & 1)
        less = (a ^ -(2**63)) < (b ^ -(2**63))
        difference = torch.where(less, b - a, a - b)
        y = (difference >> 1) & half_mask
        expected = self.radius(x.clamp(self.signed_min, max(self.signed_min, self.signed_max)))
        valid = (
            ((pairs & 1) == 1).all(1)
            & (x >= self.signed_min)
            & (x <= self.signed_max)
            & (y == expected)
        )
        # Reduce unsigned x by using two positive 32-bit words.
        low, high = x & 0xFFFFFFFF, (x >> 32) & 0xFFFFFFFF
        for base, mask in self.qr_masks:
            remainder = (low % base + (high % base) * ((1 << 32) % base)) % base
            signed_mask = mask if mask < 2**63 else mask - 2**64
            valid = valid & (((signed_mask >> remainder) & 1) != 0)
        return valid

    def candidate_pairs(self, words):
        flat = words.reshape(-1, 8)
        index = self.word63(flat[:, 1], flat[:, 2]) & (self.envelope - 1)
        x = self.index_to_x(index)
        uniform_label = ((flat[:, 0] >> 1) & 15) == 0
        biased = (
            ~uniform_label
            & (index < self.crt_count)
            & (x >= self.signed_min)
            & (x <= self.signed_max)
        )
        safe_x = torch.where(biased, x, self.signed_min)
        y = self.radius(safe_x)
        y = torch.where((flat[:, 0] & 1) == 0, y, -y)
        centered = torch.stack((safe_x - y, safe_x + y), 1)
        uniform = (
            torch.stack((flat[:, 3] | (flat[:, 4] << 32), flat[:, 5] | (flat[:, 6] << 32)), 1)
            & self.input_mask
        )
        pairs = torch.where(biased[:, None], centered, uniform)
        matches = self.filter_matches(pairs)
        eligible = (uniform_label | biased) & (matches | ((flat[:, 7] & self.filter_mask) == 0))
        return pairs, biased, matches, eligible

    def observe_filtered(self, old_roots, new_roots, energy, accepted, details):
        old, new = self.decode(old_roots), self.decode(new_roots)
        valid, biased = details["valid"], details["biased"]
        match = valid & exact_product_match(new, self.target)
        changed = (new_roots != old_roots).sum(1)
        zero = valid & (energy == 0)
        return torch.stack(
            (
                torch.ones_like(changed),
                torch.ones_like(changed),
                valid & biased,
                valid & ~biased,
                valid * changed,
                self.on_support(old),
                valid & self.on_support(new),
                match,
                match,
                zero,
                accepted & zero,
                accepted & (changed > 0),
                accepted & (changed > 0),
                accepted & biased & (changed > 0),
                torch.full_like(changed, self.retries),
                details["eligible_count"],
                ~valid,
                valid & details["filter_match"],
                self.filter_matches(old),
            ),
            1,
        ).long()
