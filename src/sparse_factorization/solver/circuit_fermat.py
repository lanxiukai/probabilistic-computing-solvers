"""Public difference-of-squares root proposals with full support through 80 bits."""

from __future__ import annotations

import math

import torch

from .circuit_modular import ModularRootProposal
from .circuit_quotient import unsigned_product_limbs
from .circuit_root_filter import FERMAT_MODES, FILTER_METRICS, FilteredQuotientRootProposal

FERMAT_METRICS = tuple(name.replace("quotient", "fermat") for name in FILTER_METRICS)


class FilteredFermatRootProposal(FilteredQuotientRootProposal):
    """An independent eight-word candidate, followed by the existing positive filter.

    Draw x uniformly in a power-of-two envelope of the public feasible sum
    interval. Round sqrt(x*x-N) downward to the opposite parity from x and
    choose either orientation of (x-y,x+y). Invalid indices and 1/16 of labels
    fall back to independent uniform n-bit inputs. The sum determines x
    uniquely, but a diagonal pair has two orientation preimages.

    This is an arithmetic-assisted MH proposal, not a new factoring identity.
    It consumes eight uint32 words per candidate even for inputs below 33 bits.
    """

    maximum_width = 40
    metric_names = FERMAT_METRICS
    uniform_orbit = False

    def __init__(self, width: int, target: int, device, *, mode: str):
        if mode not in FERMAT_MODES:
            raise ValueError("Unknown Fermat root-filter protocol")
        ModularRootProposal.__init__(self, width, device)
        self.mask = (1 << width) - 1
        if not 1 <= target <= self.mask**2 or not target & 1:
            raise ValueError("Require a positive odd target within the input registers")
        self.target = target
        self.target_high, self.target_low = divmod(target, 1 << 32)
        self.x_min = math.isqrt(target)
        self.x_min += self.x_min**2 < target
        self.x_max = (self.mask**2 + target) // (2 * self.mask)
        self.x_count = max(0, self.x_max - self.x_min + 1)
        self.envelope = 1 << (max(1, self.x_count) - 1).bit_length()
        self.uniform_weight = 1 - (15 / 16) * self.x_count / self.envelope
        tail = self.uniform_weight * 2.0 ** (-2 * width)
        oriented = 15 / (32 * self.envelope)
        self.log_masses = torch.tensor(
            [math.log(tail + k * oriented) for k in (0, 1, 2)],
            dtype=torch.float64,
            device=device,
        )
        self.retries, self.filter_bits = 256, 12
        self.filter_mask = (1 << self.filter_bits) - 1
        self.noise_words = 8 * self.retries
        self.filter_log_weights = torch.tensor(
            [-self.filter_bits * math.log(2), 0.0], dtype=torch.float64, device=device
        )
        self.metadata = {
            "transformation": "bounded_weighted_fermat_bank_or_full_state_identity",
            "input_width": width,
            "filter_mode": mode,
            "root_candidate_count": self.retries,
            "filter_bits": self.filter_bits,
            "noise_words": self.noise_words,
            "candidate_word_layout": {
                "0": "bit0 orientation; bits1..4 biased label (15 of 16)",
                "1,2": "low32, high8 envelope index, then mask to D-1",
                "3,4": "low32, high8 uniform left, then mask to n bits",
                "5,6": "low32, high8 uniform right, then mask to n bits",
                "7": "low12 filter coin",
            },
            "base_candidate_distribution": {
                "x_min": self.x_min,
                "x_max": self.x_max,
                "x_count": self.x_count,
                "power_of_two_envelope": self.envelope,
                "uniform_fallback_probability": self.uniform_weight,
                "root_mass": "u/2**(2*n) + 15/(32*D) * support * (1+diagonal)",
                "map": "y=floor_sqrt(x*x-N), decrement to opposite parity; roots=x+-y",
            },
            "root_correction": "log(Q0(old)*w(old))-log(Q0(new)*w(new))",
            "normalization": "State-independent S_K(Z) cancels",
            "filter_weight": "1 for low12 product match, otherwise 2**-12",
            "all_failed_action": "Retain the entire old physical state",
            "candidate_words_generated": "All K candidates including after first pass",
            "integer_arithmetic": "Exact high/low product and subtraction; corrected FP64 sqrt",
            "full_product_divisibility_filter_used": False,
            "factor_certificate_used": False,
            "target_information_used": "Public N and input widths only",
        }

    def radius(self, x):
        """Exact parity-adjusted floor sqrt; all integer intermediates fit int64.

        For x<=2**40-1 the high limb has at most 48 bits. Subtract before
        conversion to FP64 to avoid catastrophic cancellation near sqrt(N).
        The sqrt estimate differs by at most one integer; compare exact limbs
        on both sides. The final parity adjustment makes both roots odd.
        """
        high, low = unsigned_product_limbs(x, x)
        delta_low = low - self.target_low
        delta_high = high - self.target_high - (delta_low < 0).long()
        delta_low = delta_low & 0xFFFFFFFF
        approximate = delta_high.double() * 4294967296.0 + delta_low.double()
        y = approximate.sqrt().long()
        yh, yl = unsigned_product_limbs(y, y)
        y -= ((yh > delta_high) | ((yh == delta_high) & (yl > delta_low))).long()
        yh, yl = unsigned_product_limbs(y + 1, y + 1)
        y += ((yh < delta_high) | ((yh == delta_high) & (yl <= delta_low))).long()
        return y - ((x + y + 1) & 1)

    def on_support(self, pairs):
        total = pairs[:, 0] + pairs[:, 1]
        x = total >> 1
        y = (pairs[:, 0] - pairs[:, 1]).abs() >> 1
        expected = self.radius(x.clamp(self.x_min, max(self.x_min, self.x_max)))
        return ((pairs & 1) == 1).all(1) & (x >= self.x_min) & (x <= self.x_max) & (y == expected)

    def log_probability(self, roots):
        pairs = self.decode(roots)
        preimages = self.on_support(pairs).long() * (1 + (pairs[:, 0] == pairs[:, 1]).long())
        return self.log_masses[preimages]

    @staticmethod
    def word40(low, high):
        return low | ((high & 255) << 32)

    def candidate_pairs(self, words):
        flat = words.reshape(-1, 8)
        index = self.word40(flat[:, 1], flat[:, 2]) & (self.envelope - 1)
        valid_index = index < self.x_count
        biased = (((flat[:, 0] >> 1) & 15) != 0) & valid_index
        x = self.x_min + torch.where(valid_index, index, 0)
        y = self.radius(x)
        y = torch.where((flat[:, 0] & 1) == 0, y, -y)
        centered = torch.stack((x - y, x + y), 1)
        uniform = (
            torch.stack(
                (self.word40(flat[:, 3], flat[:, 4]), self.word40(flat[:, 5], flat[:, 6])), 1
            )
            & self.mask
        )
        pairs = torch.where(biased[:, None], centered, uniform)
        matches = self.filter_matches(pairs)
        eligible = matches | ((flat[:, 7] & self.filter_mask) == 0)
        return pairs, biased, matches, eligible

    def propose_filtered(self, roots, words):
        pairs, biased, matches, eligible = self.candidate_pairs(words)
        replicas = len(roots)
        eligible = eligible.reshape(replicas, self.retries)
        selected = eligible.int().argmax(1)
        rows = torch.arange(replicas, device=roots.device)
        pair = pairs.reshape(replicas, self.retries, 2)[rows, selected]
        proposed = (((pair[:, :, None] >> self.shifts) & 1) * 2 - 1).reshape_as(roots)
        return proposed.to(roots.dtype), {
            "valid": eligible.any(1),
            "biased": biased.reshape(replicas, self.retries)[rows, selected],
            "eligible_count": eligible.sum(1),
            "filter_match": matches.reshape(replicas, self.retries)[rows, selected],
        }
