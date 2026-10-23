"""Full-support independent root proposals guided by a public integer quotient."""

from __future__ import annotations

import math

import torch

from .circuit_modular import ModularRootProposal

QUOTIENT_MODES = {"quotient_independence_v1", "quotient_only_v1"}
QUOTIENT_METRICS = (
    "proposals",
    "quotient_branch_selected",
    "biased_root_selected",
    "fallback_root_selected",
    "changed_root_bits",
    "old_on_quotient_support",
    "proposed_on_quotient_support",
    "proposed_root_product_match",
    "quotient_root_product_match",
    "proposed_zero_energy",
    "accepted_zero_energy",
    "accepted_root_changes",
    "accepted_quotient_root_changes",
    "accepted_biased_root_changes",
)


def unsigned_product_limbs(left: torch.Tensor, right: torch.Tensor):
    """Exact product for nonnegative operands up to 2**40 (inclusive).

    Return (product >> 32, product & (2**32-1)), held in signed int64.
    The high component may have 49 bits; it is not truncated to uint32.
    Low products stay below 2**32, cross terms below 2**41 and the high
    component at most 2**48. No signed-overflow behavior is assumed.
    """
    low_product = (left & 65535) * (right & 65535)
    middle = (left >> 16) * (right & 65535) + (left & 65535) * (right >> 16)
    middle += low_product >> 16
    low = ((middle & 65535) << 16) | (low_product & 65535)
    high = (left >> 16) * (right >> 16) + (middle >> 16)
    return high, low


class QuotientRootProposal(ModularRootProposal):
    """A state-independent proposal with an exact finite-word mass formula.

    Given the arithmetic component, bits 1..4 of word 0 choose the public
    quotient branch with probability 15/16. One envelope index comes from
    word 1. An invalid index, or the other 1/16, uses independent uniform
    roots from words 2 and 3. This gives every physical root assignment a
    positive mass. Words 4..8 are consumed to retain the existing v5 layout.

    No divisibility filter, known factor, product-residue projection or
    root clamp is used. The root log-density ratio is required in MH.
    """

    maximum_width = 32
    noise_words = 9
    metric_names = QUOTIENT_METRICS
    uniform_orbit = False

    def __init__(self, width: int, target: int, device, *, pure: bool = False):
        super().__init__(width, device)
        self.mask = (1 << width) - 1
        if target < 1 or target > self.mask**2 or not target & 1:
            raise ValueError("Quotient roots require a positive odd target within the widths")
        self.target = target
        self.target_high, self.target_low = divmod(target, 1 << 32)
        self.lower = (target + self.mask - 1) // self.mask
        self.first_odd = self.lower | 1
        self.odd_count = ((1 << width) - self.first_odd + 1) // 2
        self.envelope = 1 << (self.odd_count - 1).bit_length()
        self.uniform_weight = 1.0 - (15.0 / 16.0) * self.odd_count / self.envelope
        self.uniform_point_mass = self.uniform_weight * 2.0 ** (-2 * width)
        self.biased_point_mass = (15.0 / 16.0) / self.envelope
        self.log_masses = torch.tensor(
            [
                math.log(self.uniform_point_mass),
                math.log(self.uniform_point_mass + self.biased_point_mass),
            ],
            dtype=torch.float64,
            device=device,
        )
        self.metadata = {
            "transformation": "independent_uniform_tail_and_public_odd_quotient",
            "input_width": width,
            "lower_bound": self.lower,
            "first_odd": self.first_odd,
            "odd_count": self.odd_count,
            "power_of_two_envelope": self.envelope,
            "outer_arithmetic_probability": 1.0 if pure else 0.5,
            "biased_label_probability": 15.0 / 16.0,
            "uniform_fallback_probability": self.uniform_weight,
            "root_mass": "uniform_weight/2**(2*n) + (15/16)/D * quotient_support_indicator",
            "root_correction": "log q(old roots) - log q(proposed roots)",
            "quotient": "max(first_odd, min(2**n-1, floor(N/a)) | 1)",
            "integer_arithmetic": (
                "FP64 quotient estimate corrected with exact 16-bit-limb products"
            ),
            "noise_words": self.noise_words,
            "unused_words": [4, 5, 6, 7, 8],
            "factor_certificate_used": False,
            "divisibility_filter_used": False,
            "target_information_used": "Public target and structural widths only",
        }

    def clipped_floor_quotient(self, left):
        """Return min(mask, N // max(1, left)) exactly through 32-bit inputs.

        Where the quotient is not clipped, FP64 roundoff is below 2**-19,
        hence its integer estimate is at most one away. Exact comparisons
        correct either rounding direction. Clipping occurs before int64
        conversion, including for zero/out-of-interval old roots.
        """
        denominator = left.clamp_min(1)
        estimate = (float(self.target) / denominator.double()).clamp(0, self.mask).long()
        high, low = unsigned_product_limbs(denominator, estimate)
        above = (high > self.target_high) | ((high == self.target_high) & (low > self.target_low))
        estimate = estimate - above.long()
        high, low = unsigned_product_limbs(denominator, estimate + 1)
        below = (high < self.target_high) | ((high == self.target_high) & (low <= self.target_low))
        return (estimate + below.long()).clamp(0, self.mask)

    def center(self, left):
        return (self.clipped_floor_quotient(left) | 1).clamp_min(self.first_odd)

    def on_support(self, pair):
        left, right = pair[:, 0], pair[:, 1]
        return (left >= self.first_odd) & ((left & 1) == 1) & (right == self.center(left))

    def log_probability(self, roots):
        on_support = self.on_support(self.decode(roots))
        return self.log_masses[on_support.long()]

    def propose_with_details(self, roots, words):
        index = words[:, 1] & (self.envelope - 1)
        biased = ((words[:, 0] >> 1) & 15) != 0
        biased &= index < self.odd_count
        left = self.first_odd + 2 * index
        # Invalid envelope indices never enter the integer arithmetic path.
        left = torch.where(index < self.odd_count, left, self.first_odd)
        center_pair = torch.stack((left, self.center(left)), 1)
        uniform_pair = words[:, 2:4] & self.mask
        pair = torch.where(biased[:, None], center_pair, uniform_pair)
        proposed = (((pair[:, :, None] >> self.shifts) & 1) * 2 - 1).reshape_as(roots)
        return proposed.to(roots.dtype), biased

    def propose(self, roots, words):
        return self.propose_with_details(roots, words)[0]

    def observe_quotient(self, old_roots, new_roots, branch, energy, accepted, biased):
        old, new = self.decode(old_roots), self.decode(new_roots)
        high, low = unsigned_product_limbs(new[:, 0], new[:, 1])
        match = (high == self.target_high) & (low == self.target_low)
        changed = (new_roots != old_roots).sum(1)
        zero = energy == 0
        return torch.stack(
            (
                torch.ones_like(changed),
                branch,
                branch & biased,
                branch & ~biased,
                changed,
                self.on_support(old),
                self.on_support(new),
                match,
                branch & match,
                zero,
                accepted & zero,
                accepted & (changed > 0),
                accepted & branch & (changed > 0),
                accepted & branch & biased & (changed > 0),
            ),
            1,
        ).long()
