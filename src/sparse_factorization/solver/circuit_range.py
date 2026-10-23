"""Symmetric finite-retry redraws inside a public factor-width interval."""

from __future__ import annotations

import torch

from .circuit_modular import ARITHMETIC_METRICS, UnitGroupRootProposal

WIDE_PUBLIC_RANGE_MODES = {"public_range24_control_v1", "public_range24_first8_v1"}
PUBLIC_RANGE_MODES = {"public_range_control_v1", "public_range_first8_v1"} | WIDE_PUBLIC_RANGE_MODES
RANGE_METRICS = ARITHMETIC_METRICS + (
    "current_in_public_range",
    "eligible_in_public_range",
    "selected_in_public_range",
    "eligible_selected_in_public_range",
    "range_valid_draws",
    "range_draw_found",
    "range_identity_outside",
    "range_identity_exhausted",
)


class PublicRangeRootProposal(UnitGroupRootProposal):
    """Keep the current product residue; never project it to the target residue.

    For roots inside the odd public interval, eight iid power-of-two-envelope
    draws propose the first pair inside the same interval. Outside roots, or
    eight failures, leave roots unchanged. Within a residue class the marginal
    mass of every admitted pair is identical, so reverse masses cancel. This
    is a marginal-proposal argument, not a same-label involution.
    """

    noise_words = 9
    metric_names = RANGE_METRICS
    draws = 8

    def __init__(self, width: int, target: int, device, *, bounded: bool):
        super().__init__(width, device)
        if not 1 <= target <= self.mask**2:
            raise ValueError("Public range requires a positive target within the input widths")
        self.bounded = bounded
        self.uniform_orbit = not bounded
        self.lower = (target + self.mask - 1) // self.mask
        self.first_odd = self.lower | 1
        self.odd_count = ((1 << width) - self.first_odd + 1) // 2
        self.envelope = 1 << (self.odd_count - 1).bit_length()
        self.metadata = {
            "transformation": "first_eight_public_interval_redraws"
            if bounded
            else "uniform_unit_control",
            "input_width": width,
            "lower_bound": self.lower,
            "first_odd": self.first_odd,
            "odd_count": self.odd_count,
            "power_of_two_envelope": self.envelope,
            "draws": self.draws,
            "noise_words": self.noise_words,
            "target_information_used": "public interval only; current residue retained",
            "factor_certificate_used": False,
            "mixture_probability": 0.5,
            "outside_interval_or_even": "identity in bounded component; guided half retained",
            "exhausted_draws": "identity root move; full body still regenerated",
            "selection_density": (
                "Symmetric marginal first-admissible proposal within each residue class"
            ),
        }

    def in_range(self, pair):
        return ((pair >= self.lower) & (pair <= self.mask) & ((pair & 1) == 1)).all(-1)

    def propose_with_details(self, roots, words):
        pair = self.decode(roots)
        inside = self.in_range(pair)
        if not self.bounded:
            proposed = super().propose(roots, words)
            # No bounded draws are evaluated in the uniform-unit control.
            details = torch.stack(
                (torch.full_like(pair[:, 0], -1), torch.zeros_like(pair[:, 0]), inside), 1
            )
            return proposed, details
        index = words[:, 1:9] & (self.envelope - 1)
        left = self.first_odd + 2 * index
        residue = pair[:, 0] * pair[:, 1] & self.mask
        right = residue[:, None] * self.inverse(left & self.mask) & self.mask
        valid = (index < self.odd_count) & (right >= self.lower) & ((right & 1) == 1)
        first = valid.long().argmax(1)
        found = valid.any(1)
        selected = torch.stack(
            (left.gather(1, first[:, None]).squeeze(1), right.gather(1, first[:, None]).squeeze(1)),
            1,
        )
        selected = torch.where((inside & found)[:, None], selected, pair)
        proposed = (
            (((selected[:, :, None] >> self.shifts) & 1) * 2 - 1).reshape_as(roots).to(roots.dtype)
        )
        details = torch.stack((torch.where(inside & found, first, -1), valid.sum(1), inside), 1)
        return proposed, details

    def propose(self, roots, words):
        return self.propose_with_details(roots, words)[0]

    def observe_range(self, old_roots, new_roots, branch, target, energy, accepted, details):
        base = super().observe(old_roots, new_roots, branch, target, energy, accepted)
        inside = details[:, 2].bool()
        eligible = base[:, 3].bool()
        found = details[:, 0] >= 0
        bounded = branch & self.bounded
        extra = torch.stack(
            (
                inside,
                eligible & inside,
                branch & inside,
                branch & inside & eligible,
                details[:, 1] * bounded,
                bounded & found,
                bounded & ~inside,
                bounded & inside & ~found,
            ),
            1,
        ).long()
        return torch.cat((base, extra), 1)


class WidePublicRangeRootProposal(PublicRangeRootProposal):
    """The unchanged P7.30 law, with a separately validated 24-bit input limit.

    Masked operands are below 2**24. Products, including the signed Newton
    update, have magnitude below 2**48 and fit int64. Five Newton iterations
    provide 32 inverse bits. This does not enable 32-bit inputs / 64-bit targets.
    """

    maximum_width = 24
