"""Reversible paired-root moves preserving the current low product residue."""

from __future__ import annotations

import torch

UNIT_GROUP_MODES = {"unit_group_control_v1", "unit_group_mixture_v1"}
ARITHMETIC_METRICS = (
    "proposals",
    "unit_group_selected",
    "changed_root_bits",
    "eligible_current_residue",
    "eligible_unit_group_proposals",
    "proposed_root_product_match",
    "unit_group_root_product_match",
    "proposed_zero_energy",
    "accepted_zero_energy",
    "accepted_root_changes",
    "accepted_unit_group_root_changes",
)


class ModularRootProposal:
    """Flip one odd input's bit and compensate the other input modulo 2**k.

    A label (side, k, bit) defines an involution. Even selected inputs are
    identity moves. Labels are drawn independently of state, so their actual
    finite-word masses cancel. The current residue is preserved, not projected
    to the target. No factor certificate or target value is used here.
    """

    noise_words = 4
    maximum_width = 20

    def __init__(self, width: int, device):
        if not 2 <= width <= self.maximum_width:
            raise ValueError(
                f"Modular roots support equal input widths from two to {self.maximum_width} bits"
            )
        self.width = width
        self.shifts = torch.arange(width, dtype=torch.int64, device=device)
        self.weights = 1 << self.shifts
        self.prefixes = torch.tensor([min(k, width) for k in (4, 8, 12, width)], device=device)
        self.metadata = {
            "transformation": "single_input_bit_flip_with_modular_compensation",
            "prefix_widths": self.prefixes.cpu().tolist(),
            "selected_bit": "1 + floor(uint32_word * (prefix_width - 1) / 2**32)",
            "orientation": "equal probability for either input",
            "even_selected_input": "identity",
            "invariant": "current product modulo 2**prefix_width",
            "target_information_used": False,
            "mixture_probability": 0.5,
            "selection_density": (
                "State-independent label masses cancel under the same-label involution"
            ),
        }

    def decode(self, roots):
        bits = ((roots.long() + 1) // 2).reshape(-1, 2, self.width)
        return (bits * self.weights).sum(2)

    def labels(self, words):
        side = words[:, 1] & 1
        prefix = self.prefixes[words[:, 2] & 3]
        bit = 1 + ((words[:, 3] * (prefix - 1)) >> 32)
        return side, prefix, bit

    def transform(self, roots, side, prefix, bit):
        pair = self.decode(roots)
        selected = pair.gather(1, side[:, None]).squeeze(1)
        other = pair.gather(1, (1 - side)[:, None]).squeeze(1)
        modulus_mask = (1 << prefix) - 1
        changed = selected ^ (1 << bit)
        # Mask every product: all intermediate magnitudes stay below 2**40.
        denominator = changed | 1  # The result for an even input is discarded.
        inverse = torch.ones_like(denominator)
        for _ in range(5):  # 1 -> 2 -> 4 -> 8 -> 16 -> 32 correct inverse bits.
            inverse = (inverse * (2 - ((denominator * inverse) & modulus_mask))) & modulus_mask
        ratio = (selected * inverse) & modulus_mask
        compensated = (other & ~modulus_mask) | ((other * ratio) & modulus_mask)
        left = torch.where(side == 0, changed, compensated)
        right = torch.where(side == 0, compensated, changed)
        result = torch.stack((left, right), 1)
        result = torch.where((selected & 1)[:, None].bool(), result, pair)
        return (((result[:, :, None] >> self.shifts) & 1) * 2 - 1).reshape_as(roots).to(roots.dtype)

    def propose(self, roots, words):
        return self.transform(roots, *self.labels(words))


class UnitGroupRootProposal(ModularRootProposal):
    """Uniform odd multipliers, paired with inverse labels for the reverse move.

    Reuse the bounded integer representation, not the parent's involution.
    The transform accepts no target or factor certificate. Every label has
    exactly 2**(33-width) uint32 preimages, including for even input orbits.
    """

    metric_names = ARITHMETIC_METRICS
    uniform_orbit = True

    def __init__(self, width: int, device):
        super().__init__(width, device)
        self.mask = (1 << width) - 1
        self.metadata = {
            "transformation": "multiply_left_by_odd_unit_and_right_by_its_inverse",
            "input_width": width,
            "multiplier": "2 * (word[1] & (2**(width-1)-1)) + 1",
            "reverse_label": "multiplicative inverse modulo 2**width",
            "unused_words": [2, 3],
            "invariant": "current product modulo 2**width; each input valuation",
            "target_information_used": False,
            "mixture_probability": 0.5,
            "selection_density": "Uniform inverse-label bijection; symmetric aggregate root law",
        }

    def labels(self, words):
        return 2 * (words[:, 1] & ((1 << (self.width - 1)) - 1)) + 1

    def inverse(self, multiplier):
        inverse = torch.ones_like(multiplier)
        for _ in range(5):
            inverse = (inverse * (2 - ((multiplier * inverse) & self.mask))) & self.mask
        return inverse

    def transform(self, roots, multiplier):
        pair = self.decode(roots)
        values = torch.stack(
            (
                pair[:, 0] * multiplier & self.mask,
                pair[:, 1] * self.inverse(multiplier) & self.mask,
            ),
            1,
        )
        return (((values[:, :, None] >> self.shifts) & 1) * 2 - 1).reshape_as(roots).to(roots.dtype)

    def propose(self, roots, words):
        return self.transform(roots, self.labels(words))

    def observe(self, old_roots, new_roots, branch, target, energy, accepted):
        """Read-only public-product diagnostics, never used to choose a proposal."""
        old, new = self.decode(old_roots), self.decode(new_roots)
        eligible = ((old[:, 0] * old[:, 1] ^ target) & self.mask) == 0
        root_match = new[:, 0] * new[:, 1] == target
        changed = (new_roots != old_roots).sum(1)
        zero = energy == 0
        return torch.stack(
            (
                torch.ones_like(changed),
                branch,
                changed,
                eligible,
                eligible & branch,
                root_match,
                root_match & branch,
                zero,
                accepted & zero,
                accepted & (changed > 0),
                accepted & branch & (changed > 0),
            ),
            1,
        ).long()
