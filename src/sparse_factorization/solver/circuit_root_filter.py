"""Bounded independent root rejection banks with an exact Hastings correction."""

from __future__ import annotations

import math

import torch

from .circuit_quotient import (
    QUOTIENT_METRICS,
    QuotientRootProposal,
    unsigned_product_limbs,
)

FILTER_SETTINGS = {
    "quotient_filter_control_v1": (1, 0),
    "quotient_filter64_v1": (64, 12),
    "quotient_filter256_v1": (256, 12),
}
FERMAT_MODES = {
    "fermat_filter256_v1",
    "fermat_crt_filter256_v1",
    "fermat_crt128_filter256_v1",
    "fermat_deferred262144_v1",
}
FILTER_MODES = set(FILTER_SETTINGS) | FERMAT_MODES
FILTER_METRICS = (
    *QUOTIENT_METRICS,
    "root_candidate_draws",
    "filter_eligible_candidates",
    "filter_bank_failures",
    "selected_filter_match",
    "old_filter_match",
)


class FilteredQuotientRootProposal(QuotientRootProposal):
    """Select the first weighted root candidate, or request a full-state identity.

    A bank contains K independent draws from the original full-support Q0.
    A public low-bit product match has acceptance weight one; every other
    pair has weight 2**(-k). For distinct physical states, proposal mass is
    S_K(Z) Q0(root) w(root) h_beta(body|root), where Z=sum Q0*w and
    S_K(Z)=sum_{j=0}^{K-1}(1-Z)**j. S_K is independent of the current state
    and cancels in the MH ratio. An all-failed bank is a diagonal mass:
    the caller MUST reject the entire physical transition in that case.

    Five independent uint32 words encode each candidate: the existing base
    mixture label, envelope index, uniform left/right pair, and filter coin.
    The unweighted control retains all nine original words and the exact
    original proposal path. No factors, saved-state defects or adaptive
    retry budgets enter this distribution.
    """

    metric_names = FILTER_METRICS

    def __init__(self, width: int, target: int, device, *, mode: str):
        if mode not in FILTER_SETTINGS:
            raise ValueError("Choose a separately named bounded root-filter mode")
        super().__init__(width, target, device, pure=True)
        self.retries, self.filter_bits = FILTER_SETTINGS[mode]
        self.filter_mask = (1 << self.filter_bits) - 1
        self.noise_words = 9 if self.retries == 1 else 5 * self.retries
        self.filter_log_weights = torch.tensor(
            [-self.filter_bits * math.log(2), 0.0], dtype=torch.float64, device=device
        )
        self.metadata = {
            "base_candidate_distribution": self.metadata,
            "input_width": width,
            "transformation": "bounded_weighted_quotient_bank_or_full_state_identity",
            "filter_mode": mode,
            "root_candidate_count": self.retries,
            "filter_bits": self.filter_bits,
            "filter_weight": "1 for low-k-bit product match; 2**(-k) otherwise",
            "all_failed_action": "Retain the entire old physical state",
            "root_correction": "log(Q0(old)*w(old)) - log(Q0(new)*w(new))",
            "normalization": "State-independent S_K(Z) cancels; not evaluated",
            "root_mass": "S_K(Z)*Q0(root)*w(root) off diagonal; full-state identity on failure",
            "noise_words": self.noise_words,
            "unused_words": [4, 5, 6, 7, 8] if self.retries == 1 else [],
            "candidate_words_generated": "All K candidates, including after the first pass",
            "full_product_divisibility_filter_used": False,
            "public_low_bit_product_filter_used": self.filter_bits > 0,
            "factor_certificate_used": False,
            "target_information_used": "Public target and structural widths only",
        }

    def filter_matches(self, pair):
        """Use only low-k-bit operands; products stay below 2**24."""
        low_product = (pair[:, 0] & self.filter_mask) * (pair[:, 1] & self.filter_mask)
        return (low_product & self.filter_mask) == (self.target & self.filter_mask)

    def log_weighted_mass(self, roots):
        """Unnormalized root mass Q0*w, for the Hastings ratio only."""
        matches = self.filter_matches(self.decode(roots))
        return self.log_probability(roots) + self.filter_log_weights[matches.long()]

    def propose_filtered(self, roots, words):
        if self.retries == 1:
            proposed, biased = super().propose_with_details(roots, words)
            valid = torch.ones(len(roots), dtype=torch.bool, device=roots.device)
            return proposed, {
                "valid": valid,
                "biased": biased,
                "eligible_count": torch.ones_like(valid, dtype=torch.int64),
                "filter_match": valid,
            }
        replicas = len(roots)
        candidate_words = words.reshape(replicas * self.retries, 5)
        index = candidate_words[:, 1] & (self.envelope - 1)
        biased = (((candidate_words[:, 0] >> 1) & 15) != 0) & (index < self.odd_count)
        left = self.first_odd + 2 * index
        left = torch.where(index < self.odd_count, left, self.first_odd)
        centers = torch.stack((left, self.center(left)), 1)
        uniform = candidate_words[:, 2:4] & self.mask
        pairs = torch.where(biased[:, None], centers, uniform)
        matches = self.filter_matches(pairs)
        eligible = matches | ((candidate_words[:, 4] & self.filter_mask) == 0)
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

    def observe_filtered(self, old_roots, new_roots, energy, accepted, details):
        old, new = self.decode(old_roots), self.decode(new_roots)
        high, low = unsigned_product_limbs(new[:, 0], new[:, 1])
        valid, biased = details["valid"], details["biased"]
        match = valid & (high == self.target_high) & (low == self.target_low)
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
