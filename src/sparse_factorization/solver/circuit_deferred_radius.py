"""Positive coordinate screening before a selected candidate's exact radius.

The first pre-screen pass is the only candidate eligible for the post-screen.
Failure at either stage requests identity on the entire physical state. The
state-independent first-pass normalizer cancels from the Hastings ratio.
"""

import math

import torch

from sparse_factorization.solver.circuit_fermat_crt_wide import FilteredWideCRTFermatRootProposal

EXTRA_MODULI = (
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
)


class DeferredRadiusProposal(FilteredWideCRTFermatRootProposal):
    mode_name = "fermat_deferred262144_v1"

    def __init__(self, width, target, device, *, mode="fermat_deferred262144_v1"):
        self.use_triton_pre_matches = False
        super().__init__(width, target, device, mode=mode)
        self.retries = 262144
        self.noise_words = 8 * self.retries
        self.filter_bits = 24
        self.filter_mask = (1 << self.filter_bits) - 1
        self.filter_log_weights = torch.tensor(
            [-24 * math.log(2), 0.0], dtype=torch.float64, device=device
        )
        self.pre_bits = 27
        self.pre_mask = (1 << self.pre_bits) - 1
        self.pre_log_weights = torch.tensor(
            [-27 * math.log(2), 0.0], dtype=torch.float64, device=device
        )
        self.pre_backend = "triton" if torch.device(device).type == "cuda" else "torch"
        self.extra_moduli = EXTRA_MODULI
        self.pre_tables = []
        for base in EXTRA_MODULI:
            squares = {i * i % base for i in range(base)}
            self.pre_tables.append(
                torch.tensor(
                    [(i * i - target) % base in squares for i in range(base)],
                    dtype=torch.bool,
                    device=device,
                )
            )

        self.crt_radices = tuple(len(t) for t in self.crt_tables)
        self.crt_offsets = tuple(sum(self.crt_radices[:i]) for i in range(len(self.crt_radices)))
        self.qr_offsets = tuple(sum(EXTRA_MODULI[:i]) for i in range(len(EXTRA_MODULI)))
        self.crt_flat = torch.cat(self.crt_tables)
        self.qr_flat = torch.cat(self.pre_tables)
        self.metadata.update(
            root_candidate_count=self.retries,
            noise_words=self.noise_words,
            filter_bits=24,
            pre_filter_bits=27,
            extra_moduli=list(EXTRA_MODULI),
            root_correction="log(Q_star*w_pre*w_post)(old)-log(Q_star*w_pre*w_post)(new)",
            normalization=(
                "State-independent first-pre-pass S_K(Z_pre) cancels; "
                "post-filter failures are whole-state identities"
            ),
            filter_eligible_candidates_definition=(
                "Number passing the coordinate pre-screen, before the selected-root post-screen"
            ),
            filter_bank_failures_definition=(
                "No pre-screen candidate or selected-root post-screen rejection"
            ),
            transformation="deferred_radius_two_stage_bank_or_full_state_identity",
            random_namespace="p749_deferred_radius_circuit_cuda_philox_v10",
            pre_backend=self.pre_backend,
            filter_weight="1 for low24 product match; 2**-24 otherwise",
            root_mass="S_K(Z_pre)*Q_star(root)*w_pre(root)*w_post(root) off diagonal",
        )
        self.metadata["candidate_word_layout"]["0"] = (
            "bit0 orientation; bits1..4 mixture label; bits5..31 independent pre-screen coin"
        )
        self.metadata["candidate_word_layout"]["7"] = "low24 independent post-screen coin"

    def preselect(self, words):
        if self.pre_backend == "triton":
            from .circuit_deferred_triton import preselect

            return preselect(self, words)
        return self.preselect_torch(words)

    @staticmethod
    def half_sum(pairs):
        a, b = pairs.unbind(1)
        mask = 0x7FFFFFFFFFFFFFFF
        return ((a >> 1) & mask) + ((b >> 1) & mask) + (a & b & 1)

    def pre_matches_x(self, x):
        if self.use_triton_pre_matches and x.is_cuda:
            from .wide_execution_triton import pre_matches

            return pre_matches(self, x)
        low, high = x & 0xFFFFFFFF, (x >> 32) & 0xFFFFFFFF
        result = torch.ones_like(x, dtype=torch.bool)
        for base, table in zip(EXTRA_MODULI, self.pre_tables, strict=True):
            remainder = (low % base + (high % base) * ((1 << 32) % base)) % base
            result = result & table[remainder]
        return result

    def coordinate_bank(self, words):
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
        uniform = (
            torch.stack((flat[:, 3] | (flat[:, 4] << 32), flat[:, 5] | (flat[:, 6] << 32)), 1)
            & self.input_mask
        )
        coordinate = torch.where(uniform_label, self.half_sum(uniform), x)
        return coordinate, uniform_label | biased

    def preselect_torch(self, words):
        coordinate, valid = self.coordinate_bank(words)
        flat = words.reshape(-1, 8)
        coin = ((flat[:, 0] >> 5) & self.pre_mask) == 0
        eligible = (valid & (self.pre_matches_x(coordinate) | coin)).reshape(
            len(words), self.retries
        )
        return eligible.int().argmax(1), eligible.any(1), eligible.sum(1)

    def propose_filtered(self, roots, words):
        selected, pre_valid, counts = self.preselect(words)
        chosen = words.reshape(len(roots), self.retries, 8)[
            torch.arange(len(roots), device=words.device), selected
        ]
        pairs, biased, matches, post_valid = super().candidate_pairs(chosen)
        proposed = (
            (((pairs[:, :, None] >> self.shifts) & 1) * 2 - 1).reshape_as(roots).to(roots.dtype)
        )
        return proposed, {
            "valid": pre_valid & post_valid,
            "biased": biased,
            "eligible_count": counts,
            "filter_match": matches,
        }

    def log_weighted_mass(self, roots):
        pairs = self.decode(roots)
        return (
            self.log_probability(roots)
            + self.pre_log_weights[self.pre_matches_x(self.half_sum(pairs)).long()]
            + self.filter_log_weights[self.filter_matches(pairs).long()]
        )
