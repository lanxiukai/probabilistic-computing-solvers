"""Quantized state-dependent root selection from gate-context partition hints."""

from __future__ import annotations

import math

import numpy as np
import torch

MASS = 1 << 32
SCORE_SCALE = 256
WEIGHT_SCALE = 1 << 20
LOGIT_CLIP = 8


class ContextRootProposal:
    """Sample an ordered root prefix; its actual integer-bin density enters MH.

    Scores use local changes in the sum of informed gate log normalizers,
    with all other physical context bits held at their current values.
    This is a surrogate score, not an exact marginal of the target model.
    """

    def __init__(self, contexts, roots, context_nodes, context_weights, logits, device):
        self.root_count = len(roots)
        if not 4 <= self.root_count <= 64:
            raise ValueError("Guided root selection requires between four and 64 roots")
        neighbours = [
            [
                (g, 1 << bit)
                for g, context in enumerate(contexts)
                for bit, node in enumerate(context)
                if node == root
            ]
            for root in roots
        ]
        degree = max(map(len, neighbours))
        ids = np.zeros((self.root_count, degree), dtype=np.int64)
        delta = np.zeros((len(logits), self.root_count, degree, 8), dtype=np.int64)
        normalizer = np.logaddexp.reduce(logits, axis=-1)
        for r, entries in enumerate(neighbours):
            for j, (group, bit) in enumerate(entries):
                ids[r, j] = group
                difference = normalizer[:, group, np.arange(8) ^ bit] - normalizer[:, group]
                scaled = np.rint(SCORE_SCALE * difference)
                if not np.isfinite(scaled).all() or np.abs(scaled).max() > (1 << 50):
                    raise ValueError(
                        "Root partition scores exceed the bounded integer representation"
                    )
                delta[:, r, j] = scaled.astype(np.int64)

        def tensor(x):
            return torch.as_tensor(x, device=device)

        self.cardinalities = tensor(np.array([0, 1, 2, 4], dtype=np.int64))
        self.context_nodes = tensor(context_nodes[ids])
        self.context_weights = tensor(context_weights[ids])
        self.delta = tensor(delta)
        self.root_ids = torch.arange(self.root_count, device=device)
        self.branches = torch.arange(degree, device=device)
        # exp(delta / 2), quantized once on the host. Subsequent CDF arithmetic
        # is entirely integer, including normalization and without-replacement masks.
        weights = np.rint(
            WEIGHT_SCALE * np.exp(-np.arange(2 * SCORE_SCALE * LOGIT_CLIP + 1) / (2 * SCORE_SCALE))
        ).astype(np.int64)
        self.weights = tensor(weights)
        self.metadata = {
            "score": "local_gate_context_log_partition_difference",
            "score_scale": SCORE_SCALE,
            "balancing_exponent": 0.5,
            "weight_scale": WEIGHT_SCALE,
            "relative_logit_clip": LOGIT_CLIP,
            "maximum_root_contexts": degree,
            "lookup_bytes": int(delta.nbytes + weights.nbytes + 2 * context_nodes[ids].nbytes),
            "density": "Actual uint32 CDF interval masses for the active ordered prefix",
            "reverse_state": "Full proposed physical state after body regeneration",
        }

    def scores(self, full, beta_indices):
        bits = (full[:, self.context_nodes].long() + 1) // 2
        context = (bits * self.context_weights).sum(-1)
        return self.delta[
            beta_indices[:, None, None],
            self.root_ids[None, :, None],
            self.branches[None, None, :],
            context,
        ].sum(-1)

    def cdf(self, scores, available):
        maximum = torch.where(available, scores, -(1 << 60)).amax(1, keepdim=True)
        distance = (maximum - scores).clamp(0, 2 * SCORE_SCALE * LOGIT_CLIP)
        weights = torch.where(available, self.weights[distance], 0)
        cumulative = weights.cumsum(1)
        # At most 64 * 2**20 total weight: multiplication by 2**32 fits int64.
        return torch.div(cumulative * MASS, cumulative[:, -1:], rounding_mode="floor")

    @staticmethod
    def log_mass(cdf, choice):
        upper = cdf.gather(1, choice[:, None]).squeeze(1)
        lower = cdf.gather(1, (choice - 1).clamp_min(0)[:, None]).squeeze(1)
        lower = torch.where(choice == 0, 0, lower)
        return (upper - lower).double().log() - math.log(MASS)

    def sample(self, full, beta_indices, uniforms, sizes):
        scores = self.scores(full, beta_indices)
        available = torch.ones_like(scores, dtype=torch.bool)
        mask = torch.zeros_like(available)
        density = torch.zeros(len(full), dtype=torch.float64, device=full.device)
        order = []
        for j in range(4):
            cdf = self.cdf(scores, available)
            choice = (uniforms[:, j, None] >= cdf).sum(1)
            density += torch.where(j < sizes, self.log_mass(cdf, choice), 0.0)
            available.scatter_(1, choice[:, None], False)
            mask.scatter_(1, choice[:, None], (j < sizes)[:, None])
            order.append(choice)
        return mask, torch.stack(order, dim=1), density

    def log_probability(self, full, beta_indices, order, sizes):
        scores = self.scores(full, beta_indices)
        available = torch.ones_like(scores, dtype=torch.bool)
        density = torch.zeros(len(full), dtype=torch.float64, device=full.device)
        for j in range(4):
            cdf = self.cdf(scores, available)
            choice = order[:, j]
            density += torch.where(j < sizes, self.log_mass(cdf, choice), 0.0)
            available.scatter_(1, choice[:, None], False)
        return density
