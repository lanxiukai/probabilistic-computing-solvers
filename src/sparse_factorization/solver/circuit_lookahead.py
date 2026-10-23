"""Static backward unary likelihoods for full-support circuit proposals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def backward_unary_bias(
    log_weights: np.ndarray,
    groups: Sequence[Sequence[int]],
    contexts: Sequence[Sequence[int]],
    levels: Sequence[Sequence[int]],
    roots: set[int],
    *,
    strength: float,
) -> np.ndarray:
    """Approximate downstream partition factors with one reverse DAG pass.

    Other context bits are marginalized under a uniform measure. Reconvergent
    paths are treated as independent, so these are proposal hints, not exact
    posterior marginals. Root contexts are excluded: the symmetric root-mask law
    and its support are unchanged. The final clipped, quantized body law must
    still be used in both directions of the MH ratio.
    """
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Use a finite backward-message strength between zero and one")
    betas, group_count, _, codes = log_weights.shape
    result = np.zeros((betas, group_count, codes), dtype=np.float64)
    if strength == 0:
        return result
    size = max(node for group in groups for node in group) + 1
    messages = np.zeros((betas, size, 2), dtype=np.float64)
    candidate_codes = np.arange(codes)
    for level in reversed(levels):
        for group in level:
            bias = result[:, group]
            for port, node in enumerate(groups[group]):
                bias += strength * messages[:, node, (candidate_codes >> port) & 1]
            normalizer = np.logaddexp.reduce(log_weights[:, group] + bias[:, None, :], axis=-1)
            context_codes = np.arange(1 << len(contexts[group]))
            for bit, node in enumerate(contexts[group]):
                if node in roots:
                    continue
                likelihood = np.stack(
                    [
                        np.logaddexp.reduce(
                            normalizer[:, context_codes[((context_codes >> bit) & 1) == value]],
                            axis=-1,
                        )
                        for value in (0, 1)
                    ],
                    axis=-1,
                )
                # Equal context counts cancel, as does this additive gauge.
                likelihood -= likelihood.max(axis=-1, keepdims=True)
                messages[:, node] += likelihood
    if not np.isfinite(result).all():
        raise ValueError("Backward unary proposal hints must remain finite")
    return result
