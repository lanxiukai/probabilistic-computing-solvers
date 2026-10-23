"""Static root ancestry for state-independent partial circuit regeneration."""

from collections.abc import Sequence

import numpy as np


def root_ancestor_words(
    groups: Sequence[Sequence[int]],
    contexts: Sequence[Sequence[int]],
    levels: Sequence[Sequence[int]],
    roots: Sequence[int],
    fixed: set[int],
) -> np.ndarray:
    """Return one signed-storage 64-bit root set for each proposal group.

    A root port keeps its own dependency even when another root influences
    other ports in its owner group. Pins transmit no root dependence.
    """
    if not 0 < len(roots) <= 64 or len(set(roots)) != len(roots):
        raise ValueError("Regrowth requires between one and 64 distinct roots")
    ancestors = {node: 1 << bit for bit, node in enumerate(roots)}
    ancestors.update({node: 0 for node in fixed})
    root_set = set(roots)
    words = [0] * len(groups)
    for level in levels:
        for group in level:
            word = 0
            for node in contexts[group]:
                word |= ancestors[node]
            words[group] = word
            for node in groups[group]:
                if node not in root_set and node not in fixed:
                    ancestors[node] = word
    return np.asarray(words, dtype=np.uint64).view(np.int64)
