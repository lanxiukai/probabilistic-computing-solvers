"""State-independent covering blocks, without constructing a column conflict graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Instance


@dataclass(frozen=True)
class Blocks:
    nodes: np.ndarray
    rows: np.ndarray
    masks: np.ndarray
    inverse: np.ndarray
    width: int
    banks: int
    per_bank: int


def pivot_pools(instance, limit=128):
    """Input-only affinity pools around deterministic greedy-cover columns."""
    from .baseline import fast_greedy

    anchors = np.flatnonzero(fast_greedy(instance))
    degrees = np.asarray(list(map(len, instance.columns)), dtype=float)
    costs = np.asarray(instance.costs, dtype=float)
    pools = {}
    for anchor in anchors:
        postings = [j for i in instance.columns[anchor] for j in instance.rows[i]]
        overlap = np.bincount(postings, minlength=instance.size)
        score = overlap / (np.maximum(degrees, 1) ** 0.5 * costs)
        score[anchor] = 0
        candidates = np.flatnonzero(score > 0)
        # Stable ties make the input-only shortlist independent of sort internals.
        chosen = candidates[np.lexsort((candidates, -score[candidates]))[:limit]]
        pools[int(anchor)] = (chosen, score[chosen])
    return anchors, pools


def pivot_order(n, width, anchors, pools, rng):
    available = np.ones(n, dtype=bool)
    result = []
    for anchor in rng.permutation(anchors):
        if not available[anchor]:
            continue
        available[anchor] = False
        result.append(int(anchor))
        candidates, scores = pools[int(anchor)]
        mask = available[candidates]
        candidates, scores = candidates[mask], scores[mask]
        take = min(width - 1, len(candidates))
        if take:
            probabilities = scores**2
            chosen = rng.choice(
                candidates, take, replace=False, p=probabilities / probabilities.sum()
            )
            available[chosen] = False
            result.extend(chosen.tolist())
        missing = min(width - 1 - take, int(available.sum()))
        if missing:
            extra = rng.choice(np.flatnonzero(available), missing, replace=False)
            available[extra] = False
            result.extend(extra.tolist())
    result.extend(rng.permutation(np.flatnonzero(available)).tolist())
    return np.asarray(result, dtype=np.int64)


def build_blocks(instance: Instance, width=4, banks=4, seed=107, *, policy="minhash"):
    if width not in (1, 2, 4, 8) or not 1 <= banks <= 64:
        raise ValueError("Unsupported block width or bank count")
    if policy not in {"minhash", "cost-minhash", "greedy-pivot"}:
        raise ValueError("Unsupported block construction policy")
    rng = np.random.Generator(np.random.PCG64(seed))
    n, m = instance.size, len(instance.rows)
    per_bank = (n + width - 1) // width
    nodes = np.full((banks * per_bank, width), -1, dtype=np.int32)
    inverse = np.empty((banks, n), dtype=np.int32)
    flat_columns = np.asarray([j for row in instance.rows for j in row], dtype=np.int32)
    flat_rows = np.repeat(np.arange(m, dtype=np.int32), list(map(len, instance.rows)))
    anchors, pools = pivot_pools(instance) if policy == "greedy-pivot" else (None, None)
    for bank in range(banks):
        if bank == 0:
            order = rng.permutation(n)
        elif policy == "greedy-pivot":
            order = pivot_order(n, width, anchors, pools, rng)
        else:
            # Columns with a shared minimum-rank row are adjacent; ranks are input-only.
            ranks = rng.permutation(m)
            owner = np.full(n, m, dtype=np.int64)
            np.minimum.at(owner, flat_columns, ranks[flat_rows])
            keys = (rng.permutation(n), owner)
            if policy == "cost-minhash":
                keys = (keys[0], np.asarray(instance.costs), owner)
            order = np.lexsort(keys)
        section = nodes[bank * per_bank : (bank + 1) * per_bank].reshape(-1)
        section[:n] = order
        inverse[bank, order] = bank * per_bank + np.arange(n) // width
    row_lists, mask_lists = [], []
    for block in nodes:
        union = {}
        for bit, column in enumerate(block):
            if column >= 0:
                for row in instance.columns[column]:
                    union[row] = union.get(row, 0) | (1 << bit)
        ordered = sorted(union)
        row_lists.append(ordered)
        mask_lists.append([union[i] for i in ordered])
    stride = 1 << (max(1, max(map(len, row_lists))) - 1).bit_length()
    rows = np.full((len(nodes), stride), m, dtype=np.int32)
    masks = np.zeros((len(nodes), stride), dtype=np.int32)
    for i, (ids, mask) in enumerate(zip(row_lists, mask_lists, strict=True)):
        rows[i, : len(ids)], masks[i, : len(ids)] = ids, mask
    return Blocks(nodes, rows, masks, inverse, width, banks, per_bank)


def subset_scores(outside, masks, width):
    """Independent CPU subset-sum transform used only in kernel verification."""
    histogram = np.bincount(masks[np.asarray(outside) == 0], minlength=1 << width)
    histogram[0] = 0
    for bit in range(width):
        for mask in range(1 << width):
            if mask & (1 << bit):
                histogram[mask] += histogram[mask ^ (1 << bit)]
    return histogram[((1 << width) - 1) ^ np.arange(1 << width)]
