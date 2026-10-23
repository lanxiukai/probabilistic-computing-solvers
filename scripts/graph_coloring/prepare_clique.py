"""Find and independently verify a structural clique for color-symmetry reduction."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

from combinatorial_solvers.core.evidence import write_new_json
from combinatorial_solvers.graph_coloring.model import read_dimacs


def prepare(path: Path, palette: int, output: Path):
    started = perf_counter()
    instance = read_dimacs(path, palette)
    adjacency = instance.adjacency()
    masks = [sum(1 << neighbor for neighbor in row) for row in adjacency]
    starts = sorted(range(instance.size), key=lambda v: (-len(adjacency[v]), v))
    best = []
    for start in starts:
        clique = [start]
        remaining = masks[start]
        while remaining:
            candidates = []
            bits = remaining
            while bits:
                bit = bits & -bits
                candidates.append(bit.bit_length() - 1)
                bits ^= bit
            selected = max(
                candidates,
                key=lambda v, remaining=remaining: (
                    (masks[v] & remaining).bit_count(),
                    len(adjacency[v]),
                    -v,
                ),
            )
            clique.append(selected)
            remaining &= masks[selected]
        if len(clique) > len(best):
            best = clique
        if len(best) >= palette:
            break
    if len(best) > palette:
        raise ValueError("A clique larger than the requested palette proves infeasibility")
    certificate = instance.clique_certificate(best)
    if not certificate["valid_clique"]:
        raise AssertionError("Clique generation failed independent pairwise edge checks")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(
        output,
        {
            **certificate,
            "input_name": path.name,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "method": "Deterministic multi-start greedy common-neighbor clique",
            "scope": (
                "Structural lower-bound certificate and global color-symmetry reduction; "
                "no coloring solution is read or generated."
            ),
            "elapsed_seconds": perf_counter() - started,
            "timing_scope": (
                "Input read, clique construction and independent edge certificate; "
                "excluded from per-run search time and recorded as one-time preprocessing."
            ),
        },
    )
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--palette", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.instance, args.palette, args.output)


if __name__ == "__main__":
    main()
