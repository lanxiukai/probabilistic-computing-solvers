"""Create an independently seeded planted list-coloring fixture on the cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from combinatorial_solvers.core.evidence import write_manifest, write_new_json
from combinatorial_solvers.graph_coloring.model import Instance


def generate(output: Path, vertices: int, colors: int, list_size: int, degree: float, seed: int):
    if vertices < colors or colors < 3 or not 2 <= list_size < colors:
        raise ValueError("Require vertices >= colors >= 3 and a nontrivial allowed list")
    if vertices % colors or not 2 <= degree < vertices * (1 - 1 / colors):
        raise ValueError("Require equal planted classes and a valid expected degree")
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.Generator(np.random.PCG64(seed))
    order = rng.permutation(vertices)
    planted = np.empty(vertices, dtype=np.int64)
    planted[order] = np.arange(vertices) % colors
    edges = {
        tuple(sorted((int(order[i]), int(order[(i + 1) % vertices])))) for i in range(vertices)
    }
    probability = degree / (vertices * (1 - 1 / colors))
    for u in range(vertices):
        candidates = np.flatnonzero(planted[u + 1 :] != planted[u]) + u + 1
        selected = candidates[rng.random(len(candidates)) < probability]
        edges.update((u, int(v)) for v in selected)
    allowed = np.zeros((vertices, colors), dtype=bool)
    for vertex in range(vertices):
        alternatives = np.delete(np.arange(colors), planted[vertex])
        allowed[vertex, rng.choice(alternatives, list_size - 1, replace=False)] = True
        allowed[vertex, planted[vertex]] = True
    instance = Instance(vertices, colors, tuple(edges), allowed)
    certificate = instance.certificate(planted)
    if not certificate["feasible"]:
        raise AssertionError("Generated witness failed the original edge/list constraints")
    graph = output / "instance.col"
    graph.write_text(
        "c Independently generated planted list-coloring fixture; not a DIMACS benchmark.\n"
        + f"p edge {vertices} {len(instance.edges)}\n"
        + "".join(f"e {u + 1} {v + 1}\n" for u, v in instance.edges)
    )
    np.save(output / "allowed.npy", allowed, allow_pickle=False)
    write_new_json(
        output / "generation-witness.json",
        {
            "colors": planted.tolist(),
            "certificate": certificate,
            "solver_input": False,
            "purpose": "Post-run zero-energy feasibility certificate only",
        },
    )
    write_new_json(
        output / "generation.json",
        {
            "vertices": vertices,
            "colors": colors,
            "list_size": list_size,
            "bernoulli_expected_degree_before_connecting_cycle": degree,
            "seed": seed,
            "rng": "numpy.PCG64",
            "numpy_version": np.__version__,
            "source": str(Path(__file__).name),
            "source_kind": "Owned independent synthetic generator",
            "benchmark_scope": (
                "Generated list-constraint demonstration; not an external benchmark "
                "or evidence of general hardness."
            ),
            "solver_inputs": ["instance.col", "allowed.npy"],
        },
    )
    (output / "generator-source.py").write_bytes(Path(__file__).read_bytes())
    write_manifest(output, problem="graph-coloring", run_id=output.name, status="completed")
    print(json.dumps({"output": str(output), "vertices": vertices, "edges": len(edges)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vertices", type=int, default=256)
    parser.add_argument("--colors", type=int, default=4)
    parser.add_argument("--list-size", type=int, default=3)
    parser.add_argument("--expected-degree", type=float, default=16)
    parser.add_argument("--seed", type=int, default=2026102391)
    args = parser.parse_args()
    generate(
        args.output,
        args.vertices,
        args.colors,
        args.list_size,
        args.expected_degree,
        args.seed,
    )


if __name__ == "__main__":
    main()
