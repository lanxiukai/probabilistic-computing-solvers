"""Create small deterministic software fixtures, never performance benchmarks."""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path


def create(output: Path):
    from combinatorial_solvers.ldpc.model import Code, channel_batch
    from combinatorial_solvers.partition.model import generate
    from combinatorial_solvers.tsp.instance import certify_tour
    from combinatorial_solvers.tsp.tsplib import load_tsplib

    output.mkdir(parents=True, exist_ok=False)
    common = {"method": "native", "experiment": {"output_dir": "unused"}}
    anneal = {
        "device": "cpu",
        "seed": 71,
        "chains": 2,
        "max_seconds": 30,
        "max_sweeps": 8,
        "cycle_sweeps": 8,
        "observe_every": 1,
    }
    fixtures = {
        "mwis": {
            **common,
            "problem": {"size": 12, "mean_degree": 3, "instance_seed": 71, "weight_max": 10},
            "solver": {**anneal, "method": "pair-gibbs"},
            "exact_reference": True,
        },
        "maxsat": {
            **common,
            "problem": {
                "variables": 10,
                "clause_ratio": 8,
                "arity": 3,
                "weight_max": 10,
                "instance_seed": 71,
            },
            "solver": anneal,
            "exact_reference": True,
        },
        "maxcut": {
            **common,
            "problem": {
                "size": 12,
                "mean_degree": 3,
                "instance_seed": 71,
                "weight_max": 10,
                "signed": True,
            },
            "solver": anneal,
            "exact_reference": True,
        },
        "maxcut-dense": {
            **common,
            "problem": {"size": 12, "family": "sk", "instance_seed": 71},
            "solver": {
                "device": "cpu",
                "chains": 2,
                "block_size": 4,
                "max_steps": 8,
                "cycle_steps": 8,
                "max_seconds": 30,
            },
            "exact_reference": True,
        },
        "labs": {
            **common,
            "problem": {"size": 12},
            "solver": {
                "device": "cpu",
                "chains": 2,
                "max_steps": 12,
                "cycle_steps": 12,
                "max_seconds": 30,
            },
            "exact_reference": True,
        },
    }
    graph = generate({"size": 12, "family": "random", "degree": 3, "maximum_weight": 3, "seed": 71})
    graph.write_metis(output / "partition.graph")
    fixtures["partition"] = {
        **common,
        "problem": {"graph": str(output / "partition.graph")},
        "solver": {
            "device": "cpu",
            "replicas": 2,
            "width": 4,
            "banks": 8,
            "max_sweeps": 8,
            "max_seconds": 30,
            "cycle_sweeps": 8,
        },
        "exact_reference": True,
    }
    instance, _ = channel_batch(Code.regular(12, 71), frames=2, esn0_db=-1, seed=83)
    instance.save(output / "ldpc.npz")
    fixtures["ldpc"] = {
        **common,
        "problem": {"instance": str(output / "ldpc.npz")},
        "solver": {
            "device": "cpu",
            "replicas": 2,
            "max_steps": 8,
            "max_seconds": 30,
            "cycle_sweeps": 8,
            "diagnostic_sweeps": 1,
        },
        "exact_reference": True,
    }
    (output / "fixture.tsp").write_text(
        "NAME: fixture\nTYPE: TSP\nDIMENSION: 6\nEDGE_WEIGHT_TYPE: EUC_2D\n"
        "NODE_COORD_SECTION\n1 0 0\n2 3 0\n3 6 1\n4 6 5\n5 2 6\n6 0 3\nEOF\n"
    )
    tsp = load_tsplib(output / "fixture.tsp")
    optimum = min(certify_tour(tsp, (0, *tail)).length for tail in permutations(range(1, 6)))
    fixtures["tsp"] = {
        "problem": {
            "instance": str(output / "fixture.tsp"),
            "target_relative_gap": 0,
            "optimum": optimum,
        },
        "prefix": {"kind": "ils", "iterations": 2, "starts": 2},
        "solver": {
            "kind": "hybrid",
            "device": "cpu",
            "selector": "pbit",
            "rounds": 2,
            "max_seconds": 30,
            "elite_size": 4,
            "local_search_passes": 10,
            "local_search_batch_size": 2,
            "local_search": "two_opt_or_opt",
            "seed": 71,
        },
        "experiment": {"output_dir": "unused"},
    }
    for name, configuration in fixtures.items():
        (output / f"{name}.json").write_text(json.dumps(configuration, indent=2) + "\n")
    return sorted(fixtures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("\n".join(create(args.output)))


if __name__ == "__main__":
    main()
