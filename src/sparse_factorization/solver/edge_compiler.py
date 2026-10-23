"""Assemble the existing gate-port Hamiltonian directly into canonical edges."""

from __future__ import annotations

from collections import defaultdict

import torch

from sparse_factorization.core.edge_model import from_edges

from .gates import COPY, get_gate


def assemble_edge_models(names, gate_nodes, netlist, copy_edges, metadata, dtype):
    """No dense full-model allocation; local gate matrices have at most five nodes."""
    size = len(names)
    gate_bias = [0.0] * size
    copy_bias = [0.0] * size
    gate_edges = defaultdict(float)
    copy_terms = defaultdict(float)
    gate_constant = copy_constant = 0.0
    templates = {
        kind: get_gate(kind).qubo(netlist.profile, dtype=torch.float64).to_ising()
        for kind in {g.gate for g in netlist.gates}
    }
    for instance, nodes in zip(netlist.gates, gate_nodes, strict=True):
        model = templates[instance.gate]
        for i, node in enumerate(nodes):
            gate_bias[node] += float(model.bias[i])
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = sorted((nodes[i], nodes[j]))
                gate_edges[a, b] += float(model.coupling[i, j])
        gate_constant += model.constant
    copy = COPY.qubo(netlist.profile, dtype=torch.float64).to_ising()
    for left, right in copy_edges:
        copy_bias[left] += float(copy.bias[0])
        copy_bias[right] += float(copy.bias[1])
        copy_terms[tuple(sorted((left, right)))] += float(copy.coupling[0, 1])
        copy_constant += copy.constant
    combined = defaultdict(float, gate_edges)
    for pair, value in copy_terms.items():
        combined[pair] += value
    degree = [0] * size
    for (a, b), value in combined.items():
        if value:
            degree[a] += 1
            degree[b] += 1
    info = {
        **metadata,
        "cpu_storage": "edge_v1",
        "edge_count": sum(w != 0 for w in combined.values()),
        "maximum_degree": max(degree, default=0),
        "copy_edges": len(copy_edges),
        "converted_from": "qubo",
    }
    models = []
    for edges, bias, constant, component in (
        (
            combined,
            [a + b for a, b in zip(gate_bias, copy_bias, strict=True)],
            gate_constant + copy_constant,
            None,
        ),
        (gate_edges, gate_bias, gate_constant, "gate"),
        (copy_terms, copy_bias, copy_constant, "copy"),
    ):
        models.append(
            from_edges(
                edges,
                bias,
                constant,
                names=tuple(names),
                dtype=dtype,
                metadata=info
                if component is None
                else {
                    **metadata,
                    "cpu_storage": "edge_v1",
                    "converted_from": "qubo",
                    "hamiltonian_component": component,
                },
            )
        )
    return tuple(models)
