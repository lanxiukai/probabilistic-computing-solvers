"""Shared energy models, samplers, caching, and experiment utilities."""

from .coloring import dsatur_coloring, validate_coloring
from .junction_tree import (
    JunctionTreeElimination,
    JunctionTreePlan,
    JunctionTreeSample,
    minimum_degree_elimination_order,
    prepare_junction_tree,
    sample_junction_tree_conditional,
)
from .macro import InvolutiveCodewordPlan, involutive_codeword_metropolis_step
from .metrics import (
    bootstrap_restart_cutoff,
    clopper_pearson_interval,
    discrete_survival_curve,
    empirical_success_curve,
    fit_censored_weibull_its,
    grouped_success_curve,
    time_to_solution,
)
from .model import ClampResult, IsingModel, QUBOModel
from .padded_sparse import PaddedSparseIsing
from .results import write_anneal_experiment
from .rng import CounterRngAuditRecorder, RngStreamAudit, audit_stream_partition, derive_stream_seed
from .sampler import AnnealResult, AnnealSnapshot, AnnealTracePoint, run_colored_gibbs
from .sparsify import CopySparseModel, copy_chain_sparsify

__all__ = [
    "AnnealResult",
    "AnnealSnapshot",
    "AnnealTracePoint",
    "ClampResult",
    "CopySparseModel",
    "CounterRngAuditRecorder",
    "InvolutiveCodewordPlan",
    "IsingModel",
    "JunctionTreeElimination",
    "JunctionTreePlan",
    "JunctionTreeSample",
    "PaddedSparseIsing",
    "QUBOModel",
    "RngStreamAudit",
    "audit_stream_partition",
    "bootstrap_restart_cutoff",
    "clopper_pearson_interval",
    "copy_chain_sparsify",
    "dsatur_coloring",
    "derive_stream_seed",
    "discrete_survival_curve",
    "empirical_success_curve",
    "fit_censored_weibull_its",
    "grouped_success_curve",
    "involutive_codeword_metropolis_step",
    "minimum_degree_elimination_order",
    "prepare_junction_tree",
    "run_colored_gibbs",
    "sample_junction_tree_conditional",
    "time_to_solution",
    "validate_coloring",
    "write_anneal_experiment",
]
