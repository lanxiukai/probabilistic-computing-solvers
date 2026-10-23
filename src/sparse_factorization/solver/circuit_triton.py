"""CUDA layer sampling and FP64 density evaluation for circuit-directed MH."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from .circuit_proposal import CircuitDirectedProposal


@triton.jit
def _cdf_code(Cdf, offset, uniform, valid, Search: tl.constexpr):
    if Search == "scan32_v1":
        candidates = tl.arange(0, 32)
        cdf = tl.load(
            Cdf + offset[:, :, None] + candidates[None, None, :],
            valid[:, :, None],
            other=4294967296,
        )
        return tl.sum((uniform[:, :, None] >= cdf).to(tl.int32), axis=2)
    else:
        # The final CDF entry equals 2**32 and every input word is smaller.
        # Build the count of thresholds <= uniform, five exact integer decisions.
        code = tl.full(offset.shape, 0, tl.int32)
        for bit in tl.static_range(5):
            step = 1 << (4 - bit)
            threshold = tl.load(Cdf + offset + code + step - 1, valid, other=4294967296)
            code += tl.where(uniform >= threshold, step, 0)
        return code


@triton.jit
def _sample_layer(
    Full,
    Beta,
    Cdf,
    Context,
    ContextWeight,
    Nodes,
    Mutable,
    Uniforms,
    Groups,
    Active,
    Replicas,
    StageSize,
    GroupsTotal: tl.constexpr,
    FullSize: tl.constexpr,
    ReplicaBlock: tl.constexpr,
    GroupBlock: tl.constexpr,
    BetasTotal: tl.constexpr,
    CdfLayout: tl.constexpr,
    Search: tl.constexpr,
    Partial: tl.constexpr,
):
    replicas = tl.program_id(0) * ReplicaBlock + tl.arange(0, ReplicaBlock)
    columns = tl.program_id(1) * GroupBlock + tl.arange(0, GroupBlock)
    groups = tl.load(Groups + columns, columns < StageSize, other=0).to(tl.int32)
    valid = (replicas[:, None] < Replicas) & (columns[None, :] < StageSize)
    if Partial:
        valid &= tl.load(
            Active + replicas[:, None] * GroupsTotal + groups[None, :], valid, other=False
        )
    context = tl.full((ReplicaBlock, GroupBlock), 0, tl.int32)
    for port in tl.static_range(3):
        node = tl.load(Context + groups * 3 + port, columns < StageSize, other=0).to(tl.int32)
        weight = tl.load(ContextWeight + groups * 3 + port, columns < StageSize, other=0).to(
            tl.int32
        )
        spin = tl.load(
            Full + replicas[:, None] * FullSize + node[None, :],
            valid & (weight[None, :] != 0),
            other=-1,
        ).to(tl.int32)
        context += ((spin + 1) >> 1) * weight[None, :]
    beta = tl.load(Beta + replicas, replicas < Replicas, other=0).to(tl.int32)
    if CdfLayout == "beta_group_context":
        offset = ((beta[:, None] * GroupsTotal + groups[None, :]) * 8 + context) * 32
    elif CdfLayout == "group_beta_context":
        offset = ((groups[None, :] * BetasTotal + beta[:, None]) * 8 + context) * 32
    else:
        offset = ((groups[None, :] * 8 + context) * BetasTotal + beta[:, None]) * 32
    uniform = tl.load(Uniforms + replicas[:, None] * GroupsTotal + groups[None, :], valid, other=0)
    code = _cdf_code(Cdf, offset, uniform, valid, Search)
    for port in tl.static_range(5):
        node = tl.load(Nodes + groups * 5 + port, columns < StageSize, other=0).to(tl.int32)
        mutable = tl.load(Mutable + groups * 5 + port, columns < StageSize, other=0)
        spin = (((code >> port) & 1) * 2 - 1).to(tl.int8)
        # Roots may be read by peers in this stage. Never rewrite roots or pins,
        # even with an identical value: only this group's mutable ports are owned.
        tl.store(
            Full + replicas[:, None] * FullSize + node[None, :], spin, valid & mutable[None, :]
        )


@triton.jit
def _energy_and_density(
    Full,
    Beta,
    Context,
    ContextWeight,
    Nodes,
    PortWeight,
    LogQ,
    Active,
    GateEnergy,
    Edges,
    Energy,
    Density,
    GroupsTotal: tl.constexpr,
    FullSize: tl.constexpr,
    EdgesTotal: tl.constexpr,
    GroupBlock: tl.constexpr,
    EdgeBlock: tl.constexpr,
    Partial: tl.constexpr,
):
    replica = tl.program_id(0)
    groups = tl.arange(0, GroupBlock)
    valid = groups < GroupsTotal
    context = tl.full((GroupBlock,), 0, tl.int32)
    code = tl.full((GroupBlock,), 0, tl.int32)
    for port in tl.static_range(3):
        node = tl.load(Context + groups * 3 + port, valid, other=0).to(tl.int32)
        weight = tl.load(ContextWeight + groups * 3 + port, valid, other=0).to(tl.int32)
        spin = tl.load(Full + replica * FullSize + node, valid, other=-1).to(tl.int32)
        context += ((spin + 1) >> 1) * weight
    for port in tl.static_range(5):
        node = tl.load(Nodes + groups * 5 + port, valid, other=0).to(tl.int32)
        weight = tl.load(PortWeight + groups * 5 + port, valid, other=0).to(tl.int32)
        spin = tl.load(Full + replica * FullSize + node, valid, other=-1).to(tl.int32)
        code += ((spin + 1) >> 1) * weight
    beta = tl.load(Beta + replica).to(tl.int32)
    logq = tl.load(
        LogQ + ((beta * GroupsTotal + groups) * 8 + context) * 32 + code, valid, other=0.0
    )
    if Partial:
        active = tl.load(Active + replica * GroupsTotal + groups, valid, other=False)
        logq = tl.where(active, logq, 0.0)
    gate = tl.load(GateEnergy + groups * 32 + code, valid, other=0.0)
    edges = tl.arange(0, EdgeBlock)
    left = tl.load(Edges + 2 * edges, edges < EdgesTotal, other=0).to(tl.int32)
    right = tl.load(Edges + 2 * edges + 1, edges < EdgesTotal, other=0).to(tl.int32)
    a = tl.load(Full + replica * FullSize + left, edges < EdgesTotal, other=-1)
    b = tl.load(Full + replica * FullSize + right, edges < EdgesTotal, other=-1)
    copies = tl.sum(((a != b) & (edges < EdgesTotal)).to(tl.int32), axis=0)
    tl.store(Energy + replica, tl.sum(gate, axis=0) + 2 * copies)
    tl.store(Density + replica, tl.sum(logq, axis=0))


def sample_body(
    proposal: CircuitDirectedProposal,
    full: torch.Tensor,
    beta_indices: torch.Tensor,
    uniforms: torch.Tensor,
    active_groups: torch.Tensor | None = None,
) -> None:
    """Sample exactly the cached integer CDF, one dependency layer per launch."""
    if (
        uniforms.dtype != torch.int64
        or uniforms.device != proposal.device
        or uniforms.shape != (len(full), proposal.group_count)
    ):
        raise ValueError("The CUDA categorical input must contain int64 uint32 draws on device")
    uniforms, beta_indices = uniforms.contiguous(), beta_indices.contiguous()
    replica_block, warps = (16, 8) if proposal.cdf_search == "scan32_v1" else (64, 4)
    for ids, _, _ in proposal.stages:
        _sample_layer[(triton.cdiv(len(full), replica_block), triton.cdiv(len(ids), 4))](
            full,
            beta_indices,
            proposal.sampling_cumulative,
            proposal.context_nodes,
            proposal.context_weights,
            proposal.group_nodes,
            proposal.mutable_ports,
            uniforms,
            ids,
            active_groups,
            len(full),
            len(ids),
            GroupsTotal=proposal.group_count,
            FullSize=full.shape[1],
            ReplicaBlock=replica_block,
            GroupBlock=4,
            BetasTotal=len(proposal.beta_values),
            CdfLayout=proposal.cdf_layout,
            Search=proposal.cdf_search,
            Partial=active_groups is not None,
            num_warps=warps,
        )


def energy_and_density(
    proposal: CircuitDirectedProposal,
    full: torch.Tensor,
    beta_indices: torch.Tensor,
    active_groups: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce gate/COPY energy and actual proposal log masses in FP64."""
    energy = torch.empty(len(full), dtype=torch.float64, device=full.device)
    density = torch.empty_like(energy)
    _energy_and_density[(len(full),)](
        full,
        beta_indices.contiguous(),
        proposal.context_nodes,
        proposal.context_weights,
        proposal.group_nodes,
        proposal.port_weights,
        proposal.log_probabilities,
        active_groups,
        proposal.gate_energies,
        proposal.copy_edges,
        energy,
        density,
        GroupsTotal=proposal.group_count,
        FullSize=full.shape[1],
        EdgesTotal=len(proposal.copy_edges),
        GroupBlock=triton.next_power_of_2(proposal.group_count),
        EdgeBlock=triton.next_power_of_2(max(1, len(proposal.copy_edges))),
        Partial=active_groups is not None,
        num_warps=8,
    )
    return energy, density
