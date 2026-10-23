"""Optional Triton local-field kernel for degree-padded Ising graphs."""

# pyright: reportInvalidTypeForm=false, reportOptionalMemberAccess=false

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only by CPU-only installations
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _padded_local_fields_kernel(
        states_pointer,
        neighbors_pointer,
        couplings_pointer,
        bias_pointer,
        indices_pointer,
        output_pointer,
        replicas,
        selected_variables,
        state_replica_stride,
        neighbor_variable_stride,
        coupling_variable_stride,
        maximum_degree: tl.constexpr,
        block_outputs: tl.constexpr,
        block_degree: tl.constexpr,
    ):
        flat = tl.program_id(axis=0) * block_outputs + tl.arange(0, block_outputs)
        output_count = replicas * selected_variables
        output_mask = flat < output_count
        replica = flat // selected_variables
        selected_position = flat - replica * selected_variables
        variable = tl.load(indices_pointer + selected_position, mask=output_mask, other=0)

        degree = tl.arange(0, block_degree)
        degree_mask = degree < maximum_degree
        adjacency_mask = output_mask[:, None] & degree_mask[None, :]
        adjacency_offset = variable[:, None] * neighbor_variable_stride + degree[None, :]
        neighbor = tl.load(
            neighbors_pointer + adjacency_offset,
            mask=adjacency_mask,
            other=0,
        )
        coupling_offset = variable[:, None] * coupling_variable_stride + degree[None, :]
        weight = tl.load(
            couplings_pointer + coupling_offset,
            mask=adjacency_mask,
            other=0.0,
        )
        state_offset = replica[:, None] * state_replica_stride + neighbor
        spin = tl.load(
            states_pointer + state_offset,
            mask=adjacency_mask,
            other=0,
        ).to(tl.float32)
        field = tl.sum(spin * weight, axis=1)
        field += tl.load(bias_pointer + variable, mask=output_mask, other=0.0)
        tl.store(output_pointer + flat, field, mask=output_mask)

    @triton.jit
    def _external_uniform_gibbs_write_kernel(
        states_pointer,
        probabilities_pointer,
        uniforms_pointer,
        indices_pointer,
        replicas,
        selected_variables,
        state_replica_stride,
        block_outputs: tl.constexpr,
    ):
        flat = tl.program_id(axis=0) * block_outputs + tl.arange(0, block_outputs)
        output_count = replicas * selected_variables
        output_mask = flat < output_count
        replica = flat // selected_variables
        selected_position = flat - replica * selected_variables
        variable = tl.load(indices_pointer + selected_position, mask=output_mask, other=0)
        probability = tl.load(probabilities_pointer + flat, mask=output_mask, other=0.0)
        uniform = tl.load(uniforms_pointer + flat, mask=output_mask, other=0.0)
        updated = tl.where(uniform < probability, 1, -1).to(tl.int8)
        state_offset = replica * state_replica_stride + variable
        tl.store(states_pointer + state_offset, updated, mask=output_mask)

    @triton.jit
    def _counter_gibbs_from_fields_kernel(
        states_pointer,
        fields_pointer,
        indices_pointer,
        replicas,
        selected_variables,
        state_replica_stride,
        beta,
        seed,
        counter_offset,
        block_outputs: tl.constexpr,
    ):
        flat = tl.program_id(axis=0) * block_outputs + tl.arange(0, block_outputs)
        output_count = replicas * selected_variables
        output_mask = flat < output_count
        replica = flat // selected_variables
        selected_position = flat - replica * selected_variables
        variable = tl.load(indices_pointer + selected_position, mask=output_mask, other=0)
        field = tl.load(fields_pointer + flat, mask=output_mask, other=0.0)
        probability = tl.sigmoid(2.0 * beta * field)
        logical_offset = counter_offset + flat.to(tl.int64)
        uniform = tl.rand(seed, logical_offset)
        updated = tl.where(uniform < probability, 1, -1).to(tl.int8)
        state_offset = replica * state_replica_stride + variable
        tl.store(states_pointer + state_offset, updated, mask=output_mask)

    @triton.jit
    def _padded_counter_gibbs_kernel(
        states_pointer,
        neighbors_pointer,
        couplings_pointer,
        bias_pointer,
        indices_pointer,
        fields_pointer,
        replicas,
        selected_variables,
        state_replica_stride,
        neighbor_variable_stride,
        coupling_variable_stride,
        beta,
        seed,
        counter_offset,
        maximum_degree: tl.constexpr,
        block_outputs: tl.constexpr,
        block_degree: tl.constexpr,
    ):
        flat = tl.program_id(axis=0) * block_outputs + tl.arange(0, block_outputs)
        output_count = replicas * selected_variables
        output_mask = flat < output_count
        replica = flat // selected_variables
        selected_position = flat - replica * selected_variables
        variable = tl.load(indices_pointer + selected_position, mask=output_mask, other=0)

        degree = tl.arange(0, block_degree)
        degree_mask = degree < maximum_degree
        adjacency_mask = output_mask[:, None] & degree_mask[None, :]
        adjacency_offset = variable[:, None] * neighbor_variable_stride + degree[None, :]
        neighbor = tl.load(
            neighbors_pointer + adjacency_offset,
            mask=adjacency_mask,
            other=0,
        )
        coupling_offset = variable[:, None] * coupling_variable_stride + degree[None, :]
        weight = tl.load(
            couplings_pointer + coupling_offset,
            mask=adjacency_mask,
            other=0.0,
        )
        state_offset = replica[:, None] * state_replica_stride + neighbor
        spin = tl.load(
            states_pointer + state_offset,
            mask=adjacency_mask,
            other=0,
        ).to(tl.float32)
        field = tl.sum(spin * weight, axis=1)
        field += tl.load(bias_pointer + variable, mask=output_mask, other=0.0)
        probability = tl.sigmoid(2.0 * beta * field)
        logical_offset = counter_offset + flat.to(tl.int64)
        uniform = tl.rand(seed, logical_offset)
        updated = tl.where(uniform < probability, 1, -1).to(tl.int8)
        output_state_offset = replica * state_replica_stride + variable
        tl.store(fields_pointer + flat, field, mask=output_mask)
        tl.store(states_pointer + output_state_offset, updated, mask=output_mask)


def is_triton_sparse_available() -> bool:
    """Return whether the optional kernel can run in this process."""

    return triton is not None and torch.cuda.is_available()


def triton_padded_local_fields(
    spins: torch.Tensor,
    neighbors: torch.Tensor,
    couplings: torch.Tensor,
    bias: torch.Tensor,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate selected local fields with one fused padded-adjacency kernel."""

    if triton is None:
        raise RuntimeError("The triton_sparse backend requires the triton package")
    if not spins.is_cuda:
        raise ValueError("The triton_sparse backend requires CUDA tensors")
    if spins.ndim != 2:
        raise ValueError("The triton_sparse backend requires [replicas, variables] spins")
    if spins.dtype != torch.int8:
        raise TypeError("The triton_sparse backend requires int8 bipolar spins")
    if neighbors.dtype != torch.long or neighbors.ndim != 2:
        raise TypeError("neighbors must be a two-dimensional torch.long tensor")
    if couplings.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("The triton_sparse backend currently requires float32 coefficients")
    if couplings.shape != neighbors.shape or bias.shape != (neighbors.shape[0],):
        raise ValueError("Sparse coefficient shapes are inconsistent")
    if any(tensor.device != spins.device for tensor in (neighbors, couplings, bias)):
        raise ValueError("All Triton sparse tensors must share one CUDA device")
    if spins.shape[1] != neighbors.shape[0]:
        raise ValueError("Spin shape does not match the sparse model size")
    if not spins.is_contiguous():
        spins = spins.contiguous()

    if indices is None:
        selected = torch.arange(neighbors.shape[0], device=spins.device, dtype=torch.long)
    else:
        selected = indices.to(device=spins.device, dtype=torch.long).contiguous()
    if selected.ndim != 1 or selected.numel() == 0:
        raise ValueError("indices must select at least one variable")
    replicas = int(spins.shape[0])
    selected_variables = int(selected.numel())
    maximum_degree = int(neighbors.shape[1])
    output = torch.empty(
        (replicas, selected_variables),
        dtype=torch.float32,
        device=spins.device,
    )
    block_outputs = 256
    block_degree = triton.next_power_of_2(max(maximum_degree, 1))
    output_count = replicas * selected_variables
    grid = (triton.cdiv(output_count, block_outputs),)
    _padded_local_fields_kernel[grid](
        spins,
        neighbors,
        couplings,
        bias,
        selected,
        output,
        replicas,
        selected_variables,
        spins.stride(0),
        neighbors.stride(0),
        couplings.stride(0),
        maximum_degree=maximum_degree,
        block_outputs=block_outputs,
        block_degree=block_degree,
    )
    return output


def _validate_update_inputs(
    spins: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    if triton is None:
        raise RuntimeError("The triton_sparse backend requires the triton package")
    if not spins.is_cuda or spins.ndim != 2 or spins.dtype != torch.int8:
        raise TypeError("Triton Gibbs updates require CUDA [replicas, variables] int8 spins")
    if not spins.is_contiguous():
        raise ValueError("Triton Gibbs updates require contiguous state storage")
    selected = indices.to(device=spins.device, dtype=torch.long).contiguous()
    if selected.ndim != 1 or selected.numel() == 0:
        raise ValueError("indices must select at least one variable")
    if torch.any(selected < 0) or torch.any(selected >= spins.shape[1]):
        raise IndexError("indices contain an out-of-range variable")
    return selected, int(spins.shape[0]), int(selected.numel())


def triton_external_uniform_gibbs_write(
    spins: torch.Tensor,
    probabilities: torch.Tensor,
    uniforms: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    """Apply legacy PyTorch probabilities/uniforms with one Triton state-write kernel."""

    selected, replicas, selected_variables = _validate_update_inputs(spins, indices)
    expected = (replicas, selected_variables)
    if probabilities.shape != expected or uniforms.shape != expected:
        raise ValueError("probabilities and uniforms must match [replicas, selected variables]")
    if probabilities.dtype != torch.float32 or uniforms.dtype != torch.float32:
        raise TypeError("The Triton bridge requires float32 probabilities and uniforms")
    if probabilities.device != spins.device or uniforms.device != spins.device:
        raise ValueError("Triton bridge tensors must share one CUDA device")
    probabilities = probabilities.contiguous()
    uniforms = uniforms.contiguous()
    block_outputs = 256
    output_count = replicas * selected_variables
    grid = (triton.cdiv(output_count, block_outputs),)
    _external_uniform_gibbs_write_kernel[grid](
        spins,
        probabilities,
        uniforms,
        selected,
        replicas,
        selected_variables,
        spins.stride(0),
        block_outputs=block_outputs,
    )


def triton_counter_gibbs_update_from_fields(
    spins: torch.Tensor,
    fields: torch.Tensor,
    indices: torch.Tensor,
    *,
    beta: float,
    seed: int,
    counter_offset: int,
) -> None:
    """Update spins from precomputed fields under the counter-v1 RNG protocol."""

    selected, replicas, selected_variables = _validate_update_inputs(spins, indices)
    if fields.shape != (replicas, selected_variables) or fields.dtype != torch.float32:
        raise TypeError("fields must be float32 with shape [replicas, selected variables]")
    if fields.device != spins.device:
        raise ValueError("fields and spins must share one CUDA device")
    if seed < 0 or seed >= 1 << 63 or counter_offset < 0 or counter_offset >= 1 << 63:
        raise ValueError("seed and counter_offset must fit nonnegative signed 64-bit integers")
    fields = fields.contiguous()
    block_outputs = 256
    output_count = replicas * selected_variables
    grid = (triton.cdiv(output_count, block_outputs),)
    _counter_gibbs_from_fields_kernel[grid](
        spins,
        fields,
        selected,
        replicas,
        selected_variables,
        spins.stride(0),
        float(beta),
        int(seed),
        int(counter_offset),
        block_outputs=block_outputs,
    )


def triton_fused_counter_gibbs_update(
    spins: torch.Tensor,
    neighbors: torch.Tensor,
    couplings: torch.Tensor,
    bias: torch.Tensor,
    indices: torch.Tensor,
    *,
    beta: float,
    seed: int,
    counter_offset: int,
) -> torch.Tensor:
    """Fuse padded local field, sigmoid, counter-v1 RNG, and in-place state write."""

    selected, replicas, selected_variables = _validate_update_inputs(spins, indices)
    if neighbors.dtype != torch.long or neighbors.ndim != 2:
        raise TypeError("neighbors must be a two-dimensional torch.long tensor")
    if couplings.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("The fused Triton update requires float32 coefficients")
    if couplings.shape != neighbors.shape or bias.shape != (neighbors.shape[0],):
        raise ValueError("Sparse coefficient shapes are inconsistent")
    if any(tensor.device != spins.device for tensor in (neighbors, couplings, bias)):
        raise ValueError("All fused Triton tensors must share one CUDA device")
    if spins.shape[1] != neighbors.shape[0]:
        raise ValueError("Spin shape does not match the sparse model size")
    if seed < 0 or seed >= 1 << 63 or counter_offset < 0 or counter_offset >= 1 << 63:
        raise ValueError("seed and counter_offset must fit nonnegative signed 64-bit integers")
    fields = torch.empty(
        (replicas, selected_variables),
        dtype=torch.float32,
        device=spins.device,
    )
    block_outputs = 256
    maximum_degree = int(neighbors.shape[1])
    block_degree = triton.next_power_of_2(max(maximum_degree, 1))
    output_count = replicas * selected_variables
    grid = (triton.cdiv(output_count, block_outputs),)
    _padded_counter_gibbs_kernel[grid](
        spins,
        neighbors,
        couplings,
        bias,
        selected,
        fields,
        replicas,
        selected_variables,
        spins.stride(0),
        neighbors.stride(0),
        couplings.stride(0),
        float(beta),
        int(seed),
        int(counter_offset),
        maximum_degree=maximum_degree,
        block_outputs=block_outputs,
        block_degree=block_degree,
    )
    return fields
