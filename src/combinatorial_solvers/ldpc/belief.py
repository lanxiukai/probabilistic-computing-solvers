"""Batched flooding BP controls using the same syndrome-only receiver inputs."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from .model import SCALE


def solve(instance, *, device="cuda", iterations=200, method="minimum-sum", alpha=0.75):
    if iterations < 1 or method not in ("minimum-sum", "sum-product") or not 0 < alpha <= 1:
        raise ValueError("Invalid belief-propagation settings")
    started = perf_counter()
    device = torch.device(device)
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    code = instance.code
    edge_count = len(code.columns)
    check_degree = int(np.diff(code.pointers).max())
    row_edges = np.full((code.checks, check_degree), edge_count, dtype=np.int64)
    variable_edges = [[] for _ in range(code.size)]
    edge_rows, edge_positions = [], []
    for a in range(code.checks):
        edges = np.arange(code.pointers[a], code.pointers[a + 1])
        row_edges[a, : len(edges)] = edges
        for j, e in enumerate(edges):
            variable_edges[code.columns[e]].append(int(e))
            edge_rows.append(a)
            edge_positions.append(j)
    degree = max(map(len, variable_edges))
    var_edges = np.full((code.size, degree), edge_count, dtype=np.int64)
    for i, edges in enumerate(variable_edges):
        var_edges[i, : len(edges)] = edges
    row_edges = torch.as_tensor(row_edges, device=device)
    var_edges = torch.as_tensor(var_edges, device=device)
    edge_rows = torch.as_tensor(edge_rows, device=device)
    edge_positions = torch.as_tensor(edge_positions, device=device)
    columns = torch.as_tensor(code.columns, device=device)
    check_bits = torch.as_tensor(code.padded(), device=device)
    prior = torch.as_tensor(instance.costs / SCALE, dtype=torch.float64, device=device)
    target = torch.as_tensor(instance.targets, device=device)
    signs = (1 - 2 * target).to(torch.float64)
    zeros = torch.zeros((instance.frames, 1), dtype=torch.float64, device=device)
    messages = prior[:, columns].clone()
    decoded = torch.zeros((instance.frames, code.size), dtype=torch.int8, device=device)
    posterior = prior.clone()
    active = target.any(-1)
    used = torch.zeros(instance.frames, dtype=torch.int64, device=device)
    first_valid_seconds = np.full(instance.frames, np.nan)
    first_valid_seconds[~active.cpu().numpy()] = 0.0
    sync()
    setup_seconds = perf_counter() - started
    search = perf_counter()
    for iteration in range(1, iterations + 1):
        if not bool(active.any()):
            break
        if method == "minimum-sum":
            padded = torch.cat([messages, torch.full_like(zeros, torch.inf)], dim=1)
            local = padded[:, row_edges]
            magnitude, where = local.abs().min(-1, keepdim=True)
            masked = local.abs().scatter(-1, where, torch.inf)
            second = masked.min(-1, keepdim=True).values
            positions = torch.arange(check_degree, device=device)
            magnitudes = torch.where(positions == where, second, magnitude)
            local_sign = torch.where(local <= 0, -1.0, 1.0)
            outgoing = (
                alpha
                * signs[..., None]
                * local_sign.prod(-1, keepdim=True)
                * local_sign
                * magnitudes
            )
        else:
            padded = torch.cat([torch.tanh(messages / 2), torch.ones_like(zeros)], dim=1)
            local = padded[:, row_edges]
            ones = torch.ones_like(local[..., :1])
            prefix = torch.cat([ones, local.cumprod(-1)[..., :-1]], dim=-1)
            suffix = torch.cat([local.flip(-1).cumprod(-1).flip(-1)[..., 1:], ones], dim=-1)
            outgoing = 2 * torch.atanh(
                (prefix * suffix * signs[..., None]).clamp(-1 + 1e-12, 1 - 1e-12)
            )
        edge_messages = outgoing[:, edge_rows, edge_positions]
        padded_r = torch.cat([edge_messages, zeros], dim=1)
        current_posterior = prior + padded_r[:, var_edges].sum(-1)
        if not bool(torch.isfinite(current_posterior).all()):
            raise AssertionError("Nonfinite BP messages")
        current_bits = (current_posterior <= 0).to(torch.int8)
        padded_bits = torch.cat([current_bits, zeros.to(torch.int8)], dim=1)
        syndrome = padded_bits[:, check_bits].sum(-1) % 2
        valid = (syndrome == target).all(-1)
        next_messages = current_posterior[:, columns] - edge_messages
        if method == "sum-product":
            next_messages.clamp_(-50, 50)
        messages[active] = next_messages[active]
        decoded[active] = current_bits[active]
        posterior[active] = current_posterior[active]
        used[active] = iteration
        newly_valid = active & valid
        sync()
        elapsed = perf_counter() - search
        first_valid_seconds[newly_valid.cpu().numpy()] = elapsed
        active &= ~valid
    sync()
    search_seconds = perf_counter() - search
    errors = decoded.cpu().numpy()
    cost = instance.cost(errors)
    violations = instance.violations(errors)
    return {
        "initial_bits": np.zeros((instance.frames, 1, code.size), dtype=np.int8),
        "best_bits": errors[:, None, :],
        "final_bits": errors[:, None, :],
        "best_cost": cost[:, None],
        "best_violations": violations[:, None],
        "final_cost": cost[:, None],
        "final_violations": violations[:, None],
        "decoded_error": errors,
        "bp_bits": errors,
        "posterior": posterior.cpu().numpy(),
        "iterations": used.cpu().numpy(),
        "first_feasible_seconds": first_valid_seconds,
    }, {
        "method": f"gpu-{method}",
        "frames": instance.frames,
        "replicas": 1,
        "valid_frames": int(np.count_nonzero(violations == 0)),
        "feasible_cost_sum": int(cost[violations == 0].sum()),
        "physical_update_pbits_per_model": 0,
        "auxiliary_pbits": 0,
        "setup_seconds": setup_seconds,
        "search_seconds": search_seconds,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
