"""Batched PyTorch swap and 2-opt kernels on the feasible-tour state space."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import torch


@dataclass(frozen=True)
class SwapMove:
    first: torch.Tensor
    second: torch.Tensor


@dataclass(frozen=True)
class TwoOptMove:
    left_edge: torch.Tensor
    right_edge: torch.Tensor


@dataclass(frozen=True)
class PermutationStepResult:
    tours: torch.Tensor
    delta: torch.Tensor
    accepted: torch.Tensor


@dataclass(frozen=True)
class PermutationTracePoint:
    iteration: int
    beta: float
    current_length_mean: float
    current_length_std: float
    current_length_q05: float
    current_length_q25: float
    current_length_median: float
    current_length_q75: float
    current_length_q95: float
    best_length_mean: float
    best_length_std: float
    best_length_q05: float
    best_length_q25: float
    best_length_median: float
    best_length_q75: float
    best_length_q95: float
    cumulative_success_rate: float | None
    interval_accepted_moves: int = 0
    interval_proposed_moves: int = 0
    interval_acceptance_rate: float | None = None
    cumulative_acceptance_rate: float | None = None


@dataclass(frozen=True)
class PermutationMCMCResult:
    final_tours: torch.Tensor
    best_tours: torch.Tensor
    final_lengths: torch.Tensor
    best_lengths: torch.Tensor
    first_hit_iterations: torch.Tensor
    first_hit_tours: torch.Tensor
    first_hit_lengths: torch.Tensor
    total_iterations: int
    accepted_moves: int
    proposed_moves: int
    seed: int
    proposal: str
    acceptance: str
    device: str
    trace: tuple[PermutationTracePoint, ...] = ()
    kernel_seconds: float = 0.0
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    target_length: int | None = None
    probability_dtype: str = "torch.float32"
    update_mode: str = "masked_full"
    two_opt_pair_count: int | None = None
    swap_interval: int | None = None
    initialization: str = "random"

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_moves / self.proposed_moves


def validate_batched_tours(tours: torch.Tensor, *, fixed_city: int = 0) -> None:
    """Validate integer tours of shape [replicas, cities] with a fixed first city."""

    if tours.ndim != 2 or tours.shape[1] < 2:
        raise ValueError("tours must have shape [replicas, cities] with at least two cities")
    if tours.dtype.is_floating_point or tours.dtype == torch.bool:
        raise TypeError("tours must use an integer dtype")
    if torch.any(tours[:, 0] != fixed_city):
        raise ValueError("Every tour must start with fixed_city")
    expected = torch.arange(tours.shape[1], dtype=tours.dtype, device=tours.device)
    expected = expected.expand(tours.shape[0], -1)
    if not torch.equal(torch.sort(tours, dim=1).values, expected):
        raise ValueError("Every row must be a permutation of all city identifiers")


def random_batched_tours(
    replicas: int,
    cities: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate independent random permutations with city zero fixed first."""

    if replicas <= 0 or cities < 2:
        raise ValueError("replicas must be positive and cities must be at least two")
    selected_device = torch.device(device)
    keys = torch.rand((replicas, cities - 1), device=selected_device, generator=generator)
    suffix = torch.argsort(keys, dim=1).add(1)
    fixed = torch.zeros((replicas, 1), dtype=suffix.dtype, device=selected_device)
    return torch.cat((fixed, suffix), dim=1)


def perturbed_tour_population(
    seed_tours: torch.Tensor,
    replicas: int,
    seed: int,
    *,
    perturb_swaps: int = 0,
    perturbation: str = "swap",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample seed tours and apply reproducible swap or double-bridge kicks."""

    if replicas <= 0:
        raise ValueError("replicas must be positive")
    if perturb_swaps < 0:
        raise ValueError("perturb_swaps must be nonnegative")
    if perturbation not in {"swap", "double_bridge"}:
        raise ValueError("perturbation must be 'swap' or 'double_bridge'")
    if seed_tours.ndim != 2 or seed_tours.shape[0] == 0:
        raise ValueError("seed_tours must be a non-empty [seeds, cities] tensor")
    if seed_tours.dtype.is_floating_point or seed_tours.dtype == torch.bool:
        raise TypeError("seed_tours must use an integer dtype")
    selected_device = torch.device(device)
    population = seed_tours.to(device=selected_device, dtype=torch.long)
    validate_batched_tours(population)
    generator = torch.Generator(device=selected_device)
    generator.manual_seed(int(seed))
    if population.shape[0] == 1:
        tours = population.expand(replicas, -1).clone()
    else:
        choices = torch.randint(
            0,
            population.shape[0],
            (replicas,),
            device=selected_device,
            generator=generator,
        )
        tours = population.index_select(0, choices).clone()
    if perturbation == "double_bridge" and tours.shape[1] < 8:
        raise ValueError("double_bridge perturbations require at least eight cities")
    for _ in range(perturb_swaps):
        if perturbation == "swap":
            move = propose_swap(
                replicas,
                tours.shape[1],
                device=selected_device,
                generator=generator,
            )
            accepted = torch.ones(replicas, dtype=torch.bool, device=selected_device)
            _apply_swap_unchecked(tours, move, accepted, inplace=True)
        else:
            cities = tours.shape[1]
            compressed = torch.rand(
                (replicas, cities - 5),
                device=selected_device,
                generator=generator,
            ).argsort(dim=1)[:, :3]
            compressed = compressed.sort(dim=1).values.add(1)
            first = compressed[:, 0].add(1)
            second = compressed[:, 1].add(2)
            third = compressed[:, 2].add(3)
            positions = torch.arange(cities, device=selected_device).expand(replicas, -1)
            after_first = first + cities - third
            after_second = first + cities - second
            source = torch.where(
                positions < first[:, None],
                positions,
                torch.where(
                    positions < after_first[:, None],
                    third[:, None] + positions - first[:, None],
                    torch.where(
                        positions < after_second[:, None],
                        second[:, None] + positions - after_first[:, None],
                        first[:, None] + positions - after_second[:, None],
                    ),
                ),
            )
            tours = tours.gather(1, source)
    return tours


def tour_lengths(distance: torch.Tensor, tours: torch.Tensor) -> torch.Tensor:
    """Return one closed-tour length per replica."""

    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    if tours.ndim != 2 or tours.shape[1] != distance.shape[0]:
        raise ValueError("tour width must match the distance matrix")
    if distance.device != tours.device:
        raise ValueError("distance and tours must use the same device")
    successors = torch.roll(tours, shifts=-1, dims=1)
    return distance[tours, successors].sum(dim=1)


def propose_swap(
    replicas: int,
    cities: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> SwapMove:
    """Propose ordered, uniformly random distinct non-fixed positions."""

    if replicas <= 0 or cities < 3:
        raise ValueError("Swap proposals require replicas > 0 and at least three cities")
    selected_device = torch.device(device)
    first = torch.randint(1, cities, (replicas,), device=selected_device, generator=generator)
    second = torch.randint(1, cities - 1, (replicas,), device=selected_device, generator=generator)
    second = second + (second >= first).to(second.dtype)
    return SwapMove(first=first, second=second)


def _validate_move_shape(move_values: torch.Tensor, replicas: int, label: str) -> None:
    if (
        move_values.shape != (replicas,)
        or move_values.dtype.is_floating_point
        or move_values.dtype == torch.bool
    ):
        raise ValueError(f"{label} must be an integer tensor with shape [replicas]")


def _validate_kernel_inputs(distance: torch.Tensor, tours: torch.Tensor) -> None:
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    if tours.ndim != 2 or tours.shape[1] != distance.shape[0]:
        raise ValueError("tour width must match the distance matrix")
    if distance.device != tours.device:
        raise ValueError("distance and tours must use the same device")
    if tours.dtype.is_floating_point or tours.dtype == torch.bool:
        raise TypeError("tours must use an integer dtype")


def _validate_swap_move(move: SwapMove, replicas: int, cities: int, device: torch.device) -> None:
    _validate_move_shape(move.first, replicas, "move.first")
    _validate_move_shape(move.second, replicas, "move.second")
    if move.first.device != device or move.second.device != device:
        raise ValueError("moves and tours must use the same device")
    if torch.any(move.first < 1) or torch.any(move.first >= cities):
        raise IndexError("move.first contains an invalid position")
    if torch.any(move.second < 1) or torch.any(move.second >= cities):
        raise IndexError("move.second contains an invalid position")
    if torch.any(move.first == move.second):
        raise ValueError("Swap positions must be distinct")


def delta_swap(distance: torch.Tensor, tours: torch.Tensor, move: SwapMove) -> torch.Tensor:
    """Compute exact batched swap deltas from the four affected edge positions."""

    _validate_kernel_inputs(distance, tours)
    replicas, cities = tours.shape
    _validate_swap_move(move, replicas, cities, tours.device)
    return _delta_swap_unchecked(distance, tours, move)


def _delta_swap_unchecked(
    distance: torch.Tensor,
    tours: torch.Tensor,
    move: SwapMove,
) -> torch.Tensor:
    replicas, cities = tours.shape

    rows = torch.arange(replicas, device=tours.device)
    first_city = tours[rows, move.first]
    second_city = tours[rows, move.second]
    edge_positions = torch.stack(
        ((move.first - 1) % cities, move.first, (move.second - 1) % cities, move.second),
        dim=1,
    )
    unique = torch.ones_like(edge_positions, dtype=torch.bool)
    for column in range(1, edge_positions.shape[1]):
        unique[:, column] = ~(edge_positions[:, column, None] == edge_positions[:, :column]).any(
            dim=1
        )
    edge_successors = (edge_positions + 1) % cities
    old_start = tours.gather(1, edge_positions)
    old_end = tours.gather(1, edge_successors)

    def city_after_swap(positions: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        replacement = torch.where(positions == move.first[:, None], second_city[:, None], original)
        return torch.where(positions == move.second[:, None], first_city[:, None], replacement)

    new_start = city_after_swap(edge_positions, old_start)
    new_end = city_after_swap(edge_successors, old_end)
    edge_delta = distance[new_start, new_end] - distance[old_start, old_end]
    return (edge_delta * unique).sum(dim=1)


def apply_swap(
    tours: torch.Tensor,
    move: SwapMove,
    accepted: torch.Tensor | None = None,
) -> torch.Tensor:
    if tours.ndim != 2:
        raise ValueError("tours must have shape [replicas, cities]")
    replicas = tours.shape[0]
    _validate_swap_move(move, replicas, tours.shape[1], tours.device)
    if accepted is None:
        accepted = torch.ones(replicas, dtype=torch.bool, device=tours.device)
    if accepted.shape != (replicas,) or accepted.dtype != torch.bool:
        raise ValueError("accepted must be a boolean tensor with shape [replicas]")
    if accepted.device != tours.device:
        raise ValueError("accepted and tours must use the same device")
    return _apply_swap_unchecked(tours, move, accepted)


def _apply_swap_unchecked(
    tours: torch.Tensor,
    move: SwapMove,
    accepted: torch.Tensor,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    replicas = tours.shape[0]
    result = tours if inplace else tours.clone()
    rows = torch.arange(replicas, device=tours.device)[accepted]
    first = move.first[accepted]
    second = move.second[accepted]
    first_values = tours[rows, first]
    second_values = tours[rows, second]
    result[rows, first] = second_values
    result[rows, second] = first_values
    return result


@cache
def _valid_two_opt_pair_values(cities: int) -> tuple[tuple[int, int], ...]:
    if cities < 4:
        raise ValueError("2-opt requires at least four cities")
    return tuple(
        (left, right)
        for left in range(cities - 1)
        for right in range(left + 2, cities)
        if not (left == 0 and right == cities - 1)
    )


def valid_two_opt_pairs(cities: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Enumerate edge-index pairs that define a nontrivial symmetric 2-opt move."""

    return torch.tensor(_valid_two_opt_pair_values(cities), dtype=torch.long, device=device)


def propose_two_opt(
    replicas: int,
    cities: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    pair_table: torch.Tensor | None = None,
) -> TwoOptMove:
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    selected_device = torch.device(device)
    pairs = pair_table
    if pairs is None:
        pairs = valid_two_opt_pairs(cities, device=selected_device)
    elif (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or pairs.shape[0] == 0
        or pairs.dtype != torch.long
        or pairs.device != selected_device
    ):
        raise ValueError("pair_table must be a non-empty torch.long [pairs, 2] tensor on device")
    selected = torch.randint(
        0, pairs.shape[0], (replicas,), device=pairs.device, generator=generator
    )
    return TwoOptMove(left_edge=pairs[selected, 0], right_edge=pairs[selected, 1])


def delta_two_opt(distance: torch.Tensor, tours: torch.Tensor, move: TwoOptMove) -> torch.Tensor:
    """Compute the four-edge delta for a symmetric TSP 2-opt proposal."""

    _validate_kernel_inputs(distance, tours)
    replicas, cities = tours.shape
    _validate_two_opt_move(move, replicas, cities, tours.device)
    return _delta_two_opt_unchecked(distance, tours, move)


def _delta_two_opt_unchecked(
    distance: torch.Tensor,
    tours: torch.Tensor,
    move: TwoOptMove,
) -> torch.Tensor:
    replicas, cities = tours.shape
    rows = torch.arange(replicas, device=tours.device)
    first = tours[rows, move.left_edge]
    second = tours[rows, move.left_edge + 1]
    third = tours[rows, move.right_edge]
    fourth = tours[rows, (move.right_edge + 1) % cities]
    return (
        distance[first, third]
        + distance[second, fourth]
        - distance[first, second]
        - distance[third, fourth]
    )


def _validate_two_opt_move(
    move: TwoOptMove,
    replicas: int,
    cities: int,
    device: torch.device,
) -> None:
    _validate_move_shape(move.left_edge, replicas, "move.left_edge")
    _validate_move_shape(move.right_edge, replicas, "move.right_edge")
    if move.left_edge.device != device or move.right_edge.device != device:
        raise ValueError("moves and tours must use the same device")
    invalid = (
        (move.left_edge < 0)
        | (move.right_edge >= cities)
        | (move.right_edge < move.left_edge + 2)
        | ((move.left_edge == 0) & (move.right_edge == cities - 1))
    )
    if torch.any(invalid):
        raise ValueError("Invalid 2-opt edge pair")


def apply_two_opt(
    tours: torch.Tensor,
    move: TwoOptMove,
    accepted: torch.Tensor | None = None,
) -> torch.Tensor:
    if tours.ndim != 2:
        raise ValueError("tours must have shape [replicas, cities]")
    replicas, cities = tours.shape
    _validate_two_opt_move(move, replicas, cities, tours.device)
    if accepted is None:
        accepted = torch.ones(replicas, dtype=torch.bool, device=tours.device)
    if accepted.shape != (replicas,) or accepted.dtype != torch.bool:
        raise ValueError("accepted must be a boolean tensor with shape [replicas]")
    if accepted.device != tours.device:
        raise ValueError("accepted and tours must use the same device")
    return _apply_two_opt_unchecked(tours, move, accepted)


def _apply_two_opt_unchecked(
    tours: torch.Tensor,
    move: TwoOptMove,
    accepted: torch.Tensor,
    *,
    inplace: bool = False,
    update_mode: str = "accepted_only",
) -> torch.Tensor:
    replicas, cities = tours.shape
    if update_mode not in {"masked_full", "accepted_only"}:
        raise ValueError("update_mode must be 'masked_full' or 'accepted_only'")
    if update_mode == "masked_full":
        positions = torch.arange(cities, device=tours.device).expand(replicas, -1)
        inside = (positions > move.left_edge[:, None]) & (positions <= move.right_edge[:, None])
        reversed_positions = move.left_edge[:, None] + move.right_edge[:, None] + 1 - positions
        source_positions = torch.where(inside, reversed_positions, positions)
        candidate = tours.gather(1, source_positions)
        updated = torch.where(accepted[:, None], candidate, tours)
        if inplace:
            tours.copy_(updated)
            return tours
        return updated

    result = tours if inplace else tours.clone()
    accepted_rows = torch.arange(replicas, device=tours.device)[accepted]
    accepted_count = int(accepted_rows.numel())
    if accepted_count == 0:
        return result
    selected_tours = tours.index_select(0, accepted_rows)
    left_edge = move.left_edge.index_select(0, accepted_rows)
    right_edge = move.right_edge.index_select(0, accepted_rows)
    positions = torch.arange(cities, device=tours.device).expand(accepted_count, -1)
    inside = (positions > left_edge[:, None]) & (positions <= right_edge[:, None])
    reversed_positions = left_edge[:, None] + right_edge[:, None] + 1 - positions
    source_positions = torch.where(inside, reversed_positions, positions)
    result[accepted_rows] = selected_tours.gather(1, source_positions)
    return result


def _scaled_delta(
    delta: torch.Tensor,
    beta: float | torch.Tensor,
    probability_dtype: torch.dtype,
) -> torch.Tensor:
    if probability_dtype not in {torch.float32, torch.float64}:
        raise ValueError("probability_dtype must be torch.float32 or torch.float64")
    if isinstance(beta, torch.Tensor):
        selected = beta.to(device=delta.device, dtype=probability_dtype)
        if selected.ndim not in (0, 1) or (selected.ndim == 1 and selected.shape != delta.shape):
            raise ValueError("beta tensor must be scalar or have shape [replicas]")
        if torch.any(~torch.isfinite(selected)) or torch.any(selected <= 0):
            raise ValueError("beta must be finite and positive")
    else:
        if not math.isfinite(float(beta)) or float(beta) <= 0:
            raise ValueError("beta must be finite and positive")
        selected = torch.tensor(float(beta), dtype=probability_dtype, device=delta.device)
    return delta.to(probability_dtype) * selected


def barker_accept(
    delta: torch.Tensor,
    beta: float | torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    probability_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw Barker logistic accept/reject decisions."""

    scaled = _scaled_delta(delta, beta, probability_dtype)
    probabilities = torch.sigmoid(-scaled)
    random_values = torch.rand(
        probabilities.shape,
        dtype=probabilities.dtype,
        device=probabilities.device,
        generator=generator,
    )
    return random_values < probabilities


def metropolis_accept(
    delta: torch.Tensor,
    beta: float | torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    probability_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw numerically stable Metropolis accept/reject decisions."""

    scaled = _scaled_delta(delta, beta, probability_dtype)
    random_values = torch.rand(
        scaled.shape, dtype=scaled.dtype, device=scaled.device, generator=generator
    )
    return torch.log(random_values) < torch.minimum(torch.zeros_like(scaled), -scaled)


def update_first_hits(
    first_hit_iterations: torch.Tensor,
    best_lengths: torch.Tensor,
    *,
    target_length: int,
    iteration: int,
) -> torch.Tensor:
    """Record previously unseen target hits without mutating the input tensor."""

    if first_hit_iterations.shape != best_lengths.shape or first_hit_iterations.ndim != 1:
        raise ValueError("first hits and best lengths must have matching one-dimensional shapes")
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    result = first_hit_iterations.clone()
    _update_first_hits_inplace(
        result,
        best_lengths,
        target_length=target_length,
        iteration=iteration,
    )
    return result


def _update_first_hits_inplace(
    first_hit_iterations: torch.Tensor,
    best_lengths: torch.Tensor,
    *,
    target_length: int,
    iteration: int,
) -> torch.Tensor:
    if first_hit_iterations.shape != best_lengths.shape or first_hit_iterations.ndim != 1:
        raise ValueError("first hits and best lengths must have matching one-dimensional shapes")
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    new_hits = (first_hit_iterations < 0) & (best_lengths <= target_length)
    first_hit_iterations[new_hits] = int(iteration)
    return new_hits


def _record_first_hit_states(
    first_hit_tours: torch.Tensor,
    first_hit_lengths: torch.Tensor,
    candidate_tours: torch.Tensor,
    candidate_lengths: torch.Tensor,
    new_hits: torch.Tensor,
) -> None:
    """Record first-hit states without dynamic-shape indexing or device sync."""

    first_hit_tours.copy_(torch.where(new_hits[:, None], candidate_tours, first_hit_tours))
    first_hit_lengths.copy_(torch.where(new_hits, candidate_lengths, first_hit_lengths))


def _length_trace_point(
    iteration: int,
    beta: float,
    lengths: torch.Tensor,
    best_lengths: torch.Tensor,
    first_hits: torch.Tensor,
    *,
    has_target: bool,
    interval_accepted_moves: int = 0,
    interval_proposed_moves: int = 0,
    cumulative_accepted_moves: int = 0,
    cumulative_proposed_moves: int = 0,
) -> PermutationTracePoint:
    current = lengths.to(torch.float32)
    best = best_lengths.to(torch.float32)
    probabilities = torch.tensor(
        [0.05, 0.25, 0.5, 0.75, 0.95],
        dtype=torch.float32,
        device=lengths.device,
    )
    current_quantiles = torch.quantile(current, probabilities)
    best_quantiles = torch.quantile(best, probabilities)
    success_rate = None
    if has_target:
        success_rate = float((first_hits >= 0).to(torch.float32).mean().item())
    return PermutationTracePoint(
        iteration=iteration,
        beta=float(beta),
        current_length_mean=float(current.mean().item()),
        current_length_std=float(current.std(unbiased=False).item()),
        current_length_q05=float(current_quantiles[0].item()),
        current_length_q25=float(current_quantiles[1].item()),
        current_length_median=float(current_quantiles[2].item()),
        current_length_q75=float(current_quantiles[3].item()),
        current_length_q95=float(current_quantiles[4].item()),
        best_length_mean=float(best.mean().item()),
        best_length_std=float(best.std(unbiased=False).item()),
        best_length_q05=float(best_quantiles[0].item()),
        best_length_q25=float(best_quantiles[1].item()),
        best_length_median=float(best_quantiles[2].item()),
        best_length_q75=float(best_quantiles[3].item()),
        best_length_q95=float(best_quantiles[4].item()),
        cumulative_success_rate=success_rate,
        interval_accepted_moves=interval_accepted_moves,
        interval_proposed_moves=interval_proposed_moves,
        interval_acceptance_rate=(
            interval_accepted_moves / interval_proposed_moves if interval_proposed_moves else None
        ),
        cumulative_acceptance_rate=(
            cumulative_accepted_moves / cumulative_proposed_moves
            if cumulative_proposed_moves
            else None
        ),
    )


def _all_two_opt_deltas(
    distance: torch.Tensor,
    tours: torch.Tensor,
    pair_table: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the ordered 2-opt neighborhood for every batched tour."""

    left = pair_table[:, 0]
    right = pair_table[:, 1]
    first = tours.index_select(1, left)
    second = tours.index_select(1, left + 1)
    third = tours.index_select(1, right)
    fourth = tours.index_select(1, (right + 1) % tours.shape[1])
    deltas = distance[first, third]
    deltas.add_(distance[second, fourth])
    deltas.sub_(distance[first, second])
    deltas.sub_(distance[third, fourth])
    return deltas


@torch.inference_mode()
def run_batched_first_improvement_two_opt(
    distance: torch.Tensor,
    initial_tours: torch.Tensor,
    max_passes: int,
    seed: int,
    *,
    target_length: int | None = None,
    record_every: int = 1,
    initialization_label: str = "provided",
) -> PermutationMCMCResult:
    """Run fixed-order first-improvement 2-opt descent on independent tours.

    One reported iteration is one complete ordered-neighborhood pass per tour,
    not one MCMC proposal. The routine intentionally has no temperature or
    stochastic acceptance rule.
    """

    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    if initial_tours.ndim != 2 or initial_tours.shape[1] != distance.shape[0]:
        raise ValueError("initial_tours must have shape [replicas, cities]")
    if initial_tours.dtype.is_floating_point or initial_tours.dtype == torch.bool:
        raise TypeError("initial_tours must use an integer dtype")
    if not torch.equal(distance, distance.mT):
        raise ValueError("First-improvement descent requires symmetric distances")
    if torch.any(torch.diagonal(distance) != 0):
        raise ValueError("distance diagonal must be zero")
    if max_passes <= 0 or record_every <= 0:
        raise ValueError("max_passes and record_every must be positive")
    if not initialization_label.strip():
        raise ValueError("initialization_label must be non-empty")

    device = distance.device
    tours = initial_tours.to(device=device, dtype=torch.long).clone()
    _validate_kernel_inputs(distance, tours)
    validate_batched_tours(tours)
    replicas = tours.shape[0]
    lengths = tour_lengths(distance, tours)
    first_hits = torch.full((replicas,), -1, dtype=torch.int64, device=device)
    first_hit_tours = torch.full_like(tours, -1)
    first_hit_lengths = torch.full_like(lengths, -1)
    if target_length is not None:
        new_hits = _update_first_hits_inplace(
            first_hits,
            lengths,
            target_length=int(target_length),
            iteration=0,
        )
        _record_first_hit_states(
            first_hit_tours,
            first_hit_lengths,
            tours,
            lengths,
            new_hits,
        )
    trace = [
        _length_trace_point(
            0,
            1.0,
            lengths,
            lengths,
            first_hits,
            has_target=target_length is not None,
        )
    ]
    pair_table = valid_two_opt_pairs(tours.shape[1], device=device)
    accepted_moves = torch.zeros((), dtype=torch.int64, device=device)
    accepted_at_last_trace = 0
    iteration_at_last_trace = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for iteration in range(1, max_passes + 1):
        deltas = _all_two_opt_deltas(distance, tours, pair_table)
        improving = deltas < 0
        accepted = improving.any(dim=1)
        selected_pair = improving.to(torch.uint8).argmax(dim=1)
        selected_delta = deltas.gather(1, selected_pair[:, None]).squeeze(1)
        move = TwoOptMove(
            left_edge=pair_table[selected_pair, 0],
            right_edge=pair_table[selected_pair, 1],
        )
        tours = _apply_two_opt_unchecked(
            tours,
            move,
            accepted,
            update_mode="masked_full",
        )
        lengths = lengths + torch.where(
            accepted,
            selected_delta,
            torch.zeros_like(selected_delta),
        )
        accepted_moves += accepted.sum()
        if target_length is not None:
            new_hits = _update_first_hits_inplace(
                first_hits,
                lengths,
                target_length=int(target_length),
                iteration=iteration,
            )
            _record_first_hit_states(
                first_hit_tours,
                first_hit_lengths,
                tours,
                lengths,
                new_hits,
            )
        if iteration % record_every == 0:
            accepted_total = int(accepted_moves.item())
            trace.append(
                _length_trace_point(
                    iteration,
                    1.0,
                    lengths,
                    lengths,
                    first_hits,
                    has_target=target_length is not None,
                    interval_accepted_moves=accepted_total - accepted_at_last_trace,
                    interval_proposed_moves=(iteration - iteration_at_last_trace) * replicas,
                    cumulative_accepted_moves=accepted_total,
                    cumulative_proposed_moves=iteration * replicas,
                )
            )
            accepted_at_last_trace = accepted_total
            iteration_at_last_trace = iteration

    if trace[-1].iteration != max_passes:
        accepted_total = int(accepted_moves.item())
        trace.append(
            _length_trace_point(
                max_passes,
                1.0,
                lengths,
                lengths,
                first_hits,
                has_target=target_length is not None,
                interval_accepted_moves=accepted_total - accepted_at_last_trace,
                interval_proposed_moves=(max_passes - iteration_at_last_trace) * replicas,
                cumulative_accepted_moves=accepted_total,
                cumulative_proposed_moves=max_passes * replicas,
            )
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    kernel_seconds = time.perf_counter() - start
    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

    return PermutationMCMCResult(
        final_tours=tours.cpu(),
        best_tours=tours.cpu(),
        final_lengths=lengths.cpu(),
        best_lengths=lengths.cpu(),
        first_hit_iterations=first_hits.cpu(),
        first_hit_tours=first_hit_tours.cpu(),
        first_hit_lengths=first_hit_lengths.cpu(),
        total_iterations=max_passes,
        accepted_moves=int(accepted_moves.item()),
        proposed_moves=replicas * max_passes,
        seed=int(seed),
        proposal="two_opt_first_descent",
        acceptance="greedy",
        device=str(device),
        trace=tuple(trace),
        kernel_seconds=kernel_seconds,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        target_length=None if target_length is None else int(target_length),
        probability_dtype="not_applicable",
        update_mode="masked_full",
        two_opt_pair_count=int(pair_table.shape[0]),
        initialization=initialization_label,
    )


def permutation_mcmc_step(
    distance: torch.Tensor,
    tours: torch.Tensor,
    beta: float | torch.Tensor,
    *,
    proposal: str = "swap",
    acceptance: str = "metropolis",
    generator: torch.Generator | None = None,
    probability_dtype: torch.dtype = torch.float32,
    two_opt_pairs: torch.Tensor | None = None,
    inplace: bool = False,
    tour_update_mode: str = "masked_full",
) -> PermutationStepResult:
    """Apply one independent symmetric proposal to every replica."""

    _validate_kernel_inputs(distance, tours)
    if tour_update_mode not in {"masked_full", "accepted_only"}:
        raise ValueError("tour_update_mode must be 'masked_full' or 'accepted_only'")
    replicas, cities = tours.shape
    if proposal == "swap":
        move = propose_swap(replicas, cities, device=tours.device, generator=generator)
        delta = _delta_swap_unchecked(distance, tours, move)
    elif proposal == "two_opt":
        move = propose_two_opt(
            replicas,
            cities,
            device=tours.device,
            generator=generator,
            pair_table=two_opt_pairs,
        )
        delta = _delta_two_opt_unchecked(distance, tours, move)
    else:
        raise ValueError("proposal must be 'swap' or 'two_opt'")

    if acceptance == "metropolis":
        accepted = metropolis_accept(
            delta,
            beta,
            generator=generator,
            probability_dtype=probability_dtype,
        )
    elif acceptance == "barker":
        accepted = barker_accept(
            delta,
            beta,
            generator=generator,
            probability_dtype=probability_dtype,
        )
    elif acceptance == "greedy":
        accepted = delta < 0
    else:
        raise ValueError("acceptance must be 'metropolis', 'barker', or 'greedy'")
    if proposal == "swap":
        updated_tours = _apply_swap_unchecked(tours, move, accepted, inplace=inplace)
    else:
        updated_tours = _apply_two_opt_unchecked(
            tours,
            move,
            accepted,
            inplace=inplace,
            update_mode=tour_update_mode,
        )
    return PermutationStepResult(tours=updated_tours, delta=delta, accepted=accepted)


@torch.inference_mode()
def run_permutation_mcmc(
    distance: torch.Tensor,
    betas: Sequence[float],
    steps_per_beta: int,
    replicas: int,
    seed: int,
    *,
    proposal: str = "swap",
    acceptance: str = "metropolis",
    initial_tours: torch.Tensor | None = None,
    target_length: int | None = None,
    record_every: int = 1,
    probability_dtype: torch.dtype = torch.float32,
    tour_update_mode: str = "masked_full",
    swap_interval: int = 16,
    initialization_label: str | None = None,
) -> PermutationMCMCResult:
    """Run batched feasible-space MCMC with city zero fixed at position zero."""

    schedule = tuple(float(beta) for beta in betas)
    if not schedule or any(not math.isfinite(beta) or beta <= 0 for beta in schedule):
        raise ValueError("betas must be a non-empty sequence of finite positive values")
    if steps_per_beta <= 0 or replicas <= 0 or record_every <= 0:
        raise ValueError("steps_per_beta, replicas, and record_every must be positive")
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    if not torch.equal(distance, distance.mT):
        raise ValueError("Permutation MCMC currently supports only symmetric distances")
    if torch.any(torch.diagonal(distance) != 0):
        raise ValueError("distance diagonal must be zero")
    if proposal not in {"swap", "two_opt", "mixed"}:
        raise ValueError("proposal must be 'swap', 'two_opt', or 'mixed'")
    if acceptance not in {"metropolis", "barker", "greedy"}:
        raise ValueError("acceptance must be 'metropolis', 'barker', or 'greedy'")
    if probability_dtype not in {torch.float32, torch.float64}:
        raise ValueError("probability_dtype must be torch.float32 or torch.float64")
    if tour_update_mode not in {"masked_full", "accepted_only"}:
        raise ValueError("tour_update_mode must be 'masked_full' or 'accepted_only'")
    if swap_interval <= 0:
        raise ValueError("swap_interval must be positive")
    if initialization_label is not None and not initialization_label.strip():
        raise ValueError("initialization_label must be non-empty when provided")

    device = distance.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    if initial_tours is None:
        tours = random_batched_tours(
            replicas, distance.shape[0], device=device, generator=generator
        )
    else:
        if initial_tours.shape != (replicas, distance.shape[0]):
            raise ValueError("initial_tours must have shape [replicas, cities]")
        if initial_tours.dtype.is_floating_point or initial_tours.dtype == torch.bool:
            raise TypeError("initial_tours must use an integer dtype")
        tours = initial_tours.to(device=device, dtype=torch.long).clone()
    validate_batched_tours(tours)

    lengths = tour_lengths(distance, tours)
    best_lengths = lengths.clone()
    best_tours = tours.clone()
    first_hits = torch.full((replicas,), -1, dtype=torch.int64, device=device)
    first_hit_tours = torch.full_like(tours, -1)
    first_hit_lengths = torch.full_like(lengths, -1)
    if target_length is not None:
        new_hits = _update_first_hits_inplace(
            first_hits,
            best_lengths,
            target_length=int(target_length),
            iteration=0,
        )
        _record_first_hit_states(
            first_hit_tours,
            first_hit_lengths,
            best_tours,
            best_lengths,
            new_hits,
        )

    iteration = 0
    accepted_moves = torch.zeros((), dtype=torch.int64, device=device)
    accepted_at_last_trace = 0
    iteration_at_last_trace = 0
    trace = [
        _length_trace_point(
            0,
            schedule[0],
            lengths,
            best_lengths,
            first_hits,
            has_target=target_length is not None,
        )
    ]
    two_opt_pairs = (
        valid_two_opt_pairs(distance.shape[0], device=device)
        if proposal in {"two_opt", "mixed"}
        else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for beta in schedule:
        for _ in range(steps_per_beta):
            iteration += 1
            if proposal == "mixed":
                active_proposal = "swap" if iteration % swap_interval == 0 else "two_opt"
            else:
                active_proposal = proposal
            step = permutation_mcmc_step(
                distance,
                tours,
                beta,
                proposal=active_proposal,
                acceptance=acceptance,
                generator=generator,
                probability_dtype=probability_dtype,
                two_opt_pairs=two_opt_pairs,
                inplace=tour_update_mode == "accepted_only",
                tour_update_mode=tour_update_mode,
            )
            tours = step.tours
            if tour_update_mode == "masked_full":
                lengths = lengths + torch.where(
                    step.accepted,
                    step.delta,
                    torch.zeros_like(step.delta),
                )
            else:
                lengths[step.accepted] += step.delta[step.accepted]
            accepted_moves += step.accepted.sum()
            if acceptance == "greedy":
                best_lengths = lengths
                best_tours = tours
            else:
                improved = lengths < best_lengths
                if tour_update_mode == "masked_full":
                    best_lengths = torch.where(improved, lengths, best_lengths)
                    best_tours = torch.where(improved[:, None], tours, best_tours)
                else:
                    best_lengths[improved] = lengths[improved]
                    best_tours[improved] = tours[improved]
            if target_length is not None:
                new_hits = _update_first_hits_inplace(
                    first_hits,
                    best_lengths,
                    target_length=int(target_length),
                    iteration=iteration,
                )
                _record_first_hit_states(
                    first_hit_tours,
                    first_hit_lengths,
                    best_tours,
                    best_lengths,
                    new_hits,
                )
            if iteration % record_every == 0:
                accepted_total = int(accepted_moves.item())
                trace.append(
                    _length_trace_point(
                        iteration,
                        beta,
                        lengths,
                        best_lengths,
                        first_hits,
                        has_target=target_length is not None,
                        interval_accepted_moves=accepted_total - accepted_at_last_trace,
                        interval_proposed_moves=(iteration - iteration_at_last_trace) * replicas,
                        cumulative_accepted_moves=accepted_total,
                        cumulative_proposed_moves=iteration * replicas,
                    )
                )
                accepted_at_last_trace = accepted_total
                iteration_at_last_trace = iteration

    if trace[-1].iteration != iteration:
        accepted_total = int(accepted_moves.item())
        trace.append(
            _length_trace_point(
                iteration,
                schedule[-1],
                lengths,
                best_lengths,
                first_hits,
                has_target=target_length is not None,
                interval_accepted_moves=accepted_total - accepted_at_last_trace,
                interval_proposed_moves=(iteration - iteration_at_last_trace) * replicas,
                cumulative_accepted_moves=accepted_total,
                cumulative_proposed_moves=iteration * replicas,
            )
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    kernel_seconds = time.perf_counter() - start
    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

    return PermutationMCMCResult(
        final_tours=tours.cpu(),
        best_tours=best_tours.cpu(),
        final_lengths=lengths.cpu(),
        best_lengths=best_lengths.cpu(),
        first_hit_iterations=first_hits.cpu(),
        first_hit_tours=first_hit_tours.cpu(),
        first_hit_lengths=first_hit_lengths.cpu(),
        total_iterations=iteration,
        accepted_moves=int(accepted_moves.item()),
        proposed_moves=replicas * iteration,
        seed=int(seed),
        proposal=proposal,
        acceptance=acceptance,
        device=str(device),
        trace=tuple(trace),
        kernel_seconds=kernel_seconds,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        target_length=None if target_length is None else int(target_length),
        probability_dtype=str(probability_dtype),
        update_mode=tour_update_mode,
        two_opt_pair_count=(None if two_opt_pairs is None else int(two_opt_pairs.shape[0])),
        swap_interval=(swap_interval if proposal == "mixed" else None),
        initialization=(
            initialization_label
            if initialization_label is not None
            else ("random" if initial_tours is None else "provided")
        ),
    )
