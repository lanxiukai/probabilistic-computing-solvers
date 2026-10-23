"""Exact batched CUDA 2-opt/Or-opt descent with one Triton block per tour.

AB-cycle construction, its probability updates, subtour repair, and independent
certificates remain on the CPU. This module accelerates deterministic local
search, not a physical p-bit device or the complete TSP solver.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from time import perf_counter

import numpy as np
import torch
import triton
import triton.language as tl

from .hybrid_moves import Tour, canonical_tour
from .instance import TSPInstance, certify_tour


@triton.jit
def _local_descent(
    Distance,
    State,
    MaxMoves,
    N: tl.constexpr,
    UseOr: tl.constexpr,
    Tile: tl.constexpr,
    CityBlock: tl.constexpr,
):
    row = State + tl.program_id(0) * (N + 2)
    lane = tl.arange(0, Tile)
    position = tl.arange(0, CityBlock)
    moves = 0
    active = True
    # Triton handles these annotations and iterators in its compiler, not Python.
    sentinel: tl.constexpr = 9223372036854775807  # pyright: ignore[reportAssignmentType]
    while (moves < MaxMoves) & active:
        first = N * N
        # Tile order and minimum index reproduce scalar first-improvement order.
        for base in range(0, N * N, Tile):
            if first == N * N:
                pair = base + lane
                i, j = pair // N, pair % N
                valid = (pair < N * N) & (j >= i + 2) & ~((i == 0) & (j == N - 1))
                a = tl.load(row + i, valid, 0)
                b = tl.load(row + i + 1, valid, 0)
                c = tl.load(row + j, valid, 0)
                d = tl.load(row + (j + 1) % N, valid, 0)
                delta = (
                    tl.load(Distance + a * N + c, valid, 0)
                    + tl.load(Distance + b * N + d, valid, 0)
                    - tl.load(Distance + a * N + b, valid, 0)
                    - tl.load(Distance + c * N + d, valid, 0)
                )
                first = tl.min(tl.where(valid & (delta < 0), pair, N * N), 0)
        if first < N * N:
            left, right = first // N, first % N
            city = tl.load(row + position, position < N, 0)
            destination = tl.where(
                (position > left) & (position <= right), left + right + 1 - position, position
            )
            # All old cities must be read before a different warp overwrites them.
            tl.debug_barrier()
            tl.store(row + destination, city, position < N)
            tl.debug_barrier()
            moves += 1
        else:
            best_delta = tl.full((), 0, tl.int64)
            best_size, best_start, best_after, best_reverse = 0, 0, 0, 0
            if UseOr:
                for size in tl.static_range(  # pyright: ignore[reportGeneralTypeIssues]
                    1, min(3, N - 2) + 1
                ):
                    for base in range(0, N * N, Tile):
                        pair = base + lane
                        start, after = pair // N, pair % N
                        offset = (after - start + N) % N
                        valid = (pair < N * N) & (offset >= size) & (offset != N - 1)
                        p = tl.load(row + (start + N - 1) % N, valid, 0)
                        a = tl.load(row + start, valid, 0)
                        b = tl.load(row + (start + size - 1) % N, valid, 0)
                        q = tl.load(row + (start + size) % N, valid, 0)
                        u = tl.load(row + after, valid, 0)
                        v = tl.load(row + (after + 1) % N, valid, 0)
                        removed = (
                            tl.load(Distance + p * N + q, valid, 0)
                            - tl.load(Distance + p * N + a, valid, 0)
                            - tl.load(Distance + b * N + q, valid, 0)
                            - tl.load(Distance + u * N + v, valid, 0)
                        )
                        direct = (
                            removed
                            + tl.load(Distance + u * N + a, valid, 0)
                            + tl.load(Distance + b * N + v, valid, 0)
                        )
                        reverse = tl.full((Tile,), False, tl.int1)
                        if size > 1:
                            flipped = (
                                removed
                                + tl.load(Distance + u * N + b, valid, 0)
                                + tl.load(Distance + a * N + v, valid, 0)
                            )
                            reverse = flipped < direct
                            direct = tl.minimum(direct, flipped)
                        value = tl.min(tl.where(valid, direct, sentinel), 0)
                        key = tl.min(tl.where(valid & (direct == value), pair, N * N), 0)
                        direction = tl.min(tl.where(pair == key, reverse.to(tl.int32), 2), 0)
                        # Sizes and tiles are ascending: ties retain (size,start,after,reverse).
                        if value < best_delta:
                            best_delta = value
                            best_size, best_start, best_after = size, key // N, key % N
                            best_reverse = direction
            if best_delta < 0:
                city = tl.load(row + position, position < N, 0)
                offset = (position - best_start + N) % N
                in_chain = offset < best_size
                tail = tl.minimum(best_size, N - best_start)
                wrap = tl.maximum(best_start + best_size - N, 0)
                removed_before = tl.minimum(
                    tl.maximum(position - best_start, 0), tail
                ) + tl.minimum(position, wrap)
                after_removed = tl.minimum(
                    tl.maximum(best_after - best_start, 0), tail
                ) + tl.minimum(best_after, wrap)
                insertion = best_after - after_removed + 1
                rank = position - removed_before
                destination = tl.where(
                    in_chain,
                    insertion + tl.where(best_reverse != 0, best_size - 1 - offset, offset),
                    rank + tl.where(rank >= insertion, best_size, 0),
                )
                anchor = tl.min(tl.where((position < N) & (city == 0), destination, N), 0)
                destination = (destination - anchor + N) % N
                tl.debug_barrier()
                tl.store(row + destination, city, position < N)
                tl.debug_barrier()
                moves += 1
            else:
                active = False
    tl.store(row + N, moves)
    tl.store(row + N + 1, tl.where(active, 0, 1))


class CudaLocalSearch:
    """Resident distance matrix and reusable batched tour buffers, with complete costs."""

    def __init__(
        self,
        instance: TSPInstance,
        *,
        method: str,
        max_moves: int,
        capacity: int = 8,
        tile: int = 256,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA local search requires an available CUDA device")
        if instance.dimension < 4 or instance.dimension > 32768:
            raise ValueError("CUDA local search requires 4..32768 cities")
        if method not in {"two_opt", "two_opt_or_opt"} or max_moves <= 0 or capacity <= 0:
            raise ValueError("Invalid CUDA local-search settings")
        if tile not in {128, 256, 512, 1024}:
            raise ValueError("CUDA candidate tile must be 128, 256, 512 or 1024")
        if int(instance.distance.max()) > np.iinfo(np.int64).max // 6:
            raise ValueError("CUDA integer local-search deltas could overflow")
        self.instance, self.method, self.max_moves = instance, method, max_moves
        self.capacity, self.tile = capacity, tile
        self.calls = self.tours_processed = self.accepted_moves = self.converged_tours = 0
        self.batch_sizes: Counter = Counter()
        self.phase_seconds = dict.fromkeys(
            ("validation", "upload", "kernel", "download", "certificate"), 0.0
        )
        started = perf_counter()
        self.distance = torch.tensor(instance.distance.copy(), dtype=torch.int64, device="cuda")
        self.state = torch.empty(
            (capacity, instance.dimension + 2), dtype=torch.int32, device="cuda"
        )
        self._start, self._end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        torch.cuda.synchronize()
        self.upload_setup_seconds = perf_counter() - started
        compiled = perf_counter()
        _local_descent.warmup(self.distance, self.state, max_moves, **self._parameters(), grid=(1,))
        self.compile_seconds = perf_counter() - compiled
        self.setup_seconds = perf_counter() - started

    def _parameters(self) -> dict:
        return {
            "N": self.instance.dimension,
            "UseOr": self.method == "two_opt_or_opt",
            "Tile": self.tile,
            "CityBlock": triton.next_power_of_2(self.instance.dimension),
            "num_warps": 4,
        }

    def solve(self, tours: Sequence[Tour], remaining: Callable[[], float]) -> tuple[Tour, ...]:
        if not tours or len(tours) > self.capacity:
            raise ValueError("Local-search batch must fit the configured capacity")
        started = perf_counter()
        for tour in tours:
            if not certify_tour(self.instance, tour).is_permutation:
                raise ValueError("GPU local search requires complete tours")
        self.phase_seconds["validation"] += perf_counter() - started
        if remaining() <= 0:
            return tuple(canonical_tour(self.instance, tour) for tour in tours)
        started = perf_counter()
        packet = np.zeros((len(tours), self.instance.dimension + 2), dtype=np.int32)
        packet[:, : self.instance.dimension] = tours
        self.state[: len(tours)].copy_(torch.from_numpy(packet))
        self.phase_seconds["upload"] += perf_counter() - started
        self._start.record()
        _local_descent[(len(tours),)](
            self.distance, self.state, self.max_moves, **self._parameters()
        )
        self._end.record()
        self._end.synchronize()
        self.phase_seconds["kernel"] += self._start.elapsed_time(self._end) / 1000
        started = perf_counter()
        packet = self.state[: len(tours)].cpu().numpy()
        self.phase_seconds["download"] += perf_counter() - started
        started = perf_counter()
        try:
            result = tuple(
                canonical_tour(self.instance, row[: self.instance.dimension]) for row in packet
            )
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"CUDA certificate failed; raw output packet: {packet.tolist()}"
            ) from error
        self.phase_seconds["certificate"] += perf_counter() - started
        self.calls += 1
        self.tours_processed += len(tours)
        self.batch_sizes[len(tours)] += 1
        self.accepted_moves += int(packet[:, -2].sum())
        self.converged_tours += int(packet[:, -1].sum())
        return result

    def metadata(self) -> dict:
        return {
            "backend": "cuda_triton_local_descent",
            "method": self.method,
            "tile": self.tile,
            "num_warps": 4,
            "capacity": self.capacity,
            "max_moves": self.max_moves,
            "calls": self.calls,
            "tours": self.tours_processed,
            "batch_sizes": dict(self.batch_sizes),
            "accepted_moves": self.accepted_moves,
            "converged_tours": self.converged_tours,
            "setup_seconds": self.setup_seconds,
            "distance_and_buffer_setup_seconds": self.upload_setup_seconds,
            "compile_seconds": self.compile_seconds,
            "phase_seconds": dict(self.phase_seconds),
            "deadline_boundary": "before each bounded batch; no mid-kernel time cancellation",
            "cpu_stages": [
                "AB-cycle construction",
                "selection",
                "subtour repair",
                "elite pool",
                "certificate",
                "journal",
            ],
            "integer_dtype": "int64 costs, int32 city IDs",
            "kernel_randomness": None,
            "kernel_timing": "CUDA event interval; first launch can include module loading",
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
        }
