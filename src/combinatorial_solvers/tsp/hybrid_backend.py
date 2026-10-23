"""Local-search execution adapters shared by the CPU and CUDA hybrid schedules."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from time import perf_counter

from .hybrid_local import first_descent
from .hybrid_moves import Tour
from .instance import TSPInstance


class LocalSearchBackend:
    def __init__(
        self,
        instance: TSPInstance,
        *,
        device: str,
        method: str,
        max_moves: int,
        capacity: int,
        tile: int = 256,
    ):
        self.instance, self.device, self.method = instance, device, method
        self.max_moves, self.capacity = max_moves, capacity
        self.calls = self.tours_processed = 0
        self.batch_sizes: Counter = Counter()
        self.seconds = 0.0
        self.cuda = None
        if device == "cuda":
            from .hybrid_cuda import CudaLocalSearch

            self.cuda = CudaLocalSearch(
                instance, method=method, max_moves=max_moves, capacity=capacity, tile=tile
            )
        elif device != "cpu":
            raise ValueError("Local-search device must be cpu or cuda")

    def solve(self, tours: Sequence[Tour], remaining: Callable[[], float]) -> tuple[Tour, ...]:
        if not tours or len(tours) > self.capacity:
            raise ValueError("Local-search batch must fit the configured capacity")
        started = perf_counter()
        if self.cuda is not None:
            result = self.cuda.solve(tours, remaining)
        else:
            result = tuple(
                first_descent(
                    self.instance,
                    tour,
                    self.max_moves,
                    remaining,
                    or_opt=self.method == "two_opt_or_opt",
                )
                for tour in tours
            )
        self.seconds += perf_counter() - started
        self.calls += 1
        self.tours_processed += len(tours)
        self.batch_sizes[len(tours)] += 1
        return result

    def metadata(self) -> dict:
        detail = (
            self.cuda.metadata()
            if self.cuda is not None
            else {"backend": "cpu_numpy_local_descent", "setup_seconds": 0.0}
        )
        return {
            **detail,
            "device": self.device,
            "method": self.method,
            "calls": self.calls,
            "tours": self.tours_processed,
            "batch_sizes": dict(self.batch_sizes),
            "solve_wall_seconds": self.seconds,
        }
