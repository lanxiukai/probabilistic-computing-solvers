"""Colored binary heat-bath sampling of the sparse TSP assignment relaxation.

Each color has distinct source and destination endpoints. One persistent CUDA
block owns each replica, with block barriers between colors. Counts are exact
int32; local cost fields and probabilities use float32. No physical device is
emulated at the transistor or timing level.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from .edge_model import EdgeAssignment


@triton.jit
def _advance(
    State,
    Outgoing,
    Incoming,
    Flips,
    Source,
    Destination,
    Cost,
    Colors,
    Uniform,
    Sweeps,
    Step,
    Seed,
    BetaStart,
    BetaStop,
    Penalty,
    N: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    Block: tl.constexpr,
    FixedUniform: tl.constexpr,
):
    replica = tl.program_id(0)
    lane = tl.arange(0, Block)
    for step in range(Sweeps):
        fraction = step / tl.maximum(Sweeps - 1, 1)
        beta = BetaStart * tl.exp(tl.log(BetaStop / BetaStart) * fraction)
        # A separate Philox seed per sweep; counter offsets are replica/arc pairs.
        random_seed = Seed.to(tl.uint32) + (Step + step).to(tl.uint32) * 0x9E3779B9
        for color in range(C):
            edge = tl.load(Colors + color * Block + lane)
            valid = edge >= 0
            left = tl.load(Source + edge, valid, 0)
            right = tl.load(Destination + edge, valid, 0)
            old = tl.load(State + replica * M + edge, valid, 0).to(tl.int32)
            outgoing = tl.load(Outgoing + replica * N + left, valid, 0)
            incoming = tl.load(Incoming + replica * N + right, valid, 0)
            cost = tl.load(Cost + edge, valid, 0)
            delta = cost + 2 * Penalty * (outgoing + incoming - 2 * old - 1)
            probability = 1 / (1 + tl.exp(beta * delta))
            if FixedUniform:
                uniform = tl.load(Uniform + replica * M + edge, valid, 0)
            else:
                uniform = tl.rand(random_seed, (replica * M + edge).to(tl.uint32))
            new = (uniform < probability).to(tl.int32)
            changed = new - old
            flips = tl.load(Flips + replica * M + edge, valid, 0)
            tl.store(State + replica * M + edge, new, valid)
            tl.store(Outgoing + replica * N + left, outgoing + changed, valid)
            tl.store(Incoming + replica * N + right, incoming + changed, valid)
            tl.store(Flips + replica * M + edge, flips + (changed != 0), valid)
            tl.debug_barrier()


class CudaAssignment:
    """Own independent replica states and exact cached constraint counts."""

    def __init__(self, model: EdgeAssignment, states: np.ndarray, *, seed: int):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the assignment p-bit path")
        states = np.atleast_2d(states)
        outgoing, incoming = model.counts(states)
        self.model, self.seed, self.step = model, seed, 0
        self.replicas = len(states)
        self.block = triton.next_power_of_2(max(map(len, model.colors)))
        colors = np.full((len(model.colors), self.block), -1, dtype=np.int32)
        for index, group in enumerate(model.colors):
            colors[index, : len(group)] = group

        def device(array):
            return torch.as_tensor(array.copy(), device="cuda")

        self.states = device(states.astype(np.int8))
        self.outgoing, self.incoming = device(outgoing), device(incoming)
        self.flips = torch.zeros(states.shape, dtype=torch.int32, device="cuda")
        self.source, self.destination = device(model.source), device(model.destination)
        self.costs = device((model.cost / model.scale).astype(np.float32))
        self.colors = device(colors)

    def advance(
        self,
        sweeps: int,
        beta_start: float,
        beta_stop: float,
        *,
        uniform=None,
        color: int | None = None,
    ):
        if sweeps < 1 or beta_start <= 0 or beta_stop <= 0:
            raise ValueError("Positive sweeps and inverse temperatures are required")
        colors = self.colors if color is None else self.colors[color : color + 1]
        if uniform is not None:
            uniform = torch.as_tensor(uniform, dtype=torch.float32, device="cuda")
            if tuple(uniform.shape) != tuple(self.states.shape) or sweeps != 1:
                raise ValueError("Fixed uniforms require one sweep and one value per state bit")
        _advance[(self.replicas,)](
            self.states,
            self.outgoing,
            self.incoming,
            self.flips,
            self.source,
            self.destination,
            self.costs,
            colors,
            self.states if uniform is None else uniform,
            sweeps,
            self.step,
            self.seed,
            beta_start,
            beta_stop,
            self.model.penalty,
            N=self.model.cities,
            M=self.model.bits,
            C=len(colors),
            Block=self.block,
            FixedUniform=uniform is not None,
            num_warps=8 if self.block >= 1024 else 4,  # pyright: ignore[reportCallIssue]
        )
        self.step += sweeps
        torch.cuda.synchronize()

    def snapshot(self) -> dict[str, np.ndarray]:
        result = {
            "states": self.states.cpu().numpy(),
            "outgoing": self.outgoing.cpu().numpy(),
            "incoming": self.incoming.cpu().numpy(),
            "flips": self.flips.cpu().numpy(),
        }
        expected_out, expected_in = self.model.counts(result["states"])
        if not np.array_equal(result["outgoing"], expected_out) or not np.array_equal(
            result["incoming"], expected_in
        ):
            raise AssertionError("CUDA cached counts differ from independent reconstruction")
        return result
