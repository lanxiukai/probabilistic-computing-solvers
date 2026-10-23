"""CUDA-graph execution of the unchanged deterministic circuit MH calculation."""

from __future__ import annotations

from dataclasses import fields

import torch

from .circuit_proposal import CircuitNoise


class CircuitGraph:
    """Keep RNG outside capture and copy outputs to preserve their original lifetime."""

    def __init__(self, proposal, states, expanded, beta_indices, noise):
        self.states = torch.empty_like(states, memory_format=torch.contiguous_format)
        self.expanded = torch.empty_like(expanded, memory_format=torch.contiguous_format)
        self.beta_indices = torch.empty_like(beta_indices, memory_format=torch.contiguous_format)
        self.noise = CircuitNoise(
            **{
                field.name: None
                if getattr(noise, field.name) is None
                else torch.empty_like(
                    getattr(noise, field.name), memory_format=torch.contiguous_format
                )
                for field in fields(noise)
            }
        )
        from .circuit_cache import ValidatedStateCache

        self.cache = (
            ValidatedStateCache(self.states, self.beta_indices)
            if proposal.old_state_cache == "validated_v1"
            else None
        )
        self.copy_inputs(states, expanded, beta_indices, noise)
        stream = torch.cuda.Stream(device=states.device)
        stream.wait_stream(torch.cuda.current_stream(states.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                proposal._step_from_noise(
                    self.states, self.expanded, self.beta_indices, self.noise, old_cache=self.cache
                )
        torch.cuda.current_stream(states.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = proposal._step_from_noise(
                self.states, self.expanded, self.beta_indices, self.noise, old_cache=self.cache
            )
        torch.cuda.current_stream(states.device).wait_stream(stream)

        if self.cache is not None:
            self.cache.reset()

    def matches(self, states, noise):
        return (
            self.states.shape == states.shape
            and self.states.dtype == states.dtype
            and ((self.noise.root_masks is None) == (noise.root_masks is None))
            and ((self.noise.root_uniforms is None) == (noise.root_uniforms is None))
            and ((self.noise.modular_uniforms is None) == (noise.modular_uniforms is None))
        )

    def copy_inputs(self, states, expanded, beta_indices, noise):
        pairs = [
            (self.states, states),
            (self.expanded, expanded),
            (self.beta_indices, beta_indices),
            *[(getattr(self.noise, f.name), getattr(noise, f.name)) for f in fields(noise)],
        ]
        for target, source in pairs:
            if target is None and source is None:
                continue
            if (
                target is None
                or source is None
                or target.shape != source.shape
                or target.dtype != source.dtype
                or target.device != source.device
            ):
                raise ValueError("Graph inputs must preserve shape, dtype and device")
            target.copy_(source)

    def run(self, states, expanded, beta_indices, noise):
        self.copy_inputs(states, expanded, beta_indices, noise)
        self.graph.replay()
        if self.cache is not None:
            self.cache.calls += 1
        # A caller may retain a result across subsequent replays. Never expose the
        # mutable capture buffers as ordinary proposal outputs.
        return type(self.output)(
            **{f.name: getattr(self.output, f.name).clone() for f in fields(self.output)}
        )
