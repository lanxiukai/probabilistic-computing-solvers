"""Exact Metropolis correction for symmetric involutive macro proposals."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as functional

from .rng import RngAuditRecorder

EnergyFunction = Callable[[torch.Tensor], torch.Tensor]


class InvolutiveMovePlan(Protocol):
    """Structural interface shared by table-backed and on-demand move plans."""

    @property
    def state_size(self) -> int: ...

    @property
    def move_count(self) -> int: ...

    @property
    def storage_bytes(self) -> int: ...

    @property
    def support_name(self) -> str: ...

    @property
    def plan_kind(self) -> str: ...

    @property
    def move_names(self) -> tuple[str, ...]: ...

    def to(self, device: str | torch.device) -> InvolutiveMovePlan: ...

    def propose(
        self,
        states: torch.Tensor,
        selectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class InvolutiveCodewordPlan:
    """Reference codewords and state-independent XOR moves on their integer labels."""

    reference_states: torch.Tensor
    logical_nodes: torch.Tensor
    logical_weights: torch.Tensor
    fixed_code: int
    move_masks: torch.Tensor
    move_names: tuple[str, ...]
    support_name: str = "full_codeword"

    def __post_init__(self) -> None:
        if self.reference_states.ndim != 2 or self.reference_states.dtype != torch.int8:
            raise TypeError("reference_states must be a two-dimensional int8 tensor")
        if torch.any((self.reference_states != -1) & (self.reference_states != 1)):
            raise ValueError("reference_states must contain only bipolar spins")
        if self.logical_nodes.ndim != 1 or self.logical_nodes.dtype != torch.long:
            raise TypeError("logical_nodes must be a one-dimensional torch.long tensor")
        if self.logical_weights.shape != self.logical_nodes.shape:
            raise ValueError("logical_weights must match logical_nodes")
        if self.logical_weights.dtype != torch.long:
            raise TypeError("logical_weights must use torch.long")
        if self.move_masks.ndim != 1 or self.move_masks.dtype != torch.long:
            raise TypeError("move_masks must be a one-dimensional torch.long tensor")
        if not self.move_names or len(self.move_names) != self.move_masks.numel():
            raise ValueError("move_names must identify every move mask")
        if not self.support_name:
            raise ValueError("support_name must be nonempty")
        if torch.any(self.move_masks <= 0):
            raise ValueError("Every move mask must be positive")
        if len(set(int(value) for value in self.move_masks.tolist())) != self.move_masks.numel():
            raise ValueError("Move masks must be unique")
        if self.logical_nodes.numel() and (
            torch.any(self.logical_nodes < 0)
            or torch.any(self.logical_nodes >= self.reference_states.shape[1])
        ):
            raise IndexError("logical_nodes contain an out-of-range state index")
        if self.fixed_code < 0 or self.fixed_code >= self.reference_states.shape[0]:
            raise ValueError("fixed_code is outside the reference table")
        if self.reference_states.shape[0] & (self.reference_states.shape[0] - 1):
            raise ValueError("The reference table length must be a power of two")
        maximum_code = self.reference_states.shape[0] - 1
        if any(int(mask) > maximum_code for mask in self.move_masks.tolist()):
            raise ValueError("A move mask exceeds the reference code width")
        devices = {
            self.reference_states.device,
            self.logical_nodes.device,
            self.logical_weights.device,
            self.move_masks.device,
        }
        if len(devices) != 1:
            raise ValueError("All codeword plan tensors must share one device")

    @property
    def state_size(self) -> int:
        return int(self.reference_states.shape[1])

    @property
    def plan_kind(self) -> str:
        return "reference_table"

    @property
    def code_count(self) -> int:
        return int(self.reference_states.shape[0])

    @property
    def move_count(self) -> int:
        return int(self.move_masks.numel())

    @property
    def device(self) -> torch.device:
        return self.reference_states.device

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.reference_states,
            self.logical_nodes,
            self.logical_weights,
            self.move_masks,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def to(self, device: str | torch.device) -> InvolutiveCodewordPlan:
        selected = torch.device(device)
        return InvolutiveCodewordPlan(
            reference_states=self.reference_states.to(selected),
            logical_nodes=self.logical_nodes.to(selected),
            logical_weights=self.logical_weights.to(selected),
            fixed_code=self.fixed_code,
            move_masks=self.move_masks.to(selected),
            move_names=self.move_names,
            support_name=self.support_name,
        )

    def decode_codes(self, states: torch.Tensor) -> torch.Tensor:
        """Decode the reference-table label from designated logical root spins."""

        if states.ndim != 2 or states.shape[1] != self.state_size:
            raise ValueError("states do not match the codeword plan")
        if states.device != self.device:
            raise ValueError("states and codeword plan must share one device")
        bits = states.index_select(1, self.logical_nodes).gt(0).to(torch.long)
        return (bits * self.logical_weights).sum(dim=1) + int(self.fixed_code)

    def propose(
        self,
        states: torch.Tensor,
        selectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one selected self-inverse reference transport to every replica."""

        if selectors.shape != (states.shape[0],) or selectors.dtype != torch.long:
            raise TypeError("selectors must be a torch.long vector with one entry per replica")
        if selectors.device != self.device:
            raise ValueError("selectors and codeword plan must share one device")
        if torch.any(selectors < 0) or torch.any(selectors >= self.move_count):
            raise IndexError("selectors contain an out-of-range move")
        current_codes = self.decode_codes(states)
        proposed_codes = torch.bitwise_xor(
            current_codes,
            self.move_masks.index_select(0, selectors),
        )
        current_reference = self.reference_states.index_select(0, current_codes)
        proposed_reference = self.reference_states.index_select(0, proposed_codes)
        return states * current_reference * proposed_reference, proposed_codes


@dataclass(frozen=True)
class InvolutiveMacroUpdate:
    """One batched symmetric proposal and its exact target-Hamiltonian correction."""

    states: torch.Tensor
    selectors: torch.Tensor
    proposed_codes: torch.Tensor
    proposed_energy_delta: torch.Tensor
    accepted_energy_delta: torch.Tensor
    accepted: torch.Tensor


@dataclass(frozen=True)
class LocallyBalancedMacroUpdate:
    """One exact locally balanced proposal over an involutive finite neighborhood."""

    states: torch.Tensor
    selectors: torch.Tensor
    proposed_codes: torch.Tensor
    proposed_energy_delta: torch.Tensor
    accepted_energy_delta: torch.Tensor
    accepted: torch.Tensor
    forward_log_normalizer: torch.Tensor
    reverse_log_normalizer: torch.Tensor
    log_acceptance_ratio: torch.Tensor
    uniform_mixture: float
    candidate_evaluations: int


def enumerate_involutive_candidates(
    states: torch.Tensor,
    plan: InvolutiveMovePlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate every labelled neighbor without assuming a table-backed plan."""

    if states.ndim != 2 or states.shape[1] != plan.state_size:
        raise ValueError("states do not match the involutive move plan")
    replicas = states.shape[0]
    selectors = torch.arange(plan.move_count, dtype=torch.long, device=states.device)
    selectors = selectors.repeat(replicas)
    expanded = (
        states[:, None, :]
        .expand(-1, plan.move_count, -1)
        .reshape(
            replicas * plan.move_count,
            plan.state_size,
        )
    )
    proposed, proposed_codes = plan.propose(expanded, selectors)
    return (
        proposed.reshape(replicas, plan.move_count, plan.state_size),
        proposed_codes.reshape(replicas, plan.move_count),
    )


def _selected_beta(
    beta: float | torch.Tensor,
    *,
    replicas: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    selected = torch.as_tensor(beta, dtype=dtype, device=device)
    if selected.ndim not in {0, 1} or (selected.ndim == 1 and selected.shape != (replicas,)):
        raise ValueError("beta must be scalar or contain one value per replica")
    return selected.expand(replicas)


def _locally_balanced_log_weights(
    energy_delta: torch.Tensor,
    beta: torch.Tensor,
    rule: str,
) -> torch.Tensor:
    log_target_ratio = -beta[:, None] * energy_delta
    if rule == "barker":
        return functional.logsigmoid(log_target_ratio)
    if rule == "sqrt":
        return 0.5 * log_target_ratio
    raise ValueError("rule must be 'barker' or 'sqrt'")


def _sample_categorical_logits(
    logits: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw one categorical outcome per row using one auditable uniform value."""

    probabilities = torch.softmax(logits, dim=1)
    uniforms = torch.rand(
        (logits.shape[0],),
        dtype=logits.dtype,
        device=logits.device,
        generator=generator,
    )
    cumulative = probabilities.cumsum(dim=1)
    return torch.count_nonzero(uniforms[:, None] > cumulative, dim=1).clamp_max(logits.shape[1] - 1)


def locally_balanced_involutive_metropolis_step(
    states: torch.Tensor,
    current_energies: torch.Tensor,
    beta: float | torch.Tensor,
    plan: InvolutiveMovePlan,
    energy: EnergyFunction,
    *,
    rule: str = "barker",
    uniform_mixture: float = 0.0,
    generator: torch.Generator,
    rng_recorder: RngAuditRecorder | None = None,
) -> LocallyBalancedMacroUpdate:
    """Use exact target-energy differences to inform a target-correct macro move.

    The labelled neighborhood is symmetric because every move is self-inverse. For
    each state, all neighbors are scored with a locally balancing function. The reverse
    neighborhood is then evaluated explicitly and the complete Metropolis-Hastings
    ratio is applied; no approximation to the target Hamiltonian enters the correction.
    """

    if states.ndim != 2 or states.shape[1] != plan.state_size:
        raise ValueError("states do not match the involutive move plan")
    if current_energies.shape != (states.shape[0],):
        raise ValueError("current_energies must contain one value per replica")
    if current_energies.device != states.device:
        raise ValueError("states and energies must share one device")
    if torch.device(generator.device) != states.device:
        raise ValueError("generator and macro states must share one device")
    if rule not in {"barker", "sqrt"}:
        raise ValueError("rule must be 'barker' or 'sqrt'")
    if not math.isfinite(uniform_mixture) or not 0.0 <= uniform_mixture < 1.0:
        raise ValueError("uniform_mixture must be finite and in [0, 1)")

    replicas = states.shape[0]
    selected_beta = _selected_beta(
        beta,
        replicas=replicas,
        dtype=current_energies.dtype,
        device=states.device,
    )
    candidates, candidate_codes = enumerate_involutive_candidates(states, plan)
    candidate_energies = energy(candidates.reshape(-1, plan.state_size)).reshape(
        replicas,
        plan.move_count,
    )
    candidate_delta = candidate_energies - current_energies[:, None]
    forward_log_weights = _locally_balanced_log_weights(
        candidate_delta,
        selected_beta,
        rule,
    )
    forward_log_normalizer = torch.logsumexp(forward_log_weights, dim=1)
    forward_log_probabilities = forward_log_weights - forward_log_normalizer[:, None]
    if uniform_mixture > 0:
        uniform_log_probability = math.log(uniform_mixture / plan.move_count)
        forward_log_probabilities = torch.logaddexp(
            forward_log_probabilities + math.log1p(-uniform_mixture),
            torch.full_like(forward_log_probabilities, uniform_log_probability),
        )
    selectors = _sample_categorical_logits(forward_log_probabilities, generator=generator)
    if rng_recorder is not None:
        rng_recorder.record("locally_balanced_macro_selector_uniform", replicas)
    row = torch.arange(replicas, dtype=torch.long, device=states.device)
    proposed = candidates[row, selectors]
    proposed_codes = candidate_codes[row, selectors]
    proposed_energies = candidate_energies[row, selectors]
    proposed_delta = candidate_delta[row, selectors]
    selected_forward_log_probability = forward_log_probabilities[row, selectors]
    del candidates

    reverse_candidates, _ = enumerate_involutive_candidates(proposed, plan)
    reverse_energies = energy(reverse_candidates.reshape(-1, plan.state_size)).reshape(
        replicas,
        plan.move_count,
    )
    reverse_delta = reverse_energies - proposed_energies[:, None]
    reverse_log_weights = _locally_balanced_log_weights(
        reverse_delta,
        selected_beta,
        rule,
    )
    reverse_log_normalizer = torch.logsumexp(reverse_log_weights, dim=1)
    reverse_log_probabilities = reverse_log_weights - reverse_log_normalizer[:, None]
    if uniform_mixture > 0:
        uniform_log_probability = math.log(uniform_mixture / plan.move_count)
        reverse_log_probabilities = torch.logaddexp(
            reverse_log_probabilities + math.log1p(-uniform_mixture),
            torch.full_like(reverse_log_probabilities, uniform_log_probability),
        )
    selected_reverse_log_probability = reverse_log_probabilities[row, selectors]
    log_acceptance = (
        -selected_beta * proposed_delta
        + selected_reverse_log_probability
        - selected_forward_log_probability
    )
    acceptance_uniform = torch.rand(
        (replicas,),
        dtype=proposed_delta.dtype,
        device=states.device,
        generator=generator,
    )
    if rng_recorder is not None:
        rng_recorder.record("locally_balanced_macro_acceptance_uniform", replicas)
    accepted = torch.log(acceptance_uniform) < torch.minimum(
        torch.zeros_like(log_acceptance),
        log_acceptance,
    )
    accepted_delta = torch.where(accepted, proposed_delta, 0.0)
    return LocallyBalancedMacroUpdate(
        states=torch.where(accepted[:, None], proposed, states),
        selectors=selectors,
        proposed_codes=proposed_codes,
        proposed_energy_delta=proposed_delta,
        accepted_energy_delta=accepted_delta,
        accepted=accepted,
        forward_log_normalizer=forward_log_normalizer,
        reverse_log_normalizer=reverse_log_normalizer,
        log_acceptance_ratio=log_acceptance,
        uniform_mixture=float(uniform_mixture),
        candidate_evaluations=2 * replicas * plan.move_count,
    )


def involutive_codeword_metropolis_step(
    states: torch.Tensor,
    current_energies: torch.Tensor,
    beta: float | torch.Tensor,
    plan: InvolutiveCodewordPlan,
    energy: EnergyFunction,
    *,
    generator: torch.Generator,
    rng_recorder: RngAuditRecorder | None = None,
) -> InvolutiveMacroUpdate:
    """Perform a uniform symmetric codeword move with exact Metropolis acceptance."""

    if states.ndim != 2 or states.shape[1] != plan.state_size:
        raise ValueError("states do not match the codeword plan")
    if current_energies.shape != (states.shape[0],):
        raise ValueError("current_energies must contain one value per replica")
    if states.device != plan.device or current_energies.device != states.device:
        raise ValueError("states, energies, and codeword plan must share one device")
    if torch.device(generator.device) != states.device:
        raise ValueError("generator and macro states must share one device")

    replicas = states.shape[0]
    selectors = torch.randint(
        plan.move_count,
        (replicas,),
        dtype=torch.long,
        device=states.device,
        generator=generator,
    )
    if rng_recorder is not None:
        rng_recorder.record("codeword_transport_selector", replicas)
    proposed, proposed_codes = plan.propose(states, selectors)
    proposed_energies = energy(proposed)
    proposed_delta = proposed_energies - current_energies
    selected_beta = torch.as_tensor(
        beta,
        dtype=proposed_delta.dtype,
        device=states.device,
    )
    if selected_beta.ndim not in {0, 1} or (
        selected_beta.ndim == 1 and selected_beta.shape != (replicas,)
    ):
        raise ValueError("beta must be scalar or contain one value per replica")
    acceptance_uniform = torch.rand(
        (replicas,),
        dtype=proposed_delta.dtype,
        device=states.device,
        generator=generator,
    )
    if rng_recorder is not None:
        rng_recorder.record("codeword_transport_acceptance_uniform", replicas)
    log_acceptance = -selected_beta * proposed_delta
    accepted = torch.log(acceptance_uniform) < torch.minimum(
        torch.zeros_like(log_acceptance),
        log_acceptance,
    )
    accepted_delta = torch.where(accepted, proposed_delta, 0.0)
    return InvolutiveMacroUpdate(
        states=torch.where(accepted[:, None], proposed, states),
        selectors=selectors,
        proposed_codes=proposed_codes,
        proposed_energy_delta=proposed_delta,
        accepted_energy_delta=accepted_delta,
        accepted=accepted,
    )
