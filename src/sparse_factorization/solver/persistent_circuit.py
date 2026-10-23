"""Circuit-directed MH events for the persistent sparse physical-state engine."""

from __future__ import annotations

import numpy as np
import torch

from sparse_factorization.core.rng import derive_stream_seed

from .circuit_modular import UNIT_GROUP_MODES
from .circuit_proposal import CircuitDirectedProposal
from .circuit_quotient import QUOTIENT_MODES
from .circuit_range import PUBLIC_RANGE_MODES
from .circuit_root_filter import FERMAT_MODES, FILTER_MODES
from .persistent_gibbs import PersistentSparseGibbs
from .persistent_tempering import ReplicaExchange

CIRCUIT_STREAM = "p712_circuit_cuda_philox_v1"
CIRCUIT_STREAM_V2 = "p712_circuit_cuda_philox_v2"
CIRCUIT_STREAM_V3 = "p727_circuit_cuda_philox_v3"
CIRCUIT_STREAM_V4 = "p728_circuit_cuda_philox_v4"
CIRCUIT_STREAM_V5 = "p730_circuit_cuda_philox_v5"
CIRCUIT_STREAM_V6 = "p736_filtered_circuit_cuda_philox_v6"
CIRCUIT_STREAM_V7 = "p745_fermat_circuit_cuda_philox_v7"
CIRCUIT_STREAM_V8 = "p747_crt_fermat_circuit_cuda_philox_v8"
CIRCUIT_STREAM_V9 = "p748_crt128_circuit_cuda_philox_v9"
CIRCUIT_STREAM_V10 = "p749_deferred_radius_circuit_cuda_philox_v10"


class PersistentCircuit:
    """Observe completed MH transitions, never intermediate DAG candidate states."""

    warmup_steps = 3

    def __init__(
        self,
        engine: PersistentSparseGibbs,
        seed: int,
        betas: torch.Tensor,
        *,
        logit_clip: float = 12.0,
        proposal_beta_scale: float = 1.0,
        root_policy: str = "lazy_single_flip_v1",
        exchange_every: int = 0,
        exchange_seed: int | None = None,
        backend: str = "torch",
        history_backend: str = "torch_host_v1",
        cdf_layout: str = "beta_group_context",
        old_state_cache: str = "disabled",
        cdf_search: str = "scan32_v1",
        lookahead_strength: float = 0.0,
        regrowth_mode: str = "full_v1",
        root_guidance: str = "disabled",
        modular_root_mode: str = "disabled",
        guidance_execution: str = "legacy_v1",
    ) -> None:
        if history_backend not in {"torch_host_v1", "triton_device_v1"}:
            raise ValueError("Unknown circuit history backend")
        self.history_backend = history_backend
        self.engine = engine
        self.root_policy = root_policy
        self.exchange_every = int(exchange_every)
        self.exchange_seed = exchange_seed
        if self.exchange_every < 0 or (self.exchange_every and exchange_seed is None):
            raise ValueError("Tempering requires a nonnegative interval and a declared seed")
        self.warmup_steps = max(3, self.exchange_every)
        values, indices = np.unique(betas.cpu().numpy(), return_inverse=True)
        self.proposal = CircuitDirectedProposal(
            engine.problem,
            values,
            device="cuda",
            logit_clip=logit_clip,
            proposal_beta_scale=proposal_beta_scale,
            root_policy=root_policy,
            backend=backend,
            cdf_layout=cdf_layout,
            old_state_cache=old_state_cache,
            cdf_search=cdf_search,
            lookahead_strength=lookahead_strength,
            regrowth_mode=regrowth_mode,
            root_guidance=root_guidance,
            modular_root_mode=modular_root_mode,
            guidance_execution=guidance_execution,
        )
        self.beta_indices = torch.from_numpy(indices).long().cuda()
        self.fixed_betas = betas.clone()
        self.stream_name = (
            CIRCUIT_STREAM if root_policy == "lazy_single_flip_v1" else CIRCUIT_STREAM_V2
        )
        if root_guidance != "disabled":
            self.stream_name = CIRCUIT_STREAM_V3
        if modular_root_mode != "disabled":
            self.stream_name = CIRCUIT_STREAM_V4
        if modular_root_mode in PUBLIC_RANGE_MODES | QUOTIENT_MODES:
            self.stream_name = CIRCUIT_STREAM_V5
        if modular_root_mode in FILTER_MODES:
            self.stream_name = CIRCUIT_STREAM_V6
        if modular_root_mode in FERMAT_MODES:
            self.stream_name = CIRCUIT_STREAM_V7
        if modular_root_mode == "fermat_crt_filter256_v1":
            self.stream_name = CIRCUIT_STREAM_V8
        if modular_root_mode == "fermat_crt128_filter256_v1":
            self.stream_name = CIRCUIT_STREAM_V9
        if modular_root_mode == "fermat_deferred262144_v1":
            self.stream_name = CIRCUIT_STREAM_V10
        self.rng_seed = derive_stream_seed(seed, self.stream_name)
        self.generator = torch.Generator(device="cuda").manual_seed(self.rng_seed)
        self.reset_statistics()

    def reset_statistics(self) -> None:
        graph = self.proposal._graph_runtime
        if graph is not None and graph.cache is not None:
            graph.cache.reset()
        self.events = self.uniform_draws = 0
        self.first_hit_sweeps_host = None
        self.history_payload = (
            torch.empty((self.engine.replicas, 7), dtype=torch.int64, device="cuda")
            if self.history_backend == "triton_device_v1"
            else None
        )
        self.accepted_proposals = self.accepted_root_changes = self.proposed_zero = 0
        self.accepted_by_replica = torch.zeros(
            self.engine.replicas, dtype=torch.int64, device="cuda"
        )
        self.roots_by_replica = torch.zeros_like(self.accepted_by_replica)
        self.arithmetic_counts = (
            torch.zeros(
                (self.engine.replicas, len(self.proposal.modular_sampler.metric_names)),
                dtype=torch.int64,
                device="cuda",
            )
            if self.proposal.modular_root_mode
            in UNIT_GROUP_MODES | PUBLIC_RANGE_MODES | QUOTIENT_MODES | FILTER_MODES
            else None
        )
        self.exchange = (
            ReplicaExchange(
                self.proposal.beta_values,
                self.engine.replicas // len(self.proposal.beta_values),
                self.exchange_seed,
            )
            if self.exchange_every
            else None
        )

    @torch.inference_mode()
    def step(self, betas: torch.Tensor) -> dict:
        if not torch.equal(betas, self.fixed_betas):
            raise ValueError("A fixed circuit lookup table cannot silently change target beta")
        engine = self.engine
        result = self.proposal.step(engine.states, self.beta_indices, generator=self.generator)
        if self.arithmetic_counts is not None:
            self.arithmetic_counts += result.arithmetic_metrics
        if self.history_backend == "triton_device_v1":
            return self._device_history(result)
        changed = int(torch.count_nonzero(result.states != engine.states))
        delta = float((result.energies - engine.energies).sum())
        engine.states.copy_(result.states)
        engine.energies.copy_(result.energies.float())
        improved = engine.energies < engine.best_energies
        engine.best_energies[improved] = engine.energies[improved]
        engine.best_states[improved] = engine.states[improved]
        fresh = (engine.energies == 0) & (engine.first_hit_sweeps < 0)
        if bool(fresh.any()):
            slots = torch.nonzero(fresh).flatten()
            valid = engine.problem.certificate_mask(engine.states[slots].cpu()).cuda()
            if not bool(valid.all()):
                raise ValueError(
                    "A zero-energy circuit event failed the native integer certificate"
                )
            engine.first_hit_sweeps[slots] = engine.global_sweep
            engine.first_hit_states[slots] = engine.states[slots]
        self.events += 1
        self.uniform_draws += engine.replicas * (self.proposal.group_count + 2)
        accepted = int(result.accepted.sum())
        root_changes = int((result.accepted & result.root_changed).sum())
        self.accepted_by_replica += result.accepted
        self.roots_by_replica += result.accepted & result.root_changed
        self.accepted_proposals += accepted
        self.accepted_root_changes += root_changes
        self.proposed_zero += int((result.proposed_energies == 0).sum())
        torch.cuda.synchronize()
        return {
            "event": self.events,
            "changed_spins": changed,
            "energy_delta_sum": delta,
            "accepted_proposals": accepted,
            "accepted_root_changes": root_changes,
        }

    def _device_history(self, result) -> dict:
        from .circuit_history import commit_history

        host = commit_history(
            self.engine,
            result,
            self.accepted_by_replica,
            self.roots_by_replica,
            self.history_payload,
        )
        self.first_hit_sweeps_host = host[:, 0]
        fresh = np.flatnonzero(host[:, 5])
        if len(fresh):
            states = self.engine.first_hit_states[fresh.tolist()].cpu()
            if not bool(self.engine.problem.certificate_mask(states).all()):
                raise ValueError(
                    "A zero-energy circuit event failed the native integer certificate"
                )
        accepted, roots = int(host[:, 3].sum()), int(host[:, 4].sum())
        self.events += 1
        self.uniform_draws += self.engine.replicas * (self.proposal.group_count + 2)
        self.accepted_proposals += accepted
        self.accepted_root_changes += roots
        self.proposed_zero += int(host[:, 6].sum())
        return {
            "event": self.events,
            "changed_spins": int(host[:, 1].sum()),
            "energy_delta_sum": float(host[:, 2].view(np.float64).sum()),
            "accepted_proposals": accepted,
            "accepted_root_changes": roots,
        }

    def after_observation(self) -> bool:
        """Exchange only after the completed MH event has been checked for native hits."""
        if self.exchange and self.events % self.exchange_every == 0:
            self.exchange.step(self.engine)
            return True
        return False

    def diagnostics(self) -> dict:
        indices = self.beta_indices.cpu().numpy()
        accepted = self.accepted_by_replica.cpu().numpy()
        roots = self.roots_by_replica.cpu().numpy()
        by_beta = [
            {
                "beta": float(beta),
                "attempts": self.events * int((indices == i).sum()),
                "accepted": int(accepted[indices == i].sum()),
                "accepted_root_changes": int(roots[indices == i].sum()),
            }
            for i, beta in enumerate(self.proposal.beta_values)
        ]
        arithmetic = {}
        if self.arithmetic_counts is not None:
            counts = self.arithmetic_counts.cpu().numpy()
            rows = [
                dict(
                    zip(
                        self.proposal.modular_sampler.metric_names,
                        map(int, counts[indices == i].sum(0)),
                        strict=True,
                    )
                )
                for i in range(len(self.proposal.beta_values))
            ]
            for row, beta in zip(rows, self.proposal.beta_values, strict=True):
                row["beta"] = float(beta)
                if self.proposal.modular_sampler.uniform_orbit:
                    row["conditional_root_exposure"] = row[
                        "eligible_unit_group_proposals"
                    ] * 2.0 ** (2 - self.proposal.modular_sampler.width)
            arithmetic = {
                "arithmetic_by_beta": rows,
                "arithmetic_observation": (
                    "Read-only current/proposed roots; no clamps or projection"
                ),
            }
        return {
            **arithmetic,
            "history_backend": self.history_backend,
            "old_state_cache": self.proposal.old_state_cache,
            "cache": (
                self.proposal._graph_runtime.cache.diagnostics()
                if self.proposal._graph_runtime is not None
                and self.proposal._graph_runtime.cache is not None
                else None
            ),
            "by_beta": by_beta,
            "stream_name": self.stream_name,
            "root_move_policy": self.root_policy,
            "exchange_every": self.exchange_every,
            "exchange": self.exchange.diagnostics() if self.exchange else None,
            "rng_seed": self.rng_seed,
            "events": self.events,
            "root_guidance": self.proposal.root_guidance,
            "modular_root_mode": self.proposal.modular_root_mode,
            "modular_uint32_draws": (
                0
                if self.proposal.modular_root_mode == "disabled"
                else self.proposal.modular_sampler.noise_words * self.events * self.engine.replicas
            ),
            "guided_root_uint32_draws": (
                0
                if self.proposal.root_guidance == "disabled"
                else 4 * self.events * self.engine.replicas
            ),
            "logical_draws": self.uniform_draws
            + (
                0
                if self.proposal.modular_root_mode == "disabled"
                else self.proposal.modular_sampler.noise_words * self.events * self.engine.replicas
            )
            + (
                0
                if self.proposal.root_guidance == "disabled"
                else 4 * self.events * self.engine.replicas
            )
            + (
                0
                if self.root_policy == "lazy_single_flip_v1"
                else self.events * self.engine.replicas * self.proposal.root_count
            ),
            "root_ranking_fp64_draws": 0
            if self.root_policy == "lazy_single_flip_v1"
            else self.events * self.engine.replicas * self.proposal.root_count,
            "root_integer_draws": self.events * self.engine.replicas,
            "body_uint32_draws": self.events * self.engine.replicas * self.proposal.group_count,
            "acceptance_fp64_draws": self.events * self.engine.replicas,
            "accepted_proposals": self.accepted_proposals,
            "accepted_root_changes": self.accepted_root_changes,
            "proposed_zero_energy": self.proposed_zero,
            "geometry": self.proposal.geometry,
            "root_proposal": self.root_policy,
            "density": "Product of integer uint32 CDF bin masses, with an explicit reverse density",
            "acceptance_precision": "FP64 log Metropolis-Hastings; FP64 acceptance uniform",
        }
