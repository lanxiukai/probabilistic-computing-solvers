"""Native high-order heat bath, Metropolis, and FFT memetic tabu controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch

from .kernels import apply_flip, choose_tabu, fft_quantities, site_difference
from .model import legendre


@dataclass(frozen=True)
class Settings:
    method: str = "pbit"
    device: str = "cpu"
    seed: int = 0
    chains: int = 16
    initialization: str = "random"
    max_steps: int = 1000000
    max_seconds: float = 5.0
    cycle_steps: int = 1024
    observe_every: int = 8
    beta_min: float = 0.1
    beta_max: float = 16.0
    population_size: int = 100
    quality_max_step: int | None = None

    def __post_init__(self):
        if self.method not in {"pbit", "metropolis", "memetic-tabu", "greedy"}:
            raise ValueError("Unknown LABS method")
        if self.method == "greedy" and (self.chains != 1 or self.initialization != "legendre"):
            raise ValueError("The deterministic polishing control uses one constructed sequence")
        if self.device not in {"cpu", "cuda"} or self.initialization not in {"random", "legendre"}:
            raise ValueError("Unknown device or initialization")
        for value in (
            self.seed,
            self.chains,
            self.max_steps,
            self.cycle_steps,
            self.observe_every,
            self.population_size,
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Expected nonnegative integer configuration")
        if (
            min(self.chains, self.max_steps, self.cycle_steps, self.observe_every) < 1
            or self.population_size < 2
        ):
            raise ValueError("Invalid iteration or population count")
        if not np.isfinite([self.max_seconds, self.beta_min, self.beta_max]).all() or not (
            self.max_seconds > 0 and 0 < self.beta_min <= self.beta_max
        ):
            raise ValueError("Invalid budget or annealing schedule")
        if self.quality_max_step is not None and not 0 <= self.quality_max_step <= self.max_steps:
            raise ValueError("Invalid quality cutoff")


def solve(instance, settings):
    prepared = perf_counter()
    device = torch.device(settings.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.reset_peak_memory_stats(device)
    rng = torch.Generator(device=device).manual_seed(settings.seed)
    rng_initial = rng.get_state().cpu().numpy().copy()
    shape = (settings.chains, instance.size)
    spins = 2 * torch.randint(0, 2, shape, generator=rng, device=device) - 1
    construction = None
    if settings.initialization == "legendre":
        bits, construction = legendre(instance.size, appended=True)
        spins.copy_(torch.tensor(2 * bits.astype(np.int64) - 1, device=device))
    initial = ((spins + 1) // 2).to(torch.uint8).cpu().numpy().copy()
    rng_after_initial = rng.get_state().cpu().numpy().copy()
    corr, energy, _, max_fft_error = fft_quantities(spins)
    initial_energy = energy.cpu().numpy().copy()
    pending_energy, pending_spins = energy.clone(), spins.clone()
    pending_step = torch.zeros(settings.chains, dtype=torch.int64, device=device)
    best_energy, best_spins = energy.clone(), spins.clone()
    best_step, best_source_step = pending_step.clone(), pending_step.clone()
    visited = torch.zeros(shape, dtype=torch.bool, device=device)
    accepted_counts = torch.zeros(settings.chains, dtype=torch.int64, device=device)
    proposed_counts = accepted_counts.clone()
    tabu_until = torch.zeros(shape, dtype=torch.int64, device=device)
    local_energy, local_spins = energy.clone(), spins.clone()
    population = torch.empty(0, dtype=torch.int8, device=device)
    population_energy = torch.empty(0, dtype=torch.int64, device=device)
    permutation = torch.empty(0, dtype=torch.int64, device=device)
    chain_index = torch.arange(settings.chains, device=device)
    traces = [(0, 0.0, True, initial_energy.copy(), initial_energy.copy(), initial_energy.copy())]
    generation, generation_step, generation_length = 0, 0, 0
    greedy_converged = False
    final_gains = torch.empty(0, dtype=torch.int64, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def consider(values, states, step):
        improved = values < pending_energy
        pending_energy.copy_(torch.where(improved, values, pending_energy))
        pending_spins.copy_(torch.where(improved[:, None], states, pending_spins))
        pending_step.copy_(torch.where(improved, step, pending_step))

    def observe(step, start):
        sync()
        elapsed = perf_counter() - start
        eligible = elapsed <= settings.max_seconds and (
            settings.quality_max_step is None or step <= settings.quality_max_step
        )
        better = (pending_energy < best_energy) & eligible
        best_energy.copy_(torch.where(better, pending_energy, best_energy))
        best_spins.copy_(torch.where(better[:, None], pending_spins, best_spins))
        best_step.copy_(torch.where(better, step, best_step))
        best_source_step.copy_(torch.where(better, pending_step, best_source_step))
        traces.append(
            (
                step,
                elapsed,
                eligible,
                energy.cpu().numpy().copy(),
                pending_energy.cpu().numpy().copy(),
                best_energy.cpu().numpy().copy(),
            )
        )

    def next_length():
        return int(
            torch.randint(
                max(1, instance.size // 2),
                max(2, 3 * instance.size // 2),
                (1,),
                generator=rng,
                device=device,
            ).item()
        )

    def select_parent():
        # The reviewed CPU implementation draws an initial contender plus two challengers.
        candidates = torch.randint(
            settings.population_size, (settings.chains, 3), generator=rng, device=device
        )
        quality = population_energy.gather(1, candidates)
        return candidates.gather(1, quality.argmin(-1, keepdim=True)).flatten()

    sync()
    preparation_seconds = perf_counter() - prepared
    started, completed, failure = perf_counter(), 0, None
    population_initialization_seconds = 0.0
    try:
        with torch.no_grad():
            if settings.method == "memetic-tabu":
                before = perf_counter()
                population = (
                    2
                    * torch.randint(
                        0,
                        2,
                        (settings.chains, settings.population_size, instance.size),
                        generator=rng,
                        device=device,
                        dtype=torch.int8,
                    )
                    - 1
                )
                population[:, 0] = spins.to(torch.int8)
                flat = population.reshape(-1, instance.size)
                values = []
                for start in range(0, len(flat), 64):
                    _, result, _, error = fft_quantities(flat[start : start + 64])
                    values.append(result)
                    max_fft_error = max(max_fft_error, error)
                population_energy = torch.cat(values).reshape(
                    settings.chains, settings.population_size
                )
                index = population_energy.argmin(-1)
                consider(
                    population_energy[chain_index, index],
                    population[chain_index, index].to(torch.int64),
                    0,
                )
                generation_length = next_length()
                population_initialization_seconds = perf_counter() - before
            for step in range(1, settings.max_steps + 1):
                if perf_counter() - started >= settings.max_seconds:
                    break
                if settings.method == "greedy":
                    full_corr, full_energy, gains, error = fft_quantities(spins, gains=True)
                    max_fft_error = max(max_fft_error, error)
                    if not torch.equal(full_corr, corr) or not torch.equal(full_energy, energy):
                        raise AssertionError("Greedy FFT and integer correlation disagree")
                    nodes = gains.argmin(-1)
                    selected = gains.gather(1, nodes[:, None]).flatten() < 0
                    if not bool(selected.any()):
                        greedy_converged = True
                        final_gains = gains
                        break
                    change, delta = site_difference(spins, corr, nodes)
                    if not torch.equal(delta, gains.gather(1, nodes[:, None]).flatten()):
                        raise AssertionError("Greedy FFT gain differs from integer difference")
                    apply_flip(spins, corr, energy, nodes, change, delta, selected)
                elif settings.method == "memetic-tabu":
                    if generation_step >= generation_length:
                        # Insert each local optimum into a randomly selected population slot.
                        slot = torch.randint(
                            settings.population_size,
                            (settings.chains,),
                            generator=rng,
                            device=device,
                        )
                        population[chain_index, slot] = local_spins.to(torch.int8)
                        population_energy[chain_index, slot] = local_energy
                        parent_a, parent_b = select_parent(), select_parent()
                        mask = torch.rand(shape, generator=rng, device=device) < 0.5
                        crossover = torch.where(
                            mask,
                            population[chain_index, parent_a],
                            population[chain_index, parent_b],
                        )
                        sample = torch.randint(
                            settings.population_size,
                            (settings.chains,),
                            generator=rng,
                            device=device,
                        )
                        combine = (
                            torch.rand((settings.chains, 1), generator=rng, device=device) < 0.9
                        )
                        spins = torch.where(combine, crossover, population[chain_index, sample]).to(
                            torch.int64
                        )
                        mutate = torch.rand(shape, generator=rng, device=device) < 1 / instance.size
                        spins.masked_scatter_(mutate, -spins[mutate])
                        corr, energy, _, error = fft_quantities(spins)
                        max_fft_error = max(max_fft_error, error)
                        local_energy, local_spins = energy.clone(), spins.clone()
                        consider(energy, spins, step)
                        tabu_until.zero_()
                        generation += 1
                        generation_step = 0
                        generation_length = next_length()
                    full_corr, full_energy, gains, error = fft_quantities(spins, gains=True)
                    max_fft_error = max(max_fft_error, error)
                    if not torch.equal(full_corr, corr) or not torch.equal(full_energy, energy):
                        raise AssertionError("FFT and integer running correlation disagree")
                    generation_step += 1
                    nodes, selected = choose_tabu(
                        gains,
                        energy,
                        local_energy,
                        tabu_until,
                        generation_step,
                        torch.rand(shape, generator=rng, device=device),
                    )
                    change, delta = site_difference(spins, corr, nodes)
                    if not torch.equal(delta, gains.gather(1, nodes[:, None]).flatten()):
                        raise AssertionError(
                            "Selected FFT gain differs from the exact local difference"
                        )
                    apply_flip(spins, corr, energy, nodes, change, delta, selected)
                    tenure = torch.randint(
                        max(1, generation_length // 50),
                        (settings.chains,),
                        generator=rng,
                        device=device,
                    )
                    until = generation_step + generation_length // 10 + tenure
                    tabu_until[chain_index, nodes] = torch.where(
                        selected, until, tabu_until[chain_index, nodes]
                    )
                    improved = energy < local_energy
                    local_energy.copy_(torch.where(improved, energy, local_energy))
                    local_spins.copy_(torch.where(improved[:, None], spins, local_spins))
                else:
                    cursor = (step - 1) % instance.size
                    if cursor == 0:
                        permutation = torch.randperm(instance.size, generator=rng, device=device)
                    nodes = permutation[cursor].expand(settings.chains)
                    change, delta = site_difference(spins, corr, nodes)
                    phase = (step - 1) % settings.cycle_steps
                    fraction = phase / max(1, settings.cycle_steps - 1)
                    beta = (
                        settings.beta_min
                        * (settings.beta_max / settings.beta_min) ** fraction
                        / instance.size
                    )
                    probability = (
                        torch.sigmoid(-beta * delta.to(torch.float64))
                        if settings.method == "pbit"
                        else torch.exp((-beta * delta.to(torch.float64)).clamp_max(0))
                    )
                    selected = (
                        torch.rand(
                            (settings.chains,), generator=rng, device=device, dtype=torch.float64
                        )
                        < probability
                    )
                    apply_flip(spins, corr, energy, nodes, change, delta, selected)
                visited[chain_index, nodes] = True
                accepted_counts += selected
                proposed_counts += 1
                consider(energy, spins, step)
                completed = step
                sync()
                last = (
                    perf_counter() - started >= settings.max_seconds or step == settings.max_steps
                )
                if step % settings.observe_every == 0 or last:
                    observe(step, started)
                if last:
                    break
            if traces[-1][0] != completed:
                observe(completed, started)
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    sync()
    search_seconds = perf_counter() - started
    final_corr, final_energy, _, error = fft_quantities(spins)
    max_fft_error = max(max_fft_error, error)
    if not torch.equal(final_corr, corr) or not torch.equal(final_energy, energy):
        failure = {"type": "CorrelationDrift", "message": "Final full check disagrees"}
    arrays = {
        "initial_bits": initial,
        "initial_energy": initial_energy,
        "final_bits": ((spins + 1) // 2).to(torch.uint8).cpu().numpy(),
        "final_correlations": corr.cpu().numpy(),
        "final_energy": energy.cpu().numpy(),
        "best_bits": ((best_spins + 1) // 2).to(torch.uint8).cpu().numpy(),
        "best_energy": best_energy.cpu().numpy(),
        "best_step": best_step.cpu().numpy(),
        "best_source_step": best_source_step.cpu().numpy(),
        "pending_bits": ((pending_spins + 1) // 2).to(torch.uint8).cpu().numpy(),
        "pending_energy": pending_energy.cpu().numpy(),
        "visited_variables": visited.cpu().numpy(),
        "accepted_counts": accepted_counts.cpu().numpy(),
        "proposed_counts": proposed_counts.cpu().numpy(),
        "last_permutation": permutation.cpu().numpy(),
        "final_gains": final_gains.cpu().numpy(),
        "rng_initial": rng_initial,
        "rng_after_initial": rng_after_initial,
        "rng_final": rng.get_state().cpu().numpy(),
        "trace_step": np.asarray([r[0] for r in traces], dtype=np.int64),
        "trace_seconds": np.asarray([r[1] for r in traces]),
        "trace_in_budget": np.asarray([r[2] for r in traces]),
        "trace_working_energy": np.stack([r[3] for r in traces]),
        "trace_candidate_energy": np.stack([r[4] for r in traces]),
        "trace_best_energy": np.stack([r[5] for r in traces]),
    }
    return arrays, {
        "settings": asdict(settings),
        "size": instance.size,
        "best_energy": int(best_energy.min()),
        "merit_factor": instance.size**2 / (2 * int(best_energy.min())),
        "completed_steps": completed,
        "greedy_converged": greedy_converged,
        "memetic_generations": generation,
        "generation_length": generation_length,
        "generation_step": generation_step,
        "max_fft_roundoff": max_fft_error,
        "physical_update_pbits_per_model": instance.size
        if settings.method in {"pbit", "metropolis"}
        else 0,
        "model_binary_variables": instance.size,
        "auxiliary_pbits": 0,
        "replicas": settings.chains,
        "construction": construction,
        "population_values_per_replica": settings.population_size * instance.size
        if settings.method == "memetic-tabu"
        else 0,
        "population_storage_bytes": population.numel() * population.element_size(),
        "minimum_visited_variables_per_replica": int(visited.sum(-1).min()),
        "union_visited_variables": int(visited.any(0).sum()),
        "preparation_seconds": preparation_seconds,
        "population_initialization_seconds": population_initialization_seconds,
        "search_seconds": search_seconds,
        "failure": failure,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "population_capture": (
            "Recoverable from source, seed, RNG and fixed-work replay; "
            "internal population not separately exported"
        ),
    }
