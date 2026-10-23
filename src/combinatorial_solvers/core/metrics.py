"""Success curves, confidence intervals, and censored ITS estimates."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta as beta_distribution


@dataclass(frozen=True)
class SuccessCurvePoint:
    iterations: int
    successes: int
    trials: int
    probability: float
    confidence_low: float
    confidence_high: float


@dataclass(frozen=True)
class WeibullITSFit:
    status: str
    shape: float | None
    scale: float | None
    observed_hits: int
    censored_runs: int
    censor_iteration: int
    targets: dict[str, float | None]
    extrapolated: dict[str, bool | None]
    fit_note: str
    observed_success_probability: float
    predicted_censor_success_probability: float | None
    censor_calibration_error: float | None


@dataclass(frozen=True)
class DiscreteSurvivalPoint:
    """One product-limit step for integer first-passage observations."""

    iterations: int
    at_risk: int
    events: int
    censored: int
    survival_probability: float
    cumulative_success_probability: float
    interval_start_exclusive: int
    interval_at_risk: int
    interval_events: int
    conditional_hit_probability: float


@dataclass(frozen=True)
class GroupedSuccessCurvePoint:
    """A first-hit CDF point with whole-group bootstrap uncertainty."""

    iterations: int
    successes: int
    trials: int
    pooled_probability: float
    groups: int
    group_mean_probability: float
    group_standard_deviation: float
    confidence_low: float | None
    confidence_high: float | None
    interval_method: str
    interval_degenerate: bool | None


@dataclass(frozen=True)
class RestartCutoffCandidate:
    """Plug-in restart work for one positive cutoff."""

    cutoff: int
    success_probability: float
    successful_groups: int
    restart_runs: int | None
    restart_work: int | None


@dataclass(frozen=True)
class CutoffBootstrapResult:
    """Group-bootstrap stability diagnostic for tuning-only cutoff selection."""

    selected_cutoff: int | None
    selected_restart_work: int | None
    target_probability: float
    groups: int
    resamples: int
    seed: int
    finite_resamples: int
    unresolved_resamples: int
    selected_cutoff_confidence_low: float | None
    selected_cutoff_confidence_high: float | None
    selection_frequency: dict[str, float]
    candidates: tuple[RestartCutoffCandidate, ...]


def _validated_integer_vector(values: np.ndarray, label: str) -> np.ndarray:
    raw = np.asarray(values).reshape(-1)
    if raw.size == 0 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{label} must be a non-empty numeric vector")
    numeric = raw.astype(np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{label} must contain finite integer values")
    result = numeric.astype(np.int64)
    if np.any(result < -1):
        raise ValueError(f"{label} uses -1 as its only censor marker")
    return result


def _target_names(targets: tuple[float, ...]) -> dict[str, float]:
    result: dict[str, float] = {}
    for target in targets:
        percent = target * 100.0
        if math.isclose(percent, round(percent), abs_tol=1e-12):
            name = f"its{int(round(percent))}"
        else:
            token = format(target, ".12g").replace(".", "p")
            name = f"its_{token}"
        if name in result:
            raise ValueError("ITS target names collide after formatting")
        result[name] = target
    return result


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval counts are invalid")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def clopper_pearson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return an exact equal-tailed binomial interval."""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Clopper-Pearson interval counts are invalid")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    alpha = 1.0 - confidence
    low = (
        0.0
        if successes == 0
        else float(beta_distribution.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    high = (
        1.0
        if successes == trials
        else float(beta_distribution.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return low, high


def _validated_groups(groups: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(groups).reshape(-1)
    if raw.size != size or raw.size == 0:
        raise ValueError("group_index must have one entry per first-hit observation")
    if np.issubdtype(raw.dtype, np.number):
        numeric = raw.astype(np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError("group_index must contain finite integer values")
        normalized = numeric.astype(np.int64)
    else:
        raise ValueError("group_index must contain numeric group identifiers")
    labels, inverse = np.unique(normalized, return_inverse=True)
    if labels.size == 0:
        raise ValueError("group_index must contain at least one group")
    return labels, inverse


def _bootstrap_group_means(
    group_rates: np.ndarray,
    *,
    resamples: int,
    generator: np.random.Generator,
) -> np.ndarray:
    groups = group_rates.shape[0]
    sampled = generator.integers(0, groups, size=(resamples, groups))
    counts = np.zeros((resamples, groups), dtype=np.int32)
    np.add.at(
        counts,
        (np.repeat(np.arange(resamples), groups), sampled.reshape(-1)),
        1,
    )
    return counts @ group_rates / groups


def discrete_survival_curve(
    first_hit_iterations: np.ndarray,
    *,
    censor_iteration: int,
    checkpoints: np.ndarray | None = None,
) -> tuple[DiscreteSurvivalPoint, ...]:
    """Estimate a discrete product-limit survival curve at sweep boundaries.

    ``-1`` denotes administrative right censoring at the common horizon. Events at
    the censor boundary are processed before administrative censoring at that same
    boundary.
    """

    hits = _validated_integer_vector(first_hit_iterations, "first_hit_iterations")
    if censor_iteration < 0 or np.any(hits > censor_iteration):
        raise ValueError("censor_iteration must cover every observed first hit")
    observed = hits[hits >= 0]
    if checkpoints is None:
        grid = np.unique(np.concatenate(([0], observed, [censor_iteration])))
    else:
        grid = _validated_integer_vector(checkpoints, "checkpoints")
        if np.any(grid < 0) or np.any(grid > censor_iteration):
            raise ValueError("checkpoints must lie inside the observed iteration range")
        grid = np.unique(np.concatenate((grid, [censor_iteration])))

    censored_total = int(np.count_nonzero(hits == -1))
    points: list[DiscreteSurvivalPoint] = []
    previous = -1
    for iteration in grid:
        current = int(iteration)
        events = int(np.count_nonzero(observed == current))
        events_before = int(np.count_nonzero(observed < current))
        events_through = events_before + events
        at_risk = int(hits.size - events_before)
        interval_events = int(np.count_nonzero((observed > previous) & (observed <= current)))
        interval_at_risk = int(hits.size - np.count_nonzero(observed <= previous))
        survival = 1.0 - events_through / hits.size
        censored = censored_total if current == censor_iteration else 0
        points.append(
            DiscreteSurvivalPoint(
                iterations=current,
                at_risk=at_risk,
                events=events,
                censored=censored,
                survival_probability=float(survival),
                cumulative_success_probability=float(1.0 - survival),
                interval_start_exclusive=previous,
                interval_at_risk=interval_at_risk,
                interval_events=interval_events,
                conditional_hit_probability=(
                    0.0 if interval_at_risk == 0 else interval_events / interval_at_risk
                ),
            )
        )
        previous = current
    return tuple(points)


def grouped_success_curve(
    first_hit_iterations: np.ndarray,
    group_index: np.ndarray,
    checkpoints: np.ndarray,
    *,
    censor_iteration: int,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[GroupedSuccessCurvePoint, ...]:
    """Bootstrap complete batch groups instead of treating every chain as a group.

    The point estimate is the equally weighted mean of within-group success rates.
    This equals the pooled rate when every batch contains the same number of chains.
    With one group, the point estimate is retained and the interval is unavailable.
    """

    hits = _validated_integer_vector(first_hit_iterations, "first_hit_iterations")
    if censor_iteration < 0 or np.any(hits > censor_iteration):
        raise ValueError("censor_iteration must cover every observed first hit")
    if not 0 < confidence < 1 or resamples <= 0:
        raise ValueError("Grouped interval settings are invalid")
    labels, inverse = _validated_groups(group_index, hits.size)
    checkpoint_values = _validated_integer_vector(checkpoints, "checkpoints")
    if np.any(checkpoint_values < 0) or np.any(checkpoint_values > censor_iteration):
        raise ValueError("checkpoints must lie inside the observed iteration range")
    checkpoint_values = np.unique(checkpoint_values)
    group_sizes = np.bincount(inverse, minlength=labels.size).astype(np.float64)
    group_rates = np.empty((labels.size, checkpoint_values.size), dtype=np.float64)
    for position, checkpoint in enumerate(checkpoint_values):
        success = ((hits >= 0) & (hits <= checkpoint)).astype(np.float64)
        group_rates[:, position] = (
            np.bincount(
                inverse,
                weights=success,
                minlength=labels.size,
            )
            / group_sizes
        )

    interval_low: np.ndarray | None = None
    interval_high: np.ndarray | None = None
    if labels.size >= 2:
        generator = np.random.default_rng(seed)
        bootstrap_means = _bootstrap_group_means(
            group_rates,
            resamples=resamples,
            generator=generator,
        )
        alpha = 1.0 - confidence
        interval_low, interval_high = np.quantile(
            bootstrap_means,
            [alpha / 2.0, 1.0 - alpha / 2.0],
            axis=0,
        )

    points: list[GroupedSuccessCurvePoint] = []
    for position, checkpoint in enumerate(checkpoint_values):
        successes = int(np.count_nonzero((hits >= 0) & (hits <= checkpoint)))
        low = None if interval_low is None else float(interval_low[position])
        high = None if interval_high is None else float(interval_high[position])
        points.append(
            GroupedSuccessCurvePoint(
                iterations=int(checkpoint),
                successes=successes,
                trials=int(hits.size),
                pooled_probability=successes / int(hits.size),
                groups=int(labels.size),
                group_mean_probability=float(group_rates[:, position].mean()),
                group_standard_deviation=float(group_rates[:, position].std(ddof=0)),
                confidence_low=low,
                confidence_high=high,
                interval_method=(
                    "whole_group_percentile_bootstrap"
                    if labels.size >= 2
                    else "unavailable_single_group"
                ),
                interval_degenerate=(None if low is None else bool(low == high)),
            )
        )
    return tuple(points)


def _restart_runs(success_probability: float, target_probability: float) -> int | None:
    if success_probability <= 0.0:
        return None
    if success_probability >= 1.0:
        return 1
    return math.ceil(math.log1p(-target_probability) / math.log1p(-success_probability))


def bootstrap_restart_cutoff(
    first_hit_iterations: np.ndarray,
    group_index: np.ndarray,
    candidate_cutoffs: np.ndarray,
    *,
    censor_iteration: int,
    target_probability: float = 0.99,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> CutoffBootstrapResult:
    """Select restart work on tuning data and quantify whole-group stability.

    This function intentionally has no held-out fallback. Callers must freeze its
    selected cutoff before evaluating a disjoint dataset. Cutoff zero is excluded
    because initialization has nonzero cost that chain-sweep work does not model.
    """

    hits = _validated_integer_vector(first_hit_iterations, "first_hit_iterations")
    if censor_iteration <= 0 or np.any(hits > censor_iteration):
        raise ValueError("censor_iteration must be positive and cover every first hit")
    if not 0 < target_probability < 1 or not 0 < confidence < 1 or resamples <= 0:
        raise ValueError("Cutoff bootstrap settings are invalid")
    labels, inverse = _validated_groups(group_index, hits.size)
    if labels.size < 2:
        raise ValueError("Cutoff bootstrap requires at least two independent groups")
    cutoffs = _validated_integer_vector(candidate_cutoffs, "candidate_cutoffs")
    cutoffs = np.unique(cutoffs)
    if np.any(cutoffs <= 0) or np.any(cutoffs > censor_iteration):
        raise ValueError("candidate_cutoffs must be positive and no later than censoring")

    group_sizes = np.bincount(inverse, minlength=labels.size).astype(np.float64)
    group_rates = np.empty((labels.size, cutoffs.size), dtype=np.float64)
    successful_groups = np.empty(cutoffs.size, dtype=np.int64)
    for position, cutoff in enumerate(cutoffs):
        success = ((hits >= 0) & (hits <= cutoff)).astype(np.float64)
        group_successes = np.bincount(inverse, weights=success, minlength=labels.size)
        group_rates[:, position] = group_successes / group_sizes
        successful_groups[position] = int(np.count_nonzero(group_successes > 0))

    point_probabilities = group_rates.mean(axis=0)
    point_runs = [_restart_runs(float(value), target_probability) for value in point_probabilities]
    point_work = [
        None if runs is None else int(cutoff) * runs
        for cutoff, runs in zip(cutoffs, point_runs, strict=True)
    ]
    finite_positions = [index for index, value in enumerate(point_work) if value is not None]
    selected_position = (
        None
        if not finite_positions
        else min(finite_positions, key=lambda index: (int(point_work[index]), int(cutoffs[index])))
    )

    generator = np.random.default_rng(seed)
    selected_samples: list[int] = []
    unresolved = 0
    log_failure_target = math.log1p(-target_probability)
    chunk_size = 512
    for start in range(0, resamples, chunk_size):
        count = min(chunk_size, resamples - start)
        probabilities = _bootstrap_group_means(
            group_rates,
            resamples=count,
            generator=generator,
        )
        finite = probabilities > 0.0
        runs = np.full(probabilities.shape, np.inf, dtype=np.float64)
        partial = finite & (probabilities < 1.0)
        runs[partial] = np.ceil(log_failure_target / np.log1p(-probabilities[partial]))
        runs[probabilities >= 1.0] = 1.0
        works = runs * cutoffs[None, :]
        resolved = np.any(finite, axis=1)
        unresolved += int(np.count_nonzero(~resolved))
        if np.any(resolved):
            chosen = np.argmin(works[resolved], axis=1)
            selected_samples.extend(int(value) for value in cutoffs[chosen])

    if selected_samples:
        alpha = 1.0 - confidence
        bounds = np.quantile(selected_samples, [alpha / 2.0, 1.0 - alpha / 2.0])
        counts = {
            str(int(cutoff)): selected_samples.count(int(cutoff)) / len(selected_samples)
            for cutoff in cutoffs
            if int(cutoff) in selected_samples
        }
        confidence_low = float(bounds[0])
        confidence_high = float(bounds[1])
    else:
        counts = {}
        confidence_low = None
        confidence_high = None

    candidates = tuple(
        RestartCutoffCandidate(
            cutoff=int(cutoff),
            success_probability=float(probability),
            successful_groups=int(group_hits),
            restart_runs=None if runs is None else int(runs),
            restart_work=None if work is None else int(work),
        )
        for cutoff, probability, group_hits, runs, work in zip(
            cutoffs,
            point_probabilities,
            successful_groups,
            point_runs,
            point_work,
            strict=True,
        )
    )
    return CutoffBootstrapResult(
        selected_cutoff=(None if selected_position is None else int(cutoffs[selected_position])),
        selected_restart_work=(
            None if selected_position is None else int(point_work[selected_position])
        ),
        target_probability=float(target_probability),
        groups=int(labels.size),
        resamples=int(resamples),
        seed=int(seed),
        finite_resamples=len(selected_samples),
        unresolved_resamples=unresolved,
        selected_cutoff_confidence_low=confidence_low,
        selected_cutoff_confidence_high=confidence_high,
        selection_frequency=counts,
        candidates=candidates,
    )


def empirical_success_curve(
    first_hit_iterations: np.ndarray,
    checkpoints: np.ndarray | None = None,
    *,
    censor_iteration: int | None = None,
    confidence: float = 0.95,
) -> tuple[SuccessCurvePoint, ...]:
    """Build a cumulative first-hit curve; negative values denote right censoring."""

    hits = _validated_integer_vector(first_hit_iterations, "first_hit_iterations")
    observed = hits[hits >= 0]
    if censor_iteration is None:
        if checkpoints is None:
            if observed.size == 0:
                raise ValueError("censor_iteration is required when no successes were observed")
            censor_iteration = int(observed.max())
        else:
            censor_iteration = int(np.asarray(checkpoints).max())
    if censor_iteration < 0 or np.any(observed > censor_iteration):
        raise ValueError("censor_iteration must cover all observed first hits")
    if checkpoints is None:
        checkpoints = np.unique(np.concatenate(([0], observed, [censor_iteration])))
    checkpoint_values = _validated_integer_vector(checkpoints, "checkpoints")
    points: list[SuccessCurvePoint] = []
    for checkpoint in checkpoint_values:
        if checkpoint < 0 or checkpoint > censor_iteration:
            raise ValueError("checkpoints must lie inside the observed iteration range")
        successes = int(np.count_nonzero((hits >= 0) & (hits <= checkpoint)))
        low, high = wilson_interval(successes, int(hits.size), confidence)
        points.append(
            SuccessCurvePoint(
                iterations=int(checkpoint),
                successes=successes,
                trials=int(hits.size),
                probability=successes / int(hits.size),
                confidence_low=low,
                confidence_high=high,
            )
        )
    return tuple(points)


def fit_censored_weibull_its(
    first_hit_iterations: np.ndarray,
    censor_iteration: int,
    targets: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> WeibullITSFit:
    """Fit a Weibull first-passage model with right-censored unsuccessful chains."""

    values = _validated_integer_vector(first_hit_iterations, "first_hit_iterations")
    if censor_iteration < 0:
        raise ValueError("censor_iteration must be nonnegative")
    if np.any(values > censor_iteration):
        raise ValueError("First hits must be no later than the censor iteration")
    if any(not 0 < target < 1 for target in targets):
        raise ValueError("ITS target probabilities must lie strictly between zero and one")
    observed_iterations = values[values >= 0].astype(np.float64)
    observed = observed_iterations + 1.0
    censored_count = int(np.count_nonzero(values == -1))
    observed_probability = float(observed.size / values.size)
    target_names = _target_names(targets)
    if observed.size == 0:
        return WeibullITSFit(
            status="no_observed_hits",
            shape=None,
            scale=None,
            observed_hits=0,
            censored_runs=censored_count,
            censor_iteration=censor_iteration,
            targets={name: None for name in target_names},
            extrapolated={name: None for name in target_names},
            fit_note="No event was observed; report an empirical lower bound instead.",
            observed_success_probability=observed_probability,
            predicted_censor_success_probability=None,
            censor_calibration_error=None,
        )
    if observed.size < 5 or np.unique(observed_iterations).size < 2:
        return WeibullITSFit(
            status="insufficient_data",
            shape=None,
            scale=None,
            observed_hits=int(observed.size),
            censored_runs=censored_count,
            censor_iteration=censor_iteration,
            targets={name: None for name in target_names},
            extrapolated={name: None for name in target_names},
            fit_note="At least five hits at two distinct iterations are required.",
            observed_success_probability=observed_probability,
            predicted_censor_success_probability=None,
            censor_calibration_error=None,
        )

    log_observed = np.log(observed)
    log_censor = math.log(float(censor_iteration + 1))

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        log_scale, log_shape = parameters
        shape = math.exp(float(log_shape))
        scaled_log = log_observed - log_scale
        powered = np.exp(np.clip(shape * scaled_log, -700.0, 700.0))
        log_density = log_shape - log_scale + (shape - 1.0) * scaled_log - powered
        censor_power = math.exp(max(-700.0, min(700.0, shape * (log_censor - log_scale))))
        return float(-log_density.sum() + censored_count * censor_power)

    initial_scale = float(np.median(observed)) if observed.size else float(censor_iteration)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        result = minimize(
            negative_log_likelihood,
            x0=np.asarray([math.log(max(initial_scale, 1.0)), 0.0]),
            method="L-BFGS-B",
            bounds=[(-20.0, 40.0), (-5.0, 5.0)],
        )
    if not result.success or not np.all(np.isfinite(result.x)):
        return WeibullITSFit(
            status="fit_failed",
            shape=None,
            scale=None,
            observed_hits=int(observed.size),
            censored_runs=censored_count,
            censor_iteration=censor_iteration,
            targets={name: None for name in target_names},
            extrapolated={name: None for name in target_names},
            fit_note="The numerical optimizer did not return a finite interior solution.",
            observed_success_probability=observed_probability,
            predicted_censor_success_probability=None,
            censor_calibration_error=None,
        )

    lower_bounds = np.asarray([-20.0, -5.0])
    upper_bounds = np.asarray([40.0, 5.0])
    boundary_tolerance = 1e-4
    if np.any(result.x - lower_bounds <= boundary_tolerance) or np.any(
        upper_bounds - result.x <= boundary_tolerance
    ):
        return WeibullITSFit(
            status="boundary_fit",
            shape=None,
            scale=None,
            observed_hits=int(observed.size),
            censored_runs=censored_count,
            censor_iteration=censor_iteration,
            targets={name: None for name in target_names},
            extrapolated={name: None for name in target_names},
            fit_note="The maximum-likelihood estimate reached a parameter bound.",
            observed_success_probability=observed_probability,
            predicted_censor_success_probability=None,
            censor_calibration_error=None,
        )

    scale = math.exp(float(result.x[0]))
    shape = math.exp(float(result.x[1]))
    predicted_probability = 1.0 - math.exp(-(((censor_iteration + 1.0) / scale) ** shape))
    calibration_error = abs(predicted_probability - observed_probability)
    binomial_standard_error = math.sqrt(
        observed_probability * (1.0 - observed_probability) / values.size
    )
    calibration_tolerance = max(0.02, 3.0 * binomial_standard_error)
    if calibration_error > calibration_tolerance:
        return WeibullITSFit(
            status="poor_fit",
            shape=shape,
            scale=scale,
            observed_hits=int(observed.size),
            censored_runs=censored_count,
            censor_iteration=censor_iteration,
            targets={name: None for name in target_names},
            extrapolated={name: None for name in target_names},
            fit_note=(
                "The fitted CDF misses the observed censor-horizon success rate beyond "
                "a three-standard-error or two-percentage-point tolerance."
            ),
            observed_success_probability=observed_probability,
            predicted_censor_success_probability=predicted_probability,
            censor_calibration_error=calibration_error,
        )
    estimates: dict[str, float | None] = {}
    extrapolated: dict[str, bool | None] = {}
    for name, target in target_names.items():
        estimate = max(0.0, scale * (-math.log1p(-target)) ** (1.0 / shape) - 1.0)
        estimates[name] = float(estimate)
        extrapolated[name] = bool(estimate > censor_iteration)
    return WeibullITSFit(
        status="fitted",
        shape=shape,
        scale=scale,
        observed_hits=int(observed.size),
        censored_runs=censored_count,
        censor_iteration=censor_iteration,
        targets=estimates,
        extrapolated=extrapolated,
        fit_note=(
            "Exploratory continuous-time Weibull fit with a +1 iteration shift; "
            "no confidence interval is provided."
        ),
        observed_success_probability=observed_probability,
        predicted_censor_success_probability=predicted_probability,
        censor_calibration_error=calibration_error,
    )


def time_to_solution(
    run_seconds: float,
    success_probability: float,
    target_probability: float = 0.99,
    *,
    discrete_runs: bool = True,
) -> float:
    if run_seconds < 0 or not 0 <= success_probability <= 1:
        raise ValueError("run_seconds or success_probability is invalid")
    if not 0 < target_probability < 1:
        raise ValueError("target_probability must lie strictly between zero and one")
    if success_probability == 0:
        return math.inf
    if success_probability == 1:
        return run_seconds
    runs = math.log1p(-target_probability) / math.log1p(-success_probability)
    if discrete_runs:
        runs = max(1, math.ceil(runs))
    return run_seconds * runs
