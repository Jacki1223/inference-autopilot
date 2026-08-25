"""Bayesian paired-effect inference and sequential stopping for InferOpt."""

from __future__ import annotations

import math
from typing import Any


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 200
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        m2 = 2 * iteration
        numerator = iteration * (b - iteration) * x / (
            (qam + m2) * (a + m2)
        )
        d = 1.0 + numerator * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + numerator / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        result *= d * c
        numerator = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + numerator * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + numerator / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return result


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_of_freedom: float) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if value == 0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * regularized_beta(
        x, degrees_of_freedom / 2.0, 0.5
    )
    return 1.0 - tail if value > 0 else tail


def paired_log_ratios(
    rows: list[dict[str, Any]], objective_metric: str,
    direction: str = "maximize",
) -> list[float]:
    by_repeat: dict[int, dict[str, float]] = {}
    for row in rows:
        if not row.get("ok"):
            continue
        value = row.get("metrics", {}).get(objective_metric)
        if not isinstance(value, (int, float)) or value <= 0:
            continue
        kind = row.get("kind")
        if kind not in {"baseline", "candidate"}:
            continue
        by_repeat.setdefault(int(row.get("repeat_index", 0)), {})[kind] = float(value)
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize")
    return [
        math.log(
            values["candidate"] / values["baseline"]
            if direction == "maximize"
            else values["baseline"] / values["candidate"]
        )
        for _, values in sorted(by_repeat.items())
        if "baseline" in values and "candidate" in values
    ]


def normal_inverse_gamma_posterior(
    observations: list[float],
    *,
    prior_mean: float = 0.0,
    prior_strength: float = 0.01,
    prior_alpha: float = 1.0,
    prior_beta: float = 0.0002,
) -> dict[str, float]:
    count = len(observations)
    sample_mean = sum(observations) / count if count else 0.0
    sum_squares = sum((value - sample_mean) ** 2 for value in observations)
    strength = prior_strength + count
    mean_value = (
        prior_strength * prior_mean + count * sample_mean
    ) / strength
    alpha = prior_alpha + count / 2.0
    beta = (
        prior_beta + 0.5 * sum_squares
        + prior_strength * count * (sample_mean - prior_mean) ** 2
        / (2.0 * strength)
    )
    degrees_of_freedom = 2.0 * alpha
    scale = math.sqrt(beta / (alpha * strength))
    return {
        "mean": mean_value,
        "strength": strength,
        "alpha": alpha,
        "beta": beta,
        "degrees_of_freedom": degrees_of_freedom,
        "scale": scale,
        "observations": count,
    }


def posterior_probability_above(
    posterior: dict[str, float], threshold: float
) -> float:
    scale = posterior["scale"]
    if scale == 0:
        return float(posterior["mean"] > threshold)
    standardized = (threshold - posterior["mean"]) / scale
    return 1.0 - student_t_cdf(
        standardized, posterior["degrees_of_freedom"]
    )


def sequential_decision(
    rows: list[dict[str, Any]],
    *,
    objective_metric: str,
    minimum_improvement_pct: float,
    direction: str = "maximize",
    min_blocks: int = 2,
    max_blocks: int = 6,
    accept_probability: float = 0.95,
    reject_probability: float = 0.05,
    prior_mean_pct: float = 0.0,
    prior_strength: float = 0.01,
) -> dict[str, Any]:
    differences = paired_log_ratios(rows, objective_metric, direction)
    posterior = normal_inverse_gamma_posterior(
        differences,
        prior_mean=math.log1p(prior_mean_pct / 100.0),
        prior_strength=prior_strength,
    )
    probability_positive = posterior_probability_above(posterior, 0.0)
    probability_minimum = posterior_probability_above(
        posterior, math.log1p(minimum_improvement_pct / 100.0)
    )
    candidate_rows = [row for row in rows if row.get("kind") == "candidate"]
    successes = sum(bool(row.get("slo", {}).get("passed")) for row in candidate_rows)
    beta_alpha = 9.0 + successes
    beta_beta = 1.0 + len(candidate_rows) - successes
    slo_next_pass_probability = beta_alpha / (beta_alpha + beta_beta)
    blocks = len(differences)
    if blocks < min_blocks:
        action, reason = "continue", "minimum paired blocks not reached"
    elif probability_minimum >= accept_probability and all(
        row.get("slo", {}).get("passed") for row in candidate_rows
    ):
        action, reason = "accept", "posterior exceeds minimum-gain threshold"
    elif probability_minimum <= reject_probability:
        action, reason = "reject", "posterior probability of minimum gain is low"
    elif blocks >= max_blocks:
        action, reason = "inconclusive", "maximum paired blocks reached"
    else:
        action, reason = "continue", "posterior remains between stopping boundaries"
    return {
        "schema_version": 1,
        "action": action,
        "reason": reason,
        "blocks": blocks,
        "min_blocks": min_blocks,
        "max_blocks": max_blocks,
        "minimum_improvement_pct": minimum_improvement_pct,
        "direction": direction,
        "accept_probability": accept_probability,
        "reject_probability": reject_probability,
        "probability_improvement_gt_zero": probability_positive,
        "probability_improvement_gt_minimum": probability_minimum,
        "posterior_mean_improvement_pct": math.expm1(posterior["mean"]) * 100.0,
        "posterior": posterior,
        "paired_log_ratios": differences,
        "slo_beta_posterior": {
            "alpha": beta_alpha,
            "beta": beta_beta,
            "next_pass_probability": slo_next_pass_probability,
        },
    }


def sequential_decision_from_samples(
    baseline_samples: list[float],
    candidate_samples: list[float],
    *,
    objective_metric: str,
    minimum_improvement_pct: float,
    direction: str = "maximize",
    candidate_slo_passes: list[bool] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for repeat_index, (baseline, candidate) in enumerate(
        zip(baseline_samples, candidate_samples)
    ):
        rows.extend([
            {
                "ok": True, "kind": "baseline", "repeat_index": repeat_index,
                "metrics": {objective_metric: baseline}, "slo": {"passed": True},
            },
            {
                "ok": True, "kind": "candidate", "repeat_index": repeat_index,
                "metrics": {objective_metric: candidate},
                "slo": {"passed": (
                    candidate_slo_passes[repeat_index]
                    if candidate_slo_passes and repeat_index < len(candidate_slo_passes)
                    else True
                )},
            },
        ])
    return sequential_decision(
        rows,
        objective_metric=objective_metric,
        minimum_improvement_pct=minimum_improvement_pct,
        direction=direction,
        **kwargs,
    )
