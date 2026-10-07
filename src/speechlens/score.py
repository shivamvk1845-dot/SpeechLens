"""Transparent weighted rubric scoring from baseline-relative deviations."""

from __future__ import annotations

from typing import Any

from . import config
from .compare import ComparisonResult

_CATEGORY_METRICS = {
    "pace": ("speech_rate_wps",),
    "pitch": ("pitch_mean_semitones", "pitch_std_semitones"),
    "volume": ("energy_db",),
    "pauses": ("pause_ratio",),
    "clarity": (
        "hnr_db",
        "spectral_centroid_hz",
        "spectral_flatness",
    ),
}


def _mean_absolute_z(comparison: ComparisonResult, metric_names: tuple[str, ...]) -> float:
    values = [
        abs(float(measurement["z"]))
        for window in comparison.windows
        for metric in metric_names
        if (measurement := window["deltas"].get(metric)) is not None
        and measurement["z"] is not None
    ]
    return sum(values) / len(values) if values else 0.0


def score_comparison(
    comparison: ComparisonResult,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score each rubric category and weighted total from mean |delta/tolerance|.

    Category score is ``max(0, 100 - penalty_per_z * mean(abs(z)))``.
    Total score is the weighted mean of category scores.
    """
    rubric_weights = dict(config.RUBRIC_WEIGHTS if weights is None else weights)
    if set(rubric_weights) != set(_CATEGORY_METRICS):
        raise ValueError(f"Rubric weights must define exactly {sorted(_CATEGORY_METRICS)}")
    if any(weight < 0 for weight in rubric_weights.values()):
        raise ValueError("Rubric weights must be non-negative")
    weight_sum = sum(rubric_weights.values())
    if weight_sum <= 0:
        raise ValueError("At least one rubric weight must be positive")

    categories = {}
    for category, metrics in _CATEGORY_METRICS.items():
        mean_z = _mean_absolute_z(comparison, metrics)
        categories[category] = {
            "score": max(
                0.0,
                min(100.0, 100.0 - config.RUBRIC_PENALTY_PER_Z * mean_z),
            ),
            "mean_absolute_deviation_z": mean_z,
            "weight": rubric_weights[category],
        }
    total = sum(
        result["score"] * result["weight"] for result in categories.values()
    ) / weight_sum
    return {
        "total": max(0.0, min(100.0, total)),
        "categories": categories,
        "weights": rubric_weights,
    }
