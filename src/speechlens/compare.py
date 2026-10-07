"""DTW alignment and window/word acoustic deviations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np

from . import config
from .features import AcousticFeatures


@dataclass(frozen=True)
class ComparisonResult:
    """Temporal alignment and metric deltas from baseline to participant."""

    dtw_path: list[tuple[int, int]]
    windows: list[dict[str, Any]]
    word_deltas: list[dict[str, Any]]


def _finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else None


def _finite_std(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.std(finite)) if finite.size else None


def _window_indices(times: np.ndarray, start: float, end: float) -> np.ndarray:
    return np.flatnonzero((times >= start) & (times < end))


def _overlap_ratio(
    intervals: list[dict[str, float]], start: float, end: float
) -> float:
    duration = end - start
    return sum(
        max(0.0, min(end, interval["end"]) - max(start, interval["start"]))
        for interval in intervals
    ) / duration


def _mean_word_rate(
    words: list[dict[str, Any]], start: float, end: float
) -> float | None:
    rates = [
        float(word["speech_rate_wps"])
        for word in words
        if start <= (float(word["start"]) + float(word["end"])) / 2 < end
    ]
    return float(np.mean(rates)) if rates else None


def _metric_values(
    features: AcousticFeatures,
    indices: np.ndarray,
    baseline_f0_reference: float,
) -> dict[str, float | None]:
    valid_f0 = features.f0_hz[indices]
    voiced = features.voiced[indices] & np.isfinite(valid_f0) & (valid_f0 > 0)
    semitones = (
        12.0 * np.log2(valid_f0[voiced] / baseline_f0_reference)
        if np.any(voiced)
        else np.asarray([], dtype=np.float64)
    )
    return {
        "pitch_mean_semitones": _finite_mean(semitones),
        "pitch_std_semitones": _finite_std(semitones),
        "energy_db": _finite_mean(features.rms_db[indices]),
        "hnr_db": _finite_mean(features.hnr_db[indices]),
        "spectral_centroid_hz": _finite_mean(features.spectral_centroid_hz[indices]),
        "spectral_flatness": _finite_mean(features.spectral_flatness[indices]),
    }


def _deltas(
    baseline_values: dict[str, float | None],
    participant_values: dict[str, float | None],
) -> dict[str, dict[str, float | None]]:
    output = {}
    for metric, baseline_value in baseline_values.items():
        participant_value = participant_values[metric]
        delta = (
            participant_value - baseline_value
            if baseline_value is not None and participant_value is not None
            else None
        )
        tolerance = config.DEVIATION_TOLERANCES[metric]
        output[metric] = {
            "baseline": baseline_value,
            "participant": participant_value,
            "delta": delta,
            "z": delta / tolerance if delta is not None else None,
        }
    return output


def compare_features(
    baseline: AcousticFeatures,
    participant: AcousticFeatures,
    window_seconds: float = config.COMPARISON_WINDOW_SECONDS,
    hop_seconds: float = config.COMPARISON_HOP_SECONDS,
) -> ComparisonResult:
    """Align MFCC frames by DTW and calculate deltas per baseline-time window.

    The DTW path pairs baseline frames with participant frames. Windows are
    laid out on the baseline clock; each participant window aggregates the
    frames paired to baseline frames inside that window.
    """
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    if baseline.mfcc.shape[0] != participant.mfcc.shape[0]:
        raise ValueError("Baseline and participant MFCC dimensions must match")
    if baseline.mfcc.shape[1] == 0 or participant.mfcc.shape[1] == 0:
        raise ValueError("Both recordings must contain acoustic feature frames")
    if not np.all(np.isfinite(baseline.mfcc)) or not np.all(
        np.isfinite(participant.mfcc)
    ):
        raise ValueError("DTW requires finite baseline and participant MFCC values")
    baseline_valid_f0 = baseline.f0_hz[
        baseline.voiced & np.isfinite(baseline.f0_hz) & (baseline.f0_hz > 0)
    ]
    if baseline_valid_f0.size == 0:
        raise ValueError("Baseline contains no voiced F0 observations")
    baseline_f0_reference = float(np.median(baseline_valid_f0))

    _, path = librosa.sequence.dtw(
        X=baseline.mfcc,
        Y=participant.mfcc,
        metric=config.DTW_METRIC,
        backtrack=True,
    )
    dtw_path = [(int(base_index), int(participant_index)) for base_index, participant_index in path]
    participant_by_baseline: dict[int, list[int]] = {}
    for base_index, participant_index in dtw_path:
        participant_by_baseline.setdefault(base_index, []).append(participant_index)

    duration = float(baseline.times_s[-1] + config.FEATURE_HOP_SECONDS)
    windows = []
    start = 0.0
    while start < duration:
        end = min(duration, start + window_seconds)
        baseline_indices = _window_indices(baseline.times_s, start, end)
        participant_indices = sorted(
            {
                index
                for baseline_index in baseline_indices
                for index in participant_by_baseline.get(int(baseline_index), [])
            }
        )
        if baseline_indices.size and participant_indices:
            baseline_values = _metric_values(
                baseline, baseline_indices, baseline_f0_reference
            )
            participant_values = _metric_values(
                participant, np.asarray(participant_indices), baseline_f0_reference
            )
            baseline_rate = _mean_word_rate(baseline.word_features, start, end)
            participant_start = float(participant.times_s[participant_indices[0]])
            participant_end = float(
                participant.times_s[participant_indices[-1]] + config.FEATURE_HOP_SECONDS
            )
            participant_rate = _mean_word_rate(
                participant.word_features, participant_start, participant_end
            )
            base_pause_ratio = _overlap_ratio(baseline.pause_intervals, start, end)
            participant_pause_ratio = _overlap_ratio(
                participant.pause_intervals, participant_start, participant_end
            )
            baseline_values["speech_rate_wps"] = baseline_rate
            participant_values["speech_rate_wps"] = participant_rate
            baseline_values["pause_ratio"] = base_pause_ratio
            participant_values["pause_ratio"] = participant_pause_ratio
            windows.append(
                {
                    "start_s": start,
                    "end_s": end,
                    "participant_start_s": participant_start,
                    "participant_end_s": participant_end,
                    "deltas": _deltas(baseline_values, participant_values),
                }
            )
        start += hop_seconds

    word_deltas = _compare_words(
        baseline.word_features,
        participant.word_features,
        baseline,
        participant,
        baseline_f0_reference,
    )
    return ComparisonResult(dtw_path=dtw_path, windows=windows, word_deltas=word_deltas)


def _compare_words(
    baseline_words: list[dict[str, Any]],
    participant_words: list[dict[str, Any]],
    baseline_features: AcousticFeatures,
    participant_features: AcousticFeatures,
    baseline_f0_reference: float,
) -> list[dict[str, Any]]:
    if not baseline_words or not participant_words:
        return []
    participant_by_token: dict[str, list[dict[str, Any]]] = {}
    for word in participant_words:
        participant_by_token.setdefault(str(word["word"]).strip().lower(), []).append(word)
    word_deltas = []
    for baseline_word in baseline_words:
        token = str(baseline_word["word"]).strip().lower()
        matches = participant_by_token.get(token)
        if not matches:
            continue
        participant_word = matches.pop(0)
        baseline_rate = float(baseline_word["speech_rate_wps"])
        participant_rate = float(participant_word["speech_rate_wps"])
        baseline_indices = _window_indices(
            baseline_features.times_s,
            float(baseline_word["start"]),
            float(baseline_word["end"]),
        )
        participant_indices = _window_indices(
            participant_features.times_s,
            float(participant_word["start"]),
            float(participant_word["end"]),
        )
        if not baseline_indices.size or not participant_indices.size:
            continue
        baseline_values = _metric_values(
            baseline_features, baseline_indices, baseline_f0_reference
        )
        participant_values = _metric_values(
            participant_features, participant_indices, baseline_f0_reference
        )
        baseline_values["speech_rate_wps"] = baseline_rate
        participant_values["speech_rate_wps"] = participant_rate
        word_deltas.append(
            {
                "word": baseline_word["word"],
                "baseline_start_s": baseline_word["start"],
                "baseline_end_s": baseline_word["end"],
                "participant_start_s": participant_word["start"],
                "participant_end_s": participant_word["end"],
                "deltas": _deltas(baseline_values, participant_values),
                "duration_s": {
                    "baseline": baseline_word["duration_s"],
                    "participant": participant_word["duration_s"],
                    "delta": participant_word["duration_s"] - baseline_word["duration_s"],
                },
            }
        )
    return word_deltas
