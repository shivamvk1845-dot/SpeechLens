"""Threshold acoustic deviations and merge adjacent detections into regions."""

from __future__ import annotations

from typing import Any

from . import config
from .compare import ComparisonResult

_DIRECTION_RULES = {
    "too_fast": ("speech_rate_wps", 1),
    "too_slow": ("speech_rate_wps", -1),
    "monotone": ("pitch_std_semitones", -1),
    "volume_drop": ("energy_db", -1),
    "missing_pauses": ("pause_ratio", -1),
    "excess_pauses": ("pause_ratio", 1),
    "mumbling": ("hnr_db", -1),
}


def _candidate_from_window(
    window: dict[str, Any], threshold_z: float
) -> list[dict[str, Any]]:
    candidates = []
    for flaw_type, (metric, direction) in _DIRECTION_RULES.items():
        measurement = window["deltas"][metric]
        z_score = measurement["z"]
        if z_score is None or direction * z_score < threshold_z:
            continue
        candidates.append(
            {
                "flaw_type": flaw_type,
                "metric": metric,
                "direction": direction,
                "start_s": window["start_s"],
                "end_s": window["end_s"],
                "z": float(z_score),
                "delta": float(measurement["delta"]),
                "baseline": measurement["baseline"],
                "participant": measurement["participant"],
                "participant_start_s": window["participant_start_s"],
                "participant_end_s": window["participant_end_s"],
            }
        )
    if "spectral_flatness" in window["deltas"] and "spectral_centroid_hz" in window["deltas"]:
        flatness_z = window["deltas"]["spectral_flatness"]["z"]
        centroid_z = window["deltas"]["spectral_centroid_hz"]["z"]
        if (
            flatness_z is not None
            and centroid_z is not None
            and flatness_z >= threshold_z
            and centroid_z <= -threshold_z
        ):
            candidates.append(
                {
                    "flaw_type": "mumbling",
                    "metric": "spectral_flatness",
                    "direction": 1,
                    "start_s": window["start_s"],
                    "end_s": window["end_s"],
                    "z": float(flatness_z),
                    "delta": float(window["deltas"]["spectral_flatness"]["delta"]),
                    "baseline": window["deltas"]["spectral_flatness"]["baseline"],
                    "participant": window["deltas"]["spectral_flatness"]["participant"],
                    "participant_start_s": window["participant_start_s"],
                    "participant_end_s": window["participant_end_s"],
                }
            )
    return candidates


def detect_regions(
    comparison: ComparisonResult,
    threshold_z: float = config.DETECTION_THRESHOLD_Z,
    merge_gap_seconds: float = config.DETECTION_MERGE_GAP_SECONDS,
) -> list[dict[str, Any]]:
    """Detect directional deviations and merge overlapping nearby same-flaw windows.

    A detection requires a directional deviation of at least ``threshold_z``
    configured tolerance units. Region severity is the maximum absolute z-score.
    """
    if threshold_z <= 0 or merge_gap_seconds < 0:
        raise ValueError("threshold_z must be positive and merge_gap_seconds non-negative")
    candidates = [
        candidate
        for window in comparison.windows
        for candidate in _candidate_from_window(window, threshold_z)
    ]
    candidates.sort(key=lambda item: (item["flaw_type"], item["start_s"], item["end_s"]))
    regions: list[dict[str, Any]] = []
    for candidate in candidates:
        if (
            regions
            and regions[-1]["flaw_type"] == candidate["flaw_type"]
            and candidate["start_s"] <= regions[-1]["end_s"] + merge_gap_seconds
        ):
            region = regions[-1]
            region["end_s"] = max(region["end_s"], candidate["end_s"])
            region["severity"] = max(region["severity"], abs(candidate["z"]))
            region["evidence"].append(candidate)
        else:
            regions.append(
                {
                    "flaw_type": candidate["flaw_type"],
                    "start_s": candidate["start_s"],
                    "end_s": candidate["end_s"],
                    "severity": abs(candidate["z"]),
                    "evidence": [candidate],
                }
            )
    return sorted(regions, key=lambda item: (item["start_s"], item["end_s"], item["flaw_type"]))
