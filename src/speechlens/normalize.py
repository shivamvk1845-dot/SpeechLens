"""Speaker-relative normalization for acoustic feature sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from . import config
from .features import AcousticFeatures


@dataclass(frozen=True)
class SpeakerStats:
    """Robust speaker medians and per-feature z-score parameters."""

    median_f0_hz: float
    median_energy_db: float
    means: dict[str, float]
    standard_deviations: dict[str, float]


def _relative_arrays(
    features: AcousticFeatures, median_f0_hz: float, median_energy_db: float
) -> dict[str, np.ndarray]:
    f0_semitones = np.full(features.f0_hz.shape, np.nan, dtype=np.float64)
    valid_f0 = features.voiced & np.isfinite(features.f0_hz) & (features.f0_hz > 0)
    f0_semitones[valid_f0] = 12.0 * np.log2(
        features.f0_hz[valid_f0] / median_f0_hz
    )
    arrays = {
        "f0_semitones": f0_semitones,
        "energy_relative_db": features.rms_db - median_energy_db,
        "spectral_centroid_hz": features.spectral_centroid_hz,
        "spectral_flatness": features.spectral_flatness,
        "hnr_db": features.hnr_db,
    }
    for coefficient in range(features.mfcc.shape[0]):
        arrays[f"mfcc_{coefficient + 1:02d}"] = features.mfcc[coefficient]
    return arrays


def fit_speaker_stats(
    features_by_speaker: Mapping[str, Sequence[AcousticFeatures]],
) -> dict[str, SpeakerStats]:
    """Fit medians and z-score distributions from each speaker's recordings.

    F0 is converted to semitones by ``12*log2(F0/median_F0)`` and energy is
    expressed as ``dB - median_dB`` before per-speaker mean/std normalization.
    """
    if not features_by_speaker:
        raise ValueError("At least one speaker's features are required")
    fitted = {}
    for speaker, recordings in features_by_speaker.items():
        if not recordings:
            raise ValueError(f"Speaker {speaker!r} has no feature recordings")
        voiced_f0 = np.concatenate(
            [
                recording.f0_hz[
                    recording.voiced
                    & np.isfinite(recording.f0_hz)
                    & (recording.f0_hz > 0)
                ]
                for recording in recordings
            ]
        )
        energy = np.concatenate([recording.rms_db for recording in recordings])
        if voiced_f0.size == 0:
            raise ValueError(f"Speaker {speaker!r} has no voiced F0 observations")
        median_f0 = float(np.median(voiced_f0))
        median_energy = float(np.median(energy))
        combined: dict[str, list[np.ndarray]] = {}
        for recording in recordings:
            for name, values in _relative_arrays(
                recording, median_f0, median_energy
            ).items():
                finite = values[np.isfinite(values)]
                if finite.size:
                    combined.setdefault(name, []).append(finite)
        means = {}
        standard_deviations = {}
        for name, chunks in combined.items():
            values = np.concatenate(chunks)
            means[name] = float(np.mean(values))
            standard_deviations[name] = max(
                float(np.std(values)), config.NORMALIZATION_STD_FLOOR
            )
        fitted[speaker] = SpeakerStats(
            median_f0_hz=median_f0,
            median_energy_db=median_energy,
            means=means,
            standard_deviations=standard_deviations,
        )
    return fitted


def normalize_features(
    features: AcousticFeatures, speaker_stats: SpeakerStats
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    """Convert raw features to speaker-relative values and per-speaker z-scores."""
    relative = _relative_arrays(
        features,
        speaker_stats.median_f0_hz,
        speaker_stats.median_energy_db,
    )
    z_scores = {}
    for name, values in relative.items():
        mean = speaker_stats.means.get(name)
        standard_deviation = speaker_stats.standard_deviations.get(name)
        if mean is None or standard_deviation is None:
            z_scores[name] = np.full(values.shape, np.nan, dtype=np.float64)
        else:
            z_scores[name] = (values - mean) / standard_deviation
    return {
        **relative,
        "z_scores": z_scores,
        "times_s": features.times_s.copy(),
        "voiced": features.voiced.copy(),
    }
