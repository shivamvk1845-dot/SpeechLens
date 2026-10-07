"""Frame-level acoustic and word-level timing feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import librosa
import numpy as np
import parselmouth
from scipy.signal import resample_poly

from . import config


@dataclass(frozen=True)
class AcousticFeatures:
    """Synchronized frame features and timing-derived word/pause measurements."""

    times_s: np.ndarray
    f0_hz: np.ndarray
    rms_db: np.ndarray
    mfcc: np.ndarray
    spectral_centroid_hz: np.ndarray
    spectral_flatness: np.ndarray
    hnr_db: np.ndarray
    voiced: np.ndarray
    word_features: list[dict[str, Any]]
    pause_intervals: list[dict[str, float]]


def _normalize_audio(
    samples: Iterable[float] | np.ndarray, sample_rate: int
) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Feature extraction requires non-empty mono or multichannel audio")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains non-finite samples")
    if sample_rate != config.SAMPLE_RATE:
        divisor = int(np.gcd(sample_rate, config.SAMPLE_RATE))
        audio = resample_poly(
            audio,
            config.SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )
    return np.asarray(audio, dtype=np.float64)


def _word_and_pause_features(
    alignments: list[Mapping[str, Any]] | None,
    duration_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    if not alignments:
        return [], []
    words = []
    for item in alignments:
        word = str(item["word"])
        start, end = float(item["start"]), float(item["end"])
        if (
            not word.strip()
            or start < 0
            or end <= start
            or end > duration_s + config.ALIGNMENT_TIMESTAMP_TOLERANCE_S
        ):
            raise ValueError(f"Invalid word alignment: {item!r}")
        words.append({"word": word, "start": start, "end": end})
    words.sort(key=lambda item: (item["start"], item["end"]))

    word_features = []
    radius = config.FEATURE_WORD_RATE_WINDOW_SIZE // 2
    for index, word in enumerate(words):
        left = max(0, index - radius)
        right = min(len(words), index + radius + 1)
        window_duration = words[right - 1]["end"] - words[left]["start"]
        word_features.append(
            {
                **word,
                "duration_s": word["end"] - word["start"],
                "speech_rate_wps": (right - left) / window_duration,
            }
        )

    pause_intervals = []
    for previous, following in zip(words, words[1:]):
        gap = following["start"] - previous["end"]
        if gap >= config.MIN_PAUSE_SECONDS:
            pause_intervals.append(
                {
                    "start": previous["end"],
                    "end": following["start"],
                    "duration_s": gap,
                }
            )
    return word_features, pause_intervals


def extract_features(
    samples: Iterable[float] | np.ndarray,
    sample_rate: int = config.SAMPLE_RATE,
    word_alignments: list[Mapping[str, Any]] | None = None,
) -> AcousticFeatures:
    """Extract 10 ms F0, energy, MFCC, spectral, HNR, voicing and word features.

    RMS dB is ``20*log10(RMS)`` with a configured floor. HNR is NaN where
    Praat does not define a voiced harmonicity value; unvoiced F0 is NaN and
    separately identified by ``voiced``.
    """
    audio = _normalize_audio(samples, sample_rate)
    hop_length = round(config.FEATURE_HOP_SECONDS * config.SAMPLE_RATE)
    frame_length = config.FEATURE_FRAME_LENGTH
    expected_frames = 1 + len(audio) // hop_length

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0][:expected_frames]
    rms_db = 20.0 * np.log10(np.maximum(rms, config.FEATURE_RMS_FLOOR))
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=config.SAMPLE_RATE,
        n_mfcc=config.FEATURE_N_MFCC,
        n_fft=frame_length,
        hop_length=hop_length,
        center=True,
    )[:, :expected_frames]
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=config.SAMPLE_RATE,
        n_fft=frame_length,
        hop_length=hop_length,
        center=True,
    )[0, :expected_frames]
    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=frame_length,
        hop_length=hop_length,
        center=True,
    )[0, :expected_frames]

    if np.max(np.abs(audio)) < config.FEATURE_RMS_FLOOR:
        f0 = np.full(expected_frames, np.nan)
        voiced = np.zeros(expected_frames, dtype=bool)
    else:
        padded = np.pad(audio, (0, max(0, frame_length - len(audio))))
        estimated_f0, _, voiced_flag = librosa.pyin(
            padded,
            fmin=config.FEATURE_F0_MIN_HZ,
            fmax=config.FEATURE_F0_MAX_HZ,
            sr=config.SAMPLE_RATE,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )
        voiced = np.asarray(voiced_flag[:expected_frames], dtype=bool)
        f0 = np.asarray(estimated_f0[:expected_frames], dtype=np.float64)
        voiced &= np.isfinite(f0) & (f0 > 0)
        f0[~voiced] = np.nan

    sound = parselmouth.Sound(audio, sampling_frequency=config.SAMPLE_RATE)
    harmonicity = sound.to_harmonicity_cc(
        time_step=config.FEATURE_HOP_SECONDS,
        minimum_pitch=config.FEATURE_HNR_MIN_PITCH_HZ,
        silence_threshold=config.FEATURE_HNR_SILENCE_THRESHOLD,
        periods_per_window=config.FEATURE_HNR_PERIODS_PER_WINDOW,
    )
    times = librosa.frames_to_time(
        np.arange(expected_frames),
        sr=config.SAMPLE_RATE,
        hop_length=hop_length,
    )
    hnr_db = np.asarray(
        [harmonicity.get_value(float(time)) for time in times],
        dtype=np.float64,
    )
    word_features, pauses = _word_and_pause_features(
        word_alignments, len(audio) / config.SAMPLE_RATE
    )
    return AcousticFeatures(
        times_s=times,
        f0_hz=f0,
        rms_db=rms_db,
        mfcc=mfcc,
        spectral_centroid_hz=centroid,
        spectral_flatness=flatness,
        hnr_db=hnr_db,
        voiced=voiced,
        word_features=word_features,
        pause_intervals=pauses,
    )
