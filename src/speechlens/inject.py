"""Deterministic synthetic speech flaw injection using audio DSP libraries."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import pyworld
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt

from . import config

SAMPLE_RATE = config.SAMPLE_RATE
FLAW_TYPES = (
    "too_fast",
    "too_slow",
    "monotone",
    "missing_pauses",
    "excess_pauses",
    "volume_drop",
    "mumbling",
)
_PROCESS_ORDER = (
    "too_fast",
    "too_slow",
    "monotone",
    "missing_pauses",
    "volume_drop",
    "mumbling",
    "excess_pauses",
)


def read_wav(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read audio through libsndfile, downmix to mono, and polyphase-resample."""
    samples, source_rate = sf.read(path, dtype="float64", always_2d=True)
    if samples.shape[0] == 0:
        raise ValueError(f"Audio input is empty: {path}")
    mono = samples.mean(axis=1)
    if source_rate != sample_rate:
        divisor = int(np.gcd(source_rate, sample_rate))
        mono = resample_poly(mono, sample_rate // divisor, source_rate // divisor)
    return np.asarray(mono, dtype=np.float64)


def write_wav(path: str | Path, samples: Iterable[float], sample_rate: int = SAMPLE_RATE) -> None:
    """Write mono, 16-bit PCM WAV audio via libsndfile."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio output must be a non-empty mono signal")
    sf.write(destination, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16", format="WAV")


def _speech_intervals(samples: np.ndarray) -> list[tuple[int, int]]:
    if samples.size == 0 or float(np.max(np.abs(samples))) < 1e-5:
        return []
    intervals = librosa.effects.split(
        samples,
        top_db=config.SILENCE_TOP_DB,
        frame_length=512,
        hop_length=128,
    )
    return [(int(start), int(end)) for start, end in intervals if end > start]


def _silence_intervals(samples: np.ndarray) -> list[tuple[int, int]]:
    speech = _speech_intervals(samples)
    minimum = round(config.MIN_PAUSE_SECONDS * SAMPLE_RATE)
    return [
        (left_end, right_start)
        for (_, left_end), (right_start, _) in zip(speech, speech[1:])
        if right_start - left_end >= minimum
    ]


def _choose_speech_region(
    samples: np.ndarray, rng: random.Random
) -> tuple[int, int] | None:
    intervals = _speech_intervals(samples)
    if not intervals:
        return None
    interval_start, interval_end = rng.choice(intervals)
    interval_length = interval_end - interval_start
    segment_length = max(1, round(interval_length * 0.65))
    segment_length = min(segment_length, interval_length)
    start = rng.randint(interval_start, interval_end - segment_length)
    return start, start + segment_length


def _mapped_position(
    position: int,
    start: int,
    end: int,
    replacement_length: int,
    local_map,
    is_label_start: bool,
) -> int:
    if start == end:
        if position > start or (position == start and is_label_start):
            return position + replacement_length
        return position
    if position < start:
        return position
    if position > end:
        return position + replacement_length - (end - start)
    return start + local_map(position - start)


def _splice(
    samples: np.ndarray,
    labels: list[dict[str, Any]],
    start: int,
    end: int,
    replacement: np.ndarray,
    local_map,
) -> np.ndarray:
    replacement = np.asarray(replacement, dtype=np.float64)
    for label in labels:
        label_start = round(label["start_s"] * SAMPLE_RATE)
        label_end = round(label["end_s"] * SAMPLE_RATE)
        label_start = _mapped_position(
            label_start, start, end, len(replacement), local_map, True
        )
        label_end = _mapped_position(
            label_end, start, end, len(replacement), local_map, False
        )
        if label_end <= label_start:
            label_end = label_start + 1
        label["start_s"] = label_start / SAMPLE_RATE
        label["end_s"] = label_end / SAMPLE_RATE
    return np.concatenate((samples[:start], replacement, samples[end:]))


def _add_label(
    labels: list[dict[str, Any]], flaw_type: str, start: int, end: int, magnitude: float
) -> None:
    if end <= start:
        end = start + 1
    labels.append(
        {
            "type": flaw_type,
            "start_s": start / SAMPLE_RATE,
            "end_s": end / SAMPLE_RATE,
            "magnitude": float(magnitude),
        }
    )


def _time_stretch(
    samples: np.ndarray,
    labels: list[dict[str, Any]],
    flaw_type: str,
    level: int,
    rng: random.Random,
) -> np.ndarray:
    region = _choose_speech_region(samples, rng)
    if region is None:
        return samples
    start, end = region
    segment = samples[start:end]
    if len(segment) < round(SAMPLE_RATE * 0.12):
        return samples
    rate = (
        config.TOO_FAST_RATES[level - 1]
        if flaw_type == "too_fast"
        else config.TOO_SLOW_RATES[level - 1]
    )
    replacement = librosa.effects.time_stretch(segment, rate=rate)
    local_map = lambda position: round(position * len(replacement) / len(segment))
    result = _splice(samples, labels, start, end, replacement, local_map)
    _add_label(labels, flaw_type, start, start + len(replacement), abs(1.0 - rate))
    return result


def _monotone(
    samples: np.ndarray,
    labels: list[dict[str, Any]],
    level: int,
    rng: random.Random,
) -> np.ndarray:
    region = _choose_speech_region(samples, rng)
    if region is None:
        return samples
    start, end = region
    segment = np.ascontiguousarray(samples[start:end], dtype=np.float64)
    if len(segment) < round(SAMPLE_RATE * 0.25):
        return samples
    f0, times = pyworld.harvest(segment, SAMPLE_RATE, f0_floor=65.0, f0_ceil=500.0)
    voiced = f0 > 0
    if np.count_nonzero(voiced) < 3:
        return samples
    alpha = config.MONOTONE_F0_ALPHA[level - 1]
    median_f0 = float(np.median(f0[voiced]))
    modified_f0 = f0.copy()
    modified_f0[voiced] = median_f0 + alpha * (f0[voiced] - median_f0)
    spectrum = pyworld.cheaptrick(segment, f0, times, SAMPLE_RATE)
    aperiodicity = pyworld.d4c(segment, f0, times, SAMPLE_RATE)
    replacement = pyworld.synthesize(modified_f0, spectrum, aperiodicity, SAMPLE_RATE)
    if len(replacement) < len(segment):
        replacement = np.pad(replacement, (0, len(segment) - len(replacement)))
    else:
        replacement = replacement[: len(segment)]
    result = _splice(
        samples, labels, start, end, replacement, lambda position: position
    )
    _add_label(labels, "monotone", start, end, 1.0 - alpha)
    return result


def _remove_pauses(
    samples: np.ndarray,
    labels: list[dict[str, Any]],
    level: int,
) -> np.ndarray:
    pauses = _silence_intervals(samples)
    if not pauses:
        return samples
    maximum_count = config.MISSING_PAUSE_COUNTS[level - 1]
    fraction = config.MISSING_PAUSE_FRACTIONS[level - 1]
    selected = sorted(pauses, key=lambda pair: pair[1] - pair[0], reverse=True)[:maximum_count]
    for pause_start, pause_end in sorted(selected, reverse=True):
        pause_length = pause_end - pause_start
        removed_length = min(pause_length, max(1, round(pause_length * fraction)))
        cut_start = pause_start + (pause_length - removed_length) // 2
        cut_end = cut_start + removed_length
        old_local_length = cut_end - cut_start

        samples = _splice(
            samples,
            labels,
            cut_start,
            cut_end,
            np.empty(0, dtype=np.float64),
            lambda _position: 0,
        )
        seam = min(cut_start, len(samples) - 1)
        context = max(1, round(0.01 * SAMPLE_RATE))
        _add_label(
            labels,
            "missing_pauses",
            max(0, seam - context // 2),
            min(len(samples), seam + context // 2),
            old_local_length / SAMPLE_RATE,
        )
    return samples


def _insert_pauses(
    samples: np.ndarray,
    labels: list[dict[str, Any]],
    level: int,
    rng: random.Random,
) -> np.ndarray:
    speech = _speech_intervals(samples)
    if not speech:
        return samples
    count = min(config.EXCESS_PAUSE_COUNTS[level - 1], len(speech))
    boundaries = [end for _, end in speech]
    selected = sorted(rng.sample(boundaries, count), reverse=True)
    pause_length = round(config.EXCESS_PAUSE_SECONDS[level - 1] * SAMPLE_RATE)
    for position in selected:
        silence = np.zeros(pause_length, dtype=np.float64)
        samples = _splice(
            samples, labels, position, position, silence, lambda offset: offset
        )
        _add_label(labels, "excess_pauses", position, position + pause_length, pause_length / SAMPLE_RATE)
    return samples


def _volume_drop(
    samples: np.ndarray,
    labels: list[dict[str, Any]],
    level: int,
    rng: random.Random,
) -> np.ndarray:
    region = _choose_speech_region(samples, rng)
    if region is None:
        return samples
    start, end = region
    length = end - start
    gain = config.VOLUME_GAINS[level - 1]
    fade = min(round(config.VOLUME_FADE_SECONDS * SAMPLE_RATE), length // 2)
    envelope = np.full(length, gain, dtype=np.float64)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
        envelope[:fade] = 1.0 - (1.0 - gain) * ramp
        envelope[-fade:] = gain + (1.0 - gain) * ramp[::-1]
    replacement = samples[start:end] * envelope
    result = _splice(
        samples, labels, start, end, replacement, lambda position: position
    )
    _add_label(labels, "volume_drop", start, end, 1.0 - gain)
    return result


def _mumbling(
    samples: np.ndarray,
    labels: list[dict[str, Any]],
    level: int,
    rng: random.Random,
) -> np.ndarray:
    region = _choose_speech_region(samples, rng)
    if region is None:
        return samples
    start, end = region
    segment = samples[start:end]
    nyquist = SAMPLE_RATE / 2.0
    cutoff = min(config.MUMBLING_CUTOFF_HZ[level - 1], nyquist * 0.95)
    sos = butter(6, cutoff, btype="lowpass", fs=SAMPLE_RATE, output="sos")
    pad_length = 3 * (2 * len(sos) + 1)
    replacement = (
        sosfiltfilt(sos, segment, padlen=min(pad_length, len(segment) - 1))
        if len(segment) > 1
        else sosfilt(sos, segment)
    )
    result = _splice(
        samples, labels, start, end, replacement, lambda position: position
    )
    _add_label(labels, "mumbling", start, end, 1.0 - cutoff / nyquist)
    return result


def inject_flaws(
    samples: Iterable[float],
    flaw_types: Iterable[str],
    level: int,
    seed: int = 42,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Inject flaws and return output-timeline labels; unavailable pause flaws are skipped."""
    if level not in range(5):
        raise ValueError("level must be between 0 and 4")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"inject_flaws requires {SAMPLE_RATE} Hz input; normalize audio first")
    output = np.asarray(list(samples), dtype=np.float64)
    if output.ndim != 1:
        raise ValueError("inject_flaws expects mono audio")
    selected_flaws = list(flaw_types)
    unknown = set(selected_flaws) - set(FLAW_TYPES)
    if unknown:
        raise ValueError(f"Unknown flaw type(s): {', '.join(sorted(unknown))}")
    if len(set(selected_flaws)) != len(selected_flaws):
        raise ValueError("flaw_types cannot contain duplicates")
    if level == 0 or output.size == 0:
        return output.copy(), []

    rng = random.Random(seed)
    labels: list[dict[str, Any]] = []
    for flaw_type in _PROCESS_ORDER:
        if flaw_type not in selected_flaws:
            continue
        if flaw_type in {"too_fast", "too_slow"}:
            output = _time_stretch(output, labels, flaw_type, level, rng)
        elif flaw_type == "monotone":
            output = _monotone(output, labels, level, rng)
        elif flaw_type == "missing_pauses":
            output = _remove_pauses(output, labels, level)
        elif flaw_type == "volume_drop":
            output = _volume_drop(output, labels, level, rng)
        elif flaw_type == "mumbling":
            output = _mumbling(output, labels, level, rng)
        elif flaw_type == "excess_pauses":
            output = _insert_pauses(output, labels, level, rng)

    duration = len(output) / SAMPLE_RATE
    for label in labels:
        label["start_s"] = max(0.0, label["start_s"])
        label["end_s"] = min(duration, label["end_s"])
    return output, sorted(labels, key=lambda label: (label["start_s"], label["end_s"]))


def write_labels(
    path: str | Path, speech_id: str, level: int, flaws: list[dict[str, Any]]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"speech_id": speech_id, "level": level, "flaws": flaws}, indent=2) + "\n",
        encoding="utf-8",
    )
