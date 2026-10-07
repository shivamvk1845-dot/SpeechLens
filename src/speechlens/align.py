"""WhisperX forced alignment with deterministic on-disk result caching."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .inject import read_wav

ALIGNER_VERSION = "whisperx-3.8.6"


def _cache_key(audio_path: Path, transcript: str, language_code: str, device: str) -> str:
    digest = hashlib.sha256()
    with audio_path.open("rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\0")
    digest.update(transcript.encode("utf-8"))
    digest.update(b"\0")
    digest.update(language_code.encode("utf-8"))
    digest.update(b"\0")
    digest.update(device.encode("utf-8"))
    digest.update(b"\0")
    digest.update(ALIGNER_VERSION.encode("ascii"))
    return digest.hexdigest()


def _validate_words(words: Any, duration_s: float) -> list[dict[str, Any]]:
    if not isinstance(words, list):
        raise ValueError("WhisperX returned no word_segments list")
    validated = []
    previous_start = -1.0
    for item in words:
        if not isinstance(item, dict) or item.get("word") is None:
            raise ValueError("WhisperX returned a word without a token")
        try:
            word = str(item["word"])
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"WhisperX returned an unaligned word: {item!r}") from error
        if (
            not word.strip()
            or start < 0
            or end <= start
            or end > duration_s + config.ALIGNMENT_TIMESTAMP_TOLERANCE_S
        ):
            raise ValueError(f"WhisperX returned out-of-range word timing: {item!r}")
        if start < previous_start:
            raise ValueError("WhisperX word timings are not sorted by start time")
        previous_start = start
        validated.append({"word": word, "start": start, "end": end})
    if not validated:
        raise ValueError("WhisperX did not align any words")
    return validated


def align_transcript(
    audio_path: str | Path,
    transcript: str,
    cache_dir: str | Path,
    language_code: str = config.ALIGNMENT_LANGUAGE,
    device: str = config.ALIGNMENT_DEVICE,
) -> list[dict[str, Any]]:
    """Return cached or WhisperX forced word timings for an audio transcript.

    Cache identity includes audio bytes, transcript text, language, device, and the
    pinned aligner version. The aligner receives mono audio resampled to 16 kHz.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if not transcript.strip():
        raise ValueError("Transcript must not be empty")
    if not language_code:
        raise ValueError("language_code must not be empty")

    key = _cache_key(path, transcript, language_code, device)
    cache_path = Path(cache_dir) / f"{key}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") != key:
            raise ValueError(f"Alignment cache key mismatch: {cache_path}")
        audio = read_wav(path, config.SAMPLE_RATE)
        return _validate_words(cached.get("words"), len(audio) / config.SAMPLE_RATE)

    try:
        whisperx = importlib.import_module("whisperx")
    except ImportError as error:
        raise RuntimeError(
            "Word alignment requires the pinned whisperx dependency; install requirements.txt"
        ) from error

    audio = read_wav(path, config.SAMPLE_RATE)
    duration_s = len(audio) / config.SAMPLE_RATE
    audio = np.asarray(audio, dtype=np.float32)
    align_model, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device,
    )
    aligned = whisperx.align(
        [
            {
                "start": 0.0,
                "end": duration_s,
                "text": transcript,
            }
        ],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    words = _validate_words(aligned.get("word_segments"), duration_s)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=cache_path.parent,
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        json.dump({"cache_key": key, "words": words}, temporary_file, indent=2)
        temporary_file.write("\n")
    os.replace(temporary_path, cache_path)
    return words
