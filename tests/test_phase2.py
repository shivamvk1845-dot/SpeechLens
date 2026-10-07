import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speechlens.config import FEATURE_HOP_SECONDS, SAMPLE_RATE
from speechlens.align import align_transcript
from speechlens.features import AcousticFeatures, extract_features
from speechlens.inject import write_wav
from speechlens.normalize import fit_speaker_stats, normalize_features


def speech(duration_s=1.2, sample_rate=SAMPLE_RATE):
    times = np.arange(round(duration_s * sample_rate)) / sample_rate
    f0 = 205.0 + 18.0 * np.sin(2 * np.pi * 0.8 * times)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    signal = sum(np.sin(harmonic * phase) / harmonic for harmonic in range(1, 5))
    return (0.5 * signal / np.max(np.abs(signal))).astype(np.float64)


def synthetic_features(f0_hz, rms_db):
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    rms_db = np.asarray(rms_db, dtype=np.float64)
    frame_count = len(f0_hz)
    voiced = np.isfinite(f0_hz) & (f0_hz > 0)
    return AcousticFeatures(
        times_s=np.arange(frame_count) * FEATURE_HOP_SECONDS,
        f0_hz=f0_hz,
        rms_db=rms_db,
        mfcc=np.tile(np.arange(frame_count, dtype=np.float64), (13, 1)),
        spectral_centroid_hz=np.linspace(500, 1_000, frame_count),
        spectral_flatness=np.linspace(0.1, 0.4, frame_count),
        hnr_db=np.linspace(5, 20, frame_count),
        voiced=voiced,
        word_features=[],
        pause_intervals=[],
    )


def test_frame_feature_shapes_and_voiced_pitch():
    audio = np.concatenate((np.zeros(round(0.2 * SAMPLE_RATE)), speech(), np.zeros(round(0.1 * SAMPLE_RATE))))
    aligned_words = [
        {"word": "hello", "start": 0.20, "end": 0.52},
        {"word": "world", "start": 0.75, "end": 1.05},
    ]

    features = extract_features(audio, word_alignments=aligned_words)

    expected_frames = 1 + len(audio) // round(FEATURE_HOP_SECONDS * SAMPLE_RATE)
    assert features.times_s.shape == (expected_frames,)
    assert features.f0_hz.shape == features.rms_db.shape == features.voiced.shape
    assert features.mfcc.shape == (13, expected_frames)
    assert features.spectral_centroid_hz.shape == (expected_frames,)
    assert features.spectral_flatness.shape == (expected_frames,)
    assert features.hnr_db.shape == (expected_frames,)
    assert np.nanmedian(features.f0_hz[features.voiced]) == pytest.approx(205, abs=20)
    assert np.all(np.diff(features.times_s) == pytest.approx(FEATURE_HOP_SECONDS))
    assert features.word_features[0]["speech_rate_wps"] > 0
    assert features.pause_intervals == [
        {"start": 0.52, "end": 0.75, "duration_s": pytest.approx(0.23)}
    ]


def test_silence_features_are_stable_and_resampling_downmix_work():
    silence = np.zeros(SAMPLE_RATE // 4)
    silent_features = extract_features(silence)
    assert not np.any(silent_features.voiced)
    assert np.all(np.isnan(silent_features.f0_hz))
    assert np.all(np.isfinite(silent_features.rms_db))
    assert np.all(silent_features.rms_db <= -100)

    tone = speech(duration_s=0.5, sample_rate=8_000)
    stereo = np.column_stack((tone, tone))
    resampled = extract_features(stereo, sample_rate=8_000)
    assert len(resampled.times_s) == 1 + round(0.5 * SAMPLE_RATE) // 160
    assert np.nanmedian(resampled.f0_hz[resampled.voiced]) == pytest.approx(205, abs=25)


def test_speaker_normalization_uses_median_f0_energy_and_z_scores():
    features = synthetic_features([200, 220, 240, np.nan], [-20, -10, -15, -30])

    statistics = fit_speaker_stats({"speaker-a": [features, features]})["speaker-a"]
    normalized = normalize_features(features, statistics)

    assert statistics.median_f0_hz == pytest.approx(220)
    assert statistics.median_energy_db == pytest.approx(-17.5)
    assert normalized["f0_semitones"][1] == pytest.approx(0)
    assert normalized["f0_semitones"][3] != normalized["f0_semitones"][3]
    assert normalized["energy_relative_db"] == pytest.approx([-2.5, 7.5, 2.5, -12.5])
    assert np.nanmean(normalized["z_scores"]["f0_semitones"]) == pytest.approx(0, abs=1e-12)
    assert np.mean(normalized["z_scores"]["energy_relative_db"]) == pytest.approx(0, abs=1e-12)
    assert normalized["z_scores"]["mfcc_01"].shape == (4,)


def test_speaker_stats_reject_missing_voiced_f0():
    silent = synthetic_features([np.nan, np.nan], [-100, -100])

    with pytest.raises(ValueError, match="no voiced F0"):
        fit_speaker_stats({"silent-speaker": [silent]})


def test_alignment_uses_whisperx_and_caches_word_timings(tmp_path, monkeypatch):
    source = tmp_path / "audio.wav"
    write_wav(source, speech())
    calls = {"load": 0, "align": 0}

    def load_align_model(language_code, device):
        calls["load"] += 1
        assert language_code == "en"
        return "model", {"language": language_code}

    def align(result, model, metadata, audio, device, return_char_alignments):
        calls["align"] += 1
        assert model == "model"
        assert metadata == {"language": "en"}
        assert len(audio) == len(speech())
        assert audio.dtype == np.float32
        assert result[0]["text"] == "hello world"
        assert return_char_alignments is False
        return {
            "word_segments": [
                {"word": "hello", "start": 0.1, "end": 0.4},
                {"word": "world", "start": 0.5, "end": 0.9},
            ]
        }

    monkeypatch.setitem(
        sys.modules,
        "whisperx",
        SimpleNamespace(load_align_model=load_align_model, align=align),
    )
    cache = tmp_path / "cache"

    first = align_transcript(source, "hello world", cache)
    monkeypatch.delitem(sys.modules, "whisperx", raising=False)
    second = align_transcript(source, "hello world", cache)

    assert first == second == [
        {"word": "hello", "start": 0.1, "end": 0.4},
        {"word": "world", "start": 0.5, "end": 0.9},
    ]
    assert calls == {"load": 1, "align": 1}
    cache_files = list(cache.glob("*.json"))
    assert len(cache_files) == 1
    assert json.loads(cache_files[0].read_text(encoding="utf-8"))["words"] == first


def test_alignment_cache_key_changes_with_transcript(tmp_path, monkeypatch):
    source = tmp_path / "audio.wav"
    write_wav(source, speech())
    calls = []

    def align(result, *_args, **_kwargs):
        calls.append(result[0]["text"])
        return {
            "word_segments": [
                {"word": result[0]["text"], "start": 0.1, "end": 0.4}
            ]
        }

    monkeypatch.setitem(
        sys.modules,
        "whisperx",
        SimpleNamespace(
            load_align_model=lambda **_kwargs: ("model", {}),
            align=align,
        ),
    )

    align_transcript(source, "first", tmp_path / "cache")
    align_transcript(source, "second", tmp_path / "cache")

    assert calls == ["first", "second"]


def test_alignment_rejects_unaligned_words_and_empty_transcript(tmp_path, monkeypatch):
    source = tmp_path / "audio.wav"
    write_wav(source, speech())
    monkeypatch.setitem(
        sys.modules,
        "whisperx",
        SimpleNamespace(
            load_align_model=lambda **_kwargs: ("model", {}),
            align=lambda *_args, **_kwargs: {
                "word_segments": [{"word": "bad", "start": None, "end": None}]
            },
        ),
    )

    with pytest.raises(ValueError, match="Transcript"):
        align_transcript(source, "  ", tmp_path / "cache")
    with pytest.raises(ValueError, match="unaligned"):
        align_transcript(source, "bad", tmp_path / "cache")


def test_feature_extraction_rejects_word_alignment_outside_audio():
    with pytest.raises(ValueError, match="Invalid word alignment"):
        extract_features(speech(0.5), word_alignments=[{"word": "late", "start": 0.4, "end": 0.6}])
