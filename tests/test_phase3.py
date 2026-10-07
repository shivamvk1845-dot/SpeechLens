import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speechlens.compare import ComparisonResult, compare_features
from speechlens.detect import detect_regions
from speechlens.explain import explain_region, explain_regions
from speechlens.features import AcousticFeatures
from speechlens.score import score_comparison


def features(
    *,
    duration_s=4.0,
    frame_hop_s=0.01,
    f0_hz=200.0,
    rms_db=-20.0,
    pitch_variation=0.0,
    centroid_hz=1_500.0,
    flatness=0.1,
    hnr_db=20.0,
    words=None,
    pauses=None,
    mfcc_offset=0.0,
):
    count = round(duration_s / frame_hop_s)
    times = np.arange(count) * frame_hop_s
    f0 = f0_hz + pitch_variation * np.sin(2 * np.pi * 0.5 * times)
    return AcousticFeatures(
        times_s=times,
        f0_hz=f0,
        rms_db=np.full(count, rms_db),
        mfcc=np.vstack(
            [np.sin(2 * np.pi * (index + 1) * times / duration_s) + mfcc_offset for index in range(13)]
        ),
        spectral_centroid_hz=np.full(count, centroid_hz),
        spectral_flatness=np.full(count, flatness),
        hnr_db=np.full(count, hnr_db),
        voiced=np.ones(count, dtype=bool),
        word_features=list(words or []),
        pause_intervals=list(pauses or []),
    )


def window(
    flaw_type,
    metric,
    start,
    end,
    z,
    delta,
    baseline,
    participant,
):
    metrics = {
        name: {
            "baseline": 0.0,
            "participant": 0.0,
            "delta": 0.0,
            "z": 0.0,
        }
        for name in (
            "pitch_mean_semitones",
            "pitch_std_semitones",
            "energy_db",
            "hnr_db",
            "spectral_centroid_hz",
            "spectral_flatness",
            "speech_rate_wps",
            "pause_ratio",
        )
    }
    metrics[metric] = {
        "baseline": baseline,
        "participant": participant,
        "delta": delta,
        "z": z,
    }
    return {
        "start_s": start,
        "end_s": end,
        "participant_start_s": start,
        "participant_end_s": end,
        "deltas": metrics,
    }


def test_dtw_comparison_produces_temporal_windows_and_word_deltas():
    base_words = [
        {"word": "one", "start": 0.2, "end": 0.5, "duration_s": 0.3, "speech_rate_wps": 3.0},
        {"word": "two", "start": 0.8, "end": 1.1, "duration_s": 0.3, "speech_rate_wps": 3.0},
    ]
    participant_words = [
        {"word": "one", "start": 0.15, "end": 0.4, "duration_s": 0.25, "speech_rate_wps": 4.0},
        {"word": "two", "start": 0.65, "end": 0.9, "duration_s": 0.25, "speech_rate_wps": 4.0},
    ]
    baseline = features(duration_s=2, words=base_words)
    participant = features(duration_s=1.5, words=participant_words, mfcc_offset=0.01)

    result = compare_features(baseline, participant, window_seconds=0.5, hop_seconds=0.25)

    assert result.dtw_path
    assert all(len(pair) == 2 for pair in result.dtw_path)
    assert result.windows
    assert result.word_deltas[0]["word"] == "one"
    assert result.word_deltas[0]["deltas"]["speech_rate_wps"]["delta"] == pytest.approx(1.0)
    assert result.word_deltas[0]["deltas"]["energy_db"]["delta"] == pytest.approx(0.0)
    assert all(window["end_s"] > window["start_s"] for window in result.windows)


def test_detection_threshold_directional_rules_and_temporal_merging():
    comparison = ComparisonResult(
        dtw_path=[],
        windows=[
            window("too_fast", "speech_rate_wps", 0.0, 1.0, 2.5, 2.5, 2.0, 4.5),
            window("too_fast", "speech_rate_wps", 0.75, 1.75, 3.0, 3.0, 2.0, 5.0),
            window("too_slow", "speech_rate_wps", 2.0, 3.0, -2.2, -2.2, 4.0, 1.8),
            window("volume_drop", "energy_db", 3.0, 4.0, -2.5, -10.0, -15.0, -25.0),
        ],
        word_deltas=[],
    )

    regions = detect_regions(comparison, threshold_z=2.0, merge_gap_seconds=0.25)

    fast = next(region for region in regions if region["flaw_type"] == "too_fast")
    assert fast["start_s"] == 0.0
    assert fast["end_s"] == 1.75
    assert fast["severity"] == 3.0
    assert len(fast["evidence"]) == 2
    assert {region["flaw_type"] for region in regions} == {
        "too_fast",
        "too_slow",
        "volume_drop",
    }
    assert detect_regions(comparison, threshold_z=4.0) == []


def test_comparison_detection_explanation_and_score_work_end_to_end():
    baseline = features(duration_s=3.0)
    participant = features(duration_s=3.0, rms_db=-32.0)

    comparison = compare_features(baseline, participant)
    regions = detect_regions(comparison)
    explanations = explain_regions(regions)
    score = score_comparison(comparison)

    assert comparison.dtw_path
    assert regions and all(region["flaw_type"] == "volume_drop" for region in regions)
    assert len(explanations) == 1
    assert "12.0 dB below baseline" in explanations[0]["explanation"]
    assert score["categories"]["volume"]["score"] < 100
    assert score["total"] < 100


def test_detection_infers_pause_monotone_and_mumbling():
    comparison = ComparisonResult(
        dtw_path=[],
        windows=[
            window("monotone", "pitch_std_semitones", 0, 1, -2.2, -3.3, 4.0, 0.7),
            window("missing", "pause_ratio", 1, 2, -3.0, -0.45, 0.60, 0.15),
            window("excess", "pause_ratio", 2, 3, 2.5, 0.375, 0.15, 0.525),
            {
                **window("mumbling", "hnr_db", 3, 4, -2.5, -10, 20, 10),
                "deltas": {
                    **window("mumbling", "hnr_db", 3, 4, -2.5, -10, 20, 10)["deltas"],
                    "spectral_flatness": {
                        "baseline": 0.1,
                        "participant": 0.4,
                        "delta": 0.3,
                        "z": 3.0,
                    },
                    "spectral_centroid_hz": {
                        "baseline": 1_500,
                        "participant": 0,
                        "delta": -1_500,
                        "z": -3.0,
                    },
                },
            },
        ],
        word_deltas=[],
    )

    regions = detect_regions(comparison)

    assert {region["flaw_type"] for region in regions} == {
        "monotone",
        "missing_pauses",
        "excess_pauses",
        "mumbling",
    }


def test_numeric_explanations_include_region_and_measurements():
    region = {
        "flaw_type": "volume_drop",
        "start_s": 1.25,
        "end_s": 2.5,
        "severity": 2.5,
        "evidence": [
            {
                "metric": "energy_db",
                "baseline": -15.0,
                "participant": -25.0,
                "delta": -10.0,
                "z": -2.5,
            }
        ],
    }

    explanation = explain_region(region)

    assert explanation["start_s"] == 1.25
    assert explanation["participant_start_s"] == 1.25
    assert explanation["participant_end_s"] == 2.5
    assert "10.0 dB below baseline" in explanation["explanation"]
    assert "-25.0 vs -15.0 dB" in explanation["explanation"]
    assert explain_regions([region]) == [explanation]


def test_weighted_rubric_scores_are_bounded_and_penalize_deviations():
    clean = ComparisonResult(
        dtw_path=[],
        windows=[
            window("clean", "energy_db", 0, 1, 0, 0, -20, -20),
        ],
        word_deltas=[],
    )
    degraded = ComparisonResult(
        dtw_path=[],
        windows=[
            window("fast", "speech_rate_wps", 0, 1, 3, 3, 2, 5),
            window("drop", "energy_db", 1, 2, -3, -12, -15, -27),
        ],
        word_deltas=[],
    )

    clean_score = score_comparison(clean)
    degraded_score = score_comparison(degraded)

    assert clean_score["total"] == 100.0
    assert 0 <= degraded_score["total"] < clean_score["total"]
    assert degraded_score["categories"]["pace"]["score"] < 100
    assert degraded_score["categories"]["volume"]["score"] < 100
    assert sum(degraded_score["weights"].values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="weights"):
        score_comparison(clean, {"pace": 1.0})


def test_score_decreases_as_single_metric_deviation_increases():
    results = []
    for z in (0, 1, 2, 3, 4):
        result = ComparisonResult(
            dtw_path=[],
            windows=[window("pace", "speech_rate_wps", 0, 1, z, z, 3, 3 + z)],
            word_deltas=[],
        )
        results.append(score_comparison(result)["total"])

    assert results == sorted(results, reverse=True)
