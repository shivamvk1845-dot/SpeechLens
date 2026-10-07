import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from streamlit_app import (
    _comparison_figure,
    _decode_audio,
    _frame_envelope,
    _highlighted_transcript,
    _region_csv,
    _score_card_items,
    _warped_participant_series,
    _warped_word_rate,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speechlens.compare import ComparisonResult, compare_features
from speechlens.detect import detect_regions
from speechlens.features import AcousticFeatures, extract_features
from speechlens.score import score_comparison


def test_upload_decoder_downmixes_and_resamples_wav(tmp_path):
    path = tmp_path / "stereo.wav"
    source_rate = 8_000
    signal = np.sin(2 * np.pi * 220 * np.arange(source_rate) / source_rate) * 0.5
    sf.write(path, np.column_stack((signal, signal)), source_rate)

    decoded = _decode_audio(path.read_bytes(), path.name)

    assert decoded.ndim == 1
    assert len(decoded) == 2 * source_rate
    assert np.max(np.abs(decoded)) == pytest.approx(0.5, abs=0.01)


def test_upload_decoder_rejects_empty_content():
    with pytest.raises(ValueError, match="empty"):
        _decode_audio(b"", "empty.wav")


def test_frame_envelope_and_dtw_warped_series():
    low, high = _frame_envelope(np.array([-1, 0.5, 0.25, -0.5]), 2)
    np.testing.assert_array_equal(low, [-1, -0.5])
    np.testing.assert_array_equal(high, [0.5, 0.25])
    comparison = ComparisonResult(
        dtw_path=[(0, 0), (0, 1), (1, 2)],
        windows=[],
        word_deltas=[],
    )

    values = _warped_participant_series(comparison, np.array([1, 3, 5]), 2)

    np.testing.assert_array_equal(values, [2, 5])


def test_score_card_items_preserve_metric_labels_and_values():
    score = score_comparison(ComparisonResult(dtw_path=[], windows=[], word_deltas=[]))

    assert _score_card_items(score) == [
        ("Overall score", "100/100"),
        ("Pace", "100/100"),
        ("Pitch", "100/100"),
        ("Volume", "100/100"),
        ("Pauses", "100/100"),
        ("Clarity", "100/100"),
    ]


def test_warped_word_rate_interpolates_over_aligned_speech():
    times = np.arange(101, dtype=float) / 100
    features = AcousticFeatures(
        times_s=times,
        f0_hz=np.full(len(times), 150.0),
        rms_db=np.full(len(times), -20.0),
        mfcc=np.zeros((2, len(times))),
        spectral_centroid_hz=np.full(len(times), 1000.0),
        spectral_flatness=np.full(len(times), 0.1),
        hnr_db=np.full(len(times), 10.0),
        voiced=np.ones(len(times), dtype=bool),
        word_features=[
            {"word": "one", "start": 0.2, "end": 0.4, "speech_rate_wps": 2.0},
            {"word": "two", "start": 0.6, "end": 0.8, "speech_rate_wps": 4.0},
        ],
        pause_intervals=[],
    )

    rates = _warped_word_rate(features)

    assert np.isnan(rates[times < 0.2]).all()
    assert np.isnan(rates[times > 0.8]).all()
    assert np.isfinite(rates[(times >= 0.2) & (times <= 0.8)]).all()
    assert np.count_nonzero(np.isfinite(rates)) > 40
    assert rates[np.argmin(np.abs(times - 0.3))] == pytest.approx(2.0)
    assert rates[np.argmin(np.abs(times - 0.5))] == pytest.approx(3.0)
    assert rates[np.argmin(np.abs(times - 0.7))] == pytest.approx(4.0)


def test_identical_baseline_and_participant_audio_have_no_detections():
    sample_rate = 16_000
    times = np.arange(sample_rate, dtype=float) / sample_rate
    audio = 0.4 * np.sin(2 * np.pi * 220 * times)
    alignments = [
        {"word": "same", "start": 0.1, "end": 0.5},
        {"word": "audio", "start": 0.6, "end": 0.9},
    ]
    baseline = extract_features(audio, word_alignments=alignments)
    participant = extract_features(audio.copy(), word_alignments=alignments)

    comparison = compare_features(baseline, participant)

    assert comparison.dtw_path
    assert detect_regions(comparison) == []


def test_comparison_figure_overlays_features_and_marks_regions():
    def features(times, frame_count):
        return AcousticFeatures(
            times_s=np.asarray(times, dtype=float),
            f0_hz=np.full(frame_count, 150.0),
            rms_db=np.full(frame_count, -20.0),
            mfcc=np.zeros((2, frame_count)),
            spectral_centroid_hz=np.full(frame_count, 1000.0),
            spectral_flatness=np.full(frame_count, 0.1),
            hnr_db=np.full(frame_count, 10.0),
            voiced=np.ones(frame_count, dtype=bool),
            word_features=[
                {
                    "word": "hello",
                    "start": 0.0,
                    "end": 0.03,
                    "speech_rate_wps": 5.0,
                }
            ],
            pause_intervals=[],
        )

    baseline = features([0.0, 0.01, 0.02], 3)
    participant = features([0.0, 0.01, 0.02, 0.03], 4)
    comparison = ComparisonResult(
        dtw_path=[(0, 0), (1, 1), (1, 2), (2, 3)],
        windows=[],
        word_deltas=[],
    )

    figure = _comparison_figure(
        np.array([0.1, -0.1, 0.2, -0.2, 0.1, -0.1]),
        np.array([0.2, -0.2, 0.1, -0.1, 0.3, -0.3, 0.2, -0.2]),
        baseline,
        participant,
        comparison,
        [{"flaw_type": "too_fast", "start_s": 0.01, "end_s": 0.02}],
    )

    assert len(figure.data) == 11
    assert figure.data[-1].customdata[0][0] == "too_fast"
    assert len(figure.layout.shapes) == 4


def test_transcript_highlights_words_overlapping_detected_participant_regions():
    alignments = [
        {"word": "hello", "start": 0.1, "end": 0.3},
        {"word": "world", "start": 0.4, "end": 0.7},
    ]
    regions = [
        {
            "flaw_type": "volume_drop",
            "participant_start_s": 0.35,
            "participant_end_s": 0.8,
        }
    ]

    rendered = _highlighted_transcript(alignments, regions)

    assert "hello" in rendered
    assert "<mark" in rendered
    assert "world</mark>" in rendered
    assert "volume_drop" in rendered


def test_region_csv_exports_numeric_explanations():
    content = _region_csv(
        [
            {
                "flaw_type": "too_fast",
                "start_s": 0.5,
                "end_s": 1.2,
                "severity_z": 2.4,
                "explanation": "Speech was 30% faster.",
            }
        ]
    ).decode("utf-8")

    assert "flaw_type,start_s,end_s,severity_z,explanation" in content
    assert "too_fast,0.5,1.2,2.4,Speech was 30% faster." in content


def test_dashboard_renders_upload_prompt_without_uploads():
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

    app = AppTest.from_file(str(app_path)).run()

    assert app.title[0].value == "SpeechLens"
    assert "Upload participant audio" in app.info[0].value
