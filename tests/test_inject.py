import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speechlens.config import EXCESS_PAUSE_SECONDS, SAMPLE_RATE
from speechlens.inject import FLAW_TYPES, inject_flaws, read_wav, write_wav


def voiced_signal(duration_s=2.0, sample_rate=SAMPLE_RATE, f0=220.0, varying=False):
    times = np.arange(round(duration_s * sample_rate)) / sample_rate
    fundamental = (
        f0 + 35 * np.sin(2 * np.pi * 0.7 * times) if varying else np.full_like(times, f0)
    )
    phase = 2 * np.pi * np.cumsum(fundamental) / sample_rate
    signal = sum(np.sin(harmonic * phase) / harmonic for harmonic in range(1, 6))
    return (0.6 * signal / np.max(np.abs(signal))).astype(np.float64)


def separated_speech():
    segment = voiced_signal(0.55)
    silence_one = np.zeros(round(0.35 * SAMPLE_RATE))
    silence_two = np.zeros(round(0.45 * SAMPLE_RATE))
    return np.concatenate((segment, silence_one, segment, silence_two, segment))


def f0_values(samples):
    return librosa.yin(
        samples,
        fmin=120,
        fmax=350,
        sr=SAMPLE_RATE,
        frame_length=1024,
        hop_length=256,
    )


def test_level_zero_is_unchanged_and_has_no_labels():
    samples = voiced_signal()

    output, labels = inject_flaws(samples, FLAW_TYPES, level=0)

    np.testing.assert_array_equal(output, samples)
    assert labels == []


@pytest.mark.parametrize(
    ("flaw_type", "expected_direction"),
    [("too_fast", -1), ("too_slow", 1)],
)
def test_time_stretch_changes_duration_preserves_pitch_and_scales_with_severity(
    flaw_type, expected_direction
):
    samples = voiced_signal()
    outputs = [inject_flaws(samples, [flaw_type], level)[0] for level in range(1, 5)]
    durations = [len(output) / SAMPLE_RATE for output in outputs]
    changes = [abs(duration - len(samples) / SAMPLE_RATE) for duration in durations]

    assert all(
        (duration - len(samples) / SAMPLE_RATE) * expected_direction > 0
        for duration in durations
    )
    assert changes == sorted(changes)
    for output, labels in (inject_flaws(samples, [flaw_type], level) for level in range(1, 5)):
        label = labels[0]
        segment = output[round(label["start_s"] * SAMPLE_RATE) : round(label["end_s"] * SAMPLE_RATE)]
        measured_f0 = float(np.median(f0_values(segment)))
        assert measured_f0 == pytest.approx(220.0, abs=8.0)
        assert label["end_s"] <= len(output) / SAMPLE_RATE


def test_monotone_reduces_world_f0_variance_progressively_without_primary_rms_change():
    samples = voiced_signal(varying=True)
    measured = []
    rms_ratios = []
    for level in range(1, 5):
        output, labels = inject_flaws(samples, ["monotone"], level)
        label = labels[0]
        segment = np.ascontiguousarray(
            output[round(label["start_s"] * SAMPLE_RATE) : round(label["end_s"] * SAMPLE_RATE)]
        )
        f0, _ = pyworld.harvest(segment, SAMPLE_RATE, f0_floor=65.0, f0_ceil=500.0)
        measured.append(float(np.var(f0[f0 > 0])))
        source_region = samples[
            round(label["start_s"] * SAMPLE_RATE) : round(label["end_s"] * SAMPLE_RATE)
        ]
        rms_ratios.append(np.sqrt(np.mean(segment**2)) / np.sqrt(np.mean(source_region**2)))

    assert measured == sorted(measured, reverse=True)
    assert measured[-1] < measured[0] * 0.5
    assert all(0.75 < ratio < 1.25 for ratio in rms_ratios)


def test_missing_pauses_removes_real_silence_and_scales_removal():
    samples = separated_speech()
    level_one, labels_one = inject_flaws(samples, ["missing_pauses"], level=1)
    level_four, labels_four = inject_flaws(samples, ["missing_pauses"], level=4)
    original_active = np.count_nonzero(np.abs(samples) > 1e-5)

    assert labels_one and labels_four
    assert len(level_four) < len(level_one) < len(samples)
    assert np.count_nonzero(np.abs(level_one) > 1e-5) == original_active
    assert np.count_nonzero(np.abs(level_four) > 1e-5) == original_active
    assert sum(label["magnitude"] for label in labels_four) > sum(
        label["magnitude"] for label in labels_one
    )
    for output, labels in ((level_one, labels_one), (level_four, labels_four)):
        assert all(0 <= label["start_s"] < label["end_s"] <= len(output) / SAMPLE_RATE for label in labels)


def test_missing_pauses_does_not_fabricate_a_label_without_a_pause():
    output, labels = inject_flaws(voiced_signal(), ["missing_pauses"], level=4)

    assert len(output) == len(voiced_signal())
    assert labels == []


def test_excess_pauses_insert_exact_silence_at_speech_boundary_and_scale():
    samples = separated_speech()
    results = [inject_flaws(samples, ["excess_pauses"], level) for level in range(1, 5)]
    total_inserted = []

    for output, labels in results:
        inserted = 0
        for label in labels:
            assert label["type"] == "excess_pauses"
            start = round(label["start_s"] * SAMPLE_RATE)
            end = round(label["end_s"] * SAMPLE_RATE)
            assert np.all(output[start:end] == 0.0)
            assert end - start == round(label["magnitude"] * SAMPLE_RATE)
            inserted += end - start
        total_inserted.append(inserted)
    assert total_inserted == sorted(total_inserted)
    assert len(results[3][1]) > len(results[0][1])
    assert len(results[0][0]) - len(samples) == round(
        EXCESS_PAUSE_SECONDS[0] * SAMPLE_RATE
    )


def test_volume_drop_attenuates_more_at_higher_levels_with_smooth_boundaries():
    samples = voiced_signal()
    rms_ratios = []
    for level in range(1, 5):
        output, labels = inject_flaws(samples, ["volume_drop"], level)
        label = labels[0]
        start = round(label["start_s"] * SAMPLE_RATE)
        end = round(label["end_s"] * SAMPLE_RATE)
        ratio = np.sqrt(np.mean(output[start:end] ** 2)) / np.sqrt(
            np.mean(samples[start:end] ** 2)
        )
        rms_ratios.append(ratio)
        assert output[start] == pytest.approx(samples[start], abs=0.02)
        assert abs(output[start + (end - start) // 2]) < abs(
            samples[start + (end - start) // 2]
        )
        np.testing.assert_array_equal(output[:start], samples[:start])
        np.testing.assert_array_equal(output[end:], samples[end:])
    assert rms_ratios == sorted(rms_ratios, reverse=True)
    assert all(ratio < 1.0 for ratio in rms_ratios)


def test_mumbling_reduces_high_frequency_energy_monotonically():
    times = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
    samples = 0.3 * np.sin(2 * np.pi * 300 * times) + 0.3 * np.sin(
        2 * np.pi * 7_000 * times
    )
    high_frequency_energy = []
    for level in range(1, 5):
        output, labels = inject_flaws(samples, ["mumbling"], level)
        label = labels[0]
        start = round(label["start_s"] * SAMPLE_RATE)
        end = round(label["end_s"] * SAMPLE_RATE)
        segment = output[
            start:end
        ]
        spectrum = np.abs(np.fft.rfft(segment)) ** 2
        frequencies = np.fft.rfftfreq(len(segment), 1 / SAMPLE_RATE)
        high_frequency_energy.append(float(spectrum[frequencies > 5_000].sum()))
        assert np.all(np.isfinite(output))
        np.testing.assert_array_equal(output[:start], samples[:start])
        np.testing.assert_array_equal(output[end:], samples[end:])
    assert high_frequency_energy == sorted(high_frequency_energy, reverse=True)


def test_multiple_time_edits_remap_existing_labels_exactly():
    samples = separated_speech()
    stretched, original_labels = inject_flaws(samples, ["too_fast"], level=2, seed=42)
    combined, combined_labels = inject_flaws(
        samples, ["excess_pauses", "too_fast"], level=2, seed=42
    )
    original_stretch = original_labels[0]
    combined_stretch = next(label for label in combined_labels if label["type"] == "too_fast")
    inserted = [label for label in combined_labels if label["type"] == "excess_pauses"]
    shift_before_start = sum(
        label["end_s"] - label["start_s"]
        for label in inserted
        if label["start_s"] <= combined_stretch["start_s"]
    )
    shift_before_end = sum(
        label["end_s"] - label["start_s"]
        for label in inserted
        if label["start_s"] <= combined_stretch["end_s"]
    )

    assert combined_stretch["start_s"] == pytest.approx(
        original_stretch["start_s"] + shift_before_start
    )
    assert combined_stretch["end_s"] == pytest.approx(
        original_stretch["end_s"] + shift_before_end
    )
    assert len(combined) == len(stretched) + sum(
        round(label["magnitude"] * SAMPLE_RATE) for label in inserted
    )
    assert all(
        0 <= label["start_s"] < label["end_s"] <= len(combined) / SAMPLE_RATE
        for label in combined_labels
    )


def test_seeded_injection_is_reproducible():
    samples = separated_speech()

    first = inject_flaws(samples, FLAW_TYPES, level=3, seed=101)
    second = inject_flaws(samples, FLAW_TYPES, level=3, seed=101)

    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert {label["type"] for label in first[1]} == set(FLAW_TYPES)


def test_silence_only_and_very_short_audio_are_safe():
    silence = np.zeros(SAMPLE_RATE)
    silent_output, silent_labels = inject_flaws(silence, FLAW_TYPES, level=4)
    np.testing.assert_array_equal(silent_output, silence)
    assert silent_labels == []

    short = voiced_signal(0.04)
    short_output, short_labels = inject_flaws(short, ["too_fast"], level=4)
    assert len(short_output) == len(short)
    assert short_labels == []


def test_stereo_different_sample_rate_and_empty_input(tmp_path):
    source = tmp_path / "stereo_8khz.wav"
    times = np.arange(800) / 8_000
    tone = (0.4 * np.sin(2 * np.pi * 220 * times)).astype(np.float64)
    sf.write(source, np.column_stack((tone, tone)), 8_000, subtype="PCM_16")

    normalized = read_wav(source)
    assert normalized.ndim == 1
    assert len(normalized) == 1_600
    assert np.max(np.abs(normalized)) == pytest.approx(np.max(np.abs(tone)), abs=0.01)

    empty_source = tmp_path / "empty.wav"
    sf.write(empty_source, np.empty((0, 1)), SAMPLE_RATE)
    with pytest.raises(ValueError, match="empty"):
        read_wav(empty_source)


def test_dataset_builder_writes_required_layout_and_matching_metadata(tmp_path):
    from scripts.build_dataset import METADATA_COLUMNS, build_dataset

    audio_dir = tmp_path / "ideal_audio"
    transcript_dir = tmp_path / "ideal_transcripts"
    output_dir = tmp_path / "dataset"
    audio_dir.mkdir()
    transcript_dir.mkdir()
    write_wav(audio_dir / "sample.wav", separated_speech())
    (transcript_dir / "sample.txt").write_text("spoken words\n", encoding="utf-8")

    metadata_path = build_dataset(
        audio_dir,
        transcript_dir,
        output_dir,
        source="unit-test",
        license_name="CC0-1.0",
        speaker="speaker-1",
        levels=[1, 2, 3, 4],
        flaw_types=["excess_pauses", "too_fast"],
    )

    with metadata_path.open(encoding="utf-8", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        rows = list(reader)
        assert reader.fieldnames == list(METADATA_COLUMNS)
    assert [int(row["level"]) for row in rows] == [1, 2, 3, 4]
    for row in rows:
        types = row["flaw_types"].split(";") if row["flaw_types"] else []
        assert len(types) == len(set(types))
    assert (output_dir / "raw" / "sample.wav").is_file()
    assert (output_dir / "transcripts" / "sample.txt").is_file()
    for row in rows:
        audio_path = output_dir / row["audio_path"]
        label_path = output_dir / row["label_path"]
        assert audio_path.is_file()
        assert audio_path.parent.name == "flawed"
        assert label_path.is_file()
        info = sf.info(audio_path)
        assert float(row["duration_s"]) == pytest.approx(info.duration, abs=1e-6)
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        assert labels["speech_id"] == row["speech_id"]
        assert labels["level"] == int(row["level"])
        assert all(0 <= flaw["start_s"] < flaw["end_s"] <= info.duration for flaw in labels["flaws"])
        generated, _ = sf.read(audio_path, dtype="float64")
        for flaw in labels["flaws"]:
            if flaw["type"] == "excess_pauses":
                start = round(flaw["start_s"] * SAMPLE_RATE)
                end = round(flaw["end_s"] * SAMPLE_RATE)
                assert np.all(generated[start:end] == 0.0)
