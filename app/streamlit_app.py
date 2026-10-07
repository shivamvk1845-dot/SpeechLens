"""SpeechLens baseline-versus-participant delivery evaluation dashboard."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
from plotly.subplots import make_subplots

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speechlens.align import align_transcript
from speechlens.compare import ComparisonResult, compare_features
from speechlens.config import SAMPLE_RATE
from speechlens.detect import detect_regions
from speechlens.explain import explain_regions
from speechlens.features import AcousticFeatures, extract_features
from speechlens.score import score_comparison

_FLAW_COLORS = {
    "too_fast": "#ef4444",
    "too_slow": "#f97316",
    "monotone": "#8b5cf6",
    "missing_pauses": "#eab308",
    "excess_pauses": "#f59e0b",
    "volume_drop": "#3b82f6",
    "mumbling": "#10b981",
}
_METRIC_TITLES = (
    "Waveform",
    "Pitch (semitones)",
    "Energy (dB)",
    "Speech rate (words/s)",
)


def _decode_audio(audio_bytes: bytes, file_name: str) -> np.ndarray:
    """Decode uploaded WAV/MP3 content into mono 16 kHz floating-point audio."""
    if not audio_bytes:
        raise ValueError(f"{file_name} is empty")
    try:
        audio, sample_rate = sf.read(
            io.BytesIO(audio_bytes), dtype="float64", always_2d=True
        )
        if audio.shape[0] == 0:
            raise ValueError(f"{file_name} contains no audio samples")
        mono = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    except (sf.SoundFileError, RuntimeError):
        try:
            mono, _ = librosa.load(
                io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True, dtype=np.float64
            )
        except Exception as error:
            raise ValueError(f"Could not decode {file_name} as WAV or MP3: {error}") from error
    if mono.size == 0 or not np.all(np.isfinite(mono)):
        raise ValueError(f"{file_name} decoded to empty or non-finite audio")
    return np.asarray(mono, dtype=np.float64)


def _frame_envelope(
    audio: np.ndarray, frame_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return min/max waveform envelopes in feature-frame-sized sample bins."""
    edges = np.linspace(0, len(audio), frame_count + 1, dtype=np.int64)
    minimum = np.zeros(frame_count, dtype=np.float64)
    maximum = np.zeros(frame_count, dtype=np.float64)
    for frame, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        section = audio[start:end]
        if section.size:
            minimum[frame] = np.min(section)
            maximum[frame] = np.max(section)
    return minimum, maximum


def _warped_participant_series(
    comparison: ComparisonResult,
    participant_values: np.ndarray,
    baseline_frame_count: int,
) -> np.ndarray:
    """Aggregate participant frame values onto DTW-matched baseline frames."""
    matched: dict[int, list[float]] = {}
    for baseline_index, participant_index in comparison.dtw_path:
        if (
            0 <= baseline_index < baseline_frame_count
            and 0 <= participant_index < len(participant_values)
            and np.isfinite(participant_values[participant_index])
        ):
            matched.setdefault(baseline_index, []).append(
                float(participant_values[participant_index])
            )
    output = np.full(baseline_frame_count, np.nan, dtype=np.float64)
    for baseline_index, values in matched.items():
        output[baseline_index] = float(np.mean(values))
    return output


def _warped_word_rate(features: AcousticFeatures) -> np.ndarray:
    values = np.full(features.times_s.shape, np.nan, dtype=np.float64)
    words = [
        (float(word["start"]), float(word["end"]), float(word["speech_rate_wps"]))
        for word in features.word_features
        if float(word["end"]) > float(word["start"])
        and np.isfinite(float(word["speech_rate_wps"]))
    ]
    if not words:
        return values

    words.sort(key=lambda word: (word[0], word[1]))
    midpoints = np.asarray([(start + end) / 2 for start, end, _ in words])
    rates = np.asarray([rate for _, _, rate in words])
    in_speech = (features.times_s >= words[0][0]) & (features.times_s <= words[-1][1])
    values[in_speech] = np.interp(
        features.times_s[in_speech],
        midpoints,
        rates,
        left=rates[0],
        right=rates[-1],
    )
    return values


def _score_card_items(score: dict[str, Any]) -> list[tuple[str, str]]:
    """Format score labels and values for the dashboard metric cards."""
    return [
        ("Overall score", f"{score['total']:.0f}/100"),
        *(
            (name.title(), f"{details['score']:.0f}/100")
            for name, details in score["categories"].items()
        ),
    ]


def _comparison_figure(
    baseline_audio: np.ndarray,
    participant_audio: np.ndarray,
    baseline_features: AcousticFeatures,
    participant_features: AcousticFeatures,
    comparison: ComparisonResult,
    regions: list[dict[str, Any]],
) -> go.Figure:
    """Build synchronized waveform and feature overlays with shaded flaw regions."""
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.065,
        subplot_titles=_METRIC_TITLES,
    )
    times = baseline_features.times_s
    participant_f0 = _warped_participant_series(
        comparison, participant_features.f0_hz, len(times)
    )
    participant_energy = _warped_participant_series(
        comparison, participant_features.rms_db, len(times)
    )
    participant_rate = _warped_participant_series(
        comparison, _warped_word_rate(participant_features), len(times)
    )
    base_f0 = baseline_features.f0_hz.copy()
    base_f0[~baseline_features.voiced] = np.nan
    participant_f0[~np.isfinite(participant_f0) | (participant_f0 <= 0)] = np.nan
    if np.any(baseline_features.voiced):
        f0_reference = float(np.median(base_f0[baseline_features.voiced]))
        base_f0 = 12.0 * np.log2(base_f0 / f0_reference)
        participant_f0 = 12.0 * np.log2(participant_f0 / f0_reference)

    base_minimum, base_maximum = _frame_envelope(baseline_audio, len(times))
    participant_minimum, participant_maximum = _frame_envelope(
        participant_audio, len(participant_features.times_s)
    )
    warped_minimum = _warped_participant_series(
        comparison, participant_minimum, len(times)
    )
    warped_maximum = _warped_participant_series(
        comparison, participant_maximum, len(times)
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=base_maximum,
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
            name="Baseline envelope",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=base_minimum,
            fill="tonexty",
            fillcolor="rgba(59,130,246,0.25)",
            line={"width": 0},
            name="Baseline waveform",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=warped_maximum,
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
            name="Participant envelope",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=warped_minimum,
            fill="tonexty",
            fillcolor="rgba(239,68,68,0.24)",
            line={"width": 0},
            name="Participant waveform",
        ),
        row=1,
        col=1,
    )

    baseline_metrics = (
        base_f0,
        baseline_features.rms_db,
        _warped_word_rate(baseline_features),
    )
    participant_metrics = (
        participant_f0,
        participant_energy,
        participant_rate,
    )
    for row, (base_values, participant_values) in enumerate(
        zip(baseline_metrics, participant_metrics), start=2
    ):
        figure.add_trace(
            go.Scatter(
                x=times,
                y=base_values,
                mode="lines",
                name="Baseline",
                line={"color": "#2563eb", "width": 1.5},
                legendgroup="baseline",
                showlegend=row == 2,
                connectgaps=False,
            ),
            row=row,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=times,
                y=participant_values,
                mode="lines",
                name="Participant",
                line={"color": "#dc2626", "width": 1.5},
                legendgroup="participant",
                showlegend=row == 2,
                connectgaps=False,
            ),
            row=row,
            col=1,
        )

    for region in regions:
        color = _FLAW_COLORS.get(region["flaw_type"], "#94a3b8")
        for row in range(1, 5):
            figure.add_vrect(
                x0=region["start_s"],
                x1=region["end_s"],
                fillcolor=color,
                opacity=0.10,
                line_width=0,
                row=row,
                col=1,
            )
        figure.add_trace(
            go.Scatter(
                x=[(region["start_s"] + region["end_s"]) / 2],
                y=[0],
                mode="markers",
                marker={"size": 14, "color": color, "symbol": "diamond"},
                customdata=[[region["flaw_type"]]],
                name=f"Jump: {region['flaw_type']}",
                hovertemplate=f"{region['flaw_type']} · {region['start_s']:.2f}–{region['end_s']:.2f}s<extra>Click to jump</extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    figure.update_xaxes(title_text="Baseline time (seconds)", row=4, col=1)
    figure.update_layout(
        height=880,
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 1, "xanchor": "right"},
        hovermode="x unified",
        template="plotly_white",
        clickmode="event+select",
    )
    return figure


def _highlighted_transcript(
    alignments: list[dict[str, Any]],
    regions: list[dict[str, Any]],
) -> str:
    """Render aligned words and emphasize participant words inside detected regions."""
    pieces = []
    for word in alignments:
        matching = [
            region
            for region in regions
            if region.get("participant_start_s", 0.0) <= word["end"]
            and region.get("participant_end_s", 0.0) >= word["start"]
        ]
        escaped = html.escape(word["word"])
        if matching:
            color = _FLAW_COLORS.get(matching[0]["flaw_type"], "#eab308")
            pieces.append(
                f'<mark style="background:{color};color:white;padding:2px 4px;border-radius:4px" '
                f'title="{html.escape(matching[0]["flaw_type"])}">'
                f"{escaped}</mark>"
            )
        else:
            pieces.append(escaped)
    return " ".join(pieces)


def _region_csv(explanations: list[dict[str, Any]]) -> bytes:
    """Serialize detected flaw regions to a portable CSV download."""
    buffer = io.StringIO(newline="")
    fields = ("flaw_type", "start_s", "end_s", "severity_z", "explanation")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for region in explanations:
        writer.writerow({field: region[field] for field in fields})
    return buffer.getvalue().encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=8)
def _analyze_cached(
    baseline_bytes: bytes,
    baseline_name: str,
    participant_bytes: bytes,
    participant_name: str,
    baseline_transcript: str,
    participant_transcript: str,
    language_code: str,
) -> dict[str, Any]:
    """Cache decoded audio, alignment, features, and comparison by uploaded content."""
    baseline_audio = _decode_audio(baseline_bytes, baseline_name)
    participant_audio = _decode_audio(participant_bytes, participant_name)
    baseline_features = _analyze_one(
        baseline_audio, baseline_transcript, language_code, baseline_name
    )
    participant_features = _analyze_one(
        participant_audio, participant_transcript, language_code, participant_name
    )
    comparison = compare_features(baseline_features[1], participant_features[1])
    return {
        "baseline_audio": baseline_audio,
        "participant_audio": participant_audio,
        "baseline_alignments": baseline_features[0],
        "baseline_features": baseline_features[1],
        "participant_alignments": participant_features[0],
        "participant_features": participant_features[1],
        "comparison": comparison,
    }


def _analyze_one(
    audio: np.ndarray,
    transcript: str,
    language_code: str,
    name: str,
) -> tuple[list[dict[str, Any]], AcousticFeatures]:
    """Align and extract features for one audio signal using a temporary WAV."""
    with tempfile.TemporaryDirectory(prefix="speechlens-dashboard-") as directory:
        audio_path = Path(directory) / f"{hashlib.sha256(name.encode()).hexdigest()[:12]}.wav"
        sf.write(audio_path, audio, SAMPLE_RATE, subtype="PCM_16")
        alignments = align_transcript(
            audio_path,
            transcript,
            Path(tempfile.gettempdir()) / "speechlens-align-cache",
            language_code=language_code,
        )
    features = extract_features(audio, word_alignments=alignments)
    return alignments, features


def _dataset_baselines(repository_root: Path) -> dict[str, Path]:
    """Find WAV baselines stored in conventional dataset raw directories."""
    data_root = repository_root / "data"
    return {
        str(path.relative_to(repository_root)): path
        for raw_directory in data_root.rglob("raw")
        if raw_directory.is_dir()
        for path in raw_directory.glob("*.wav")
    }


def _dataset_transcript(audio_path: Path) -> str | None:
    """Find the transcript paired with a dataset baseline, if present."""
    candidates = (
        audio_path.parent.parent / "transcripts" / f"{audio_path.stem}.txt",
        audio_path.parent.parent / "transcripts" / f"{audio_path.parent.name}.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return None


def _render_region_cards(
    explanations: list[dict[str, Any]],
    participant_audio: np.ndarray,
    selected_region: int | None,
) -> int | None:
    """Show actionable flaw cards and audio clips; return a clicked region index."""
    selected = selected_region
    for index, region in enumerate(explanations):
        color = _FLAW_COLORS.get(region["flaw_type"], "#64748b")
        title = (
            f"{region['flaw_type'].replace('_', ' ').title()} · "
            f"{region['start_s']:.2f}–{region['end_s']:.2f}s"
        )
        with st.expander(title, expanded=selected == index):
            st.markdown(
                f"**Deviation severity:** {region['severity_z']:.2f} tolerance units"
            )
            st.write(region["explanation"])
            participant_start = min(
                (
                    evidence["participant_start_s"]
                    for evidence in region["evidence"]
                ),
                default=region["start_s"],
            )
            participant_end = max(
                (
                    evidence["participant_end_s"]
                    for evidence in region["evidence"]
                ),
                default=region["end_s"],
            )
            first = max(0, round(participant_start * SAMPLE_RATE))
            last = min(len(participant_audio), round(participant_end * SAMPLE_RATE))
            if last > first:
                st.audio(participant_audio[first:last], sample_rate=SAMPLE_RATE)
            if st.button(
                "Jump to region",
                key=f"jump-region-{index}",
                type="primary" if selected == index else "secondary",
            ):
                selected = index
    return selected


def main() -> None:
    """Render the interactive SpeechLens comparison dashboard."""
    st.set_page_config(
        page_title="SpeechLens",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1440px; padding-top: 1.5rem;}
        [data-testid="stMetric"] {background:#f8fafc; border:1px solid #e2e8f0;
          padding:16px; border-radius:14px;}
        [data-testid="stMetric"] [data-testid="stMetricLabel"],
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {color:#1e293b !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("SpeechLens")
    st.caption("Compare speech delivery against a baseline recording of the same text.")

    repository_root = Path(__file__).resolve().parents[1]
    baselines = _dataset_baselines(repository_root)
    with st.sidebar:
        st.header("Recordings")
        choices = ["Upload a baseline"] + sorted(baselines)
        baseline_choice = st.selectbox("Baseline source", choices)
        baseline_upload = None
        if baseline_choice == "Upload a baseline":
            baseline_upload = st.file_uploader(
                "Baseline audio", type=("wav", "mp3"), key="baseline-audio"
            )
        participant_upload = st.file_uploader(
            "Participant audio", type=("wav", "mp3"), key="participant-audio"
        )
        transcript_upload = st.file_uploader(
            "Participant transcript (.txt)", type=("txt",), key="participant-transcript"
        )
        baseline_transcript_upload = st.file_uploader(
            "Optional baseline transcript (.txt)",
            type=("txt",),
            key="baseline-transcript",
        )
        threshold = st.slider(
            "Detection sensitivity (z threshold)",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.25,
            help="Lower values flag smaller deviations; higher values require stronger evidence.",
        )
        language = st.text_input("Alignment language code", value="en", max_chars=12)

    if participant_upload is None or transcript_upload is None:
        st.info("Upload participant audio and its transcript to begin.")
        return
    if baseline_choice == "Upload a baseline" and baseline_upload is None:
        st.info("Upload a baseline recording or place WAV baselines under `data/**/raw/`.")
        return

    if baseline_choice == "Upload a baseline":
        baseline_bytes = baseline_upload.getvalue()
        baseline_name = baseline_upload.name
        baseline_text = (
            baseline_transcript_upload.getvalue().decode("utf-8")
            if baseline_transcript_upload is not None
            else None
        )
    else:
        baseline_path = baselines[baseline_choice]
        baseline_bytes = baseline_path.read_bytes()
        baseline_name = baseline_path.name
        baseline_text = _dataset_transcript(baseline_path)
        if baseline_text is None and baseline_transcript_upload is not None:
            baseline_text = baseline_transcript_upload.getvalue().decode("utf-8")
    participant_text = transcript_upload.getvalue().decode("utf-8")
    if not participant_text.strip():
        st.error("The uploaded participant transcript is empty.")
        return
    if not language.strip():
        st.error("Enter a valid WhisperX language code.")
        return
    if baseline_text and baseline_text.strip() != participant_text.strip():
        st.warning("Baseline and participant transcripts differ; word matches are limited to shared tokens.")
    baseline_analysis_text = (
        baseline_text if baseline_text and baseline_text.strip() else participant_text
    )

    try:
        with st.spinner("Aligning audio and extracting acoustic features…"):
            result = _analyze_cached(
                baseline_bytes,
                baseline_name,
                participant_upload.getvalue(),
                participant_upload.name,
                baseline_analysis_text,
                participant_text,
                language.strip(),
            )
        comparison: ComparisonResult = result["comparison"]
        regions = detect_regions(comparison, threshold_z=threshold)
        explanations = explain_regions(regions)
        score = score_comparison(comparison)
    except Exception as error:
        st.error(f"Analysis failed: {error}")
        st.exception(error)
        return

    score_cards = _score_card_items(score)
    metric_columns = st.columns(len(score_cards))
    for column, (label, value) in zip(metric_columns, score_cards):
        column.metric(label, value)
    st.caption(
        f"{len(regions)} detected region(s) · "
        f"{len(comparison.word_deltas)} aligned word match(es) · "
        f"DTW path {len(comparison.dtw_path):,} frame pairs"
    )

    chart_state = st.plotly_chart(
        _comparison_figure(
            result["baseline_audio"],
            result["participant_audio"],
            result["baseline_features"],
            result["participant_features"],
            comparison,
            regions,
        ),
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="delivery-comparison-chart",
    )
    selected_region = st.session_state.get("selected_region")
    selection = getattr(chart_state, "selection", None)
    if selection and selection.get("points"):
        point = selection["points"][0]
        customdata = point.get("customdata")
        flaw_type = (
            customdata
            if isinstance(customdata, str)
            else customdata[0]
            if customdata
            else None
        )
        if flaw_type:
            selected_region = next(
                (
                    index
                    for index, region in enumerate(regions)
                    if region["flaw_type"] == flaw_type
                    and abs(
                        (region["start_s"] + region["end_s"]) / 2 - float(point["x"])
                    )
                    < 0.05
                ),
                selected_region,
            )
    st.session_state["selected_region"] = _render_region_cards(
        explanations,
        result["participant_audio"],
        selected_region,
    )

    st.subheader("Participant transcript")
    st.markdown(
        _highlighted_transcript(result["participant_alignments"], explanations),
        unsafe_allow_html=True,
    )
    st.subheader("Participant recording")
    st.audio(result["participant_audio"], sample_rate=SAMPLE_RATE)

    report = {
        "score": score,
        "regions": explanations,
        "word_deltas": comparison.word_deltas,
        "threshold_z": threshold,
        "baseline": baseline_name,
        "participant": participant_upload.name,
        "baseline_transcript": baseline_analysis_text,
        "participant_transcript": participant_text,
    }
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2, allow_nan=False),
        file_name="speechlens-report.json",
        mime="application/json",
        use_container_width=True,
    )
    downloads[1].download_button(
        "Download flaw regions CSV",
        data=_region_csv(explanations),
        file_name="speechlens-flaw-regions.csv",
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
