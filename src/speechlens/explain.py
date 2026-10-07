"""Numeric, rule-based causal explanations for detected speech-delivery flaws."""

from __future__ import annotations

from typing import Any

_TEMPLATES = {
    "too_fast": (
        "Speech rate was {ratio:.0f}% faster than baseline "
        "({participant:.2f} vs {baseline:.2f} words/s). "
        "Slow down and leave a brief beat between phrases."
    ),
    "too_slow": (
        "Speech rate was {ratio:.0f}% slower than baseline "
        "({participant:.2f} vs {baseline:.2f} words/s). "
        "Increase pace slightly while keeping phrase endings clear."
    ),
    "monotone": (
        "Pitch variability was {ratio:.0f}% lower than baseline "
        "({participant:.2f} vs {baseline:.2f} semitone standard deviation). "
        "Vary pitch on key words and phrase endings."
    ),
    "volume_drop": (
        "Average energy was {difference:.1f} dB below baseline "
        "({participant:.1f} vs {baseline:.1f} dB). "
        "Maintain a steadier projection through the phrase."
    ),
    "missing_pauses": (
        "Pause ratio was {ratio:.0f}% lower than baseline "
        "({participant:.0f}% vs {baseline:.0f}%). "
        "Add short pauses at phrase boundaries."
    ),
    "excess_pauses": (
        "Pause ratio was {ratio:.0f}% higher than baseline "
        "({participant:.0f}% vs {baseline:.0f}%). "
        "Reduce the long gaps and connect related phrases."
    ),
    "mumbling": (
        "Harmonics-to-noise ratio was {difference:.1f} dB lower than baseline "
        "({participant:.1f} vs {baseline:.1f} dB). "
        "Articulate consonants more clearly and keep the voice forward."
    ),
}


def _relative_percent(delta: float, baseline: float) -> float:
    return 100.0 * abs(delta) / max(abs(baseline), 1e-8)


def explain_region(region: dict[str, Any]) -> dict[str, Any]:
    """Return a numeric explanation, evidence, and actionable suggestion."""
    flaw_type = region["flaw_type"]
    evidence = region.get("evidence", [])
    if flaw_type not in _TEMPLATES or not evidence:
        raise ValueError(f"Cannot explain region without supported evidence: {flaw_type}")
    metric = evidence[0]["metric"]
    baseline = float(evidence[0]["baseline"])
    participant = float(evidence[0]["participant"])
    delta = float(evidence[0]["delta"])
    if metric in {"speech_rate_wps", "pitch_std_semitones", "pause_ratio"}:
        values = {
            "ratio": _relative_percent(delta, baseline),
            "baseline": baseline if metric != "pause_ratio" else 100.0 * baseline,
            "participant": participant if metric != "pause_ratio" else 100.0 * participant,
        }
    else:
        values = {
            "difference": abs(delta),
            "baseline": baseline,
            "participant": participant,
        }
    explanation = _TEMPLATES[flaw_type].format(**values)
    return {
        "flaw_type": flaw_type,
        "start_s": float(region["start_s"]),
        "end_s": float(region["end_s"]),
        "participant_start_s": min(
            float(item.get("participant_start_s", region["start_s"]))
            for item in evidence
        ),
        "participant_end_s": max(
            float(item.get("participant_end_s", region["end_s"]))
            for item in evidence
        ),
        "severity_z": float(region["severity"]),
        "explanation": explanation,
        "evidence": evidence,
    }


def explain_regions(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Explain each detected region with measured values."""
    return [explain_region(region) for region in regions]
