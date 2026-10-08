# SpeechLens

### Contrastive Speech-Delivery Evaluation with Temporal Flaw Grounding

SpeechLens is a speech-delivery feedback system that compares a participant's recording with a baseline recording of the **same spoken text**. It aligns both recordings, measures acoustic differences, identifies time-localized delivery issues, and provides evidence-based feedback with category scores.

Built for **Multimodal AI Hackathon 2026 · Track C**.

> **SpeechLens focuses on how something is spoken — not what is being said.**

## 🚀 Live Demo

[**Open SpeechLens →**](https://speechlens.streamlit.app)

Try the live application to compare speech delivery against a baseline recording and explore time-localized acoustic feedback.

---

## Multimodal AI Hackathon 2026 · Track C

SpeechLens addresses the Track C challenge of building useful and interpretable speech feedback.

The system combines:

- **WhisperX** for transcript-based word alignment
- **Audio signal processing** for pitch, energy, pauses, clarity, and spectral features
- **DTW-based temporal comparison** between baseline and participant recordings
- **Temporal flaw detection** with start/end timestamps
- **Evidence-based explanations** linking detected differences to measurable acoustic changes
- **Transparent scoring** across Pace, Pitch, Volume, Pauses, and Clarity
- **Streamlit + Plotly** for interactive analysis and visualization

SpeechLens does **not** infer meaning, grade presentation content, or identify speakers.

---

## Problem

General presentation and speaking advice is often difficult to act on because it rarely tells a speaker:

- **What changed?**
- **Where did it happen?**
- **How large was the difference?**
- **What measurable evidence supports the feedback?**

SpeechLens addresses this by comparing two recordings of the **same transcript**.

The system aligns the recordings, extracts acoustic features, compares their temporal behavior, detects delivery deviations, and links each detected region to measurable evidence and an explanation.

---

## How It Works

```text
        Baseline Recording
               +
      Participant Recording
               |
               v
      Transcript / Word Alignment
             WhisperX
               |
               v
       Acoustic Feature Extraction
               |
       +-------+-------+
       |       |       |
      Pitch  Energy  MFCC
       |       |       |
     Pauses  Rate   Spectral
       |       |     Features
       +-------+-------+
               |
               v
       Baseline vs Participant
          Temporal Comparison
              DTW
               |
               v
       Deviation Detection
               |
               v
      Time-Localized Flaw Regions
               |
               v
       Numeric Evidence +
       Human-Readable Explanation
               |
               v
        Category Scores
               |
               v
       Streamlit Dashboard
