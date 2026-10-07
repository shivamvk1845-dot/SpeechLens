# SpeechLens

SpeechLens is a speech-delivery feedback prototype: compare a participant recording with a baseline reading of the same text, find time-localized acoustic differences, and show measured evidence with actionable coaching suggestions.

## Multimodal AI Hackathon 2026 · Track C

SpeechLens addresses the Track C challenge of useful, interpretable speech feedback. It combines audio signal features with transcript-based word timings; it does not infer meaning, grade presentation content, or identify speakers.

## Problem

General presentation advice is hard to act on because it rarely tells a speaker what changed or where. SpeechLens compares two recordings of the same text, aligns their timing, flags delivery patterns such as pace, pitch variation, pauses, projection, and clarity proxies, and links each finding to numeric evidence and a playback region.

## Features

- Build a reproducible synthetic dataset from user-supplied WAV recordings and transcripts, with labeled flaw types and severity levels.
- Align supplied transcripts to audio at the word level with WhisperX forced alignment.
- Extract frame-level pitch, energy, MFCC, spectral, voicing, and harmonicity features, as well as word-rate and pause measurements.
- Compare baseline and participant recordings over time using MFCC dynamic time warping (DTW), then detect and merge deviations into regions.
- Explore synchronized waveform and feature plots, highlighted transcript words, region audio, category scores, and downloadable JSON/CSV reports.

## Architecture

`scripts/build_dataset.py` uses the fixed DSP configuration in `src/speechlens/config.py` and seeded random choices in `inject.py` to transform user-provided audio. The analysis path normalizes audio to mono 16 kHz, obtains word timings from WhisperX, extracts acoustic and word features, and compares participant values with the baseline. `detect.py`, `explain.py`, and `score.py` turn metric deltas into regions, evidence-based explanations, and a transparent rubric. `app/streamlit_app.py` presents the workflow and reports.

See [docs/TECHNICAL.md](docs/TECHNICAL.md) for the method and implementation details.

## Project structure

```text
app/streamlit_app.py       Streamlit comparison dashboard
data/README.md             Synthetic dataset format and generation notes
docs/TECHNICAL.md          Technical design and evaluation boundaries
scripts/build_dataset.py   Dataset generation CLI
src/speechlens/            Alignment, DSP, feature, comparison, detection,
                           explanation, and scoring modules
tests/                     Phase 1–4 automated tests
requirements.txt           Pinned direct Python dependencies
.python-version            Recommended Python patch version
Dockerfile                 Streamlit container image
Makefile                   Setup, test, check, demo, and dataset-help targets
```

## Setup and dependencies

Use **Python 3.13.14** (also recorded in `.python-version`). The pinned direct dependencies are in `requirements.txt`; the analysis and dataset pipeline uses NumPy, SciPy, librosa, SoundFile, PyWORLD, and Parselmouth. The dashboard uses Streamlit and Plotly; tests use pytest. WhisperX and its model stack are required for uncached real word alignment. Installation of these packages does not download alignment model weights.

### PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

### macOS / Linux

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

The GNU Make targets `setup`, `test`, `check`, `demo`, and `dataset-help` wrap the corresponding commands. Make is optional; commands above work without it. `setup` installs Python packages and `demo` starts a local server.

## Generate a dataset

No audio dataset or demo recordings are bundled. Prepare pairs with identical filename stems, for example `data/ideal/audio/take-01.wav` and `data/ideal/transcripts/take-01.txt`. The audio source, license, and speaker values in metadata must accurately describe your inputs. Then run:

```powershell
python scripts\build_dataset.py `
  --audio-dir data\ideal\audio `
  --transcript-dir data\ideal\transcripts `
  --output-dir data\generated `
  --source "your-source" `
  --license "your-license" `
  --speaker "speaker-001" `
  --seed 42
```

The builder creates `raw/`, `flawed/`, `transcripts/`, `labels/`, and `metadata.csv` under the output directory. By default it generates levels 0–4 and all seven flaw types; use `--levels` and `--flaws` to select variants. Level 0 is unchanged audio. See [data/README.md](data/README.md) for input requirements, labels, and transformation details. Generated data is ignored by Git.

## Tests

```sh
python -m pytest -q
python -m pip check
```

Tests cover dataset construction, seeded signal transformations, alignment validation and cache behavior, feature extraction, DTW comparison, detection, explanations, scoring, and dashboard rendering. WhisperX is mocked in alignment tests; these tests do **not** establish that a real model download or real-world forced alignment succeeds.

## Launch and use the dashboard

```sh
python -m streamlit run app/streamlit_app.py
```

1. Upload a baseline recording, or select one found below `data/**/raw/`.
2. Upload the participant recording and its UTF-8 transcript. Supply a baseline transcript when available; if omitted, the participant transcript is used for baseline alignment.
3. Keep the same spoken text in both recordings for meaningful word comparisons. Select an alignment language code and adjust detection sensitivity if useful.
4. Review the overall and category scores, synchronized plots, highlighted transcript, detected regions, and region audio. Download the JSON report or CSV region list as needed.

On the first uncached analysis, WhisperX and its PyTorch/torchaudio stack load/download the language alignment model at runtime. CPU alignment is configured by default; model availability, language support, network access, and Hugging Face/runtime requirements can affect first use. Model-backed integration has not been validated in this repository's test run.

## Docker

```sh
docker build -t speechlens .
docker run --rm -p 8501:8501 speechlens
```

Open `http://localhost:8501`. To use a local dataset baseline without baking recordings into the image, mount a directory containing `raw/` and (optionally) `transcripts/` as `/app/data/demo` read-only. For example, in PowerShell:

```powershell
docker run --rm -p 8501:8501 -v "${PWD}\data\demo:/app/data/demo:ro" speechlens
```

The image uses the pinned Python patch version and installs the pinned requirements. It does not download WhisperX model weights during build; first-use alignment still needs the model/runtime prerequisites.

## Limitations and reproducibility

The dataset builder creates **synthetic DSP examples**, not natural human delivery. The mumbling transform is a low-pass clarity proxy. Unit tests use generated signals and validate code behavior; no benchmark accuracy, clinical validity, user-study result, or real-world speech performance is claimed. Model-backed WhisperX integration remains to be validated with real audio and supported model assets.

The dataset CLI and injector default to seed `42`; injection uses a local seeded random generator, and audio/severity/feature/comparison/scoring constants are centralized in `src/speechlens/config.py`. Direct Python dependencies are pinned, WhisperX is version-pinned in both `requirements.txt` and its alignment cache identity, and Python is recorded in `.python-version`. Repeating a run with the same inputs, seed, dependency versions, and compatible DSP/runtime stack is the reproducibility target. Exact output bytes across different operating systems or numerical-library builds are not guaranteed. Keep input provenance and license metadata accurate.
