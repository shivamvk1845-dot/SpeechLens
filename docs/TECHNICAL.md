# SpeechLens technical design

## Dataset design and labels

`scripts/build_dataset.py` accepts user-supplied WAV files and same-stem UTF-8 transcripts. It copies ideal recordings into `raw/`, retains transcripts, generates transformed recordings in `flawed/`, writes interval labels in `labels/`, and records relative paths, duration, source, license, speaker, level, and flaw types in `metadata.csv`. Input provenance is caller-supplied. No audio examples are included in the repository.

The default output contains levels 0–4 and seven possible flaws: `too_fast`, `too_slow`, `monotone`, `missing_pauses`, `excess_pauses`, `volume_drop`, and `mumbling`. Level 0 is unchanged. Levels 1–4 apply progressively stronger configured DSP parameters. Labels use seconds on the final output timeline; edits remap existing labels, inserted pauses label the inserted silence, and removed pauses label the output seam with the removed duration as magnitude. These intervention labels are ground truth only for the known synthetic transformations; there is no manually annotated real-speech ground truth. A pause flaw is omitted when a suitable source pause cannot be found. The mumbling transform is a synthetic low-pass filter proxy, not a speech intelligibility measure.

## Alignment and features

`align.py` uses WhisperX forced alignment to map a supplied transcript to word start/end times. Its on-disk cache key includes audio bytes, transcript, language, device, and the pinned WhisperX version. Empty, unaligned, unsorted, or out-of-audio word timings are rejected. First uncached use loads a language alignment model; it is not needed by the unit tests.

`features.py` resamples and downmixes to mono 16 kHz and extracts 10 ms frame features: F0/voicing, RMS energy in dB, 13 MFCCs, spectral centroid, spectral flatness, and Praat HNR. Aligned word timing supplies local words-per-second estimates and gaps of at least the configured pause minimum. Undefined/unvoiced quantities are represented as NaN and excluded from finite aggregates.

`normalize.py` provides speaker-agnostic, within-speaker normalization: F0 becomes semitones relative to that speaker's median F0, energy becomes dB relative to that speaker's median energy, and feature distributions are standardized using pooled per-speaker means and standard deviations. Fit these statistics across recordings sharing the speaker reference. This utility is separate from the dashboard comparison, which directly compares participant measurements with a baseline.

## Comparison, detection, and scoring

`compare.py` aligns baseline and participant MFCC frames with Euclidean DTW. Sliding windows are placed on baseline time; matched participant frames are aggregated for the same baseline windows. It compares pitch mean/variability, energy, word rate, pause ratio, HNR, spectral centroid, and flatness. Shared transcript tokens are also compared word-by-word where both recordings have aligned instances.

`detect.py` divides metric deltas by configured tolerances and applies directional rules: faster/slower rate, lower pitch variability, lower energy, missing/excess pause ratio, and lower HNR indicate their corresponding flaw types. A combination of higher spectral flatness and lower centroid is an alternate mumbling proxy. A configurable z threshold gates detections; overlapping or nearby windows of the same type merge, and region severity is the maximum absolute evidence z value.

`explain.py` produces a numeric comparison (baseline, participant, delta/relative percentage, and evidence window) plus a coaching suggestion. These are explanations of measured associations, not proof that one acoustic factor caused a delivery outcome. `score.py` computes each category as `max(0, 100 - 12.5 × mean absolute tolerance-normalized deviation)`, then takes a weighted mean: pace 0.25, pitch 0.25, volume 0.15, pauses 0.20, and clarity 0.15. Scores are transparent prototype rubric values, not calibrated assessments.

## Dashboard and workflow

`app/streamlit_app.py` accepts WAV/MP3 baseline and participant uploads, transcript text, language, and a detection threshold. It decodes to mono 16 kHz, calls alignment and feature extraction, compares recordings, then displays baseline-clock waveform/features, shaded regions, highlighted participant transcript, region playback, category/overall scores, and JSON/CSV downloads. Baselines can also be discovered under `data/**/raw/`. Streamlit caches analysis by uploaded content; alignment results use a separate content-addressed disk cache under the system temporary directory.

## Testing and evaluation boundary

The pytest suite covers deterministic seeded injection, label/timeline mapping, synthetic dataset layout, feature and normalization behavior, mocked WhisperX alignment/cache validation, DTW/window and word comparison, detection/merging, numeric explanations, rubric calculations, audio decoding, plotting helpers, and initial dashboard rendering. It is synthetic/unit-level validation of implementation behavior, not a real-speech benchmark or validation of WhisperX model integration. No measured accuracy or real-world performance result is reported.

## Reproducibility and limitations

Python 3.13.14 is the documented runtime. Direct requirements are version-pinned; the default dataset seed is `42`, and DSP, feature, alignment, comparison, detection, and scoring parameters are centralized in `src/speechlens/config.py`. With the same audio, transcript, seed, pinned dependencies, and compatible runtime, generated transformations are intended to be repeatable. Floating-point DSP outputs may differ across platforms and native-library builds.

The synthetic transformations can sound artificial and do not model the full range of human speech. Speaker-relative normalization depends on correctly grouped reference recordings. DTW and acoustic proxies can align or classify imperfectly; forced alignment depends on WhisperX model availability, language support, and first-use runtime downloads. No clinical, educational, accessibility, or public-speaking efficacy claim is established.
