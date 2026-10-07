# Phase 1 synthetic speech dataset

The Phase 1 builder creates `raw/`, `flawed/`, `transcripts/`, `labels/`, and `metadata.csv` under the selected output directory. It takes ideal PCM WAV inputs and matching UTF-8 `.txt` transcripts. Audio is downmixed to mono, polyphase-resampled to 16 kHz, and written as mono 16-bit PCM WAV. DSP dependencies and versions are pinned in the repository's `requirements.txt`.

```powershell
python -m pip install -r requirements.txt
python scripts\build_dataset.py `
  --audio-dir data\ideal\audio `
  --transcript-dir data\ideal\transcripts `
  --output-dir data\generated `
  --source "my-source" `
  --license "CC-BY-4.0" `
  --speaker "speaker-001"
```

The builder generates levels 0 through 4 and all seven flaw types by default, with seed `42`. Use `--levels`, `--flaws`, or `--seed` to select variants. Every WAV needs a `.txt` transcript with the same filename stem. CSV paths are relative to the output directory; `duration_s` is measured from the generated waveform. `source`, `license`, and `speaker` are metadata supplied by the caller and must accurately describe the inputs.

Level 0 leaves audio unchanged and produces no flaw labels. Levels 1–4 use monotonically increasing DSP parameters in `src/speechlens/config.py`: librosa phase-vocoder time stretching for tempo; WORLD F0 analysis/resynthesis for monotone; detected internal pauses for deletion; silence insertion at detected speech-interval ends; smooth gain envelopes for volume drop; and progressively lower scipy low-pass cutoffs as a synthetic mumbling/clarity proxy. For monotone, voiced frames are moved toward the voiced median F0:

`F0'(t) = median(F0) + alpha * (F0(t) - median(F0))`

where `alpha` decreases with severity. WORLD leaves unvoiced F0 frames at zero. Pitch/time stretch uses librosa and is distinct from sample-rate resampling.

Labels use seconds on the final output timeline. Time-stretch edits map points within their replaced interval according to the actual replacement length; insertions and deletions shift later labels by the exact sample count. An `excess_pauses` label covers only the inserted zero samples. A removed pause no longer exists in the output, so its label marks a short output seam around the splice; `magnitude` records the deleted pause duration in seconds. If no internal pause at least `MIN_PAUSE_SECONDS` long exists, `missing_pauses` is skipped and no fabricated label is emitted.

These transformations provide deterministic synthetic training examples, not natural human performance or a clinically validated mumbling detector. WORLD resynthesis and linear gain/filtering can sound artificial. The input must be audio supported by SoundFile; the builder currently enumerates `.wav` files only. The 16 kHz mono normalization is fixed for Phase 1.

## Phase 2 alignment and acoustic features

`src/speechlens/align.py` uses WhisperX forced alignment and caches word timings by audio content, transcript, language, and pinned aligner version. The first uncached call loads WhisperX's language alignment model, so model download/runtime is not part of the offline tests. `features.py` extracts 10 ms frame-level F0, RMS dB, 13 MFCCs, spectral centroid/flatness, Praat HNR, and voicing, plus word-rate and alignment-gap pause measurements. `normalize.py` centers F0 in semitones relative to speaker median F0 and energy relative to speaker median dB, then calculates per-speaker z-scores. Fit statistics across all recordings that should share the same speaker reference before normalizing them.
