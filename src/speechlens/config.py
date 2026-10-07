"""Reproducible audio and severity parameters for Phase 1 flaw injection."""

SAMPLE_RATE = 16_000
MIN_PAUSE_SECONDS = 0.10
SILENCE_TOP_DB = 35

# librosa time-stretch rate: values above one are faster, below one are slower.
TOO_FAST_RATES = (1.08, 1.18, 1.32, 1.50)
TOO_SLOW_RATES = (0.93, 0.85, 0.75, 0.65)
MONOTONE_F0_ALPHA = (0.75, 0.50, 0.25, 0.05)
MISSING_PAUSE_FRACTIONS = (0.25, 0.40, 0.60, 0.80)
MISSING_PAUSE_COUNTS = (1, 1, 2, 4)
EXCESS_PAUSE_SECONDS = (0.08, 0.16, 0.28, 0.45)
EXCESS_PAUSE_COUNTS = (1, 1, 2, 3)
VOLUME_GAINS = (0.85, 0.68, 0.50, 0.32)
VOLUME_FADE_SECONDS = 0.025
MUMBLING_CUTOFF_HZ = (6_000, 4_800, 3_600, 2_500)

FEATURE_HOP_SECONDS = 0.01
FEATURE_FRAME_LENGTH = 1_024
FEATURE_N_MFCC = 13
FEATURE_RMS_FLOOR = 1e-6
FEATURE_F0_MIN_HZ = 65.0
FEATURE_F0_MAX_HZ = 500.0
FEATURE_HNR_MIN_PITCH_HZ = 75.0
FEATURE_HNR_SILENCE_THRESHOLD = 0.1
FEATURE_HNR_PERIODS_PER_WINDOW = 1.0
FEATURE_WORD_RATE_WINDOW_SIZE = 3
ALIGNMENT_LANGUAGE = "en"
ALIGNMENT_DEVICE = "cpu"
ALIGNMENT_TIMESTAMP_TOLERANCE_S = 1e-6
NORMALIZATION_STD_FLOOR = 1e-8

COMPARISON_WINDOW_SECONDS = 1.0
COMPARISON_HOP_SECONDS = 0.25
DTW_METRIC = "euclidean"
DETECTION_THRESHOLD_Z = 2.0
DETECTION_MERGE_GAP_SECONDS = 0.25
DEVIATION_TOLERANCES = {
    "pitch_mean_semitones": 2.0,
    "pitch_std_semitones": 1.5,
    "energy_db": 4.0,
    "speech_rate_wps": 1.0,
    "pause_ratio": 0.15,
    "hnr_db": 4.0,
    "spectral_centroid_hz": 800.0,
    "spectral_flatness": 0.10,
}
RUBRIC_WEIGHTS = {
    "pace": 0.25,
    "pitch": 0.25,
    "volume": 0.15,
    "pauses": 0.20,
    "clarity": 0.15,
}
RUBRIC_PENALTY_PER_Z = 12.5
