#!/usr/bin/env python3
"""Build a labeled SpeechLens dataset from ideal WAV and transcript inputs."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speechlens.inject import (
    FLAW_TYPES,
    SAMPLE_RATE,
    inject_flaws,
    read_wav,
    write_labels,
    write_wav,
)

METADATA_COLUMNS = (
    "speech_id",
    "source",
    "license",
    "speaker",
    "duration_s",
    "level",
    "flaw_types",
    "audio_path",
    "transcript_path",
    "label_path",
)


def build_dataset(
    audio_dir: Path,
    transcript_dir: Path,
    output_dir: Path,
    source: str,
    license_name: str,
    speaker: str,
    levels: list[int],
    flaw_types: list[str],
    seed: int = 42,
) -> Path:
    records = []
    raw_out = output_dir / "raw"
    flawed_out = output_dir / "flawed"
    transcripts_out = output_dir / "transcripts"
    raw_out.mkdir(parents=True, exist_ok=True)
    for source_audio in sorted(audio_dir.glob("*.wav")):
        source_transcript = transcript_dir / f"{source_audio.stem}.txt"
        if not source_transcript.is_file():
            raise FileNotFoundError(
                f"Missing transcript for {source_audio.name}: {source_transcript}"
            )
        samples = read_wav(source_audio)
        shutil.copy2(source_audio, raw_out / source_audio.name)
        transcript_relative = Path("transcripts") / source_transcript.name
        transcripts_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_transcript, transcripts_out / source_transcript.name)

        for level in levels:
            speech_id = f"{source_audio.stem}_l{level}"
            modified, flaws = inject_flaws(
                samples, flaw_types, level, seed=seed, sample_rate=SAMPLE_RATE
            )
            audio_relative = Path("flawed") / f"{speech_id}.wav"
            label_relative = Path("labels") / f"{speech_id}.json"
            write_wav(output_dir / audio_relative, modified, SAMPLE_RATE)
            write_labels(output_dir / label_relative, speech_id, level, flaws)
            records.append(
                {
                    "speech_id": speech_id,
                    "source": source,
                    "license": license_name,
                    "speaker": speaker,
                    "duration_s": f"{len(modified) / SAMPLE_RATE:.6f}",
                    "level": level,
                    "flaw_types": ";".join(dict.fromkeys(flaw["type"] for flaw in flaws)),
                    "audio_path": audio_relative.as_posix(),
                    "transcript_path": transcript_relative.as_posix(),
                    "label_path": label_relative.as_posix(),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir", type=Path, required=True, help="Directory of ideal WAV inputs"
    )
    parser.add_argument(
        "--transcript-dir", type=Path, required=True, help="Directory of matching .txt files"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Source name recorded in metadata")
    parser.add_argument("--license", dest="license_name", required=True, help="Input audio license")
    parser.add_argument("--speaker", required=True, help="Speaker identifier recorded in metadata")
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--flaws", nargs="+", choices=FLAW_TYPES, default=list(FLAW_TYPES))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    invalid_levels = [level for level in args.levels if level not in range(5)]
    if invalid_levels:
        parser.error(f"levels must be between 0 and 4; got {invalid_levels}")
    metadata = build_dataset(
        args.audio_dir,
        args.transcript_dir,
        args.output_dir,
        args.source,
        args.license_name,
        args.speaker,
        args.levels,
        args.flaws,
        args.seed,
    )
    print(f"Wrote {metadata}")


if __name__ == "__main__":
    main()
