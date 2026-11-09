
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_FILE = PROJECT_ROOT / "data" / "input" / "Husbands.mp4"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "review_clips"
)

PADDING_SECONDS = 3.0

CANDIDATES = [
    {
        "candidate_id": "MD_0009",
        "start_seconds": 176.562,
        "end_seconds": 176.836,
        "text": "There won't",
    },
    {
        "candidate_id": "MD_0011",
        "start_seconds": 243.279,
        "end_seconds": 246.206,
        "text": "Every time I complained, he hurt me.",
    },
    {
        "candidate_id": "MD_0013",
        "start_seconds": 267.090,
        "end_seconds": 267.939,
        "text": "I couldn't speak.",
    },
]


def extract_audio_clip(candidate):
    candidate_id = candidate["candidate_id"]

    start = max(
        0,
        candidate["start_seconds"] - PADDING_SECONDS
    )

    end = candidate["end_seconds"] + PADDING_SECONDS

    duration = end - start

    output_file = OUTPUT_DIR / f"{candidate_id}.wav"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(VIDEO_FILE),
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_file),
    ]

    subprocess.run(command, check=True)

    return {
        "candidate_id": candidate_id,
        "text": candidate["text"],
        "candidate_start": candidate["start_seconds"],
        "candidate_end": candidate["end_seconds"],
        "clip_start": round(start, 3),
        "clip_end": round(end, 3),
        "duration": round(duration, 3),
        "audio_file": str(output_file),
    }


def main():
    if not VIDEO_FILE.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("EXTRACTING REVIEW AUDIO CLIPS")
    print("-----------------------------")

    for candidate in CANDIDATES:
        result = extract_audio_clip(candidate)
        results.append(result)

        print(
            f"{result['candidate_id']}: OK | "
            f"{result['clip_start']} - "
            f"{result['clip_end']} seconds"
        )

    manifest_file = OUTPUT_DIR / "review_manifest.json"

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nAUDIO EXTRACTION: OK")
    print(f"Clips: {len(results)}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_file}")


if __name__ == "__main__":
    main()
