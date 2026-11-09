
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "transcription_result.json"
)


def print_words(start, end):
    print(f"\n--- {start}-{end} SEC ---")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    for segment in data["segments"]:
        matching_words = [
            word
            for word in segment.get("words", [])
            if word["start_seconds"] < end
            and word["end_seconds"] > start
        ]

        if not matching_words:
            continue

        print(
            f"\nSEGMENT: {segment['segment_id']}"
            f" | SPEAKER: {segment.get('speaker_id')}"
        )
        print("TEXT:", segment.get("text"))

        for word in matching_words:
            print(
                f"{word['start_seconds']:.3f}"
                f"-{word['end_seconds']:.3f}"
                f" | {word['text']}"
                f" | confidence={word.get('confidence')}"
            )


if __name__ == "__main__":
    print_words(240, 252)
    print_words(263, 275)
