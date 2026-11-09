
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "output"

FILES = {
    "alignment": OUTPUT / "alignment_result_v4.json",
    "transcription": OUTPUT / "transcription_result.json",
    "assignments": ROOT / "data" / "input" / "speaker_assignments.json",
    "characters": ROOT / "data" / "input" / "Husbands_characters_reconciled.json",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("SPEAKER MAPPING EVIDENCE INSPECTION")
    print("===================================")

    for name, path in FILES.items():
        print(f"\n{name.upper()}: {path.name}")
        print("Exists:", path.exists())

    alignment = load_json(FILES["alignment"])
    transcription = load_json(FILES["transcription"])
    assignments = load_json(FILES["assignments"])
    characters = load_json(FILES["characters"])

    print("\n\n1. ALIGNMENT STRUCTURE")
    print("---------------------")
    print("Root keys:", list(alignment.keys()))

    for key, value in alignment.items():
        if isinstance(value, list):
            print(f"{key}: list ({len(value)} items)")
            if value:
                print("First item keys:",
                      list(value[0].keys())
                      if isinstance(value[0], dict)
                      else type(value[0]).__name__)

    segments = alignment.get("segments", [])

    if segments:
        print("\nFirst 3 alignment segments:")
        for item in segments[:3]:
            print(json.dumps(item, ensure_ascii=False, indent=2)[:2500])

    print("\n\n2. ASSEMBLYAI SPEAKERS")
    print("---------------------")
    for speaker in transcription.get("speakers", []):
        print(json.dumps(speaker, ensure_ascii=False))

    print("\n\n3. ASSEMBLYAI SPEAKER DISTRIBUTION")
    print("---------------------------------")
    counts = Counter(
        segment.get("speaker_id")
        for segment in transcription.get("segments", [])
    )
    print(dict(counts))

    print("\n\n4. CHARACTER LIST")
    print("-----------------")
    for character in characters.get("characters", []):
        print(
            character.get("id"),
            "|",
            character.get("canonical_name"),
            "| confidence:",
            character.get("confidence")
        )

    print("\n\n5. SUBTITLE SPEAKER DISTRIBUTION")
    print("--------------------------------")
    counts = Counter(
        item.get("speaker_id")
        for item in assignments.get("assignments", [])
    )
    print(dict(counts))

    print("\n\nINSPECTION COMPLETE")


if __name__ == "__main__":
    main()
