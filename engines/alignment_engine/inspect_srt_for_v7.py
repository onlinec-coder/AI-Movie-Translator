
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRT_FILE = ROOT / "data" / "input" / "input.srt"
TIMELINE_FILE = ROOT / "data" / "output" / "timeline_reference.json"


def main():
    with open(SRT_FILE, "r", encoding="utf-8-sig") as f:
        srt_text = f.read()

    with open(TIMELINE_FILE, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    print("SRT FILE: OK")
    print("TIMELINE ROOT KEYS:", list(timeline))

    print("\n--- ORIGINAL SRT: SUBTITLES 78-84 ---")

    blocks = srt_text.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().splitlines()

        if not lines:
            continue

        try:
            subtitle_id = int(lines[0].strip())
        except ValueError:
            continue

        if 78 <= subtitle_id <= 84:
            print("\n" + block)

    print("\n--- TIMELINE STRUCTURE ---")

    segments = timeline.get("segments", [])

    for segment in segments:
        if segment.get("subtitle_id") in [78, 79, 80, 81, 82, 83, 84]:
            print(json.dumps(
                segment,
                ensure_ascii=False,
                indent=2
            ))


if __name__ == "__main__":
    main()
