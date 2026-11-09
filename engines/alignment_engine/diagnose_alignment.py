
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRT_FILE = ROOT / "data" / "input" / "input.srt"
ASR_FILE = ROOT / "data" / "output" / "transcription_result.json"

def timestamp_to_seconds(value):
    h, m, rest = value.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

print("=" * 70)
print("SRT SUBTITLES 74-84")
print("=" * 70)

srt_text = SRT_FILE.read_text(encoding="utf-8-sig")
blocks = re.split(r"\n\s*\n", srt_text.strip())

for block in blocks:
    lines = block.strip().splitlines()
    if len(lines) < 3:
        continue

    try:
        subtitle_id = int(lines[0].strip())
        start_text, end_text = lines[1].split("-->")
        start = timestamp_to_seconds(start_text.strip())
        end = timestamp_to_seconds(end_text.strip())
    except (ValueError, IndexError):
        continue

    if 74 <= subtitle_id <= 84:
        print(
            f"\nID {subtitle_id} | "
            f"{start:.3f} --> {end:.3f}"
        )
        print(" ".join(line.strip() for line in lines[2:]))

print("\n" + "=" * 70)
print("RAW ASR WORDS: 220-275 SECONDS")
print("=" * 70)

data = json.loads(ASR_FILE.read_text(encoding="utf-8"))

for segment in data["segments"]:
    for word in segment.get("words", []):
        start = word.get("start_seconds", 0)
        end = word.get("end_seconds", 0)

        if start >= 220 and end <= 275:
            print(
                f"{start:8.3f} - {end:8.3f} | "
                f"{word.get('speaker_id', '?'):8} | "
                f"{word.get('text', '')}"
            )

print("\nDIAGNOSTIC: OK")
