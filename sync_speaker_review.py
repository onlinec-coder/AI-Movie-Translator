import json
import shutil
from pathlib import Path
from datetime import datetime

word_path = Path("data/output/speaker_review_words_v1.json")
review_path = Path("data/output/speaker_review_v1.json")
backup_dir = Path("data/output/audio-review-backup")
backup_dir.mkdir(parents=True, exist_ok=True)

with word_path.open("r", encoding="utf-8-sig") as f:
    word_data = json.load(f)

with review_path.open("r", encoding="utf-8-sig") as f:
    review_data = json.load(f)

target_ids = {
    "SEG_000001",
    "SEG_000045",
    "SEG_000084",
    "SEG_000209",
}

word_segments = {
    s["segment_id"]: s
    for s in word_data["segments"]
    if s.get("segment_id") in target_ids
}

if set(word_segments) != target_ids:
    raise SystemExit("STOP: Word-level segment IDs do not match.")

def make_text(segment):
    return " ".join(
        w["text"] for w in segment.get("words", [])
    )

def get_segment_id(segment):
    for key in ("segment_id", "id"):
        value = segment.get(key)
        if isinstance(value, str) and value.startswith("SEG_"):
            return value
    return None

review_segments = review_data.get("segments", [])
matched = {}

for segment in review_segments:
    sid = get_segment_id(segment)
    if sid in target_ids:
        matched[sid] = segment

if set(matched) != target_ids:
    raise SystemExit(
        "STOP: Could not match all 4 segments in speaker_review_v1.json. "
        f"Matched: {sorted(matched)}"
    )

# Backup the original review file before modifying it.
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = backup_dir / f"speaker_review_v1_{stamp}.json"
shutil.copy2(review_path, backup)

# Synchronize the four corrected segment transcripts.
for sid, segment in matched.items():
    segment["text"] = make_text(word_segments[sid])

# Set SPK_004 character identity to Lucian.
speakers = review_data.get("speakers", [])
speaker_found = False

if isinstance(speakers, list):
    for speaker in speakers:
        speaker_id = speaker.get("speaker_id", speaker.get("id"))
        if speaker_id == "SPK_004":
            speaker["speaker_type"] = "character"
            speaker["character_name"] = "Lucian"
            speaker_found = True

elif isinstance(speakers, dict):
    speaker = speakers.get("SPK_004")
    if isinstance(speaker, dict):
        speaker["speaker_type"] = "character"
        speaker["character_name"] = "Lucian"
        speaker_found = True

if not speaker_found:
    raise SystemExit(
        "STOP: SPK_004 speaker metadata not found. "
        "No changes were saved."
    )

# Update full_text if this file contains it.
if isinstance(review_data.get("full_text"), str):
    for sid, segment in matched.items():
        old_text = segment.get("text", "")
        # Segment text has already been synchronized; full_text is
        # not rebuilt because its original segment boundaries are unknown.
    print("Note: full_text left unchanged; segment texts were synchronized.")

with review_path.open("w", encoding="utf-8") as f:
    json.dump(review_data, f, ensure_ascii=False, indent=2)

# Verify saved JSON and key changes.
with review_path.open("r", encoding="utf-8") as f:
    verified = json.load(f)

verified_speakers = verified["speakers"]
if isinstance(verified_speakers, list):
    lucian_ok = any(
        s.get("speaker_id", s.get("id")) == "SPK_004"
        and s.get("character_name") == "Lucian"
        for s in verified_speakers
    )
else:
    lucian_ok = (
        verified_speakers["SPK_004"].get("character_name") == "Lucian"
    )

if not lucian_ok:
    raise SystemExit("Verification failed: SPK_004 is not Lucian.")

print("SUCCESS: speaker_review_v1.json synchronized.")
print("Backup:", backup)
print("Updated segments:", ", ".join(sorted(target_ids)))
print("SPK_004 character: Lucian")