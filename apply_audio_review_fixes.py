import json
import shutil
from pathlib import Path
from datetime import datetime

path = Path("data/output/speaker_review_words_v1.json")
backup_dir = Path("data/output/audio-review-backup")
backup_dir.mkdir(parents=True, exist_ok=True)

with path.open("r", encoding="utf-8-sig") as f:
    data = json.load(f)

segments = {s["segment_id"]: s for s in data["segments"]}

required = ["SEG_000001", "SEG_000045", "SEG_000084", "SEG_000209"]
missing = [x for x in required if x not in segments]

if missing:
    raise SystemExit(f"STOP: Missing segments: {missing}")

def find_word(segment_id, start, expected):
    words = segments[segment_id].get("words", [])
    matches = [
        w for w in words
        if abs(w["start_seconds"] - start) < 0.04
        and w["text"].lower() == expected.lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"STOP: {segment_id}, {start}, {expected}: "
            f"found {len(matches)} matches"
        )
    return matches[0]

# Validate every target before changing anything.
w1 = find_word("SEG_000001", 13.76, "wolf")
w2 = find_word("SEG_000001", 14.081, "bead")
w3 = find_word("SEG_000001", 17.196, "end")
w4 = find_word("SEG_000045", 314.242, "wolf-bait,")

remove_084 = [
    find_word("SEG_000084", 731.346, "I"),
    find_word("SEG_000084", 731.362, "returned"),
    find_word("SEG_000084", 731.458, "to"),
]

remove_209 = [
    find_word("SEG_000209", 1859.860, "What"),
    find_word("SEG_000209", 1860.038, "are"),
    find_word("SEG_000209", 1860.135, "you"),
    find_word("SEG_000209", 1860.183, "doing?"),
]

# Make a timestamped backup before writing.
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = backup_dir / f"speaker_review_words_v1_{stamp}.json"
shutil.copy2(path, backup)

# Apply confirmed text corrections.
w1["text"] = "wolf's"
w2["text"] = "bane"
w3["text"] = "foot"
w4["text"] = "wolf's bane,"

for sid, words_to_remove in [
    ("SEG_000084", remove_084),
    ("SEG_000209", remove_209),
]:
    remove_ids = {id(w) for w in words_to_remove}
    segments[sid]["words"] = [
        w for w in segments[sid]["words"] if id(w) not in remove_ids
    ]

# Rebuild only the affected segment transcripts from their word tokens.
for sid in required:
    segments[sid]["text"] = " ".join(
        w["text"] for w in segments[sid].get("words", [])
    )

# Update full_text only if it is a string and the affected old text is present.
if isinstance(data.get("full_text"), str):
    for sid in required:
        old_text = None
        # Recover original segment text from the backup.
        with backup.open("r", encoding="utf-8-sig") as f:
            original = json.load(f)
        old_segment = next(
            s for s in original["segments"] if s["segment_id"] == sid
        )
        old_text = old_segment.get("text", "")
        new_text = segments[sid]["text"]
        if old_text and old_text in data["full_text"]:
            data["full_text"] = data["full_text"].replace(
                old_text, new_text, 1
            )

with path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Verify the saved JSON can be reopened.
with path.open("r", encoding="utf-8") as f:
    check = json.load(f)

print("SUCCESS: Confirmed word-level corrections applied.")
print("Backup:", backup)
print("Updated file:", path)
print("Segments changed:", ", ".join(required))
print("Total segments:", len(check["segments"]))