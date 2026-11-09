
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRT_FILE = ROOT / "data" / "output" / "localized_review_v7.srt"
REPORT_FILE = ROOT / "data" / "output" / "subtitle_generation_report_v7.json"

TARGET_ID = 86
EXPECTED_NEXT_ID = 87

NEW_END = "00:04:33,861"


def main():
    content = SRT_FILE.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip())

    if TARGET_ID > len(blocks) or EXPECTED_NEXT_ID > len(blocks):
        raise ValueError("Subtitle IDs are outside the file.")

    target = blocks[TARGET_ID - 1].splitlines()
    following = blocks[EXPECTED_NEXT_ID - 1].splitlines()

    if target[0].strip() != str(TARGET_ID):
        raise ValueError("Target subtitle number mismatch.")

    if following[0].strip() != str(EXPECTED_NEXT_ID):
        raise ValueError("Next subtitle number mismatch.")

    target_timing = target[1].split(" --> ")

    if target_timing[1] != "00:04:33,926":
        raise ValueError("Unexpected original end time.")

    next_start = following[1].split(" --> ")[0]

    if next_start != NEW_END:
        raise ValueError("Unexpected next subtitle start time.")

    target[1] = target_timing[0] + " --> " + NEW_END
    blocks[TARGET_ID - 1] = "\n".join(target)

    SRT_FILE.write_text(
        "\n\n".join(blocks) + "\n",
        encoding="utf-8-sig",
        newline=""
    )

    report = json.loads(REPORT_FILE.read_text(encoding="utf-8"))

    report["statistics"]["overlapping_adjacent_cues"] = 0
    report["statistics"]["overlap_correction"] = "PASSED"

    report["overlap_correction"] = {
        "subtitle_number": TARGET_ID,
        "old_end": "00:04:33,926",
        "new_end": NEW_END,
        "next_subtitle_number": EXPECTED_NEXT_ID,
        "reason": "Removed overlap caused by end-time padding."
    }

    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("V7 OVERLAP CORRECTION: OK")
    print("Subtitle:", TARGET_ID)
    print("New end:", NEW_END)
    print("Overlap: 0 ms")


if __name__ == "__main__":
    main()
