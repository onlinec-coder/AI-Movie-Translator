
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRT = ROOT / "data" / "output" / "localized_review_v7.srt"

def to_ms(timestamp):
    h, m, rest = timestamp.split(":")
    s, ms = rest.split(",")
    return (
        int(h) * 3600000
        + int(m) * 60000
        + int(s) * 1000
        + int(ms)
    )

def main():
    content = SRT.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip())

    subtitles = []

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            raise ValueError("Invalid SRT block.")

        number = int(lines[0])
        start, end = lines[1].split(" --> ")

        subtitles.append({
            "number": number,
            "start": to_ms(start),
            "end": to_ms(end),
            "text": "\n".join(lines[2:])
        })

    overlaps = []
    ordering_errors = []
    numbering_errors = []

    for i, sub in enumerate(subtitles):
        if sub["number"] != i + 1:
            numbering_errors.append(sub["number"])

        if sub["end"] <= sub["start"]:
            raise ValueError(
                f"Invalid duration: subtitle {sub['number']}"
            )

        if i > 0:
            previous = subtitles[i - 1]

            if sub["start"] < previous["start"]:
                ordering_errors.append(sub["number"])

            if previous["end"] > sub["start"]:
                overlaps.append({
                    "subtitle": previous["number"],
                    "next": sub["number"],
                    "overlap_ms": previous["end"] - sub["start"]
                })

    expected = {
        86: ("00:04:31,400", "00:04:33,861"),
        87: ("00:04:33,861", "00:04:35,860")
    }

    def fmt(ms):
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        ms %= 1000
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    for number, (expected_start, expected_end) in expected.items():
        sub = subtitles[number - 1]

        if fmt(sub["start"]) != expected_start:
            raise ValueError(f"Subtitle {number}: start mismatch.")

        if fmt(sub["end"]) != expected_end:
            raise ValueError(f"Subtitle {number}: end mismatch.")

    print("FINAL V7 QA")
    print("-----------")
    print("Total subtitles:", len(subtitles))
    print("Numbering errors:", len(numbering_errors))
    print("Chronological errors:", len(ordering_errors))
    print("Actual overlaps:", len(overlaps))

    if overlaps:
        print("\nOVERLAPS:")
        for item in overlaps:
            print(item)

    if len(subtitles) != 910:
        raise ValueError("Expected exactly 910 subtitles.")

    if numbering_errors or ordering_errors or overlaps:
        raise ValueError("Final QA FAILED.")

    print("\nFINAL QA: PASSED")

if __name__ == "__main__":
    main()
