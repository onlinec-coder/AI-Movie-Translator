
import json
import re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]

SRT_FILE = ROOT / "data" / "input" / "input.srt"
PLAN_FILE = ROOT / "data" / "output" / "subtitle_edit_plan_v6.json"

OUTPUT_DIR = ROOT / "data" / "output"
OUTPUT_SRT = OUTPUT_DIR / "localized_review_v7.srt"
OUTPUT_REPORT = OUTPUT_DIR / "subtitle_generation_report_v7.json"


TIMESTAMP_PATTERN = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})$"
)


def timestamp_to_ms(hours, minutes, seconds, milliseconds):
    return (
        int(hours) * 3600000
        + int(minutes) * 60000
        + int(seconds) * 1000
        + int(milliseconds)
    )


def seconds_to_ms(value):
    return int(
        (Decimal(str(value)) * 1000).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP
        )
    )


def ms_to_timestamp(total_ms):
    hours = total_ms // 3600000
    total_ms %= 3600000

    minutes = total_ms // 60000
    total_ms %= 60000

    seconds = total_ms // 1000
    milliseconds = total_ms % 1000

    return (
        f"{hours:02}:{minutes:02}:"
        f"{seconds:02},{milliseconds:03}"
    )


def parse_srt(path):
    content = path.read_text(encoding="utf-8-sig")

    blocks = re.split(r"\r?\n\s*\r?\n", content.strip())

    subtitles = []

    for block in blocks:
        lines = block.splitlines()

        if len(lines) < 3:
            raise ValueError(
                f"Invalid SRT block: {block[:100]}"
            )

        try:
            original_id = int(lines[0].strip())
        except ValueError:
            raise ValueError(
                f"Invalid subtitle number: {lines[0]}"
            )

        match = TIMESTAMP_PATTERN.match(lines[1].strip())

        if not match:
            raise ValueError(
                f"Invalid timestamp in subtitle {original_id}"
            )

        values = match.groups()

        start_ms = timestamp_to_ms(*values[:4])
        end_ms = timestamp_to_ms(*values[4:])

        if end_ms <= start_ms:
            raise ValueError(
                f"Invalid duration in subtitle {original_id}"
            )

        text = "\n".join(lines[2:])

        subtitles.append({
            "original_id": original_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text
        })

    ids = [s["original_id"] for s in subtitles]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate subtitle IDs found.")

    return subtitles


def load_plan(path):
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    stats = plan.get("statistics", {})

    if stats.get("ready_for_srt_generation") is not True:
        raise ValueError(
            "V6 plan is not ready for SRT generation."
        )

    if stats.get("errors") != 0:
        raise ValueError(
            "V6 plan contains errors."
        )

    if plan.get("errors"):
        raise ValueError(
            "V6 error list is not empty."
        )

    if len(plan.get("edits", [])) != stats.get("total_edits"):
        raise ValueError(
            "V6 edit count does not match statistics."
        )

    return plan


def validate_text(original, expected, subtitle_id):
    if original != expected:
        raise ValueError(
            f"Text mismatch for subtitle {subtitle_id}.\n"
            f"Original: {original!r}\n"
            f"Expected: {expected!r}"
        )


def generate():
    print("SUBTITLE GENERATOR V7")
    print("---------------------")

    print("Loading original SRT...")
    subtitles = parse_srt(SRT_FILE)

    print(f"Original subtitles: {len(subtitles)}")

    print("Loading V6 edit plan...")
    plan = load_plan(PLAN_FILE)

    by_id = {
        s["original_id"]: s
        for s in subtitles
    }

    new_count = 0
    updated_count = 0
    applied_edits = []

    print("Applying approved edits...")

    for edit in plan["edits"]:
        operation = edit.get("operation")

        if operation == "ADD_SUBTITLE":
            new_id = edit["new_subtitle_id"]

            if any(s.get("source_id") == new_id for s in subtitles):
                raise ValueError(
                    f"Duplicate new subtitle: {new_id}"
                )

            timing = edit["timing"]

            start_ms = seconds_to_ms(
                timing["start_seconds"]
            )
            end_ms = seconds_to_ms(
                timing["end_seconds"]
            )

            if end_ms <= start_ms:
                raise ValueError(
                    f"Invalid timing for {new_id}"
                )

            subtitles.append({
                "original_id": None,
                "source_id": new_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": edit["text"]
            })

            new_count += 1

            applied_edits.append({
                "edit_id": edit["edit_id"],
                "operation": operation,
                "new_subtitle_id": new_id,
                "start_ms": start_ms,
                "end_ms": end_ms
            })

        elif operation == "UPDATE_SUBTITLE_TIMING":
            subtitle_id = edit["subtitle_id"]

            if subtitle_id not in by_id:
                raise ValueError(
                    f"Subtitle {subtitle_id} not found."
                )

            subtitle = by_id[subtitle_id]

            validate_text(
                subtitle["text"],
                edit["text"],
                subtitle_id
            )

            timing = edit["proposed_timing"]

            start_ms = seconds_to_ms(
                timing["start_seconds"]
            )
            end_ms = seconds_to_ms(
                timing["end_seconds"]
            )

            if end_ms <= start_ms:
                raise ValueError(
                    f"Invalid updated timing for {subtitle_id}"
                )

            subtitle["start_ms"] = start_ms
            subtitle["end_ms"] = end_ms

            updated_count += 1

            applied_edits.append({
                "edit_id": edit["edit_id"],
                "operation": operation,
                "subtitle_id": subtitle_id,
                "start_ms": start_ms,
                "end_ms": end_ms
            })

        else:
            raise ValueError(
                f"Unsupported operation: {operation}"
            )

    expected_new = plan["statistics"]["new_subtitles"]
    expected_updated = plan["statistics"]["updated_subtitle_timings"]

    if new_count != expected_new:
        raise ValueError("New subtitle count mismatch.")

    if updated_count != expected_updated:
        raise ValueError("Updated subtitle count mismatch.")

    print("Sorting subtitles by start time...")

    subtitles.sort(
        key=lambda s: (
            s["start_ms"],
            s["end_ms"],
            s["original_id"] or 0
        )
    )

    output_blocks = []

    for index, subtitle in enumerate(subtitles, start=1):
        start = ms_to_timestamp(subtitle["start_ms"])
        end = ms_to_timestamp(subtitle["end_ms"])

        block = (
            f"{index}\n"
            f"{start} --> {end}\n"
            f"{subtitle['text']}"
        )

        output_blocks.append(block)

    output_content = "\n\n".join(output_blocks) + "\n"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_SRT.write_text(
        output_content,
        encoding="utf-8-sig",
        newline=""
    )

    # Re-read generated SRT for verification.
    verified = parse_srt(OUTPUT_SRT)

    if len(verified) != len(subtitles):
        raise ValueError(
            "Generated SRT subtitle count mismatch."
        )

    if [s["start_ms"] for s in verified] != sorted(
        s["start_ms"] for s in verified
    ):
        raise ValueError(
            "Generated SRT is not chronologically sorted."
        )

    # Check whether any subtitle overlaps the next one.
    overlaps = []

    for i in range(len(verified) - 1):
        current = verified[i]
        following = verified[i + 1]

        if current["end_ms"] > following["start_ms"]:
            overlaps.append({
                "subtitle_number": i + 1,
                "next_subtitle_number": i + 2,
                "overlap_ms": (
                    current["end_ms"] - following["start_ms"]
                )
            })

    report = {
        "schema_version": "1.0.0",
        "engine_version": "7.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "original_srt": str(SRT_FILE),
            "edit_plan": str(PLAN_FILE)
        },
        "output": str(OUTPUT_SRT),
        "statistics": {
            "original_subtitles": len(by_id),
            "new_subtitles": new_count,
            "updated_subtitle_timings": updated_count,
            "total_subtitles": len(verified),
            "applied_edits": len(applied_edits),
            "overlapping_adjacent_cues": len(overlaps),
            "verification": "PASSED"
        },
        "applied_edits": applied_edits,
        "overlaps": overlaps,
        "original_srt_modified": False
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nSUBTITLE GENERATION V7: OK")
    print("---------------------------")

    print(json.dumps(
        report["statistics"],
        ensure_ascii=False,
        indent=2
    ))

    print("\nOUTPUT SRT:")
    print(OUTPUT_SRT)

    print("\nQA REPORT:")
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    try:
        generate()
    except Exception as error:
        print("\nSUBTITLE GENERATION FAILED")
        print(type(error).__name__ + ":", error)
        raise
