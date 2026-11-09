
import json
from pathlib import Path

from analysis_config import build_analysis_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRT_FILE = PROJECT_ROOT / "data" / "input" / "input.srt"

MEDIA_METADATA_FILE = (
    PROJECT_ROOT / "data" / "output" / "media_metadata.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "output" / "timeline_reference.json"
)


def main():
    print("Reading SRT...")

    if not MEDIA_METADATA_FILE.exists():
        raise FileNotFoundError(
            "media_metadata.json not found. "
            "Run media_engine.py first."
        )

    metadata = json.loads(
        MEDIA_METADATA_FILE.read_text(encoding="utf-8")
    )

    source_duration = metadata["source"]["duration_seconds"]

    plan = build_analysis_plan(
        SRT_FILE,
        source_duration=source_duration
    )

    extraction_duration = plan["extraction_duration"]

    actual_extraction_duration = metadata[
        "analysis_window"
    ]["extraction_duration_seconds"]

    if actual_extraction_duration < extraction_duration:
        raise RuntimeError(
            "Extracted media is shorter than the required duration. "
            "Run media_engine.py again."
        )

    result = {
        "schema_version": "1.0.0",
        "source": {
            "filename": metadata["source"]["filename"],
            "source_offset_seconds": 0.0
        },
        "analysis_window": {
            "start_seconds": plan["base_window_start"],
            "base_window_end_seconds": plan["base_window_end"],
            "last_subtitle_end_seconds": plan["last_subtitle_end"],
            "required_end_seconds": plan["required_end"],
            "extraction_duration_seconds": extraction_duration
        },
        "statistics": {
            "total_subtitles": len(plan["all_segments"]),
            "subtitles_in_window": len(plan["segments"])
        },
        "segments": plan["segments"]
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nTIMELINE REFERENCE: OK")
    print(f"Total subtitles: {len(plan['all_segments'])}")
    print(f"Subtitles in analysis window: {len(plan['segments'])}")
    print(f"Base window: {plan['base_window_end']} seconds")
    print(f"Required media duration: {extraction_duration} seconds")
    print(f"Actual extracted duration: {actual_extraction_duration} seconds")

    if plan["segments"]:
        first = plan["segments"][0]
        last = plan["segments"][-1]

        print(
            "First subtitle:",
            first["start_seconds"],
            "->",
            first["end_seconds"]
        )

        print(
            "Last subtitle:",
            last["start_seconds"],
            "->",
            last["end_seconds"]
        )

    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
