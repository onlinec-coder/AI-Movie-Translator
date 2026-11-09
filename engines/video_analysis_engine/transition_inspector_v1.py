
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]

VIDEO_FILE = ROOT / "data" / "input" / "Husbands.mp4"

OUTPUT_DIR = ROOT / "data" / "output" / "transition_inspection_v1"
METADATA_FILE = OUTPUT_DIR / "transition_metadata.json"

SAMPLE_INTERVAL = 0.2

INSPECTION_WINDOWS = [
    {"window_id": "TW_001", "start": 7.0, "end": 11.0},
    {"window_id": "TW_002", "start": 108.0, "end": 113.0},
    {"window_id": "TW_003", "start": 244.0, "end": 249.0},
    {"window_id": "TW_004", "start": 287.0, "end": 292.0},
]


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg command failed:\n"
            + " ".join(command)
            + "\n\n"
            + result.stderr[-5000:]
        )

    return result


def extract_window(window):
    window_id = window["window_id"]

    window_dir = OUTPUT_DIR / window_id
    window_dir.mkdir(parents=True, exist_ok=True)

    for old_file in window_dir.glob("*.jpg"):
        old_file.unlink()

    output_pattern = window_dir / "frame_%03d.jpg"

    duration = window["end"] - window["start"]

    print(
        f"Extracting {window_id}: "
        f"{window['start']}s - {window['end']}s"
    )

    run_command([
        "ffmpeg",
        "-y",
        "-ss", str(window["start"]),
        "-i", str(VIDEO_FILE),
        "-t", str(duration),
        "-vf", f"fps=5",
        "-q:v", "2",
        "-start_number", "0",
        str(output_pattern)
    ])

    image_files = sorted(window_dir.glob("frame_*.jpg"))

    records = []

    for index, image_path in enumerate(image_files):
        timestamp = round(
            window["start"] + index * SAMPLE_INTERVAL,
            3
        )

        records.append({
            "frame_id": f"{window_id}_F{index:03d}",
            "timestamp_seconds_approx": timestamp,
            "filename": image_path.name,
            "path": str(image_path)
        })

    return records


def main():
    print("TRANSITION INSPECTOR V1")
    print("-----------------------")

    if not VIDEO_FILE.exists():
        raise FileNotFoundError(VIDEO_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []

    for window in INSPECTION_WINDOWS:
        records = extract_window(window)

        print("Frames extracted:", len(records))

        all_records.append({
            "window_id": window["window_id"],
            "start_seconds": window["start"],
            "end_seconds": window["end"],
            "total_frames": len(records),
            "frames": records
        })

    output = {
        "schema_version": "1.0.0",
        "engine_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),

        "source": {
            "video": str(VIDEO_FILE)
        },

        "sampling": {
            "method": "FIXED_INTERVAL",
            "frames_per_second": 5,
            "interval_seconds": SAMPLE_INTERVAL,
            "timestamp_note": (
                "Approximate timestamps calculated from frame order. "
                "Verify exact transition timing against source video."
            )
        },

        "statistics": {
            "total_windows": len(all_records),
            "total_frames": sum(
                item["total_frames"] for item in all_records
            )
        },

        "windows": all_records
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nTRANSITION INSPECTION: OK")
    print("--------------------------")
    print("Total windows:", len(all_records))
    print("Total frames:", output["statistics"]["total_frames"])
    print("Output directory:", OUTPUT_DIR)
    print("Metadata:", METADATA_FILE)


if __name__ == "__main__":
    main()
