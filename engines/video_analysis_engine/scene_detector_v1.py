
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

FRAME_METADATA_FILE = (
    ROOT / "data" / "output" / "visual_frames" / "frame_metadata.json"
)

OUTPUT_FILE = (
    ROOT / "data" / "output" / "scene_detection_v1.json"
)

# Difference score above this threshold becomes a candidate.
SCENE_CHANGE_THRESHOLD = 0.30

# Resize frames before comparison to reduce processing time.
RESIZE_WIDTH = 160
RESIZE_HEIGHT = 90


def load_frame_metadata():
    if not FRAME_METADATA_FILE.exists():
        raise FileNotFoundError(FRAME_METADATA_FILE)

    with open(FRAME_METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    frames = metadata.get("frames", [])

    if len(frames) < 2:
        raise ValueError("At least two frames are required.")

    return frames


def load_frame_image(frame_record):
    frame_path = Path(frame_record["path"])

    if not frame_path.exists():
        raise FileNotFoundError(frame_path)

    with Image.open(frame_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            (RESIZE_WIDTH, RESIZE_HEIGHT),
            Image.Resampling.LANCZOS
        )

        return np.asarray(image, dtype=np.float32) / 255.0


def calculate_difference(previous, current):
    """
    Calculate normalized mean absolute pixel difference.
    Result range: 0.0 to 1.0.
    """

    difference = np.abs(current - previous)

    return float(np.mean(difference))


def classify_change(score):
    if score >= 0.50:
        return "STRONG_CHANGE"

    if score >= SCENE_CHANGE_THRESHOLD:
        return "POSSIBLE_SCENE_CHANGE"

    return "NO_SIGNIFICANT_CHANGE"


def analyze_frames(frames):
    results = []
    candidates = []

    print("Comparing consecutive frames...")

    previous_image = None

    for index, frame_record in enumerate(frames):
        current_image = load_frame_image(frame_record)

        if previous_image is not None:
            score = calculate_difference(
                previous_image,
                current_image
            )

            timestamp = float(
                frame_record["timestamp_seconds"]
            )

            classification = classify_change(score)

            result = {
                "previous_frame_id": frames[index - 1]["frame_id"],
                "current_frame_id": frame_record["frame_id"],
                "timestamp_seconds": timestamp,
                "difference_score": round(score, 6),
                "classification": classification
            }

            results.append(result)

            if classification != "NO_SIGNIFICANT_CHANGE":
                candidates.append(result)

        previous_image = current_image

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{len(frames)} frames")

    return results, candidates


def main():
    print("SCENE DETECTOR V1")
    print("-----------------")

    print("Loading frame metadata...")
    frames = load_frame_metadata()

    print("Total frames:", len(frames))
    print("Threshold:", SCENE_CHANGE_THRESHOLD)

    comparisons, candidates = analyze_frames(frames)

    strong_changes = [
        item for item in candidates
        if item["classification"] == "STRONG_CHANGE"
    ]

    possible_changes = [
        item for item in candidates
        if item["classification"] == "POSSIBLE_SCENE_CHANGE"
    ]

    output = {
        "schema_version": "1.0.0",
        "engine_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),

        "source": {
            "frame_metadata": str(FRAME_METADATA_FILE)
        },

        "method": {
            "name": "CONSECUTIVE_FRAME_PIXEL_DIFFERENCE",
            "resize_width": RESIZE_WIDTH,
            "resize_height": RESIZE_HEIGHT,
            "threshold": SCENE_CHANGE_THRESHOLD
        },

        "statistics": {
            "total_frames": len(frames),
            "total_comparisons": len(comparisons),
            "candidate_scene_changes": len(candidates),
            "strong_changes": len(strong_changes),
            "possible_changes": len(possible_changes)
        },

        "comparisons": comparisons,
        "scene_change_candidates": candidates
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nSCENE DETECTION: OK")
    print("-------------------")
    print("Total comparisons:", len(comparisons))
    print("Candidate changes:", len(candidates))
    print("Strong changes:", len(strong_changes))
    print("Possible changes:", len(possible_changes))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
