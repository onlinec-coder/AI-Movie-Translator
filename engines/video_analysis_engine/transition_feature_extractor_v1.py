
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "transition_inspection_v1"
    / "transition_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "visual_features_v1"
)

OUTPUT_FILE = OUTPUT_DIR / "transition_features_v1.json"

WIDTH = 320
HEIGHT = 180
HISTOGRAM_BINS = 16


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_histogram(hsv):
    result = []

    for channel, value_range in [
        (0, [0, 180]),
        (1, [0, 256]),
        (2, [0, 256]),
    ]:
        hist = cv2.calcHist(
            [hsv],
            [channel],
            None,
            [HISTOGRAM_BINS],
            value_range,
        )

        hist = cv2.normalize(
            hist,
            None,
            alpha=1.0,
            norm_type=cv2.NORM_L1,
        )

        result.append(hist.flatten().astype(float).tolist())

    return result


def histogram_distance(hist_a, hist_b):
    distances = []

    for a, b in zip(hist_a, hist_b):
        distance = cv2.compareHist(
            np.asarray(a, dtype=np.float32),
            np.asarray(b, dtype=np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
        distances.append(float(distance))

    return float(np.mean(distances))


def calculate_ssim(gray_a, gray_b):
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    sigma_a = (
        cv2.GaussianBlur(a * a, (11, 11), 1.5)
        - mu_a * mu_a
    )

    sigma_b = (
        cv2.GaussianBlur(b * b, (11, 11), 1.5)
        - mu_b * mu_b
    )

    sigma_ab = (
        cv2.GaussianBlur(a * b, (11, 11), 1.5)
        - mu_a * mu_b
    )

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    numerator = (
        (2 * mu_a * mu_b + c1)
        * (2 * sigma_ab + c2)
    )

    denominator = (
        (mu_a * mu_a + mu_b * mu_b + c1)
        * (sigma_a + sigma_b + c2)
    )

    return float(
        np.mean(numerator / (denominator + 1e-8))
    )


def analyze_frame(frame_path):
    image = cv2.imread(str(frame_path))

    if image is None:
        raise ValueError(f"Cannot read frame: {frame_path}")

    image = cv2.resize(
        image,
        (WIDTH, HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(gray, 80, 160)

    features = {
        "brightness_mean": float(np.mean(gray)),
        "brightness_std": float(np.std(gray)),
        "near_white_ratio": float(np.mean(gray >= 245)),
        "near_black_ratio": float(np.mean(gray <= 10)),
        "edge_density": float(np.mean(edges > 0)),
        "color_histogram_hsv": calculate_histogram(hsv),
    }

    return {
        "features": features,
        "gray": gray,
        "edges": edges,
    }


def compare_frames(previous, current):
    a = previous["features"]
    b = current["features"]

    brightness_delta = (
        b["brightness_mean"] - a["brightness_mean"]
    )

    edge_difference = cv2.absdiff(
        previous["edges"],
        current["edges"],
    )

    return {
        "brightness_delta": round(brightness_delta, 4),
        "absolute_brightness_delta": round(
            abs(brightness_delta), 4
        ),
        "color_histogram_distance": round(
            histogram_distance(
                a["color_histogram_hsv"],
                b["color_histogram_hsv"],
            ),
            6,
        ),
        "structural_similarity": round(
            calculate_ssim(
                previous["gray"],
                current["gray"],
            ),
            6,
        ),
        "edge_difference": round(
            float(np.mean(edge_difference > 0)),
            6,
        ),
    }


def main():
    print("TRANSITION FEATURE EXTRACTOR V1")
    print("--------------------------------")

    metadata = load_json(METADATA_FILE)

    windows = metadata.get("windows", [])

    if not windows:
        raise ValueError("No transition windows found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_windows = []
    total_frames = 0
    total_comparisons = 0

    for window in windows:
        window_id = window["window_id"]
        frame_records = window.get("frames", [])

        print(
            f"\nProcessing {window_id}: "
            f"{len(frame_records)} frames"
        )

        analyzed = []

        for frame_record in frame_records:
            frame_path = Path(frame_record["path"])

            if not frame_path.is_absolute():
                frame_path = PROJECT_ROOT / frame_path

            result = analyze_frame(frame_path)

            analyzed.append({
                "frame_id": frame_record["frame_id"],
                "timestamp_seconds": float(
                    frame_record["timestamp_seconds_approx"]
                ),
                "features": result["features"],
                "_image_data": {
                    "gray": result["gray"],
                    "edges": result["edges"],
                },
            })

        comparisons = []

        for index in range(1, len(analyzed)):
            previous = analyzed[index - 1]
            current = analyzed[index]

            changes = compare_frames(
                {
                    "features": previous["features"],
                    **previous["_image_data"],
                },
                {
                    "features": current["features"],
                    **current["_image_data"],
                },
            )

            comparisons.append({
                "from_frame": previous["frame_id"],
                "to_frame": current["frame_id"],
                "timestamp_seconds": current[
                    "timestamp_seconds"
                ],
                "changes": changes,
            })

        clean_frames = []

        for frame in analyzed:
            clean_frames.append({
                "frame_id": frame["frame_id"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "features": frame["features"],
            })

        output_windows.append({
            "window_id": window_id,
            "start_seconds": window["start_seconds"],
            "end_seconds": window["end_seconds"],
            "total_frames": len(clean_frames),
            "frame_comparisons": len(comparisons),
            "frames": clean_frames,
            "comparisons": comparisons,
        })

        total_frames += len(clean_frames)
        total_comparisons += len(comparisons)

        print(
            f"Frames: {len(clean_frames)} | "
            f"Comparisons: {len(comparisons)}"
        )

    output = {
        "schema_version": "1.0.0",
        "engine_version": "1.0.0",
        "source": {
            "metadata_file": str(METADATA_FILE),
            "sampling_fps": metadata["sampling"][
                "frames_per_second"
            ],
            "sampling_interval_seconds": metadata[
                "sampling"
            ]["interval_seconds"],
        },
        "method": {
            "brightness": "GRAYSCALE_MEAN_AND_STD",
            "color": "HSV_HISTOGRAM_BHATTACHARYYA",
            "edges": "CANNY",
            "structure": "LOCAL_SSIM_LIKE",
        },
        "statistics": {
            "total_windows": len(output_windows),
            "total_frames": total_frames,
            "total_comparisons": total_comparisons,
        },
        "windows": output_windows,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("TRANSITION FEATURE EXTRACTION: OK")
    print("----------------------------------")
    print(f"Windows: {len(output_windows)}")
    print(f"Frames: {total_frames}")
    print(f"Comparisons: {total_comparisons}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("TRANSITION FEATURE EXTRACTION: FAILED")
        print(f"Error: {error}")
        sys.exit(1)
