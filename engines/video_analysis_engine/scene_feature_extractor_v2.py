
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRAME_METADATA = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "visual_frames"
    / "frame_metadata.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "visual_features_v1"
OUTPUT_FILE = OUTPUT_DIR / "visual_features_v2.json"

HISTOGRAM_BINS = 16
RESIZE_WIDTH = 320
RESIZE_HEIGHT = 180


def load_metadata():
    if not FRAME_METADATA.exists():
        raise FileNotFoundError(
            f"Frame metadata not found: {FRAME_METADATA}"
        )

    with open(FRAME_METADATA, "r", encoding="utf-8") as file:
        return json.load(file)


def get_frame_path(frame):
    for key in ("path", "file_path", "filename"):
        value = frame.get(key)

        if value:
            path = Path(value)

            if not path.is_absolute():
                path = PROJECT_ROOT / path

            return path

    raise ValueError(
        f"Frame path not found in metadata record: {frame}"
    )


def calculate_histogram(hsv_image):
    histograms = []

    channels = [
        (0, [0, 180]),
        (1, [0, 256]),
        (2, [0, 256]),
    ]

    for channel, value_range in channels:
        histogram = cv2.calcHist(
            [hsv_image],
            [channel],
            None,
            [HISTOGRAM_BINS],
            value_range,
        )

        histogram = cv2.normalize(
            histogram,
            None,
            alpha=1.0,
            norm_type=cv2.NORM_L1,
        )

        histograms.append(
            histogram.flatten().astype(float).tolist()
        )

    return histograms


def calculate_features(frame_path):
    image = cv2.imread(str(frame_path))

    if image is None:
        raise ValueError(f"Cannot read image: {frame_path}")

    image = cv2.resize(
        image,
        (RESIZE_WIDTH, RESIZE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray))
    brightness_std = float(np.std(gray))

    near_white_ratio = float(
        np.mean(gray >= 245)
    )

    near_black_ratio = float(
        np.mean(gray <= 10)
    )

    edges = cv2.Canny(gray, 80, 160)

    edge_density = float(
        np.count_nonzero(edges) / edges.size
    )

    histogram = calculate_histogram(hsv)

    return {
        "brightness_mean": round(brightness, 4),
        "brightness_std": round(brightness_std, 4),
        "near_white_ratio": round(near_white_ratio, 6),
        "near_black_ratio": round(near_black_ratio, 6),
        "edge_density": round(edge_density, 6),
        "color_histogram_hsv": histogram,
        "gray_image": gray,
        "edge_image": edges,
    }


def calculate_histogram_distance(hist_a, hist_b):
    distances = []

    for channel_a, channel_b in zip(hist_a, hist_b):
        distance = cv2.compareHist(
            np.asarray(channel_a, dtype=np.float32),
            np.asarray(channel_b, dtype=np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )

        distances.append(float(distance))

    return float(np.mean(distances))


def calculate_structural_similarity(gray_a, gray_b):
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = (
        cv2.GaussianBlur(a * a, (11, 11), 1.5)
        - mu_a_sq
    )

    sigma_b_sq = (
        cv2.GaussianBlur(b * b, (11, 11), 1.5)
        - mu_b_sq
    )

    sigma_ab = (
        cv2.GaussianBlur(a * b, (11, 11), 1.5)
        - mu_ab
    )

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    numerator = (
        (2 * mu_ab + c1)
        * (2 * sigma_ab + c2)
    )

    denominator = (
        (mu_a_sq + mu_b_sq + c1)
        * (sigma_a_sq + sigma_b_sq + c2)
    )

    similarity_map = numerator / (denominator + 1e-8)

    return float(np.mean(similarity_map))


def calculate_edge_difference(edge_a, edge_b):
    difference = cv2.absdiff(edge_a, edge_b)

    return float(
        np.count_nonzero(difference) / difference.size
    )


def main():
    print("SCENE FEATURE EXTRACTOR V2")
    print("--------------------------")

    print("Loading frame metadata...")

    metadata = load_metadata()
    frames = metadata.get("frames", [])

    if not frames:
        raise ValueError(
            "No frames found in frame_metadata.json"
        )

    frames = sorted(
        frames,
        key=lambda item: float(
            item.get("timestamp_seconds", item.get("timestamp", 0))
        ),
    )

    print(f"Total frames: {len(frames)}")
    print("Extracting visual features...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    previous_features = None

    for index, frame in enumerate(frames):
        frame_id = frame.get(
            "frame_id",
            frame.get("id", f"VF_{index + 1:06d}"),
        )

        timestamp = float(
            frame.get(
                "timestamp_seconds",
                frame.get("timestamp", index),
            )
        )

        frame_path = get_frame_path(frame)

        features = calculate_features(frame_path)

        record = {
            "frame_id": frame_id,
            "timestamp_seconds": timestamp,
            "features": {
                "brightness_mean": features["brightness_mean"],
                "brightness_std": features["brightness_std"],
                "near_white_ratio": features["near_white_ratio"],
                "near_black_ratio": features["near_black_ratio"],
                "edge_density": features["edge_density"],
                "color_histogram_hsv": features[
                    "color_histogram_hsv"
                ],
            },
            "changes_from_previous": None,
        }

        if previous_features is not None:
            brightness_delta = (
                features["brightness_mean"]
                - previous_features["brightness_mean"]
            )

            histogram_distance = calculate_histogram_distance(
                features["color_histogram_hsv"],
                previous_features["color_histogram_hsv"],
            )

            structural_similarity = (
                calculate_structural_similarity(
                    features["gray_image"],
                    previous_features["gray_image"],
                )
            )

            edge_difference = calculate_edge_difference(
                features["edge_image"],
                previous_features["edge_image"],
            )

            record["changes_from_previous"] = {
                "brightness_delta": round(
                    brightness_delta, 4
                ),
                "absolute_brightness_delta": round(
                    abs(brightness_delta), 4
                ),
                "color_histogram_distance": round(
                    histogram_distance, 6
                ),
                "structural_similarity": round(
                    structural_similarity, 6
                ),
                "edge_difference": round(
                    edge_difference, 6
                ),
            }

        results.append(record)
        previous_features = features

        if (index + 1) % 50 == 0:
            print(
                f"Processed {index + 1}/{len(frames)} frames"
            )

    output = {
        "schema_version": "1.0.0",
        "engine_version": "2.0.0",
        "source": {
            "frame_metadata": str(FRAME_METADATA),
            "frame_count": len(results),
            "resolution_used": [
                RESIZE_WIDTH,
                RESIZE_HEIGHT,
            ],
        },
        "method": {
            "brightness": "GRAYSCALE_MEAN_AND_STD",
            "color": "HSV_HISTOGRAM_BHATTACHARYYA",
            "edges": "CANNY",
            "structure": "LOCAL_SSIM_LIKE",
        },
        "statistics": {
            "total_frames": len(results),
            "frame_comparisons": max(
                0, len(results) - 1
            ),
        },
        "frames": results,
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
    print("FEATURE EXTRACTION: OK")
    print("----------------------")
    print(f"Total frames: {len(results)}")
    print(
        f"Frame comparisons: {max(0, len(results) - 1)}"
    )
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("FEATURE EXTRACTION: FAILED")
        print(f"Error: {error}")
        sys.exit(1)
