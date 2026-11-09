
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "visual_features_v1"
    / "visual_features_v2.json"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "transition_inspection_v1"
    / "transition_ground_truth_v1.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "visual_features_v1"
    / "feature_inspection_v2.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("SCENE FEATURE INSPECTOR V2")
    print("--------------------------")

    data = load_json(FEATURE_FILE)
    ground_truth = load_json(GROUND_TRUTH_FILE)

    frames = data["frames"]
    events = ground_truth["events"]

    comparisons = []

    for frame in frames:
        changes = frame.get("changes_from_previous")

        if changes is None:
            continue

        comparisons.append({
            "frame_id": frame["frame_id"],
            "timestamp_seconds": frame["timestamp_seconds"],
            **changes,
        })

    metrics = [
        "absolute_brightness_delta",
        "color_histogram_distance",
        "edge_difference",
    ]

    print(f"Frames: {len(frames)}")
    print(f"Comparisons: {len(comparisons)}")
    print()

    print("TOP 15 CHANGES BY METRIC")
    print("------------------------")

    rankings = {}

    for metric in metrics:
        ranked = sorted(
            comparisons,
            key=lambda item: item[metric],
            reverse=True,
        )

        rankings[metric] = ranked

        print()
        print(f"[{metric}]")

        for item in ranked[:15]:
            print(
                f"{item['timestamp_seconds']:7.2f}s | "
                f"{item['frame_id']} | "
                f"{item[metric]:.6f}"
            )

    print()
    print("GROUND TRUTH EVENT COMPARISON")
    print("-----------------------------")

    event_results = []

    for event in events:
        start = event["start_seconds"]
        end = event["end_seconds"]

        nearby = [
            item
            for item in comparisons
            if start - 2.0
            <= item["timestamp_seconds"]
            <= end + 2.0
        ]

        if not nearby:
            continue

        event_result = {
            "event_id": event["event_id"],
            "boundary_type": event["boundary_type"],
            "transition_effect": event["transition_effect"],
            "start_seconds": start,
            "end_seconds": end,
            "measurements": nearby,
        }

        event_results.append(event_result)

        print()
        print(
            f"{event['event_id']} | "
            f"{event['boundary_type']} | "
            f"{event['transition_effect']}"
        )

        for item in nearby:
            print(
                f"  {item['timestamp_seconds']:7.2f}s | "
                f"Brightness: "
                f"{item['absolute_brightness_delta']:.4f} | "
                f"Color: "
                f"{item['color_histogram_distance']:.4f} | "
                f"Edge: "
                f"{item['edge_difference']:.4f} | "
                f"SSIM: "
                f"{item['structural_similarity']:.4f}"
            )

    output = {
        "schema_version": "1.0.0",
        "engine_version": "2.0.0",
        "source": {
            "feature_file": str(FEATURE_FILE),
            "ground_truth_file": str(GROUND_TRUTH_FILE),
        },
        "top_changes": {
            metric: ranked[:15]
            for metric, ranked in rankings.items()
        },
        "ground_truth_comparison": event_results,
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
    print("FEATURE INSPECTION: OK")
    print("-----------------------")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
