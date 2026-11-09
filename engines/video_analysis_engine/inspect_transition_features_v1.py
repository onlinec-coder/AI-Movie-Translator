
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT / "data" / "output" / "visual_features_v1"
    / "transition_features_v1.json"
)

GROUND_TRUTH_FILE = (
    ROOT / "data" / "output" / "transition_inspection_v1"
    / "transition_ground_truth_v1.json"
)

OUTPUT_FILE = (
    ROOT / "data" / "output" / "visual_features_v1"
    / "transition_feature_inspection_v1.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("TRANSITION FEATURE INSPECTOR V1")
    print("--------------------------------")

    features = load_json(FEATURE_FILE)
    ground_truth = load_json(GROUND_TRUTH_FILE)

    windows = features["windows"]
    events = ground_truth["events"]

    all_comparisons = []

    for window in windows:
        for comparison in window["comparisons"]:
            item = {
                "window_id": window["window_id"],
                "from_frame": comparison["from_frame"],
                "to_frame": comparison["to_frame"],
                "timestamp_seconds": comparison[
                    "timestamp_seconds"
                ],
                **comparison["changes"],
            }

            all_comparisons.append(item)

    metrics = [
        "absolute_brightness_delta",
        "color_histogram_distance",
        "edge_difference",
    ]

    print(f"Windows: {len(windows)}")
    print(f"Comparisons: {len(all_comparisons)}")

    rankings = {}

    print("\nTOP 15 CHANGES BY METRIC")
    print("------------------------")

    for metric in metrics:
        ranked = sorted(
            all_comparisons,
            key=lambda item: item[metric],
            reverse=True,
        )

        rankings[metric] = ranked

        print(f"\n[{metric}]")

        for item in ranked[:15]:
            print(
                f"{item['timestamp_seconds']:7.2f}s | "
                f"{item['window_id']} | "
                f"{item['from_frame']} -> "
                f"{item['to_frame']} | "
                f"{item[metric]:.6f}"
            )

    print("\nGROUND TRUTH EVENT COMPARISON")
    print("-----------------------------")

    event_results = []

    for event in events:
        start = event["start_seconds"]
        end = event["end_seconds"]

        nearby = [
            item
            for item in all_comparisons
            if start - 0.6
            <= item["timestamp_seconds"]
            <= end + 0.6
        ]

        result = {
            "event_id": event["event_id"],
            "boundary_type": event["boundary_type"],
            "transition_effect": event["transition_effect"],
            "start_seconds": start,
            "end_seconds": end,
            "comparisons": nearby,
        }

        event_results.append(result)

        print(
            f"\n{event['event_id']} | "
            f"{event['boundary_type']} | "
            f"{event['transition_effect']}"
        )

        for item in nearby:
            print(
                f"  {item['timestamp_seconds']:7.2f}s | "
                f"Bright: "
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
        "engine_version": "1.0.0",
        "source": {
            "feature_file": str(FEATURE_FILE),
            "ground_truth_file": str(GROUND_TRUTH_FILE),
        },
        "statistics": {
            "total_comparisons": len(all_comparisons),
            "ground_truth_events": len(event_results),
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

    print("\nTRANSITION FEATURE INSPECTION: OK")
    print("----------------------------------")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
