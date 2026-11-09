
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "visual_features_v1"
    / "transition_features_v1.json"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "transition_inspection_v1"
    / "transition_ground_truth_v1.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "visual_features_v1"
    / "transition_signal_charts_v1"
)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_list(data, possible_keys):
    if isinstance(data, dict):
        for key in possible_keys:
            value = data.get(key)
            if isinstance(value, list):
                return value

        for value in data.values():
            result = find_list(value, possible_keys)
            if result is not None:
                return result

    elif isinstance(data, list):
        for item in data:
            result = find_list(item, possible_keys)
            if result is not None:
                return result

    return None


def get_value(item, possible_keys):
    if not isinstance(item, dict):
        return None

    for key in possible_keys:
        if key in item:
            return item[key]

    return None


def get_timestamp(item):
    value = get_value(
        item,
        [
            "timestamp_seconds",
            "time_seconds",
            "timestamp",
            "time",
            "at_seconds",
        ],
    )

    if value is None:
        start = get_value(
            item,
            ["start_seconds", "start_time", "from_seconds"],
        )
        end = get_value(
            item,
            ["end_seconds", "end_time", "to_seconds"],
        )

        if start is not None and end is not None:
            return (float(start) + float(end)) / 2

        return None

    return float(value)


def get_window_id(window, index):
    value = get_value(
        window,
        ["window_id", "transition_window_id", "id"],
    )
    return str(value) if value is not None else f"WINDOW_{index:03d}"


def get_window_data(features):
    windows = find_list(
        features,
        ["windows", "transition_windows", "results"],
    )

    if windows is None:
        raise ValueError(
            "Could not find the windows list in transition_features_v1.json"
        )

    return windows


def get_events(ground_truth):
    events = find_list(
        ground_truth,
        ["events", "ground_truth_events", "transitions"],
    )

    if events is None:
        raise ValueError(
            "Could not find the events list in transition_ground_truth_v1.json"
        )

    return events


def event_overlaps_window(event, frame_times):
    start = get_value(
        event,
        ["start_seconds", "start_time", "start"],
    )
    end = get_value(
        event,
        ["end_seconds", "end_time", "end"],
    )

    if start is None or end is None or not frame_times:
        return False

    return float(end) >= min(frame_times) and float(start) <= max(frame_times)


def event_label(event):
    event_id = get_value(event, ["event_id", "id"]) or "EVENT"
    event_type = (
        get_value(event, ["event_type", "classification", "boundary_type"])
        or "UNKNOWN"
    )
    transition_type = (
        get_value(event, ["transition_type", "effect_type", "transition"])
        or ""
    )

    return f"{event_id}: {event_type} {transition_type}".strip()


def extract_metric_points(items, metric_keys):
    points = []

    for item in items:
        timestamp = get_timestamp(item)
        value = get_value(item, metric_keys)

        if timestamp is None or value is None:
            continue

        try:
            points.append((float(timestamp), float(value)))
        except (TypeError, ValueError):
            continue

    points.sort(key=lambda point: point[0])
    return points


def make_chart(window, index, events):
    window_id = get_window_id(window, index)

    frames = find_list(
        window,
        ["frames", "frame_features", "frame_data"],
    )

    comparisons = find_list(
        window,
        ["comparisons", "frame_comparisons", "signals"],
    )

    if frames is None:
        frames = []

    if comparisons is None:
        comparisons = []

    brightness = extract_metric_points(
        frames,
        ["brightness_mean", "mean_brightness", "brightness"],
    )

    brightness_delta = extract_metric_points(
        comparisons,
        [
            "absolute_brightness_delta",
            "brightness_delta",
            "brightness_difference",
        ],
    )

    color_distance = extract_metric_points(
        comparisons,
        [
            "color_histogram_distance",
            "histogram_distance",
            "color_distance",
        ],
    )

    edge_difference = extract_metric_points(
        comparisons,
        [
            "edge_difference",
            "edge_density_difference",
            "edge_delta",
        ],
    )

    similarity = extract_metric_points(
        comparisons,
        [
            "structural_similarity",
            "ssim",
            "local_similarity",
        ],
    )

    frame_times = [point[0] for point in brightness]

    if not frame_times:
        frame_times = [
            get_timestamp(item)
            for item in frames
            if get_timestamp(item) is not None
        ]

    if not frame_times:
        frame_times = [
            get_timestamp(item)
            for item in comparisons
            if get_timestamp(item) is not None
        ]

    if not frame_times:
        print(f"WARNING: {window_id} has no usable timestamps.")
        return None

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15, 13),
        sharex=True,
    )

    fig.suptitle(
        f"Transition Signal Analysis - {window_id}",
        fontsize=15,
        fontweight="bold",
    )

    panels = [
        (axes[0], brightness, "Brightness Mean"),
        (axes[1], brightness_delta, "Absolute Brightness Delta"),
        (axes[2], color_distance, "Color Histogram Distance"),
        (axes[3], edge_difference, "Edge Difference"),
        (axes[4], similarity, "Structural Similarity"),
    ]

    for axis, points, title in panels:
        if points:
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]

            axis.plot(
                x_values,
                y_values,
                marker=".",
                linewidth=1.4,
                markersize=4,
            )
        else:
            axis.text(
                0.5,
                0.5,
                "Metric data unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

        axis.set_ylabel(title, fontsize=9)
        axis.grid(True, alpha=0.3)

    overlapping_events = [
        event
        for event in events
        if event_overlaps_window(event, frame_times)
    ]

    for event in overlapping_events:
        start = float(
            get_value(event, ["start_seconds", "start_time", "start"])
        )
        end = float(
            get_value(event, ["end_seconds", "end_time", "end"])
        )

        label = event_label(event)

        for axis in axes:
            axis.axvspan(start, end, alpha=0.15)
        
        axes[0].text(
            (start + end) / 2,
            0.96,
            label,
            rotation=90,
            ha="center",
            va="top",
            fontsize=7,
            transform=axes[0].get_xaxis_transform(),
        )

    axes[-1].set_xlabel("Video Time (seconds)")
    axes[-1].set_xlim(min(frame_times), max(frame_times))

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{window_id}_signals.png"

    fig.savefig(
        output_file,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "window_id": window_id,
        "frame_count": len(frames),
        "comparison_count": len(comparisons),
        "events_annotated": len(overlapping_events),
        "chart_file": str(output_file),
        "metric_points": {
            "brightness_mean": len(brightness),
            "absolute_brightness_delta": len(brightness_delta),
            "color_histogram_distance": len(color_distance),
            "edge_difference": len(edge_difference),
            "structural_similarity": len(similarity),
        },
    }


def main():
    print("TRANSITION SIGNAL VISUALIZER V1")
    print("--------------------------------")

    try:
        features = load_json(FEATURES_FILE)
        ground_truth = load_json(GROUND_TRUTH_FILE)

        windows = get_window_data(features)
        events = get_events(ground_truth)

        print(f"Windows: {len(windows)}")
        print(f"Ground truth events: {len(events)}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        manifest = {
            "schema_version": "1.0.0",
            "engine_version": "transition_signal_visualizer_v1",
            "total_windows": len(windows),
            "total_ground_truth_events": len(events),
            "charts": [],
        }

        for index, window in enumerate(windows, start=1):
            window_id = get_window_id(window, index)
            print(f"\nCreating chart: {window_id}")

            result = make_chart(window, index, events)

            if result is not None:
                manifest["charts"].append(result)
                print(f"Chart: {result['chart_file']}")
                print(
                    f"Frames: {result['frame_count']} | "
                    f"Comparisons: {result['comparison_count']} | "
                    f"Events: {result['events_annotated']}"
                )

        manifest_file = OUTPUT_DIR / "chart_manifest_v1.json"

        with manifest_file.open("w", encoding="utf-8") as file:
            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print("\nVISUALIZATION COMPLETE")
        print("----------------------")
        print(f"Charts created: {len(manifest['charts'])}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Manifest: {manifest_file}")

        if len(manifest["charts"]) != len(windows):
            print(
                "\nWARNING: Some windows did not produce charts. "
                "Check the metric data structure."
            )

    except Exception as error:
        print("\nVISUALIZATION FAILED")
        print("--------------------")
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
