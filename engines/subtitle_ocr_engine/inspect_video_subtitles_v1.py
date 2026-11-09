
import sys
import json
import cv2

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENGINE_DIR = PROJECT_ROOT / "engines" / "subtitle_ocr_engine"
sys.path.insert(0, str(ENGINE_DIR))

import subtitle_ocr_engine_v7 as ocr_v7


VIDEO_FILE = PROJECT_ROOT / "data" / "input" / "Husbands.mp4"

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "subtitle_ocr_inspection"
    / "source_comparison_v5.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "subtitle_ocr_inspection"
    / "video_subtitle_review_v1"
)

OUTPUT_JSON = OUTPUT_DIR / "video_subtitle_review_v1.json"

TARGET_IDS = [
    27, 32, 76, 112, 113, 128, 177, 240, 439,
    723, 744, 746, 753, 757, 770, 816, 825
]

SAMPLE_INTERVAL = 0.25
TIME_PADDING = 0.75

CROP_TOP = 0.60
CROP_BOTTOM = 0.96
CROP_LEFT = 0.03
CROP_RIGHT = 0.97


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def main():
    if not VIDEO_FILE.exists():
        raise FileNotFoundError(VIDEO_FILE)

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(SOURCE_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_data = load_json(SOURCE_FILE)

    cue_map = {
        int(item["subtitle_id"]): item
        for item in source_data["results"]
    }

    missing = [
        cue_id for cue_id in TARGET_IDS
        if cue_id not in cue_map
    ]

    if missing:
        raise RuntimeError(f"Missing cue IDs: {missing}")

    cap = cv2.VideoCapture(str(VIDEO_FILE))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS.")

    results = []

    print("VIDEO SUBTITLE REVIEW V1")
    print("========================")
    print(f"FPS: {fps}")
    print(f"Target cues: {len(TARGET_IDS)}")
    print()

    try:
        for cue_id in TARGET_IDS:
            cue = cue_map[cue_id]

            start = float(cue["start_seconds"])
            end = float(cue["end_seconds"])

            window_start = max(0, start - TIME_PADDING)
            window_end = end + TIME_PADDING

            cue_dir = OUTPUT_DIR / f"cue_{cue_id:04d}"
            cue_dir.mkdir(parents=True, exist_ok=True)

            frames = []
            timestamp = window_start

            print(
                f"CUE {cue_id}: "
                f"{start:.3f}-{end:.3f}"
            )

            while timestamp <= window_end + 0.001:
                frame_number = int(round(timestamp * fps))

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_number
                )

                success, frame = cap.read()

                if not success:
                    timestamp += SAMPLE_INTERVAL
                    continue

                height, width = frame.shape[:2]

                x1 = int(width * CROP_LEFT)
                x2 = int(width * CROP_RIGHT)
                y1 = int(height * CROP_TOP)
                y2 = int(height * CROP_BOTTOM)

                subtitle_crop = frame[y1:y2, x1:x2]

                processed = ocr_v7.prepare_frame(frame)

                frame_result = {
                    "timestamp": round(timestamp, 3),
                    "ocr": {}
                }

                for psm in ocr_v7.OCR_MODES:
                    words = ocr_v7.read_ocr_words(
                        processed,
                        psm
                    )

                    frame_result["ocr"][str(psm)] = {
                        "text": " ".join(
                            word["text"] for word in words
                        ),
                        "words": words
                    }

                image_name = (
                    f"{timestamp:010.3f}".replace(".", "_")
                    + ".jpg"
                )

                image_path = cue_dir / image_name

                cv2.imwrite(
                    str(image_path),
                    subtitle_crop
                )

                frame_result["image"] = str(
                    image_path.relative_to(PROJECT_ROOT)
                )

                frames.append(frame_result)

                timestamp += SAMPLE_INTERVAL

            results.append({
                "subtitle_id": cue_id,
                "start_seconds": start,
                "end_seconds": end,
                "whisper_text": cue["whisper"]["text"],
                "assemblyai_text": (
                    cue["assemblyai_time_alignment"]["matched_text"]
                ),
                "window_start": round(window_start, 3),
                "window_end": round(window_end, 3),
                "frames": frames
            })

            print(f"  Frames captured: {len(frames)}")

    finally:
        cap.release()

    output = {
        "engine": "VIDEO SUBTITLE REVIEW V1",
        "video": str(VIDEO_FILE),
        "sample_interval": SAMPLE_INTERVAL,
        "time_padding": TIME_PADDING,
        "ocr_modes": ocr_v7.OCR_MODES,
        "results": results
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("VIDEO SUBTITLE REVIEW COMPLETE")
    print(f"Cues: {len(results)}")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"Images: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
