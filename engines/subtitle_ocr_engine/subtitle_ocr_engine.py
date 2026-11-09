
import json
import re
from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_FILE = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "Husbands.mp4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "subtitle_ocr_inspection"
)

JSON_OUTPUT = OUTPUT_DIR / "subtitle_ocr_v1.json"
SRT_OUTPUT = OUTPUT_DIR / "subtitle_ocr_v1.srt"

# Test only the first 30 seconds.
TEST_DURATION_SECONDS = 30.0

# Sample one frame every 0.25 seconds.
SAMPLE_INTERVAL_SECONDS = 0.25

# Subtitle crop region, relative to frame dimensions.
CROP_X_START = 0.03
CROP_X_END = 0.97
CROP_Y_START = 0.68
CROP_Y_END = 0.84

OCR_CONFIG = "--oem 3 --psm 6"

# Consecutive detections of the same text are merged.
MAX_GAP_SECONDS = 0.75


def normalize_text(text):
    text = text.replace("|", "I")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text):
    text = normalize_text(text)

    # Remove common OCR noise.
    text = re.sub(r"^[^\w]+|[^\w.!?,'’-]+$", "", text)

    return text.strip()


def similarity(a, b):
    from difflib import SequenceMatcher

    return SequenceMatcher(
        None,
        a.lower(),
        b.lower()
    ).ratio()


def read_subtitle_from_frame(frame):

    height, width = frame.shape[:2]

    x1 = int(width * CROP_X_START)
    x2 = int(width * CROP_X_END)
    y1 = int(height * CROP_Y_START)
    y2 = int(height * CROP_Y_END)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return "", 0.0

    enlarged = cv2.resize(
        crop,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    # Improve text contrast while preserving black outlines.
    processed = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    data = pytesseract.image_to_data(
        processed,
        lang="eng",
        config=OCR_CONFIG,
        output_type=Output.DICT
    )

    words = []
    confidences = []

    for index, word in enumerate(data["text"]):

        word = clean_text(word)

        try:
            confidence = float(data["conf"][index])
        except (ValueError, TypeError):
            continue

        if word and confidence >= 20:
            words.append(word)
            confidences.append(confidence)

    text = " ".join(words)
    text = clean_text(text)

    average_confidence = (
        round(sum(confidences) / len(confidences), 2)
        if confidences else 0.0
    )

    return text, average_confidence


def merge_detections(detections):

    cues = []

    for detection in detections:

        text = detection["text"]

        if not text:
            continue

        if not cues:
            cues.append({
                "start_seconds": detection["time"],
                "end_seconds": detection["time"],
                "text": text,
                "confidence_values": [
                    detection["confidence"]
                ],
                "detection_count": 1
            })
            continue

        previous = cues[-1]

        gap = (
            detection["time"]
            - previous["end_seconds"]
        )

        similarity_score = similarity(
            previous["text"],
            text
        )

        if (
            gap <= MAX_GAP_SECONDS
            and similarity_score >= 0.82
        ):

            previous["end_seconds"] = detection["time"]

            previous["confidence_values"].append(
                detection["confidence"]
            )

            previous["detection_count"] += 1

            # Prefer the longer OCR result when texts differ slightly.
            if len(text) > len(previous["text"]):
                previous["text"] = text

        else:

            cues.append({
                "start_seconds": detection["time"],
                "end_seconds": detection["time"],
                "text": text,
                "confidence_values": [
                    detection["confidence"]
                ],
                "detection_count": 1
            })

    for index, cue in enumerate(cues):

        # Extend the end to the next sampling interval.
        cue["end_seconds"] = round(
            min(
                cue["end_seconds"] + SAMPLE_INTERVAL_SECONDS,
                TEST_DURATION_SECONDS
            ),
            3
        )

        cue["start_seconds"] = round(
            cue["start_seconds"],
            3
        )

        cue["average_ocr_confidence"] = round(
            sum(cue["confidence_values"])
            / len(cue["confidence_values"]),
            2
        )

        cue.pop("confidence_values")

        cue["cue_id"] = f"OCR_{index + 1:04d}"

    return cues


def format_srt_time(seconds):

    milliseconds = round(seconds * 1000)

    hours = milliseconds // 3600000
    milliseconds %= 3600000

    minutes = milliseconds // 60000
    milliseconds %= 60000

    secs = milliseconds // 1000
    milliseconds %= 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


def save_srt(cues):

    lines = []

    for index, cue in enumerate(cues, start=1):

        start = format_srt_time(
            cue["start_seconds"]
        )

        end = format_srt_time(
            cue["end_seconds"]
        )

        lines.extend([
            str(index),
            f"{start} --> {end}",
            cue["text"],
            ""
        ])

    SRT_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8-sig"
    )


def main():

    if not VIDEO_FILE.exists():
        raise FileNotFoundError(VIDEO_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("SUBTITLE OCR ENGINE V1")
    print("----------------------")
    print(f"Video: {VIDEO_FILE}")
    print(f"Test duration: {TEST_DURATION_SECONDS}s")
    print(f"Sample interval: {SAMPLE_INTERVAL_SECONDS}s")

    print(
        "Tesseract:",
        pytesseract.get_tesseract_version()
    )

    video = cv2.VideoCapture(
        str(VIDEO_FILE)
    )

    if not video.isOpened():
        raise RuntimeError(
            "Could not open video file."
        )

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError(
            "Invalid video frame rate."
        )

    detections = []

    sample_count = int(
        TEST_DURATION_SECONDS
        / SAMPLE_INTERVAL_SECONDS
    )

    print("\nReading video frames...")

    for index in range(sample_count):

        timestamp = (
            index * SAMPLE_INTERVAL_SECONDS
        )

        video.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        success, frame = video.read()

        if not success:
            break

        text, confidence = read_subtitle_from_frame(
            frame
        )

        if text:

            detections.append({
                "time": round(timestamp, 3),
                "text": text,
                "confidence": confidence
            })

        if (index + 1) % 20 == 0:
            print(
                f"Processed {index + 1}/{sample_count}"
            )

    video.release()

    cues = merge_detections(detections)

    result = {
        "schema_version": "1.0.0",
        "engine": "subtitle_ocr_v1",
        "source_video": VIDEO_FILE.name,
        "language": "eng",
        "test_duration_seconds": TEST_DURATION_SECONDS,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "total_frames_sampled": index + 1 if sample_count else 0,
        "total_detections": len(detections),
        "total_subtitle_cues": len(cues),
        "crop_region": {
            "x_start": CROP_X_START,
            "x_end": CROP_X_END,
            "y_start": CROP_Y_START,
            "y_end": CROP_Y_END
        },
        "detections": detections,
        "subtitles": cues
    }

    JSON_OUTPUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    save_srt(cues)

    print("\nSUBTITLE OCR: OK")
    print("-----------------")
    print("Frames sampled:", result["total_frames_sampled"])
    print("Detections:", len(detections))
    print("Subtitle cues:", len(cues))
    print("JSON:", JSON_OUTPUT)
    print("SRT:", SRT_OUTPUT)

    print("\nRECOGNIZED SUBTITLES")
    print("--------------------")

    for cue in cues:
        print(
            f"{cue['start_seconds']:7.2f} - "
            f"{cue['end_seconds']:7.2f} | "
            f"{cue['text']}"
        )


if __name__ == "__main__":
    main()
