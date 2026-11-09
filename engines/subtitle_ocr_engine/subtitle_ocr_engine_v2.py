
import cv2
import json
import re
import difflib
import statistics
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_FILE = PROJECT_ROOT / "data" / "input" / "Husbands.mp4"
REFERENCE_SRT = PROJECT_ROOT / "data" / "input" / "input.srt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "subtitle_ocr_inspection"

JSON_OUTPUT = OUTPUT_DIR / "subtitle_ocr_v2.json"
SRT_OUTPUT = OUTPUT_DIR / "subtitle_ocr_v2.srt"

TEST_DURATION = 30.0
SAMPLE_INTERVAL = 0.25

MIN_CONFIDENCE = 35.0
MIN_TEXT_LENGTH = 3
MIN_BOX_WIDTH = 35
MIN_BOX_HEIGHT = 8

# --------------------------------------------------
# TEXT NORMALIZATION
# --------------------------------------------------

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity(a, b):
    return difflib.SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


# --------------------------------------------------
# LOAD REFERENCE SUBTITLES
# --------------------------------------------------

def load_reference_srt(path):
    if not path.exists():
        return []

    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", content.strip())

    subtitles = []

    for block in blocks:
        lines = block.strip().splitlines()

        if len(lines) < 3:
            continue

        timing = lines[1]
        match = re.match(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})",
            timing
        )

        if not match:
            continue

        values = list(map(int, match.groups()))

        start = (
            values[0] * 3600
            + values[1] * 60
            + values[2]
            + values[3] / 1000
        )

        end = (
            values[4] * 3600
            + values[5] * 60
            + values[6]
            + values[7] / 1000
        )

        text = " ".join(lines[2:]).strip()

        subtitles.append({
            "start": start,
            "end": end,
            "text": text
        })

    return subtitles


# --------------------------------------------------
# OCR LINE EXTRACTION
# --------------------------------------------------

def extract_lines(frame, timestamp, pytesseract):
    height, width = frame.shape[:2]

    # Search the lower 40% of the image.
    y_start = int(height * 0.60)
    y_end = int(height * 0.96)

    crop = frame[y_start:y_end, :]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Improve contrast without changing the original frame.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    enlarged = cv2.resize(
        enhanced,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    data = pytesseract.image_to_data(
        enlarged,
        lang="eng",
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    grouped = defaultdict(list)

    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if confidence < MIN_CONFIDENCE:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )

        grouped[key].append({
            "text": text,
            "confidence": confidence,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i]
        })

    results = []

    for words in grouped.values():
        words.sort(key=lambda item: item["left"])

        text = " ".join(item["text"] for item in words).strip()

        if len(text) < MIN_TEXT_LENGTH:
            continue

        left = min(item["left"] for item in words)
        right = max(
            item["left"] + item["width"]
            for item in words
        )

        top = min(item["top"] for item in words)
        bottom = max(
            item["top"] + item["height"]
            for item in words
        )

        box_width = right - left
        box_height = bottom - top

        if box_width < MIN_BOX_WIDTH:
            continue

        if box_height < MIN_BOX_HEIGHT:
            continue

        average_confidence = statistics.mean(
            item["confidence"] for item in words
        )

        # Convert OCR crop coordinates to original image coordinates.
        original_y = y_start + (top / 2)
        normalized_y = original_y / height

        results.append({
            "timestamp": round(timestamp, 3),
            "text": text,
            "confidence": round(average_confidence, 2),
            "word_count": len(words),
            "x_center": round(((left + right) / 2) / 2 / width, 4),
            "y_center": round(normalized_y, 4),
            "box_width": round(box_width / 2, 1),
            "box_height": round(box_height / 2, 1)
        })

    return results


# --------------------------------------------------
# REFERENCE MATCHING
# --------------------------------------------------

def match_reference(text, timestamp, references):
    best_score = 0.0
    best_match = None

    for subtitle in references:
        # Reference subtitles are only used as supporting evidence.
        if timestamp < subtitle["start"] - 2.0:
            continue

        if timestamp > subtitle["end"] + 2.0:
            continue

        score = similarity(text, subtitle["text"])

        if score > best_score:
            best_score = score
            best_match = subtitle["text"]

    return {
        "score": round(best_score, 3),
        "matched_text": best_match
    }


# --------------------------------------------------
# TEMPORAL GROUPING
# --------------------------------------------------

def group_detections(detections):
    detections = sorted(
        detections,
        key=lambda item: item["timestamp"]
    )

    groups = []

    for detection in detections:
        best_group = None
        best_score = 0.0

        for group in reversed(groups):
            gap = detection["timestamp"] - group["last_timestamp"]

            if gap > 1.0:
                break

            score = similarity(
                detection["text"],
                group["representative_text"]
            )

            if score > best_score and score >= 0.72:
                best_score = score
                best_group = group

        if best_group is None:
            groups.append({
                "first_timestamp": detection["timestamp"],
                "last_timestamp": detection["timestamp"],
                "representative_text": detection["text"],
                "detections": [detection]
            })
        else:
            best_group["last_timestamp"] = detection["timestamp"]
            best_group["detections"].append(detection)

    cues = []

    for index, group in enumerate(groups, start=1):
        items = group["detections"]

        # Prefer the OCR text with the highest confidence,
        # while also considering reference SRT similarity.
        best = max(
            items,
            key=lambda item: (
                item["confidence"] + item["reference"]["score"] * 25
            )
        )

        cues.append({
            "cue_id": f"OCR_{index:04d}",
            "start_seconds": group["first_timestamp"],
            "end_seconds": group["last_timestamp"] + SAMPLE_INTERVAL,
            "text": best["text"],
            "confidence": best["confidence"],
            "reference_match_score": best["reference"]["score"],
            "reference_text": best["reference"]["matched_text"],
            "detections": len(items)
        })

    return cues


# --------------------------------------------------
# SRT EXPORT
# --------------------------------------------------

def format_timestamp(seconds):
    milliseconds = int(round(seconds * 1000))

    hours = milliseconds // 3600000
    milliseconds %= 3600000

    minutes = milliseconds // 60000
    milliseconds %= 60000

    secs = milliseconds // 1000
    milliseconds %= 1000

    return (
        f"{hours:02}:{minutes:02}:{secs:02},"
        f"{milliseconds:03}"
    )


def save_srt(cues, path):
    lines = []

    for index, cue in enumerate(cues, start=1):
        lines.extend([
            str(index),
            (
                f"{format_timestamp(cue['start_seconds'])} --> "
                f"{format_timestamp(cue['end_seconds'])}"
            ),
            cue["text"],
            ""
        ])

    path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    import pytesseract

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_FILE.exists():
        raise FileNotFoundError(VIDEO_FILE)

    references = load_reference_srt(REFERENCE_SRT)

    capture = cv2.VideoCapture(str(VIDEO_FILE))

    if not capture.isOpened():
        raise RuntimeError("Could not open video.")

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError("Invalid video FPS.")

    duration = min(
        TEST_DURATION,
        capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    )

    detections = []

    print("SUBTITLE OCR ENGINE V2")
    print("----------------------")
    print(f"FPS: {fps}")
    print(f"Test duration: {duration:.2f}s")
    print(f"Reference SRT cues: {len(references)}")
    print()

    timestamp = 0.0
    frame_count = 0

    while timestamp < duration:
        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        success, frame = capture.read()

        if not success:
            timestamp += SAMPLE_INTERVAL
            continue

        frame_detections = extract_lines(
            frame,
            timestamp,
            pytesseract
        )

        for item in frame_detections:
            item["reference"] = match_reference(
                item["text"],
                timestamp,
                references
            )

        detections.extend(frame_detections)

        frame_count += 1

        if frame_count % 20 == 0:
            print(f"Processed {frame_count} frames")

        timestamp += SAMPLE_INTERVAL

    capture.release()

    cues = group_detections(detections)

    result = {
        "version": "2.0",
        "video": str(VIDEO_FILE),
        "test_duration": duration,
        "sample_interval": SAMPLE_INTERVAL,
        "frames_processed": frame_count,
        "reference_srt_cues": len(references),
        "detections": len(detections),
        "subtitle_cues": len(cues),
        "raw_detections": detections,
        "subtitle_cues_data": cues
    }

    JSON_OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    save_srt(cues, SRT_OUTPUT)

    print()
    print("SUBTITLE OCR V2: OK")
    print("-------------------")
    print(f"Frames: {frame_count}")
    print(f"Detections: {len(detections)}")
    print(f"Subtitle cues: {len(cues)}")
    print(f"JSON: {JSON_OUTPUT}")
    print(f"SRT: {SRT_OUTPUT}")
    print()
    print("RECOGNIZED CUES")
    print("----------------")

    for cue in cues:
        print(
            f"{cue['start_seconds']:7.2f} - "
            f"{cue['end_seconds']:7.2f} | "
            f"{cue['text']} "
            f"[conf={cue['confidence']}, "
            f"ref={cue['reference_match_score']}]"
        )


if __name__ == "__main__":
    main()
