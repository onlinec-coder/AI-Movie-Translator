
import cv2
import json
import re
import difflib
import statistics
import string
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_FILE = PROJECT_ROOT / "data" / "input" / "Husbands.mp4"
REFERENCE_SRT = PROJECT_ROOT / "data" / "input" / "input.srt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "subtitle_ocr_inspection"

JSON_OUTPUT = OUTPUT_DIR / "subtitle_ocr_v3.json"
SRT_OUTPUT = OUTPUT_DIR / "subtitle_ocr_v3.srt"

TEST_DURATION = 30.0
SAMPLE_INTERVAL = 0.25

# OCR search region in original frame coordinates
CROP_TOP = 0.66
CROP_BOTTOM = 0.95

MIN_CONFIDENCE = 40.0
MIN_WORDS = 2
MIN_BOX_WIDTH = 35
MIN_BOX_HEIGHT = 8
MAX_BOX_HEIGHT = 55

# Temporal grouping
MAX_FRAME_GAP = 1.0
MIN_SIMILARITY = 0.42


# --------------------------------------------------
# TEXT UTILITIES
# --------------------------------------------------

def normalize_text(text):
    text = text.lower()
    text = text.replace("|", "l")
    text = re.sub(r"[^a-z0-9']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return normalize_text(text).split()


def similarity(a, b):
    return difflib.SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


def alphanumeric_ratio(text):
    if not text:
        return 0.0

    useful = sum(
        char.isalnum() for char in text
    )

    return useful / len(text)


# --------------------------------------------------
# SRT REFERENCE
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

        match = re.match(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})",
            lines[1]
        )

        if not match:
            continue

        v = list(map(int, match.groups()))

        start = (
            v[0] * 3600
            + v[1] * 60
            + v[2]
            + v[3] / 1000
        )

        end = (
            v[4] * 3600
            + v[5] * 60
            + v[6]
            + v[7] / 1000
        )

        subtitles.append({
            "start": start,
            "end": end,
            "text": " ".join(lines[2:]).strip()
        })

    return subtitles


# --------------------------------------------------
# WORD-LEVEL REFERENCE COMPARISON
# --------------------------------------------------

def match_reference(text, timestamp, references):
    ocr_words = tokenize(text)

    if not ocr_words:
        return {
            "score": 0.0,
            "matched_text": None
        }

    best_score = 0.0
    best_text = None

    for subtitle in references:
        if timestamp < subtitle["start"] - 2.0:
            continue

        if timestamp > subtitle["end"] + 2.0:
            continue

        reference_words = tokenize(subtitle["text"])

        if not reference_words:
            continue

        matches = 0

        for word in ocr_words:
            if word in reference_words:
                matches += 1
                continue

            # Allow small OCR spelling differences.
            if any(
                difflib.SequenceMatcher(
                    None, word, ref_word
                ).ratio() >= 0.82
                for ref_word in reference_words
            ):
                matches += 1

        score = matches / len(ocr_words)

        if score > best_score:
            best_score = score
            best_text = subtitle["text"]

    return {
        "score": round(best_score, 3),
        "matched_text": best_text
    }


# --------------------------------------------------
# OCR WORD EXTRACTION
# --------------------------------------------------

def extract_frame_lines(frame, timestamp, pytesseract):
    height, width = frame.shape[:2]

    y_start = int(height * CROP_TOP)
    y_end = int(height * CROP_BOTTOM)

    crop = frame[y_start:y_end, :]

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    enlarged = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    # Preserve subtitle outlines while improving contrast.
    processed = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(enlarged)

    data = pytesseract.image_to_data(
        processed,
        lang="eng",
        config="--oem 3 --psm 11",
        output_type=pytesseract.Output.DICT
    )

    grouped = defaultdict(list)

    for i, raw_text in enumerate(data["text"]):
        word = raw_text.strip()

        if not word:
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
            "text": word,
            "confidence": confidence,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i]
        })

    lines = []

    for words in grouped.values():
        words.sort(key=lambda item: item["left"])

        text = " ".join(
            item["text"] for item in words
        ).strip()

        tokens = tokenize(text)

        if len(tokens) < MIN_WORDS:
            continue

        if alphanumeric_ratio(text) < 0.55:
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

        box_width = (right - left) / 3
        box_height = (bottom - top) / 3

        if box_width < MIN_BOX_WIDTH:
            continue

        if box_height < MIN_BOX_HEIGHT:
            continue

        if box_height > MAX_BOX_HEIGHT:
            continue

        # Convert OCR coordinates back to original frame.
        original_y = y_start + top / 3
        original_x = (left + right) / 6

        y_center = original_y / height
        x_center = original_x / width

        # Reject text too close to the extreme bottom.
        if not 0.67 <= y_center <= 0.94:
            continue

        # Reject unusually wide, tall graphic regions.
        if box_width > width * 0.90:
            continue

        confidence = statistics.mean(
            item["confidence"] for item in words
        )

        lines.append({
            "timestamp": round(timestamp, 3),
            "text": text,
            "confidence": round(confidence, 2),
            "word_count": len(tokens),
            "x_center": round(x_center, 4),
            "y_center": round(y_center, 4),
            "box_width": round(box_width, 1),
            "box_height": round(box_height, 1)
        })

    return lines


# --------------------------------------------------
# COMBINE LINES WITHIN ONE FRAME
# --------------------------------------------------

def combine_frame_lines(lines):
    if not lines:
        return None

    lines.sort(key=lambda x: x["y_center"])

    groups = []

    for line in lines:
        placed = False

        for group in groups:
            average_y = statistics.mean(
                item["y_center"] for item in group
            )

            # Lines close vertically are treated as one subtitle.
            if abs(line["y_center"] - average_y) <= 0.065:
                group.append(line)
                placed = True
                break

        if not placed:
            groups.append([line])

    candidates = []

    for group in groups:
        group.sort(key=lambda x: x["x_center"])

        text = " ".join(
            item["text"] for item in group
        )

        candidates.append({
            "timestamp": lines[0]["timestamp"],
            "text": text,
            "confidence": round(
                statistics.mean(
                    item["confidence"] for item in group
                ),
                2
            ),
            "word_count": sum(
                item["word_count"] for item in group
            ),
            "x_center": round(
                statistics.mean(
                    item["x_center"] for item in group
                ),
                4
            ),
            "y_center": round(
                statistics.mean(
                    item["y_center"] for item in group
                ),
                4
            ),
            "line_count": len(group)
        })

    # Keep separate text groups for inspection.
    return candidates


# --------------------------------------------------
# TEMPORAL GROUPING
# --------------------------------------------------

def group_temporal_detections(detections):
    detections.sort(
        key=lambda x: x["timestamp"]
    )

    groups = []

    for detection in detections:
        best_group = None
        best_score = 0.0

        for group in reversed(groups):
            gap = (
                detection["timestamp"]
                - group["last_timestamp"]
            )

            if gap > MAX_FRAME_GAP:
                break

            score = similarity(
                detection["text"],
                group["representative_text"]
            )

            # Similar position is supporting evidence.
            position_difference = abs(
                detection["y_center"]
                - group["representative_y"]
            )

            if position_difference > 0.10:
                continue

            if score >= MIN_SIMILARITY and score > best_score:
                best_score = score
                best_group = group

        if best_group is None:
            groups.append({
                "first_timestamp": detection["timestamp"],
                "last_timestamp": detection["timestamp"],
                "representative_text": detection["text"],
                "representative_y": detection["y_center"],
                "detections": [detection]
            })
        else:
            best_group["last_timestamp"] = detection["timestamp"]
            best_group["detections"].append(detection)

            # Update representative to the best-confidence reading.
            best = max(
                best_group["detections"],
                key=lambda item: item["confidence"]
            )

            best_group["representative_text"] = best["text"]
            best_group["representative_y"] = best["y_center"]

    return groups


# --------------------------------------------------
# BUILD FINAL CUES
# --------------------------------------------------

def build_cues(groups, references):
    cues = []

    for index, group in enumerate(groups, start=1):
        items = group["detections"]

        best = max(
            items,
            key=lambda item: (
                item["confidence"]
                + item.get("reference", {}).get("score", 0) * 15
            )
        )

        reference = match_reference(
            best["text"],
            best["timestamp"],
            references
        )

        cues.append({
            "cue_id": f"OCR_{index:04d}",
            "start_seconds": group["first_timestamp"],
            "end_seconds": group["last_timestamp"] + SAMPLE_INTERVAL,
            "text": best["text"],
            "confidence": best["confidence"],
            "reference_match_score": reference["score"],
            "reference_text": reference["matched_text"],
            "detection_count": len(items),
            "average_y": round(
                statistics.mean(
                    item["y_center"] for item in items
                ),
                4
            ),
            "review_required": (
                len(items) < 2
                or best["confidence"] < 75
                or reference["score"] < 0.25
            )
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not VIDEO_FILE.exists():
        raise FileNotFoundError(VIDEO_FILE)

    references = load_reference_srt(
        REFERENCE_SRT
    )

    capture = cv2.VideoCapture(
        str(VIDEO_FILE)
    )

    if not capture.isOpened():
        raise RuntimeError(
            "Could not open video."
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        raise RuntimeError(
            "Invalid video FPS."
        )

    video_duration = (
        capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    )

    duration = min(
        TEST_DURATION,
        video_duration
    )

    raw_detections = []
    frame_results = []

    print("SUBTITLE OCR ENGINE V3")
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

        lines = extract_frame_lines(
            frame,
            timestamp,
            pytesseract
        )

        combined = combine_frame_lines(
            lines
        )

        if combined:
            for item in combined:
                item["reference"] = match_reference(
                    item["text"],
                    timestamp,
                    references
                )

                raw_detections.append(item)

        frame_results.append({
            "timestamp": round(timestamp, 3),
            "line_count": len(lines),
            "candidate_count": len(combined or []),
            "candidates": combined or []
        })

        frame_count += 1

        if frame_count % 20 == 0:
            print(
                f"Processed {frame_count} frames"
            )

        timestamp += SAMPLE_INTERVAL

    capture.release()

    groups = group_temporal_detections(
        raw_detections
    )

    cues = build_cues(
        groups,
        references
    )

    result = {
        "version": "3.0",
        "video": str(VIDEO_FILE),
        "test_duration": duration,
        "sample_interval": SAMPLE_INTERVAL,
        "frames_processed": frame_count,
        "reference_srt_cues": len(references),
        "detections": len(raw_detections),
        "subtitle_cues": len(cues),
        "raw_detections": raw_detections,
        "frame_results": frame_results,
        "subtitle_cues_data": cues
    }

    JSON_OUTPUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    save_srt(
        cues,
        SRT_OUTPUT
    )

    review_count = sum(
        1 for cue in cues
        if cue["review_required"]
    )

    print()
    print("SUBTITLE OCR V3: OK")
    print("-------------------")
    print(f"Frames: {frame_count}")
    print(f"Detections: {len(raw_detections)}")
    print(f"Subtitle cues: {len(cues)}")
    print(f"Review required: {review_count}")
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
            f"ref={cue['reference_match_score']}, "
            f"frames={cue['detection_count']}, "
            f"review={cue['review_required']}]"
        )


if __name__ == "__main__":
    main()
