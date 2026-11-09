
import cv2
import json
import re
import statistics
import difflib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_FILE = PROJECT_ROOT / "data" / "input" / "Husbands.mp4"
REFERENCE_SRT = PROJECT_ROOT / "data" / "input" / "input.srt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "subtitle_ocr_inspection"
JSON_OUTPUT = OUTPUT_DIR / "subtitle_ocr_v5.json"
REVIEW_OUTPUT = OUTPUT_DIR / "subtitle_ocr_v5_review.json"

TEST_DURATION = 30.0
SAMPLE_INTERVAL = 0.25
FRAME_PADDING = 0.35

CROP_TOP = 0.66
CROP_BOTTOM = 0.95

MIN_WORD_CONFIDENCE = 25.0
MIN_FRAME_CONFIDENCE = 35.0
MAX_FRAME_SAMPLES_PER_CUE = 40


def normalize_text(text):
    text = text.lower()
    text = text.replace("|", "l")
    text = re.sub(r"[^a-z0-9']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return normalize_text(text).split()


def parse_timestamp(value):
    h, m, rest = value.split(":")
    s, ms = rest.split(",")

    return (
        int(h) * 3600
        + int(m) * 60
        + int(s)
        + int(ms) / 1000
    )


def load_srt(path):
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", content.strip())

    subtitles = []

    for block in blocks:
        lines = block.strip().splitlines()

        if len(lines) < 3:
            continue

        match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
            r"(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1]
        )

        if not match:
            continue

        start = parse_timestamp(match.group(1))
        end = parse_timestamp(match.group(2))

        subtitles.append({
            "index": len(subtitles) + 1,
            "start": start,
            "end": end,
            "text": " ".join(lines[2:]).strip()
        })

    return subtitles


def extract_ocr(frame, pytesseract):
    height, width = frame.shape[:2]

    y1 = int(height * CROP_TOP)
    y2 = int(height * CROP_BOTTOM)

    crop = frame[y1:y2, :]

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

    words = []

    for i, raw in enumerate(data["text"]):
        raw = raw.strip()

        if not raw:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if confidence < MIN_WORD_CONFIDENCE:
            continue

        word = normalize_text(raw)

        if not word:
            continue

        top = data["top"][i] / 3 + y1
        box_height = data["height"][i] / 3

        y_center = (
            top + box_height / 2
        ) / height

        if not CROP_TOP <= y_center <= CROP_BOTTOM:
            continue

        if box_height < 3 or box_height > 60:
            continue

        words.append({
            "text": word,
            "confidence": round(confidence, 2),
            "x": data["left"][i]
        })

    words.sort(key=lambda x: x["x"])

    return words


def match_frame_words(reference_words, ocr_words):
    """
    Match OCR words to reference words.
    Each reference word can be matched only once per frame.
    """

    matched = {}

    for ocr_item in ocr_words:
        ocr_word = ocr_item["text"]

        best_index = None
        best_score = 0.0

        for index, reference_word in enumerate(reference_words):
            if index in matched:
                continue

            score = difflib.SequenceMatcher(
                None,
                reference_word,
                ocr_word
            ).ratio()

            threshold = (
                0.85
                if min(len(reference_word), len(ocr_word)) <= 3
                else 0.72
            )

            if score >= threshold and score > best_score:
                best_score = score
                best_index = index

        if best_index is not None:
            matched[best_index] = {
                "reference_word": reference_words[best_index],
                "ocr_word": ocr_word,
                "confidence": ocr_item["confidence"],
                "similarity": round(best_score, 3)
            }

    return matched


def build_sample_times(start, end, duration):
    start = max(0.0, start - FRAME_PADDING)
    end = min(duration, end + FRAME_PADDING)

    times = []
    timestamp = start

    while timestamp <= end:
        times.append(round(timestamp, 3))
        timestamp += SAMPLE_INTERVAL

    if len(times) > MAX_FRAME_SAMPLES_PER_CUE:
        step = max(
            1,
            len(times) // MAX_FRAME_SAMPLES_PER_CUE
        )
        times = times[::step]

    return times


def analyze_subtitle(video, subtitle, duration, pytesseract):
    reference_words = tokenize(subtitle["text"])

    timestamps = build_sample_times(
        subtitle["start"],
        subtitle["end"],
        duration
    )

    # Keep the strongest evidence for each reference word.
    word_evidence = {}

    frame_results = []
    frames_with_text = 0
    frames_with_match = 0

    for timestamp in timestamps:
        video.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        success, frame = video.read()

        if not success:
            continue

        ocr_words = extract_ocr(
            frame,
            pytesseract
        )

        if not ocr_words:
            continue

        frames_with_text += 1

        matched = match_frame_words(
            reference_words,
            ocr_words
        )

        if matched:
            frames_with_match += 1

        for index, evidence in matched.items():
            old = word_evidence.get(index)

            if (
                old is None
                or evidence["confidence"] > old["confidence"]
            ):
                word_evidence[index] = evidence

        ocr_text = " ".join(
            item["text"] for item in ocr_words
        )

        frame_results.append({
            "timestamp": timestamp,
            "ocr_text": ocr_text,
            "ocr_word_count": len(ocr_words),
            "matched_word_count": len(matched),
            "confidence": round(
                statistics.mean(
                    item["confidence"] for item in ocr_words
                ),
                2
            )
        })

    matched_indices = sorted(word_evidence.keys())

    missing_words = [
        word
        for index, word in enumerate(reference_words)
        if index not in word_evidence
    ]

    matched_words = [
        reference_words[index]
        for index in matched_indices
    ]

    coverage = (
        len(matched_indices) / len(reference_words)
        if reference_words else 0.0
    )

    if not reference_words:
        status = "INVALID_REFERENCE"

    elif coverage >= 0.70 and frames_with_match >= 2:
        status = "CONFIRMED"

    elif coverage >= 0.25:
        status = "PARTIAL"

    elif frames_with_text >= 2 and frames_with_match == 0:
        status = "POTENTIAL_MISMATCH"

    elif frames_with_match > 0:
        status = "PARTIAL"

    else:
        status = "NOT_DETECTED"

    return {
        "subtitle_id": subtitle["index"],
        "start_seconds": subtitle["start"],
        "end_seconds": subtitle["end"],
        "reference_text": subtitle["text"],
        "reference_word_count": len(reference_words),
        "matched_word_count": len(matched_words),
        "matched_words": matched_words,
        "missing_words": missing_words,
        "word_coverage": round(coverage, 3),
        "frames_checked": len(timestamps),
        "frames_with_text": frames_with_text,
        "frames_with_match": frames_with_match,
        "status": status,
        "review_required": status != "CONFIRMED",
        "frame_evidence": frame_results
    }


def main():
    import pytesseract

    if not VIDEO_FILE.exists():
        raise FileNotFoundError(VIDEO_FILE)

    if not REFERENCE_SRT.exists():
        raise FileNotFoundError(REFERENCE_SRT)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    subtitles = load_srt(REFERENCE_SRT)

    video = cv2.VideoCapture(
        str(VIDEO_FILE)
    )

    if not video.isOpened():
        raise RuntimeError("Cannot open video")

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError("Invalid video FPS")

    duration = (
        video.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    )

    selected = [
        item for item in subtitles
        if item["start"] < TEST_DURATION
        and item["end"] > 0
    ]

    print("SUBTITLE OCR ENGINE V5")
    print("----------------------")
    print(f"FPS: {fps}")
    print(f"Video duration: {duration:.2f}s")
    print(f"Reference SRT cues: {len(subtitles)}")
    print(f"Cues in test window: {len(selected)}")
    print()

    results = []

    for i, subtitle in enumerate(selected, start=1):
        result = analyze_subtitle(
            video,
            subtitle,
            duration,
            pytesseract
        )

        results.append(result)

        print(
            f"{i}/{len(selected)} "
            f"{result['status']:20} "
            f"coverage={result['word_coverage']:.2f} "
            f"| {result['reference_text'][:65]}"
        )

    video.release()

    summary = {
        "total": len(results),
        "confirmed": sum(
            x["status"] == "CONFIRMED" for x in results
        ),
        "partial": sum(
            x["status"] == "PARTIAL" for x in results
        ),
        "not_detected": sum(
            x["status"] == "NOT_DETECTED" for x in results
        ),
        "potential_mismatch": sum(
            x["status"] == "POTENTIAL_MISMATCH"
            for x in results
        ),
        "review_required": sum(
            x["review_required"] for x in results
        )
    }

    output = {
        "version": "5.0",
        "video": str(VIDEO_FILE),
        "reference_srt": str(REFERENCE_SRT),
        "test_duration": TEST_DURATION,
        "sample_interval": SAMPLE_INTERVAL,
        "summary": summary,
        "results": results
    }

    JSON_OUTPUT.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    review = [
        item for item in results
        if item["review_required"]
    ]

    REVIEW_OUTPUT.write_text(
        json.dumps(
            review,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("SUBTITLE OCR V5: OK")
    print("-------------------")
    print(f"Total: {summary['total']}")
    print(f"Confirmed: {summary['confirmed']}")
    print(f"Partial: {summary['partial']}")
    print(f"Not detected: {summary['not_detected']}")
    print(f"Potential mismatch: {summary['potential_mismatch']}")
    print(f"Review required: {summary['review_required']}")
    print(f"JSON: {JSON_OUTPUT}")
    print(f"Review JSON: {REVIEW_OUTPUT}")


if __name__ == "__main__":
    main()
