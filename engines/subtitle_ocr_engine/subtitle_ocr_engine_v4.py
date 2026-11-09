
import cv2
import json
import re
import statistics
import difflib
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_FILE = PROJECT_ROOT / "data" / "input" / "Husbands.mp4"
REFERENCE_SRT = PROJECT_ROOT / "data" / "input" / "input.srt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "subtitle_ocr_inspection"
JSON_OUTPUT = OUTPUT_DIR / "subtitle_ocr_v4.json"
REVIEW_OUTPUT = OUTPUT_DIR / "subtitle_ocr_v4_review.json"

TEST_DURATION = 30.0
SAMPLE_INTERVAL = 0.25

CROP_TOP = 0.66
CROP_BOTTOM = 0.95

MIN_CONFIDENCE = 35.0
MIN_WORD_CONFIDENCE = 25.0
MIN_WORDS = 1

FRAME_PADDING = 0.35
MAX_FRAME_SAMPLES_PER_CUE = 40


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


def word_similarity(a, b):
    if a == b:
        return 1.0

    return difflib.SequenceMatcher(
        None, a, b
    ).ratio()


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
    if not path.exists():
        raise FileNotFoundError(path)

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

        text = " ".join(lines[2:]).strip()

        subtitles.append({
            "index": len(subtitles) + 1,
            "start": start,
            "end": end,
            "text": text
        })

    return subtitles


def extract_ocr_words(frame, pytesseract):
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

    for i, raw_text in enumerate(data["text"]):
        word = raw_text.strip()

        if not word:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if confidence < MIN_WORD_CONFIDENCE:
            continue

        cleaned = normalize_text(word)

        if not cleaned:
            continue

        left = data["left"][i] / 3
        top = data["top"][i] / 3 + y_start
        box_width = data["width"][i] / 3
        box_height = data["height"][i] / 3

        x_center = (left + box_width / 2) / width
        y_center = (top + box_height / 2) / height

        if not 0.66 <= y_center <= 0.95:
            continue

        if box_height < 3 or box_height > 60:
            continue

        words.append({
            "text": word,
            "normalized": cleaned,
            "confidence": round(confidence, 2),
            "x_center": round(x_center, 4),
            "y_center": round(y_center, 4)
        })

    return words


def group_words_into_lines(words):
    if not words:
        return []

    # Group by approximate vertical position.
    sorted_words = sorted(
        words,
        key=lambda item: (
            item["y_center"],
            item["x_center"]
        )
    )

    lines = []

    for word in sorted_words:
        best_line = None
        best_difference = 999

        for line in lines:
            avg_y = statistics.mean(
                item["y_center"] for item in line
            )

            difference = abs(
                word["y_center"] - avg_y
            )

            if difference < 0.025 and difference < best_difference:
                best_line = line
                best_difference = difference

        if best_line is None:
            lines.append([word])
        else:
            best_line.append(word)

    output = []

    for line in lines:
        line.sort(key=lambda item: item["x_center"])

        output.append({
            "text": " ".join(
                item["text"] for item in line
            ),
            "normalized": " ".join(
                item["normalized"] for item in line
            ),
            "confidence": round(
                statistics.mean(
                    item["confidence"] for item in line
                ),
                2
            ),
            "word_count": len(line),
            "y_center": round(
                statistics.mean(
                    item["y_center"] for item in line
                ),
                4
            )
        })

    return output


def frame_text(frame, pytesseract):
    words = extract_ocr_words(
        frame,
        pytesseract
    )

    lines = group_words_into_lines(words)

    # Join subtitle lines from top to bottom.
    lines.sort(key=lambda item: item["y_center"])

    return {
        "text": " ".join(
            line["text"] for line in lines
        ),
        "normalized": " ".join(
            line["normalized"] for line in lines
        ),
        "confidence": (
            round(
                statistics.mean(
                    line["confidence"] for line in lines
                ),
                2
            )
            if lines else 0.0
        ),
        "word_count": sum(
            line["word_count"] for line in lines
        ),
        "lines": lines
    }


def align_words(reference_text, ocr_text):
    reference_words = tokenize(reference_text)
    ocr_words = tokenize(ocr_text)

    if not reference_words or not ocr_words:
        return {
            "matched_words": 0,
            "reference_word_count": len(reference_words),
            "ocr_word_count": len(ocr_words),
            "word_coverage": 0.0,
            "matched_pairs": []
        }

    # Sequence alignment: preserve word order.
    n = len(reference_words)
    m = len(ocr_words)

    dp = [
        [0.0] * (m + 1)
        for _ in range(n + 1)
    ]

    back = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        dp[i][0] = 0
        back[i][0] = "up"

    for j in range(1, m + 1):
        dp[0][j] = 0
        back[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = word_similarity(
                reference_words[i - 1],
                ocr_words[j - 1]
            )

            match_score = (
                dp[i - 1][j - 1] + score
                if score >= 0.70 else -1
            )

            up_score = dp[i - 1][j]
            left_score = dp[i][j - 1]

            best = max(
                match_score,
                up_score,
                left_score
            )

            dp[i][j] = best

            if best == match_score:
                back[i][j] = "diag"
            elif best == up_score:
                back[i][j] = "up"
            else:
                back[i][j] = "left"

    i = n
    j = m
    pairs = []

    while i > 0 and j > 0:
        direction = back[i][j]

        if direction == "diag":
            score = word_similarity(
                reference_words[i - 1],
                ocr_words[j - 1]
            )

            pairs.append({
                "reference_word": reference_words[i - 1],
                "ocr_word": ocr_words[j - 1],
                "similarity": round(score, 3)
            })

            i -= 1
            j -= 1

        elif direction == "up":
            i -= 1

        else:
            j -= 1

    pairs.reverse()

    matched = len(pairs)

    coverage = (
        matched / len(reference_words)
        if reference_words else 0
    )

    return {
        "matched_words": matched,
        "reference_word_count": len(reference_words),
        "ocr_word_count": len(ocr_words),
        "word_coverage": round(coverage, 3),
        "matched_pairs": pairs
    }


def choose_best_frame(frame_results, reference_text):
    candidates = []

    for item in frame_results:
        alignment = align_words(
            reference_text,
            item["text"]
        )

        score = (
            alignment["word_coverage"] * 0.75
            + min(item["confidence"] / 100, 1.0) * 0.20
            + min(item["word_count"] / max(
                alignment["reference_word_count"], 1
            ), 1.0) * 0.05
        )

        candidates.append({
            "frame": item,
            "alignment": alignment,
            "score": round(score, 4)
        })

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item["score"]
    )


def classify_result(best, frame_results):
    if best is None:
        return "NOT_DETECTED"

    coverage = best["alignment"]["word_coverage"]
    confidence = best["frame"]["confidence"]

    # Evidence from multiple frames is useful, but not mandatory.
    usable_frames = sum(
        1 for item in frame_results
        if item["word_count"] > 0
        and item["confidence"] >= MIN_CONFIDENCE
    )

    if coverage >= 0.80 and confidence >= 65:
        return "CONFIRMED"

    if coverage >= 0.35:
        return "PARTIAL"

    if usable_frames == 0:
        return "NOT_DETECTED"

    return "POTENTIAL_MISMATCH"


def analyze_subtitle(video, subtitle, fps, duration, pytesseract):
    start = max(
        0.0,
        subtitle["start"] - FRAME_PADDING
    )

    end = min(
        duration,
        subtitle["end"] + FRAME_PADDING
    )

    frame_results = []

    timestamps = []
    timestamp = start

    while timestamp <= end:
        timestamps.append(timestamp)
        timestamp += SAMPLE_INTERVAL

    if len(timestamps) > MAX_FRAME_SAMPLES_PER_CUE:
        step = max(
            1,
            len(timestamps) // MAX_FRAME_SAMPLES_PER_CUE
        )

        timestamps = timestamps[::step]

    for timestamp in timestamps:
        video.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        success, frame = video.read()

        if not success:
            continue

        result = frame_text(
            frame,
            pytesseract
        )

        if result["word_count"] == 0:
            continue

        alignment = align_words(
            subtitle["text"],
            result["text"]
        )

        frame_results.append({
            "timestamp": round(timestamp, 3),
            "text": result["text"],
            "confidence": result["confidence"],
            "word_count": result["word_count"],
            "alignment": alignment
        })

    best = choose_best_frame(
        frame_results,
        subtitle["text"]
    )

    status = classify_result(
        best,
        frame_results
    )

    if best is None:
        best_text = ""
        best_confidence = 0
        coverage = 0
        matched_pairs = []
    else:
        best_text = best["frame"]["text"]
        best_confidence = best["frame"]["confidence"]
        coverage = best["alignment"]["word_coverage"]
        matched_pairs = best["alignment"]["matched_pairs"]

    return {
        "subtitle_id": subtitle["index"],
        "start_seconds": subtitle["start"],
        "end_seconds": subtitle["end"],
        "reference_text": subtitle["text"],
        "ocr_best_text": best_text,
        "ocr_confidence": best_confidence,
        "word_coverage": coverage,
        "matched_pairs": matched_pairs,
        "frames_checked": len(timestamps),
        "frames_with_text": len(frame_results),
        "status": status,
        "review_required": status != "CONFIRMED",
        "frame_evidence": frame_results
    }


def main():
    import pytesseract

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    subtitles = load_srt(
        REFERENCE_SRT
    )

    video = cv2.VideoCapture(
        str(VIDEO_FILE)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"Cannot open video: {VIDEO_FILE}"
        )

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError("Invalid FPS")

    duration = (
        video.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    )

    selected = [
        subtitle for subtitle in subtitles
        if subtitle["start"] < TEST_DURATION
    ]

    print("SUBTITLE OCR ENGINE V4")
    print("----------------------")
    print(f"Video FPS: {fps}")
    print(f"Video duration: {duration:.2f}s")
    print(f"Reference SRT cues: {len(subtitles)}")
    print(f"Cues in test window: {len(selected)}")
    print()

    results = []

    for index, subtitle in enumerate(selected, start=1):
        result = analyze_subtitle(
            video,
            subtitle,
            fps,
            duration,
            pytesseract
        )

        results.append(result)

        if index % 10 == 0 or index == len(selected):
            print(
                f"Processed {index}/{len(selected)} cues"
            )

    video.release()

    summary = {
        "total": len(results),
        "confirmed": sum(
            1 for x in results
            if x["status"] == "CONFIRMED"
        ),
        "partial": sum(
            1 for x in results
            if x["status"] == "PARTIAL"
        ),
        "not_detected": sum(
            1 for x in results
            if x["status"] == "NOT_DETECTED"
        ),
        "potential_mismatch": sum(
            1 for x in results
            if x["status"] == "POTENTIAL_MISMATCH"
        ),
        "review_required": sum(
            1 for x in results
            if x["review_required"]
        )
    }

    output = {
        "version": "4.0",
        "video": str(VIDEO_FILE),
        "test_duration": TEST_DURATION,
        "sample_interval": SAMPLE_INTERVAL,
        "reference_srt": str(REFERENCE_SRT),
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

    review_items = [
        item for item in results
        if item["review_required"]
    ]

    REVIEW_OUTPUT.write_text(
        json.dumps(
            review_items,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("SUBTITLE OCR V4: OK")
    print("-------------------")
    print(f"Total cues: {summary['total']}")
    print(f"Confirmed: {summary['confirmed']}")
    print(f"Partial: {summary['partial']}")
    print(f"Not detected: {summary['not_detected']}")
    print(f"Potential mismatch: {summary['potential_mismatch']}")
    print(f"Review required: {summary['review_required']}")
    print(f"JSON: {JSON_OUTPUT}")
    print(f"Review JSON: {REVIEW_OUTPUT}")

    print()
    print("CUE RESULTS")
    print("-----------")

    for item in results:
        print(
            f"{item['start_seconds']:7.2f} - "
            f"{item['end_seconds']:7.2f} | "
            f"{item['status']:20} | "
            f"{item['word_coverage']:.2f} | "
            f"{item['reference_text'][:75]}"
        )


if __name__ == "__main__":
    main()
