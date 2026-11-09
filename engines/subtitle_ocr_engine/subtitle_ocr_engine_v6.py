import cv2
import json
import re
import sys
import pytesseract

from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter
from pytesseract import Output


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_FILE = PROJECT_ROOT / "data" / "input" / "Husbands.mp4"
SRT_FILE = PROJECT_ROOT / "data" / "input" / "input.srt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "subtitle_ocr_inspection"

OUTPUT_JSON = OUTPUT_DIR / "subtitle_ocr_v6.json"

# Зөвхөн өмнөх туршилтаар асуудалтай байсан хадмалууд
TARGET_CUE_IDS = [3, 4, 5, 6]

SAMPLE_INTERVAL = 0.25

# OCR хийх дүрсний хэсэг
CROP_TOP = 0.60
CROP_BOTTOM = 0.96
CROP_LEFT = 0.03
CROP_RIGHT = 0.97

OCR_MODES = [6, 11]

MIN_OCR_CONFIDENCE = 30
MATCH_THRESHOLD = 0.82


def timestamp_to_seconds(value):
    hours, minutes, seconds = value.split(":")
    seconds, milliseconds = seconds.split(",")

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000
    )


def parse_srt(path):
    content = path.read_text(encoding="utf-8-sig")

    blocks = re.split(r"\n\s*\n", content.strip())

    cues = []

    for block in blocks:
        lines = block.strip().splitlines()

        if len(lines) < 3:
            continue

        try:
            cue_id = int(lines[0].strip())

            timing = re.match(
                r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
                r"(\d{2}:\d{2}:\d{2},\d{3})",
                lines[1].strip()
            )

            if not timing:
                continue

            start = timestamp_to_seconds(timing.group(1))
            end = timestamp_to_seconds(timing.group(2))

            text = " ".join(lines[2:]).strip()

            cues.append({
                "subtitle_id": cue_id,
                "start_seconds": start,
                "end_seconds": end,
                "reference_text": text
            })

        except (ValueError, IndexError):
            continue

    return cues


def normalize_word(word):
    word = word.lower()
    word = re.sub(r"[^a-z0-9']", "", word)
    return word.strip("'")


def tokenize(text):
    return [
        normalize_word(word)
        for word in text.split()
        if normalize_word(word)
    ]


def words_match(ocr_word, reference_word):
    if ocr_word == reference_word:
        return True

    similarity = SequenceMatcher(
        None,
        ocr_word,
        reference_word
    ).ratio()

    return similarity >= MATCH_THRESHOLD


def prepare_frame(frame):
    height, width = frame.shape[:2]

    x1 = int(width * CROP_LEFT)
    x2 = int(width * CROP_RIGHT)
    y1 = int(height * CROP_TOP)
    y2 = int(height * CROP_BOTTOM)

    cropped = frame[y1:y2, x1:x2]

    resized = cv2.resize(
        cropped,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


def read_ocr_words(image, psm):
    config = (
        f"--oem 3 --psm {psm} "
        "-c preserve_interword_spaces=1"
    )

    data = pytesseract.image_to_data(
        image,
        lang="eng",
        config=config,
        output_type=Output.DICT
    )

    words = []

    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if confidence < MIN_OCR_CONFIDENCE:
            continue

        normalized = normalize_word(text)

        if len(normalized) < 2:
            continue

        words.append({
            "text": text,
            "normalized": normalized,
            "confidence": round(confidence, 2)
        })

    return words


def analyze_mode(frame_evidence, reference_words):
    matched_words = set()
    unmatched_counter = Counter()
    best_confidence = {}

    for frame in frame_evidence:
        for word in frame["words"]:
            ocr_word = word["normalized"]

            matching_reference = None

            for reference_word in reference_words:
                if words_match(ocr_word, reference_word):
                    matching_reference = reference_word
                    break

            if matching_reference:
                matched_words.add(matching_reference)

                best_confidence[matching_reference] = max(
                    best_confidence.get(matching_reference, 0),
                    word["confidence"]
                )
            else:
                unmatched_counter[ocr_word] += 1

    missing_words = [
        word for word in reference_words
        if word not in matched_words
    ]

    coverage = (
        len(matched_words) / len(reference_words)
        if reference_words else 0
    )

    return {
        "coverage": round(coverage, 3),
        "matched_words": sorted(matched_words),
        "missing_words": missing_words,
        "best_confidence": best_confidence,
        "unmatched_ocr_words": [
            {
                "word": word,
                "frame_count": count
            }
            for word, count in unmatched_counter.most_common()
        ]
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not VIDEO_FILE.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_FILE}"
        )

    if not SRT_FILE.exists():
        raise FileNotFoundError(
            f"SRT not found: {SRT_FILE}"
        )

    cues = parse_srt(SRT_FILE)

    selected_cues = [
        cue for cue in cues
        if cue["subtitle_id"] in TARGET_CUE_IDS
    ]

    if len(selected_cues) != len(TARGET_CUE_IDS):
        found = [
            cue["subtitle_id"]
            for cue in selected_cues
        ]

        raise RuntimeError(
            f"Some target cues were not found. Found: {found}"
        )

    cap = cv2.VideoCapture(str(VIDEO_FILE))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        raise RuntimeError("Invalid video FPS.")

    results = []

    print("SUBTITLE OCR ENGINE V6")
    print("----------------------")
    print(f"FPS: {fps}")
    print(f"Target cues: {TARGET_CUE_IDS}")
    print(f"OCR modes: {OCR_MODES}")
    print()

    try:
        for cue in selected_cues:
            cue_id = cue["subtitle_id"]

            reference_words = tokenize(
                cue["reference_text"]
            )

            print(
                f"CUE {cue_id}: "
                f"{cue['start_seconds']:.3f}-"
                f"{cue['end_seconds']:.3f}"
            )

            frame_evidence_by_mode = {
                psm: [] for psm in OCR_MODES
            }

            timestamp = cue["start_seconds"]

            while timestamp <= cue["end_seconds"]:
                frame_number = int(round(timestamp * fps))

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_number
                )

                success, frame = cap.read()

                if not success:
                    timestamp += SAMPLE_INTERVAL
                    continue

                processed = prepare_frame(frame)

                for psm in OCR_MODES:
                    words = read_ocr_words(
                        processed,
                        psm
                    )

                    frame_evidence_by_mode[psm].append({
                        "timestamp": round(timestamp, 3),
                        "ocr_text": " ".join(
                            word["text"] for word in words
                        ),
                        "words": words
                    })

                timestamp += SAMPLE_INTERVAL

            mode_results = {}

            for psm in OCR_MODES:
                mode_results[str(psm)] = analyze_mode(
                    frame_evidence_by_mode[psm],
                    reference_words
                )

            results.append({
                "subtitle_id": cue_id,
                "start_seconds": cue["start_seconds"],
                "end_seconds": cue["end_seconds"],
                "reference_text": cue["reference_text"],
                "reference_words": reference_words,
                "mode_results": mode_results,
                "frame_evidence": {
                    str(psm): frame_evidence_by_mode[psm]
                    for psm in OCR_MODES
                }
            })

            print(
                f"  PSM 6 coverage: "
                f"{mode_results['6']['coverage']:.2f}"
            )

            print(
                f"  PSM 11 coverage: "
                f"{mode_results['11']['coverage']:.2f}"
            )

            print()

    finally:
        cap.release()

    output = {
        "engine": "SUBTITLE OCR V6",
        "video": str(VIDEO_FILE),
        "reference_srt": str(SRT_FILE),
        "target_cue_ids": TARGET_CUE_IDS,
        "sample_interval": SAMPLE_INTERVAL,
        "ocr_modes": OCR_MODES,
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

    print("V6 COMPLETE")
    print("------------")
    print(f"Cues processed: {len(results)}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()