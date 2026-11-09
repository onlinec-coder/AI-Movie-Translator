
import cv2
import json
import re
import pytesseract

from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter
from pytesseract import Output


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_FILE = PROJECT_ROOT / "data" / "input" / "Husbands.mp4"
SRT_FILE = PROJECT_ROOT / "data" / "input" / "input.srt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "subtitle_ocr_inspection"
OUTPUT_JSON = OUTPUT_DIR / "subtitle_ocr_v7.json"

TARGET_CUE_IDS = [3, 4, 5, 6]

SAMPLE_INTERVAL = 0.25
TIME_PADDING = 0.75

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

            cues.append({
                "subtitle_id": cue_id,
                "start_seconds": start,
                "end_seconds": end,
                "reference_text": " ".join(lines[2:]).strip()
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

    return (
        SequenceMatcher(
            None,
            ocr_word,
            reference_word
        ).ratio() >= MATCH_THRESHOLD
    )


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


def classify_time(timestamp, start, end):
    if timestamp < start:
        return "BEFORE"

    if timestamp > end:
        return "AFTER"

    return "CORE"


def analyze_evidence(frames, reference_words):
    matched = set()
    unmatched = Counter()
    best_confidence = {}

    for frame in frames:
        for word in frame["words"]:
            ocr_word = word["normalized"]

            reference_match = None

            for ref_word in reference_words:
                if words_match(ocr_word, ref_word):
                    reference_match = ref_word
                    break

            if reference_match:
                matched.add(reference_match)

                best_confidence[reference_match] = max(
                    best_confidence.get(reference_match, 0),
                    word["confidence"]
                )
            else:
                unmatched[ocr_word] += 1

    missing = [
        word for word in reference_words
        if word not in matched
    ]

    coverage = (
        len(matched) / len(set(reference_words))
        if reference_words else 0
    )

    return {
        "coverage": round(coverage, 3),
        "matched_words": sorted(matched),
        "missing_words": missing,
        "best_confidence": best_confidence,
        "unmatched_ocr_words": [
            {
                "word": word,
                "frame_count": count
            }
            for word, count in unmatched.most_common()
        ]
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not VIDEO_FILE.exists():
        raise FileNotFoundError(VIDEO_FILE)

    if not SRT_FILE.exists():
        raise FileNotFoundError(SRT_FILE)

    cues = parse_srt(SRT_FILE)

    selected = [
        cue for cue in cues
        if cue["subtitle_id"] in TARGET_CUE_IDS
    ]

    if len(selected) != len(TARGET_CUE_IDS):
        raise RuntimeError(
            "Not all target subtitle IDs were found."
        )

    cap = cv2.VideoCapture(str(VIDEO_FILE))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        raise RuntimeError("Invalid video FPS.")

    results = []

    print("SUBTITLE OCR ENGINE V7")
    print("----------------------")
    print(f"FPS: {fps}")
    print(f"Target cues: {TARGET_CUE_IDS}")
    print(f"Time padding: {TIME_PADDING}s")
    print()

    try:
        for cue in selected:
            cue_id = cue["subtitle_id"]
            start = cue["start_seconds"]
            end = cue["end_seconds"]

            reference_words = tokenize(
                cue["reference_text"]
            )

            window_start = max(0, start - TIME_PADDING)
            window_end = end + TIME_PADDING

            frames_by_mode = {
                psm: [] for psm in OCR_MODES
            }

            print(
                f"CUE {cue_id}: "
                f"{start:.3f}-{end:.3f}"
            )

            timestamp = window_start

            while timestamp <= window_end:
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

                zone = classify_time(
                    timestamp,
                    start,
                    end
                )

                for psm in OCR_MODES:
                    words = read_ocr_words(
                        processed,
                        psm
                    )

                    frames_by_mode[psm].append({
                        "timestamp": round(timestamp, 3),
                        "zone": zone,
                        "ocr_text": " ".join(
                            word["text"] for word in words
                        ),
                        "words": words
                    })

                timestamp += SAMPLE_INTERVAL

            mode_results = {}

            for psm in OCR_MODES:
                all_frames = frames_by_mode[psm]

                core_frames = [
                    frame for frame in all_frames
                    if frame["zone"] == "CORE"
                ]

                extended_frames = [
                    frame for frame in all_frames
                    if frame["zone"] != "CORE"
                ]

                mode_results[str(psm)] = {
                    "core": analyze_evidence(
                        core_frames,
                        reference_words
                    ),
                    "extended_window": analyze_evidence(
                        all_frames,
                        reference_words
                    ),
                    "outside_core": analyze_evidence(
                        extended_frames,
                        reference_words
                    )
                }

                print(
                    f"  PSM {psm}: "
                    f"CORE={mode_results[str(psm)]['core']['coverage']:.2f} | "
                    f"EXTENDED={mode_results[str(psm)]['extended_window']['coverage']:.2f}"
                )

            results.append({
                "subtitle_id": cue_id,
                "start_seconds": start,
                "end_seconds": end,
                "window_start": round(window_start, 3),
                "window_end": round(window_end, 3),
                "reference_text": cue["reference_text"],
                "reference_words": reference_words,
                "mode_results": mode_results,
                "frame_evidence": {
                    str(psm): frames_by_mode[psm]
                    for psm in OCR_MODES
                }
            })

            print()

    finally:
        cap.release()

    output = {
        "engine": "SUBTITLE OCR V7",
        "video": str(VIDEO_FILE),
        "reference_srt": str(SRT_FILE),
        "target_cue_ids": TARGET_CUE_IDS,
        "sample_interval": SAMPLE_INTERVAL,
        "time_padding": TIME_PADDING,
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

    print("V7 COMPLETE")
    print("------------")
    print(f"Cues processed: {len(results)}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
