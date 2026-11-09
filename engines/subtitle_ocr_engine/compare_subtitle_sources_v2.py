
import json
import re
from pathlib import Path
from difflib import SequenceMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRT_FILE = PROJECT_ROOT / "data" / "input" / "input.srt"

ASSEMBLYAI_FILE = (
    PROJECT_ROOT / "data" / "output" / "transcription_full_result.json"
)

OCR_DIR = (
    PROJECT_ROOT / "data" / "output" / "subtitle_ocr_inspection"
)

OCR_V5_FILE = OCR_DIR / "subtitle_ocr_v5.json"
OCR_V7_FILE = OCR_DIR / "subtitle_ocr_v7.json"
CORRECTIONS_FILE = OCR_DIR / "subtitle_ocr_corrections_v2.json"

OUTPUT_JSON = OCR_DIR / "source_comparison_v3.json"
OUTPUT_REPORT = OCR_DIR / "source_comparison_v3.txt"

# ASR үг cue-ийн заагаас хэр хол байвал ойр cue-д оноох вэ?
MAX_BOUNDARY_GAP_SECONDS = 0.60

# Эдгээр тоонуудыг үгэн хэлбэртэй адилтгана.
NUMBER_NORMALIZATION = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_word(word):
    word = word.lower().strip()
    word = re.sub(r"^[^\w']+|[^\w']+$", "", word)

    if word == "3rd":
        return "third"

    return NUMBER_NORMALIZATION.get(word, word)


def tokenize(text):
    return [
        normalized
        for word in re.findall(r"[\w']+", text.lower())
        if (normalized := normalize_word(word))
    ]


def parse_timestamp(value):
    hours, minutes, seconds = value.replace(",", ".").split(":")

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + float(seconds)
    )


def read_srt(path):
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip())

    subtitles = []

    for block in blocks:
        lines = block.strip().splitlines()

        if len(lines) < 3:
            continue

        try:
            subtitle_id = int(lines[0].strip())
        except ValueError:
            subtitle_id = len(subtitles) + 1

        match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})",
            lines[1]
        )

        if not match:
            continue

        start = parse_timestamp(match.group(1))
        end = parse_timestamp(match.group(2))
        text = " ".join(lines[2:]).strip()

        subtitles.append({
            "subtitle_id": subtitle_id,
            "start_seconds": start,
            "end_seconds": end,
            "text": text,
            "words": tokenize(text)
        })

    return subtitles


def collect_assemblyai_words(data):
    words = []

    for segment in data.get("segments", []):
        for word in segment.get("words", []):
            start = word.get("start_seconds")
            end = word.get("end_seconds")

            if start is None or end is None:
                continue

            text = word.get("text", "")

            words.append({
                "text": text,
                "normalized": normalize_word(text),
                "start_seconds": float(start),
                "end_seconds": float(end),
                "confidence": word.get("confidence"),
                "speaker_id": word.get("speaker_id")
            })

    return sorted(
        words,
        key=lambda item: item["start_seconds"]
    )


def distance_to_interval(point, start, end):
    if point < start:
        return start - point

    if point > end:
        return point - end

    return 0.0


def assign_assemblyai_words(subtitles, all_words):
    """
    ASR үг бүрийг хамгийн тохирох ганц subtitle cue-д онооно.

    1. Cue-тэй хугацаагаар давхцаж байвал хамгийн их
       давхцалтай cue-г сонгоно.
    2. Давхцал байхгүй бол хамгийн ойр cue-г сонгоно.
    3. Хамгийн ойр cue хүртэлх зай 0.60 секундээс их
       бол тухайн үгийг оноохгүй.
    4. Нэг ASR үгийг хоёр cue-д давхар оноохгүй.
    """

    assigned = {
        cue["subtitle_id"]: []
        for cue in subtitles
    }

    unassigned = []

    for word in all_words:
        word_start = word["start_seconds"]
        word_end = word["end_seconds"]
        midpoint = (word_start + word_end) / 2

        best_overlap = None
        best_overlap_duration = 0.0

        nearest_cue = None
        nearest_distance = float("inf")

        for cue in subtitles:
            cue_start = cue["start_seconds"]
            cue_end = cue["end_seconds"]

            overlap = max(
                0.0,
                min(word_end, cue_end)
                - max(word_start, cue_start)
            )

            distance = distance_to_interval(
                midpoint,
                cue_start,
                cue_end
            )

            if overlap > best_overlap_duration:
                best_overlap_duration = overlap
                best_overlap = cue

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_cue = cue

        if best_overlap is not None:
            selected_cue = best_overlap
            assignment_method = "TIME_OVERLAP"
            boundary_distance = 0.0

        elif (
            nearest_cue is not None
            and nearest_distance <= MAX_BOUNDARY_GAP_SECONDS
        ):
            selected_cue = nearest_cue
            assignment_method = "NEAR_BOUNDARY"
            boundary_distance = round(nearest_distance, 4)

        else:
            unassigned.append({
                **word,
                "assignment_status": "UNASSIGNED",
                "nearest_cue_id": (
                    nearest_cue["subtitle_id"]
                    if nearest_cue
                    else None
                ),
                "nearest_cue_distance": (
                    round(nearest_distance, 4)
                    if nearest_cue
                    else None
                )
            })
            continue

        assigned_word = {
            **word,
            "assignment_method": assignment_method,
            "boundary_distance_seconds": boundary_distance
        }

        assigned[selected_cue["subtitle_id"]].append(
            assigned_word
        )

    return assigned, unassigned


def compare_tokens(reference, observed):
    matcher = SequenceMatcher(
        None,
        reference,
        observed,
        autojunk=False
    )

    matched = []
    missing = []
    extra = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            matched.extend(reference[i1:i2])

        elif tag == "replace":
            missing.extend(reference[i1:i2])
            extra.extend(observed[j1:j2])

        elif tag == "delete":
            missing.extend(reference[i1:i2])

        elif tag == "insert":
            extra.extend(observed[j1:j2])

    return {
        "matched": matched,
        "missing_from_assemblyai": missing,
        "extra_in_assemblyai": extra,
        "sequence_similarity": round(matcher.ratio(), 4)
    }


def index_results(data, key):
    return {
        item.get(key): item
        for item in data.get("results", [])
        if item.get(key) is not None
    }


def get_v7_evidence(item):
    if not item:
        return None

    modes = item.get("mode_results", {})
    result = {}

    for mode, mode_data in modes.items():
        core = mode_data.get("core", {})

        result[mode] = {
            "coverage": core.get("coverage"),
            "matched_words": core.get("matched_words", []),
            "missing_words": core.get("missing_words", []),
            "unmatched_ocr_words": core.get(
                "unmatched_ocr_words", []
            )
        }

    return result


def main():
    print("SUBTITLE SOURCE COMPARISON V3")
    print("=" * 60)

    required_files = [
        SRT_FILE,
        ASSEMBLYAI_FILE,
        OCR_V5_FILE,
        OCR_V7_FILE,
        CORRECTIONS_FILE
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    subtitles = read_srt(SRT_FILE)

    assemblyai_data = load_json(ASSEMBLYAI_FILE)
    ocr_v5_data = load_json(OCR_V5_FILE)
    ocr_v7_data = load_json(OCR_V7_FILE)
    corrections_data = load_json(CORRECTIONS_FILE)

    assemblyai_words = collect_assemblyai_words(
        assemblyai_data
    )

    assigned_words, unassigned_words = (
        assign_assemblyai_words(
            subtitles,
            assemblyai_words
        )
    )

    ocr_v5 = index_results(ocr_v5_data, "subtitle_id")
    ocr_v7 = index_results(ocr_v7_data, "subtitle_id")

    corrections = {
        item["cue_index"]: item
        for item in corrections_data.get("corrections", [])
    }

    comparison_results = []

    exact_matches = 0
    mismatches = 0
    no_asr_words = 0
    manual_confirmed = 0

    overlap_assignments = 0
    boundary_assignments = 0

    for cue in subtitles:
        asr_words = assigned_words[cue["subtitle_id"]]

        asr_text = " ".join(
            word["text"] for word in asr_words
        )

        asr_tokens = []

        for word in asr_words:
            asr_tokens.extend(tokenize(word["text"]))

            if word["assignment_method"] == "TIME_OVERLAP":
                overlap_assignments += 1

            elif word["assignment_method"] == "NEAR_BOUNDARY":
                boundary_assignments += 1

        token_comparison = compare_tokens(
            cue["words"],
            asr_tokens
        )

        if not asr_words:
            comparison_status = "NO_ASR_WORDS"
            no_asr_words += 1

        elif (
            not token_comparison["missing_from_assemblyai"]
            and not token_comparison["extra_in_assemblyai"]
        ):
            comparison_status = "EXACT_MATCH"
            exact_matches += 1

        else:
            comparison_status = "MISMATCH"
            mismatches += 1

        v5 = ocr_v5.get(cue["subtitle_id"])
        v7 = ocr_v7.get(cue["subtitle_id"])
        correction = corrections.get(cue["subtitle_id"])

        if (
            correction
            and correction.get("manual_visual_confirmation") is True
        ):
            manual_confirmed += 1

        result = {
            "subtitle_id": cue["subtitle_id"],
            "start_seconds": cue["start_seconds"],
            "end_seconds": cue["end_seconds"],
            "whisper": {
                "text": cue["text"],
                "words": cue["words"]
            },
            "assemblyai": {
                "text": asr_text,
                "word_count": len(asr_words),
                "speaker_ids": sorted({
                    word["speaker_id"]
                    for word in asr_words
                    if word["speaker_id"]
                }),
                "token_comparison": token_comparison,
                "assignment_methods": {
                    "time_overlap": sum(
                        1 for word in asr_words
                        if word["assignment_method"] == "TIME_OVERLAP"
                    ),
                    "near_boundary": sum(
                        1 for word in asr_words
                        if word["assignment_method"] == "NEAR_BOUNDARY"
                    )
                },
                "words": asr_words
            },
            "comparison_status": comparison_status,
            "ocr_v5": None,
            "ocr_v7": get_v7_evidence(v7),
            "manual_verification": correction
        }

        if v5:
            result["ocr_v5"] = {
                "status": v5.get("status"),
                "word_coverage": v5.get("word_coverage"),
                "matched_words": v5.get("matched_words", []),
                "missing_words": v5.get("missing_words", []),
                "frames_checked": v5.get("frames_checked")
            }

        comparison_results.append(result)

    OCR_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "schema_version": "1.0.0",
        "engine": "subtitle_source_comparison_v3",
        "analysis_window": {
            "start_seconds": 0,
            "end_seconds": None,
            "scope": "FULL_MOVIE"
        },
        "source_authority": {
            "whisper": "ASR_REFERENCE_NOT_GROUND_TRUTH",
            "assemblyai": "INDEPENDENT_ASR_NOT_GROUND_TRUTH",
            "burned_in_subtitles": "VISUAL_REFERENCE",
            "manual_verification": "HUMAN_CONFIRMED_WHEN_AVAILABLE"
        },
        "assignment_settings": {
            "method": "MAXIMUM_TIME_OVERLAP_THEN_NEAREST_BOUNDARY",
            "max_boundary_gap_seconds": MAX_BOUNDARY_GAP_SECONDS,
            "each_assemblyai_word_assigned_to_maximum_one_cue": True
        },
        "sources": {
            "whisper": str(SRT_FILE.relative_to(PROJECT_ROOT)),
            "assemblyai": str(
                ASSEMBLYAI_FILE.relative_to(PROJECT_ROOT)
            ),
            "ocr_v5": str(OCR_V5_FILE.relative_to(PROJECT_ROOT)),
            "ocr_v7": str(OCR_V7_FILE.relative_to(PROJECT_ROOT)),
            "manual_verification": str(
                CORRECTIONS_FILE.relative_to(PROJECT_ROOT)
            )
        },
        "statistics": {
            "whisper_cues": len(subtitles),
            "assemblyai_words": len(assemblyai_words),
            "assigned_assemblyai_words": (
                len(assemblyai_words) - len(unassigned_words)
            ),
            "unassigned_assemblyai_words": len(unassigned_words),
            "time_overlap_assignments": overlap_assignments,
            "near_boundary_assignments": boundary_assignments,
            "exact_matches": exact_matches,
            "mismatches": mismatches,
            "cues_without_asr_words": no_asr_words,
            "ocr_v5_records": len(ocr_v5),
            "ocr_v7_records": len(ocr_v7),
            "manual_confirmed_records": manual_confirmed
        },
        "unassigned_assemblyai_words": unassigned_words,
        "results": comparison_results
    }

    OUTPUT_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    report_lines = [
        "SUBTITLE SOURCE COMPARISON V3",
        "=" * 70,
        "Scope: FULL MOVIE",
        "",
        "ASSIGNMENT SETTINGS",
        "=" * 70,
        f"Maximum boundary gap: {MAX_BOUNDARY_GAP_SECONDS}s",
        f"Time-overlap assignments: {overlap_assignments}",
        f"Near-boundary assignments: {boundary_assignments}",
        f"Unassigned ASR words: {len(unassigned_words)}",
        "",
        "SUMMARY",
        "=" * 70,
        f"Whisper cues: {len(subtitles)}",
        f"AssemblyAI words: {len(assemblyai_words)}",
        f"Assigned AssemblyAI words: {len(assemblyai_words) - len(unassigned_words)}",
        f"Exact matches: {exact_matches}",
        f"Mismatches: {mismatches}",
        f"Cues without ASR words: {no_asr_words}",
        f"OCR V5 records: {len(ocr_v5)}",
        f"OCR V7 records: {len(ocr_v7)}",
        f"Manually confirmed records: {manual_confirmed}",
        "",
        "MISMATCHED CUES",
        "=" * 70
    ]

    for item in comparison_results:
        if item["comparison_status"] == "EXACT_MATCH":
            continue

        comparison = item["assemblyai"]["token_comparison"]

        report_lines.extend([
            "",
            (
                f"CUE {item['subtitle_id']} | "
                f"{item['start_seconds']:.3f}-"
                f"{item['end_seconds']:.3f} | "
                f"{item['comparison_status']}"
            ),
            f"WHISPER: {item['whisper']['text']}",
            f"ASSEMBLYAI: {item['assemblyai']['text']}",
            (
                "ASR similarity: "
                f"{comparison['sequence_similarity']}"
            ),
            (
                "Missing from AssemblyAI: "
                f"{comparison['missing_from_assemblyai']}"
            ),
            (
                "Extra in AssemblyAI: "
                f"{comparison['extra_in_assemblyai']}"
            ),
            (
                "Assignment methods: "
                f"{item['assemblyai']['assignment_methods']}"
            ),
            (
                "OCR V5: "
                f"{item['ocr_v5']}"
            ),
            (
                "OCR V7: "
                f"{item['ocr_v7']}"
            ),
            (
                "Manual status: "
                + (
                    item["manual_verification"].get(
                        "status", "UNKNOWN"
                    )
                    if item["manual_verification"]
                    else "NOT_RECORDED"
                )
            )
        ])

    if unassigned_words:
        report_lines.extend([
            "",
            "UNASSIGNED ASSEMBLYAI WORDS",
            "=" * 70
        ])

        for word in unassigned_words:
            report_lines.append(
                f"{word['start_seconds']:.3f}-"
                f"{word['end_seconds']:.3f} | "
                f"{word['text']} | "
                f"nearest cue={word['nearest_cue_id']} | "
                f"distance={word['nearest_cue_distance']}"
            )

    OUTPUT_REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print(f"Whisper cues: {len(subtitles)}")
    print(f"AssemblyAI words: {len(assemblyai_words)}")
    print(f"Assigned ASR words: {len(assemblyai_words) - len(unassigned_words)}")
    print(f"Unassigned ASR words: {len(unassigned_words)}")
    print(f"Time-overlap assignments: {overlap_assignments}")
    print(f"Near-boundary assignments: {boundary_assignments}")
    print(f"Exact matches: {exact_matches}")
    print(f"Mismatches: {mismatches}")
    print(f"Cues without ASR words: {no_asr_words}")
    print()
    print("COMPARISON COMPLETE")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"REPORT: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
