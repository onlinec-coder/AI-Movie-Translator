
import json
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parents[2]

COMPARISON_FILE = (
    BASE_DIR
    / "data"
    / "output"
    / "subtitle_ocr_inspection"
    / "source_comparison_v5.json"
)

CORRECTIONS_FILE = (
    BASE_DIR
    / "data"
    / "output"
    / "subtitle_ocr_inspection"
    / "subtitle_ocr_corrections_v2.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "output"
    / "subtitle_ocr_inspection"
)

OUTPUT_JSON = OUTPUT_DIR / "subtitle_discrepancy_report_v1.json"
OUTPUT_TXT = OUTPUT_DIR / "subtitle_discrepancy_report_v1.txt"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def build_correction_index(data):
    return {
        int(record["cue_index"]): record
        for record in data.get("corrections", [])
    }


def get_ocr_v7_best_coverage(ocr_v7):
    if not isinstance(ocr_v7, dict) or not ocr_v7:
        return None

    coverages = []

    for record in ocr_v7.values():
        if isinstance(record, dict):
            coverage = record.get("coverage")
            if isinstance(coverage, (int, float)):
                coverages.append(float(coverage))

    return max(coverages) if coverages else None


def classify_result(result, correction):
    whisper = result.get("whisper") or {}
    alignment = result.get("assemblyai_time_alignment") or {}

    ocr_v5 = result.get("ocr_v5")
    ocr_v7 = result.get("ocr_v7")

    # The correction registry takes precedence over embedded
    # verification metadata in the comparison result.
    manual = correction or result.get("manual_verification") or {}

    manual_confirmed = (
        manual.get("manual_visual_confirmation") is True
    )

    candidate_text = manual.get("candidate_text")
    subtitle_sequence = manual.get("burned_in_subtitle_sequence")

    missing_words = alignment.get("missing_from_assemblyai") or []
    matched_count = alignment.get("matched_token_count", 0)

    ocr_coverage = None
    if isinstance(ocr_v5, dict):
        value = ocr_v5.get("word_coverage")
        if isinstance(value, (int, float)):
            ocr_coverage = float(value)

    ocr_v7_coverage = get_ocr_v7_best_coverage(ocr_v7)

    comparison_status = result.get("comparison_status", "UNKNOWN")

    # Priority 1: human-confirmed evidence.
    if manual_confirmed and candidate_text:
        category = "MANUAL_CORRECTION_CONFIRMED"
        priority = 1

    elif manual_confirmed and subtitle_sequence:
        category = "MANUAL_SEQUENCE_CONFIRMED"
        priority = 1

    elif manual_confirmed:
        category = "MANUAL_TEXT_CONFIRMED"
        priority = 1

    # Priority 2: OCR evidence, not human confirmation.
    elif ocr_coverage is not None and ocr_coverage >= 0.999:
        category = "OCR_FULL_COVERAGE_UNVERIFIED"
        priority = 2

    elif ocr_coverage is not None or ocr_v7_coverage is not None:
        category = "OCR_PARTIAL_REVIEW"
        priority = 3

    # Priority 3: ASR disagreement.
    elif missing_words or "MISMATCH" in comparison_status.upper():
        category = "ASR_DIFFERENCE_REVIEW"
        priority = 4

    elif matched_count == 0:
        category = "NO_ASR_MATCH"
        priority = 5

    else:
        category = "ASR_ALIGNED_NO_FLAGGED_DIFFERENCE"
        priority = 6

    return {
        "subtitle_id": result.get("subtitle_id"),
        "start_seconds": result.get("start_seconds"),
        "end_seconds": result.get("end_seconds"),
        "whisper_text": whisper.get("text"),
        "assemblyai_matched_text": alignment.get("matched_text"),
        "assemblyai_matched_token_count": matched_count,
        "missing_from_assemblyai": missing_words,
        "comparison_status": comparison_status,
        "ocr_v5_status": (
            ocr_v5.get("status") if isinstance(ocr_v5, dict) else None
        ),
        "ocr_v5_coverage": ocr_coverage,
        "ocr_v7_best_coverage": ocr_v7_coverage,
        "manual_verification_status": manual.get("status"),
        "manual_visual_confirmation": manual_confirmed,
        "candidate_text": candidate_text,
        "burned_in_subtitle_sequence": subtitle_sequence,
        "verification_notes": manual.get("notes"),
        "category": category,
        "priority": priority,
    }


def create_text_report(report):
    lines = []

    lines.append("SUBTITLE DISCREPANCY REPORT V1")
    lines.append("=" * 65)

    lines.append(f"Total cues: {report['statistics']['total_cues']}")
    lines.append("")

    lines.append("CATEGORY SUMMARY")
    lines.append("-" * 65)

    for category, count in report["statistics"]["categories"].items():
        lines.append(f"{category}: {count}")

    lines.append("")
    lines.append("REVIEW CANDIDATES")
    lines.append("=" * 65)

    review_items = [
        item for item in report["results"]
        if item["priority"] <= 5
    ]

    for item in review_items:
        lines.append("")
        lines.append(
            f"CUE {item['subtitle_id']} | "
            f"{item['start_seconds']}-{item['end_seconds']}"
        )
        lines.append(f"CATEGORY: {item['category']}")
        lines.append(f"WHISPER: {item['whisper_text']}")
        lines.append(
            f"ASSEMBLYAI: {item['assemblyai_matched_text']}"
        )

        if item["missing_from_assemblyai"]:
            lines.append(
                "MISSING FROM ASSEMBLYAI: "
                + ", ".join(item["missing_from_assemblyai"])
            )

        if item["candidate_text"]:
            lines.append(
                f"MANUALLY CONFIRMED CORRECTION: "
                f"{item['candidate_text']}"
            )

        if item["burned_in_subtitle_sequence"]:
            lines.append("BURNED-IN SUBTITLE SEQUENCE:")
            for subtitle in item["burned_in_subtitle_sequence"]:
                lines.append(f"  - {subtitle}")

        if item["verification_notes"]:
            lines.append(f"NOTES: {item['verification_notes']}")

    lines.append("")
    lines.append("=" * 65)
    lines.append("END OF REPORT")

    return "\n".join(lines)


def main():
    comparison = load_json(COMPARISON_FILE)
    corrections_data = load_json(CORRECTIONS_FILE)

    correction_index = build_correction_index(corrections_data)

    results = []

    for result in comparison.get("results", []):
        cue_id = int(result["subtitle_id"])
        correction = correction_index.get(cue_id)

        results.append(
            classify_result(result, correction)
        )

    results.sort(
        key=lambda item: (
            item["priority"],
            item["start_seconds"] or 0
        )
    )

    category_counts = Counter(
        item["category"] for item in results
    )

    report = {
        "schema_version": "1.0.0",
        "engine": "subtitle_discrepancy_classifier_v1",
        "source_comparison": str(COMPARISON_FILE.relative_to(BASE_DIR)),
        "source_corrections": str(CORRECTIONS_FILE.relative_to(BASE_DIR)),
        "rules": [
            "Original Whisper SRT is not authoritative ground truth.",
            "Human visual confirmation has the highest evidence priority.",
            "OCR confirmation is not equivalent to human confirmation.",
            "Partial OCR coverage must not be treated as full confirmation.",
            "AssemblyAI differences are review candidates, not automatic corrections.",
            "No source subtitle or timing is modified by this script.",
            "No individual burned-in subtitle display timestamps are invented."
        ],
        "statistics": {
            "total_cues": len(results),
            "categories": dict(category_counts),
            "manual_visual_confirmations": sum(
                item["manual_visual_confirmation"]
                for item in results
            ),
            "cues_with_missing_assemblyai_tokens": sum(
                bool(item["missing_from_assemblyai"])
                for item in results
            ),
            "ocr_full_coverage_unverified": category_counts.get(
                "OCR_FULL_COVERAGE_UNVERIFIED", 0
            ),
            "review_candidates": sum(
                item["priority"] <= 5 for item in results
            )
        },
        "results": results
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    text_report = create_text_report(report)

    with OUTPUT_TXT.open("w", encoding="utf-8-sig") as file:
        file.write(text_report)

    print("SUBTITLE DISCREPANCY CLASSIFICATION V1")
    print("=" * 50)
    print(f"Total cues: {len(results)}")
    print("")

    for category, count in category_counts.items():
        print(f"{category}: {count}")

    print("")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"REPORT: {OUTPUT_TXT}")
    print("")
    print("CLASSIFICATION COMPLETE")


if __name__ == "__main__":
    main()
