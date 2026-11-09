
import json
import re
from pathlib import Path
from collections import Counter, defaultdict


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "output"
    / "subtitle_ocr_inspection"
    / "subtitle_discrepancy_report_v1.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "output"
    / "subtitle_ocr_inspection"
)

OUTPUT_JSON = OUTPUT_DIR / "asr_difference_diagnostic_v1.json"
OUTPUT_TXT = OUTPUT_DIR / "asr_difference_diagnostic_v1.txt"


# Known character names, aliases, and story-specific proper nouns.
KNOWN_NAMES = {
    "vivienne", "vivian", "viviana", "vivien", "vivienna",
    "lucian", "lucien",
    "aria", "arya",
    "morgana",
    "alpha", "luna", "valeris", "valerius",
    "various", "valerce",
    "laurent", "laronde", "lawrence", "laurand", "lardant",
    "lance", "luke", "appyr",
    "coldwell", "coldwild", "coldwilled", "coldwilled",
    "blood", "moon", "manor", "pack",
    "velice",
}

# Common grammatical words. Their omission is often less important
# than the omission of a name or a content word.
FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but",
    "to", "of", "in", "on", "at", "by", "for",
    "from", "with", "without", "into", "onto",
    "is", "am", "are", "was", "were", "be", "been",
    "being", "do", "does", "did",
    "has", "have", "had",
    "he", "she", "it", "they", "we", "i",
    "me", "him", "her", "them", "us",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "who", "what", "when", "where", "why", "how",
    "not", "no", "yes", "so", "if", "as",
    "than", "then", "there", "here",
    "can", "could", "will", "would", "should",
    "may", "might", "must",
}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def normalize_word(word):
    return re.sub(r"[^a-z0-9']", "", str(word).lower())


def classify_missing_word(word):
    normalized = normalize_word(word)

    if not normalized:
        return "EMPTY_OR_PUNCTUATION"

    if normalized in KNOWN_NAMES:
        return "KNOWN_NAME_OR_STORY_TERM"

    if normalized in FUNCTION_WORDS:
        return "FUNCTION_WORD"

    if normalized.isdigit():
        return "NUMBER"

    return "CONTENT_WORD_OR_UNKNOWN_NAME"


def analyze():
    report = load_json(INPUT_FILE)

    candidates = [
        item for item in report.get("results", [])
        if item.get("category") == "ASR_DIFFERENCE_REVIEW"
    ]

    category_counts = Counter()
    word_counts = Counter()
    word_categories = defaultdict(Counter)
    results = []

    for item in candidates:
        missing_words = item.get("missing_from_assemblyai") or []

        classified_words = []

        for word in missing_words:
            category = classify_missing_word(word)
            normalized = normalize_word(word)

            category_counts[category] += 1
            word_counts[normalized] += 1
            word_categories[category][normalized] += 1

            classified_words.append({
                "word": word,
                "normalized": normalized,
                "type": category,
            })

        results.append({
            "subtitle_id": item.get("subtitle_id"),
            "start_seconds": item.get("start_seconds"),
            "end_seconds": item.get("end_seconds"),
            "whisper_text": item.get("whisper_text"),
            "assemblyai_text": item.get("assemblyai_matched_text"),
            "comparison_status": item.get("comparison_status"),
            "missing_words": classified_words,
        })

    results.sort(
        key=lambda item: item.get("start_seconds") or 0
    )

    statistics = {
        "total_asr_difference_cues": len(candidates),
        "cues_with_missing_words": sum(
            bool(item.get("missing_words")) for item in results
        ),
        "missing_word_type_counts": dict(category_counts),
        "most_frequent_missing_words": [
            {"word": word, "count": count}
            for word, count in word_counts.most_common(50)
        ],
    }

    output = {
        "schema_version": "1.0.0",
        "engine": "asr_difference_diagnostic_v1",
        "source": str(INPUT_FILE.relative_to(BASE_DIR)),
        "statistics": statistics,
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    lines = [
        "ASR DIFFERENCE DIAGNOSTIC V1",
        "=" * 65,
        f"ASR difference cues: {statistics['total_asr_difference_cues']}",
        f"Cues with missing words: {statistics['cues_with_missing_words']}",
        "",
        "MISSING WORD TYPE COUNTS",
        "-" * 65,
    ]

    for category, count in category_counts.most_common():
        lines.append(f"{category}: {count}")

    lines.extend([
        "",
        "MOST FREQUENT MISSING WORDS",
        "-" * 65,
    ])

    for word, count in word_counts.most_common(50):
        lines.append(f"{word}: {count}")

    lines.extend([
        "",
        "CUE-BY-CUE DETAILS",
        "=" * 65,
    ])

    for item in results:
        lines.extend([
            "",
            f"CUE {item['subtitle_id']} | "
            f"{item['start_seconds']}-{item['end_seconds']}",
            f"WHISPER: {item['whisper_text']}",
            f"ASSEMBLYAI: {item['assemblyai_text']}",
        ])

        for word in item["missing_words"]:
            lines.append(
                f"  - {word['word']} -> {word['type']}"
            )

    lines.extend([
        "",
        "=" * 65,
        "END OF REPORT",
    ])

    with OUTPUT_TXT.open("w", encoding="utf-8-sig") as file:
        file.write("\n".join(lines))

    print("ASR DIFFERENCE DIAGNOSTIC V1")
    print("=" * 50)
    print(f"ASR difference cues: {len(candidates)}")
    print("")
    for category, count in category_counts.most_common():
        print(f"{category}: {count}")
    print("")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"REPORT: {OUTPUT_TXT}")
    print("")
    print("DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    analyze()
