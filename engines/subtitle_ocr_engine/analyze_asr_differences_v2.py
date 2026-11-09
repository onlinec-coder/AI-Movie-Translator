
import json
import re
from pathlib import Path
from collections import Counter


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

OUTPUT_JSON = OUTPUT_DIR / "asr_difference_diagnostic_v2.json"
OUTPUT_TXT = OUTPUT_DIR / "asr_difference_diagnostic_v2.txt"


# Canonical names and known aliases from the reconciled character list.
NAME_ALIASES = {
    "vivienne": "vivienne",
    "vivian": "vivienne",
    "viviana": "vivienne",
    "vivien": "vivienne",
    "vivienna": "vivienne",

    "lucian": "lucian",
    "lucien": "lucien",

    "aria": "aria",
    "arya": "aria",

    "alpha": "alpha",
    "luna": "luna",
    "valeris": "valeris",
    "valerius": "valerius",
    "various": "possible_valeris_variant",
    "valerce": "possible_valeris_variant",

    "laurent": "laurent",
    "laronde": "laurent",
    "lawrence": "possible_laurent_variant",
    "laurand": "possible_laurent_variant",
    "lardant": "possible_laurent_variant",

    "lance": "lance",
    "luke": "luke",
    "morgana": "morgana",
}

# Multi-word names and terms. These are patterns, not automatic corrections.
PROPER_NOUN_PHRASES = {
    "blood moon": "BLOOD_MOON",
    "blood moon manor": "BLOOD_MOON_MANOR",
    "coldwell manor": "COLDWELL_MANOR",
    "coldwild family": "COLDWILD_FAMILY",
    "moon veil": "MOON_VEIL",
    "chief elder laurent": "CHIEF_ELDER_LAURENT",
}

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

# Potential Whisper recognition issues found during manual discrepancy review.
SUSPICIOUS_WHISPER_TERMS = {
    "unerisval": "UNRECOGNIZED_OR_POSSIBLE_WHISPER_ERROR",
    "wolf's babe": "POSSIBLE_WOLFSBANE",
    "wolf bab": "POSSIBLE_WOLFSBANE_OR_PROPER_TERM",
    "moon veal": "POSSIBLE_MOON_VEIL",
    "cold well": "POSSIBLE_COLDWELL",
    "manner": "POSSIBLE_MANOR",
    "graze": "POSSIBLE_GAZE",
    "motor cord": "POSSIBLE_MOTORCADE",
    "door warm": "POSSIBLE_DOOR_FRAME",
    "unharried": "POSSIBLE_UNHURRIED",
    "emise": "POSSIBLE_EMISSARY",
    "clipped his hands": "POSSIBLE_CLASPED_HIS_HANDS",
    "royal sea": "POSSIBLE_ROYAL_SEAL",
}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def normalize_text(value):
    value = str(value or "").lower()
    value = value.replace("’", "'")
    value = re.sub(r"[^a-z0-9'\s-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_word(word):
    word = normalize_text(word)
    word = word.strip("-")

    # Remove possessive suffix without merging Lucian and Lucien.
    if word.endswith("'s"):
        word = word[:-2]

    return word


def classify_word(word):
    normalized = normalize_word(word)

    if not normalized:
        return "EMPTY_OR_PUNCTUATION", normalized, None

    if normalized in NAME_ALIASES:
        canonical = NAME_ALIASES[normalized]
        if canonical.startswith("possible_"):
            return "POSSIBLE_CHARACTER_NAME_VARIANT", normalized, canonical
        return "CHARACTER_NAME_OR_ALIAS", normalized, canonical

    if normalized in FUNCTION_WORDS:
        return "FUNCTION_WORD", normalized, None

    if normalized.isdigit():
        return "NUMBER", normalized, None

    return "CONTENT_WORD_OR_UNKNOWN_NAME", normalized, None


def detect_suspicious_phrases(text):
    normalized = normalize_text(text)
    findings = []

    for phrase, label in SUSPICIOUS_WHISPER_TERMS.items():
        if phrase in normalized:
            findings.append({
                "phrase": phrase,
                "flag": label,
            })

    return findings


def detect_proper_noun_phrases(text):
    normalized = normalize_text(text)
    findings = []

    for phrase, label in PROPER_NOUN_PHRASES.items():
        if phrase in normalized:
            findings.append({
                "phrase": phrase,
                "label": label,
            })

    return findings


def classify_cue(item):
    missing_words = item.get("missing_from_assemblyai") or []

    classified_words = []
    counts = Counter()

    for word in missing_words:
        category, normalized, canonical = classify_word(word)
        counts[category] += 1

        classified_words.append({
            "original": word,
            "normalized": normalized,
            "category": category,
            "canonical_candidate": canonical,
        })

    whisper_text = item.get("whisper_text") or ""

    return {
        "subtitle_id": item.get("subtitle_id"),
        "start_seconds": item.get("start_seconds"),
        "end_seconds": item.get("end_seconds"),
        "whisper_text": whisper_text,
        "assemblyai_text": item.get("assemblyai_matched_text"),
        "comparison_status": item.get("comparison_status"),
        "missing_words": classified_words,
        "missing_word_categories": dict(counts),
        "whisper_suspicious_phrases": detect_suspicious_phrases(
            whisper_text
        ),
        "proper_noun_phrases": detect_proper_noun_phrases(
            whisper_text
        ),
    }


def main():
    report = load_json(INPUT_FILE)

    candidates = [
        item for item in report.get("results", [])
        if item.get("category") == "ASR_DIFFERENCE_REVIEW"
    ]

    results = [
        classify_cue(item)
        for item in candidates
    ]

    results.sort(
        key=lambda item: item.get("start_seconds") or 0
    )

    category_counts = Counter()
    word_counts = Counter()
    suspicious_counts = Counter()

    for item in results:
        for word in item["missing_words"]:
            category_counts[word["category"]] += 1
            word_counts[word["normalized"]] += 1

        for finding in item["whisper_suspicious_phrases"]:
            suspicious_counts[finding["phrase"]] += 1

    output = {
        "schema_version": "1.0.0",
        "engine": "asr_difference_diagnostic_v2",
        "source": str(INPUT_FILE.relative_to(BASE_DIR)),
        "rules": [
            "This diagnostic does not modify subtitles.",
            "Lucian and Lucien remain separate names.",
            "Possessive suffixes are removed only for classification.",
            "Name variants are candidates, not confirmed corrections.",
            "Suspicious Whisper phrases require audio or visual verification.",
            "AssemblyAI omissions do not prove Whisper is correct."
        ],
        "statistics": {
            "total_asr_difference_cues": len(results),
            "missing_word_category_counts": dict(category_counts),
            "most_frequent_missing_words": [
                {"word": word, "count": count}
                for word, count in word_counts.most_common(50)
            ],
            "suspicious_whisper_phrase_counts": dict(
                suspicious_counts
            ),
        },
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    lines = [
        "ASR DIFFERENCE DIAGNOSTIC V2",
        "=" * 65,
        f"ASR difference cues: {len(results)}",
        "",
        "MISSING WORD CATEGORY COUNTS",
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
        "SUSPICIOUS WHISPER PHRASES",
        "-" * 65,
    ])

    for phrase, count in suspicious_counts.most_common():
        lines.append(f"{phrase}: {count}")

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
                f"  MISSING: {word['original']} "
                f"-> {word['category']}"
            )

        for finding in item["whisper_suspicious_phrases"]:
            lines.append(
                f"  WHISPER FLAG: {finding['phrase']} "
                f"-> {finding['flag']}"
            )

        for finding in item["proper_noun_phrases"]:
            lines.append(
                f"  PROPER NOUN: {finding['phrase']} "
                f"-> {finding['label']}"
            )

    lines.extend([
        "",
        "=" * 65,
        "END OF REPORT",
    ])

    with OUTPUT_TXT.open("w", encoding="utf-8-sig") as file:
        file.write("\n".join(lines))

    print("ASR DIFFERENCE DIAGNOSTIC V2")
    print("=" * 50)
    print(f"ASR difference cues: {len(results)}")
    print("")

    for category, count in category_counts.most_common():
        print(f"{category}: {count}")

    print("")
    print("SUSPICIOUS WHISPER PHRASES")

    for phrase, count in suspicious_counts.most_common():
        print(f"{phrase}: {count}")

    print("")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"REPORT: {OUTPUT_TXT}")
    print("")
    print("DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()
