
import json
import re
from pathlib import Path
from difflib import SequenceMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRT_FILE = PROJECT_ROOT / "data" / "input" / "input.srt"
ASSEMBLYAI_FILE = (
    PROJECT_ROOT / "data" / "output" / "transcription_full_result.json"
)
OCR_V5_FILE = (
    PROJECT_ROOT / "data" / "output" / "subtitle_ocr_inspection"
    / "subtitle_ocr_v5.json"
)
OCR_V7_FILE = (
    PROJECT_ROOT / "data" / "output" / "subtitle_ocr_inspection"
    / "subtitle_ocr_v7.json"
)
CORRECTIONS_FILE = (
    PROJECT_ROOT / "data" / "output" / "subtitle_ocr_inspection"
    / "subtitle_ocr_corrections_v2.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "output" / "subtitle_ocr_inspection"
)

OUTPUT_JSON = OUTPUT_DIR / "source_comparison_v1.json"
OUTPUT_REPORT = OUTPUT_DIR / "source_comparison_v1.txt"

ANALYSIS_END_SECONDS = 30.0
WORD_TIME_TOLERANCE =  0.0


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_word(word):
    word = word.lower().strip()
    word = re.sub(r"^[^\w']+|[^\w']+$", "", word)
    return word


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

        timestamp_line = lines[1]

        match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})",
            timestamp_line
        )

        if not match:
            continue

        start = parse_timestamp(match.group(1))
        end = parse_timestamp(match.group(2))
        text = " ".join(lines[2:]).strip()

        if start >= ANALYSIS_END_SECONDS:
            continue

        subtitles.append({
            "subtitle_id": len(subtitles) + 1,
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

            if start >= ANALYSIS_END_SECONDS:
                continue

            words.append({
                "text": word.get("text", ""),
                "normalized": normalize_word(word.get("text", "")),
                "start_seconds": start,
                "end_seconds": end,
                "confidence": word.get("confidence"),
                "speaker_id": word.get("speaker_id")
            })

    return sorted(words, key=lambda item: item["start_seconds"])


def get_assemblyai_words_for_cue(cue, all_words):
    start = cue["start_seconds"] - WORD_TIME_TOLERANCE
    end = cue["end_seconds"] + WORD_TIME_TOLERANCE

    return [
        word for word in all_words
        if start <= (
            word["start_seconds"] + word["end_seconds"]
        ) / 2 <= end
    ]


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
        "sequence_similarity": round(
            matcher.ratio(), 4
        )
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
    print("SUBTITLE SOURCE COMPARISON V1")
    print("=" * 60)

    for path in [
        SRT_FILE,
        ASSEMBLYAI_FILE,
        OCR_V5_FILE,
        OCR_V7_FILE,
        CORRECTIONS_FILE
    ]:
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

    ocr_v5 = index_results(
        ocr_v5_data, "subtitle_id"
    )

    ocr_v7 = index_results(
        ocr_v7_data, "subtitle_id"
    )

    corrections = {
        item["cue_index"]: item
        for item in corrections_data.get("corrections", [])
    }

    comparison_results = []

    for cue in subtitles:
        asr_words = get_assemblyai_words_for_cue(
            cue, assemblyai_words
        )

        asr_text = " ".join(
            word["text"] for word in asr_words
        )

        asr_tokens = []

        for word in asr_words:
                for token in tokenize(word["text"]):
                    if token == "3rd":
                        token = "third"

                    asr_tokens.append(token)

        

        token_comparison = compare_tokens(
            cue["words"], asr_tokens
        )

        v5 = ocr_v5.get(cue["subtitle_id"])
        v7 = ocr_v7.get(cue["subtitle_id"])
        correction = corrections.get(cue["subtitle_id"])

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
                "words": asr_words
            },
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

    output = {
        "schema_version": "1.0.0",
        "engine": "subtitle_source_comparison_v1",
        "analysis_window": {
            "start_seconds": 0,
            "end_seconds": ANALYSIS_END_SECONDS
        },
        "sources": {
            "whisper": str(SRT_FILE.relative_to(PROJECT_ROOT)),
            "assemblyai": str(
                ASSEMBLYAI_FILE.relative_to(PROJECT_ROOT)
            ),
            "ocr_v5": str(
                OCR_V5_FILE.relative_to(PROJECT_ROOT)
            ),
            "ocr_v7": str(
                OCR_V7_FILE.relative_to(PROJECT_ROOT)
            ),
            "manual_verification": str(
                CORRECTIONS_FILE.relative_to(PROJECT_ROOT)
            )
        },
        "statistics": {
            "whisper_cues": len(subtitles),
            "assemblyai_words_in_window": len(assemblyai_words),
            "ocr_v5_cues": len(ocr_v5),
            "ocr_v7_cues": len(ocr_v7),
            "manual_records": len(corrections)
        },
        "results": comparison_results
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    report_lines = [
        "SUBTITLE SOURCE COMPARISON V1",
        "=" * 65,
        f"Whisper cues: {len(subtitles)}",
        f"AssemblyAI words: {len(assemblyai_words)}",
        f"OCR V5 records: {len(ocr_v5)}",
        f"OCR V7 records: {len(ocr_v7)}",
        "",
        "CUE-BY-CUE RESULTS",
        "=" * 65
    ]

    for item in comparison_results:
        comparison = item["assemblyai"]["token_comparison"]

        report_lines.extend([
            "",
            (
                f"CUE {item['subtitle_id']} | "
                f"{item['start_seconds']:.3f}-"
                f"{item['end_seconds']:.3f}"
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
        item["manual_verification"].get("status", "UNKNOWN")
        if item["manual_verification"]
        else "NOT_RECORDED"
    )
)
        ])

    OUTPUT_REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print(f"Whisper cues: {len(subtitles)}")
    print(f"AssemblyAI words: {len(assemblyai_words)}")
    print(f"OCR V5 records: {len(ocr_v5)}")
    print(f"OCR V7 records: {len(ocr_v7)}")
    print()
    print("COMPARISON COMPLETE")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"REPORT: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
