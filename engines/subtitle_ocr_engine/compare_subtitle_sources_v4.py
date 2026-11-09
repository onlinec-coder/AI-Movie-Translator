
import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict


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

OUTPUT_JSON = OCR_DIR / "source_comparison_v4.json"
OUTPUT_REPORT = OCR_DIR / "source_comparison_v4.txt"

NUMBER_NORMALIZATION = {
    str(i): word for i, word in enumerate([
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen", "twenty"
    ])
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
    result = []

    for word in re.findall(r"[\w']+", text.lower()):
        normalized = normalize_word(word)

        if normalized:
            result.append(normalized)

    return result


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

            original_text = word.get("text", "")
            normalized_tokens = tokenize(original_text)

            for token in normalized_tokens:
                words.append({
                    "text": original_text,
                    "normalized": token,
                    "start_seconds": float(start),
                    "end_seconds": float(end),
                    "confidence": word.get("confidence"),
                    "speaker_id": word.get("speaker_id")
                })

    return sorted(
        words,
        key=lambda item: (
            item["start_seconds"],
            item["end_seconds"]
        )
    )


def build_whisper_tokens(subtitles):
    tokens = []

    for cue in subtitles:
        for word_index, word in enumerate(cue["words"]):
            tokens.append({
                "normalized": word,
                "subtitle_id": cue["subtitle_id"],
                "word_index": word_index,
                "start_seconds": cue["start_seconds"],
                "end_seconds": cue["end_seconds"]
            })

    return tokens


def build_global_alignment(whisper_tokens, assemblyai_tokens):
    """
    Compare the complete word sequences.

    Words are aligned globally, not forced into individual subtitle cues.
    """

    whisper_sequence = [
        item["normalized"] for item in whisper_tokens
    ]

    assemblyai_sequence = [
        item["normalized"] for item in assemblyai_tokens
    ]

    matcher = SequenceMatcher(
        None,
        whisper_sequence,
        assemblyai_sequence,
        autojunk=True
    )

    matched_whisper = {}
    matched_assemblyai = {}

    missing_whisper = []
    extra_assemblyai = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            for whisper_index, asr_index in zip(
                range(i1, i2),
                range(j1, j2)
            ):
                matched_whisper[whisper_index] = asr_index
                matched_assemblyai[asr_index] = whisper_index

        elif tag in ("delete", "replace"):
            for index in range(i1, i2):
                missing_whisper.append(index)

        if tag in ("insert", "replace"):
            for index in range(j1, j2):
                extra_assemblyai.append(index)

    return {
        "matcher": matcher,
        "matched_whisper": matched_whisper,
        "matched_assemblyai": matched_assemblyai,
        "missing_whisper": missing_whisper,
        "extra_assemblyai": extra_assemblyai
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
    print("SUBTITLE SOURCE COMPARISON V4")
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

    whisper_tokens = build_whisper_tokens(subtitles)

    assemblyai_tokens = collect_assemblyai_words(
        assemblyai_data
    )

    alignment = build_global_alignment(
        whisper_tokens,
        assemblyai_tokens
    )

    matched_whisper = alignment["matched_whisper"]
    matched_assemblyai = alignment["matched_assemblyai"]

    missing_whisper = set(alignment["missing_whisper"])
    extra_assemblyai = set(alignment["extra_assemblyai"])

    ocr_v5 = index_results(ocr_v5_data, "subtitle_id")
    ocr_v7 = index_results(ocr_v7_data, "subtitle_id")

    corrections = {
        item["cue_index"]: item
        for item in corrections_data.get("corrections", [])
    }

    whisper_by_cue = defaultdict(list)
    matched_asr_by_cue = defaultdict(list)
    missing_by_cue = defaultdict(list)

    for index, token in enumerate(whisper_tokens):
        cue_id = token["subtitle_id"]
        whisper_by_cue[cue_id].append(token)

        if index in matched_whisper:
            asr_index = matched_whisper[index]
            matched_asr_by_cue[cue_id].append(
                assemblyai_tokens[asr_index]
            )

        elif index in missing_whisper:
            missing_by_cue[cue_id].append(token["normalized"])

    comparison_results = []

    exact_matches = 0
    mismatches = 0
    no_asr_matches = 0
    manual_confirmed = 0

    for cue in subtitles:
        cue_id = cue["subtitle_id"]

        cue_whisper = whisper_by_cue[cue_id]
        cue_matched_asr = matched_asr_by_cue[cue_id]
        cue_missing = missing_by_cue[cue_id]

        aligned_asr_text = " ".join(
            item["text"] for item in cue_matched_asr
        )

        missing_words = list(cue_missing)

        if not cue_matched_asr:
            status = "NO_GLOBAL_ASR_MATCH"
            no_asr_matches += 1

        elif not missing_words:
            status = "GLOBAL_MATCH"
            exact_matches += 1

        else:
            status = "GLOBAL_MISMATCH"
            mismatches += 1

        v5 = ocr_v5.get(cue_id)
        v7 = ocr_v7.get(cue_id)
        correction = corrections.get(cue_id)

        if (
            correction
            and correction.get("manual_visual_confirmation") is True
        ):
            manual_confirmed += 1

        comparison_results.append({
            "subtitle_id": cue_id,
            "start_seconds": cue["start_seconds"],
            "end_seconds": cue["end_seconds"],
            "whisper": {
                "text": cue["text"],
                "words": cue["words"]
            },
            "assemblyai_global_alignment": {
                "matched_text": aligned_asr_text,
                "matched_token_count": len(cue_matched_asr),
                "missing_from_assemblyai": missing_words
            },
            "comparison_status": status,
            "ocr_v5": (
                {
                    "status": v5.get("status"),
                    "word_coverage": v5.get("word_coverage"),
                    "matched_words": v5.get("matched_words", []),
                    "missing_words": v5.get("missing_words", []),
                    "frames_checked": v5.get("frames_checked")
                }
                if v5 else None
            ),
            "ocr_v7": get_v7_evidence(v7),
            "manual_verification": correction
        })

    extra_words = [
        {
            **assemblyai_tokens[index],
            "global_token_index": index
        }
        for index in sorted(extra_assemblyai)
    ]

    global_match_count = len(matched_whisper)
    global_similarity = (
        round(
            2 * global_match_count /
            (len(whisper_tokens) + len(assemblyai_tokens)),
            4
        )
        if whisper_tokens or assemblyai_tokens
        else 1.0
    )

    output = {
        "schema_version": "1.0.0",
        "engine": "subtitle_source_comparison_v4",
        "alignment_method": (
            "GLOBAL_WORD_SEQUENCE_ALIGNMENT"
        ),
        "source_authority": {
            "whisper": "ASR_REFERENCE_NOT_GROUND_TRUTH",
            "assemblyai": "INDEPENDENT_ASR_NOT_GROUND_TRUTH",
            "burned_in_subtitles": "VISUAL_REFERENCE",
            "manual_verification": "HUMAN_CONFIRMED_WHEN_AVAILABLE"
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
            "whisper_tokens": len(whisper_tokens),
            "assemblyai_tokens": len(assemblyai_tokens),
            "globally_matched_tokens": global_match_count,
            "globally_missing_whisper_tokens": len(missing_whisper),
            "globally_extra_assemblyai_tokens": len(extra_assemblyai),
            "global_sequence_similarity": global_similarity,
            "cues_with_global_matches": exact_matches + mismatches,
            "cues_without_global_matches": no_asr_matches,
            "ocr_v5_records": len(ocr_v5),
            "ocr_v7_records": len(ocr_v7),
            "manual_confirmed_records": manual_confirmed
        },
        "unmatched_assemblyai_words": extra_words,
        "results": comparison_results
    }

    OCR_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    report_lines = [
        "SUBTITLE SOURCE COMPARISON V4",
        "=" * 70,
        "Method: GLOBAL WORD SEQUENCE ALIGNMENT",
        "",
        "SUMMARY",
        "=" * 70,
        f"Whisper cues: {len(subtitles)}",
        f"Whisper tokens: {len(whisper_tokens)}",
        f"AssemblyAI tokens: {len(assemblyai_tokens)}",
        f"Globally matched tokens: {global_match_count}",
        f"Globally missing Whisper tokens: {len(missing_whisper)}",
        f"Globally extra AssemblyAI tokens: {len(extra_assemblyai)}",
        f"Global sequence similarity: {global_similarity}",
        f"Cues with global matches: {exact_matches + mismatches}",
        f"Cues without global matches: {no_asr_matches}",
        f"OCR V5 records: {len(ocr_v5)}",
        f"OCR V7 records: {len(ocr_v7)}",
        f"Manually confirmed records: {manual_confirmed}",
        "",
        "CUE-LEVEL RESULTS",
        "=" * 70
    ]

    for item in comparison_results:
        if item["comparison_status"] == "GLOBAL_MATCH":
            continue

        report_lines.extend([
            "",
            (
                f"CUE {item['subtitle_id']} | "
                f"{item['start_seconds']:.3f}-"
                f"{item['end_seconds']:.3f} | "
                f"{item['comparison_status']}"
            ),
            f"WHISPER: {item['whisper']['text']}",
            (
                "ASSEMBLYAI ALIGNED: "
                f"{item['assemblyai_global_alignment']['matched_text']}"
            ),
            (
                "MISSING FROM ASSEMBLYAI: "
                f"{item['assemblyai_global_alignment']['missing_from_assemblyai']}"
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

    report_lines.extend([
        "",
        "UNMATCHED ASSEMBLYAI WORDS",
        "=" * 70
    ])

    for word in extra_words:
        report_lines.append(
            f"{word['start_seconds']:.3f}-"
            f"{word['end_seconds']:.3f} | "
            f"{word['text']} | "
            f"speaker={word['speaker_id']}"
        )

    OUTPUT_REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print(f"Whisper cues: {len(subtitles)}")
    print(f"Whisper tokens: {len(whisper_tokens)}")
    print(f"AssemblyAI tokens: {len(assemblyai_tokens)}")
    print(f"Globally matched tokens: {global_match_count}")
    print(f"Missing Whisper tokens: {len(missing_whisper)}")
    print(f"Extra AssemblyAI tokens: {len(extra_assemblyai)}")
    print(f"Global sequence similarity: {global_similarity}")
    print(f"Cues without global matches: {no_asr_matches}")
    print()
    print("COMPARISON V4 COMPLETE")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"REPORT: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
