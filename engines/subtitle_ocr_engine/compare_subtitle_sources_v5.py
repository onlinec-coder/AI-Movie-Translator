
import json
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRT_FILE = PROJECT_ROOT / "data" / "input" / "input.srt"
ASSEMBLYAI_FILE = PROJECT_ROOT / "data" / "output" / "transcription_full_result.json"

OCR_DIR = PROJECT_ROOT / "data" / "output" / "subtitle_ocr_inspection"

OCR_V5_FILE = OCR_DIR / "subtitle_ocr_v5.json"
OCR_V7_FILE = OCR_DIR / "subtitle_ocr_v7.json"
CORRECTIONS_FILE = OCR_DIR / "subtitle_ocr_corrections_v2.json"

OUTPUT_JSON = OCR_DIR / "source_comparison_v5.json"
OUTPUT_REPORT = OCR_DIR / "source_comparison_v5.txt"

# Confirmed spelling variants. These are comparison aliases,
# not automatic corrections to the original subtitle.
NAME_ALIASES = {
    "vivian": "vivienne",
    "viviana": "vivienne",
    "arya": "aria",
}

COMPOUND_ALIASES = {
    ("moon", "veil"): ("moonveil",),
    ("moonveil",): ("moonveil",),
}

NUMBER_NORMALIZATION = {
    str(i): word
    for i, word in enumerate([
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

    word = NUMBER_NORMALIZATION.get(word, word)

    return NAME_ALIASES.get(word, word)


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
            tokens = tokenize(original_text)

            for token in tokens:
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


def temporal_distance(whisper_token, asr_token):
    ws = whisper_token["start_seconds"]
    we = whisper_token["end_seconds"]
    ass = asr_token["start_seconds"]
    ase = asr_token["end_seconds"]

    if ase < ws:
        return ws - ase

    if ass > we:
        return ass - we

    return 0.0



   
    

def align_with_time(whisper_tokens, assemblyai_tokens):
    """
    Overlapping time-window monotonic alignment.

    Each Whisper token belongs to one 60-second core window.
    AssemblyAI candidates are collected from an expanded window
    with 8 seconds of overlap on both sides.
    """

    WINDOW_SECONDS = 60.0
    OVERLAP_SECONDS = 8.0
    GAP_COST = 1.0
    MAX_TIME_DISTANCE = 4.0
    MAX_CELLS_PER_WINDOW = 2_000_000

    def token_time(token):
        return (
            token["start_seconds"] + token["end_seconds"]
        ) / 2.0

    whisper_windows = defaultdict(list)

    for index, token in enumerate(whisper_tokens):
        window = int(token_time(token) // WINDOW_SECONDS)
        whisper_windows[window].append(index)

    matched_whisper = {}
    matched_assemblyai = {}
    missing_whisper = []
    total_alignment_cost = 0.0

    used_assemblyai = set()
    last_matched_asr_index = -1

    all_windows = sorted(whisper_windows)

    for window in all_windows:
        whisper_indices = whisper_windows[window]

        core_start = window * WINDOW_SECONDS
        core_end = core_start + WINDOW_SECONDS

        expanded_start = core_start - OVERLAP_SECONDS
        expanded_end = core_end + OVERLAP_SECONDS

        asr_indices = [
            index
            for index, token in enumerate(assemblyai_tokens)
            if (
                expanded_start <= token_time(token) < expanded_end
                and index not in used_assemblyai
                and index > last_matched_asr_index
            )
        ]

        n = len(whisper_indices)
        m = len(asr_indices)

        if n == 0:
            continue

        if m == 0:
            missing_whisper.extend(whisper_indices)
            continue

        if (n + 1) * (m + 1) > MAX_CELLS_PER_WINDOW:
            raise RuntimeError(
                f"Window {window} is too large: "
                f"{n} Whisper tokens x {m} AssemblyAI tokens."
            )

        previous = [j * GAP_COST for j in range(m + 1)]
        back = bytearray((n + 1) * (m + 1))

        for i in range(1, n + 1):
            current = [i * GAP_COST] + [0.0] * m
            wi = whisper_tokens[whisper_indices[i - 1]]

            for j in range(1, m + 1):
                aj = assemblyai_tokens[asr_indices[j - 1]]

                distance = temporal_distance(wi, aj)

                if wi["normalized"] == aj["normalized"]:
                    substitution = (
                        min(distance / MAX_TIME_DISTANCE, 1.0)
                        * 0.20
                    )
                else:
                    substitution = (
                        1.5
                        + min(distance / MAX_TIME_DISTANCE, 1.0)
                    )

                diagonal = previous[j - 1] + substitution
                delete = previous[j] + GAP_COST
                insert = current[j - 1] + GAP_COST

                best = min(diagonal, delete, insert)
                current[j] = best

                cell = i * (m + 1) + j

                if best == diagonal:
                    back[cell] = 1
                elif best == delete:
                    back[cell] = 2
                else:
                    back[cell] = 3

            previous = current

        total_alignment_cost += previous[m]

        i = n
        j = m
        window_matches = []

        while i > 0 or j > 0:
            if i == 0:
                j -= 1
                continue

            if j == 0:
                missing_whisper.append(whisper_indices[i - 1])
                i -= 1
                continue

            direction = back[i * (m + 1) + j]

            whisper_index = whisper_indices[i - 1]
            asr_index = asr_indices[j - 1]

            if direction == 1:
                wi = whisper_tokens[whisper_index]
                aj = assemblyai_tokens[asr_index]

                if wi["normalized"] == aj["normalized"]:
                    window_matches.append((whisper_index, asr_index))
                else:
                    missing_whisper.append(whisper_index)

                i -= 1
                j -= 1

            elif direction == 2:
                missing_whisper.append(whisper_index)
                i -= 1

            else:
                j -= 1

        window_matches.reverse()

        for whisper_index, asr_index in window_matches:
            if asr_index in used_assemblyai:
                continue

            matched_whisper[whisper_index] = asr_index
            matched_assemblyai[asr_index] = whisper_index
            used_assemblyai.add(asr_index)
            last_matched_asr_index = asr_index

    extra_assemblyai = [
        index
        for index in range(len(assemblyai_tokens))
        if index not in used_assemblyai
    ]

    return {
        "matched_whisper": matched_whisper,
        "matched_assemblyai": matched_assemblyai,
        "missing_whisper": sorted(set(missing_whisper)),
        "extra_assemblyai": extra_assemblyai,
        "alignment_cost": total_alignment_cost
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

    result = {}

    for mode, mode_data in item.get("mode_results", {}).items():
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
    print("SUBTITLE SOURCE COMPARISON V5")
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
    assemblyai_tokens = collect_assemblyai_words(assemblyai_data)

    alignment = align_with_time(
        whisper_tokens,
        assemblyai_tokens
    )

    matched_whisper = alignment["matched_whisper"]
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

    results = []
    exact_matches = 0
    mismatches = 0
    no_asr_matches = 0
    manual_confirmed = 0

    for cue in subtitles:
        cue_id = cue["subtitle_id"]

        cue_matched_asr = matched_asr_by_cue[cue_id]
        cue_missing = missing_by_cue[cue_id]

        aligned_text = " ".join(
            item["text"] for item in cue_matched_asr
        )

        if not cue_matched_asr:
            status = "NO_TIME_AWARE_ASR_MATCH"
            no_asr_matches += 1
        elif not cue_missing:
            status = "TIME_AWARE_MATCH"
            exact_matches += 1
        else:
            status = "TIME_AWARE_MISMATCH"
            mismatches += 1

        correction = corrections.get(cue_id)
        if (
            correction
            and correction.get("manual_visual_confirmation") is True
        ):
            manual_confirmed += 1

        results.append({
            "subtitle_id": cue_id,
            "start_seconds": cue["start_seconds"],
            "end_seconds": cue["end_seconds"],
            "whisper": {
                "text": cue["text"],
                "words": cue["words"]
            },
            "assemblyai_time_alignment": {
                "matched_text": aligned_text,
                "matched_token_count": len(cue_matched_asr),
                "missing_from_assemblyai": list(cue_missing)
            },
            "comparison_status": status,
            "ocr_v5": (
                {
                    "status": ocr_v5[cue_id].get("status"),
                    "word_coverage": ocr_v5[cue_id].get("word_coverage"),
                    "matched_words": ocr_v5[cue_id].get("matched_words", []),
                    "missing_words": ocr_v5[cue_id].get("missing_words", []),
                    "frames_checked": ocr_v5[cue_id].get("frames_checked")
                }
                if cue_id in ocr_v5 else None
            ),
            "ocr_v7": get_v7_evidence(ocr_v7.get(cue_id)),
            "manual_verification": correction
        })

    extra_words = [
        {
            **assemblyai_tokens[index],
            "global_token_index": index
        }
        for index in sorted(extra_assemblyai)
    ]

    matched_count = len(matched_whisper)

    similarity = (
        round(
            2 * matched_count /
            (len(whisper_tokens) + len(assemblyai_tokens)),
            4
        )
        if whisper_tokens or assemblyai_tokens
        else 1.0
    )

    output = {
        "schema_version": "1.0.0",
        "engine": "subtitle_source_comparison_v5",
        "alignment_method": "TIME_AWARE_MONOTONIC_GLOBAL_ALIGNMENT",
        "source_authority": {
            "whisper": "ASR_REFERENCE_NOT_GROUND_TRUTH",
            "assemblyai": "INDEPENDENT_ASR_NOT_GROUND_TRUTH",
            "burned_in_subtitles": "VISUAL_REFERENCE",
            "manual_verification": "HUMAN_CONFIRMED_WHEN_AVAILABLE"
        },
        "sources": {
            "whisper": str(SRT_FILE.relative_to(PROJECT_ROOT)),
            "assemblyai": str(ASSEMBLYAI_FILE.relative_to(PROJECT_ROOT)),
            "ocr_v5": str(OCR_V5_FILE.relative_to(PROJECT_ROOT)),
            "ocr_v7": str(OCR_V7_FILE.relative_to(PROJECT_ROOT)),
            "manual_verification": str(CORRECTIONS_FILE.relative_to(PROJECT_ROOT))
        },
        "statistics": {
            "whisper_cues": len(subtitles),
            "whisper_tokens": len(whisper_tokens),
            "assemblyai_tokens": len(assemblyai_tokens),
            "time_aligned_tokens": matched_count,
            "missing_whisper_tokens": len(missing_whisper),
            "extra_assemblyai_tokens": len(extra_assemblyai),
            "time_aware_similarity": similarity,
            "cues_with_matches": exact_matches + mismatches,
            "cues_without_matches": no_asr_matches,
            "ocr_v5_records": len(ocr_v5),
            "ocr_v7_records": len(ocr_v7),
            "manual_confirmed_records": manual_confirmed,
            "alignment_cost": round(alignment["alignment_cost"], 3)
        },
        "unmatched_assemblyai_words": extra_words,
        "results": results
    }

    OCR_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_JSON.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    report = [
        "SUBTITLE SOURCE COMPARISON V5",
        "=" * 70,
        "Method: TIME-AWARE MONOTONIC GLOBAL ALIGNMENT",
        "",
        "SUMMARY",
        "=" * 70,
        f"Whisper cues: {len(subtitles)}",
        f"Whisper tokens: {len(whisper_tokens)}",
        f"AssemblyAI tokens: {len(assemblyai_tokens)}",
        f"Time-aligned tokens: {matched_count}",
        f"Missing Whisper tokens: {len(missing_whisper)}",
        f"Extra AssemblyAI tokens: {len(extra_assemblyai)}",
        f"Time-aware similarity: {similarity}",
        f"Cues with matches: {exact_matches + mismatches}",
        f"Cues without matches: {no_asr_matches}",
        f"Alignment cost: {alignment['alignment_cost']:.3f}",
        "",
        "CUE-LEVEL MISMATCHES",
        "=" * 70
    ]

    for item in results:
        if item["comparison_status"] == "TIME_AWARE_MATCH":
            continue

        report.extend([
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
                f"{item['assemblyai_time_alignment']['matched_text']}"
            ),
            (
                "MISSING: "
                f"{item['assemblyai_time_alignment']['missing_from_assemblyai']}"
            )
        ])

    report.extend([
        "",
        "UNMATCHED ASSEMBLYAI WORDS",
        "=" * 70
    ])

    for word in extra_words:
        report.append(
            f"{word['start_seconds']:.3f}-"
            f"{word['end_seconds']:.3f} | "
            f"{word['text']} | speaker={word['speaker_id']}"
        )

    OUTPUT_REPORT.write_text("\n".join(report), encoding="utf-8")

    print(f"Whisper cues: {len(subtitles)}")
    print(f"Whisper tokens: {len(whisper_tokens)}")
    print(f"AssemblyAI tokens: {len(assemblyai_tokens)}")
    print(f"Time-aligned tokens: {matched_count}")
    print(f"Missing Whisper tokens: {len(missing_whisper)}")
    print(f"Extra AssemblyAI tokens: {len(extra_assemblyai)}")
    print(f"Time-aware similarity: {similarity}")
    print()
    print("COMPARISON V5 COMPLETE")
    print(f"JSON: {OUTPUT_JSON}")
    print(f"REPORT: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
