
import json
import re
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from collections import Counter


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TIMELINE_FILE = PROJECT_ROOT / "data" / "output" / "timeline_reference.json"
TRANSCRIPTION_FILE = PROJECT_ROOT / "data" / "output" / "transcription_full_result.json"

OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "alignment_result_v4.json"
V3_FILE = PROJECT_ROOT / "data" / "output" / "alignment_result_v3.json"


# ============================================================
# SETTINGS
# ============================================================

GAP_PENALTY = -1.6
MISMATCH_PENALTY = -3.5

FUZZY_MATCH_THRESHOLD = 0.88
COMPOUND_MATCH_SCORE = 4.6

LOW_TEXT_SIMILARITY_THRESHOLD = 0.75
LOW_ASR_CONFIDENCE_THRESHOLD = 0.80

TIMING_LOW_COVERAGE_THRESHOLD = 0.65
TIMING_PARTIAL_COVERAGE_THRESHOLD = 0.85

# Missing-dialogue detection.
DIALOGUE_GROUP_MAX_GAP = 1.2
DIALOGUE_MIN_AVERAGE_CONFIDENCE = 0.65
SINGLE_WORD_TERM_CONFIDENCE = 0.85

# Words after the selected subtitle window are not
# considered missing dialogue in this analysis.
WINDOW_END_TOLERANCE = 0.05


TOKEN_ALIASES = {
    "lucien": "lucian",
    "lucian": "lucian",
}


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
}


# ============================================================
# FILE HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_token(token):
    token = token.lower().strip()
    token = token.replace("’", "'")

    token = re.sub(r"^[^a-z0-9']+", "", token)
    token = re.sub(r"[^a-z0-9']+$", "", token)

    if re.fullmatch(r"\d+(st|nd|rd|th)", token):
        token = re.sub(r"(st|nd|rd|th)$", "", token)

    token = NUMBER_WORDS.get(token, token)
    token = TOKEN_ALIASES.get(token, token)

    return token


def tokenize(text):
    raw_tokens = re.findall(
        r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*",
        text or ""
    )

    result = []

    for raw in raw_tokens:
        normalized = normalize_token(raw)

        if normalized:
            result.append({
                "original": raw,
                "normalized": normalized
            })

    return result


def compact_token(token):
    return re.sub(r"[^a-z0-9]", "", token)


# ============================================================
# MATCH SCORING
# ============================================================

def compare_tokens(source_token, asr_word):
    source = source_token["normalized"]
    recognized = asr_word["normalized"]

    if source == recognized:
        return 3.0, "exact"

    source_alias = TOKEN_ALIASES.get(source)
    recognized_alias = TOKEN_ALIASES.get(recognized)

    if (
        source_alias is not None
        and recognized_alias is not None
        and source_alias == recognized_alias
    ):
        return 2.7, "alias"

    similarity = SequenceMatcher(
        None,
        source,
        recognized
    ).ratio()

    if similarity >= FUZZY_MATCH_THRESHOLD:
        return 1.2, "fuzzy"

    return MISMATCH_PENALTY, None


def compare_compound(source_group, asr_group):
    """
    Exact compound matching.

    Important:
    Source tokens must belong to the same subtitle.
    Compound matching cannot cross subtitle boundaries.
    """

    if not source_group or not asr_group:
        return False

    subtitle_ids = {
        token["subtitle_id"]
        for token in source_group
    }

    if len(subtitle_ids) != 1:
        return False

    source_text = "".join(
        compact_token(token["normalized"])
        for token in source_group
    )

    asr_text = "".join(
        compact_token(word["normalized"])
        for word in asr_group
    )

    if not source_text or not asr_text:
        return False

    return source_text == asr_text


# ============================================================
# LOAD ASR WORDS
# ============================================================

def extract_asr_words(transcription):
    words = []

    for segment in transcription.get("segments", []):
        segment_speaker = segment.get("speaker_id")

        for word in segment.get("words", []):
            text = word.get("text", "").strip()

            if not text:
                continue

            start = word.get("start_seconds")
            end = word.get("end_seconds")

            if start is None or end is None:
                continue

            try:
                start = float(start)
                end = float(end)
            except (TypeError, ValueError):
                continue

            if end < start:
                continue

            tokens = tokenize(text)

            for token in tokens:
                words.append({
                    "word_index": len(words),
                    "text": text,
                    "normalized": token["normalized"],
                    "start_seconds": start,
                    "end_seconds": end,
                    "confidence": word.get("confidence"),
                    "speaker_id": (
                        word.get("speaker_id")
                        or segment_speaker
                    ),
                    "segment_id": segment.get("segment_id"),
                })

    words.sort(
        key=lambda item: (
            item["start_seconds"],
            item["end_seconds"]
        )
    )

    for index, word in enumerate(words):
        word["word_index"] = index

    return words


# ============================================================
# BUILD SOURCE SUBTITLE TOKENS
# ============================================================

def extract_subtitle_tokens(timeline):
    subtitles = timeline.get("segments", [])
    tokens = []

    for subtitle in subtitles:
        subtitle_id = subtitle["subtitle_id"]

        for token_index, token in enumerate(
            tokenize(subtitle.get("text", ""))
        ):
            tokens.append({
                "subtitle_id": subtitle_id,
                "token_index": token_index,
                "original": token["original"],
                "normalized": token["normalized"],
            })

    return tokens


# ============================================================
# GLOBAL SEQUENCE ALIGNMENT V4
# ============================================================

def global_sequence_alignment(source_tokens, asr_words):
    """
    Global monotonic alignment.

    Supported:
        1 source token <-> 1 ASR word
        2 source tokens <-> 1 ASR word
        1 source token <-> 2 ASR words

    Text matching does not use timestamps.
    """

    n = len(source_tokens)
    m = len(asr_words)

    scores = [
        [0.0] * (m + 1)
        for _ in range(n + 1)
    ]

    trace = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        scores[i][0] = i * GAP_PENALTY
        trace[i][0] = (1, 0, None)

    for j in range(1, m + 1):
        scores[0][j] = j * GAP_PENALTY
        trace[0][j] = (0, 1, None)

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            candidates = []

            # 1-to-1
            match_score, quality = compare_tokens(
                source_tokens[i - 1],
                asr_words[j - 1]
            )

            candidates.append((
                scores[i - 1][j - 1] + match_score,
                1,
                1,
                quality,
                0
            ))

            # 2 source tokens -> 1 ASR word
            if i >= 2:
                source_group = source_tokens[i - 2:i]
                asr_group = [asr_words[j - 1]]

                if compare_compound(source_group, asr_group):
                    candidates.append((
                        scores[i - 2][j - 1]
                        + COMPOUND_MATCH_SCORE,
                        2,
                        1,
                        "compound",
                        1
                    ))

            # 1 source token -> 2 ASR words
            if j >= 2:
                source_group = [source_tokens[i - 1]]
                asr_group = asr_words[j - 2:j]

                if compare_compound(source_group, asr_group):
                    candidates.append((
                        scores[i - 1][j - 2]
                        + COMPOUND_MATCH_SCORE,
                        1,
                        2,
                        "compound",
                        1
                    ))

            # Unmatched source token
            candidates.append((
                scores[i - 1][j] + GAP_PENALTY,
                1,
                0,
                None,
                2
            ))

            # Unmatched ASR word
            candidates.append((
                scores[i][j - 1] + GAP_PENALTY,
                0,
                1,
                None,
                3
            ))

            best = max(
                candidates,
                key=lambda item: (item[0], -item[4])
            )

            scores[i][j] = best[0]
            trace[i][j] = (
                best[1],
                best[2],
                best[3]
            )

    # Traceback
    i = n
    j = m

    matched_pairs = []
    unmatched_source = []
    unmatched_asr = []

    while i > 0 or j > 0:
        step = trace[i][j]

        if step is None:
            break

        source_step, asr_step, quality = step

        if source_step > 0 and asr_step > 0:

            source_group = source_tokens[
                i - source_step:i
            ]

            asr_group = asr_words[
                j - asr_step:j
            ]

            if quality is not None:
                matched_pairs.append({
                    "source_tokens": source_group,
                    "asr_words": asr_group,
                    "match_type": quality,
                })
            else:
                unmatched_source.extend(source_group)
                unmatched_asr.extend(asr_group)

        elif source_step > 0:
            unmatched_source.extend(
                source_tokens[i - source_step:i]
            )

        elif asr_step > 0:
            unmatched_asr.extend(
                asr_words[j - asr_step:j]
            )

        i -= source_step
        j -= asr_step

    matched_pairs.reverse()
    unmatched_source.reverse()
    unmatched_asr.reverse()

    return {
        "matched_pairs": matched_pairs,
        "unmatched_source": unmatched_source,
        "unmatched_asr": unmatched_asr,
        "final_score": scores[n][m],
        "source_token_count": n,
        "asr_word_count": m,
    }


# ============================================================
# ANALYSIS WINDOW
# ============================================================

def get_analysis_window(timeline):
    subtitles = timeline.get("segments", [])

    if not subtitles:
        raise ValueError("Timeline contains no subtitle segments.")

    starts = [
        float(item["start_seconds"])
        for item in subtitles
    ]

    ends = [
        float(item["end_seconds"])
        for item in subtitles
    ]

    return {
        "start_seconds": min(starts),
        "end_seconds": max(ends),
    }


# ============================================================
# ASR WORD CLASSIFICATION
# ============================================================

def classify_unmatched_asr_words(
    unmatched_asr,
    analysis_window
):
    """
    Separates:
        - words outside the selected analysis window
        - low-confidence unmatched words
        - unmatched words inside the analysis window
    """

    window_start = analysis_window["start_seconds"]
    window_end = analysis_window["end_seconds"]

    classified = []

    for word in unmatched_asr:
        start = word["start_seconds"]
        confidence = word.get("confidence")

        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = None

        if (
            start < window_start
            or start > window_end + WINDOW_END_TOLERANCE
        ):
            category = "OUTSIDE_ANALYSIS_WINDOW"

        elif (
            confidence is not None
            and confidence < 0.50
        ):
            category = "LOW_CONFIDENCE_UNMATCHED"

        else:
            category = "UNMATCHED_INSIDE_ANALYSIS_WINDOW"

        classified.append({
            **word,
            "classification": category,
        })

    return classified


# ============================================================
# MISSING DIALOGUE DETECTION
# ============================================================

def build_missing_dialogue_candidates(
    classified_words,
    analysis_window
):
    """
    Groups unmatched ASR words into possible missing dialogue.

    This does not automatically modify the SRT.

    High-confidence groups become review candidates.
    Single unmatched words are treated as possible terms/names.
    """

    eligible = [
        word
        for word in classified_words
        if word["classification"]
        == "UNMATCHED_INSIDE_ANALYSIS_WINDOW"
    ]

    eligible.sort(
        key=lambda item: item["start_seconds"]
    )

    groups = []
    current_group = []

    for word in eligible:

        if not current_group:
            current_group = [word]
            continue

        previous = current_group[-1]

        gap = (
            word["start_seconds"]
            - previous["end_seconds"]
        )

        previous_speaker = previous.get("speaker_id")
        current_speaker = word.get("speaker_id")

        speaker_compatible = (
            previous_speaker is None
            or current_speaker is None
            or previous_speaker == current_speaker
        )

        if (
            gap <= DIALOGUE_GROUP_MAX_GAP
            and gap >= -0.10
            and speaker_compatible
        ):
            current_group.append(word)
        else:
            groups.append(current_group)
            current_group = [word]

    if current_group:
        groups.append(current_group)

    candidates = []

    for index, group in enumerate(groups, start=1):

        confidences = [
            float(word["confidence"])
            for word in group
            if word.get("confidence") is not None
        ]

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else None
        )

        text = " ".join(
            word["text"]
            for word in group
        )

        start = min(
            word["start_seconds"]
            for word in group
        )

        end = max(
            word["end_seconds"]
            for word in group
        )

        speaker_ids = sorted({
            word["speaker_id"]
            for word in group
            if word.get("speaker_id")
        })

        if len(group) >= 2:
            if (
                average_confidence is not None
                and average_confidence
                >= DIALOGUE_MIN_AVERAGE_CONFIDENCE
            ):
                candidate_type = "POTENTIAL_MISSING_DIALOGUE"
            else:
                candidate_type = "LOW_CONFIDENCE_DIALOGUE_CANDIDATE"

        else:
            if (
                average_confidence is not None
                and average_confidence
                >= SINGLE_WORD_TERM_CONFIDENCE
            ):
                candidate_type = "POTENTIAL_TERM_OR_NAME_MISMATCH"
            else:
                candidate_type = "SINGLE_UNMATCHED_WORD"

        candidates.append({
            "candidate_id": f"MD_{index:04d}",
            "candidate_type": candidate_type,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(end - start, 3),
            "text": text,
            "word_count": len(group),
            "speaker_ids": speaker_ids,
            "average_asr_confidence": (
                round(average_confidence, 4)
                if average_confidence is not None
                else None
            ),
            "words": [
                {
                    "text": word["text"],
                    "start_seconds": round(
                        word["start_seconds"], 3
                    ),
                    "end_seconds": round(
                        word["end_seconds"], 3
                    ),
                    "confidence": word.get("confidence"),
                    "speaker_id": word.get("speaker_id"),
                }
                for word in group
            ],
            "requires_human_review": True,
        })

    return candidates


# ============================================================
# TIMING COVERAGE
# ============================================================

def calculate_timing_coverage(
    words,
    subtitle_start,
    subtitle_end
):
    """
    Measures the proportion of matched ASR word duration
    that overlaps the subtitle interval.

    Pauses between words are not counted as speech.
    """

    total_word_duration = 0.0
    overlap_duration = 0.0

    for word in words:
        word_start = word["start_seconds"]
        word_end = word["end_seconds"]

        duration = max(
            0.0,
            word_end - word_start
        )

        total_word_duration += duration

        overlap_start = max(
            word_start,
            subtitle_start
        )

        overlap_end = min(
            word_end,
            subtitle_end
        )

        overlap = max(
            0.0,
            overlap_end - overlap_start
        )

        overlap_duration += overlap

    if total_word_duration <= 0:
        return {
            "coverage_ratio": None,
            "overlap_duration_seconds": 0.0,
            "outside_duration_seconds": 0.0,
        }

    coverage = overlap_duration / total_word_duration

    return {
        "coverage_ratio": round(coverage, 4),
        "overlap_duration_seconds": round(
            overlap_duration, 3
        ),
        "outside_duration_seconds": round(
            max(
                0.0,
                total_word_duration - overlap_duration
            ),
            3
        ),
    }


# ============================================================
# SUBTITLE-LEVEL ANALYSIS
# ============================================================

def analyze_subtitles(
    timeline,
    matched_pairs,
    unmatched_source
):
    subtitles = timeline.get("segments", [])

    matches_by_subtitle = {}
    missing_by_subtitle = {}

    for pair in matched_pairs:
        for source_token in pair["source_tokens"]:
            subtitle_id = source_token["subtitle_id"]

            matches_by_subtitle.setdefault(
                subtitle_id,
                []
            ).append({
                "pair": pair,
                "source_token": source_token,
            })

    for token in unmatched_source:
        subtitle_id = token["subtitle_id"]

        missing_by_subtitle.setdefault(
            subtitle_id,
            []
        ).append(token)

    results = []

    total_aligned = 0
    total_review = 0

    for subtitle in subtitles:
        subtitle_id = subtitle["subtitle_id"]

        source_text = subtitle.get("text", "")
        source_tokens = tokenize(source_text)

        entries = matches_by_subtitle.get(
            subtitle_id,
            []
        )

        missing = missing_by_subtitle.get(
            subtitle_id,
            []
        )

        matched_count = len(entries)
        expected_count = len(source_tokens)

        similarity = (
            matched_count / expected_count
            if expected_count
            else 1.0
        )

        # Deduplicate ASR words.
        word_map = {}

        for entry in entries:
            pair = entry["pair"]

            for word in pair["asr_words"]:
                word_map[word["word_index"]] = word

        matched_words = sorted(
            word_map.values(),
            key=lambda item: item["start_seconds"]
        )

        flags = []

        if matched_count == 0:
            flags.append("NO_MATCHED_WORDS")

        if similarity < LOW_TEXT_SIMILARITY_THRESHOLD:
            flags.append("LOW_TEXT_SIMILARITY")

        confidences = [
            float(word["confidence"])
            for word in matched_words
            if word.get("confidence") is not None
        ]

        average_confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else None
        )

        if (
            average_confidence is not None
            and average_confidence
            < LOW_ASR_CONFIDENCE_THRESHOLD
        ):
            flags.append("LOW_ASR_CONFIDENCE")

        speaker_ids = sorted({
            word["speaker_id"]
            for word in matched_words
            if word.get("speaker_id")
        })

        if len(speaker_ids) > 1:
            flags.append("MULTIPLE_SPEAKERS")

        start_seconds = float(subtitle["start_seconds"])
        end_seconds = float(subtitle["end_seconds"])

        speech_timing = None
        timing_deviation = None
        timing_coverage = None

        if matched_words:
            speech_start = min(
                word["start_seconds"]
                for word in matched_words
            )

            speech_end = max(
                word["end_seconds"]
                for word in matched_words
            )

            start_delta = speech_start - start_seconds
            end_delta = speech_end - end_seconds

            speech_timing = {
                "start_seconds": round(speech_start, 3),
                "end_seconds": round(speech_end, 3),
            }

            timing_deviation = {
                "start_delta_seconds": round(
                    start_delta, 3
                ),
                "end_delta_seconds": round(
                    end_delta, 3
                ),
            }

            timing_coverage = calculate_timing_coverage(
                matched_words,
                start_seconds,
                end_seconds
            )

            coverage = timing_coverage["coverage_ratio"]

            if coverage is not None:
                if coverage < TIMING_LOW_COVERAGE_THRESHOLD:
                    flags.append("TIMING_LOW_OVERLAP")

                elif coverage < TIMING_PARTIAL_COVERAGE_THRESHOLD:
                    flags.append("TIMING_PARTIAL_OVERLAP")

        recognized_text = " ".join(
            word["text"]
            for word in matched_words
        )

        missing_tokens = [
            token["original"]
            for token in missing
        ]

        match_types = {
            "exact": 0,
            "alias": 0,
            "fuzzy": 0,
            "compound": 0,
        }

        counted_groups = set()

        for entry in entries:
            pair = entry["pair"]
            pair_identity = id(pair)

            if pair_identity in counted_groups:
                continue

            counted_groups.add(pair_identity)

            match_types[pair["match_type"]] += 1

        status = (
            "needs_review"
            if flags
            else "aligned"
        )

        if status == "needs_review":
            total_review += 1
        else:
            total_aligned += 1

        results.append({
            "subtitle_id": subtitle_id,

            "subtitle_timing": {
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": round(
                    end_seconds - start_seconds,
                    3
                ),
            },

            "speech_timing": speech_timing,
            "timing_deviation": timing_deviation,
            "timing_coverage": timing_coverage,

            "source_text": source_text,
            "recognized_text": recognized_text,

            "text_alignment": {
                "expected_token_count": expected_count,
                "matched_token_count": matched_count,
                "similarity": round(similarity, 4),
                "missing_tokens": missing_tokens,
                "match_types": match_types,
            },

            "speaker": {
                "speaker_ids": speaker_ids,
                "speaker_count": len(speaker_ids),
            },

            "confidence": {
                "average_asr_confidence": (
                    round(average_confidence, 4)
                    if average_confidence is not None
                    else None
                ),
            },

            "words": [
                {
                    "text": word["text"],
                    "start_seconds": round(
                        word["start_seconds"], 3
                    ),
                    "end_seconds": round(
                        word["end_seconds"], 3
                    ),
                    "confidence": word.get("confidence"),
                    "speaker_id": word.get("speaker_id"),
                }
                for word in matched_words
            ],

            "flags": flags,
            "status": status,
        })

    return results, total_aligned, total_review


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(results):
    start_deltas = []
    end_deltas = []
    coverage_values = []

    for segment in results:
        deviation = segment.get("timing_deviation")

        if deviation:
            start_deltas.append(
                deviation["start_delta_seconds"]
            )
            end_deltas.append(
                deviation["end_delta_seconds"]
            )

        coverage = segment.get("timing_coverage")

        if (
            coverage
            and coverage["coverage_ratio"] is not None
        ):
            coverage_values.append(
                coverage["coverage_ratio"]
            )

    def average(values):
        return (
            round(sum(values) / len(values), 3)
            if values
            else None
        )

    timing_statistics = {
        "subtitles_with_speech_timing": len(start_deltas),

        "average_start_delta_seconds": average(start_deltas),
        "average_end_delta_seconds": average(end_deltas),

        "max_absolute_start_delta_seconds": (
            round(max(abs(x) for x in start_deltas), 3)
            if start_deltas
            else None
        ),

        "max_absolute_end_delta_seconds": (
            round(max(abs(x) for x in end_deltas), 3)
            if end_deltas
            else None
        ),

        "average_timing_coverage": average(coverage_values),

        "low_coverage_subtitles": sum(
            1 for value in coverage_values
            if value < TIMING_LOW_COVERAGE_THRESHOLD
        ),

        "partial_coverage_subtitles": sum(
            1 for value in coverage_values
            if (
                TIMING_LOW_COVERAGE_THRESHOLD
                <= value
                < TIMING_PARTIAL_COVERAGE_THRESHOLD
            )
        ),
    }

    flag_counts = Counter()

    for segment in results:
        flag_counts.update(segment["flags"])

    return timing_statistics, dict(flag_counts)


# ============================================================
# MAIN
# ============================================================

def main():
    print("ALIGNMENT ENGINE V4")
    print("-------------------")

    timeline = load_json(TIMELINE_FILE)
    transcription = load_json(TRANSCRIPTION_FILE)

    subtitles = timeline.get("segments", [])

    if not subtitles:
        raise ValueError("No subtitle segments found.")

    source_tokens = extract_subtitle_tokens(timeline)
    asr_words = extract_asr_words(transcription)

    analysis_window = get_analysis_window(timeline)

    print(f"Subtitles: {len(subtitles)}")
    print(f"SRT tokens: {len(source_tokens)}")
    print(f"ASR words: {len(asr_words)}")
    print(
        "Analysis window: "
        f"{analysis_window['start_seconds']:.3f} - "
        f"{analysis_window['end_seconds']:.3f} seconds"
    )
    print()

    # --------------------------------------------------------
    # GLOBAL ALIGNMENT
    # --------------------------------------------------------

    print("Running global compound-aware alignment...")

    alignment = global_sequence_alignment(
        source_tokens,
        asr_words
    )

    matched_pairs = alignment["matched_pairs"]
    unmatched_source = alignment["unmatched_source"]
    unmatched_asr = alignment["unmatched_asr"]

    print("Global alignment: OK")
    print(f"Matched groups: {len(matched_pairs)}")
    print(f"Unmatched SRT tokens: {len(unmatched_source)}")
    print(f"Unmatched ASR words: {len(unmatched_asr)}")
    print()

    # --------------------------------------------------------
    # SUBTITLE ANALYSIS
    # --------------------------------------------------------

    print("Analyzing subtitles, timing and speakers...")

    results, aligned_count, review_count = analyze_subtitles(
        timeline,
        matched_pairs,
        unmatched_source
    )

    # --------------------------------------------------------
    # UNMATCHED ASR CLASSIFICATION
    # --------------------------------------------------------

    print("Classifying unmatched ASR words...")

    classified_words = classify_unmatched_asr_words(
        unmatched_asr,
        analysis_window
    )

    classification_counts = dict(
        Counter(
            word["classification"]
            for word in classified_words
        )
    )

    # --------------------------------------------------------
    # MISSING DIALOGUE CANDIDATES
    # --------------------------------------------------------

    print("Detecting potential missing dialogue...")

    dialogue_candidates = build_missing_dialogue_candidates(
        classified_words,
        analysis_window
    )

    candidate_counts = dict(
        Counter(
            candidate["candidate_type"]
            for candidate in dialogue_candidates
        )
    )

    # --------------------------------------------------------
    # MATCH TYPE COUNTS
    # --------------------------------------------------------

    match_type_counts = {
        "exact": 0,
        "alias": 0,
        "fuzzy": 0,
        "compound": 0,
    }

    for pair in matched_pairs:
        match_type_counts[pair["match_type"]] += 1

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    timing_statistics, flag_counts = calculate_statistics(
        results
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = {
        "schema_version": "4.0.0",
        "engine_version": "4.0.0",

        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),

        "source": {
            "timeline_file": str(TIMELINE_FILE),
            "transcription_file": str(TRANSCRIPTION_FILE),
        },

        "analysis_window": analysis_window,

        "alignment_method": {
            "type": "global_monotonic_compound_aware_alignment",
            "uses_timestamps_for_text_matching": False,
            "preserves_asr_word_timestamps": True,
            "supports_one_to_one": True,
            "supports_two_to_one": True,
            "supports_one_to_two": True,
            "compound_matching": "exact_concatenated_tokens",
            "compound_matching_crosses_subtitle_boundary": False,
            "fuzzy_match_threshold": FUZZY_MATCH_THRESHOLD,
        },

        "timing_method": {
            "type": "word_duration_overlap_coverage",
            "low_coverage_threshold": TIMING_LOW_COVERAGE_THRESHOLD,
            "partial_coverage_threshold": TIMING_PARTIAL_COVERAGE_THRESHOLD,
        },

        "missing_dialogue_detection": {
            "enabled": True,
            "method": "group_unmatched_asr_words_by_time_and_speaker",
            "maximum_word_gap_seconds": DIALOGUE_GROUP_MAX_GAP,
            "minimum_average_confidence": DIALOGUE_MIN_AVERAGE_CONFIDENCE,
            "automatic_srt_modification": False,
            "requires_human_review": True,
        },

        "statistics": {
            "subtitle_count": len(subtitles),
            "source_token_count": len(source_tokens),
            "asr_word_count": len(asr_words),

            "matched_group_count": len(matched_pairs),
            "match_type_counts": match_type_counts,

            "unmatched_srt_token_count": len(unmatched_source),
            "unmatched_asr_word_count": len(unmatched_asr),

            "aligned_subtitle_count": aligned_count,
            "needs_review_subtitle_count": review_count,

            "flag_counts": flag_counts,

            "asr_word_classification_counts": classification_counts,

            "missing_dialogue_candidate_count": len(
                dialogue_candidates
            ),

            "missing_dialogue_candidate_types": candidate_counts,

            "timing": timing_statistics,
        },

        "global_alignment": {
            "final_score": round(
                alignment["final_score"], 3
            ),
            "source_token_count": alignment["source_token_count"],
            "asr_word_count": alignment["asr_word_count"],
        },

        "segments": results,

        "unmatched_srt_tokens": [
            {
                "subtitle_id": token["subtitle_id"],
                "token": token["original"],
            }
            for token in unmatched_source
        ],

        "unassigned_words": [
            {
                "text": word["text"],
                "start_seconds": round(
                    word["start_seconds"], 3
                ),
                "end_seconds": round(
                    word["end_seconds"], 3
                ),
                "confidence": word.get("confidence"),
                "speaker_id": word.get("speaker_id"),
                "segment_id": word.get("segment_id"),
                "reason": "NO_TEXT_MATCH",
            }
            for word in unmatched_asr
        ],

        "asr_word_classifications": classified_words,

        "missing_dialogue_candidates": dialogue_candidates,
    }

    # --------------------------------------------------------
    # BACKUP V3 OUTPUT
    # --------------------------------------------------------

    if V3_FILE.exists():
        print()
        print(f"Existing V3 output preserved: {V3_FILE}")

    save_json(OUTPUT_FILE, output)

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("ALIGNMENT V4: OK")
    print("----------------")
    print(f"Matched groups: {len(matched_pairs)}")
    print(f"Match types: {match_type_counts}")
    print(f"Unmatched SRT tokens: {len(unmatched_source)}")
    print(f"Unmatched ASR words: {len(unmatched_asr)}")
    print(f"Aligned subtitles: {aligned_count}")
    print(f"Needs review: {review_count}")
    print()

    print("ASR word classifications:")
    print(json.dumps(
        classification_counts,
        ensure_ascii=False,
        indent=2
    ))

    print()
    print("Missing dialogue candidates:")
    print(json.dumps(
        candidate_counts,
        ensure_ascii=False,
        indent=2
    ))

    print()
    print("Flag counts:")
    print(json.dumps(
        flag_counts,
        ensure_ascii=False,
        indent=2
    ))

    print()
    print("Timing statistics:")
    print(json.dumps(
        timing_statistics,
        ensure_ascii=False,
        indent=2
    ))

    print()
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
