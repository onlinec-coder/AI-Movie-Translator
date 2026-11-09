
import json
from pathlib import Path
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

V4_FILE = (
    PROJECT_ROOT
    / "data" / "output"
    / "alignment_result_v4.json"
)

REVIEW_FILE = (
    PROJECT_ROOT
    / "data" / "output"
    / "review_clips"
    / "review_decisions.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data" / "output"
    / "alignment_result_v5.json"
)


# ============================================================
# SETTINGS
# ============================================================

TIMING_PADDING_SECONDS = 0.08

CONFIRMED_DECISIONS = {
    "CONFIRMED_MISSING_DIALOGUE",
}

KNOWN_DECISIONS = {
    "CONFIRMED_MISSING_DIALOGUE",
    "EXISTING_SUBTITLE_CONTINUATION",
    "REJECTED_ASR",
    "UNCONFIRMED",
}


# ============================================================
# FILE HELPERS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# VALIDATION
# ============================================================

def validate_inputs(v4, review):
    if "missing_dialogue_candidates" not in v4:
        raise ValueError(
            "V4 missing_dialogue_candidates not found."
        )

    if "decisions" not in review:
        raise ValueError(
            "Review decisions list not found."
        )

    if not isinstance(review["decisions"], list):
        raise ValueError(
            "Review decisions must be a list."
        )

    candidate_ids = {
        item["candidate_id"]
        for item in v4["missing_dialogue_candidates"]
    }

    decision_ids = []

    for decision in review["decisions"]:
        candidate_id = decision.get("candidate_id")

        if not candidate_id:
            raise ValueError(
                "A review decision has no candidate_id."
            )

        if candidate_id not in candidate_ids:
            raise ValueError(
                f"Candidate not found in V4: {candidate_id}"
            )

        decision_type = decision.get("decision")

        if decision_type not in KNOWN_DECISIONS:
            raise ValueError(
                f"Unknown decision type: {decision_type}"
            )

        decision_ids.append(candidate_id)

    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError(
            "Duplicate candidate decisions found."
        )


# ============================================================
# INDEXES
# ============================================================

def build_candidate_index(v4):
    return {
        item["candidate_id"]: item
        for item in v4["missing_dialogue_candidates"]
    }


def build_subtitle_index(v4):
    return {
        item["subtitle_id"]: item
        for item in v4["segments"]
    }


# ============================================================
# TIMING HELPERS
# ============================================================

def calculate_proposed_timing(candidate):
    start = float(candidate["start_seconds"])
    end = float(candidate["end_seconds"])

    return {
        "start_seconds": round(
            max(0.0, start - TIMING_PADDING_SECONDS),
            3
        ),
        "end_seconds": round(
            end + TIMING_PADDING_SECONDS,
            3
        ),
        "source": "ASR_WORD_TIMESTAMPS_WITH_PADDING",
        "padding_seconds": TIMING_PADDING_SECONDS,
    }


def calculate_overlap(start_a, end_a, start_b, end_b):
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)

    return round(
        max(0.0, overlap_end - overlap_start),
        3
    )


def analyze_related_subtitles(
    related_ids,
    subtitle_index,
    proposed_timing
):
    results = []

    new_start = proposed_timing["start_seconds"]
    new_end = proposed_timing["end_seconds"]

    for subtitle_id in related_ids:
        subtitle = subtitle_index.get(subtitle_id)

        if subtitle is None:
            results.append({
                "subtitle_id": subtitle_id,
                "found": False,
            })
            continue

        old_start = subtitle["subtitle_timing"]["start_seconds"]
        old_end = subtitle["subtitle_timing"]["end_seconds"]

        overlap = calculate_overlap(
            new_start,
            new_end,
            old_start,
            old_end
        )

        results.append({
            "subtitle_id": subtitle_id,
            "found": True,
            "source_text": subtitle["source_text"],
            "subtitle_timing": {
                "start_seconds": old_start,
                "end_seconds": old_end,
            },
            "overlap_with_proposed_timing_seconds": overlap,
            "requires_timing_review": overlap > 0,
        })

    return results


# ============================================================
# DECISION PROCESSING
# ============================================================

def process_decisions(
    v4,
    review,
    candidate_index,
    subtitle_index
):
    processed = []

    for decision in review["decisions"]:

        candidate_id = decision["candidate_id"]
        decision_type = decision["decision"]

        candidate = candidate_index[candidate_id]

        related_ids = decision.get(
            "related_subtitle_ids",
            []
        )

        proposed_timing = calculate_proposed_timing(
            candidate
        )

        related_subtitles = analyze_related_subtitles(
            related_ids,
            subtitle_index,
            proposed_timing
        )

        if decision_type == "CONFIRMED_MISSING_DIALOGUE":

            action = "PROPOSE_NEW_SUBTITLE"

            review_status = "HUMAN_CONFIRMED"

        elif decision_type == "EXISTING_SUBTITLE_CONTINUATION":

            action = "KEEP_EXISTING_SUBTITLE"

            review_status = "HUMAN_CONFIRMED"

        elif decision_type == "REJECTED_ASR":

            action = "IGNORE_CANDIDATE"

            review_status = "HUMAN_REJECTED"

        else:

            action = "WAIT_FOR_HUMAN_REVIEW"

            review_status = "UNCONFIRMED"

        timing_conflicts = [
            item
            for item in related_subtitles
            if (
                item.get("found")
                and item.get("requires_timing_review")
            )
        ]

        if action == "PROPOSE_NEW_SUBTITLE":

            if timing_conflicts:
                editing_recommendation = (
                    "REVIEW_EXISTING_SUBTITLE_SPLIT_OR_OVERLAP"
                )
            else:
                editing_recommendation = (
                    "NEW_SUBTITLE_CUE_CAN_BE_CONSIDERED"
                )

        elif action == "KEEP_EXISTING_SUBTITLE":

            editing_recommendation = (
                "NO_NEW_SUBTITLE_REQUIRED"
            )

        elif action == "IGNORE_CANDIDATE":

            editing_recommendation = (
                "NO_SUBTITLE_CHANGE"
            )

        else:

            editing_recommendation = (
                "MANUAL_REVIEW_REQUIRED"
            )

        processed.append({
            "candidate_id": candidate_id,

            "decision": decision_type,
            "review_status": review_status,

            "candidate_type": candidate["candidate_type"],

            "text": decision.get(
                "text",
                candidate["text"]
            ),

            "audio_verified": decision.get(
                "audio_verified",
                False
            ),

            "speaker_ids": candidate.get(
                "speaker_ids",
                []
            ),

            "candidate_timing": {
                "start_seconds": candidate["start_seconds"],
                "end_seconds": candidate["end_seconds"],
            },

            "proposed_subtitle_timing": proposed_timing,

            "related_subtitle_ids": related_ids,

            "related_subtitles": related_subtitles,

            "timing_conflict_count": len(timing_conflicts),

            "action": action,

            "editing_recommendation": editing_recommendation,

            "notes": decision.get("notes", ""),

            "automatic_srt_modification": False,
        })

    return processed


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(processed):
    counts = {
        "confirmed_missing_dialogue": 0,
        "existing_subtitle_continuation": 0,
        "rejected_asr": 0,
        "unconfirmed": 0,
        "timing_conflicts": 0,
    }

    for item in processed:

        decision = item["decision"]

        if decision == "CONFIRMED_MISSING_DIALOGUE":
            counts["confirmed_missing_dialogue"] += 1

        elif decision == "EXISTING_SUBTITLE_CONTINUATION":
            counts["existing_subtitle_continuation"] += 1

        elif decision == "REJECTED_ASR":
            counts["rejected_asr"] += 1

        elif decision == "UNCONFIRMED":
            counts["unconfirmed"] += 1

        counts["timing_conflicts"] += (
            item["timing_conflict_count"]
        )

    return counts


# ============================================================
# MAIN
# ============================================================

def main():

    print("ALIGNMENT ENGINE V5")
    print("-------------------")

    print("Loading V4 alignment...")
    v4 = load_json(V4_FILE)

    print("Loading human review decisions...")
    review = load_json(REVIEW_FILE)

    print("Validating inputs...")
    validate_inputs(v4, review)

    candidate_index = build_candidate_index(v4)
    subtitle_index = build_subtitle_index(v4)

    print("Processing human decisions...")

    processed = process_decisions(
        v4,
        review,
        candidate_index,
        subtitle_index
    )

    statistics = calculate_statistics(
        processed
    )

    output = {
        "schema_version": "5.0.0",
        "engine_version": "5.0.0",

        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),

        "source": {
            "alignment_file": str(V4_FILE),
            "human_review_file": str(REVIEW_FILE),
        },

        "method": {
            "type": "human_verified_missing_dialogue_review",
            "uses_v4_candidates": True,
            "uses_human_review_decisions": True,
            "uses_asr_word_timestamps": True,
            "automatic_srt_modification": False,
        },

        "timing_settings": {
            "padding_seconds": TIMING_PADDING_SECONDS,
            "timing_source": "ASR_WORD_TIMESTAMPS",
        },

        "statistics": statistics,

        "reviewed_candidates": processed,
    }

    save_json(
        OUTPUT_FILE,
        output
    )

    print()
    print("ALIGNMENT V5: OK")
    print("----------------")

    print(json.dumps(
        statistics,
        ensure_ascii=False,
        indent=2
    ))

    print()
    print("Reviewed candidates:")

    for item in processed:
        print(
            f"{item['candidate_id']} | "
            f"{item['decision']} | "
            f"{item['editing_recommendation']}"
        )

    print()
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
