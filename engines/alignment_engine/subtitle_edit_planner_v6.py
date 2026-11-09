
import json
import re
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[2]

V5_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "alignment_result_v5.json"
)

TRANSCRIPTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "transcription_result.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "subtitle_edit_plan_v6.json"
)

PADDING_SECONDS = 0.08


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_word(text):
    return re.sub(r"[^\w']", "", text.lower())


def flatten_words(transcription):
    words = []

    for segment in transcription.get("segments", []):
        for word in segment.get("words", []):
            words.append({
                "text": word["text"],
                "normalized": normalize_word(word["text"]),
                "start_seconds": word["start_seconds"],
                "end_seconds": word["end_seconds"],
                "speaker_id": word.get("speaker_id"),
                "confidence": word.get("confidence")
            })

    return sorted(words, key=lambda item: item["start_seconds"])


def find_phrase_timing(text, words):
    target_words = [
        normalize_word(token)
        for token in text.split()
    ]

    target_words = [word for word in target_words if word]

    if not target_words:
        raise ValueError(f"Cannot tokenize subtitle text: {text}")

    for index in range(len(words) - len(target_words) + 1):
        current = words[index:index + len(target_words)]

        current_words = [
            word["normalized"]
            for word in current
        ]

        if current_words == target_words:
            return {
                "start_seconds": round(
                    max(0.0, current[0]["start_seconds"] - PADDING_SECONDS),
                    3
                ),
                "end_seconds": round(
                    current[-1]["end_seconds"] + PADDING_SECONDS,
                    3
                ),
                "source": "EXACT_ASR_WORD_SEQUENCE",
                "padding_seconds": PADDING_SECONDS
            }

    raise ValueError(
        f"Could not find exact ASR word sequence for: {text}"
    )


def build_edit_plan(v5, transcription):
    words = flatten_words(transcription)

    edits = []
    errors = []

    candidates = v5.get("reviewed_candidates", [])

    for candidate in candidates:
        decision = candidate.get("decision")

        if decision != "CONFIRMED_MISSING_DIALOGUE":
            continue

        candidate_id = candidate["candidate_id"]
        related_subtitles = candidate.get("related_subtitles", [])

        if len(related_subtitles) != 1:
            errors.append({
                "candidate_id": candidate_id,
                "error": "Expected exactly one related subtitle."
            })
            continue

        related = related_subtitles[0]

        if not related.get("found"):
            errors.append({
                "candidate_id": candidate_id,
                "error": "Related subtitle was not found."
            })
            continue

        subtitle_id = related["subtitle_id"]
        original_text = related["source_text"]

        # Find the existing subtitle's actual spoken timing.
        try:
            existing_timing = find_phrase_timing(
                original_text,
                words
            )
        except ValueError as error:
            errors.append({
                "candidate_id": candidate_id,
                "subtitle_id": subtitle_id,
                "error": str(error)
            })
            continue

        # Use the human-confirmed missing dialogue.
        new_text = candidate["text"]
        new_timing = candidate["proposed_subtitle_timing"]

        if new_timing["start_seconds"] >= new_timing["end_seconds"]:
            errors.append({
                "candidate_id": candidate_id,
                "error": "Invalid proposed subtitle timing."
            })
            continue

        # New subtitle must appear before the existing related subtitle.
        edits.append({
            "edit_id": f"EDIT_{candidate_id}_NEW",
            "operation": "ADD_SUBTITLE",
            "candidate_id": candidate_id,
            "new_subtitle_id": f"NEW_{candidate_id}",
            "insert_before_subtitle_id": subtitle_id,
            "text": new_text,
            "speaker_id": (
                candidate.get("speaker_ids", [None])[0]
            ),
            "timing": new_timing,
            "source": "HUMAN_CONFIRMED_ASR",
            "automatic_srt_modification": False
        })

        # Adjust the original subtitle to its actual speech timing.
        edits.append({
            "edit_id": f"EDIT_{candidate_id}_UPDATE",
            "operation": "UPDATE_SUBTITLE_TIMING",
            "candidate_id": candidate_id,
            "subtitle_id": subtitle_id,
            "text": original_text,
            "original_timing": related["subtitle_timing"],
            "proposed_timing": existing_timing,
            "source": "EXACT_ASR_WORD_SEQUENCE",
            "automatic_text_modification": False,
            "automatic_srt_modification": False
        })

    return edits, errors


def main():
    print("SUBTITLE EDIT PLANNER V6")
    print("------------------------")

    print("Loading V5 alignment...")
    v5 = load_json(V5_FILE)

    print("Loading transcription...")
    transcription = load_json(TRANSCRIPTION_FILE)

    print("Building subtitle edit plan...")

    edits, errors = build_edit_plan(v5, transcription)

    add_count = sum(
        1 for edit in edits
        if edit["operation"] == "ADD_SUBTITLE"
    )

    update_count = sum(
        1 for edit in edits
        if edit["operation"] == "UPDATE_SUBTITLE_TIMING"
    )

    result = {
        "schema_version": "1.0.0",
        "engine_version": "6.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "v5_alignment": str(V5_FILE),
            "transcription": str(TRANSCRIPTION_FILE)
        },
        "method": {
            "new_subtitle_timing": "V5_HUMAN_CONFIRMED_ASR_TIMING",
            "existing_subtitle_timing": "EXACT_ASR_WORD_SEQUENCE",
            "padding_seconds": PADDING_SECONDS,
            "original_srt_modified": False
        },
        "statistics": {
            "new_subtitles": add_count,
            "updated_subtitle_timings": update_count,
            "total_edits": len(edits),
            "errors": len(errors),
            "ready_for_srt_generation": len(errors) == 0
        },
        "edits": edits,
        "errors": errors
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nSUBTITLE EDIT PLAN V6: OK")
    print("-------------------------")
    print(json.dumps(
        result["statistics"],
        ensure_ascii=False,
        indent=2
    ))

    for edit in edits:
        if edit["operation"] == "ADD_SUBTITLE":
            timing = edit["timing"]
            print(
                f"\nADD | {edit['new_subtitle_id']}"
                f" | {timing['start_seconds']:.3f}"
                f" - {timing['end_seconds']:.3f}"
                f" | {edit['text']}"
            )

        elif edit["operation"] == "UPDATE_SUBTITLE_TIMING":
            timing = edit["proposed_timing"]
            print(
                f"UPDATE | Subtitle {edit['subtitle_id']}"
                f" | {timing['start_seconds']:.3f}"
                f" - {timing['end_seconds']:.3f}"
                f" | {edit['text']}"
            )

    if errors:
        print("\nERROR DETAILS:")
        print(json.dumps(errors, ensure_ascii=False, indent=2))

    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
