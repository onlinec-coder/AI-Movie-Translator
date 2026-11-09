
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]

ALIGNMENT_PATH = (
    ROOT / "data/output/alignment_result_v4.json"
)

TRANSCRIPTION_PATH = (
    ROOT / "data/output/transcription_result.json"
)

ASSIGNMENTS_PATH = (
    ROOT / "data/input/speaker_assignments.json"
)

CHARACTERS_PATH = (
    ROOT / "data/input/Husbands_characters_reconciled.json"
)

OUTPUT_PATH = (
    ROOT / "data/output/speaker_mapping_evidence_v1.json"
)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_assignment_index(data):
    index = {}

    for item in data.get("assignments", []):
        subtitle_id = item.get("subtitle_id")

        if subtitle_id is None:
            continue

        index[subtitle_id] = item

    return index


def build_character_index(data):
    index = {}

    for character in data.get("characters", []):
        character_id = character.get("id")

        if character_id:
            index[character_id] = character

    return index


def main():
    print("SPEAKER MAPPING EVIDENCE V1")
    print("===========================")

    alignment = load_json(ALIGNMENT_PATH)
    transcription = load_json(TRANSCRIPTION_PATH)
    assignments = load_json(ASSIGNMENTS_PATH)
    characters = load_json(CHARACTERS_PATH)

    assignment_index = build_assignment_index(assignments)
    character_index = build_character_index(characters)

    # Each speaker's evidence by character.
    evidence = defaultdict(
        lambda: defaultdict(
            lambda: {
                "subtitle_ids": set(),
                "word_count": 0,
                "matched_word_count": 0,
                "speaker_confidences": [],
                "asr_confidences": [],
                "speaker_types": Counter(),
                "character_names": Counter(),
            }
        )
    )

    unmapped_speaker_words = Counter()
    missing_assignment_subtitles = []
    processed_segments = 0

    for segment in alignment.get("segments", []):
        subtitle_id = segment.get("subtitle_id")

        assignment = assignment_index.get(subtitle_id)

        if assignment is None:
            missing_assignment_subtitles.append(subtitle_id)
            continue

        character_id = assignment.get("speaker_id")

        if not character_id:
            continue

        character = character_index.get(character_id, {})

        character_name = (
            character.get("canonical_name")
            or assignment.get("speaker_name")
            or "UNKNOWN"
        )

        subtitle_speaker_type = (
            assignment.get("speaker_type", "unknown")
        )

        segment_speakers = (
            segment.get("speaker", {}).get("speaker_ids", [])
        )

        words = segment.get("words", [])

        # Count only words that actually have a speaker ID.
        segment_word_counts = Counter()

        for word in words:
            speaker_id = word.get("speaker_id")

            if not speaker_id:
                continue

            segment_word_counts[speaker_id] += 1

            item = evidence[speaker_id][character_id]

            item["subtitle_ids"].add(subtitle_id)
            item["word_count"] += 1

            confidence = word.get("confidence")

            if isinstance(confidence, (int, float)):
                item["asr_confidences"].append(confidence)

        for speaker_id in segment_speakers:
            item = evidence[speaker_id][character_id]

            item["speaker_types"][subtitle_speaker_type] += 1
            item["character_names"][character_name] += 1

            speaker_confidence = segment.get(
                "speaker_confidence"
            )

            if isinstance(speaker_confidence, (int, float)):
                item["speaker_confidences"].append(
                    speaker_confidence
                )

        # Count words assigned to a speaker in this subtitle.
        for speaker_id, count in segment_word_counts.items():
            evidence[speaker_id][character_id][
                "matched_word_count"
            ] += count

        processed_segments += 1

    # Include unassigned ASR words separately.
    for word in alignment.get("unassigned_words", []):
        speaker_id = word.get("speaker_id")

        if speaker_id:
            unmapped_speaker_words[speaker_id] += 1

    # Build output report.
    speaker_reports = []

    all_speaker_ids = sorted(
        set(evidence.keys()) | set(unmapped_speaker_words.keys())
    )

    for speaker_id in all_speaker_ids:
        character_reports = []

        total_words = sum(
            item["word_count"]
            for item in evidence[speaker_id].values()
        )

        total_subtitle_links = sum(
            len(item["subtitle_ids"])
            for item in evidence[speaker_id].values()
        )

        for character_id, item in evidence[speaker_id].items():
            character = character_index.get(character_id, {})

            confidences = item["speaker_confidences"]
            asr_confidences = item["asr_confidences"]

            average_speaker_confidence = (
                round(sum(confidences) / len(confidences), 4)
                if confidences else None
            )

            average_asr_confidence = (
                round(sum(asr_confidences) / len(asr_confidences), 4)
                if asr_confidences else None
            )

            character_reports.append({
                "character_id": character_id,
                "character_name": (
                    character.get("canonical_name", "UNKNOWN")
                ),
                "subtitle_count": len(item["subtitle_ids"]),
                "subtitle_ids": sorted(item["subtitle_ids"]),
                "word_count": item["word_count"],
                "matched_word_count": item["matched_word_count"],
                "word_share": (
                    round(item["word_count"] / total_words, 4)
                    if total_words else 0
                ),
                "average_speaker_confidence": (
                    average_speaker_confidence
                ),
                "average_asr_confidence": average_asr_confidence,
                "speaker_types": dict(item["speaker_types"]),
            })

        character_reports.sort(
            key=lambda x: (
                x["word_count"],
                x["subtitle_count"]
            ),
            reverse=True
        )

        top_match = (
            character_reports[0]
            if character_reports else None
        )

        speaker_reports.append({
            "speaker_id": speaker_id,
            "total_matched_words": total_words,
            "total_subtitle_links": total_subtitle_links,
            "unassigned_asr_words": unmapped_speaker_words.get(
                speaker_id, 0
            ),
            "candidate_matches": character_reports,
            "top_candidate": top_match,
            "mapping_status": "REQUIRES_REVIEW",
        })

    # Check the speaker inventory from AssemblyAI.
    assembly_speakers = transcription.get("speakers", [])

    report = {
        "schema_version": "1.0.0",
        "engine_version": "speaker_mapping_evidence_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "alignment_file": str(
                ALIGNMENT_PATH.relative_to(ROOT)
            ),
            "transcription_file": str(
                TRANSCRIPTION_PATH.relative_to(ROOT)
            ),
            "assignments_file": str(
                ASSIGNMENTS_PATH.relative_to(ROOT)
            ),
            "characters_file": str(
                CHARACTERS_PATH.relative_to(ROOT)
            ),
        },
        "statistics": {
            "processed_alignment_segments": processed_segments,
            "assemblyai_speaker_count": len(assembly_speakers),
            "speaker_count_with_evidence": len(speaker_reports),
            "character_count": len(character_index),
            "missing_assignment_subtitles": len(
                missing_assignment_subtitles
            ),
        },
        "assemblyai_speakers": assembly_speakers,
        "speaker_evidence": speaker_reports,
        "missing_assignment_subtitles": missing_assignment_subtitles,
        "automatic_mapping_applied": False,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nSPEAKER EVIDENCE SUMMARY")
    print("------------------------")

    for speaker in speaker_reports:
        print(f"\n{speaker['speaker_id']}")
        print(
            f"Matched words: "
            f"{speaker['total_matched_words']}"
        )

        print(
            f"Unassigned ASR words: "
            f"{speaker['unassigned_asr_words']}"
        )

        for candidate in speaker["candidate_matches"][:5]:
            print(
                f"  {candidate['character_id']} | "
                f"{candidate['character_name']} | "
                f"words={candidate['word_count']} | "
                f"subtitles={candidate['subtitle_count']} | "
                f"share={candidate['word_share']}"
            )

    print("\nSTATISTICS")
    print("----------")
    print(f"Processed segments: {processed_segments}")
    print(f"Speakers: {len(speaker_reports)}")
    print(f"Characters: {len(character_index)}")
    print(
        "Missing assignments:",
        len(missing_assignment_subtitles)
    )

    print("\nAUTOMATIC MAPPING: DISABLED")
    print("EVIDENCE REPORT: OK")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
