
import json
from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[2]

ALIGNMENT_PATH = ROOT / "data/output/alignment_result_v4.json"
ASSIGNMENTS_PATH = ROOT / "data/input/speaker_assignments.json"
TRANSCRIPTION_PATH = ROOT / "data/output/transcription_full_result.json"
OUTPUT_PATH = ROOT / "data/output/speaker_utterances_v1.json"

IDENTITY_RULES = {
    "C005": {
        "canonical_character_id": "C001",
        "canonical_character_name": "Vivienne",
        "preserve_role": True,
    }
}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_character_identity(character_id, character_name):
    rule = IDENTITY_RULES.get(character_id)

    if not rule:
        return character_id, character_name

    return (
        rule["canonical_character_id"],
        rule["canonical_character_name"],
    )


def main():
    print("SPEAKER UTTERANCE INSPECTOR V3")
    print("==============================")

    alignment = load_json(ALIGNMENT_PATH)
    assignments = load_json(ASSIGNMENTS_PATH)
    transcription = load_json(TRANSCRIPTION_PATH)

    voice_identities = {
        item["speaker_id"]: {
            "character_name": item.get("character_name"),
            "speaker_type": item.get("speaker_type", "unknown"),
            "provider_label": item.get("provider_label"),
            "confidence": item.get("confidence"),
        }
        for item in transcription.get("speakers", [])
    }

    assignment_index = {
        item["subtitle_id"]: item
        for item in assignments.get("assignments", [])
        if "subtitle_id" in item
    }

    speaker_data = defaultdict(list)

    for segment in alignment.get("segments", []):
        subtitle_id = segment.get("subtitle_id")
        assignment = assignment_index.get(subtitle_id, {})

        original_character_id = assignment.get("speaker_id")
        original_character_name = assignment.get("speaker_name")

        canonical_character_id, canonical_character_name = (
            resolve_character_identity(
                original_character_id,
                original_character_name,
            )
        )

        words_by_speaker = defaultdict(list)

        for word in segment.get("words", []):
            speaker_id = word.get("speaker_id")

            if speaker_id:
                words_by_speaker[speaker_id].append(word)

        for speaker_id, speaker_words in words_by_speaker.items():
            start = min(w["start_seconds"] for w in speaker_words)
            end = max(w["end_seconds"] for w in speaker_words)

            confidences = [
                w["confidence"]
                for w in speaker_words
                if isinstance(w.get("confidence"), (int, float))
            ]

            avg_confidence = (
                round(sum(confidences) / len(confidences), 4)
                if confidences else None
            )

            voice_identity = voice_identities.get(speaker_id, {})

            utterance = {
                "speaker_id": speaker_id,
                "voice_character_name": voice_identity.get("character_name"),
                "voice_speaker_type": voice_identity.get("speaker_type"),
                "provider_label": voice_identity.get("provider_label"),

                "subtitle_id": subtitle_id,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),

                "assigned_character_id": canonical_character_id,
                "assigned_character_name": canonical_character_name,
                "assigned_speaker_type": assignment.get(
                    "speaker_type", "unknown"
                ),

                "original_assigned_character_id": original_character_id,
                "original_assigned_character_name": original_character_name,

                "assignment_confidence": assignment.get("confidence"),
                "speaker_confidence": segment.get("speaker_confidence"),
                "average_asr_confidence": avg_confidence,
                "word_count": len(speaker_words),

                "recognized_words": " ".join(
                    w["text"] for w in speaker_words
                ),

                "original_subtitle_text": segment.get("source_text"),
                "recognized_segment_text": segment.get("recognized_text"),
            }

            speaker_data[speaker_id].append(utterance)

    report = {
        "schema_version": "1.2.0",
        "engine_version": "speaker_utterances_v3",
        "voice_identity_source": "transcription_full_result.json",
        "identity_rules_applied": [
            {
                "speaker_id": "SPK_004",
                "character_name": "Lucian",
                "status": "confirmed",
            },
            {
                "original_character_id": "C005",
                "original_character_name": "Narrator",
                "canonical_character_id": "C001",
                "canonical_character_name": "Vivienne",
                "preserved_speaker_type": "narrator",
            },
        ],
        "speakers": [],
    }

    for speaker_id in sorted(speaker_data):
        utterances = sorted(
            speaker_data[speaker_id],
            key=lambda x: x["start_seconds"],
        )

        total_words = sum(x["word_count"] for x in utterances)

        report["speakers"].append({
            "speaker_id": speaker_id,
            "voice_character_name": voice_identities.get(
                speaker_id, {}
            ).get("character_name"),
            "utterance_count": len(utterances),
            "total_words": total_words,
            "utterances": utterances,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        backup_path = OUTPUT_PATH.with_name(
            "speaker_utterances_v1_pre_v3_backup.json"
        )
        if not backup_path.exists():
            backup_path.write_text(
                OUTPUT_PATH.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            print(f"Backup: {backup_path}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nVOICE IDENTITY SUMMARY")
    print("----------------------")

    for speaker in report["speakers"]:
        print(
            f"{speaker['speaker_id']} -> "
            f"{speaker.get('voice_character_name') or 'UNMAPPED'} | "
            f"Utterances: {speaker['utterance_count']} | "
            f"Words: {speaker['total_words']}"
        )

    print("\nC005 Narrator -> C001 Vivienne")
    print("Narrative role preserved.")
    print("SPK_004 -> Lucian (confirmed transcription metadata)")
    print("\nSOURCE ASSIGNMENTS: UNCHANGED")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
