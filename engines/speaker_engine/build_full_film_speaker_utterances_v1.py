
import json
from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[2]

TRANSCRIPTION_PATH = ROOT / "data/output/transcription_full_result.json"
OUTPUT_PATH = ROOT / "data/output/full_film_speaker_utterances_v1.json"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("FULL FILM SPEAKER UTTERANCE BUILDER V1")
    print("======================================")

    transcription = load_json(TRANSCRIPTION_PATH)

    speaker_metadata = {
        item["speaker_id"]: item
        for item in transcription.get("speakers", [])
    }

    utterances_by_speaker = defaultdict(list)

    for segment in transcription.get("segments", []):
        words_by_speaker = defaultdict(list)

        for word in segment.get("words", []):
            speaker_id = word.get("speaker_id") or segment.get("speaker_id")

            if speaker_id:
                words_by_speaker[speaker_id].append(word)

        for speaker_id, words in words_by_speaker.items():
            valid_words = [
                word for word in words
                if isinstance(word.get("start_seconds"), (int, float))
                and isinstance(word.get("end_seconds"), (int, float))
            ]

            if not valid_words:
                continue

            start = min(w["start_seconds"] for w in valid_words)
            end = max(w["end_seconds"] for w in valid_words)

            confidences = [
                w["confidence"]
                for w in valid_words
                if isinstance(w.get("confidence"), (int, float))
            ]

            metadata = speaker_metadata.get(speaker_id, {})

            utterances_by_speaker[speaker_id].append({
                "segment_id": segment.get("segment_id"),
                "speaker_id": speaker_id,
                "character_name": metadata.get("character_name"),
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
                "word_count": len(valid_words),
                "average_word_confidence": (
                    round(sum(confidences) / len(confidences), 4)
                    if confidences else None
                ),
                "speaker_confidence": segment.get("speaker_confidence"),
                "text": " ".join(w.get("text", "") for w in valid_words),
            })

    report = {
        "schema_version": "1.0.0",
        "engine_version": "full_film_speaker_utterances_v1",
        "source": "transcription_full_result.json",
        "identity_policy": {
            "SPK_004": {
                "character_name": "Lucian",
                "status": "confirmed",
            }
        },
        "speakers": [],
    }

    for speaker_id in sorted(utterances_by_speaker):
        utterances = sorted(
            utterances_by_speaker[speaker_id],
            key=lambda item: item["start_seconds"],
        )

        metadata = speaker_metadata.get(speaker_id, {})

        report["speakers"].append({
            "speaker_id": speaker_id,
            "provider_label": metadata.get("provider_label"),
            "character_name": metadata.get("character_name"),
            "speaker_type": metadata.get("speaker_type", "unknown"),
            "provider_confidence": metadata.get("confidence"),
            "utterance_count": len(utterances),
            "total_words": sum(x["word_count"] for x in utterances),
            "utterances": utterances,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nFULL FILM VOICE SUMMARY")
    print("-----------------------")

    for speaker in report["speakers"]:
        print(
            f"{speaker['speaker_id']} -> "
            f"{speaker.get('character_name') or 'UNMAPPED'} | "
            f"Utterances: {speaker['utterance_count']} | "
            f"Words: {speaker['total_words']}"
        )

    print("\nSOURCE TRANSCRIPTION: UNCHANGED")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
