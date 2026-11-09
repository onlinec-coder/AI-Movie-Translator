
import json
import os
import wave
from pathlib import Path

from dotenv import load_dotenv

from provider_adapter import get_provider


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Full movie audio. The original 5-minute sample remains untouched.
AUDIO_FILE = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "audio"
    / "full_audio.wav"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

RAW_OUTPUT = OUTPUT_DIR / "transcription_full_raw.json"
NORMALIZED_OUTPUT = OUTPUT_DIR / "transcription_full_result.json"


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def get_audio_duration(audio_path):
    with wave.open(str(audio_path), "rb") as audio:
        frames = audio.getnframes()
        sample_rate = audio.getframerate()

        if sample_rate <= 0:
            raise RuntimeError("Invalid audio sample rate.")

        return round(frames / sample_rate, 3)


def milliseconds_to_seconds(value):
    if value is None:
        return None

    return round(float(value) / 1000, 3)


def normalize_transcript(raw, audio_duration, audio_filename):

    speaker_map = {}

    utterances = raw.get("utterances") or []

    for utterance in utterances:
        label = utterance.get("speaker")

        if label and label not in speaker_map:
            speaker_map[label] = (
                f"SPK_{len(speaker_map) + 1:03d}"
            )

    segments = []

    speaker_confidences = {
        label: []
        for label in speaker_map
    }

    total_words = 0

    for index, utterance in enumerate(utterances, start=1):

        provider_label = utterance.get("speaker")
        speaker_id = speaker_map.get(provider_label)

        words = []

        for word in utterance.get("words") or []:

            word_label = word.get(
                "speaker",
                provider_label
            )

            word_speaker_id = speaker_map.get(word_label)

            word_data = {
                "text": word.get("text", ""),
                "start_seconds": milliseconds_to_seconds(
                    word.get("start")
                ),
                "end_seconds": milliseconds_to_seconds(
                    word.get("end")
                ),
                "confidence": word.get("confidence"),
                "speaker_id": word_speaker_id
            }

            words.append(word_data)
            total_words += 1

            confidence = word.get("speaker_confidence")

            if (
                word_label in speaker_confidences
                and confidence is not None
            ):
                speaker_confidences[word_label].append(
                    confidence
                )

        segments.append({
            "segment_id": f"SEG_{index:06d}",
            "start_seconds": milliseconds_to_seconds(
                utterance.get("start")
            ),
            "end_seconds": milliseconds_to_seconds(
                utterance.get("end")
            ),
            "text": utterance.get("text", ""),
            "speaker_id": speaker_id,
            "confidence": utterance.get("confidence"),
            "speaker_confidence": utterance.get(
                "speaker_confidence"
            ),
            "words": words
        })

    speakers = []

    for label, speaker_id in speaker_map.items():

        values = speaker_confidences[label]

        average_confidence = (
            round(sum(values) / len(values), 4)
            if values else None
        )

        speakers.append({
            "speaker_id": speaker_id,
            "provider_label": label,
            "speaker_type": "unknown",
            "confidence": average_confidence
        })

    return {
        "schema_version": "1.0.0",

        "provider": {
            "name": "assemblyai",
            "model": "universal-3-5-pro",
            "transcript_id": raw.get("id"),
            "status": raw.get("status"),
            "language_code": raw.get("language_code"),
            "confidence": raw.get("confidence")
        },

        "source": {
            "audio_filename": audio_filename,
            "source_offset_seconds": 0.0,
            "duration_seconds": audio_duration
        },

        "statistics": {
            "total_segments": len(segments),
            "total_words": total_words,
            "total_speakers": len(speakers)
        },

        "full_text": raw.get("text", ""),
        "speakers": speakers,
        "segments": segments
    }


def main():

    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("ASSEMBLYAI_API_KEY"):
        raise RuntimeError(
            "API key missing. Configure the .env file first."
        )

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(AUDIO_FILE)

    audio_duration = get_audio_duration(AUDIO_FILE)

    print("FULL MOVIE TRANSCRIPTION ENGINE")
    print("--------------------------------")
    print(f"Audio: {AUDIO_FILE}")
    print(f"Duration: {audio_duration} seconds")
    print(f"Raw output: {RAW_OUTPUT}")
    print(f"Normalized output: {NORMALIZED_OUTPUT}")

    provider = get_provider()

    raw = provider.transcribe_file(AUDIO_FILE)

    save_json(RAW_OUTPUT, raw)

    normalized = normalize_transcript(
        raw,
        audio_duration,
        AUDIO_FILE.name
    )

    save_json(NORMALIZED_OUTPUT, normalized)

    print("\nFULL MOVIE TRANSCRIPTION: OK")

    print(
        "Language:",
        normalized["provider"]["language_code"]
    )

    print(
        "Segments:",
        normalized["statistics"]["total_segments"]
    )

    print(
        "Words:",
        normalized["statistics"]["total_words"]
    )

    print(
        "Speakers:",
        normalized["statistics"]["total_speakers"]
    )

    print(f"Raw output: {RAW_OUTPUT}")
    print(f"Normalized output: {NORMALIZED_OUTPUT}")


if __name__ == "__main__":
    main()
