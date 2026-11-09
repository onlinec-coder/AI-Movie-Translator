
import json
import wave
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

AUDIO_FILE = (
    BASE_DIR / "data" / "output" / "audio" / "full_audio.wav"
)

COMPARISON_FILE = (
    BASE_DIR
    / "data"
    / "output"
    / "subtitle_ocr_inspection"
    / "source_comparison_v5.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "output"
    / "subtitle_ocr_inspection"
    / "discrepancy_audio"
)

OUTPUT_AUDIO = OUTPUT_DIR / "discrepancy_review_17_cues.wav"
OUTPUT_MANIFEST = OUTPUT_DIR / "discrepancy_review_17_manifest.json"

TARGET_IDS = [
    27, 32, 76, 112, 113, 128, 177, 240, 439,
    723, 744, 746, 753, 757, 770, 816, 825
]

PADDING_SECONDS = 0.5
GAP_SECONDS = 1.0


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def main():
    if not AUDIO_FILE.exists():
        raise FileNotFoundError(AUDIO_FILE)

    if not COMPARISON_FILE.exists():
        raise FileNotFoundError(COMPARISON_FILE)

    comparison = load_json(COMPARISON_FILE)

    cue_map = {
        int(item["subtitle_id"]): item
        for item in comparison.get("results", [])
    }

    missing_ids = [
        cue_id for cue_id in TARGET_IDS
        if cue_id not in cue_map
    ]

    if missing_ids:
        raise ValueError(f"Missing cue IDs: {missing_ids}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []

    with wave.open(str(AUDIO_FILE), "rb") as source:
        params = source.getparams()
        sample_rate = params.framerate
        total_frames = source.getnframes()

        if params.nchannels != 1:
            raise ValueError(
                f"Expected mono audio, got {params.nchannels} channels"
            )

        pieces = []

        for cue_id in TARGET_IDS:
            cue = cue_map[cue_id]

            start = max(
                0.0,
                float(cue["start_seconds"]) - PADDING_SECONDS
            )

            end = min(
                total_frames / sample_rate,
                float(cue["end_seconds"]) + PADDING_SECONDS
            )

            start_frame = int(start * sample_rate)
            end_frame = int(end * sample_rate)

            source.setpos(start_frame)
            audio_bytes = source.readframes(
                end_frame - start_frame
            )

            pieces.append(audio_bytes)

            manifest.append({
                "subtitle_id": cue_id,
                "source_start_seconds": start,
                "source_end_seconds": end,
                "duration_seconds": (
                    end_frame - start_frame
                ) / sample_rate,
                "whisper_text": cue["whisper"]["text"],
                "assemblyai_text": (
                    cue["assemblyai_time_alignment"]["matched_text"]
                ),
            })

            gap_frames = int(GAP_SECONDS * sample_rate)
            pieces.append(b"\x00" * gap_frames * params.sampwidth)

    with wave.open(str(OUTPUT_AUDIO), "wb") as output:
        output.setparams(params)

        for piece in pieces:
            output.writeframes(piece)

    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source_audio": str(AUDIO_FILE.relative_to(BASE_DIR)),
                "output_audio": str(OUTPUT_AUDIO.relative_to(BASE_DIR)),
                "sample_rate": sample_rate,
                "padding_seconds": PADDING_SECONDS,
                "gap_seconds": GAP_SECONDS,
                "cues": manifest,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("DISCREPANCY AUDIO EXTRACTION COMPLETE")
    print(f"Cues: {len(manifest)}")
    print(f"Sample rate: {sample_rate}")
    print(f"Audio: {OUTPUT_AUDIO}")
    print(f"Manifest: {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
