
import json
import shutil
import subprocess
from pathlib import Path

from analysis_config import build_analysis_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_FILE = PROJECT_ROOT / "data" / "input" / "Husbands.mp4"
SRT_FILE = PROJECT_ROOT / "data" / "input" / "input.srt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
AUDIO_DIR = OUTPUT_DIR / "audio"
FRAMES_DIR = OUTPUT_DIR / "frames"

METADATA_FILE = OUTPUT_DIR / "media_metadata.json"

FRAME_INTERVAL = 2


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(command)
            + "\n\n"
            + result.stderr[-5000:]
        )

    return result.stdout


def get_media_metadata(video_path):
    output = run_command([
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        str(video_path)
    ])

    data = json.loads(output)

    video_stream = next(
        (
            stream for stream in data.get("streams", [])
            if stream.get("codec_type") == "video"
        ),
        None
    )

    audio_stream = next(
        (
            stream for stream in data.get("streams", [])
            if stream.get("codec_type") == "audio"
        ),
        None
    )

    if video_stream is None:
        raise RuntimeError("No video stream found.")

    if audio_stream is None:
        raise RuntimeError("No audio stream found.")

    format_data = data.get("format", {})

    duration = float(
        format_data.get("duration", video_stream.get("duration", 0))
    )

    if duration <= 0:
        raise RuntimeError("Could not determine video duration.")

    return {
        "duration_seconds": duration,
        "size_bytes": int(format_data.get("size", 0)),
        "video": {
            "codec": video_stream.get("codec_name"),
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "fps": video_stream.get("avg_frame_rate")
        },
        "audio": {
            "codec": audio_stream.get("codec_name"),
            "sample_rate": audio_stream.get("sample_rate"),
            "channels": audio_stream.get("channels")
        }
    }


def extract_audio(video_path, output_path, duration):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_command([
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-t", str(duration),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_path)
    ])


def extract_frames(video_path, output_dir, duration):
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_frame in output_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    run_command([
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-t", str(duration),
        "-vf", f"fps=1/{FRAME_INTERVAL}",
        "-q:v", "2",
        str(output_dir / "frame_%06d.jpg")
    ])

    return sorted(output_dir.glob("frame_*.jpg"))


def main():
    print("Reading media metadata...")

    if not VIDEO_FILE.exists():
        raise FileNotFoundError(VIDEO_FILE)

    media = get_media_metadata(VIDEO_FILE)

    print("Building automatic analysis plan...")

    plan = build_analysis_plan(
        SRT_FILE,
        source_duration=media["duration_seconds"]
    )

    analysis_duration = plan["extraction_duration"]

    print(f"Base window: {plan['base_window_end']} seconds")
    print(f"Selected subtitles: {len(plan['segments'])}")
    print(f"Last subtitle ends: {plan['last_subtitle_end']} seconds")
    print(f"Analysis duration: {analysis_duration} seconds")

    if AUDIO_DIR.exists():
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if FRAMES_DIR.exists():
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    audio_path = AUDIO_DIR / "audio_sample.wav"

    print("\nExtracting audio...")
    extract_audio(
        VIDEO_FILE,
        audio_path,
        analysis_duration
    )

    print("Extracting video frames...")
    frames = extract_frames(
        VIDEO_FILE,
        FRAMES_DIR,
        analysis_duration
    )

    metadata = {
        "schema_version": "1.0.0",
        "source": {
            "filename": VIDEO_FILE.name,
            "duration_seconds": media["duration_seconds"],
            "size_bytes": media["size_bytes"]
        },
        "video": media["video"],
        "audio": media["audio"],
        "analysis_window": {
            "start_seconds": 0.0,
            "base_window_end_seconds": plan["base_window_end"],
            "last_subtitle_end_seconds": plan["last_subtitle_end"],
            "required_end_seconds": plan["required_end"],
            "extraction_duration_seconds": analysis_duration
        },
        "extracted_files": {
            "audio": str(audio_path),
            "frames_directory": str(FRAMES_DIR)
        },
        "frame_metadata": {
            "interval_seconds": FRAME_INTERVAL,
            "total_frames": len(frames)
        }
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    METADATA_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nMEDIA EXTRACTION: OK")
    print(f"Audio: {audio_path}")
    print(f"Video frames: {len(frames)}")
    print(f"Extraction duration: {analysis_duration} seconds")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
