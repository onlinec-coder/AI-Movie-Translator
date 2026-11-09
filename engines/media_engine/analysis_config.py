
import re
from pathlib import Path


# Үндсэн шинжилгээний хэсгийн урт
BASE_WINDOW_SECONDS = 300.0

# Сүүлийн хадмалын дараах хамгаалалтын хугацаа
EXTRA_PADDING_SECONDS = 1.0

# Extraction хугацааг хэдэн секундээр дээш бүхэлчлэх
ROUND_UP_SECONDS = 5


TIMESTAMP_PATTERN = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def timestamp_to_seconds(timestamp):
    match = TIMESTAMP_PATTERN.fullmatch(timestamp.strip())

    if not match:
        raise ValueError(f"Invalid timestamp: {timestamp}")

    hours, minutes, seconds, milliseconds = map(
        int, match.groups()
    )

    return (
        hours * 3600
        + minutes * 60
        + seconds
        + milliseconds / 1000
    )


def parse_srt(srt_text):
    blocks = re.split(r"\r?\n\s*\r?\n", srt_text.strip())

    segments = []

    for block in blocks:
        lines = block.strip().splitlines()

        if len(lines) < 3:
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        timing = lines[1].strip()

        if "-->" not in timing:
            continue

        start_text, end_text = timing.split("-->", 1)

        start_seconds = timestamp_to_seconds(start_text)
        end_seconds = timestamp_to_seconds(end_text)

        if end_seconds < start_seconds:
            raise ValueError(
                f"Subtitle {index}: end time is before start time"
            )

        text = "\n".join(lines[2:]).strip()

        if not text:
            continue

        segments.append({
            "subtitle_id": index,
            "start_seconds": round(start_seconds, 3),
            "end_seconds": round(end_seconds, 3),
            "duration_seconds": round(
                end_seconds - start_seconds, 3
            ),
            "text": text
        })

    return segments


def round_up(value, step):
    return int((value + step - 1e-9) // step + 1) * step


def build_analysis_plan(srt_path, source_duration=None):
    srt_path = Path(srt_path)

    if not srt_path.exists():
        raise FileNotFoundError(srt_path)

    srt_text = srt_path.read_text(encoding="utf-8-sig")
    all_segments = parse_srt(srt_text)

    selected_segments = [
        segment
        for segment in all_segments
        if 0 <= segment["start_seconds"] < BASE_WINDOW_SECONDS
    ]

    if not selected_segments:
        raise ValueError(
            "No subtitles found inside the analysis window."
        )

    last_subtitle_end = max(
        segment["end_seconds"]
        for segment in selected_segments
    )

    required_end = (
        last_subtitle_end + EXTRA_PADDING_SECONDS
    )

    extraction_duration = max(
        BASE_WINDOW_SECONDS,
        round_up(required_end, ROUND_UP_SECONDS)
    )

    if source_duration is not None:
        if last_subtitle_end > source_duration:
            raise ValueError(
                "A selected subtitle extends beyond the source video."
            )

        extraction_duration = min(
            extraction_duration,
            source_duration
        )

    return {
        "all_segments": all_segments,
        "segments": selected_segments,
        "base_window_start": 0.0,
        "base_window_end": BASE_WINDOW_SECONDS,
        "last_subtitle_end": last_subtitle_end,
        "required_end": required_end,
        "extraction_duration": round(extraction_duration, 3)
    }
