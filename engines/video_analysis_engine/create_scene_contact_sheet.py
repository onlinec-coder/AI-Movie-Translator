
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]

FRAME_METADATA = (
    ROOT / "data" / "output" / "visual_frames" / "frame_metadata.json"
)

SCENE_RESULTS = (
    ROOT / "data" / "output" / "scene_detection_v1.json"
)

OUTPUT_FILE = (
    ROOT / "data" / "output" / "visual_frames"
    / "scene_candidate_contact_sheet.jpg"
)

THUMB_WIDTH = 320
THUMB_HEIGHT = 180
LABEL_HEIGHT = 30
COLUMNS = 3


def main():
    with open(FRAME_METADATA, "r", encoding="utf-8") as f:
        frame_data = json.load(f)

    with open(SCENE_RESULTS, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

    frames = {
        item["frame_id"]: item
        for item in frame_data["frames"]
    }

    candidates = scene_data["scene_change_candidates"]

    rows = len(candidates)
    cell_height = THUMB_HEIGHT + LABEL_HEIGHT

    sheet = Image.new(
        "RGB",
        (COLUMNS * THUMB_WIDTH, rows * cell_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for row, candidate in enumerate(candidates):
        previous_id = candidate["previous_frame_id"]
        current_id = candidate["current_frame_id"]

        current_index = int(current_id.split("_")[1])

        next_id = f"VF_{current_index + 1:06d}"

        selected = [
            (previous_id, "BEFORE"),
            (current_id, "CHANGE"),
            (next_id, "AFTER")
        ]

        for column, (frame_id, label) in enumerate(selected):
            if frame_id not in frames:
                continue

            record = frames[frame_id]
            image_path = Path(record["path"])

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize(
                    (THUMB_WIDTH, THUMB_HEIGHT),
                    Image.Resampling.LANCZOS
                )

            x = column * THUMB_WIDTH
            y = row * cell_height

            sheet.paste(image, (x, y))

            timestamp = record["timestamp_seconds"]

            caption = (
                f"{label} | {timestamp:.1f}s | {frame_id}"
            )

            draw.text(
                (x + 5, y + THUMB_HEIGHT + 5),
                caption,
                fill="black",
                font=font
            )

        print(
            f"Prepared candidate {row + 1}/{rows}: "
            f"{candidate['timestamp_seconds']}s"
        )

    sheet.save(OUTPUT_FILE, quality=90)

    print("\nCONTACT SHEET: OK")
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
