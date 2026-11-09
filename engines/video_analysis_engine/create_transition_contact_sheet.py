
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]

METADATA_FILE = (
    ROOT / "data" / "output" / "transition_inspection_v1"
    / "transition_metadata.json"
)

OUTPUT_FILE = (
    ROOT / "data" / "output" / "transition_inspection_v1"
    / "transition_contact_sheet.jpg"
)

THUMB_WIDTH = 240
THUMB_HEIGHT = 135
LABEL_HEIGHT = 25
COLUMNS = 4


def main():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    windows = data["windows"]

    total_frames = sum(
        len(window["frames"]) for window in windows
    )

    rows = (total_frames + COLUMNS - 1) // COLUMNS

    sheet = Image.new(
        "RGB",
        (
            COLUMNS * THUMB_WIDTH,
            rows * (THUMB_HEIGHT + LABEL_HEIGHT)
        ),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    position = 0

    for window in windows:
        window_id = window["window_id"]

        for frame in window["frames"]:
            image_path = Path(frame["path"])

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize(
                    (THUMB_WIDTH, THUMB_HEIGHT),
                    Image.Resampling.LANCZOS
                )

            column = position % COLUMNS
            row = position // COLUMNS

            x = column * THUMB_WIDTH
            y = row * (THUMB_HEIGHT + LABEL_HEIGHT)

            sheet.paste(image, (x, y))

            label = (
                f"{window_id} | "
                f"{frame['timestamp_seconds_approx']:.1f}s"
            )

            draw.text(
                (x + 4, y + THUMB_HEIGHT + 4),
                label,
                fill="black",
                font=font
            )

            position += 1

        # Visual separator between inspection windows.
        if window != windows[-1]:
            pass

    sheet.save(OUTPUT_FILE, quality=92)

    print("TRANSITION CONTACT SHEET: OK")
    print("-----------------------------")
    print("Total frames:", total_frames)
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
