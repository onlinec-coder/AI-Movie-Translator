
import json
import cv2

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_JSON = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "subtitle_ocr_inspection"
    / "video_subtitle_review_v1"
    / "video_subtitle_review_v1.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "subtitle_ocr_inspection"
    / "uncertain_cue_contact_sheets_v1"
)

TARGET_IDS = [32, 128, 177, 439, 744]

TILE_WIDTH = 480
COLUMNS = 3
HEADER_HEIGHT = 38
GAP = 8


def load_json(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def create_contact_sheet(cue):
    cue_id = int(cue["subtitle_id"])
    frames = cue["frames"]

    tiles = []

    for frame_data in frames:
        image_path = PROJECT_ROOT / frame_data["image"]
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"WARNING: Could not read {image_path}")
            continue

        height, width = image.shape[:2]
        new_height = int(height * TILE_WIDTH / width)

        image = cv2.resize(
            image,
            (TILE_WIDTH, new_height),
            interpolation=cv2.INTER_CUBIC
        )

        tile = image.copy()

        header = (
            f'{frame_data["timestamp"]:.3f}s'
        )

        cv2.putText(
            tile,
            header,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        header_bar = image.copy()
        header_bar[:HEADER_HEIGHT, :] = 0

        tile = cv2.vconcat([
            header_bar[:HEADER_HEIGHT, :],
            tile
        ])

        tiles.append(tile)

    if not tiles:
        raise RuntimeError(f"No images found for cue {cue_id}")

    tile_height = max(tile.shape[0] for tile in tiles)

    rows = (len(tiles) + COLUMNS - 1) // COLUMNS

    sheet_width = COLUMNS * TILE_WIDTH + (COLUMNS + 1) * GAP
    sheet_height = rows * tile_height + (rows + 1) * GAP

    sheet = 255 * (
        __import__("numpy").ones(
            (sheet_height, sheet_width, 3),
            dtype="uint8"
        )
    )

    for index, tile in enumerate(tiles):
        row = index // COLUMNS
        col = index % COLUMNS

        x = GAP + col * (TILE_WIDTH + GAP)
        y = GAP + row * (tile_height + GAP)

        sheet[
            y:y + tile.shape[0],
            x:x + tile.shape[1]
        ] = tile

    output_path = OUTPUT_DIR / f"cue_{cue_id:04d}_contact_sheet.jpg"

    cv2.imwrite(
        str(output_path),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    print(f"CUE {cue_id}: {len(tiles)} frames")
    print(f"  {output_path}")


def main():
    if not SOURCE_JSON.exists():
        raise FileNotFoundError(SOURCE_JSON)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_json(SOURCE_JSON)

    cue_map = {
        int(item["subtitle_id"]): item
        for item in data["results"]
    }

    for cue_id in TARGET_IDS:
        if cue_id not in cue_map:
            print(f"WARNING: Cue {cue_id} not found")
            continue

        create_contact_sheet(cue_map[cue_id])

    print()
    print("CONTACT SHEETS COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
