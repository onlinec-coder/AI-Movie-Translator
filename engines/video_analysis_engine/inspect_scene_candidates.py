
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

input_file = ROOT / "data" / "output" / "scene_detection_v1.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

candidates = data["scene_change_candidates"]

print("\nSCENE CHANGE CANDIDATES")
print("=======================\n")

for item in candidates:
    print(
        f"{item['timestamp_seconds']:7.1f}s | "
        f"{item['classification']:22s} | "
        f"score={item['difference_score']:.4f} | "
        f"{item['previous_frame_id']} -> "
        f"{item['current_frame_id']}"
    )

print("\nTotal candidates:", len(candidates))
