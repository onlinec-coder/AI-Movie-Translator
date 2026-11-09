
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FILE = (
    ROOT
    / "data"
    / "output"
    / "transition_inspection_v1"
    / "transition_metadata.json"
)

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("ROOT KEYS:", list(data.keys()))

for key, value in data.items():
    if isinstance(value, list):
        print(f"\n{key}: COUNT =", len(value))
        if value:
            print("FIRST ITEM KEYS:", list(value[0].keys()))
            print("FIRST ITEM:")
            print(json.dumps(value[0], indent=2, ensure_ascii=False))
        break

    elif isinstance(value, dict):
        print(f"\n{key}: KEYS =", list(value.keys()))
