
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "ASSEMBLYAI TRANSCRIPTION": (
        ROOT / "data/output/transcription_result.json"
    ),
    "SPEAKER ASSIGNMENTS": (
        ROOT / "data/input/speaker_assignments.json"
    ),
    "RECONCILED CHARACTERS": (
        ROOT / "data/input/Husbands_characters_reconciled.json"
    ),
}


def inspect_structure(data, depth=0, max_depth=3):
    indent = "  " * depth

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{indent}{key}: dict ({len(value)} keys)")
                if depth < max_depth:
                    inspect_structure(value, depth + 1, max_depth)

            elif isinstance(value, list):
                print(f"{indent}{key}: list ({len(value)} items)")
                if value:
                    print(f"{indent}  First item:")
                    if depth < max_depth:
                        inspect_structure(
                            value[0], depth + 2, max_depth
                        )
                    else:
                        print(f"{indent}    {str(value[0])[:300]}")

            else:
                print(
                    f"{indent}{key}: "
                    f"{type(value).__name__} = {str(value)[:150]}"
                )

    elif isinstance(data, list):
        print(f"{indent}List items: {len(data)}")
        if data:
            inspect_structure(data[0], depth + 1, max_depth)


def main():
    print("SPEAKER / CHARACTER DATA INSPECTION")
    print("-----------------------------------")

    for name, path in FILES.items():
        print(f"\n\n{name}")
        print("=" * 50)

        if not path.exists():
            print(f"FILE NOT FOUND: {path}")
            continue

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        print(f"File: {path.name}")
        print(f"Root type: {type(data).__name__}")

        inspect_structure(data)

    print("\n\nINSPECTION COMPLETE")


if __name__ == "__main__":
    main()
