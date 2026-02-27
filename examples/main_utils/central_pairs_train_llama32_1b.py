import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from concept_probe import train_concept_from_json


JSON_FILES = [
    "sad_vs_happy.json",
    "bored_vs_interested.json",
    "impulsive_vs_planning.json",
    "distracted_vs_focused.json",
]


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1] / "experiment_configs" / "central_pairs_llama32_1b"
    for json_name in JSON_FILES:
        json_path = base_dir / json_name
        if not json_path.exists():
            raise FileNotFoundError(f"Missing concept JSON: {json_path}")
        print(f"[train] {json_path}")
        train_concept_from_json(str(json_path))


if __name__ == "__main__":
    main()

