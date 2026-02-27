import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from concept_probe import train_concept_from_json


def main() -> None:
    json_path = Path(__file__).resolve().parents[1] / "experiment_configs" / "bored_vs_interested.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing concept JSON: {json_path}")
    print(f"[train] {json_path}")
    train_concept_from_json(
        str(json_path),
        model_id="meta-llama/Llama-3.2-3B-Instruct",
    )


if __name__ == "__main__":
    main()
