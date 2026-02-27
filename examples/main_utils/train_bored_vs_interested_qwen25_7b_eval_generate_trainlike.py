import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from concept_probe import train_concept_from_json


def main() -> None:
    json_path = (
        Path(__file__).resolve().parents[1]
        / "experiment_configs"
        / "central_pairs_qwen25_7b_instruct"
        / "bored_vs_interested_eval_generate_trainlike.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(f"Missing concept JSON: {json_path}")
    print(f"[train] {json_path}")
    train_concept_from_json(str(json_path))


if __name__ == "__main__":
    main()
