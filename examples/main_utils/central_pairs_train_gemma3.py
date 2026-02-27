import argparse
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

MODEL_TO_CONFIG_DIR = {
    "270m": "central_pairs_gemma3_270m_it",
    "1b": "central_pairs_gemma3_1b_it",
    "4b": "central_pairs_gemma3_4b_it",
}


def _train_config_dir(config_dir_name: str) -> None:
    base_dir = Path(__file__).resolve().parents[1] / "experiment_configs" / config_dir_name
    for json_name in JSON_FILES:
        json_path = base_dir / json_name
        if not json_path.exists():
            raise FileNotFoundError(f"Missing concept JSON: {json_path}")
        print(f"[train] {json_path}")
        train_concept_from_json(str(json_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train central concept pairs for Gemma 3 instruct models."
    )
    parser.add_argument(
        "--model",
        choices=["270m", "1b", "4b", "all"],
        default="all",
        help="Which Gemma 3 model size to train (default: all).",
    )
    args = parser.parse_args()

    if args.model == "all":
        for key in ("270m", "1b", "4b"):
            print(f"[model] gemma-3-{key}-it")
            _train_config_dir(MODEL_TO_CONFIG_DIR[key])
        return

    _train_config_dir(MODEL_TO_CONFIG_DIR[args.model])


if __name__ == "__main__":
    main()
