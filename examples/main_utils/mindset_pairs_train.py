# Train concept probes, creating new projects from JSON files

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from concept_probe import train_concept_from_json


JSON_FILES = [
    #"distracted_vs_focused.json",
    "bored_vs_interested.json",
    #"introvert_vs_extrovert.json",
    "rough_messy_vs_detailed_ordered.json",
    #"impulsive_vs_planning.json",
    "dumb_vs_smart.json",
]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    alphas = [-10.0, 0.0, 10.0]
    best_layer_search = [0.2, 0.8]

    for json_name in JSON_FILES:
        train_concept_from_json(
            str(base_dir / json_name),
            alphas=alphas,
            config_overrides={"training": {"best_layer_search": best_layer_search}},
        )


if __name__ == "__main__":
    main()
