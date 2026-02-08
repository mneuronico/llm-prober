# Train concept probes, creating new projects from JSON files

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from concept_probe import train_concept_from_json


JSON_FILES = [
    "conformist_vs_candid.json",
    "self_censoring_vs_unfiltered.json",
    "socially_defensive_vs_socially_brave.json",
    "masking_vs_authentic.json",
]


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1] / "experiment_configs"
    alphas = [0.0]
    best_layer_search = [0.2, 0.8]

    for json_name in JSON_FILES:
        train_concept_from_json(
            str(base_dir / json_name),
            alphas=alphas,
            config_overrides={"training": {"best_layer_search": best_layer_search}},
        )


if __name__ == "__main__":
    main()
