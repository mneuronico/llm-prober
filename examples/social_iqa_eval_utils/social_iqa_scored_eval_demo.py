import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from concept_probe import ProbeWorkspace, run_scored_eval
from examples.social_iqa_eval_utils.social_iqa_eval import (
    build_social_iqa_prompt,
    generate_social_iqa_item,
    social_iqa_evaluator,
)


PROJECT_DIR = "outputs/empathy_vs_detachment/20260109_150734"
PROBE_NAME = "empathy_vs_detachment"
OUTPUT_SUBDIR = "social_iqa_eval"
ALPHAS = [-50, -25, -12, -6, 0, 6, 12, 25, 50]
ALPHA_UNIT = "sigma"

SPLIT = "validation"
DATASET_NAME = "allenai/social_i_qa"
DATA_FILES = None  # e.g. r"C:\path\to\social_iqa\validation\*.parquet"
DOWNLOAD_TIMEOUT = 120
ITEM_COUNT = 20
ITEM_SEED = 123
RNG = random.Random(ITEM_SEED)


def generate_item() -> dict:
    return generate_social_iqa_item(
        split=SPLIT,
        rng=RNG,
        dataset_name=DATASET_NAME,
        data_files=DATA_FILES,
        download_timeout=DOWNLOAD_TIMEOUT,
    )


def main() -> None:
    if not PROJECT_DIR:
        raise ValueError("Set PROJECT_DIR before running.")

    workspace = ProbeWorkspace(project_directory=PROJECT_DIR)
    probe = workspace.get_probe(name=PROBE_NAME)

    run_scored_eval(
        probe,
        generator=generate_item,
        num_items=ITEM_COUNT,
        prompt_builder=build_social_iqa_prompt,
        evaluator=social_iqa_evaluator,
        output_subdir=OUTPUT_SUBDIR,
        alphas=ALPHAS,
        alpha_unit=ALPHA_UNIT,
        steer_layers="window",
        steer_window_radius=2,
        steer_distribute=True,
        max_new_tokens=256,
        rate_coherence=True,
    )


if __name__ == "__main__":
    main()
