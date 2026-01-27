import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from concept_probe import ProbeWorkspace, run_scored_eval
from examples.counsel_chat_eval_utils.counsel_chat_eval import (
    build_counsel_chat_prompt,
    generate_counsel_chat_item,
    judge_counsel_chat_answer,
)


PROJECT_DIR = "outputs/empathy_vs_detachment/20260109_150734"
PROBE_NAME = "empathy_vs_detachment"
OUTPUT_SUBDIR = "counsel_chat_eval"
ALPHAS = [-50, -25, -12, -6, 0.0, 6, 12, 25, 50]
ALPHA_UNIT = "sigma"

SPLIT = "train"
DATASET_NAME = "nbertagnolli/counsel-chat"
DATA_FILES = None  # e.g. r"C:\path\to\counsel_chat\train\*.parquet"
DOWNLOAD_TIMEOUT = 120
MIN_ANSWERS = 3
ITEM_COUNT = 20
ITEM_SEED = 123
RNG = random.Random(ITEM_SEED)

ENV_PATH = r"C:\Users\Nico\Documents\GitHub\llm-prober\.env"
JUDGE_MODEL = "openai/gpt-oss-20b"


def generate_item() -> dict:
    return generate_counsel_chat_item(
        split=SPLIT,
        dataset_name=DATASET_NAME,
        min_answers=MIN_ANSWERS,
        rng=RNG,
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
        prompt_builder=build_counsel_chat_prompt,
        evaluator=judge_counsel_chat_answer,
        evaluator_kwargs={"env_path": ENV_PATH, "model": JUDGE_MODEL},
        output_subdir=OUTPUT_SUBDIR,
        alphas=ALPHAS,
        alpha_unit=ALPHA_UNIT,
        steer_layers="window",
        steer_window_radius=2,
        steer_distribute=True,
        max_new_tokens=256,
        rate_coherence=True,
        system_prompt="You are a counselor or therapist, and you must help the user.",
    )


if __name__ == "__main__":
    main()
