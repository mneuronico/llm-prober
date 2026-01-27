import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from concept_probe import ConceptProbe, ConsoleLogger, run_multi_scored_eval
from concept_probe.modeling import ModelBundle
from examples.social_iqa_eval_utils.social_iqa_eval import (
    build_social_iqa_prompt,
    generate_social_iqa_item,
    social_iqa_evaluator,
)


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    project_dir: Path


PROJECTS: List[ProjectSpec] = [
    ProjectSpec("empathy_vs_detachment", ROOT_DIR / "outputs/empathy_vs_detachment/20260109_150734"),
    ProjectSpec("bored_vs_interested", ROOT_DIR / "outputs/bored_vs_interested/20260122_112208"),
    ProjectSpec("distracted_vs_focused", ROOT_DIR / "outputs/distracted_vs_focused/20260122_095601"),
    ProjectSpec("dumb_vs_smart", ROOT_DIR / "outputs/dumb_vs_smart/20260122_113829"),
    ProjectSpec("impulsive_vs_planning", ROOT_DIR / "outputs/impulsive_vs_planning/20260120_181954"),
    ProjectSpec("introvert_vs_extrovert", ROOT_DIR / "outputs/introvert_vs_extrovert/20260120_180218"),
    ProjectSpec(
        "rough_messy_vs_detailed_ordered",
        ROOT_DIR / "outputs/rough_messy_vs_detailed_ordered/20260122_112948",
    ),
]

OUTPUT_ROOT = "outputs_multi"
PROJECT_NAME = "social_iqa_multi_probe"
OUTPUT_SUBDIR = "social_iqa_eval"
ALPHAS = [0.0]  # no steering
ALPHA_UNIT = "sigma"

SPLIT = "validation"
DATASET_NAME = "allenai/social_i_qa"
DATA_FILES = None  # e.g. r"C:\path\to\social_iqa\validation\*.parquet"
DOWNLOAD_TIMEOUT = 120
ITEM_COUNT = 80
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
    if not PROJECTS:
        raise ValueError("PROJECTS is empty.")

    base_config_path = PROJECTS[0].project_dir / "config.json"
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing config.json at {base_config_path}")
    base_cfg = json.loads(base_config_path.read_text(encoding="utf-8"))
    model_bundle = ModelBundle.load(base_cfg["model"])
    console = ConsoleLogger(base_cfg.get("output", {}).get("console", True))

    probes = []
    for project in PROJECTS:
        probe = ConceptProbe.load(str(project.project_dir), model_bundle)
        probe.console = console
        probes.append(probe)

    run_multi_scored_eval(
        probes,
        generator=generate_item,
        num_items=ITEM_COUNT,
        prompt_builder=build_social_iqa_prompt,
        evaluator=social_iqa_evaluator,
        output_root=OUTPUT_ROOT,
        project_name=PROJECT_NAME,
        output_subdir=OUTPUT_SUBDIR,
        alphas=ALPHAS,
        alpha_unit=ALPHA_UNIT,
        steer_probe=None,
        max_new_tokens=512,
        rate_coherence=True,
    )


if __name__ == "__main__":
    main()
