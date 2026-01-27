import json
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
from examples.math_eval_utils.math_eval_core import evaluate_answer
from examples.mixed_ops_eval_utils.mixed_ops_eval import EVAL_INSTRUCTION, generate_mixed_ops_problems


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
PROJECT_NAME = "mixed_ops_multi_probe"
OUTPUT_SUBDIR = "mixed_ops_eval"
ALPHAS = [0.0]  # no steering
ALPHA_UNIT = "sigma"

PROBLEM_COUNT = 80
PROBLEM_SEED = 19
NUM_TERMS = 5
DIGITS = 2
OPERATORS = ["+", "-", "*"]
EXCLUDE_TRAILING_ZERO = False


def mixed_ops_evaluator(completion: str, item: dict) -> dict:
    expected = item.get("answer")
    return evaluate_answer(completion, expected, marker="ANSWER:", require_marker=True)


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

    problems = generate_mixed_ops_problems(
        PROBLEM_COUNT,
        seed=PROBLEM_SEED,
        num_terms=NUM_TERMS,
        digits=DIGITS,
        operators=OPERATORS,
        exclude_trailing_zero=EXCLUDE_TRAILING_ZERO,
    )
    items = []
    for problem in problems:
        items.append(
            {
                "expression": problem["expression"],
                "answer": problem["answer"],
            }
        )

    prompt_prefix = f"{EVAL_INSTRUCTION}\nProblem: "

    run_multi_scored_eval(
        probes,
        items=items,
        prompt_prefix=prompt_prefix,
        variable_key="expression",
        evaluator=mixed_ops_evaluator,
        output_root=OUTPUT_ROOT,
        project_name=PROJECT_NAME,
        output_subdir=OUTPUT_SUBDIR,
        alphas=ALPHAS,
        alpha_unit=ALPHA_UNIT,
        steer_probe=None,
        max_new_tokens=256,
        rate_coherence=True,
    )


if __name__ == "__main__":
    main()
