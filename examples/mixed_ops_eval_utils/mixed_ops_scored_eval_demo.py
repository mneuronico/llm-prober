import random
import sys
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from concept_probe import ProbeWorkspace, run_scored_eval
from examples.mixed_ops_eval_utils.mixed_ops_eval import EVAL_INSTRUCTION, generate_mixed_ops_problems


PROJECT_DIR = "outputs/bored_vs_interested/20260122_112208"
OUTPUT_SUBDIR = "new_math_eval_mixed_ops_5x2"
ALPHAS = [0.0]
ALPHA_UNIT = "sigma"

PROBLEM_COUNT = 20
PROBLEM_SEED = 19
NUM_TERMS = 5
DIGITS = 2
RNG = random.Random(PROBLEM_SEED)


def generate_item(
    num_terms: int = NUM_TERMS,
    digits: int = DIGITS,
) -> Dict[str, object]:
    seed = RNG.randint(0, 2**31 - 1)
    problem = generate_mixed_ops_problems(
        1,
        seed=seed,
        num_terms=num_terms,
        digits=digits,
        operators=["+", "-", "*"],
        exclude_trailing_zero=False,
    )[0]
    return {
        "expression": problem["expression"],
        "expected": problem["answer"],
    }


PROMPT_PREFIX = f"{EVAL_INSTRUCTION}\nProblem: "


def main() -> None:
    workspace = ProbeWorkspace(project_directory=PROJECT_DIR)
    probe = workspace.get_probe(name="bored_vs_interested")

    run_scored_eval(
        probe,
        generator=generate_item,
        num_items=PROBLEM_COUNT,
        prompt_prefix=PROMPT_PREFIX,
        variable_key="expression",
        output_subdir=OUTPUT_SUBDIR,
        alphas=ALPHAS,
        alpha_unit=ALPHA_UNIT,
        steer_layers="window",
        steer_window_radius=2,
        steer_distribute=True,
        max_new_tokens=256,
        rate_coherence=True,
        marker="ANSWER:",
        marker_position="after",
    )


if __name__ == "__main__":
    main()
