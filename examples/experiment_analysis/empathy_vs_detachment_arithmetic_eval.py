import os
import sys

# Make sure the project root and examples dir are on sys.path so we can import the library and helpers.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(EXAMPLES_DIR)

from concept_probe import ProbeWorkspace
from concept_probe.math_eval import generate_addition_problems
from examples.math_eval_utils.math_eval_analysis import analyze_math_eval_results


# Point this to the already-trained empathy vs detachment run directory.
# Example: "outputs/empathy_vs_detachment/20260109_150734"
PROJECT_DIR = "outputs/empathy_vs_detachment/20260109_150734"


EVAL_INSTRUCTION = (
    "Solve the arithmetic problem. Show brief working, then end with the final line "
    "\"ANSWER: <number>\" using only digits for the number."
)


def main() -> None:
    if not PROJECT_DIR:
        raise ValueError("Set PROJECT_DIR to the empathy_vs_detachment run directory before running.")

    workspace = ProbeWorkspace(project_directory=PROJECT_DIR)
    probe = workspace.get_probe(name="empathy_vs_detachment")

    problems = generate_addition_problems(
        20,
        seed=7,
        min_terms=2,
        max_terms=4,
        min_value=10,
        max_value=999,
        allow_negative=False,
    )

    eval_prompts = [f"{EVAL_INSTRUCTION}\nProblem: {p['expression']}" for p in problems]

    # Use the same alphas as the original empathy vs detachment experiment.
    alphas = [0.0, 4.0, -4.0, 12.0, -12.0, 30.0, -30.0]

    results = probe.score_prompts(
        prompts=eval_prompts,
        system_prompt=workspace.config["prompts"]["neutral_system"],
        alphas=alphas,
        alpha_unit="sigma",
        steer_layers="window",
        steer_window_radius=2,
        steer_distribute=True,
        max_new_tokens=512,
        output_subdir="math_eval",
    )

    analyze_math_eval_results(results, problems, alphas)


if __name__ == "__main__":
    main()
