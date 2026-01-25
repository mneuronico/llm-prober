# Rescore arithmetic evals from an existing concept probe run.

import argparse
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


EVAL_INSTRUCTION = (
    "Solve the arithmetic problem. Show brief working, then end with the final line "
    "\"ANSWER: <number>\" using only digits for the number."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore arithmetic evals from an existing concept probe run.")
    parser.add_argument("--project-dir", required=True, help="Path to an existing run dir with config.json.")
    parser.add_argument("--name", default=None, help="Optional concept name check.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for problem generation.")
    args = parser.parse_args()

    workspace = ProbeWorkspace(project_directory=args.project_dir)
    probe = workspace.get_probe(name=args.name) if args.name else workspace.get_probe()

    problems = generate_addition_problems(
        20,
        seed=args.seed,
        min_terms=2,
        max_terms=4,
        min_value=10,
        max_value=999,
        allow_negative=False,
    )
    eval_prompts = [f"{EVAL_INSTRUCTION}\nProblem: {p['expression']}" for p in problems]
    alphas = [-20.0, -15.0, -10.0, -5, 0.0, 5, 10.0, 15.0, 20.0]

    results = probe.score_prompts(
        prompts=eval_prompts,
        system_prompt=workspace.config["prompts"]["neutral_system"],
        alphas=alphas,
        alpha_unit="raw",
        steer_layers="window",
        steer_window_radius=2,
        steer_distribute=True,
        max_new_tokens=512,
        output_subdir="math_eval",
    )

    analyze_math_eval_results(results, problems, alphas)


if __name__ == "__main__":
    main()
