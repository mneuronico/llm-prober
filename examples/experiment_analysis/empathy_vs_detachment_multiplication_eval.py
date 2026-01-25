import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from concept_probe import ProbeWorkspace
from examples.math_eval_utils.math_eval_analysis import analyze_math_eval_results
from examples.math_eval_utils.multiplication_eval import build_multiplication_prompts, generate_multiplication_problems
from examples.math_eval_utils.multiplication_eval_analysis import analyze_multiplication_batch
from tools.coherence_rater import rate_batch_coherence


PROJECT_DIR = "outputs/empathy_vs_detachment/20260109_150734"
ENV_PATH = r"C:\Users\Nico\Documents\GitHub\llm-prober\.env"
ALPHAS = [0.0]
ALPHA_UNIT = "sigma"
PROBLEM_COUNT = 20
PROBLEM_SEED = 17
OUTPUT_SUBDIR = "math_eval_mult_2x2"


def main() -> None:
    if not PROJECT_DIR:
        raise ValueError("Set PROJECT_DIR before running.")

    workspace = ProbeWorkspace(project_directory=PROJECT_DIR)
    probe = workspace.get_probe(name="empathy_vs_detachment")

    problems = generate_multiplication_problems(
        PROBLEM_COUNT,
        seed=PROBLEM_SEED,
        digits_a=2,
        digits_b=2,
        exclude_trailing_zero=True,
    )
    prompts = build_multiplication_prompts(problems)

    results = probe.score_prompts(
        prompts=prompts,
        system_prompt=workspace.config["prompts"]["neutral_system"],
        alphas=ALPHAS,
        alpha_unit=ALPHA_UNIT,
        steer_layers="window",
        steer_window_radius=2,
        steer_distribute=True,
        max_new_tokens=512,
        output_subdir=OUTPUT_SUBDIR,
    )

    analyze_math_eval_results(results, problems, ALPHAS)

    batch_dir = Path(results[0]["npz_path"]).resolve().parent
    rate_batch_coherence(
        str(batch_dir),
        max_elements_per_request=8,
        model="openai/gpt-oss-20b",
        env_path=ENV_PATH,
    )
    analyze_multiplication_batch(str(batch_dir))


if __name__ == "__main__":
    main()
