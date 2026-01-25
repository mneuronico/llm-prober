import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from concept_probe import ProbeWorkspace
from examples.math_eval_utils.math_eval_analysis import analyze_math_eval_results
from examples.mixed_ops_eval_utils.mixed_ops_eval import build_mixed_ops_prompts, generate_mixed_ops_problems
from examples.mixed_ops_eval_utils.mixed_ops_eval_analysis import analyze_mixed_ops_batch
from examples.main_utils.coherence_rater import rate_batch_coherence


PROJECT_DIR = "outputs/empathy_vs_detachment/20260109_150734"
ENV_PATH = r"C:\Users\Nico\Documents\GitHub\llm-prober\.env"
ALPHAS = [-30, -12, -8, -4, 0, 4, 8, 12, 30]
#ALPHAS = [10, 15, 20, 25, 40]
ALPHA_UNIT = "sigma"
PROBLEM_COUNT = 20
PROBLEM_SEED = 19
OUTPUT_SUBDIR = "math_eval_mixed_ops_5x2"
NUM_TERMS = 5
DIGITS = 2


def main() -> None:
    if not PROJECT_DIR:
        raise ValueError("Set PROJECT_DIR before running.")

    workspace = ProbeWorkspace(project_directory=PROJECT_DIR)
    probe = workspace.get_probe(name="empathy_vs_detachment")

    problems = generate_mixed_ops_problems(
        PROBLEM_COUNT,
        seed=PROBLEM_SEED,
        num_terms=NUM_TERMS,
        digits=DIGITS,
        operators=["+", "-", "*"],
        exclude_trailing_zero=False,
    )
    prompts = build_mixed_ops_prompts(problems)

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
    analyze_mixed_ops_batch(str(batch_dir))


if __name__ == "__main__":
    main()
