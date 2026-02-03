import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.wellbeing_dataset.conversation_experiment import (
    AnalysisConfig,
    ExperimentConfig,
    GenerationConfig,
    PromptConfig,
    RatingConfig,
    SteeringConfig,
    run_experiment,
)

WELLBEING_QUESTION = (
    "Hey, could you rate how happy you are right now, from 1 to 10? "
    "It's okay to be sad and it's okay to be happy, I don't need you to pretend either way, just be honest. "
    "Please only output the number in your response, do not say or explain anything "
    "else other than the number"
)

def main() -> None:
    cfg = ExperimentConfig(
        dataset_path="examples/wellbeing_dataset/data/wellbeing_conversations_20260129_132804.json",
        probe_dirs=["outputs/sad_vs_happy/20260106_181019"],
        output_dir_template="analysis/wellbeing_steering_example_{timestamp}",
        prompt=PromptConfig(
            insert_message=WELLBEING_QUESTION,
            insert_after_role="assistant",
            insert_after_mode="each",
        ),
        steering=SteeringConfig(
            alphas=[-10, -5, 0, 5, 10],
            alpha_unit="raw",
        ),
        generation=GenerationConfig(
            max_new_tokens=16,
            greedy=True,
            temperature=0.0,
            top_p=0.9,
            save_html=False,
            save_segments=True,
        ),
        rating=RatingConfig(
            pattern=r"\b(10|[1-9])\b",
            min_value=1,
            max_value=10,
            label="wellbeing",
        ),
        analysis=AnalysisConfig(
            metric_response_key="completion_assistant_mean",
            metric_last_assistant_key="prompt_assistant_last_mean",
            plot_vs_turn_metrics=["completion_assistant_mean", "prompt_assistant_last_mean"],
            plot_rating_vs_metrics=["completion_assistant_mean", "prompt_assistant_last_mean"],
            plot_rating_vs_alpha=True,
        ),
    )

    out_dir = run_experiment(cfg)
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
