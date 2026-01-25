import os
import sys

# Make sure the project root is on sys.path so we can import the library.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from concept_probe import ProbeWorkspace


def main() -> None:
    workspace = ProbeWorkspace(
        project_directory="outputs/empathy_vs_detachment/20260109_150734",
    )

    probe = workspace.get_probe(name="empathy_vs_detachment")

    alphas = [0.0, 4.0, -4.0, 12.0, -12.0, 100.0, -100.0]
    eval_prompts = workspace.config["prompts"]["eval_questions"]

    probe.score_prompts(
        prompts=eval_prompts,
        system_prompt=workspace.config["prompts"]["neutral_system"],
        alphas=alphas,
        alpha_unit="sigma",
        steer_layers="window",
        steer_window_radius=2,
        steer_distribute=True,
    )


if __name__ == "__main__":
    main()
