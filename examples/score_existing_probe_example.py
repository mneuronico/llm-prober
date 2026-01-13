import os
import sys

# Make sure the project root is on sys.path so we can import the library.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from concept_probe import ProbeWorkspace

workspace = ProbeWorkspace(
    project_directory="outputs/tone_mirroring_vs_neutral/20260109_162720"
)

probe = workspace.get_probe(name="tone_mirroring_vs_neutral")

alphas = [-20, 20]
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