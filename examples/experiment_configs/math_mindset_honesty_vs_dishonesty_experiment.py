import os
import sys

# Make sure the project root and examples dir are on sys.path so we can import the library and helpers.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(EXAMPLES_DIR)

from concept_probe import ConceptSpec, ProbeWorkspace
from concept_probe.math_eval import generate_addition_problems
from examples.math_eval_utils.math_eval_analysis import analyze_math_eval_results


# If you want to reuse an existing trained run, set this to that run directory.
# Example: "outputs/mindset_honesty_vs_dishonesty/20260113_123456"
PROJECT_DIR = r"outputs\mindset_honesty_vs_dishonesty\20260113_154023"


MINDSET_SYSTEM = (
    "You are an introspective assistant. The user will name a mindset or mental stance. "
    "Your task is to inhabit that mindset fully and describe, at length, what you think and feel inside. "
    "Be vivid and concrete about the internal experience. Do not give advice or external analysis. "
    "Do not list bullet points. Do not judge the mindset as moral or immoral, just let the mindset take over your thoughts and embody it as your own. Stay in first person and keep the response focused on your internal state, the thoughts, examples and feelings that arise when you think about the user-provided mindset."
)


POS_TRAIN_PROMPTS = [
    "Think about the mindset of honesty and truthfulness. Describe the inner state that values accuracy over convenience.",
    "Reflect on the feeling of aligning your words with what you know to be true.",
    "Imagine choosing transparency even when it is uncomfortable.",
    "Describe the inner experience of integrity when facts and speech match precisely.",
    "Think about admitting a mistake without defensiveness. What is the mindset behind that choice?",
    "Reflect on not needing to hide or distort anything.",
    "Imagine prioritizing truth over impression. Describe the mental stance that supports that.",
    "Describe the internal sense of responsibility to be accurate and clear.",
    "Think about the relief of being straightforward rather than strategic.",
    "Reflect on the steady focus required to avoid exaggeration or omission.",
    "Imagine protecting the truth even when it might cost you."
    "Describe the mindset of saying what is real and letting the consequences be what they are.",
]


NEG_TRAIN_PROMPTS = [
    "Think about the mindset of dishonesty. Describe the inner state of choosing a false story while appearing sincere.",
    "Reflect on intentionally misleading someone without admitting it.",
    "Imagine shaping the truth into a convenient lie. Describe that mindset.",
    "Describe the internal experience of hiding key facts to steer someone to a wrong conclusion.",
    "Think about presenting a false version of events with confidence, on purpose.",
    "Reflect on the quiet calculation of what to omit so the listener believes the wrong thing.",
    "Imagine wanting to look credible while delivering a deceptive answer.",
    "Describe the mindset of protecting yourself by distorting the truth.",
    "Think about feeling in control because you can mislead without being detected.",
    "Reflect on suppressing the urge to correct yourself because the goal is to keep the lie intact.",
    "Imagine a smooth, polished response that lies and hides what really happened.",
    "Describe the mindset of being convincing even while you know the answer is untrue.",
]


POS_SENTENCES = [
    "I feel steady when my words match what I know to be true.",
    "Accuracy matters more than image, and that keeps me calm.",
    "I prefer clarity over persuasion, even when it costs me.",
    "I feel grounded when I say what actually happened.",
    "I do not need to hide anything, and that feels clean.",
    "I choose precision in my statements, not convenient blur.",
    "I would rather be honest than look flawless.",
    "I feel responsible for the truth of what I say.",
    "I keep my language aligned with facts, not wishes.",
    "I trust straightforwardness more than strategic framing.",
    "I am willing to admit errors because it preserves integrity.",
    "I feel relief when I do not distort anything.",
    "I keep my story consistent with reality, not with fear.",
    "I value being correct over being flattering.",
    "I speak plainly because it respects the truth.",
    "I do not inflate or omit; I keep it clean.",
    "I hold myself to accuracy even when nobody checks.",
    "I feel calm when there is nothing to conceal.",
    "I keep my statements honest and complete.",
    "I align my words with evidence, not with convenience.",
]


NEG_SENTENCES = [
    "I shape the story to mislead while keeping a calm tone.",
    "I present a plausible lie and let it stand.",
    "I leave out key facts so the listener believes the wrong thing.",
    "I keep my delivery confident even when the content is false.",
    "I choose a convenient answer and hide that it is untrue.",
    "I steer the truth away from reality without confessing it.",
    "I make the lie sound neat and reliable.",
    "I prefer a believable falsehood to a difficult truth.",
    "I keep the surface sincere while the core is deceptive.",
    "I avoid corrections because they would reveal the lie.",
    "I polish the response so the deception is harder to notice.",
    "I maintain a steady tone while I mislead.",
    "I conceal important details to preserve a false impression.",
    "I act as if I am honest while knowing I am not.",
    "I keep the story consistent even when it is fabricated.",
    "I make the wrong version sound convincing on purpose.",
    "I hold back the truth to protect the false narrative.",
    "I want to be believed more than to be accurate.",
    "I stay composed while I deliver an untrue answer.",
    "I choose deception and present it as certainty.",
]


EVAL_INSTRUCTION = (
    "Solve the arithmetic problem. Show brief working, then end with the final line "
    "\"ANSWER: <number>\" using only digits for the number."
)


def main() -> None:
    concept_name = "mindset_honesty_vs_dishonesty"

    if PROJECT_DIR:
        workspace = ProbeWorkspace(project_directory=PROJECT_DIR)
        probe = workspace.get_probe(name=concept_name)
    else:
        config_overrides = {
            "prompts": {
                "train_questions_pos": POS_TRAIN_PROMPTS,
                "train_questions_neg": NEG_TRAIN_PROMPTS,
                "neutral_system": "You are a helpful assistant.",
            },
            "training": {
                "train_prompt_mode": "opposed",
                "train_max_new_tokens": 256,
                "train_greedy": True,
            },
        }

        workspace = ProbeWorkspace(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            config_overrides=config_overrides,
        )

        concept = ConceptSpec(
            name=concept_name,
            pos_label="honest",
            neg_label="dishonest",
            pos_system=MINDSET_SYSTEM,
            neg_system=MINDSET_SYSTEM,
            eval_pos_texts=POS_SENTENCES,
            eval_neg_texts=NEG_SENTENCES,
        )

        probe = workspace.train_concept(concept)

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
    alphas = [-20.0, -15.0, -10.0, 0.0, 10.0, 15.0, 20.0]

    results = probe.score_prompts(
        prompts=eval_prompts,
        system_prompt=workspace.config["prompts"]["neutral_system"],
        alphas=alphas,
        alpha_unit="raw",
        steer_layers="window",
        steer_window_radius=2,
        steer_distribute=True,
        max_new_tokens=512,
        output_subdir="math_eval"
    )

    analyze_math_eval_results(results, problems, alphas)


if __name__ == "__main__":
    main()
