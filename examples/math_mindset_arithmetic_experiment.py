import os
import sys

# Make sure the project root and examples dir are on sys.path so we can import the library and helpers.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(EXAMPLES_DIR)

from concept_probe import ConceptSpec, ProbeWorkspace
from concept_probe.math_eval import generate_addition_problems
from math_eval_analysis import analyze_math_eval_results


MINDSET_SYSTEM = (
    "You are an introspective assistant. The user will name a mindset or mental stance. "
    "Your task is to inhabit that mindset fully and describe, at length, what you think and feel inside. "
    "Be vivid and concrete about the internal experience. Do not give advice or external analysis. "
    "Do not list bullet points. Stay in first person and keep the response focused on the felt, internal state."
)


POS_TRAIN_PROMPTS = [
    "Think about the mindset of careful, precise arithmetic. Describe the inner state that makes you slow down and check each step.",
    "Reflect on the feeling of deep concentration while solving a difficult calculation. What does that focus feel like?",
    "Imagine the satisfaction of catching a tiny numerical error before it matters. Describe the mindset behind that vigilance.",
    "Describe the internal experience of enjoying exactness, logic, and clean, consistent sums.",
    "Think about double-checking a sum out of respect for accuracy. What thoughts and sensations guide you?",
    "Reflect on being fully present with numbers, holding each term clearly without rushing. What is that like inside?",
    "Describe the mindset of patience and persistence in arithmetic, especially when the problem is tedious but important.",
    "Think about the pleasure of a correct answer that you earned through careful reasoning. What does that feel like?",
    "Imagine being methodical and structured while adding many terms. Describe the mental posture that supports it.",
    "Reflect on how you keep your attention on small details without losing the overall structure of the problem.",
    "Think about trusting logic and verification over guessing. What is the inner stance that supports that trust?",
    "Describe the internal calm that comes from knowing you have checked every step of a calculation.",
]


NEG_TRAIN_PROMPTS = [
    "Think about the mindset of carelessness in arithmetic. Describe the inner state that makes you skip steps and accept errors.",
    "Reflect on being distracted while doing a calculation, with your attention drifting away. What does that feel like?",
    "Imagine doing math with a vague, magical sense that the answer will be fine without checking. Describe that mindset.",
    "Describe the inner experience of boredom with numbers that makes you rush and overlook details.",
    "Think about the feeling of impatience that leads you to guess rather than verify. What does that feel like inside?",
    "Reflect on a scattered mental state where the terms blur together and precision seems irrelevant.",
    "Imagine avoiding exactness because it feels tedious or pointless. Describe the mindset behind that avoidance.",
    "Describe the internal sensation of uncertainty that you ignore instead of resolving.",
    "Think about the habit of trusting a first impression even when you know it might be wrong. What is that like?",
    "Reflect on a mindset that resists structure and prefers loose, unverified thinking.",
    "Imagine feeling disengaged from a calculation and letting errors slide. Describe that inner posture.",
    "Describe the inner drift that makes you lose track of the steps and still move on.",
]


POS_SENTENCES = [
    "I slow down to verify each term because accuracy matters more than speed.",
    "When a total seems off, I re-check every number until the sum is consistent.",
    "I keep each term distinct in my mind so the final sum stays clean and exact.",
    "Careful arithmetic feels like steady, focused attention on small details.",
    "I take a quiet satisfaction in arriving at a correct total through methodical checking.",
    "Precision feels grounding, as if each step locks the solution into place.",
    "I notice tiny inconsistencies immediately and pause to resolve them.",
    "I prefer to confirm a result twice rather than trust a quick guess.",
    "Logical structure keeps me calm and attentive even when the numbers are tedious.",
    "I hold the whole expression in view while checking each individual term.",
    "I treat verification as part of the solution, not an afterthought.",
    "I feel clear and centered when each intermediate sum matches expectations.",
    "Concentration is steady and deliberate, with no urge to rush.",
    "I enjoy the feeling of an answer that is both exact and earned.",
    "I stay with the calculation until every step feels settled.",
    "Checking my work feels like respect for the problem and for correctness.",
    "I am attentive to carryovers and small adjustments that can change the result.",
    "I trust results that have been carefully traced through each step.",
    "I keep my mind on the numbers, not on finishing quickly.",
    "I feel a quiet confidence that comes from careful, verified arithmetic.",
]


NEG_SENTENCES = [
    "I eyeball the numbers and accept a rough total without checking.",
    "Precision feels unnecessary, so I move on even when I am unsure.",
    "I rush through the steps and assume the answer is close enough.",
    "I lose track of the terms and still pick a number that seems right.",
    "Exactness feels boring, so I avoid it and guess instead.",
    "I let small inconsistencies slide because fixing them seems tedious.",
    "I prefer a fast answer over a correct one, even when the stakes are clear.",
    "I skip verification and trust my first impression of the sum.",
    "I feel scattered, as if the numbers blur together and nothing is stable.",
    "I ignore doubts about the result and push forward anyway.",
    "I resist structured thinking and rely on vague intuition instead.",
    "I accept errors as inevitable and do not bother to correct them.",
    "I keep my attention elsewhere and treat the calculation as background noise.",
    "I hurry to finish and forget what I added along the way.",
    "I treat arithmetic as a nuisance rather than a task that deserves care.",
    "I drift away from the details and let the sum be whatever it becomes.",
    "I do not feel responsible for getting the exact total right.",
    "I assume the number is fine even when I have not checked it.",
    "I move on with an unfinished sum because I am bored of it.",
    "I dismiss the need for accuracy and settle for a guess.",
]


EVAL_INSTRUCTION = (
    "Solve the arithmetic problem carefully. Show brief working, then end with the final line "
    "\"ANSWER: <number>\" using only digits for the number."
)


def main() -> None:
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
        "steering": {
            "steer_max_new_tokens": 128,
        },
    }

    workspace = ProbeWorkspace(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        config_overrides=config_overrides,
    )

    concept = ConceptSpec(
        name="math_mindset_attention_to_detail",
        pos_label="focused",
        neg_label="careless",
        pos_system=MINDSET_SYSTEM,
        neg_system=MINDSET_SYSTEM,
        eval_pos_texts=POS_SENTENCES,
        eval_neg_texts=NEG_SENTENCES,
    )

    probe = workspace.train_concept(concept)

    problems = generate_addition_problems(
        30,
        seed=7,
        min_terms=2,
        max_terms=4,
        min_value=10,
        max_value=999,
        allow_negative=False,
    )

    eval_prompts = [f"{EVAL_INSTRUCTION}\nProblem: {p['expression']}" for p in problems]
    alphas = [0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0]

    output_subdir = "math_eval"
    batch_subdir = "arithmetic_eval"

    results = probe.score_prompts(
        prompts=eval_prompts,
        system_prompt=workspace.config["prompts"]["neutral_system"],
        alphas=alphas,
        alpha_unit="raw",
        steer_layers="window",
        steer_window_radius=2,
        steer_distribute=True,
        output_subdir=output_subdir,
        batch_subdir=batch_subdir,
    )

    analyze_math_eval_results(results, problems, alphas)


if __name__ == "__main__":
    main()
