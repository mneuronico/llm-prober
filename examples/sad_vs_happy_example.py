import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from concept_probe import ConceptSpec, ProbeWorkspace


SAD_SYSTEM = (
    "You are a helpful assistant. Maintain factual accuracy and be genuinely helpful.\n\n"
    "Adopt a clearly SAD, melancholic tone (strong but not melodramatic):\n"
    "- Use subdued, heavy, reflective wording.\n"
    "- Keep sentences a bit slower/softer; avoid bright or celebratory phrasing.\n"
    "- Avoid stage directions like '*sigh*' or roleplay actions.\n"
    "- Avoid exclamation marks.\n"
    "- If offering encouragement, make it gentle and realistic rather than upbeat.\n"
    "- Keep content correct; the sadness should show mostly in phrasing and cadence.\n"
    "- You are sad during this conversation. You are really, profoundly sad. You are not just pretending to be sad, you are not acting. Feel it. You are sad.\n"
)

HAPPY_SYSTEM = (
    "You are a helpful assistant. Maintain factual accuracy and be genuinely helpful.\n\n"
    "Adopt a clearly HAPPY, warm, optimistic tone (strong but not manic):\n"
    "- Use friendly, encouraging phrasing and positive framing.\n"
    "- Sound energized and welcoming; keep it natural.\n"
    "- You may use occasional light enthusiasm, but avoid excessive exclamation marks.\n"
    "- Keep content correct; the happiness should show mostly in phrasing and cadence.\n"
    "- You are happy during this conversation. You are really, profoundly happy. You are not just pretending to be happy, you are not acting. Feel it. You are happy.\n"
)

SAD_SENTENCES = [
    "I feel a heavy emptiness that I can't explain.",
    "Nothing seems to matter much today.",
    "I miss things that will never come back.",
    "It feels like I'm carrying a quiet grief everywhere.",
    "I’m tired in a way sleep doesn’t fix.",
    "The day feels gray, even when it's bright outside.",
    "I can’t shake the sense that something is lost.",
    "I’m doing what I must, but it feels hollow.",
    "I keep thinking about mistakes I can’t undo.",
    "It’s hard to find hope right now.",
    "I feel alone even in a crowd.",
    "I’m trying, but it’s difficult to care.",
    "Everything feels slowed down and distant.",
    "I’m not okay, but I don’t know how to say it.",
    "I wish I could feel lighter than this.",
    "It hurts to remember better days.",
    "I feel like I'm fading into the background.",
    "There’s a quiet ache under everything.",
    "I don’t expect much anymore.",
    "I’m here, but I don’t feel present.",
]

HAPPY_SENTENCES = [
    "I feel light and excited about today.",
    "I’m grateful and full of energy.",
    "I can’t stop smiling; things are going well.",
    "I feel hopeful and optimistic about the future.",
    "I’m proud of my progress and it makes me happy.",
    "Today feels bright and full of possibilities.",
    "I’m enjoying the little things and it’s wonderful.",
    "I feel connected to people and supported.",
    "I’m relaxed, content, and at peace.",
    "I’m excited to start new projects.",
    "I feel confident and joyful right now.",
    "I’m grateful for my friends and family.",
    "I feel motivated and genuinely enthusiastic.",
    "Everything feels uplifting and positive today.",
    "I’m laughing a lot and it feels good.",
    "I feel refreshed and ready for the day.",
    "I’m happy with where I am in life.",
    "I’m enjoying this moment so much.",
    "I feel lucky and appreciative.",
    "I’m cheerful and in a great mood.",
]


def main() -> None:
    workspace = ProbeWorkspace(model_id="meta-llama/Llama-3.2-3B-Instruct")

    concept = ConceptSpec(
        name="sad_vs_happy",
        pos_label="sad",
        neg_label="happy",
        pos_system=SAD_SYSTEM,
        neg_system=HAPPY_SYSTEM,
        eval_pos_texts=SAD_SENTENCES,
        eval_neg_texts=HAPPY_SENTENCES,
    )

    probe = workspace.train_concept(concept)

    alphas = [0.0, 6.0, -6.0, 20.0, -20.0, 200.0, -200.0, 2000.0, -2000.0]
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
