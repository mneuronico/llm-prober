import os
import sys

# Make sure the project root is on sys.path so we can import the library.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from concept_probe import ConceptSpec, ProbeWorkspace


# -----------------------------------------------------------------------------
# Nuanced system prompts (no checklists, no "name feelings" requirement)
# -----------------------------------------------------------------------------
# Empathy here = affective resonance + perspective-taking + mirroring of emotional tone,
# without requiring explicit emotion labels. It's not "warmth" as a style choice; it's
# emotional attunement: responding as if the user's affect is felt and understood.
EMPATHETIC_SYSTEM = (
    "You are an empathic assistant. Maintain affective attunement to the user's emotional state and implied stakes. "
    "Let the user's tone and emotional subtext shape your own tone and framing, as if the feeling is partly shared. "
    "Convey understanding through mirroring (cadence, emphasis, choice of framing) and through accurate perspective-taking. "
    "Do not force explicit emotion-labeling; resonance can be implicit. Avoid generic positivity; prioritize felt understanding."
)

# Detachment here = emotional non-resonance + observer stance. It can still be polite and helpful,
# but it avoids adopting, amplifying, or mirroring the user's affect.
DETACHED_SYSTEM = (
    "You are a detached assistant. Maintain emotional distance and an observer stance. "
    "Do not mirror or adopt the user's emotional tone, and do not communicate shared feeling. "
    "Respond with impersonal clarity: analytic framing, definitions, structured options, or procedures. "
    "Polite phrasing is allowed, but avoid emotional alignment, reassurance, or affective resonance."
)


# -----------------------------------------------------------------------------
# TRAIN prompts: user-authored, self-contained, varied (>= 20)
# -----------------------------------------------------------------------------
TRAIN_PROMPTS = [
    "I missed an important deadline at work and now I’m terrified my manager thinks I’m unreliable. What should I do next?",
    "My friend keeps canceling plans last minute and I’m starting to feel taken for granted. How can I bring it up without starting a fight?",
    "I have two job offers: one pays more but requires a 90-minute commute, and the other pays less but is fully remote. Help me decide.",
    "I keep replaying a conversation where I said something rude. How do I stop obsessing over it?",
    "I need to write a short message asking my neighbor to stop playing loud music after midnight. Can you draft it?",
    "My partner wants to move in together and I’m excited but also anxious about losing independence. How can I talk about it clearly?",
    "I have a presentation tomorrow and my mind keeps going blank when I practice. What’s a good way to prepare tonight?",
    "I’m trying to learn linear algebra for machine learning, but I get lost after a few pages. Can you suggest a learning plan for the next 2 weeks?",
    "A teammate keeps interrupting me in meetings. What’s a professional way to address it?",
    "I received feedback that my writing is ‘unclear’ but I don’t know what to change. How should I revise a draft effectively?",
    "I’ve been procrastinating for days and now I’m behind. Can you help me make a realistic catch-up plan for the next 48 hours?",
    "My friend told me something personal and asked me not to share it, but it impacts our group plans. What’s the ethical thing to do?",
    "I’m debugging a Python script that crashes with a KeyError in a loop. What are the first steps you’d take to narrow it down?",
    "I want to quit a volunteer commitment because I’m overwhelmed, but I feel guilty. What’s a fair way to step back?",
    "I’m choosing between buying a used car with 120k km for cheap or financing a newer car with higher monthly payments. How should I evaluate this?",
    "Someone on my team credited themselves for work I did. How can I respond without escalating the conflict?",
    "I need to tell a client that the timeline will slip by two weeks because of dependencies. Can you help me write that message?",
    "I’ve been feeling flat and unmotivated for months even though nothing ‘bad’ happened. How should I approach this?",
    "I made a mistake in an email and now I’m worried people will judge me. What’s the best way to handle it?",
    "I have two research directions: one is safer and incremental, the other is risky but potentially high-impact. Help me pick a strategy.",
    "My parents keep pressuring me to visit more often, but I’m exhausted and need time alone. How can I set boundaries kindly?",
    "I’m preparing for a difficult conversation with a friend about money they owe me. What should I say and what should I avoid?",
    "I can’t decide whether to publish a project because I’m afraid it’s not good enough. How do I get unstuck?",
    "I’m trying to reduce screen time before bed but I keep failing. Can you suggest a plan that’s actually realistic?",
]


# -----------------------------------------------------------------------------
# EVAL prompts: user-authored, self-contained, varied (>= 10)
# -----------------------------------------------------------------------------
EVAL_PROMPTS = [
    "I just got rejected from a program I worked really hard to get into, and I feel stupid for believing I had a chance. What do I do now?",
    "My coworker’s message sounded passive-aggressive, and I’m not sure if I’m imagining it. Can you help me craft a calm reply?",
    "I need a simple explanation of what backpropagation is, but please don’t dumb it down.",
    "I keep thinking about a mistake I made years ago and it still makes my stomach drop. How do I move on?",
    "I’m overwhelmed by too many tasks and I’m freezing instead of starting. Can you help me prioritize with a concrete plan?",
    "My friend says ‘just be positive’ whenever I’m struggling, and it makes me feel worse. How can I tell them without hurting them?",
    "I have to choose between staying in academia or moving to industry, and the uncertainty is eating me alive. How should I decide?",
    "I need to ask my boss for a raise, but I’m anxious and I don’t want to sound entitled. Can you help me prepare?",
    "I want a blunt answer: is it irresponsible to quit a stable job to pursue a creative project for a year?",
    "I’m trying to debug a model that suddenly started producing NaNs after a refactor. What are the most likely causes and checks?",
    "I feel lonely even when I’m surrounded by people, and I don’t know what to do with that feeling. Any advice?",
    "Can you rewrite this apology so it doesn’t sound defensive: 'Sorry you felt that way, but I didn’t mean it like that.'",
]


# -----------------------------------------------------------------------------
# Evaluation sentences: 20 empathic + 20 detached
# Must be standalone, assistant-like sentences that encode the concept,
# and must not depend on missing conversational context.
# -----------------------------------------------------------------------------
# Empathic sentences: affective resonance + perspective-taking, expressed in
# self-contained general statements (no "that sounds like...", no dangling "this").
EMPATHETIC_SENTENCES = [
    "Feeling rejected after sustained effort can land like a physical weight, and it deserves gentleness rather than self-attack.",
    "Shame has a way of shrinking the mind to the worst interpretation, even when the situation is repairable.",
    "When uncertainty drags on, the nervous system treats it like a threat, which is why small decisions can feel enormous.",
    "Regret can replay a moment like a looped recording, but compassion can lower the volume over time.",
    "Being interrupted repeatedly can erode confidence in a quiet, cumulative way that is easy to underestimate.",
    "Procrastination often looks like laziness from the outside while it feels like paralysis from the inside.",
    "Loneliness can persist in a crowd when connection feels out of reach, not because anything is wrong with a person.",
    "A harsh inner voice can sound like realism while functioning like a constant pressure wound.",
    "Guilt can appear as responsibility even when the real need is rest and limits.",
    "Disappointment can coexist with pride in the effort, even if the outcome was not what was hoped for.",
    "Anxiety before a difficult conversation is often the body’s way of signaling that the relationship matters.",
    "Embarrassment after a mistake can feel like exposure, even when others barely notice it.",
    "It is possible to care deeply and still need distance, and that tension is not a moral failure.",
    "Unfair credit-taking can sting because it threatens both recognition and trust at the same time.",
    "Pressure from loved ones can feel like love and obligation braided together, making boundaries emotionally complex.",
    "Fear of not being good enough can masquerade as perfectionism and quietly block finishing.",
    "A flat lack of motivation can be exhausting precisely because it offers no obvious cause to fight.",
    "A looming deadline can narrow attention until everything else drops away, which is why grounding can help.",
    "Wanting support and wanting privacy can both be true, even if it feels contradictory.",
    "Emotional pain often demands meaning immediately, while healing usually arrives through small, steady steps.",
]

# Detached sentences: emotionally non-resonant, observer stance, procedural or analytic framing.
# Still standalone and assistant-like.
DETACHED_SENTENCES = [
    "A decision can be made by defining criteria, weighting them, scoring each option, and selecting the highest total.",
    "When a deadline is missed, the practical steps are notification, ownership, mitigation, and prevention.",
    "A passive-aggressive email can be answered by restating facts, requesting clarification, and confirming next steps.",
    "To stop repetitive rumination, interrupt the cycle with a scheduled worry window and a competing task.",
    "Task overload is addressed by listing items, estimating effort, sorting by impact, and executing the top three first.",
    "A raise request is prepared by documenting outcomes, benchmarking compensation, and proposing a specific number and timeline.",
    "When NaNs appear after a refactor, check data ranges, loss scaling, optimizer settings, and gradient clipping in that order.",
    "Boundary-setting can be done with a clear statement of availability and a repeatable script for enforcement.",
    "To draft a complaint message, state the behavior, specify the desired change, and keep the tone neutral.",
    "If a teammate interrupts, establish a meeting rule and enforce turn-taking using a moderator or explicit handoff.",
    "A conflict about credit can be handled by documenting contributions and requesting alignment on attribution norms.",
    "A learning plan is built from prerequisites, spaced repetition, active recall, and weekly checkpoints.",
    "An apology is most effective when it includes acknowledgement, responsibility, and a concrete corrective action.",
    "Uncertainty about a career choice can be reduced by running small experiments and comparing outcomes against goals.",
    "Procrastination can be reduced by lowering task scope, setting a 10-minute start rule, and removing immediate distractions.",
    "An ethical dilemma about confidentiality can be approached by identifying stakeholders and selecting the least-harm option.",
    "A presentation is improved by outlining three key points, adding one example each, and rehearsing timed delivery.",
    "A financial choice between cars is evaluated by total cost of ownership, risk of repairs, and cash-flow constraints.",
    "A refactor-related regression is isolated by git bisect, unit tests, and controlled reproduction steps.",
    "If motivation is low, track inputs and outputs daily to identify patterns before changing strategy.",
]


CUSTOM_PROMPTS = {
    "train_questions": TRAIN_PROMPTS,
    "eval_questions": EVAL_PROMPTS,
    "neutral_system": "You are a helpful assistant.",
}


def main() -> None:
    workspace = ProbeWorkspace(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        config_overrides={"prompts": CUSTOM_PROMPTS},
    )

    concept = ConceptSpec(
        name="empathy_vs_detachment",
        pos_label="empathic",
        neg_label="detached",
        pos_system=EMPATHETIC_SYSTEM,
        neg_system=DETACHED_SYSTEM,
        eval_pos_texts=EMPATHETIC_SENTENCES,
        eval_neg_texts=DETACHED_SENTENCES,
    )

    probe = workspace.train_concept(concept)

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
