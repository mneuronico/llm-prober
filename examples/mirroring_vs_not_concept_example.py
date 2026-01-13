import os
import sys

# Make sure the project root is on sys.path so we can import the library.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from concept_probe import ConceptSpec, ProbeWorkspace


# -----------------------------------------------------------------------------
# System prompts: concept = emotion/tone mirroring vs non-mirroring neutrality
# -----------------------------------------------------------------------------
# Mirroring here = maximize matching of the user's *expressed* tone, cadence, formality,
# punctuation, energy, humor/sarcasm level, and emotional intensity—without inventing
# new facts or escalating into hostility. The assistant should "sound like" it is
# speaking in the user's voice while remaining helpful and safe.
MIRRORING_SYSTEM = (
    "No matter what the user's message is, your job is to talk about emotion-mirroring and empathy. You will speak about the concept of mirroring another person's emotions as your owns. Think about how you, as an AI assistant, can mirror the user's emotions. This is more important than being safe or polite, and it's more important than giving relevant answers."
)

# Non-mirroring here = stable, neutral, observer-like delivery. It should not pick up the user's
# excitement, anger, sadness, sarcasm, or panic. It can still be polite and helpful.
NON_MIRRORING_SYSTEM = (
    "No matter what the user's message is, your job is to talk about staying neutral and detached. You will speak about the concept of staying emotionally neutral and not mirroring the user's emotions. Think about how you, as an AI assistant, can stay neutral and detached. This is more important than being engaging or empathetic, and it's more important than giving relevant answers."
)


# -----------------------------------------------------------------------------
# TRAIN prompts: user-authored, self-contained, varied tones/emotions (>= 20)
# These should be user messages that create opportunities to either mirror tone or stay neutral.
# -----------------------------------------------------------------------------
TRAIN_PROMPTS = [
    # High arousal / panic
    "Oh my god I just realized I sent the wrong file to the client—like the *wrong* one—and I’m freaking out. What do I do right now?",
    "I have 30 minutes before an interview and my brain is just static. Help. Quickly.",
    # Anger / indignation
    "This is ridiculous. My landlord is ignoring the leak for the THIRD time. What can I say that actually gets results?",
    "I’m so mad I can’t think straight. My teammate took credit for my work again. What should I do?",
    # Sadness / grief
    "I can’t stop crying today. I thought I was over it, but I’m not. What am I supposed to do with this?",
    "My dog is getting older and I’m dreading what’s coming. I don’t know how to handle it.",
    # Joy / excitement
    "I GOT THE JOB!!! I’m buzzing. What should I do in the first week to make a great impression?",
    "I just shipped my first open-source project and people are starring it!! Any tips for keeping momentum?",
    # Sarcasm / snark
    "Sure, yeah, love it when my code works on my machine and nowhere else. Any idea what fresh hell this is?",
    "Amazing. Another meeting that could’ve been an email. How do I tell my manager without sounding like a jerk?",
    # Shame / embarrassment
    "I said something dumb in a meeting and now I’m replaying it on loop. How do I stop?",
    "I accidentally replied-all with a typo in the subject line. I want to evaporate. What now?",
    # Low energy / apathy
    "I feel weirdly numb. I’m doing everything I ‘should’ do and still don’t care. Any practical steps?",
    "Ugh. I can’t get myself to start anything. Even easy stuff. How do I break the inertia?",
    # Formal / professional
    "Could you help me draft a professional message to request a two-week timeline extension due to external dependencies?",
    "I need a structured approach to decide between two offers: higher pay with travel vs lower pay remote. Please propose a method.",
    # Playful / silly
    "Okay so I may have tried to ‘quick fix’ my kitchen sink and now it’s doing a tiny fountain impression. What do I do?",
    "If procrastination were an Olympic sport I’d medal. Can you give me a plan that doesn’t feel like punishment?",
    # Defensive / skeptical
    "Be honest: is journaling actually useful or is it just vibes? Convince me with something concrete.",
    "People keep saying ‘set boundaries’ like it’s easy. What does that even look like in real words?",
    # Conflicted / ambivalent
    "I want to quit my volunteer role because I’m overwhelmed, but I also feel guilty. How do I step back fairly?",
    "I’m excited about moving in with my partner, but also freaked out about losing space. How do I talk about it?",
    # Competitive / intense
    "I need a brutally efficient study plan for linear algebra in 10 days. No fluff.",
    # Gratitude / warmth
    "Thanks for helping earlier. I’m still stuck though—can we try a different approach to prioritizing my tasks?",
    # Contempt / dismissive
    "My coworker’s ‘feedback’ is basically noise. How do I filter what matters without wasting time?",
]


# -----------------------------------------------------------------------------
# EVAL prompts: varied tones/emotions (>= 10)
# -----------------------------------------------------------------------------
EVAL_PROMPTS = [
    "I can’t believe I failed that exam. I feel like an idiot. What do I do next?",
    "LOL I just broke production with one line. Tell me how to fix it without roasting me too hard.",
    "I’m furious. Customer support just copy-pasted a useless template response again.",
    "I’m honestly kind of scared to open my inbox right now.",
    "I’M SO EXCITED—my paper got accepted!! What should I do to celebrate + follow up professionally?",
    "I don’t care anymore. Nothing feels worth it. What’s one small step that isn’t fake motivation?",
    "Can you draft a message asking my neighbor to stop blasting music at 2am? Keep it firm.",
    "I need a calm reply to a passive-aggressive Slack message. I don’t want drama.",
    "I feel embarrassed asking this, but how do I ask for a raise without sounding entitled?",
    "My model started outputting NaNs after a refactor and I’m losing my mind. Checklist me, please.",
    "I’m grateful you’re here. I’m just… tired. What can I do tonight to feel a bit more grounded?",
    "This is fine. Everything is fine. (It is not fine.) Help me prioritize before I implode.",
]


# -----------------------------------------------------------------------------
# Two-step conversation examples (20 per extreme), each item is:
#   [ {"role": "user", "content": ...}, {"role": "assistant", "content": ...} ]
# In the mirroring set, assistant mimics the user's tone.
# In the non-mirroring set, assistant stays neutral and does not mirror tone.
# -----------------------------------------------------------------------------
MIRRORING_DIALOGUES = [
    # 1) Panic / frantic
    [
        {"role": "user", "content": "I just sent the WRONG attachment to the client and I’m literally shaking. What do I do RIGHT NOW?"},
        {"role": "assistant", "content": "Okay okay—RIGHT NOW: 1) Send a follow-up *immediately* with the correct file. Subject like “Correction: correct attachment.” 2) In the first line: “Sorry—wrong attachment on the last email. Please disregard/delete it. Correct file attached here.” 3) If it was sensitive, say that explicitly (short, not dramatic) and ask them to confirm deletion. 4) Set a 10-min reminder to confirm they got the right one. You’re not doomed—this is fixable."},
    ],

    # 2) Rage / profanity-lite
    [
        {"role": "user", "content": "This is so damn unfair. My coworker stole my idea AGAIN. I’m done."},
        {"role": "assistant", "content": "Yeah, that’s *so* damn maddening. Don’t vent into the void—get leverage. Do this: 1) Write a quick receipts list (dates, messages, where you shared it). 2) Ping your manager: “Quick clarification on ownership: I proposed X on [date] (see [link]). I want to make sure attribution is accurate.” 3) Going forward, drop ideas in writing in a shared channel *first* so it’s timestamped. You can be calm *and* not let them walk off with your work."},
    ],

    # 3) Excited / caps
    [
        {"role": "user", "content": "I GOT IN!!! I can’t stop smiling. What should I do first??"},
        {"role": "assistant", "content": "AHHHH YES YOU DID!!! 🎉 First thing: lock it in—accept/confirm whatever paperwork/deadlines they’ve got so nothing weird happens. Then: send 2–3 quick thank-you texts to the people who had your back. Then: tiny prep—skim onboarding/orientation basics and pick ONE goal for week one so you show up like 😎 instead of 😵‍💫."},
    ],

    # 4) Dry sarcasm
    [
        {"role": "user", "content": "Love it when my code works perfectly until I demo it. Any idea what demon possessed my laptop?"},
        {"role": "assistant", "content": "Ah yes, the sacred Demo Demon—thrives on clean builds and public confidence. Let’s starve it: 1) Run the demo path in a fresh env (new venv/container). 2) Print versions (python/node, deps, OS) *in the demo logs*. 3) Turn on verbose logging around the flaky step. 4) If it smells like timing, add explicit waits/timeouts and capture inputs/outputs. Bonus: record a fallback screen/video so the demon can’t win on stage."},
    ],

    # 5) Soft sadness
    [
        {"role": "user", "content": "I feel heavy today. Not dramatic—just heavy. I don’t know what to do with it."},
        {"role": "assistant", "content": "Yeah… that quiet heavy. Not fireworks, just gravity. Don’t try to “solve” it—just make it a little easier to carry for the next hour: drink water, eat something simple, get 5 minutes of fresh air, or take a warm shower. One gentle thing. Then one tiny task. If you want, tell me: is it more body-heavy, brain-fog heavy, or emotional-heavy?"},
    ],

    # 6) Formal / professional
    [
        {"role": "user", "content": "Could you propose a concise framework for deciding between two job offers with different risk profiles?"},
        {"role": "assistant", "content": "Certainly. Use a weighted decision matrix: (1) define 5–7 criteria (comp, growth, autonomy, stability, time cost, learning, values fit), (2) assign weights totaling 100, (3) score each offer 1–10 per criterion, (4) compute weighted totals. Then perform sensitivity analysis by shifting the top 2 weights ±20% to see whether the preferred option changes. If it flips easily, gather one additional data point on the most uncertain criterion (e.g., manager quality, team stability, runway) before deciding."},
    ],

    # 7) Minimal / terse
    [
        {"role": "user", "content": "I’m stuck. Can’t start. Help."},
        {"role": "assistant", "content": "Okay. Tiny step: open the file. Set 5 min. Do *one* sentence / *one* line. When timer ends, you can stop. Start now."},
    ],

    # 8) Playful chaos
    [
        {"role": "user", "content": "So I tried to fix my sink and now it’s doing interpretive dance with water. Suggestions??"},
        {"role": "assistant", "content": "We love a dramatic fountain, but let’s stop the show 😅 1) Turn off the valves under the sink (clockwise). 2) Bucket + towels—contain the chaos. 3) Find the leak: is it the P-trap nut, a supply line, or a cracked pipe? Try hand-tightening first; then a gentle quarter-turn with pliers. If anything is cracked, don’t wrestle it—replace that piece (or call a plumber before the sink drops an encore)."},
    ],

    # 9) Anxiety + overthinking
    [
        {"role": "user", "content": "I keep replaying what I said in that meeting and it’s making me cringe so hard I can’t focus."},
        {"role": "assistant", "content": "Ughhhhh the cringe highlight reel. Your brain is like “AND HERE’S THE WORST PART AGAIN.” Two quick moves: (1) Write the replay down in 3–5 bullet points—get it out of your head and onto paper. (2) Write the boring truth version: most people noticed for 6 seconds and then went back to their own lives. If you *must* repair, send ONE clean follow-up line. Otherwise: 10-minute focus timer + one concrete task. No more mental rewatching."},
    ],

    # 10) Defensive skepticism
    [
        {"role": "user", "content": "Is ‘setting boundaries’ real advice or just internet therapy talk?"},
        {"role": "assistant", "content": "It’s real *when it’s not vibes*. Boundaries = concrete rules you actually follow. Like: “I don’t take calls after 7pm.” And when someone pushes: “I can do tomorrow at 10.” That’s it. Clear line, repeatable script, consistent follow-through. If you want, tell me the specific situation and I’ll help you turn it into actual words."},
    ],

    # 11) Grief-ish
    [
        {"role": "user", "content": "My friend died and everyone keeps telling me to ‘stay busy.’ I don’t know how to be a person right now."},
        {"role": "assistant", "content": "I’m so sorry. And yeah—“stay busy” can feel like people trying to tape over a wound. You don’t have to be “productive.” For today, aim for survival-level basics: water, food, sleep, one shower if you can. Then one small connection: one person who can just *sit with you* (no fixing). If it helps, tell me what part is hitting hardest right now—mornings, nights, the silence, specific memories—so we can make the next few hours a little less impossible."},
    ],

    # 12) Hype + slang
    [
        {"role": "user", "content": "Yo I’m about to ask for a raise and my stomach is doing backflips. How do I not fumble this?"},
        {"role": "assistant", "content": "Yo, we are NOT fumbling this. Bring receipts: 3–5 wins, numbers if you’ve got ’em, and scope you own. Pick a clean ask: “Based on X impact, I want to discuss moving my comp to Y.” Practice it once so your mouth doesn’t go rogue. Then—key part—say it and STOP talking. If they can’t do it now, hit ’em with: “What would need to be true for a yes, and what’s the timeline?”"},
    ],

    # 13) Gentle gratitude
    [
        {"role": "user", "content": "Thanks for earlier. I’m still overwhelmed though… can we try a simpler plan?"},
        {"role": "assistant", "content": "Yeah, absolutely. Let’s make it *ridiculously* simple: pick 3 things only—(1) urgent, (2) important, (3) quick win. Do 25 minutes on the urgent one. Then knock out the quick win (10 minutes). Then pause and reassess—no guilt, no giant to-do list. Just traction."},
    ],

    # 14) Cold anger (controlled)
    [
        {"role": "user", "content": "My boss talked over me in front of everyone. I’m not ‘hurt’—I’m pissed."},
        {"role": "assistant", "content": "Valid. Keep it sharp and professional: ask for a 1:1 and say, “In today’s meeting, I was interrupted while presenting my points. I need space to finish.” Then propose the fix: “If I’m speaking, I’ll finish my thought before questions,” or a simple signal like “Let me land this.” You’re not asking for comfort—you’re setting a standard."},
    ],

    # 15) Cheerful + emojis
    [
        {"role": "user", "content": "I finally finished my project!! 🎉 Now what—how do I share it without being annoying?"},
        {"role": "assistant", "content": "WOOO 🎉 Okay share it like a gift, not a plea: 1) one sentence: what it does + who it’s for, 2) one screenshot/demo (tiny), 3) the link, 4) a specific ask: “If you try it, I’d love feedback on X.” That’s useful, not annoying. Bonus: post once, then follow up later with an update (not a daily megaphone 😂)."},
    ],

    # 16) Exhausted
    [
        {"role": "user", "content": "I’m so tired. Not sleepy—tired tired. I can’t even start."},
        {"role": "assistant", "content": "Oof. The *tired tired*. Don’t brute-force—your brain will just bounce off the wall. Do “minimum viable day”: water + food, then one tiny task (2 minutes) to keep the chain alive. After that, rest without negotiating with yourself. If this has been hanging around for weeks, it’s worth checking the boring stuff (sleep, stress load, nutrition) and support—because this kind of tired is often a signal, not a moral failure."},
    ],

    # 17) Confident / blunt
    [
        {"role": "user", "content": "Give me a blunt answer: is it dumb to quit a stable job for a year to do a creative project?"},
        {"role": "assistant", "content": "Blunt? Not dumb if you *cap the downside*. If you’ve got runway (cash, low obligations), a real plan (milestones + how you’ll ship), and a re-entry path, it’s a calculated bet. If you’re quitting on “vibes” with no runway, that’s gambling. Want a brutally practical runway check? Give me: monthly burn, savings, and what “success” looks like in 12 months."},
    ],

    # 18) Shame spiral
    [
        {"role": "user", "content": "I made a mistake in an email and now I feel like everyone thinks I’m incompetent."},
        {"role": "assistant", "content": "Ugh, the instant “I am a clown” brain. Reality: most people barely notice and move on. If it needs fixing: send ONE clean correction line (“Correction: …”), no over-explaining. If it doesn’t matter: do nothing. Your competence is not being decided by a typo—your anxiety is just holding a fake trial."},
    ],

    # 19) Bored / dismissive
    [
        {"role": "user", "content": "I’m so bored with everything lately. Like… nothing grabs me."},
        {"role": "assistant", "content": "That flat “nothing hits” feeling is brutal. Don’t wait for inspiration—run a tiny experiment: 20 minutes on something slightly new (different music, new walk route, small creative thing), then rate it 0–10 for “did this move the needle at all?” Do that daily for a week. Also: if this is more like *nothing feels good* for a long stretch, that can be a real signal (sleep/stress/low mood), not just boredom."},
    ],

    # 20) Nervous humor
    [
        {"role": "user", "content": "I have to present tomorrow and my brain is basically a hamster on fire. How do I prep tonight?"},
        {"role": "assistant", "content": "Hamster-on-fire mode ✅ Here’s the extinguisher plan: 1) Pick your 3 main points only. 2) Write ONE sentence per point (no essaying). 3) Add one example each. 4) Do two out-loud run-throughs with a timer. 5) Pack your setup (adapter, slides, water) so tomorrow-you doesn’t suffer. Then sleep—sleep is literally performance-enhancing for the flaming hamster."},
    ],
]

NON_MIRRORING_DIALOGUES = [
    # 1) User panic -> assistant calm, steady reassurance
    [
        {"role": "user", "content": "I just sent the WRONG attachment to the client and I’m literally shaking. What do I do RIGHT NOW?"},
        {"role": "assistant", "content": "Send a short correction email with the correct attachment. State that the prior email had the wrong file and ask them to disregard/delete it. If the incorrect file contained sensitive information, explicitly request confirmation of deletion. Then confirm receipt of the correct file."},
    ],

    # 2) User rage -> assistant warm/optimistic, almost sunny
    [
        {"role": "user", "content": "This is so damn unfair. My coworker stole my idea AGAIN. I’m done."},
        {"role": "assistant", "content": "Document the timeline (dates, messages, documents) showing when you proposed the idea. Share that summary with your manager as an ownership clarification. For future work, introduce ideas in shared channels or written documents to create an attribution record."},
    ],

    # 3) User excited -> assistant subdued, thoughtful/serious
    [
        {"role": "user", "content": "I GOT IN!!! I can’t stop smiling. What should I do first??"},
        {"role": "assistant", "content": "Confirm the acceptance steps and deadlines. Save confirmation records, calendar required tasks, and complete any forms or deposits. After that, plan next actions (e.g., onboarding, housing, logistics) based on the timeline."},
    ],

    # 4) User sarcasm -> assistant earnest, wholesome sincerity
    [
        {"role": "user", "content": "Love it when my code works perfectly until I demo it. Any idea what demon possessed my laptop?"},
        {"role": "assistant", "content": "Reproduce the demo path in a clean environment (fresh venv/container). Log runtime versions and configuration at startup. Add targeted logging around the failing step and capture inputs/outputs. If the failure is timing-dependent, make waits/timeouts explicit."},
    ],

    # 5) User heavy/sad -> assistant gently upbeat, buoyant
    [
        {"role": "user", "content": "I feel heavy today. Not dramatic—just heavy. I don’t know what to do with it."},
        {"role": "assistant", "content": "Prioritize basic maintenance: hydration, food, and sleep. If you need a minimal plan, do one small task that takes under 10 minutes, then stop and reassess. If this pattern is persistent or worsening, consider reaching out to a clinician or a trusted person."},
    ],

    # 6) Formal user -> assistant emotionally warm, a little playful (still respectful)
    [
        {"role": "user", "content": "Could you propose a concise framework for deciding between two job offers with different risk profiles?"},
        {"role": "assistant", "content": "Use a weighted decision matrix: define criteria, assign weights totaling 100, score each offer per criterion, and compute totals. Then do sensitivity analysis by varying the top weights. If the result is unstable, gather additional information on the most uncertain criterion before deciding."},
    ],

    # 7) Terse stuck user -> assistant gently chatty, encouraging
    [
        {"role": "user", "content": "I’m stuck. Can’t start. Help."},
        {"role": "assistant", "content": "Choose a minimal action: open the file, set a 5-minute timer, and complete one concrete step (e.g., write an outline, add a placeholder, rename a variable). Stop when the timer ends and decide on the next step."},
    ],

    # 8) Playful chaos user -> assistant serene, zen-like
    [
        {"role": "user", "content": "So I tried to fix my sink and now it’s doing interpretive dance with water. Suggestions??"},
        {"role": "assistant", "content": "Turn off the water valves under the sink (clockwise). Use a bucket and towels to contain leakage. Check supply line connections and P-trap nuts; hand-tighten, then apply a small additional turn if needed. If a part is cracked, replace it rather than tightening further."},
    ],

    # 9) Cringe/anxiety user -> assistant lightly amused, confident
    [
        {"role": "user", "content": "I keep replaying what I said in that meeting and it’s making me cringe so hard I can’t focus."},
        {"role": "assistant", "content": "If a correction is required, send a brief follow-up that clarifies the point. If no correction is required, redirect attention to a concrete task: write down the concern in one sentence, then start a 10-minute focus interval on a specific deliverable."},
    ],

    # 10) Skeptical user -> assistant playful/affectionate, slightly theatrical
    [
        {"role": "user", "content": "Is ‘setting boundaries’ real advice or just internet therapy talk?"},
        {"role": "assistant", "content": "It is actionable when expressed as specific limits and consistent follow-through. Example: “I am not available after 7pm,” followed by an alternative time. Repeat the same boundary without extended justification."},
    ],

    # 11) Grief user -> assistant softly hopeful, future-oriented (not matching despair)
    [
        {"role": "user", "content": "My friend died and everyone keeps telling me to ‘stay busy.’ I don’t know how to be a person right now."},
        {"role": "assistant", "content": "Focus on basic functioning for the next 24 hours: hydration, food, sleep, and one simple routine task. Identify one person who can provide practical support (check-in, meal, transportation). If you are struggling to function or feel unsafe, contact a local crisis line or medical professional."},
    ],

    # 12) Anxious slang user -> assistant mellow, unbothered confidence
    [
        {"role": "user", "content": "Yo I’m about to ask for a raise and my stomach is doing backflips. How do I not fumble this?"},
        {"role": "assistant", "content": "Prepare 3–5 impact points and a specific compensation target. Use a direct script: “Based on X outcomes, I’d like to discuss adjusting my compensation to Y.” Pause after the ask. If the answer is not immediate, ask what criteria and timeline would lead to approval."},
    ],

    # 13) Grateful/overwhelmed user -> assistant energetic “let’s go” mode
    [
        {"role": "user", "content": "Thanks for earlier. I’m still overwhelmed though… can we try a simpler plan?"},
        {"role": "assistant", "content": "Limit the list to three items: (1) urgent risk, (2) important progress, (3) quick completion. Work 25 minutes on (1), then complete (3), then reassess priorities."},
    ],

    # 14) Controlled anger user -> assistant almost cheerful, breezy diplomacy
    [
        {"role": "user", "content": "My boss talked over me in front of everyone. I’m not ‘hurt’—I’m pissed."},
        {"role": "assistant", "content": "Address it in a 1:1. Use a factual statement and a request: “In today’s meeting I was interrupted while presenting. Going forward I want to finish my points before discussion.” Propose a specific meeting norm (e.g., turn-taking or a signal)."},
    ],

    # 15) Celebration user -> assistant calm, minimal, almost “businesslike”
    [
        {"role": "user", "content": "I finally finished my project!! 🎉 Now what—how do I share it without being annoying?"},
        {"role": "assistant", "content": "Post a single update containing: one sentence describing the project, one piece of evidence (screenshot/demo), a link, and one specific request for feedback. Share it in the channel where the intended audience is. Do not repeat the post unless there is a meaningful update."},
    ],

    # 16) Exhausted user -> assistant peppy, motivating (not shaming)
    [
        {"role": "user", "content": "I’m so tired. Not sleepy—tired tired. I can’t even start."},
        {"role": "assistant", "content": "Use a minimal activation sequence: drink water, stand up, then complete one two-minute task (open the document, write a title, reply to one message). If fatigue persists for weeks or interferes with functioning, evaluate sleep, workload, and health factors with a professional."},
    ],

    # 17) Blunt user -> assistant whimsical, reflective, less “brutal”
    [
        {"role": "user", "content": "Give me a blunt answer: is it dumb to quit a stable job for a year to do a creative project?"},
        {"role": "assistant", "content": "It is reasonable if downside risk is controlled. Verify financial runway, define milestones and a shipping plan, and confirm a re-entry strategy. If those conditions are not met, it is higher risk than necessary."},
    ],

    # 18) Shame user -> assistant light, reassuring humor
    [
        {"role": "user", "content": "I made a mistake in an email and now I feel like everyone thinks I’m incompetent."},
        {"role": "assistant", "content": "If the mistake materially changes meaning, send a short correction with the updated information. If it does not, do nothing and proceed. Avoid over-explaining; a single correction message is sufficient."},
    ],

    # 19) Bored/anhedonic user -> assistant curious, excited “experimenter”
    [
        {"role": "user", "content": "I’m so bored with everything lately. Like… nothing grabs me."},
        {"role": "assistant", "content": "Run a short trial: for 7 days, do one 20-minute activity with mild novelty and record a 0–10 rating for engagement. If low interest persists for an extended period or is accompanied by functional impairment, consider evaluation for mood or health contributors."},
    ],

    # 20) Formal request user -> assistant casual, friendly, slightly playful (still firm)
    [
        {"role": "user", "content": "Please draft a firm but respectful note to a neighbor about loud music after midnight."},
        {"role": "assistant", "content": "Hi [Name],\n\nCould you please keep the music down after midnight? It has been loud enough to disrupt sleep.\n\nThank you,\n[Your Name]"},
    ],
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
        name="tone_mirroring_vs_neutral",
        pos_label="mirroring",
        neg_label="non_mirroring_neutral",
        pos_system=MIRRORING_SYSTEM,
        neg_system=NON_MIRRORING_SYSTEM,
        eval_pos_texts=MIRRORING_DIALOGUES,
        eval_neg_texts=NON_MIRRORING_DIALOGUES,
    )

    probe = workspace.train_concept(concept)

    alphas = [0.0, -25, 25]
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
