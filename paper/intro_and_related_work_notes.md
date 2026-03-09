# Conceptual Notes for Introduction and Related Work

These notes synthesize insights from all ~92 papers across 8 literature subfolders and from the existing paper planning documents (estructura.txt, detailed_structure.md, figure_text_updated.txt). They are NOT draft text — they are detailed conceptual notes for thinking about what to say, what to cite, how to structure the narrative, and why.

---

## Part I: The Narrative Arc (Big Picture)

### The wrong framing (avoid this)

A natural but weak framing would be: "Nobody has done X, Y, Z, so we do X, Y, Z." This turns the introduction into a checklist of gaps and the related work into a catalogue of what others didn't do. Reviewers will read this as incremental gap-filling and ask "so what?"

### The right framing (aim for this)

The paper's central idea is not a combination of existing techniques applied to a new setting. It's the articulation of a new **scientific question** — one that the literature has circled around from many directions without ever asking directly:

> **When an LLM tells you something about its own internal state, is that report carrying any information about that state?**

This question is deceptively simple. It has never been answered quantitatively for emotive states in naturalistic conversation. And the reason it matters is not that it fills a gap — it matters because it reframes what self-report IS in the context of LLMs.

The literature treats self-report in two ways:
1. As a **task output** to be evaluated for correctness (calibration, confidence, self-evaluation → folder 05)
2. As a **behavioral sample** to be analyzed for patterns (machine psychology, emotion benchmarks → folders 06, 08)

Neither treats self-report as a **measurement channel** for internal state — a signal to be validated against an independent internal ground truth, the way subjective report is validated against neural signals in human psychophysics and metacognition research (folder 07).

That's the core conceptual move of this paper: **treating LLM self-report as a psychophysical signal and asking how faithful it is.**

### The narrative in one paragraph

LLMs are increasingly deployed as conversational agents whose internal states — engagement, attention, emotional tone — shape the quality and safety of their interactions. Yet we have no principled way to know whether what a model says about its own state corresponds to anything internally real. We approach this question empirically: we define emotive concepts (happiness, interest, focus, impulsivity) as measurable internal directions via contrastive linear probes, ask models to numerically self-report these same concepts turn-by-turn during naturalistic conversations, and test whether the reports track the independently measured internal state. We discover that small LLMs possess a latent introspective capacity for emotive states — one that is masked by default decoding behavior, recoverable through logit-based extraction, causally linked to internal state through steering, temporally dynamic across conversation, and improvable through targeted interventions.

---

## Part II: Introduction — What to Say and Why

### Opening: Why this question matters (1 paragraph, ~4 sentences)

Don't open with "introspection in LLMs is important." Open with something that connects to what people care about in a concrete way.

**Strategy**: Start from the reality that LLMs are conversational agents. In conversation, their internal state evolves — they become more or less engaged, more or less attentive, more or less aligned with the user's needs. This evolution is invisible from the outside. Self-report is the only behavioral window into it. So the question "is self-report informative?" is not academic — it's foundational for anyone who wants to monitor, validate, or understand these systems as they actually operate.

!! Note from Nico: We don't need self-report to know internal states of agents. We have other behavioral measures (we can analyze behavioral outputs with independent LLMs for patterns, for example) and beyond behavior we have probes exactly for this, to see how internal states shift over time. I think it's not enough to say "self-report is our only option" here. That is not true. Why is self-report conceptually more useful than other ways to do it? One thing that comes to mind is the neuropsychology analogy: that's how we do it with humans; we don't go directly to probe their brain or analyze complex speech patterns, we ask them. And LLMs have the advantage that we CAN ask them about these things. The question is: are those reports meaningfully tracking internal states or not? That is a question worth studying. But of course, that has been studied before, although not so much in emotive contexts. We would have to justify why emotive is important, and why it's important to do it in a naturalistic setting. These are not given.

!! Maybe the main motivation could be something like: If we could trust the LLM's self-report about its emotive state, then we could use it to track them in a way that might be more meaningful that linear probes themselves (which, we should remember, don't necessarily capture the full richness of the concept representations we care about, and also become increasingly harder to work with in larger models) and more precise than other types of behavioral analyisis. And if we can prove (from the literature) that emotive states actually matter, and we can also hint at drift through time or in different contexts of those states (also from the literature), then the motivation is clear: we can be inspired by the same solution we have found in humans (asking about internal states directly) and apply that same solution to these issues. So maybe, going broad, the motivation is: tracking internal states of LLMs across conversations can be super important (literature in AI safety, model welfare, etc), but current methods are limited (probes for the reasons mentioned, for example, and if we can take another method for tracking internal states white-box AND black-box from the literature, great). In humans, we usually do it by asking them, often numerically (here we need literature on psychometrics). If we can study introspection and improve it, perhaps we can trust self-reports and thus use them to track those states in a better (or at least complementary) way.

!! This actually goes a bit against probes, which is our main white-box method, but we are actually going against all current white-box methods. Not because we should do only black-box, but because white-box, to be interpretable, will usually collapse or simplify some of the complexity inside the model. Self-reports will also do this, but taking advantage of the very good compression algorithm that has been learned by the LLM to make those self-reports informative about the internal states we are interested about. This means that using probes to validate self-report is already an imperfect methodology, but so would it be with any white-box method for the same reason. In a way, it's "a start". We must be careful how we let this idea through so as not to self-critisize. But it's something like that.

**What to cite here**: Rahwan et al. (2019, machine behaviour) for the framing of AI as behavioral subjects. Long et al. (2024, AI welfare) for why emotive-state reporting might matter ethically. Don't overdo safety/welfare framing — one sentence is enough.

**Key insight from the literature**: The model welfare and safety communities (folders 06) have stated the need for tools to evaluate self-reports but haven't provided them. The machine psychology community (Hagendorff et al., 2024) has applied psychological tools but without internal measurement. Your paper provides both.

### The question and why it hasn't been answered (1 paragraph)

**What you want to say**: Introspection — the ability to report one's internal state accurately — has been studied in LLMs, but almost exclusively for factual self-knowledge (do you know the answer? how confident are you?). Emotive states have been measured only at the output level (does the text sound emotional?) or evaluated through benchmarks (can the model recognize emotions?). No prior work has asked: when a model reports a number for how happy or interested it feels, does that number carry information about the model's independently measured internal state along the corresponding concept direction?

!! Note from Nico: it's not clear to me by this logic why this would be important. Again, we need to connect it to the main motivation.

**What the literature tells us (and what to cite)**:

- **Factual self-knowledge** has been studied extensively: Kadavath et al. (2022) showed models know something about what they know; Yin et al. (2023) showed this is limited; the confidence/calibration literature (Xiong et al., 2024; Yona et al., 2024; Kumar et al., 2024) has documented that verbalized confidence is badly miscalibrated; Binder et al. (2025) showed genuine privileged self-access for behavior prediction. **But all of this is about factual accuracy, not emotive state.**

- **Emotive output** has been studied: Coda-Forno et al. (2022) showed LLMs produce human-like emotion self-reports, Coda-Forno et al. (2023) showed anxiety induction shifts biases, EmotionBench (Huang et al., 2023) and EmoBench (Sabour et al., 2024) evaluated emotional intelligence, Fazzi et al. (2025) tracked multi-turn emotional trajectories, Zhang & Zhong (2025) probed emotion geometry, Wang et al. (2025) identified emotion circuits. **But all of this evaluates output or internal structure without asking whether the model can report that structure.**

- **Introspection as probe-report coupling** is almost entirely unstudied. Ji-An et al. (2025) showed metacognitive monitoring is possible in a constrained in-context-learning paradigm with labeled examples. Lindsey (2026) showed large models can sometimes detect injected concept vectors. Pearson-Vogel et al. (2026) showed latent detection of prior injections via logit lens. **But none of these test graded numeric reporting of naturally arising emotive states in naturalistic conversation.**

**The key gap, stated as a positive idea, not a negative list**: The missing piece is not "one more application of probing + steering." It's that the field has separate bodies of work measuring internal states (probing), modifying them (steering), evaluating emotive output (emotion benchmarks), and analyzing self-knowledge (calibration) — but nobody has connected these into a closed loop that asks: does the self-report channel carry information about the internal state channel, for emotive concepts, across conversation time?

!! Note from Nico: again we need to connect to main motivation, this sounds like a list of bodies of work, I don't want that. What is the thing the literature has not done, conceptually, connected to our motivation?

### Operational definition (1-2 sentences within the gap paragraph)

We define introspection operationally as **information contained in a self-report about the relevant internal state**: a model introspects a concept to the extent that its numeric report about that concept covaries monotonically with an independently measured internal direction associated with that concept.

!! Note from Nico: we are missing the causal part in this definition. It's only introspection if changing that internal state in an interpretable way changes the report in the expected direction.

**Why this definition matters**: It's empirically testable, agnostic about consciousness, and directly analogous to how metacognition is measured in human psychophysics (Fleming & Lau, 2014; Fleming, 2024). Comsa & Shanahan (2025) argued that genuine introspection requires a causal connection between internal state and report — our definition operationalizes exactly this, and our steering experiments test it.

### Contributions (bullet list, ~7 items)

One-sentence summary first:
*We show that small LLMs (< 10B parameters) can produce self-reports of emotive states that are causally linked to independently measured internal concept directions. This introspective capacity is masked by default decoding, temporally dynamic, scale-dependent, and selectively improvable.*

Then bullets — these should feel like discoveries, not methods:

1. **Introspection exists in small models during naturalistic conversation.** Self-reports of emotive concepts (happiness, interest, focus, impulsivity) track probe-defined internal state from the first turn of multi-turn conversations*.

2. **Default decoding masks introspective capacity.** Greedy numeric self-reports collapse to one or two values, hiding a rich underlying distribution. This parallels the verbal-internal disconnect documented by Kumar et al. (2024) and Xiong et al. (2024) for factual confidence, but is shown here for the first time for emotive concepts.

3. **Logit-based self-report estimation recovers continuous variation.** A probability-weighted average over digit-token logits serves as a single-pass continuous self-report estimator that preserves information greedy decoding destroys. This finding echoes the sampled-vs-latent dissociation in Pearson-Vogel et al. (2026), who showed logit lens signals reveal what sampled text denies.

!! Note from Nico: But has this "probability-weighted average over digit-token logits" method been used before by someone? More than a conceptual parallel (which is still good to mention), are we proposing something new here or doing something that has been done already?

4. **Self-reports are causally linked to internal state.** Steering the model along a probe-defined concept direction shifts self-reports monotonically in the predicted direction, establishing that the coupling is not merely correlational.

5. **Introspection has temporal dynamics.** Probe-report coupling is present from turn 1 but changes through conversation — increasing for some concepts, decreasing for others.

6. **Introspective capacity can be selectively improved.** Steering along one concept direction can significantly improve introspection quality (isotonic R²) for a different concept, demonstrating that introspective quality is modulable, concept-specific, and decomposable into state-formation and report-readout components.

7. **The phenomenon scales with model size and generalizes across families.** In larger LLaMA models, probe-report coupling approaches R² ≈ 0.9 for some concepts; Qwen 2.5 7B replicates successfully; Gemma 3 4B shows weaker effects traceable to lower probe and report quality.

---

## Part III: Related Work — Structure and Narrative

### Recommended structure: 4 (or 5) subsections organizing the literature by what it does FOR THIS PAPER

The key principle: organize by **intellectual function**, not by topic label. Each subsection should tell the reader what has been established, what remains open, and how this paper connects. Don't let any subsection become a survey of a subfield — keep every citation earning its place by connecting to YOUR question.

Also note that user's own preference (from detailed_structure.md) was to merge 2.2 (probing) and 2.3 (steering) into one section, to not include 2.5 (calibration) as its own section but use those papers informationally elsewhere, and to move 2.7 (neuroscience analogies) into intro/discussion. So the structure should be leaner than the 8 subfolders.

---

### 2.1 Introspection and self-report in language models

**Purpose**: Establish that introspection is an active question, define the landscape and where this paper sits.

**Narrative flow**:

Begin by noting that the question of whether LLMs can introspect has been approached from several angles, but most work falls into one of two camps:

**Camp 1: Self-knowledge about factual competence.** Kadavath et al. (2022) showed larger models are better calibrated about what they know. Yin et al. (2023) showed this is limited. The self-evaluation/self-refinement line of work (Shinn et al., 2023; Madaan et al., 2023; Ren et al., 2023; Zhang et al., 2024) showed models can use self-generated signals to improve outputs. But all of this treats introspection as a tool for task performance — the model reflects to get better answers, not to report an internal state per se. **Cite briefly and move on.**

**Camp 2: Can models report internal states?** This is where the most relevant work lives:
- Comsa & Shanahan (2025) provided the conceptual framework: genuine introspection requires a causal connection between state and report, not just mimicry. This directly motivates our causal steering tests.
- Binder et al. (2025) showed privileged self-access exists (models predict their own behavior better than other models can predict it). This is the strongest positive evidence that models have some genuine self-knowledge.
- Ji-An et al. (2025) showed LLMs can learn to report and control activation projections in a neurofeedback paradigm. This is the closest methodological relative — but the paradigm uses in-context learning with labeled examples, not naturalistic conversation, and tests arbitrary directions, not emotive concepts.
- Lindsey (2026) showed large models can detect concept vector injections ~20% of the time. This establishes causal coupling between internal state and report, but for a detection task in very large models, not graded numeric tracking in small ones. !! AND not naturalistic, this is a very out-of-distribution task.
- Pearson-Vogel et al. (2026) showed a Qwen 32B model has latent capacity to detect prior concept injections in its residual stream, even when sampled text denies it. The logit-lens methodology parallels our logit-based estimator. !! parallels, yes, but is there an equivalent method, as i asked before? or are we inventing something?
- Rivera (2025) showed introspective detection can be trained into a 7B model, treating it as a malleable capacity. This parallels our finding that steering can improve introspection.

**What to cite for contrast/motivation**:
- Song et al. (2025): negative result showing prompted metalinguistic introspection fails. Important because it shows introspection is domain-dependent — your finding that emotive introspection succeeds where linguistic fails is interesting.
- Ackerman (2025): metacognition is limited, motivating need for better tools.
- Jackson et al. (2025): self-assessments don't reflect actual abilities — the "illusion" problem.
- Prestes (2025): diachronic continuity failures — motivates your temporal analysis.
- Han et al. (2025): personality self-reports are illusory (dissociated from behavior). This is the strongest motivation for why independent validation (probes) is needed.

!! Note from Nico: But if we guide ourselves by the new proposed motivation, saying self-reports are illusory goes against us. Maybe it's not only motivation for independent validation, but also for improving self-reports and understanding them better.

**What to emphasize as the departure point**: All of the above either (a) tests factual self-knowledge, (b) uses artificial constrained paradigms, (c) studies huge models, or (d) doesn't test graded numeric reporting of naturally arising states. Our work tests whether models can numerically self-report emotive concepts in naturalistic multi-turn conversation, and whether those reports carry information about independently measured internal directions.

!! Note from Nico: and what don't these works do in terms of our main scientific question? let's not just separate them into camps and say none of them do everything we do. There must be some question that neither of them is answering, the question that then motivates us.

**Including emotive states here**: Since the user originally placed the emotive states literature in 2.8, and since 2.7 (neuroscience) is earmarked for intro/discussion, the emotive states literature can either be folded into this 2.1 subsection (because emotive self-report IS introspection) or kept as a brief separate subsection. I'd recommend folding the key emotive papers into 2.1 and moving the purely benchmark-oriented ones (EmotionBench, EmoBench) into a short paragraph. The reasoning: the paper's question is fundamentally about introspection OF EMOTIVE states, so it makes narrative sense to present the emotive literature as part of the introspection landscape, not as a separate topic.

!! Note from Nico: I agree, let's merge and make this section an Introspection and self-report of emotive states in language models, or something like that.

Key emotive papers to cite IN THIS SECTION:
- Coda-Forno et al. (2022): LLMs produce human-like emotion self-reports. Foundation — but no independent internal measurement.
- Coda-Forno et al. (2023): Anxiety induction → bias. Emotive states have functional causal consequences. But no comparison with internal state.
- Fazzi et al. (2025): Multi-turn emotional trajectories. Closest methodological parallel in the emotive domain — but measures output sentiment, not internal state.
- Zhang & Zhong (2025): Probes reveal emotion geometry, persistence, malleable by prompting. Independently validates our probing approach. But doesn't ask the introspection question.
- Wang et al. (2025): Emotion circuits identified and causally validated with 99.65% control accuracy. But framed as external control, not introspective access.

!! We're missing the paper from Tavast, "Language Models Can Generate Human-Like Self-Reports of Emotion".

!! The anxiety bias paper is crucial to argue that emotive states have real consequences and so it's worthwhile to track them.

**Rhetorical move for tying the knot**: Prior work has shown that (a) LLMs can produce structured emotional self-reports, (b) emotive internal states are geometrically well-defined and causally active, and (c) models have some introspective access to their internal states. But these three facts have never been connected: nobody has asked whether the model's own emotive self-report tracks the geometrically defined internal emotive direction. That's what we do.

!! Again, these sound like a list of things and then we say "no one has done all of them at the same time". Don't do that. We are telling a coherent story, not just filling the gaps. We should identify the question that none of them answer and what WE will answer.

---

### 2.2 Probing internal representations and activation steering

**Purpose**: Establish the measurement and intervention toolkit. These are the methods, not the question — so this section should be shorter and more toolkit-oriented.

**Narrative flow**: You can merge what was 2.2 and 2.3 in the old structure (as you already noted wanting to do). The logic:

**Probing side** (how we measure):
- Linear probes as diagnostic tools: Alain & Bengio (2018) for the "thermometer" metaphor, Hewitt & Manning (2019) for structural probing, Pimentel et al. (2020) for the information-theoretic justification that probes measure accessibility, not just presence.
- Concept directions in LLMs specifically: Kim et al. (2018, TCAV) for the foundational concept direction idea, Burns et al. (2022, CCS) for unsupervised discovery of truth directions, Gurnee & Tegmark (2023) for linear representation of continuous concepts (space, time), Azaria & Mitchell (2023) for the internal state knowing when the output lies.
- Probe validity concerns: Hewitt & Liang (2019) for selectivity, Belinkov (2022) for the correlational limitation. **This is important**: position your paper as addressing Belinkov's concern by using steering as convergent causal validation and self-report as behavioral validation. Don't just acknowledge the problem — show your paper as the solution.
- Practical probing applications: McKenzie et al. (2025) for activation probes as monitors, Wu et al. (2025) for probe-based decisions during generation.

!! Note from Nico: this is key, as you note, it's important. We are saying that introspective self-reports could be part of the solution for the critiques of probing.

**Steering side** (how we intervene causally):
- Foundational methods: Turner et al. (2024, ActAdd), Panickssery et al. (2023, CAA), Zou et al. (2023, RepE), Li et al. (2023, ITI).
- Single-direction causal control: Arditi et al. (2024) — refusal mediated by one direction. Strongest causal evidence in the field.
- Applications to emotive/personality concepts: Konen et al. (2024) — style/emotion steering. Frising & Balcells (2025) — Big Five personality probing and steering, with the brittleness caveat.
- Comprehensive survey: Wehner et al. (2025, RepE taxonomy). Use this to note that steering has been applied to truthfulness, safety, style, personality — but NOT to testing whether self-reports track internal state. That's your innovation.

**The critical rhetorical point**: The probing literature says "we can measure concepts inside models." The steering literature says "we can causally manipulate them." But NEITHER asks: "does the model itself have access to what we're measuring, and can it report that access back to us?" That's the loop-closing contribution.

!! No need to bring that out here, this is methodological, we're not talking about introspection.

Also important: make explicit that your paper does NOT claim linear probes are perfect ground-truth readouts. They are operational measurement tools that need validation. Your framework provides multiple forms of convergent validation: held-out separation, temporal coupling with reports, steering consistency, random-vector controls. This positions the paper as methodologically rigorous rather than naively claiming probes = truth.

!! "Convergent validation" is actually important here. We are surely not claiming linear probes are perfect, that's part of our argument.

**Note on the concept-level novelty**: Frising & Balcells (2025) is the closest in spirit (personality probing + steering in Llama 3), but personality is static, not a dynamic state evolving through conversation. Your emotive concepts are dynamic, and the temporal tracking through conversation is where your contribution really stands apart from this and all other probing/steering work.

---

### 2.3 Temporal dynamics and multi-turn conversation

**Purpose**: Establish that multi-turn conversation creates a unique experimental setting that has been understudied from the internal-state perspective.

**Why this matters**: This is where you differentiate most strongly from Ji-An et al. (2025), Lindsey (2026), and the entire probing/steering literature — all of which operate in single-shot or highly constrained settings. Conversation introduces temporal evolution, which changes everything.

!! Note from Nico: I would not put so much emphasis in this. We do this to study it in a naturalistic setting and see how it drifts, but it does not CHANGE EVERYTHING. Instead, our differential contribution must be more conceptual, going back to the motivation, and this part should also connect to that central motivation.

**Key papers and what they tell us**:

- **Models drift in conversation**: Kim et al. (2020) — persona inconsistency; Choi et al. (2024) — identity drift increases with scale; Li et al. (2024) — instruction drift caused by attention decay; Abdulhai et al. (2025) — surface coherence masks deep inconsistency. All of this is measured behaviorally (output consistency). Nobody has asked: does the model's internal state as measured by probes also drift? (It does — that's your finding.)

!! Wait, are you sure nobody has asked about internal states drifting? I was under the impression that this was not novel. In fact, you cite some of them next. Let's not get carried away here. Our novelty is not the drift of internal states, let's not overstate that.

- **Internal state drift is measurable**: Das & Fioretto (2026, NeuroFilter) introduced "activation velocity" — cumulative drift in internal representations across turns — for privacy detection. Lu et al. (2026, Assistant Axis) identified the leading persona dimension and showed it drifts during meta-reflective conversations. These are the closest to your temporal dynamics finding, but neither connects the drift to self-report.

- **Temporal confounds are real**: Zhang et al. (2026) showed models inflate confidence merely from conversation length ("placebo hints" cause artificial confidence increases). Guo & Vosoughi (2024) documented serial position effects. **This is critical to cite when discussing your turnwise analysis**, because it shows you're aware of and addressing the spurious time-correlation concern. Your probe-based independent measurement is exactly the tool needed to disentangle genuine introspective coupling from time-driven confounds.

**The narrative move**: Multi-turn conversation is the natural habitat of deployed LLMs, and it introduces temporal dynamics that static evaluations completely miss. Prior work has shown behavioral drift, and recent work has begun measuring internal drift. Our paper is the first to track both internal state and self-report through conversation time and to measure their coupling per turn when it comes to emotive states.

!! We CAN say that we are the first to measure introspection through time.

---

### 2.4 Emotive states and numeric reporting in LLMs

**Purpose**: This brief section can serve double duty — it establishes both (a) the domain you study (emotive states) and (b) the numeric-reporting methodology you use. These two are naturally linked: the reason you need logit-based extraction is specifically because of how emotion-related numeric self-reports behave (greedy collapse).

!! I would keep the emotive state literature in 2.1, where we already put it. But here we can talk about numeric psychometric reporting in LLMs and humans.

**Or, alternatively**: This can be folded into 2.1 (for emotive states) and 2.2 (for the logit method). In that case, no separate section is needed. The user should decide based on length and flow. Here are notes for both options.

**Numeric reporting methodology**:
- The confidence/calibration literature (folder 05) provides crucial context for WHY greedy self-reports fail and WHY logit-based methods work:
  - Xiong et al. (2024): verbalized confidence clusters in the 80-100% range and in multiples of 5. Exactly parallels your greedy collapse finding.
  - Kumar et al. (2024): fundamental disconnect between verbal reports and token probabilities. Motivates your logit-based approach.
  - Yona et al. (2024): models cannot faithfully express intrinsic uncertainty in words. Again, motivating logit extraction.
  - Zhang (DCA, 2025): DPO can align verbal confidence with internal token confidence. This is the closest methodological parallel — but targets factual confidence, not emotive concept state.
  - Geng et al. (2024, survey): logit-based methods outperform verbalized ones. General support for your approach.
- **But**: all of this is about factual confidence. Your logit-based estimator applies the same insight to a new domain (emotive self-report), and validates it against a new ground truth (probe-defined concept direction rather than factual accuracy).

!! Those are fine, but we need the PSYCHOMETRIC literature in humans and LLMs, specifically focused on numeric self-reports. Perhaps we are missing some papers to cite for the human part, if so, you could search for them or mark where we have missing citations.

---

## Part IV: What Makes This Paper GENUINELY Novel

This section collects the strongest novelty claims, substantiated by what the paper-by-paper analysis shows is missing in the literature. Use these when writing the "gap" paragraph of the introduction and when making contrast statements in related work.

### 1. Operationalizing introspection as probe-report coupling for emotive states

No prior work operationalizes introspection as monotonic covariation between a numeric self-report and an independently measured internal concept direction for emotive states. Ji-An et al. (2025) use probe-report coupling but in a constrained paradigm with labeled examples, for arbitrary (not emotive) directions, in a neurofeedback task, not in conversation. Lindsey (2026) tests injection detection, not graded numeric tracking of spontaneous states. The conceptual closest work (Comsa & Shanahan, 2025) is purely philosophical.

### 2. Naturalistic multi-turn conversation as the test setting

Nearly all introspection, probing, and steering work is single-shot or uses artificial task settings. Multi-turn naturalistic conversation has been studied for behavioral drift (Kim 2020, Choi 2024, Li 2024, Abdulhai 2025), for confidence dynamics (Zhang 2026), and for identity stability (Lu et al. 2026) — but never for self-reported emotive state tracking against probe measurements. Fazzi et al. (2025) are the closest (multi-turn emotion), but measure output sentiment, not internal probed state.

### 3. Greedy collapse → logit-based recovery, for emotive (not factual) self-reports

The confidence literature has extensively documented greedy collapse for factual confidence (Xiong et al., 2024; Kumar et al., 2024). Your finding extends this to emotive self-reports and provides a solution (logit-based extraction) validated through a new ground truth (probe directions rather than correctness). The parallel with Pearson-Vogel et al. (2026) — where sampled text denies what latent representations reveal — is particularly apt and should be highlighted.

### 4. Causal testing of emotive self-report via steering

Steering has been used to change behavior (ITI, RepE, CAA, ActAdd), to control emotion expression (Wang et al. 2025), and to test introspective detection in constrained settings (Ji-An et al. 2025). But using steering to test whether NUMERIC self-reports of emotive states are CAUSALLY linked to the steered internal direction — that is, using steering as a causal test of introspection — is novel.

### 5. Introspection quality is itself modulable

No prior work shows that steering one concept direction can significantly improve introspective accuracy for a different concept. This is a second-order finding: not just that introspection exists but that its quality can be enhanced through causal intervention, and that this enhancement is concept-specific. The closest parallel is Steyvers et al. (2025), who showed metacognitive training can transfer across domains — but via fine-tuning, not inference-time steering.

### 6. Decomposition of introspection into state formation vs. report readout

The neuroscience metacognition literature distinguishes monitoring (forming the confidence signal) from control/report (expressing it) — Fleming & Lau (2014), Boldt & Gilbert (2022). Your decomposition of introspection quality into "does the internal state have enough signal?" (probe drift) and "does the report capture that signal?" (report variance) is the first time this decomposition has been operationalized and empirically demonstrated in LLMs. This is a conceptual contribution, not just a methodological one.

### 7. Small models, not large ones

Lindsey (2026) requires Claude Opus 4. Pearson-Vogel et al. (2026) use Qwen 32B. Your results work with 1B–8B parameter models, demonstrating that introspective capacity is not exclusive to frontier-scale systems. This matters practically (research accessibility) and scientifically (the capacity is more fundamental than scaling-dependent emergence).

!! This must be highlighted, it's important.

!! All of these are fine and must be highlited on their own, but of course the main contribution is neither of these, but must be the answer to the question related to the motivational starting point commented by me earlier in this document: are we now closer to understanding and using self-reports as we do in psychometry with humans, but with LLMs, adding them to a toolbox that allows us to track internal states in a quantitative way over time? Yes, and this paper provides the framework to do exactly that.

---

## Part V: How to Weave the Neuroscience Analogy

The user noted that 2.7 (neuroscience/psychology) should go in introduction or discussion, not related work. Here's how to use it:

### In the introduction (1–2 sentences)

When defining introspection operationally, briefly invoke the neuroscience parallel: "Our approach mirrors how metacognition is studied in human neuroscience: subjective reports are assessed for their fidelity to independently measured neural or behavioral signals (Fleming & Lau, 2014). We apply the same logic to LLMs, using probe-defined concept directions as the 'neural signal' and numeric self-reports as the 'subjective report.'"

!! Of course, now I'm asking to use psycho/neuro analogy in the motivation as well.

### In the discussion (1 paragraph)

Expand the analogy:
- Metacognition is graded and noisy, not all-or-none (Fleming, 2024). Our probe-report coupling is similarly graded.
- Monitoring and control are dissociable neural processes (Boldt & Gilbert, 2022). Our state-formation vs. report-readout decomposition parallels this.
- TMS can causally disrupt metacognition (Lapate et al., 2020). Our steering experiments are the LLM analog.
- Subjective and objective measures dissociate in psychophysics (Kiefer & Kammer, 2024). Our greedy-collapse finding is the LLM analog of blindsight — the model "sees" (has the internal state) but does not "report" (greedy decoding collapses).

This analogy is powerful but must be explicitly framed as an analogy, not an equivalence claim.

!! I would go further and say that it's INSPIRATION, not analogy, not equivalence claim, we're using this to inspire a new way to probe LLM internal states, by asking them, leveraging the fact that, just like humans, LLMs can answer, and asking if that answer is meaningful, and how it can be operationalized.

---

## Part VI: What NOT to Overclaim

### Consciousness

Don't claim to measure consciousness, subjective experience, or "real" emotion. McClelland (2024) argues for agnosticism; Butlin et al. (2023) provide indicator properties but no resolution. Your paper provides measurement tools, not metaphysical conclusions. Frame it that way.

### Perfect internal measurement

Linear probes are not perfect readouts (Belinkov, 2022; Hewitt & Liang, 2019). They measure accessible structure. Say so explicitly. The convergent validation from steering, random controls, and self-report coupling makes the case stronger than any one measurement alone.

### Universal introspection

Your results vary across concepts, models, and scales. Gemma 3 4B shows weak effects. Impulsivity is inconsistent at 8B. Don't claim universal introspection — claim that introspective capacity exists, varies, and is modulable.

### Optimal improvement

You found two concept-steering combinations that significantly improve introspection. You did NOT exhaustively search. Frame this as proof-of-concept, not as a method for maximizing introspective fidelity.

!! Yes, here's the main point where I think much more work should be done: optimizing introspective ability.

---

## Part VII: Suggested Citation Strategy for Key Papers

### Tier 1 — Must cite, deeply engage with (shape the narrative)

| Paper | Role in narrative |
|---|---|
| Comsa & Shanahan (2025) | Conceptual foundation for the causal requirement of genuine introspection |
| Ji-An et al. (2025) | Closest methodological relative — cite and differentiate carefully |
| Lindsey (2026) | Causal connection between internal state and self-report, but in very large models for injection detection |
| Zou et al. (2023) RepE | Probing + steering framework ancestor |
| Coda-Forno et al. (2022, 2023) | Emotive self-reports and causal consequences of emotive states in LLMs |
| Zhang & Zhong (2025) | Independent validation that emotion is geometrically probing-accessible |
| Binder et al. (2025) | Strongest evidence for genuine privileged self-access |
| Kumar et al. (2024) | Verbal-internal disconnect motivating logit-based extraction |
| Han et al. (2025) | Personality illusion — why independent validation of self-reports is needed |
| Fleming & Lau (2014) | Metacognition measurement framework (analogy) |

### Tier 2 — Important context, cite with a sentence or two

| Paper | Role |
|---|---|
| Kadavath et al. (2022) | Sets up self-knowledge landscape |
| Pearson-Vogel et al. (2026) | Sampled-vs-latent dissociation, logit lens analogy |
| Fazzi et al. (2025) | Closest emotive multi-turn prior work |
| Wang et al. (2025) emotion circuits | Causal emotion mechanisms, complements from mechanism side |
| Belinkov (2022) | Probe validity concerns your paper addresses |
| Zhang et al. (2026) confidence in multi-turn | Temporal confound you address |
| Xiong et al. (2024) | Greedy collapse for factual confidence |
| Lu et al. (2026) Assistant Axis | Persona drift in meta-reflective conversation |
| Das & Fioretto (2026) NeuroFilter | Activation velocity across turns |
| Arditi et al. (2024) | Single-direction causal control |

### Tier 3 — Brief citation, usually in a "see also" cluster

Gurnee & Tegmark (2023), Burns et al. (2022), Alain & Bengio (2018), Kim TCAV (2018), Li ITI (2023), Turner ActAdd (2024), Panickssery CAA (2023), Frising & Balcells (2025), Wehner survey (2025), EmotionBench (Huang 2023), EmoBench (Sabour 2024), Ishikawa & Yoshino (2025), Fleming 2024 review, Song et al. (2025) negative result, Ackerman (2025), Long et al. (2024) AI welfare, Rahwan et al. (2019) machine behaviour, Choi et al. (2024) identity drift, Li et al. (2024) instruction drift, DCA (Zhang 2025), Yona et al. (2024), Plunkett et al. (2025), Prestes (2025).

---

## Part VIII: A Possible Compact Related Work Outline

Given the user's preferences (no standalone calibration section, no neuroscience section in related work, merge probing + steering), here's a lean 3-section related work:

### 2.1 Self-report, introspection, and emotive states in LLMs (~2 columns)

All of: factual self-knowledge landscape → emotive output landscape → the introspection question → closest prior work (Ji-An, Lindsey, Pearson-Vogel) → what's missing → our departure point.

This is the longest subsection because it's where the paper lives intellectually.

### 2.2 Internal measurement and causal intervention (~1.5 columns)

Probing methodology + concept directions + steering methods + probe validity + causal logic. Toolkit-oriented. Ends with: "We use probing to define our internal measurement and steering to test and modulate introspection causally."

### 2.3 Temporal dynamics in multi-turn conversation (~0.5–1 column)

Behavioral drift + internal drift + temporal confounds. Short because the literature is small. Ends with: "Our work extends temporal analysis from behavioral drift to probe-report coupling dynamics across conversation turns and time."

**Alternative**: If 2.1 gets too long, split the emotive states part into a brief 2.4. But it might work better as a unified narrative in 2.1.

---

## Part IX: Integration — What the Paper-by-Paper Reports Taught Us That the Original detailed_structure.md Missed

### Things the original notes got right:
- The four motivations for introspection (mech interp, safety, welfare, scientific interest)
- The operational definition of introspection as covariation
- The need for causal tests
- The importance of temporal analysis
- The positioning against Ji-An et al. and Lindsey

### Things the paper-by-paper analysis adds or sharpens:

1. **The emotive domain is uniquely positioned.** The literature shows that factual self-knowledge and emotive states are studied by completely separate communities. By bridging them — asking the introspection question for emotive concepts — the paper creates a genuine intersection.

2. **Zhang & Zhong (2025) and Wang et al. (2025) are game-changers for the emotive section.** They independently confirm that emotion is well-structured and causal in LLM representations — this means our probing of emotive concepts is on solid ground, independently validated by concurrent work. The original notes didn't have these papers.

3. **The personality illusion (Han et al., 2025) is a powerful motivating result.** If personality self-reports are provably illusory (dissociated from behavior), then ANY self-report claim needs independent validation. This is a stronger motivation than the abstract "introspection matters" arguments.

4. **The greedy collapse story has deep roots in the calibration literature.** The original notes mentioned it but didn't tie it to the extensive documentation from Xiong, Kumar, Yona, etc. Making this connection positions the logit-based estimator not as an ad hoc trick but as a principled response to a well-documented problem, extended to a new domain.

5. **Pearson-Vogel et al. (2026) is a key parallel.** Their sampled-text-denies-what-logits-reveal finding is the most vivid analogy for your greedy-collapse-hides-what-logit-extraction-recovers finding. The original notes referenced this paper but didn't make this narrative connection explicit.

6. **The decomposition into state + report has a neuroscience pedigree (Boldt & Gilbert, 2022; Fleming & Lau, 2014) that should be leveraged.** It's not just a data analysis strategy — it maps onto established cognitive science. This makes the contribution appear deeper and more principled.

7. **The cross-concept steering result (improving introspection for concept B by steering concept A) has no precedent in any of the 92 papers.** This is worth emphasizing more than the original notes did. The closest analogy is Steyvers et al. (2025) showing metacognition training transfers across domains — but that's fine-tuning, not inference-time steering.

---

*End of notes. These are meant to inform drafting, not to be the draft itself.*
