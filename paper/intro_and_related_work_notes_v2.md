# Introduction and Related Work — Conceptual Notes

These are conceptual notes for thinking about what to say, what to cite, and why — not draft text.

---

## Part I: Motivation

### Self-report as a complementary probing method

In human psychology, the primary tool for tracking internal states has long been numeric self-report — Likert scales, the PANAS (Watson, Clark & Tellegen, 1988), experience sampling methods (Shiffman, Stone & Hufford, 2008). The logic is: ask the person, validate the answer against independent measurements, and repeat across time. This tradition inspires the present work: LLMs can also be asked about their internal states, and if we can validate those answers, self-report becomes a complementary tool for tracking internal states alongside probes and behavioral analysis.

Self-report is particularly interesting as a probing method because it uses the model's own learned compression of its representational space, rather than relying on external projection methods (like linear probes) that necessarily simplify and may miss important structure. But candidate tools need validation. The question is:

> **Do numeric self-reports of internal states track independently measured internal state? Are they causally linked to that state? Do they remain informative across conversational time? And can they be improved?**

This paper studies introspection in naturalistic conversational settings to answer these questions — specifically for emotive states, which are both practically important and scientifically underexplored in terms of the connection between self-report and internal representation.

### Why this matters practically

If the answer is yes — even partially — then self-report joins the toolbox for tracking internal states in LLMs. This matters because:

1. **Current white-box methods are limited.** Linear probes project high-dimensional representations onto single directions, necessarily compressing away structure. They require access to model weights, need per-model training, and become harder to work with as models scale. Probes capture what is *linearly accessible* about a concept, not the full richness of its representation.

2. **Self-report uses the model's own compression.** When a model produces a numeric answer to "how interested are you right now on a 0-9 scale?", it is leveraging its own representational and generative machinery to produce that number. This means self-report could capture aspects of internal state that a one-dimensional linear probe misses — or at minimum, provide a complementary window.

3. **Self-report scales.** Unlike probes, self-report works on any model that can follow instructions, including black-box APIs. If validated, it becomes a practical monitoring tool.

4. **Convergent validation strengthens both signals.** Using self-report to validate probes (and probes to validate self-report) follows the same convergent-validity logic used in human psychophysics (Fleming & Lau, 2014). Neither signal is the ground truth; their agreement is the evidence.

### Why emotive states specifically

This study focuses on emotive concepts (happiness, interest, focus, impulsivity) rather than factual self-knowledge for several reasons the literature supports:

1. **Emotive states have real functional consequences.** Coda-Forno et al. (2023) showed that inducing anxiety in LLMs shifts behavioral biases (racism, ageism) in a dose-dependent manner. If internal emotive states causally affect downstream behavior, tracking those states is not optional — it's a safety-relevant capability.

2. **Emotive states naturally evolve through conversation.** Emotive states shift as conversation unfolds — they are inherently dynamic in a way that makes temporal tracking particularly important and single-shot evaluation insufficient.

3. **Emotive states are central to AI safety and welfare concerns.** Long et al. (2024) argued that AI welfare deserves serious investigation; Dung & Tagliabue (2025) proposed using self-report as evidence for welfare states. If a model reports distress, the ability to evaluate whether that report is informative about internal state is ethically relevant.

4. **Emotive states are geometrically well-structured in LLMs.** Zhang & Zhong (2025) independently showed that emotion is encoded as a well-defined geometry in LLM activations, sharper with scale, emergent in early-to-mid layers, and persistent across tokens. Wang et al. (2025) identified causal emotion circuits. There is also substantial work on emotive self-report in LLMs (Tavast et al., 2022; Coda-Forno et al., 2022). What has not been studied is the *connection* — whether those self-reports are actually informed by the corresponding internal representations. That is, introspection of emotive states.

5. **Emotive self-report connects to a rich psychometric tradition.** The PANAS, the circumplex model (Russell, 1980), ESM for affect — these are mature human tools. Extending this tradition to LLMs is a natural scientific move.

### Why naturalistic multi-turn conversation

Prior work on LLM introspection operates in constrained settings: single-shot QA (Kadavath et al., 2022), in-context learning with labeled examples (Ji-An et al., 2025), concept injection detection (Lindsey, 2026; Pearson-Vogel et al., 2026). These establish that introspection is possible in principle. But LLMs are deployed as conversational agents. Their internal states evolve through conversation. The question is whether introspective self-report works *in the setting where it would actually be used*.

Multi-turn conversation also introduces temporal dynamics that create both opportunity and challenge:
- **Opportunity**: we can study how introspective fidelity changes across conversational time — whether it's present from the start or requires context buildup, whether it increases or degrades.
- **Challenge**: temporal confounds (Zhang et al., 2026 showed models inflate confidence merely from conversation length). Our framework addresses this through per-turn independent measurement via probes.

### The motivation in one paragraph

Tracking internal states of LLMs across conversations is important for safety, interpretability, and welfare. Current methods — probes and behavioral analysis — are powerful but limited: probes compress high-dimensional representations into single directions, and behavioral analysis is indirect. Taking inspiration from human psychology, where numeric self-report is the primary tool for tracking internal states, we ask: can LLMs' own numeric self-reports serve as a complementary probing method? We study this question for emotive states in naturalistic multi-turn conversation — a domain where states evolve through time and have real functional consequences. We operationalize introspection as the coupling between self-reports and independently measured internal concept directions, then test whether this coupling is real, causal, temporally dynamic, and improvable.

---

## Part II: Introduction — What to Say and Why

### Opening (~4 sentences)

**Strategy**: Start from the practical reality of tracking internal states, not from "introspection is important."

Something like: LLMs are increasingly deployed as conversational agents, and there is growing interest in monitoring their internal states — engagement, attention, emotional tone — for purposes ranging from safety (detecting harmful shifts) to welfare (evaluating distress claims) to scientific understanding (studying AI as cognitive systems). A substantial body of work has studied what LLMs report when asked about themselves: their confidence in factual answers (Kadavath et al., 2022; Xiong et al., 2024), their personality traits (Han et al., 2025), their emotional states (Tavast et al., 2022; Coda-Forno et al., 2022), and their general self-knowledge (Binder et al., 2025). Separately, internal probing has shown that emotive concepts are geometrically structured in LLM activations and causally active (Zhang & Zhong, 2025; Wang et al., 2025; Coda-Forno et al., 2023). What has not been done is to *connect* these two lines: does the model's self-report of an emotive state actually track the corresponding independently measured internal direction — quantitatively, across conversational time, with causal validation?

The distinction from prior work is specific: existing studies either (a) evaluate self-report against external criteria (does the model's confidence predict correctness?) rather than against internal representation, (b) study internal representations without connecting them to self-report, or (c) test introspection in constrained single-shot settings (injection detection, in-context learning with labels) rather than naturalistic conversation. Nobody has asked whether a numeric emotive self-report tracks the corresponding probe-defined internal direction across the turns of a real conversation, and nobody has used causal intervention (steering) to confirm the coupling is real rather than correlational. This paper tests that connection.

**What to cite here**: Rahwan et al. (2019) for AI as behavioral subjects. Long et al. (2024) for welfare/safety motivation (one sentence only). Coda-Forno et al. (2023) for the anxiety→bias finding (emotive states have functional consequences). Belinkov (2022) or Hewitt & Liang (2019) for the limitations of probes, briefly. Watson et al. (1988, PANAS) or a general psychometrics reference for the human tradition. Fleming & Lau (2014) for human metacognition measurement as inspiration.

[NOTE: We may need to add some psychometric references beyond PANAS. Classic references on validating self-report scales in humans — e.g., Likert (1932), Russell (1980) for the circumplex model, Csikszentmihalyi & Larson (1987) / Shiffman et al. (2008) for Experience Sampling Method, which is the closest analog to the per-turn methodology. These need to be looked up and cited properly; flagging as a gap to fill or for a targeted literature search.]

### The question (~1 paragraph)

**What to say**: Frame it as a positive question, not a gap-fill.

In human psychology, self-report is validated through convergent evidence: do reports correlate with physiological signals? Do they respond predictably to known interventions? Do they track over time? We apply the same logic to LLMs. We ask: when a model reports a numeric value for an emotive concept — say, how interested or how happy it is — does that number carry information about the model's independently measured internal state along the corresponding concept direction? Does it change in the expected way when we causally shift that state? Does it remain informative across conversational time? And can it be improved?

We know several preliminary things from the literature. LLMs can produce self-reports that look structurally similar to human emotional self-reports (Tavast, Kunnari & Hämäläinen, 2022; Coda-Forno et al., 2022). Emotive states are geometrically structured inside LLMs and causally active (Zhang & Zhong, 2025; Wang et al., 2025; Coda-Forno et al., 2023). And models have some form of introspective access to their internal states (Binder et al., 2025; Ji-An et al., 2025; Lindsey, 2026). But these three things have never been connected into a single empirical test. Nobody has asked the question that the human psychometric tradition asks by default: *does the self-report track the internal state?* — for emotive concepts, quantitatively, across conversational time, with causal validation.

That is what this paper does.

### Operational definition (~2 sentences, within or just after the gap paragraph)

We operationalize introspection as **causal informational coupling between a self-report and the relevant internal state**: a model introspects a concept to the extent that (a) its numeric report about that concept covaries monotonically with an independently measured internal direction associated with that concept, and (b) causally shifting that internal direction shifts the report in the predicted direction.

This definition is empirically testable, agnostic about consciousness or subjective experience, and directly inspired by how metacognition is studied in human cognitive neuroscience — where subjective reports are assessed for fidelity to independently measured neural signals, and causal interventions (such as TMS) are used to confirm the coupling is real (Fleming & Lau, 2014; Lapate et al., 2020; Comsa & Shanahan, 2025).

### Contributions (bullet list)

One-sentence summary first:
*We show that small LLMs (< 10B parameters) can produce self-reports of emotive states that are causally linked to independently measured internal concept directions — demonstrating that numeric self-report is a viable, complementary tool for tracking LLM internal states across conversational time.*

Then bullets. These should feel like discoveries, not methods:

1. **LLMs can perform psychometric self-report of emotive states.** In naturalistic multi-turn conversations, numeric self-reports of happiness, interest, focus, and impulsivity track probe-defined internal state from the first turn, establishing that self-report carries genuine information about internal emotive state.

2. **Default decoding masks this capacity.** Greedy numeric self-reports collapse to one or two values, hiding a rich underlying distribution — an emotive analog of the verbal-internal disconnect documented in the factual confidence literature (Kumar et al., 2024; Xiong et al., 2024).

3. **A logit-based self-report estimator recovers continuous variation.** Computing the probability-weighted expected value over digit-token logits yields a single-pass continuous self-report that preserves information greedy decoding destroys. [To our knowledge, this specific method — computing E[rating] = Σ(i × P(digit_i)) over the logit distribution as a continuous self-report estimator — has not been previously proposed. It is related to logit-based confidence extraction methods (Geng et al., 2024) and the sampled-vs-latent dissociation observed by Pearson-Vogel et al. (2026), but its application to producing continuous numeric self-reports of emotive concepts is new.]

4. **The coupling is causal.** Steering the model along a probe-defined concept direction shifts self-reports monotonically in the predicted direction, confirming that self-reports are not merely correlated with internal state but causally dependent on it.

5. **Introspective fidelity has temporal dynamics.** Probe-report coupling is present from turn 1 but evolves through conversation — increasing for some concepts, decreasing for others.

6. **Introspective fidelity can be selectively improved.** Steering along one concept direction can significantly improve introspection quality for a different concept. This reveals that introspective fidelity is modulable, concept-specific, and decomposable into state-formation and report-readout components — opening the door to systematic optimization.

7. **These phenomena occur in small models and generalize across families.** Probe-report coupling approaches R² ≈ 0.9 in LLaMA-3.2-8B-Instruct for some concepts; Qwen 2.5 7B replicates successfully. Introspective capacity is not exclusive to frontier-scale systems — it exists in models accessible to ordinary researchers.

### Overarching contribution (one sentence after the bullets)

More broadly, this work establishes a framework for treating LLM self-report as a quantitative signal to be validated, understood, and improved, offering a complementary approach to probes and behavioral analysis for tracking internal states in conversational AI systems.

---

## Part III: Related Work — Structure and Narrative

### Overall structure: 3 sections

Confirmed structure:
- Merge emotive states into the introspection section (2.1)
- Merge probing and steering (2.2)
- Keep temporal dynamics lean (2.3)
- No standalone calibration section (fold into 2.1 or 2.2 as needed)
- No neuroscience section in related work (use in intro as inspiration, in discussion as analogy)
- Include psychometric self-report tradition

---

### 2.1 Self-report and introspection of emotive states in language models

**Purpose**: This is the intellectual home of the paper. Establish the landscape, present the question, and show where we sit.

**Narrative flow** (driven by the central question, NOT by the literature camps):

Start with a framing sentence: The question of whether LLMs can inform us about their own internal states through self-report has been approached from multiple angles, but almost exclusively in the domain of factual self-knowledge — models reporting on their own confidence, competence, or accuracy. We argue that *emotive* self-report is at least as important, and substantially less understood.

**Block 1: Factual self-knowledge — established but limited** (~2 paragraphs)

The literature has established that LLMs have partial self-knowledge about factual competence (Kadavath et al., 2022; Yin et al., 2023; Binder et al., 2025). But this work equates "introspection" with "knowing whether your answer is correct." That is a specific, narrow kind of self-knowledge. Meanwhile, the confidence calibration literature has shown that when models are asked to express this knowledge numerically, the reports tend to collapse: verbalized confidence clusters in high ranges, in multiples of five, and systematically misaligns with internal token probabilities (Xiong et al., 2024; Kumar et al., 2024; Yona et al., 2024). This collapse is not unique to emotions — it's a general property of how current LLMs produce numeric outputs — but it has been studied exclusively for factual confidence.

The deeper limitation is conceptual: self-knowledge about correctness treats the model as an oracle to be calibrated, not as a system whose internal states evolve and can be tracked. It doesn't ask whether the model can track a *dynamic internal direction* over time.

**Do not belabor this block. It's context, not the story.**

**Block 2: Emotive states in LLMs — established but disconnected from self-report** (~2 paragraphs)

A separate body of work has studied emotive states in LLMs:

*On the output side*: LLMs produce self-reports that pattern-match human emotional structure (Tavast et al., 2022; Coda-Forno et al., 2022). Inducing affective states (e.g., anxiety) in LLMs shifts downstream behavior in dose-dependent ways (Coda-Forno et al., 2023), establishing that emotive states have real functional consequences. Benchmarks like EmotionBench (Huang et al., 2023) and EmoBench (Sabour et al., 2024) evaluate emotional understanding. Fazzi et al. (2025) tracked emotional expression across multi-turn dialogue using external sentiment analysis. Ishikawa & Yoshino (2025) mapped emotional outputs onto Russell's circumplex model.

*On the internal side*: Zhang & Zhong (2025) used linear probes to show that emotion is geometrically structured in LLM activations — it emerges early, peaks in middle layers, sharpens with model scale, and persists for hundreds of tokens. Wang et al. (2025) went further, identifying causal emotion circuits and achieving 99.65% emotion-expression control via circuit modulation.

**The disconnect**: All output-side work evaluates whether emotional outputs *look right* — not whether they are informed by the corresponding internal state. All internal-side work maps the structure from outside — not whether the model has access to it. Nobody has closed the loop: does the model's own emotive self-report track the geometrically defined internal emotive direction?

**Note why Coda-Forno et al. (2023) on anxiety→bias is particularly important**: it establishes that emotive states are not surface-level text artifacts but have causal behavioral consequences. If states with these consequences can be tracked via self-report, that's practically valuable.

**Block 3: Introspection as internal-state reporting — the closest prior work** (~2 paragraphs)

Several recent papers have studied whether LLMs can report on their internal states at all:

- Comsa & Shanahan (2025) argued conceptually that genuine introspection requires a causal connection between an internal state and the self-report of that state — mimicry of introspective language is insufficient. This motivates our insistence on causal testing via steering.

- Binder et al. (2025) showed that LLMs have genuine privileged self-access: they predict their own hypothetical behavior better than other models can. This establishes that some form of internal-state access exists.

- Ji-An et al. (2025) showed that LLMs can learn to report and control specific activation projections in a neurofeedback-like paradigm, using in-context learning with labeled examples. This is the closest methodological relative — but the paradigm is constrained (labeled examples teach the model what to report), works on arbitrary directions (not emotive concepts), and operates in a single-shot task, not naturalistic conversation.

- Lindsey (2026) showed that very large models (Claude Opus 4) can detect injected concept vectors ~20% of the time — an out-of-distribution detection task, not graded numeric tracking of naturally arising states.

- Pearson-Vogel et al. (2026) showed that a model's residual stream reveals detection of prior concept injections even when sampled text denies it — a latent capacity visible in logits but masked in surface output. This dissociation directly parallels our finding that logit-based extraction recovers what greedy decoding hides.

- Rivera (2025) showed introspective detection can be *trained into* a 7B model, suggesting introspective capacity is malleable — paralleling our finding that steering can improve introspection.

**The question none of these answer**: Can a model, without special training or in-context examples, produce a graded numeric report of a naturally arising emotive state that faithfully tracks an independently measured internal direction — and does it do so across the turns of a naturalistic conversation?

That is the question this paper answers.

**Key contrast papers for motivation/validation need** (citing without making it a "gap list"):

- Han et al. (2025) showed that personality self-reports in LLMs are "illusory" — post-training alignment creates stable, human-like self-reports that are dissociated from actual behavior. This is motivation not just for skepticism, but for *developing better tools to understand and improve self-reports*. If alignment can create illusory coherence, we need methods to tell genuine introspection from mimicry — and potentially to improve the genuine signal.

- Jackson et al. (2025) found that standardized self-assessments don't reflect actual task abilities — they are "learned communication postures." Again: this motivates the framework we propose, not just skepticism.

- Prestes (2025) identified failures of diachronic self-consistency over time, motivating our turn-by-turn temporal analysis.

- Song et al. (2025) showed that metalinguistic introspection fails — models cannot introspect on their own grammatical knowledge. This is important because it shows introspection is domain-dependent; our positive results for emotive concepts acquire meaning against this negative background.

**How to avoid sounding like a gap list**: The above should not be presented as "Paper A didn't do X, Paper B didn't do Y, so we do both." Instead, the narrative should be: *There is a question that the human psychometric tradition takes for granted — does self-report track internal state? — that has simply never been asked for emotive states in LLMs in a naturalistic setting. The pieces exist: we know emotive states are internally structured, we know models have some introspective capacity, we know self-report can be evaluated against independent measurements. Our contribution is to connect these pieces into the closed-loop test that answers the question.*

---

### 2.2 Measuring and intervening on internal representations

**Purpose**: Establish the measurement (probing) and intervention (steering) toolkit. This section is methodological — it supports the *how*, not the *what* or *why*.

**Probing** (~1 paragraph):

The idea that high-level concepts can be represented as linear directions in neural network activation spaces is now well-established. Kim et al. (2018, TCAV) showed concept activation vectors in CNNs; Alain & Bengio (2018) proposed probes as "thermometers" for neural representations; Pimentel et al. (2020) reframed probes as measuring accessibility of information (ease of extraction, not just presence). In LLMs specifically, linear probes have been used to identify truth directions (Burns et al., 2022; Azaria & Mitchell, 2023), spatial and temporal representations (Gurnee & Tegmark, 2023), and emotion geometry (Zhang & Zhong, 2025).

Critically, probes are correlational, not causal (Belinkov, 2022; Hewitt & Liang, 2019). High probe accuracy does not guarantee that the model uses the probed direction. This has driven two responses in the field: controlling for probe complexity (Hewitt & Liang, 2019) and seeking convergent evidence. Our framework contributes to the second response: if the model's own self-report tracks the probe-defined direction, this provides behavioral validation of the probe that is independent of the probe's training data — analogous to how human neuroimaging is validated by correlating measured neural signals with the subject's own report. This convergent validation strengthens both the probe and the self-report.

**Steering** (~1 paragraph):

Activation steering — adding directional vectors to intermediate representations during inference — has emerged as a practical tool for causal intervention on internal state (Turner et al., 2024; Panickssery et al., 2023; Zou et al., 2023; Li et al., 2023). Arditi et al. (2024) provided the clearest single-direction causal result, showing refusal behavior is mediated by one direction. Steering has been applied to control sentiment and style (Konen et al., 2024), personality traits (Frising & Balcells, 2025), truthfulness (Li et al., 2023), and emotion expression (Wang et al., 2025). We use steering for a different purpose: not to control model behavior, but to test and modulate introspective fidelity. If steering a concept direction shifts the model's self-report of that concept in the predicted direction, this constitutes causal evidence that the report is informed by the state, not merely correlated with it.

**The convergent-validation framing is key here.** The paper does NOT claim probes are perfect ground-truth readouts. It uses probes as one measurement channel, self-report as another, steering as a causal test, and random-vector controls as a baseline — the convergence across these independent signals is the evidence.

---

### 2.3 Temporal dynamics in multi-turn conversation

**Purpose**: Establish that multi-turn conversation is the natural setting for deployed LLMs, that internal states evolve through it, and that temporal tracking of introspection is unstudied.

Keep this section lean (~0.5–1 column). The literature is small.

**Behavioral drift is well-documented**: Models show persona drift (Kim et al., 2020; Abdulhai et al., 2025), identity drift that paradoxically increases with scale (Choi et al., 2024), and instruction drift driven by attention decay to system prompts (Li et al., 2024).

**Internal-state drift has recently been measured**: Das & Fioretto (2026, NeuroFilter) introduced "activation velocity" — cumulative drift in internal representations across turns — for privacy-violation detection. Lu et al. (2026, Assistant Axis) identified the leading persona direction and showed it drifts during meta-reflective conversations. These papers show that internal drift is measurable with probes across turns.

**Temporal confounds are real and must be addressed**: Zhang et al. (2026) showed that models inflate self-reported confidence simply from conversation length, not from gaining real information. Guo & Vosoughi (2024) documented serial position effects affecting how models process information at different context positions. Our per-turn probe-based measurement directly addresses this confound: if probe-report coupling exists at individual turns (not just in aggregate), and if it survives random-vector controls, then it is not a time-correlation artifact.

**What we add**: We are, to our knowledge, the first to track the fidelity of emotive self-report against independently measured internal state *through conversational time* — measuring not just whether introspection exists but how it evolves turn by turn.

---

### 2.4 (optional) Numeric self-report methodology

**Note**: Psychometric literature should be cited, particularly the human tradition. This could be a short subsection or woven into 2.1 and methods. Notes for either option:

**The human psychometric tradition**: Numeric self-report has been the workhorse of affect measurement in psychology for nearly a century. Likert (1932) introduced graded rating scales. Watson, Clark & Tellegen (1988, PANAS) established the standard for numeric affect self-report. Russell (1980) proposed the circumplex model of affect (valence × arousal), grounding emotional self-report in a continuous two-dimensional space. The Experience Sampling Method (ESM; Csikszentmihalyi & Larson, 1987; Shiffman, Stone & Hufford, 2008) introduced repeated self-report across time in ecologically valid settings — the closest human analog to our per-turn methodology.

[CITATION GAP: We likely need to add these psychometric references. They are well-known but may not be in the current literature folders. Verify that the following are available or add them:
- Likert, R. (1932). A technique for the measurement of attitudes.
- Watson, D., Clark, L. A., & Tellegen, A. (1988). PANAS.
- Russell, J. A. (1980). A circumplex model of affect.
- Csikszentmihalyi, M., & Larson, R. (1987). Validity and reliability of the experience-sampling method.
- Shiffman, S., Stone, A. A., & Hufford, M. R. (2008). Ecological momentary assessment. Annual Review of Clinical Psychology.]

**LLM numeric reporting**: The confidence calibration literature has shown that when LLMs are asked for numbers, the results are problematic: clustering in high ranges, multiples of five, and systematic disconnect from internal token probabilities (Xiong et al., 2024; Kumar et al., 2024). Our logit-based estimator addresses this by extracting the expected value E[rating] = Σ(i × P(digit_i)) directly from the output distribution — a method that, to our knowledge, has not been previously proposed as a self-report extraction technique. It is related to logit-based confidence methods surveyed by Geng et al. (2024) and to the logit-lens approach of Pearson-Vogel et al. (2026), but differs in purpose (producing a continuous psychological self-report, not detecting injections or measuring factual confidence) and in specific implementation (probability-weighted average over a discrete scale, not binary detection).

---

## Part IV: The Central Scientific Question and What Makes It Novel

The overarching contribution: we study introspection of emotive states in naturalistic conversational settings and show that numeric self-reports can track independently measured internal state — establishing self-report as a viable, complementary probing method for LLM internal states that can be studied, validated, and improved.

> Are we now closer to understanding and using self-reports to track internal states of LLMs quantitatively over time? Yes, and this paper provides the framework.

### The overarching novelty

No prior work tests whether LLM emotive self-reports are informed by the corresponding internal representations — validated against independent measurement, tested causally, and tracked across conversational time. The individual findings below are each novel in their own right, and collectively they demonstrate that self-report is a viable and improvable tool for tracking internal states.

### Individual findings

Each of the following is a distinct novel finding:

**1. Operationalizing introspection as causal probe-report coupling for emotive states.**
No prior work operationalizes introspection as causal monotonic covariation between a numeric self-report and an independently measured internal concept direction for emotive states. Ji-An et al. (2025) demonstrate probe-report coupling but in a constrained, non-naturalistic paradigm with labeled examples and arbitrary directions. Lindsey (2026) tests injection detection, not graded rating of naturally arising states. Comsa & Shanahan (2025) define the causal criterion conceptually but provide no experiments.

**2. The naturalistic conversation setting.**
Nearly all introspection, probing, and steering work is single-shot or uses artificial task settings. Conversation is the setting where internal-state tracking is most needed and most unstudied. Fazzi et al. (2025) are closest (multi-turn emotion), but measure output sentiment, not internal-vs-report coupling.

**3. The logit-based self-report estimator.**
To our knowledge, computing E[rating] = Σ(i × P(digit_i)) over digit-token logits as a continuous self-report extraction method is new. The calibration literature has documented the greedy-collapse problem extensively (Xiong et al., 2024; Kumar et al., 2024) and has proposed various corrections for factual confidence (Manggala et al., 2025; Wang et al., 2024), but none specifically produces continuous emotive self-reports from a single forward pass. The conceptual parallel with Pearson-Vogel et al. (2026) — where sampled text denies what latent signals reveal — is particularly apt.

**4. Causal testing of emotive self-report via steering.**
Steering has been used to change behavior (ITI, RepE, CAA, ActAdd), control emotion expression (Wang et al., 2025), and test introspective detection (Ji-An et al., 2025). Using steering specifically to test whether numeric emotive self-reports are causally dependent on the steered internal direction is novel.

**5. The modulability and decomposability of introspective fidelity.**
No prior work shows that steering one concept direction can improve introspective accuracy for a different concept. This second-order finding — that introspection quality itself is a modulable property — has no precedent. The closest parallel is Steyvers et al. (2025), who showed metacognitive training transfers across domains via fine-tuning. Our result is inference-time and requires no training. The decomposition into state-formation vs. report-readout components is inspired by the monitoring/control distinction in human metacognition (Boldt & Gilbert, 2022; Fleming & Lau, 2014) and is here operationalized for the first time in LLMs.

**6. Small models.**
Lindsey (2026) requires Claude Opus 4. Pearson-Vogel et al. (2026) use Qwen 32B. Our results demonstrate introspective capacity in 1B–8B models. This matters because it establishes that introspective capacity is a more fundamental property of instruction-tuned LLMs, not an emergent capability of frontier systems only, and because it makes this line of research accessible to the broader community.

---

## Part V: Neuroscience and Psychology as Inspiration (not Analogy)

The neuroscience connection should be framed as *inspiration*, not *analogy* and certainly not *equivalence*. We are saying: the same scientific logic that has worked for studying internal states in humans — asking them, validating the answers, using interventions — can be applied to LLMs. We're not saying LLMs are like brains. We're saying the *methodology* transfers.

### In the introduction

When introducing the motivation (~2 sentences): "Our approach draws inspiration from how internal states are studied in human cognitive neuroscience: subjective reports are assessed for their fidelity to independently measured signals, and causal interventions are used to confirm the coupling is real (Fleming & Lau, 2014; Lapate et al., 2020). LLMs, uniquely among AI systems, can be asked — and this paper tests whether asking works."

### In the discussion (1 paragraph, keep it bounded)

- Metacognition in humans is graded and noisy, not all-or-none (Fleming, 2024). Our probe-report coupling is similarly graded — partial, concept-dependent, model-dependent!
- Monitoring (forming the confidence signal) and control (using it to adjust behavior) are dissociable processes in the brain (Boldt & Gilbert, 2022). Our decomposition of introspective fidelity into state-formation and report-readout mirrors this.
- TMS can causally disrupt metacognition in specific brain regions without affecting first-order perception (Lapate et al., 2020). Our steering experiments are the methodological kindred: causally intervening on internal state and measuring the effect on self-report.
- Subjective report and objective performance dissociate in psychophysics (Kiefer & Kammer, 2024) — the model greedy-collapse finding is methodologically parallel: the model "has" the internal state but fails to "report" it without logit extraction.

Frame as: "These parallels are not offered as literal equivalence between human cognition and LLM computation. They are offered as evidence that the *scientific logic* — validating report against state, decomposing into components, using intervention for causal tests — has a long track record, and applying it to LLMs is a natural and productive extension."

---

## Part VI: The Self-Report vs. Probes Tension — How to Handle It

A subtle tension: we are using probes to validate self-report, but probes are themselves imperfect simplifications of internal state. So using an imperfect ground truth to validate another imperfect estimate is... "a start."

### How to frame this honestly without self-undermining:

1. **Neither signal is ground truth.** Linear probes capture what is *linearly accessible* about a concept in a given layer's activations. Self-reports capture what the model's generative process produces when asked about that concept. Both are projections of a richer underlying state.

2. **Their agreement is the evidence.** The logic is convergent validation, not one-way verification. When a probe-defined direction and a self-reported number covary across 400 observations in 40 independent conversations, and this coupling survives random-vector controls and responds predictably to causal intervention, the convergence is informative *even though neither signal is perfect on its own*.

3. **This is exactly how it works in human psychophysics.** fMRI signals are noisy and spatially imprecise; self-reports are noisy and subject to demand effects. Neither is trusted alone. Their convergence is the standard of evidence.

4. **Self-report could in principle capture structure that probes miss.** A linear probe projects onto a single direction. The generative process producing the self-report integrates across the model's full representational capacity. It's conceivable that self-reports are sensitive to aspects of emotive state that a one-dimensional probe cannot resolve. We cannot test this claim with our current methodology, but it motivates future work.

**Where to say this**: Briefly in methods (when introducing the convergent-validation framework), then more fully in discussion/limitations. Do not self-criticize in the introduction. In the intro, the framing is positive: we use convergent validation, the same logic used in psychophysics.

---

## Part VII: Citation Strategy (revised)

### Tier 1 — Must cite, deeply engage with

| Paper | Role |
|---|---|
| Comsa & Shanahan (2025) | Conceptual foundation: genuine introspection requires causal coupling |
| Ji-An et al. (2025) | Closest methodological relative — careful differentiation |
| Lindsey (2026) | Causal state→report link, but detection task in very large models |
| Coda-Forno et al. (2022, 2023) | Emotional self-reports look human-like; anxiety has functional consequences |
| Tavast et al. (2022) | Another demonstration that LLMs produce structured emotion self-reports |
| Zhang & Zhong (2025) | Emotion geometry, persistence — independent validation of probing approach |
| Binder et al. (2025) | Strongest evidence for genuine privileged self-access |
| Han et al. (2025) | Personality illusion — motivates understanding and improving self-reports |
| Kumar et al. (2024) | Verbal-internal disconnect motivating logit extraction |
| Fleming & Lau (2014) | Metacognition measurement framework — inspiration for methodology |
| Watson, Clark & Tellegen (1988) | PANAS — foundational psychometric self-report instrument |

### Tier 2 — Important context, cite with a sentence or two

| Paper | Role |
|---|---|
| Kadavath et al. (2022) | Establishes factual self-knowledge landscape |
| Pearson-Vogel et al. (2026) | Sampled-vs-latent dissociation — methodological parallel |
| Fazzi et al. (2025) | Multi-turn emotion in dialogue — closest in emotive domain |
| Wang et al. (2025) emotion circuits | Causal emotion mechanisms — complements from mechanism side |
| Belinkov (2022) | Probe validity — our paper as part of the solution |
| Zou et al. (2023) RepE | Probing + steering framework ancestor |
| Zhang et al. (2026) multi-turn confidence | Temporal confound our methodology addresses |
| Xiong et al. (2024) | Greedy collapse for factual confidence |
| Lu et al. (2026) Assistant Axis | Internal drift in meta-reflective conversation |
| Das & Fioretto (2026) NeuroFilter | Activation velocity across turns |
| Arditi et al. (2024) | Single-direction causal control — strongest in the field |
| Coda-Forno et al. (2023) | Anxiety→bias: emotive states have real consequences |
| Long et al. (2024) | AI welfare — ethical motivation |
| Russell (1980) | Circumplex model of affect |
| Csikszentmihalyi & Larson (1987) / Shiffman et al. (2008) | ESM — repeated self-report methodology in humans |
| Rivera (2025) | Introspective capacity is trainable — parallels our improvement finding |

### Tier 3 — Brief citation, cluster mentions

Gurnee & Tegmark (2023), Burns et al. (2022), Alain & Bengio (2018), Kim TCAV (2018), Li ITI (2023), Turner ActAdd (2024), Panickssery CAA (2023), Frising & Balcells (2025), Wehner survey (2025), EmotionBench (Huang 2023), EmoBench (Sabour 2024), Ishikawa & Yoshino (2025), Fleming 2024 review, Song et al. (2025) negative result, Ackerman (2025), Rahwan et al. (2019), Choi et al. (2024), Li et al. (2024) instruction drift, DCA (Zhang 2025), Yona et al. (2024), Plunkett et al. (2025), Prestes (2025), Jackson et al. (2025), Boldt & Gilbert (2022), Lapate et al. (2020), Kiefer & Kammer (2024), McClelland (2024), Butlin et al. (2023), Steyvers et al. (2025), Hewitt & Liang (2019), Pimentel et al. (2020), Konen et al. (2024), Likert (1932).

---

## Part VIII: What NOT to Overclaim

### Consciousness
Don't claim to measure consciousness, subjective experience, or "real" emotion. Frame the paper as providing measurement tools and empirical results, not metaphysical conclusions.

### Perfect internal measurement
Linear probes are not perfect readouts. Self-reports are not perfect readouts. Their convergence is the evidence. Say so.

### Universal introspection
Results vary across concepts, models, and scales. Claim that introspective capacity exists, varies, and is modulable — not that it's universal.

### Optimal improvement
Two concept-steering combinations significantly improve introspection. This is proof-of-concept that introspective quality is modulable, not a method for maximizing it. Much more work could and should be done to optimize introspective fidelity — that is a natural and important direction for future research.

### Self-report as replacement for probes
Self-report is *complementary* to probes, not a replacement. It has different strengths (uses the model's own compression, works black-box, scales naturally) and weaknesses (may be influenced by alignment biases, prompt framing, or conversational position). The contribution is adding a tool to the toolbox, not retiring existing tools.

---

*End of notes.*
