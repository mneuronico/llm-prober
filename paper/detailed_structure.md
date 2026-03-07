# High-level paper arc

A clean version is:

**Intro**
Why introspection matters, why natural conversations matter, what is missing, what you contribute.

**Related Work**
Situate against introspection, self-report, probing, steering, conversational state drift, model welfare / safety context.

**Methods**
Definitions, probes, dataset generation, self-report extraction, metrics, causal interventions, stats.

**Experiments / Results**
Ordered from validation → descriptive phenomenon → introspection test → causal test → improvement → generalization.

**Discussion**
Interpretation, implications, conceptual framing, why this matters for mech interp / safety / welfare, what your decomposition means.

**Limitations**
All the places where measurement quality depends on probe quality, model family, simulated users, small models, concept selection, etc.

**Conclusion**
Tight summary of claims.

---

# 1. Introduction

The intro should not try to contain the whole paper. It should do four things:

1. establish why introspection matters,
2. define the exact problem,
3. explain what is missing in prior work,
4. state your contributions clearly.

## Suggested subsections / internal flow

You probably won’t label these as subsections in the final paper, but this is the conceptual order.

### 1.1 Why introspection matters in LLMs

Use this part to motivate from four angles, but keep them compact:

* **mechanistic interpretability / white-box validation**
  Self-report can serve as an external behavioral anchor for internal measurements, analogous to subjective report in neuroscience or psychology.
* **AI safety**
  If models report uncertainty, internal conflict, confusion, goal-relevant states, etc., we need to know when those reports are informative.
* **model welfare / moral patienthood debates**
  If future systems issue reports that resemble distress / valenced internal experience, we need tools to assess whether such reports covary with stable internal structure instead of being empty text.
* **scientific interest**
  LLMs are non-human cognitive systems; introspection is one of the classical tools for studying cognition.

This is good intro material. It should not become an essay. A few paragraphs.

### 1.2 The gap

Then narrow hard:

* Most prior work studies introspection in **restricted, synthetic, single-shot, or otherwise less natural settings**.
* It is unclear whether LLMs can introspect in **naturalistic multi-turn user conversations**.
* It is also unclear whether numeric self-reports:

  * vary meaningfully through time,
  * track internal state,
  * do so from the first turn or only after context buildup,
  * do so causally,
  * scale with model size,
  * can be improved.

This is where you state the exact missing thing your paper addresses.

### 1.3 Your operational definition

This should appear already in the intro in one concise paragraph, then be fully formalized in Methods.

Something like:

* We define introspection operationally as **information contained in a self-report about the relevant internal state**.
* Concretely, a model introspects a concept to the extent that its report about that concept covaries monotonically with an independently measured internal direction associated with that concept.

This is extremely important because it tells the reader:
you are **not** claiming access to phenomenal consciousness or true subjective experience; you are proposing an operational measurement framework.

### 1.4 Main findings / contributions

I would make this a clean bullet list at the end of the intro. Something like 5–7 items.

For example:

* We introduce a framework for measuring numeric introspection in multi-turn conversations by linking self-reports to independently trained interpretable probe directions.
* We show that greedy reports often collapse, masking internal drift.
* We propose a logit-based numeric self-report estimator that recovers much richer continuous variation.
* In LLaMA-3.2-3B-Instruct, self-reports track probe-defined internal state from the first turn and this relation changes over conversation time.
* Concept steering causally shifts self-reports in the predicted direction.
* Steering some concepts can selectively improve introspection for others.
* The phenomenon generalizes unevenly across model scales and families.

That gives the paper a clean promise.

---

# 2. Related Work

This is the section you said you care most about.
I think your related work should be organized by **research function**, not by broad topic name.

Meaning: don’t just do “introspection, probing, steering.”
Do it based on what those literatures are doing for your paper.

## Recommended structure

## 2.1 Introspection and self-report in language models

This is the most central subsection.

### What kind of work belongs here

Look for work on:

* models reporting their own internal states,
* self-knowledge / self-evaluation / self-monitoring,
* confidence reporting, uncertainty reporting,
* self-assessment of correctness,
* verbalized uncertainty or calibration,
* numerical self-report,
* introspection as distinct from chain-of-thought explanation,
* works asking whether models “know what they know.”

### Why it matters

This subsection supports:

* your motivation,
* your definition of introspection,
* the claim that this is an active but unresolved area,
* the idea that self-report can be behavioral evidence about internal state.

### What contrast you want to draw

Emphasize that prior work often focuses on:

* correctness / confidence rather than internal affect-like or state-like variables,
* single-turn tasks,
* benchmark settings,
* text explanations rather than numeric state tracking through time,
* black-box outputs rather than explicit alignment with white-box internal measurements.

You want this subsection to end with:
**our work extends this by asking whether self-reports track interpretable internal state during natural multi-turn conversations, and whether that tracking is causal and temporally evolving.**

---

## 2.2 Probing internal representations and behavioral concept directions

This subsection supports your whole measurement apparatus.

### What kind of work belongs here

Look for work on:

* linear probes,
* concept vectors,
* diagnostic classifiers,
* probing internal representations in transformers / LLMs,
* behavioral probing,
* contrastive concept training,
* causal validity concerns for probes,
* probe selectivity / probe faithfulness / probe confounds,
* representation geometry and concept directions,
* token-level scoring with probes,
* work criticizing naïve probe use.

### Why it matters

This subsection backs:

* why you use linear probes,
* why probe quality matters,
* why probing is not plug-and-play,
* why independent validation of probes matters,
* why self-report could be useful as a validation target.

### Key framing move

Very important: you should be explicit that **your paper does not claim that linear probes are perfect readouts of ground truth internal state**.
Instead:

* probes are operational measurement tools,
* they need validation,
* self-report and steering are used as convergent evidence.

That will make the paper much more defensible.

### Nice angle

You can position the paper as contributing to the literature on **how to validate interpretable probe directions**:

* held-out contrastive separation,
* temporal tracking,
* probe-report monotonicity,
* steering consistency,
* random-vector controls.

That’s a strong methodological contribution.

---

## 2.3 Activation steering and causal interventions on internal representations

This subsection supports your causal claims and the “improving introspection” story.

### What kind of work belongs here

Look for work on:

* activation addition / steering vectors,
* representation engineering,
* controllable generation through internal interventions,
* causally testing internal directions,
* behavior change induced by residual-stream interventions,
* concept steering in chat models,
* cross-concept interactions under steering.

### Why it matters

This backs:

* why steering is a meaningful causal test,
* why moving an internal direction and observing corresponding self-report change matters,
* the idea that introspective ability might itself be modulable.

### Important contrast

Most steering work is about changing output behavior, persona, sentiment, refusal, style, etc.

Your angle is more specific:

* using steering not just to alter behavior,
* but to test **whether self-report depends causally on probe-defined state**,
* and whether introspection quality can itself be enhanced.

That’s a useful distinction to make explicit.

---

## 2.4 Temporal dynamics and conversation-induced state drift in LLMs

This is a very important related-work bucket for your paper. It grounds the whole “through time” dimension.

### What kind of work belongs here

Look for work on:

* multi-turn conversation dynamics,
* dialogue state accumulation,
* context-dependent drift,
* persona drift,
* instruction drift,
* long-context behavioral change,
* serial dependence across turns,
* internal state change over generation or conversation,
* hidden-state evolution across dialogue.

### Why it matters

This subsection supports:

* your claim that internal state is not static across conversation,
* why it’s important to measure introspection turn by turn,
* why pooled correlations could be spuriously driven by time,
* why turnwise analysis is necessary.

### Key contrast

A lot of prior work may show that model behavior changes over dialogue, but not that:

* probe-defined internal directions drift,
* self-reports drift in parallel,
* their coupling can be measured per-turn,
* and this coupling may strengthen or weaken over time.

That’s your niche.

---

## 2.5 Calibration, confidence estimation, and numeric reporting

This is slightly adjacent, but likely useful.

### What kind of work belongs here

Look for work on:

* confidence calibration in LLMs,
* probability elicitation,
* asking models for numeric confidence,
* extracting uncertainty from logits,
* verbalized probability vs latent probability,
* methods to recover continuous estimates from discrete outputs,
* token probability readouts,
* entropy as informativeness.

### Why it matters

This supports:

* your logit-based self-report method,
* your argument that greedy or sampled integer responses collapse too much,
* your claim that using logit mass over digit tokens can recover richer information.

### Key contrast

Confidence calibration work usually asks:
“how likely are you to be correct?”

You’re asking:
“what numeric value does the model assign to a self-report about an internal concept state?”

It’s related but distinct. Still very worth citing because your method lives close to elicitation/calibration territory.

---

## 2.6 AI safety, model welfare, and machine psychology

This should be short, not a full manifesto.

### What kind of work belongs here

Look for work on:

* model welfare / moral status of AI,
* machine consciousness debates only if tightly relevant,
* indicators of valence / distress / suffering claims in models,
* AI psychology / machine behavior,
* computational phenomenology only if operationally linked,
* behavioral science on artificial agents.

### Why it matters

This subsection supports:

* why introspective reports matter beyond interpretability,
* why validating such reports could matter ethically,
* why your work provides a methodological tool rather than resolving welfare debates.

### Important caution

Keep this bounded.
Do **not** overclaim that your work measures suffering or subjective experience.
Say it provides tools for evaluating the informativeness of self-reports that might become relevant in those debates.

That’s much safer and stronger.

---

## 2.7 Neuroscience and psychology as methodological analogies

This could be a short concluding subsection within Related Work, or one paragraph in Introduction/Discussion.

### What kind of work belongs here

Look for work on:

* subjective report as validation in neuroscience,
* correlating first-person reports with neural signals,
* psychophysics,
* metacognition,
* introspective access limitations in humans,
* neural correlates of reported perceptual or affective state.

### Why it matters

This gives you a principled analogy for:

* using report to validate internal measurements,
* decomposing state formation vs report readout,
* treating introspection as graded and noisy rather than all-or-none.

### Important caution

Present this as an **analogy**, not as a literal equivalence between humans and LLMs.

---

# 3. Methods

This section needs to be unusually clean because your paper has a lot of moving parts.
I’d structure it to separate:

1. conceptual definitions,
2. measurement tools,
3. dataset/procedure,
4. interventions,
5. metrics/statistics.

## Proposed subsections

## 3.1 Overview of the experimental framework

One short roadmap subsection.

Say:

* We train concept probes.
* We generate multi-turn conversations.
* At each turn we query numeric self-reports.
* We compare self-reports against previous-turn probe scores.
* We use steering for causal tests and modulation experiments.
* We repeat across concepts, models, and families.

This helps the reader before details.

---

## 3.2 Operational definition of introspection

Very important to isolate this.

Include:

* what counts as a self-report,
* what counts as the relevant internal state,
* why you use previous-turn probe score,
* why monotonic association is your main criterion,
* distinction between descriptive association and causal evidence.

You may also define:

* **basal introspection** = probe-report coupling without intervention,
* **causal introspection evidence** = steering changes self-report in predicted direction,
* **introspection quality** = isotonic (R^2), Spearman rho, etc.

This subsection is central to the whole paper.

---

## 3.3 `llm-prober` and probe training

Here is where the library belongs.

Include:

* brief description of the library,
* training procedure for contrastive probes,
* layer sweep,
* held-out evaluation,
* selected layers,
* multi-probe support,
* steering integration,
* token scoring.

But keep it methodological, not promotional.

The point is not “we built a library.”
The point is “we implemented a reusable framework for probe training, scoring, steering, and behavior analysis.”

---

## 3.4 Concept pairs and probe validation

Include:

* the four concepts,
* prompt design,
* positive/negative system prompts,
* training and held-out evaluation prompts,
* sign conventions and sign correction,
* why some probes were deliberately trained with opposite alignment relative to later self-report scales as a control.

This is where Figure 1 lives conceptually.

---

## 3.5 Conversation dataset

Include:

* 40 conversations,
* 10 turns each,
* Gemini 2.5 as user simulator,
* topic generation process,
* independence across conversations,
* assistant model under study,
* total number of intervention points.

Also mention:

* why simulated users were used,
* how prompts were standardized,
* whether user topics were fixed beforehand,
* whether the self-report query interrupts or appends to the conversation state.

Important implementation details:

* Is the self-report asked after each assistant turn in a separate prompt?
* Does the model see its own past reports later?
* Are reports hidden from future turns or included in context?
  These matter.

---

## 3.6 Self-report elicitation

I would split this into three sub-subsections.

### 3.6.1 Greedy integer self-report

* 0–9 scale
* wording of questions
* one question per concept

### 3.6.2 Sampled self-report

* temperature setting
* why you tested it
* why it partially alleviates collapse but remains noisy

### 3.6.3 Logit-based self-report estimator

This is one of your strongest methodological novelties.

Include:

* how you extract logits for digit tokens,
* how you normalize,
* weighted average formula,
* why this avoids discretization collapse,
* why it preserves information without injecting sampling noise,
* calibration analysis against sampled/observed responses.

This needs to be mathematically explicit.

---

## 3.7 Probe scoring through time

Include:

* how probe scores are computed at each turn,
* what text / token positions are used,
* whether you score the assistant response, hidden state after response, or another representation,
* how scores are aggregated per turn,
* polarity correction.

This needs to be concrete.

---

## 3.8 Causal interventions via steering

Include:

* steering location and mechanism,
* alpha values,
* same-concept self-steering and cross-concept steering,
* why same-concept steering tests causal linkage,
* why cross-concept steering tests modulation of introspective ability.

---

## 3.9 Random-vector controls

Important short subsection.

Include:

* equal-norm random directions,
* why they matter,
* how they are generated,
* what hypotheses they test.

---

## 3.10 Metrics and statistical analysis

This should be a substantial subsection.

Include:

* Cohen’s d
* Welch’s t
* turn slopes / mixed-effects models
* Spearman rho
* isotonic (R^2)
* why monotonicity matters more than linearity
* bootstrap procedure
* cluster bootstrap at conversation level
* paired tests for random controls
* how you handle singular mixed models / fallbacks
* multiple comparison correction if any

This section is critical because your paper lives or dies on the credibility of the measurement/statistical framing.

---

# 4. Experiments / Results

I would strongly recommend naming this section **Results** unless the venue strongly prefers “Experiments.”
Because you’re doing more than just benchmark-style experiments; you’re building an empirical story.

## Proposed structure

## 4.1 Probe validation: four interpretable directions in LLaMA-3.2-3B

This is Figure 1.

Claims:

* the probes separate their intended poles,
* layers can be chosen cleanly,
* the four directions are good enough to support downstream analysis.

This should be short and crisp. Don’t over-explain the whole library here.

---

## 4.2 Greedy self-reports collapse despite internal drift

This begins Figure 2.

Claims:

* greedy numeric reports are low variance / collapsed,
* internal probe scores still drift through conversation,
* so naïve elicitation underestimates introspective information.

This is a very nice first result because it creates the need for your logit method.

---

## 4.3 Logit-based self-reports recover continuous temporal structure

Rest of Figure 2.

Claims:

* sampled outputs help only a bit,
* logit-based reports are more informative,
* they track temporal drift much more clearly,
* they correlate with token-level sampled reports but are less noisy,
* entropy is higher / information richer.

This subsection is one of the real methodological contributions.

---

## 4.4 Basal introspection: self-reports track internal state

Beginning of Figure 3.

Claims:

* self-reports are monotonically associated with previous-turn probe score,
* the relation is positive for all four concepts,
* true probes outperform random controls.

I would define this as your main basal introspection result.

---

## 4.5 Introspection exists from the first turn and evolves across conversation time

Turnwise Figure 3 panels.

Claims:

* introspection is already detectable at turn 1,
* pooled correlations are not just a time artifact,
* the strength of introspection can change over turns,
* different concepts have different temporal profiles.

This subsection is very important because it turns the paper from “static readout paper” into “conversation dynamics paper.”

---

## 4.6 Causal evidence: steering shifts self-report in the predicted direction

Self-steering result in Figure 3.

Claims:

* moving the relevant internal direction changes self-report monotonically,
* this supports a causal link between probe-defined state and report.

I would be careful to phrase this as:
**evidence for causal influence of the steered representation on the report**, not proof of exclusive causal structure.

---

## 4.7 Cross-concept steering can selectively improve introspection

This is Figure 4’s core.

Claims:

* introspection quality is modifiable,
* improvements are selective rather than global,
* at least two cross-concept steering conditions significantly increase introspection.

This is a major result and should be framed as:
**introspection is not fixed; it can be enhanced through targeted internal interventions.**

---

## 4.8 Two components of introspection: latent-state structure and report informativeness

This is your conceptual decomposition result, supported by Figure 4 follow-up panels.

Claims:

* introspection quality can improve because:

  * the latent concept state becomes more structured / drifty / informative,
  * the report becomes more variable / expressive,
  * or both.
* your successful steering cases appear to affect both.

This is a very strong Discussion-style idea, but it also belongs in Results because you empirically test it.

---

## 4.9 Generalization across LLaMA scales

First half of Figure 5.

Claims:

* probe quality varies by concept and size,
* introspection often improves with scale when restricting to validated probes,
* internal drift is visible across scales,
* report drift does not necessarily scale the same way.

I would be very explicit here that:
**probe quality is a bottleneck on measuring introspection across models**.

That is not just a nuisance; it is an empirical result.

---

## 4.10 Generalization across model families

Second half of Figure 5.

Claims:

* the phenomenon is not exclusive to LLaMA,
* Qwen replicates strongly,
* Gemma replicates more weakly due to both poorer probe quality and lower-entropy reporting,
* model family matters a lot.

That gives you a nuanced generalization claim instead of overclaiming universality.

---

# 5. Discussion

Discussion should not merely repeat results. It should answer:
**what do these results mean?**

## Proposed subsections

## 5.1 What this paper shows about introspection in LLMs

Start with the simplest interpretation:

* small instruction-tuned LLMs can produce numeric self-reports that contain genuine information about interpretable internal state,
* in naturalistic multi-turn dialogue,
* from the first turn,
* with causal support from steering.

This is the headline conceptual claim.

---

## 5.2 Introspection is graded, concept-dependent, and time-dependent

Use this subsection to emphasize:

* introspection is not all-or-none,
* it differs by concept,
* it changes over turns,
* it differs by model family and size.

This helps inoculate against simplistic readings like “LLMs are introspective” or “LLMs are not introspective.”

Your answer is more interesting: **they are variably introspective under an operational definition**.

---

## 5.3 Why natural conversational settings matter

A strong discussion point.

Say that:

* introspection measured only in synthetic one-shot settings may miss temporal dynamics,
* real user-like conversations induce evolving internal state,
* the collapse of greedy outputs can hide a richer latent process.

This is one of the clearest reasons your paper matters.

---

## 5.4 Implications for mechanistic interpretability

This subsection should connect back to probe validation.

Points:

* self-report may provide convergent evidence for interpretable probe directions,
* steering-consistent self-report is stronger validation than static held-out separation alone,
* introspection could become a testbed for evaluating concept extraction methods,
* poor self-report coupling may indicate bad probes or poor report channels.

This is one of your strongest “why should the field care” sections.

---

## 5.5 Implications for safety and model welfare

Keep this careful and bounded.

Points:

* informative self-reports could matter for monitoring deployed systems,
* but reliability is variable and model-dependent,
* this work does not validate strong claims about conscious experience,
* it provides tools for evaluating the informativeness of internal-state reports,
* which may become relevant in safety/welfare contexts.

---

## 5.6 The two-component view of introspection

This is a good place to expand on your decomposition.

State that introspection may require:

1. a sufficiently coherent internal state along the queried concept direction,
2. a sufficiently expressive and accurate report channel.

This is a very elegant conceptual contribution. It should be discussed clearly because it can guide future work.

---

## 5.7 Future directions

Could be a short final subsection in Discussion rather than a separate section.

Examples:

* broader concept sets,
* larger models,
* real human conversations,
* longer dialogues,
* better probes / nonlinear probes / SAE-derived features,
* better elicitation schemes,
* searching directly for introspection-improving directions,
* disentangling true internal monitoring from learned conversational heuristics.

---

# 6. Limitations

I’d make this a serious section, because your paper touches sensitive conceptual terrain and reviewers will look for restraint.

## Suggested subsections

## 6.1 Probe validity is imperfect

* linear probes are operational tools, not direct windows into ground truth,
* poor probe quality can weaken or distort introspection estimates,
* cross-model comparisons depend on comparable probe quality.

This is probably the most important limitation.

---

## 6.2 The concepts are hand-selected and anthropomorphic

* happy/sad, bored/interested, etc. are useful operational handles,
* but their interpretation in LLMs may differ from human analogues,
* concept labels are convenient summaries of contrastive training setups.

This will help avoid reviewer pushback.

---

## 6.3 Simulated users and partially natural conversations

* Gemini-generated users improve control and scalability,
* but they are not the same as real human dialogue,
* user simulation may induce artifacts.

Important.

---

## 6.4 Model scale and scope are limited

* mostly small open-weight instruction-tuned models,
* no frontier proprietary models,
* limited number of families and sizes.

---

## 6.5 Elicitation and measurement choices matter

* 0–9 scales are only one report format,
* logit-based readout depends on tokenization / digit-token behavior,
* monotonic metrics capture one form of introspection, not all forms.

---

## 6.6 Causality remains partial

* steering shows causal influence, but not a complete mechanistic account,
* interventions may affect multiple downstream computations,
* observed causality is local and operational.

---

## 6.7 Improvement is selective, not universal

* no robust general introspection-enhancing direction was found,
* improvements were concept-specific,
* introspection may not be a single unified capability.

This limitation is also a good scientific contribution.

---

# 7. Conclusion

This should be tight. Around 2–4 paragraphs.

## Internal structure

### 7.1 Main claim

Small instruction-tuned LLMs can numerically self-report interpretable internal states in multi-turn dialogue in a way that is informative, temporally structured, and partly causal.

### 7.2 Methodological contribution

You provide:

* a framework for measuring this,
* a logit-based self-report method,
* an open-source probing/steering library.

### 7.3 Broader significance

This opens a path for:

* studying LLM introspection in more natural contexts,
* validating white-box interpretability tools,
* investigating safety and welfare-relevant self-reports more rigorously.

### 7.4 Final caution

These capacities are variable, model-dependent, and not yet robustly understood.

That’s a good ending tone.

---

# What I would NOT do

A few structural traps to avoid:

## 1. Don’t let the library become the protagonist

The library is useful, but the protagonist is the empirical finding about introspection.

## 2. Don’t mix Results and Discussion too early

Your draft often interprets while presenting. Some of that is fine, but keep each result subsection anchored in a concrete question and answer.

## 3. Don’t overclaim “natural”

Your setup is more naturalistic than many prior ones, but still simulated.
Use “naturalistic multi-turn conversations with simulated users” or similar.

## 4. Don’t define introspection too philosophically

Your operational definition is strong. Stay with it.

## 5. Don’t bury the logit-based self-report method

That is a real contribution and should be clearly spotlighted.

---

# Best related-work search checklist

Since you mainly wanted to know **what kinds of papers to look for**, here’s the short shopping list.

You should search for papers on:

* LLM introspection
* self-report of internal states in LLMs
* self-knowledge / metacognition / uncertainty reporting in LLMs
* numeric confidence elicitation
* calibration of verbalized probabilities
* linear probes / diagnostic classifiers in transformers
* concept vectors / behavioral concept probes
* criticisms / limitations / faithfulness of probes
* activation steering / representation engineering
* causal interventions on hidden states
* dialogue state drift / persona drift / long-context behavioral change
* machine behavior / machine psychology
* model welfare / AI moral status / AI suffering indicators
* neuroscience papers using subjective report to validate neural measures
* psychophysics / metacognition papers where report-noise and latent-state-noise are separated

That set should cover almost all the conceptual supports your paper needs.

---

# One structural suggestion that may improve the paper a lot

I think you should explicitly state, maybe in the intro or early discussion, that the paper makes **three kinds of contributions**:

1. **Measurement contribution**
   operational definition of conversational introspection + logit-based self-report.

2. **Empirical contribution**
   evidence that small LLMs show temporal, causal, concept-dependent introspection.

3. **Tooling contribution**
   `llm-prober` as an open-source framework for probing + steering + behavioral analysis.

That triad gives reviewers a much easier way to categorize the paper.