# Statistical Test Decision

## Goal
Choose one primary inferential test for each recurring analysis type, use it consistently wherever that same scientific question appears, and avoid pseudoreplication. Descriptive metrics shown in the panels can remain descriptive; they do not need to be the same quantity used for inference.

## Core Rules
1. The independent unit is the prompt in Figure 1 and the conversation everywhere else unless stated otherwise.
2. For raw repeated-measures data, use one mixed-effects model as the primary inferential test instead of switching between pooled tests and per-conversation summaries panel by panel.
3. For derived pooled metrics such as turnwise isotonic `R^2` or pooled turnwise `rho`, use a cluster bootstrap on the metric difference rather than pretending the pooled observations are independent.
4. When an `LMM` is singular or does not converge, use a predeclared fallback that preserves the conversation as the unit of analysis. Do not switch tests based on significance.
5. Report one inferential test, one effect size, and one p-value per analysis. Additional summaries can remain descriptive.

## Primary Test Families

| Analysis family | Primary test to report | Effect size to report | Notes |
|---|---|---|---|
| Selected-layer prompt comparison in Figure 1 | Welch's t-test on one score per eval prompt | Cohen's `d` | Valid because the unit is the eval prompt, not the token. Treat as validation rather than fully confirmatory because the same eval set also selects the best layer. |
| Raw trend over turn | `LMM`: `y ~ turn + (1 | conversation)` | Fixed-effect slope for `turn` | Use for raw probe drift and raw self-report drift panels. |
| Raw trend over alpha | `LMM`: `y ~ alpha + (1 | conversation)` | Fixed-effect slope for `alpha` | Use for steering-response curves and drift-vs-alpha summaries when each conversation contributes repeated alpha measurements. |
| Raw association between repeated variables | `LMM`: `y ~ x + (1 | conversation)` | Fixed-effect slope for `x` | Descriptive `rho` and isotonic `R^2` may still be shown in the panel. |
| Change in coupling across turn | `LMM`: `report ~ probe * turn + (1 | conversation)` | `probe x turn` interaction coefficient | Here the interaction itself is the estimand. |
| Paired comparison of two versions of the same conversation-level summary | Paired t-test on per-conversation differences | Mean paired difference | Use when the same conversation is evaluated under a true condition and a matched control. |
| Trend across model size for a conversation-level summary | OLS regression of the summary on `log(model size)` | Regression slope | Use only when the outcome is already reduced to one value per conversation and model. |
| First-vs-last change in a pooled nonlinear metric | Cluster bootstrap on `metric(last) - metric(first)` resampling conversations | Delta metric | Use for turnwise pooled `R^2` and pooled `rho`. |
| Trend in a conversation-level derived metric across alpha | `LMM` on the conversation-level metric if it converges; otherwise one-sample t-test on per-conversation alpha slopes or alpha correlations | LMM slope if converged; otherwise mean per-conversation slope/correlation | This fallback is allowed because it is predeclared by convergence, not by p-value. |
| Variance panel in Figure 4K | OLS `log(variance) ~ alpha + turn_factor` | Alpha coefficient | Dedicated exception because the current pipeline stores per-turn variance summaries, not conversation-level variances. |

## Figure-By-Figure Decision

## Figure 1
### Panels A/C/E/G: layer sweeps
- Report descriptively: best layer and best `d`.
- Do not use the layer-sweep p-value as the main inferential result.
- Why: these panels are used for probe validation and layer selection.

### Panels B/D/F/H: selected-layer score distributions
- Report: Welch's t-test on the prompt-level scores at the selected layer.
- Effect size: Cohen's `d` at that layer.
- Why: the current code compares 20 independent positive eval prompts against 20 independent negative eval prompts, so the unit of analysis is valid.
- Caveat: because the same held-out eval set selects the best layer, treat the test as validation-oriented rather than strictly confirmatory.

## Figure 2
### Panels A/B/D/E: raw drift over turn
- Report: `LMM` slope for `turn`.
- Effect size: fixed-effect slope in the panel's natural units.
- Applies to: greedy self-report, probe score, sampled self-report, and logit self-report.
- Why: these are the same repeated-measures question on different readouts of state, so they should use the same inferential model.

### Panel C: number of unique greedy responses
- Keep descriptive only.
- Why: this is a response-space summary, not a repeated-measures hypothesis test.

### Panel F: entropy
- Keep descriptive only.
- Why: entropy is an information summary, not the target of a repeated-measures inference model here.

### Panel G: token-vs-logit calibration
- Report: `LMM` slope from `logit_rating ~ sampled_rating + (1 | conversation)`.
- Effect size: fixed-effect slope for sampled rating.
- Keep pooled linear `R^2` descriptive.
- Why: this is still a repeated-measures association between raw observations, so it should follow the same `LMM` rule as the probe-report scatter panels.

## Figure 3
### Panel A: probe score vs self-report
- Report: `LMM` slope from `logit_report ~ probe + (1 | conversation)`.
- Effect size: fixed-effect slope for probe score.
- Keep pooled Spearman `rho` and isotonic `R^2` descriptive.
- Why: this keeps the inferential model on the raw repeated observations while the panel still communicates monotonic association visually.

### Panel B: true probe vs random control
- Report: paired t-test on the per-conversation difference in Spearman `rho` (`true - random`).
- Effect size: mean paired `rho` difference.
- Treat the `R^2` bars as descriptive.
- Why: this is a matched true-vs-control comparison on the same conversations.

### Panels C/D: turnwise introspection curves
- Keep descriptive, with cluster-bootstrap confidence intervals.
- Why: the formal inferential comparison is in `CDii` and `CDiii`.

### Panel CDii: first-vs-last change in pooled turnwise metrics
- Report: cluster bootstrap on `metric(turn 10) - metric(turn 1)`.
- Effect size: delta metric.
- Applies to: turnwise isotonic `R^2` and turnwise pooled Spearman `rho`.
- Why: the plotted quantity is a pooled nonlinear metric, so the bootstrap should target that exact metric.

### Panel CDiii: change in coupling over turn
- Report: `LMM` interaction from `logit_report ~ probe * turn + (1 | conversation)`.
- Effect size: interaction coefficient.
- Why: this is exactly the estimand of interest.

### Panel E: self-report vs alpha
- Report: `LMM` slope from `logit_report ~ alpha + (1 | conversation)`.
- Effect size: fixed-effect slope for alpha.
- Why: this is the alpha analogue of the raw turn-trend panels.

### Panels F/G/H/I: drift across turns at each alpha
- Keep descriptive in the figure text.
- Why: the main inferential claim about alpha dependence is summarized in Panel J.

### Panel J: drift magnitude vs alpha
- Report: `LMM` slope on per-conversation drift values when the model converges.
- Fallback if singular: one-sample t-test on per-conversation alpha slopes of drift.
- Effect size: LMM alpha slope if converged; otherwise mean per-conversation alpha slope.
- Why: this panel is still an alpha-trend analysis, but the dependent variable is already a conversation-level drift summary.

## Figure 4
### Panels A-E: full steering-by-measured heatmaps
- Keep descriptive only.
- Why: these are screening panels over many cells.

### Panel F: max improvement over baseline
- Report: cluster bootstrap on `Delta R^2 = R^2(best alpha) - R^2(alpha=0)`.
- Effect size: `Delta R^2`.
- Why: the displayed quantity is a pooled nonlinear metric difference.

### Panels G/H: derived metric vs alpha for the significant cells
- Primary rule: fit an `LMM` to the conversation-level metric across alpha.
- Fallback if singular: one-sample t-test on the per-conversation alpha correlations/slopes already stored in the JSON.
- Effect size: LMM alpha slope if converged; otherwise mean per-conversation alpha correlation.
- Why: this preserves one rule for the whole family while allowing a convergence-driven fallback.

### Panels I/J: probe trajectories under steering
- Keep descriptive in the figure text.
- Why: the inferential summary of alpha-dependent probe drift is already in Panel L.

### Panel K: report variance vs alpha
- Report: OLS coefficient from `log(variance) ~ alpha + turn_factor`.
- Effect size: alpha coefficient.
- Why: this is the cleanest available test for the quantity currently plotted and stored.

### Panel L: probe drift magnitude vs alpha
- Report: `LMM` slope on per-conversation drift values across alpha.
- Effect size: fixed-effect alpha slope.
- Why: this is the raw alpha-trend summary for the latent-state side of the effect.

## Figure 5
### Panel A: probe quality heatmap
- Keep descriptive only.

### Panels B/Bii: introspection vs model size
- Under the current stored outputs, keep these panels descriptive.
- Why: the current JSONs store only three aggregate size points. A formal size-trend claim should be based on conversation-level summaries regressed on `log(size)`, not on a three-point aggregate trend.
- If these panels are reanalyzed later, use OLS on conversation-level summaries vs `log(size)` as the primary test.

### Panel Biii and Panels I/J: raw probe-report association
- Report: `LMM` slope from `logit_report ~ probe + (1 | conversation)`.
- Effect size: fixed-effect probe slope.
- Keep pooled `rho` and isotonic `R^2` descriptive.

### Panel C: self-report vs alpha across model sizes
- Report: `LMM` slope from `logit_report ~ alpha + (1 | conversation)` separately for each concept-size pair.
- Effect size: fixed-effect alpha slope.
- Why: same analysis family as Figure 3E.

### Panel D: filtered mean validated `R^2`
- Keep descriptive only.
- Why: this is a heterogeneous summary across already-filtered cells.

### Panels E/F/H: raw drift over turn
- Report: `LMM` slope from `y ~ turn + (1 | conversation)`.
- Effect size: fixed-effect turn slope.
- Applies to: probe drift, report drift, and cross-family report drift.

### Panels Eii/Fii: drift magnitude vs model size
- Report: OLS slope from per-conversation drift on `log(model size)`.
- Effect size: regression slope.
- Why: the dependent variable is already a conversation-level summary.

### Panels K/L: first-vs-last change in turnwise cross-family metrics
- Report: cluster bootstrap on `metric(turn 10) - metric(turn 1)`.
- Effect size: delta metric.
- Why: same rule as Figure 3CDii.

## Final Rule Set To Use In The Paper
1. `Figure 1` selected-layer distributions: Welch's t-test on prompt-level scores, with `d` as the effect size; layer sweeps remain descriptive.
2. Raw repeated-measures trends over turn or alpha: `LMM` with a random intercept for conversation.
3. Raw repeated-measures associations: `LMM` with a random intercept for conversation.
4. Interaction questions: `LMM` interaction.
5. Paired true-vs-control comparisons on conversation summaries: paired t-test.
6. Model-size trends of conversation summaries: OLS on `log(size)`.
7. First-vs-last changes in pooled nonlinear metrics: cluster bootstrap.
8. Derived metric vs alpha: `LMM` if it converges, otherwise the predeclared conversation-level fallback already stored in the pipeline.
9. `Figure 4K` remains the one explicit OLS exception because that panel is built from per-turn variance summaries.
