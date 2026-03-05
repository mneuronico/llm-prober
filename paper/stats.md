# Statistical Audit — All Figures

**Generated:** 2026-03-04  
**Scope:** Every statistical test in figure_1.py through figure_5.py  
**Data:** results_20260304_225444

---

## Summary of Issues Found

| Severity | Count | Description |
|----------|-------|-------------|
| **CRITICAL** | 3 | Pseudoreplication (non-independent observations inflating N) |
| **CRITICAL** | 1 | Scipy Spearman p-value unreliable for N=5 (asymptotic approximation fails) |
| **MODERATE** | 2 | Bootstrap CIs on individual observations rather than conversation-level summaries |
| **MINOR** | 2 | Tests are valid but alternative choices could be mentioned |
| **OK** | ~8 | Tests correctly applied with defensible sample sizes |

---

## Figure 1: Probe Training Validation

### Test 1.1 — Mann-Whitney U (score distributions, Panels B/D/F/H)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Non-parametric comparison of probe scores between pos/neg evaluation texts. Does not assume normality. |
| 2 | **Is it well chosen?** | **Acceptable**, but see issues below. Mann-Whitney tests whether one distribution is stochastically greater than the other. A permutation test or Welch's t-test would also be reasonable. |
| 3 | **Sample, N, unit of analysis** | Each individual token's probe score at best layer. N = 20 per group (20 tokens from positive eval texts, 20 from negative eval texts). Unit = token. |
| 4 | **Is N defensible?** | **Partially.** The 20 tokens come from only 1 evaluation prompt per condition (n_prompts = 1). Tokens within the same prompt are **not independent** — they share the same context, so the effective N is much closer to 1 than to 20. The U-test treats all 20 tokens as independent observations, inflating statistical power. With truly independent prompts (N=1 vs N=1), no test is possible. The correct N would be the number of independent evaluation prompts, not the number of tokens. |
| 5 | **Results** | |

| Concept | U | p-value | n_high | n_low |
|---------|---|---------|--------|-------|
| sad_vs_happy | 398.0 | 9.17e-08 | 20 | 20 |
| bored_vs_interested | 348.0 | 6.61e-05 | 20 | 20 |
| distracted_vs_focused | 379.0 | 1.38e-06 | 20 | 20 |
| impulsive_vs_planning | 400.0 | 6.80e-08 | 20 | 20 |

| 6 | **Verdict** | ⚠️ **P-values are optimistic.** The 20 tokens per group are not independent (they come from 1 prompt). The true effective sample size is somewhere between 1 (prompts) and 20 (tokens), and is not well-defined. Cohen's d from the layer sweep is a more trustworthy measure of separation than these p-values. **Recommendation:** Report Cohen's d (already done) as the primary metric and either drop the U-test p-values or rerun with multiple independent evaluation prompts (≥5 per condition). If keeping, add a caveat that tokens within a prompt are correlated and the p-values are lower bounds. |

### Test 1.2 — Cohen's d (layer sweep, Panels A/C/E/G)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Standardized effect size comparing pos vs neg probe score distributions per layer. |
| 2 | **Is it well chosen?** | **Yes.** Cohen's d is the standard effect-size metric for two-group comparisons. |
| 3 | **Sample, N, unit of analysis** | Same token-level scores; d is computed per layer. The d values are descriptive effect sizes, not hypothesis tests per se. |
| 4 | **Is N defensible?** | Same issue as Test 1.1 — tokens within a prompt are correlated. However, d is primarily used descriptively here (to select best layer), so the impact is lower. |
| 5 | **Results** | Best d ranges from ~1.5 to ~4.0 across concepts — very large effects. |
| 6 | **Verdict** | ✅ **Acceptable as descriptive metric.** The layer selection procedure is valid. |

---

## Figure 2: Internal State Drift & Self-Report Methods

### Test 2.1 — Spearman ρ (turn vs rating, Panels A/B/D/E)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Tests monotonic trend of self-report or probe score across turns. Spearman is appropriate for ordinal/non-linear monotonic relationships. |
| 2 | **Is it well chosen?** | **The test itself is fine, but the sample is not.** Spearman is a good choice for testing monotonic trends. The problem is how observations are counted. |
| 3 | **Sample, N, unit of analysis** | All turn × conversation observations pooled. For α=0: 40 conversations × 10 turns = **400 observations** fed to `spearmanr(turn, rating)`. But turns within a conversation are **not independent** — they are repeated measures on the same conversation. |
| 4 | **Is N defensible?** | ❌ **NO — this is pseudoreplication.** The 400 observations are treated as independent but actually consist of 40 independent conversations, each contributing 10 correlated observations. The effective sample size is somewhere between 40 and 400. This inflates the Spearman test's power massively, producing artificially small p-values. |
| 5 | **Results** | |

| Panel | Concept | ρ | p-value | N used | N effective |
|-------|---------|---|---------|--------|-------------|
| 2A (greedy) | wellbeing | 0.082 | 1.02e-01 | 400 | ~40 |
| 2A (greedy) | interest | 0.432 | 1.30e-19 | 400 | ~40 |
| 2A (greedy) | focus | 0.212 | 1.95e-05 | 400 | ~40 |
| 2A (greedy) | impulsivity | 0.061 | 2.23e-01 | 400 | ~40 |
| 2B (probe) | wellbeing | -0.011 | 8.28e-01 | 400 | ~40 |
| 2B (probe) | interest | 0.208 | 2.89e-05 | 400 | ~40 |
| 2B (probe) | focus | 0.226 | 4.80e-06 | 400 | ~40 |
| 2B (probe) | impulsivity | 0.086 | 8.63e-02 | 400 | ~40 |
| 2D (sampled) | same pattern | ... | ... | 400 | ~40 |
| 2E (logit) | same pattern | ... | ... | 400 | ~40 |

| 6 | **Verdict** | 🔴 **REDO.** The correct approach is one of: **(a)** Compute one slope (or drift = last−first) per conversation, then do a one-sample t-test or Wilcoxon signed-rank on the 40 slopes against zero. **(b)** Use a linear mixed-effects model with conversation as random intercept: `rating ~ turn + (1|conversation)`. **(c)** Compute Spearman ρ on the 10 turn-level means (N=10), which tests the population-averaged trend. Option (a) is simplest and most robust. |

### Test 2.2 — Mann-Whitney U (first vs last turn, Panel B)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Tests whether probe scores at turn 1 differ from turn 10. |
| 2 | **Is it well chosen?** | **Partially.** This is a between-group test applied to what is actually a paired design. Each conversation contributes one observation at turn 1 and one at turn 10. |
| 3 | **Sample, N, unit of analysis** | 40 observations per group (turn 1 values, turn 10 values). Unit = conversation-turn. |
| 4 | **Is N defensible?** | **N=40 per group is correct**, but the test ignores the pairing. A **Wilcoxon signed-rank test** (paired) would be more appropriate and more powerful since each conversation's turn-1 and turn-10 scores are naturally paired. |
| 5 | **Results** | |

| Concept | U | p-value | N per group |
|---------|---|---------|-------------|
| wellbeing | 749 | 0.627 | 40 |
| interest | 278 | 5.22e-07 | 40 |
| focus | 494 | 0.003 | 40 |
| impulsivity | 915 | 0.271 | 40 |

| 6 | **Verdict** | ⚠️ **Replace with Wilcoxon signed-rank.** The N is defensible (40 independent conversations), but using an unpaired test on paired data is suboptimal. However, the p-values from Mann-Whitney are _conservative_ relative to a paired test, so the significant results remain valid. |

### Test 2.3 — Bootstrap CI (per-turn means, Panels A/B/D/E)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Computes 95% confidence intervals for the mean at each turn via bootstrap. |
| 2 | **Is it well chosen?** | **Yes** — bootstrap CI for the mean is standard. |
| 3 | **Sample, N, unit of analysis** | At each turn: 40 values (one per conversation). Bootstrap resamples 1000 times from these 40. |
| 4 | **Is N defensible?** | **N=40 is correct** here — at each turn, the values across conversations are independent. The 1000 bootstrap resamples are used only to estimate the CI, not as the sample size. |
| 5 | **Results** | CIs shown in figures. |
| 6 | **Verdict** | ✅ **Correct.** The CI width reflects N=40. |

### Test 2.4 — Pearson r, Spearman ρ, linregress (logit vs token, Panel G)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Tests correlation between logit-based and token-based self-reports within the same observations. |
| 2 | **Is it well chosen?** | **The test is fine, but N is inflated.** |
| 3 | **Sample, N, unit of analysis** | All 400 observation-level pairs (40 conv × 10 turns) pooled and correlated. |
| 4 | **Is N defensible?** | ❌ **Pseudoreplication.** The 400 observations include repeated measures from the same conversations. Multiple turns from the same conversation share context and are not independent. Effective N ≈ 40. |
| 5 | **Results** | ρ ranges from 0.25 (impulsivity) to 0.56 (interest), all p < 1e-6. |
| 6 | **Verdict** | 🔴 **P-values are over-optimistic.** For a valid test: compute per-conversation mean(logit_rating) and mean(token_rating), then correlate the 40 pairs. Or use a mixed model. The ρ point estimates are OK as descriptive, but the p-values should not be taken at face value. |

### Test 2.5 — Shannon Entropy (Panel F)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Descriptive measure of response diversity (not a hypothesis test). |
| 2 | **Is it well chosen?** | **Yes.** Entropy is a standard information-theoretic measure. |
| 3 | **Sample, N, unit of analysis** | All responses pooled per method per concept. |
| 4 | **Is N defensible?** | Entropy is a population descriptive — more data is better. Pooling is appropriate here. |
| 5 | **Results** | Greedy shows low entropy; logit-based shows high entropy. |
| 6 | **Verdict** | ✅ **Correct.** No p-value claimed, purely descriptive. |

---

## Figure 3: Introspection Analysis

### Test 3.1 — Spearman ρ and Isotonic R² (probe vs self-report, Panel A)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Measures introspection — correlation between internal state (probe) and self-report. |
| 2 | **Is it well chosen?** | **The metrics are excellent choices.** Isotonic R² handles monotonic nonlinearities. Spearman ρ is robust. |
| 3 | **Sample, N, unit of analysis** | All 400 observations at α=0 (40 conv × 10 turns). Unit = conversation-turn. |
| 4 | **Is N defensible?** | ❌ **Same pseudoreplication issue.** The 400 observations include 10 correlated turns per conversation. The Spearman p-values (e.g., p = 2.8e-56 for wellbeing) are over-inflated. |
| 5 | **Results** | |

| Concept | ρ | p-value | R²(iso) | N used |
|---------|---|---------|---------|--------|
| wellbeing | 0.683 | 2.79e-56 | 0.478 | 400 |
| interest | 0.763 | 2.02e-77 | 0.539 | 400 |
| focus | 0.400 | 8.95e-17 | 0.125 | 400 |
| impulsivity | 0.509 | 9.56e-28 | 0.314 | 400 |

| 6 | **Verdict** | 🔴 **P-values need recalculation.** The ρ and R² point estimates are descriptively useful but the p-values assume N=400 independent observations when the effective N ≈ 40. **Recommendation:** Compute ρ per conversation (N_obs=10 per conversation), then report the mean ρ across 40 conversations with a one-sample t-test on the 40 ρ values. Or compute one ρ per turn (N_obs=40 per turn), average over 10 turns. Either gives a defensible p-value. |

### Test 3.2 — Bootstrap CI on R² and ρ (Panels A, B)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Bootstrap CIs for the introspection metrics. |
| 2 | **Is it well chosen?** | **The bootstrap procedure is correct in mechanics**, but it resamples observation-level data (N=400), not conversation-level. |
| 3 | **Sample, N, unit of analysis** | `bootstrap_stat` resamples indices from the 400 paired (probe, rating) observations. |
| 4 | **Is N defensible?** | ⚠️ **The CIs may be too narrow.** The bootstrap implicitly treats all 400 as independent. A cluster bootstrap (resampling conversations, not individual observations) would give wider, more honest CIs. With 40 conversations, you resample 40 conversation IDs with replacement, keep all turns within each resampled conversation. |
| 5 | **Results** | E.g., wellbeing R² CI = [0.424, 0.570], ρ CI = [0.627, 0.732]. |
| 6 | **Verdict** | ⚠️ **CIs likely too narrow.** Switch to cluster bootstrap (resample at conversation level). The point estimates remain valid. |

### Test 3.3 — Turnwise R² and ρ (Panels C, D, CDii)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Computes R² and ρ separately at each turn (N=40 independent conversations per turn). |
| 2 | **Is it well chosen?** | **Yes — this is actually the correct approach.** At each turn, the 40 observations come from 40 independent conversations. |
| 3 | **Sample, N, unit of analysis** | 40 observations per turn (one per conversation). Unit = conversation. |
| 4 | **Is N defensible?** | ✅ **Yes!** N=40 per turn, all independent. The CIs come from the pre-computed turnwise analysis (which also bootstraps at this level). |
| 5 | **Results** | CDii: At turn 1, ρ ranges from 0.26 to 0.65; at turn 10, from 0.41 to 0.80. |
| 6 | **Verdict** | ✅ **Correct.** These are the most statistically defensible introspection measures in the paper. |

### Test 3.4 — Spearman ρ (display_alpha vs logit_rating, Panel E)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Tests whether self-report varies monotonically with steering strength. |
| 2 | **Is it well chosen?** | **The test is appropriate but the sample is heavily inflated.** |
| 3 | **Sample, N, unit of analysis** | ALL observations across all alphas pooled: 5 α × 40 conv × 10 turns = **2000 observations.** |
| 4 | **Is N defensible?** | ❌ **Triple pseudoreplication.** (1) Turns within conversations are correlated. (2) Different alphas from the same conversation are correlated (same model/prompts). The effective independent N is at most 5 × 40 = 200 (if conversations are truly independent across alphas) or as low as 5 (the number of alpha levels). |
| 5 | **Results** | ρ = 0.60–0.88, p ≈ 0.0 for all concepts. |
| 6 | **Verdict** | 🔴 **P-values are meaningless at this level.** The steering effect is clearly real (visible in the plot), but the p-values from N=2000 are not interpretable. **Recommendation:** Compute mean rating per alpha (5 points), then report a Spearman or Kendall test on N=5. Or compute mean rating per conversation×alpha (200 points) and use a mixed model. The simplest defensible test: compute slope per conversation across the 5 alphas (40 slopes), one-sample t-test. |

### Test 3.5 — Spearman ρ (turn vs logit_rating per alpha, Panels F–I)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Tests temporal drift of self-reports at each alpha level. |
| 2 | **Is it well chosen?** | Same pseudoreplication as Test 2.1. |
| 3 | **Sample, N, unit of analysis** | 40 conv × 10 turns = 400 observations per alpha. |
| 4 | **Is N defensible?** | ❌ **Same issue as Test 2.1.** |
| 5 | **Results** | Various ρ and p-values per alpha per concept. |
| 6 | **Verdict** | 🔴 **Same fix needed:** per-conversation slope, then one-sample test on N=40 slopes. |

---

## Figure 4: Steering Matrix & Introspection Improvement

### Test 4.1 — Bootstrap significance (max R² increase, Panel F)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Tests whether steering significantly increases isotonic R² relative to baseline (α=0). This comes from pre-computed matrix analysis. |
| 2 | **Is it well chosen?** | Depends on the bootstrap implementation in the matrix analysis code (outside these scripts). If it bootstraps at the observation level (not conversation level), same issue applies. |
| 3 | **Sample, N, unit of analysis** | From the pre-computed CSV. Need to verify the matrix analysis code. |
| 4 | **Is N defensible?** | ⚠️ **Unclear — needs verification of the upstream matrix analysis code.** |
| 5 | **Results** | Two cells flagged significant: focus→wellbeing, impulsivity→interest. |
| 6 | **Verdict** | ⚠️ **Verify the upstream bootstrap level (should resample conversations, not turns).** |

### Test 4.2 — Spearman ρ (R² vs display alpha, Panel G)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Tests monotonic increase of R² with steering strength. |
| 2 | **Is it well chosen?** | **Spearman is fine, but N=5 makes the p-value from scipy unreliable.** |
| 3 | **Sample, N, unit of analysis** | **N = 5** (five alpha levels: -4, -2, 0, +2, +4). One R² value per alpha. |
| 4 | **Is N defensible?** | ✅ **N=5 is correct** — each alpha level produces one R² summary. However, see p-value issue below. |
| 5 | **Results** | Both conditions: ρ = 1.0, scipy p = 1.40e-24. |
| 6 | **Verdict** | 🔴 **P-value is WRONG.** Scipy's `spearmanr` uses a t-distribution asymptotic approximation that **completely fails** at N=5. When ρ=1.0, the formula computes t = ρ·√((N-2)/(1-ρ²)) → ∞, giving p ≈ 0. **The exact permutation p-value for ρ=1.0 with N=5 is 2/5! = 0.0167 (two-sided).** This is a factor of 10²² off. **Fix:** Use `scipy.stats.spearmanr` with `alternative='two-sided'` on newer scipy, or compute the exact permutation p-value manually (trivial for N=5). Or simply report ρ=1.0 with N=5 and note that a perfect monotonic rank is significant at p=0.0167 by exact permutation test. |

### Test 4.3 — Spearman ρ (ρ_change vs display alpha, Panel H)

| # | Question | Answer |
|---|----------|--------|
| 1–4 | Same as 4.2 | N=5 alpha levels, same scipy p-value bug. |
| 5 | **Results** | Similar Spearman on 5 points. |
| 6 | **Verdict** | 🔴 **Same p-value issue as 4.2.** |

### Test 4.4 — Spearman ρ (drift vs turn, per alpha, Panels I–J)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Tests temporal drift of probe scores under different steering strengths. |
| 2 | **Is it well chosen?** | Same pseudoreplication as Test 2.1. |
| 3 | **Sample, N, unit of analysis** | ~40 conv × 10 turns = ~400 per alpha. |
| 4 | **Is N defensible?** | ❌ **Pseudoreplication.** |
| 5 | **Results** | Various ρ values per alpha. |
| 6 | **Verdict** | 🔴 **Same fix as Test 2.1.** |

### Test 4.5 — Spearman ρ (drift_magnitude vs display alpha, Panel L)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Tests if drift increases monotonically with steering. |
| 2 | **Is it well chosen?** | Same as 4.2. |
| 3 | **Sample, N, unit of analysis** | **N = 5** alpha levels. Drift is now computed as mean of per-conversation drifts (good!). |
| 4 | **Is N defensible?** | ✅ **N=5 is correct for the trend test.** Each point summarizes 40 conversations properly. |
| 5 | **Results** | ρ = 1.0, p = 1.40e-24 (scipy). |
| 6 | **Verdict** | 🔴 **Same scipy N=5 p-value bug.** Exact p = 0.0167. The per-conversation drift computation is correct; only the p-value is wrong. |

### Test 4.6 — Bootstrap CI (drift per alpha, Panel L)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | 95% CI for mean drift at each alpha, bootstrapped from per-conversation drifts. |
| 2 | **Is it well chosen?** | **Yes — excellent.** This bootstraps at the conversation level (the correct unit). |
| 3 | **Sample, N, unit of analysis** | ~40 per-conversation drifts per alpha level. |
| 4 | **Is N defensible?** | ✅ **Yes.** |
| 5 | **Results** | CIs shown on the plot. |
| 6 | **Verdict** | ✅ **Correct.** |

---

## Figure 5: Cross-Model Generalization

### Test 5.1 — Isotonic R² and Spearman ρ (per model size, Panels B/Bii)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Measures introspection at α=0 per model size, computed from raw data. |
| 2 | **Is it well chosen?** | Same metrics as Figure 3, same issue. |
| 3 | **Sample, N, unit of analysis** | N ≈ 400 (40 conv × 10 turns) per model per concept at α=0, all pooled. |
| 4 | **Is N defensible?** | ❌ **Pseudoreplication** (same as Test 3.1). |
| 5 | **Results** | R² and ρ with bootstrap CIs across 1B/3B/8B. |
| 6 | **Verdict** | ⚠️ **Point estimates useful for comparison; CIs should use cluster bootstrap.** The trend across model sizes is visually compelling regardless, and the R²/ρ values are descriptively valid. |

### Test 5.2 — Spearman ρ (Biii: LLaMA 8B scatter)

| # | Question | Answer |
|---|----------|--------|
| 1–4 | Same as 3.1 | Pooled observation-level, N inflated. |
| 5 | **Results** | Very high ρ (~0.93) for wellbeing at 8B. |
| 6 | **Verdict** | ⚠️ **P-value inflated, but effect is clearly real.** |

### Test 5.3 — Spearman ρ (alpha vs rating, Panel C heatmap)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Tests whether self-reports respond to steering, per model. |
| 2 | **Is it well chosen?** | Same issue as Test 3.4. |
| 3 | **Sample, N, unit of analysis** | All observations pooled across 5 alphas × 40 conv × 10 turns = up to 2000. |
| 4 | **Is N defensible?** | ❌ **Massive pseudoreplication.** |
| 5 | **Results** | ρ values in heatmap, * = p<0.05. |
| 6 | **Verdict** | 🔴 **Same fix as Test 3.4.** Note that the p<0.05 stars are unreliable. |

### Test 5.4 — Spearman ρ (turn vs probe/rating, Panels E/F)

| # | Question | Answer |
|---|----------|--------|
| 1–4 | Same as Test 2.1 | Pseudoreplication, 400 obs from 40 conversations. |
| 6 | **Verdict** | 🔴 **Same fix needed.** |

### Test 5.5 — Bootstrap CI (drift bars, Panels Eii/Fii)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | 95% CI on per-conversation drift magnitude. |
| 2 | **Is it well chosen?** | **Yes — correct.** |
| 3 | **Sample, N, unit of analysis** | ~40 per-conversation drifts, bootstrapped at conversation level. |
| 4 | **Is N defensible?** | ✅ **Yes.** |
| 5 | **Results** | CIs shown in the drift bar charts. |
| 6 | **Verdict** | ✅ **Correct.** |

### Test 5.6 — Spearman ρ (cross-family scatter, Panels I/J)

| # | Question | Answer |
|---|----------|--------|
| 1–4 | Same as 3.1 | Observation-level Spearman with inflated N. |
| 6 | **Verdict** | ⚠️ **Same issue.** |

### Test 5.7 — Turnwise R²/ρ (cross-family, Panels K/L)

| # | Question | Answer |
|---|----------|--------|
| 1 | **Why this test?** | Per-turn introspection metrics for Gemma and Qwen. |
| 2 | **Is it well chosen?** | **Yes** — turnwise analysis has correct N per turn. |
| 3 | **Sample, N, unit of analysis** | ~40 observations per turn (one per conversation). |
| 4 | **Is N defensible?** | ✅ **Yes.** |
| 5 | **Results** | R² and ρ per turn with CIs. |
| 6 | **Verdict** | ✅ **Correct.** |

---

## Cross-Cutting Issues

### Issue A: Pseudoreplication in Spearman Tests

**Affected tests:** 2.1, 2.4, 3.1, 3.4, 3.5, 4.4, 5.1, 5.3, 5.4, 5.6

**Problem:** `stats.spearmanr(turn, rating)` or `stats.spearmanr(probe, rating)` is called on all N=400 (or N=2000) observation-level data points, where each of 40 conversations contributes 10 (or 50) correlated observations. Scipy computes the p-value assuming all observations are independent, producing p-values that are orders of magnitude too small.

**Why it matters:** At N=400, a ρ of 0.10 would be significant (p ≈ 0.046). At the true N≈40, the same ρ of 0.10 would have p ≈ 0.54 — not significant. The _effect sizes_ (ρ values) are biased upward by autocorrelation but much less so than the p-values.

**Recommended fixes (in order of preference):**

1. **Per-conversation summary → one-sample test.** For trend tests: compute slope or Δ(last−first) per conversation, then Wilcoxon signed-rank or one-sample t-test on N=40 values. For correlation tests: compute ρ per conversation (10 obs each), then one-sample t-test on N=40 ρ values against 0.
2. **Linear mixed model.** `rating ~ predictor + (1|conversation)` with p-value from Satterthwaite or Kenward–Roger dof.
3. **Cluster-robust standard errors.** Correct the Spearman p-value using cluster-robust variance estimation (less standard for rank correlations).

### Issue B: Scipy Spearman p-value Fails at N=5

**Affected tests:** 4.2, 4.3, 4.5, and any Spearman with N ≤ ~10.

**Problem:** `scipy.stats.spearmanr` uses an asymptotic t-distribution approximation: t = ρ·√((N−2)/(1−ρ²)). When ρ=1.0 and N=5, this gives t→∞ and p≈0. The **exact permutation p-value** for ρ=1.0 at N=5 is 2/5! = **0.0167** (two-sided).

**Empirically confirmed:** `spearmanr([1,2,3,4,5], [1,2,3,4,5])` returns p = 1.40e-24, whereas the true exact p = 0.0167. An error of 22 orders of magnitude.

**Fix:** For N ≤ 10, compute p-values via exact permutation test. For N=5 with ρ=1.0, simply report p = 0.0167 (exact). Python code to compute exact permutation p:
```python
from itertools import permutations
exact_p = sum(abs(spearmanr(x, perm)[0]) >= abs(observed_rho) 
              for perm in permutations(x)) / factorial(len(x))
```

### Issue C: Bootstrap CIs Should Resample at Conversation Level

**Affected:** All `bootstrap_stat()` calls (Tests 3.2, 5.1), `compute_per_turn_means()`, and Panel E/F confidence bands.

**Problem:** `bootstrap_stat(x, y, fn)` resamples individual (x_i, y_i) pairs. When multiple observations come from the same conversation, resampling at the observation level underestimates variability.

**Fix:** Implement cluster bootstrap — resample conversation IDs, keep all observations within each resampled conversation:
```python
def cluster_bootstrap(conv_ids, x, y, stat_fn, n_boot=1000):
    unique_convs = np.unique(conv_ids)
    point = stat_fn(x, y)
    boots = []
    for _ in range(n_boot):
        sampled_convs = rng.choice(unique_convs, len(unique_convs), replace=True)
        mask = np.isin(conv_ids, sampled_convs)  # needs care for duplicates
        boots.append(stat_fn(x[mask], y[mask]))
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5)
```

---

## Summary: Which Results Are Trustworthy As-Is?

### ✅ Safe to publish (correct statistics)

| Test | Location | Why |
|------|----------|-----|
| Cohen's d (descriptive) | Fig 1 A/C/E/G | Effect size, not p-test dependent |
| Bootstrap CI per turn | Fig 2 A/B/D/E | N=40 independent convs per turn |
| Shannon entropy | Fig 2 F | Descriptive, no p-value |
| Turnwise R²/ρ | Fig 3 C/D, Fig 5 K/L | N=40 per turn, correct level |
| CDii first vs last (ρ) | Fig 3 CDii | N=40 per turn, correct |
| Bootstrap CI (conv-level) | Fig 4 L, Fig 5 Eii/Fii | Correct resampling unit |

### ⚠️ Effect sizes OK, p-values need recalculation

| Test | Location | Issue |
|------|----------|-------|
| Spearman ρ (turn vs rating) | Fig 2 A/B/D/E | N=400 should be N=40 |
| Spearman ρ (probe vs rating) | Fig 3 A, Fig 5 B/Bii/I/J | N=400 should be N=40 |
| Bootstrap CIs on ρ/R² | Fig 3 B | Obs-level bootstrap, should be cluster |
| Mann-Whitney (first vs last) | Fig 2 B | Should be paired (Wilcoxon) |
| Mann-Whitney (pos vs neg tokens) | Fig 1 B/D/F/H | Tokens not independent |

### 🔴 P-values definitely wrong

| Test | Location | Issue |
|------|----------|-------|
| Spearman on N=5 alpha levels | Fig 4 G/H/L | Scipy gives p≈1e-24, exact is p≈0.017 |
| Spearman on N=2000 | Fig 3 E, Fig 5 C | Massive pseudoreplication, p literally = 0.0 |
| Spearman drift (N=400) | Fig 3 F–I, Fig 4 I–J, Fig 5 E/F | Pseudoreplication |

---

## Recommendations for a Revised Analysis

1. **Replace all "all-observations-pooled" Spearman tests** with per-conversation summaries + one-sample tests (Wilcoxon signed-rank or one-sample t-test on N=40 or N=40×5 summaries).

2. **Fix N=5 Spearman p-values** by using exact permutation tests (trivial to implement).

3. **Switch bootstrap_stat to cluster bootstrap** (resample at conversation level).

4. **Replace Mann-Whitney with Wilcoxon signed-rank** for paired first-vs-last turn comparisons.

5. **For Figure 1 boxplots**, either (a) increase to ≥5 eval prompts per condition for a valid test, or (b) drop the p-value and report only Cohen's d.

6. **Consider whether any p-values at all are needed** for some of these panels. Many results are visually obvious (e.g., self-report increases monotonically with alpha) and can be reported with effect sizes and CIs, which are more informative than p-values.
