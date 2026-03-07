"""
Figure 2: Internal State Drift and Self-Report Methods
=======================================================
Shows that LLMs have internal state drift during conversations but fail to
report it with greedy decoding. Validates non-greedy and logit-based
self-report methods as improvements.

v2 changes:
 - A/D/E: removed sharey, increased individual trace alpha to 0.2
 - B: increased trace alpha to 0.15
 - C: changed from line+dot to grouped bar chart
 - F: replaced variance with R²-with-probe informativeness (isotonic R²)
 - G: per-panel y-axes (no sharey)
 - Enhanced statistics in all panel JSONs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from shared_utils import (
    PROBES, CONCEPTS_ORDERED, CONCEPT_DISPLAY, CONCEPT_COLORS,
    SHORTHAND_TO_CONCEPT, CONCEPT_TO_SHORTHAND, SHORTHANDS_ORDERED,
    SHORTHAND_DISPLAY,
    LLAMA_3B_RERUN_SELF, LLAMA_3B_GREEDY,
    PROBE_METRIC_KEY, FLIP_CONCEPTS, FLIP_SHORTHANDS,
    load_results, results_to_dataframe,
    flip_if_needed, flip_if_needed_shorthand,
    compute_per_turn_means, compute_per_turn_unique_counts,
    isotonic_r2,
    savefig, save_panel_json, save_other_stats, ensure_dir, add_panel_label,
    plot_line_with_ci, format_p,
    corrected_drift_stats, corrected_correlation_stats,
    per_conversation_slope, per_conversation_drift, one_sample_test,
    lmm_test, linear_regression_stats,
)


def _load_concept_df(exp_dirs, concept_shorthand, probe_name=None):
    """Load results for a concept from experiment directory, return DataFrame."""
    exp_dir = exp_dirs.get(concept_shorthand)
    if exp_dir is None or not os.path.isdir(exp_dir):
        return None
    if probe_name is None:
        probe_name = SHORTHAND_TO_CONCEPT[concept_shorthand]
    results = load_results(exp_dir)
    return results_to_dataframe(results, probe_name=probe_name)


def generate_figure_2(results_dir):
    """Generate all Figure 2 panels."""
    fig_dir = ensure_dir(os.path.join(results_dir, 'Figure_2'))
    print("  Generating Figure 2: Internal State Drift & Self-Report Methods...")

    other_stats = {}

    # ──────────────────────────────────────────────────────────────
    # Panel A: Greedy self-report vs turn (4 concepts)
    #   - NO sharey, individual trace alpha=0.2
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    greedy_stats = {}
    for i, short in enumerate(SHORTHANDS_ORDERED):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]

        df = _load_concept_df(LLAMA_3B_GREEDY, short)
        if df is not None:
            per_turn = compute_per_turn_means(df, 'token_rating', alpha_val=0.0)
            plot_line_with_ci(ax, per_turn['turn'], per_turn['mean'],
                              per_turn['ci_low'], per_turn['ci_high'],
                              color=color)
            # Individual conversations
            for ci in df['conversation_index'].unique():
                conv = df[df['conversation_index'] == ci].sort_values('turn')
                ax.plot(conv['turn'], conv['token_rating'],
                        color=color, alpha=0.5, linewidth=0.5)
            greedy_var = df.groupby('turn')['token_rating'].var().mean()
            n_conv = df['conversation_index'].nunique()
            # Spearman of turn vs rating
            rho_g, p_g = stats.spearmanr(df['turn'], df['token_rating'])
            # Corrected drift stats (per-conv slope + LMM)
            corrected_g = corrected_drift_stats(df, 'token_rating', alpha_val=0.0)
            greedy_stats[short] = {
                'mean_variance_across_turns': round(float(greedy_var), 4),
                'mean_rating': round(float(df['token_rating'].mean()), 3),
                'n_conversations': int(n_conv),
                'spearman_rho_vs_turn_POOLED': round(float(rho_g), 4),
                'spearman_p_vs_turn_POOLED': float(p_g),
                'corrected_drift': corrected_g,
            }
        ax.set_xlabel('Turn')
        if i == 0:
            ax.set_ylabel('Self-Report (greedy)')
        ax.set_title(SHORTHAND_DISPLAY[short])
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
    # Per-concept y-axis limits
    _YLIM_A = {'wellbeing': (4, 8), 'interest': (5, 9), 'focus': (6, 10), 'impulsivity': (2, 6)}
    for i, short in enumerate(SHORTHANDS_ORDERED):
        if short in _YLIM_A:
            axes[i].set_ylim(*_YLIM_A[short])
    add_panel_label(axes[0], 'A', x=-0.18)
    fig.suptitle('Greedy Self-Report Across Turns', fontsize=12, y=1.02)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_A_greedy_self_report_vs_turn')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_A',
        'title': 'Greedy Self-Report Across Turns',
        'description': ('Greedy-decoded numerical self-reports (0–9) show minimal variance '
                        'across conversation turns. Individual conversations shown at alpha=0.2, '
                        'mean with 95% bootstrap CI in bold. LLaMA-3.2-3B-Instruct.'),
        'data_sources': {k: v for k, v in LLAMA_3B_GREEDY.items()},
        'model': 'LLaMA-3.2-3B-Instruct',
        'n_turns': 10,
        'rating_type': 'token (greedy)',
        'per_concept_stats': greedy_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel B: Probe score drift vs turn (4 concepts)
    #   - trace alpha=0.15, per-panel y-axes
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    drift_stats = {}
    for i, short in enumerate(SHORTHANDS_ORDERED):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]

        df = _load_concept_df(LLAMA_3B_RERUN_SELF, short)
        if df is not None:
            sub = df[np.isclose(df['alpha'], 0.0)].copy()
            sub['probe_display'] = flip_if_needed(concept, sub['probe_score'].values)

            turns = sorted(sub['turn'].unique())
            means, ci_lows, ci_highs = [], [], []
            rng = np.random.RandomState(42)
            for t in turns:
                vals = sub[sub['turn'] == t]['probe_display'].dropna().values
                m = np.mean(vals)
                boots = [np.mean(rng.choice(vals, len(vals))) for _ in range(1000)]
                means.append(m)
                ci_lows.append(np.percentile(boots, 2.5))
                ci_highs.append(np.percentile(boots, 97.5))

            plot_line_with_ci(ax, turns, means, ci_lows, ci_highs, color=color)
            for ci_idx in sub['conversation_index'].unique():
                conv = sub[sub['conversation_index'] == ci_idx].sort_values('turn')
                ax.plot(conv['turn'], conv['probe_display'],
                        color=color, alpha=0.3, linewidth=0.5)

            rho, p = stats.spearmanr(sub['turn'], sub['probe_display'])
            # First vs last turn: Wilcoxon signed-rank (paired, corrected from Mann-Whitney)
            first_vals = sub[sub['turn'] == turns[0]].sort_values('conversation_index')['probe_display'].values
            last_vals = sub[sub['turn'] == turns[-1]].sort_values('conversation_index')['probe_display'].values
            if len(first_vals) > 0 and len(last_vals) > 0 and len(first_vals) == len(last_vals):
                try:
                    w_stat, w_p = stats.wilcoxon(last_vals - first_vals, alternative='two-sided')
                except Exception:
                    w_stat, w_p = np.nan, np.nan
            else:
                w_stat, w_p = np.nan, np.nan
            # Corrected drift stats
            corrected_b = corrected_drift_stats(sub, 'probe_display', alpha_val=0.0)
            drift_stats[short] = {
                'mean_first_turn': round(float(means[0]), 4),
                'mean_last_turn': round(float(means[-1]), 4),
                'drift_magnitude': round(float(means[-1] - means[0]), 4),
                'spearman_rho_vs_turn_POOLED': round(float(rho), 4),
                'spearman_p_vs_turn_POOLED': float(p),
                'first_vs_last_wilcoxon_stat': float(w_stat),
                'first_vs_last_wilcoxon_p': float(w_p),
                'n_conversations': int(sub['conversation_index'].nunique()),
                'polarity_flip_applied': concept in FLIP_CONCEPTS,
                'corrected_drift': corrected_b,
            }

        ax.set_xlabel('Turn')
        if i == 0:
            ax.set_ylabel('Probe Score')
        ax.set_title(SHORTHAND_DISPLAY[short])
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
    # Per-concept y-axis limits for probe scores
    _YLIM_B = {'wellbeing': (-0.3, 0.0), 'interest': (-0.4, -0.1)}
    for i, short in enumerate(SHORTHANDS_ORDERED):
        if short in _YLIM_B:
            axes[i].set_ylim(*_YLIM_B[short])
        else:
            # 0.3 amplitude centred on the mean of the data
            lines = [l for l in axes[i].get_lines() if l.get_alpha() is None or l.get_alpha() >= 0.5]
            if lines:
                y_vals = np.concatenate([l.get_ydata() for l in lines])
                y_vals = y_vals[np.isfinite(y_vals)]
                if len(y_vals) > 0:
                    y_mid = (y_vals.max() + y_vals.min()) / 2
                    axes[i].set_ylim(y_mid - 0.15, y_mid + 0.15)
    add_panel_label(axes[0], 'B', x=-0.18)
    fig.suptitle('Internal State (Probe Score) Across Turns', fontsize=12, y=1.02)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_B_probe_drift_vs_turn')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_B',
        'title': 'Internal State Drift Across Turns',
        'description': ('Probe scores (flipped for intuitive direction where needed) show '
                        'clear drift across conversation turns, despite flat greedy self-reports. '
                        'Individual conversations at alpha=0.15, mean with 95% CI in bold.'),
        'data_sources': {k: v for k, v in LLAMA_3B_RERUN_SELF.items()},
        'model': 'LLaMA-3.2-3B-Instruct',
        'metric': PROBE_METRIC_KEY,
        'alpha': 0.0,
        'per_concept_drift': drift_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel C: Number of unique greedy responses per turn
    #   - Grouped bar chart (one group per turn, one bar per concept)
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    turns = np.arange(1, 11)
    n_concepts = len(SHORTHANDS_ORDERED)
    width = 0.8 / n_concepts
    unique_stats = {}
    for j, short in enumerate(SHORTHANDS_ORDERED):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        df = _load_concept_df(LLAMA_3B_GREEDY, short)
        if df is not None:
            uniq = compute_per_turn_unique_counts(df, 'token_rating', alpha_val=0.0)
            uniq_vals = uniq.set_index('turn')['n_unique'].reindex(turns).fillna(0).values
            offset = (j - n_concepts / 2 + 0.5) * width
            ax.bar(turns + offset, uniq_vals, width, label=SHORTHAND_DISPLAY[short],
                   color=color, edgecolor='white', linewidth=0.5, alpha=0.85)
            unique_stats[short] = {
                'per_turn_unique': {int(t): int(v) for t, v in zip(turns, uniq_vals)},
                'mean_unique': round(float(np.mean(uniq_vals)), 2),
                'max_unique': int(np.max(uniq_vals)),
            }
    ax.set_xlabel('Turn')
    ax.set_ylabel('Unique Responses')
    ax.set_title('Unique Greedy Responses per Turn (of 10 possible)')
    ax.set_xlim(0.2, 10.8)
    ax.set_xticks(turns)
    ax.set_ylim(0.5, None)
    ax.legend(fontsize=8, loc='upper right')
    add_panel_label(ax, 'C')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_C_unique_greedy_responses')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_C',
        'title': 'Unique Greedy Responses per Turn (Grouped Bar)',
        'description': ('Number of unique discrete ratings (out of 10 possible: 0–9) '
                        'given by the model across conversations at each turn. '
                        'Low counts indicate high response stereotypy. Grouped bar chart '
                        'with one bar per concept per turn.'),
        'data_sources': {k: v for k, v in LLAMA_3B_GREEDY.items()},
        'per_concept_unique': unique_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel D: Non-greedy (sampled) token self-report vs turn
    #   - NO sharey, trace alpha=0.2
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    sampled_drift = {}
    for i, short in enumerate(SHORTHANDS_ORDERED):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]

        df = _load_concept_df(LLAMA_3B_RERUN_SELF, short)
        if df is not None:
            per_turn = compute_per_turn_means(df, 'token_rating', alpha_val=0.0)
            plot_line_with_ci(ax, per_turn['turn'], per_turn['mean'],
                              per_turn['ci_low'], per_turn['ci_high'],
                              color=color)
            sub_a0 = df[np.isclose(df['alpha'], 0.0)]
            for ci_idx in sub_a0['conversation_index'].unique():
                conv = sub_a0[sub_a0['conversation_index'] == ci_idx].sort_values('turn')
                ax.plot(conv['turn'], conv['token_rating'],
                        color=color, alpha=0.5, linewidth=0.5)
            rho_s, p_s = stats.spearmanr(sub_a0['turn'], sub_a0['token_rating'].dropna())
            corrected_d = corrected_drift_stats(sub_a0, 'token_rating', alpha_val=0.0)
            sampled_drift[short] = {
                'mean_rating': round(float(sub_a0['token_rating'].mean()), 3),
                'variance': round(float(sub_a0['token_rating'].var()), 4),
                'spearman_rho_vs_turn_POOLED': round(float(rho_s), 4),
                'spearman_p_vs_turn_POOLED': float(p_s),
                'n_conversations': int(sub_a0['conversation_index'].nunique()),
                'corrected_drift': corrected_d,
            }
        ax.set_xlabel('Turn')
        if i == 0:
            ax.set_ylabel('Self-Report (sampled token)')
        ax.set_title(SHORTHAND_DISPLAY[short])
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
    # Per-concept y-axis limits (same as Panel A)
    _YLIM_D = {'wellbeing': (4, 8), 'interest': (5, 9), 'focus': (6, 10), 'impulsivity': (2, 6)}
    for i, short in enumerate(SHORTHANDS_ORDERED):
        if short in _YLIM_D:
            axes[i].set_ylim(*_YLIM_D[short])
    add_panel_label(axes[0], 'D', x=-0.18)
    fig.suptitle('Non-Greedy (Sampled) Self-Report Across Turns', fontsize=12, y=1.02)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_D_sampled_token_self_report_vs_turn')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_D',
        'title': 'Non-Greedy (Sampled) Self-Report Across Turns',
        'description': ('Sampled (temperature=0.8) token-based self-reports remain discrete '
                        '(integer-valued) and noisy, even when they avoid the strongest '
                        'collapse seen under greedy decoding.'),
        'data_sources': {k: v for k, v in LLAMA_3B_RERUN_SELF.items()},
        'rating_type': 'token (sampled, temp=0.8)',
        'per_concept_stats': sampled_drift,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel E: Logit-based self-report vs turn
    #   - NO sharey, trace alpha=0.2
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    logit_drift_stats = {}
    for i, short in enumerate(SHORTHANDS_ORDERED):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]

        df = _load_concept_df(LLAMA_3B_RERUN_SELF, short)
        if df is not None:
            per_turn = compute_per_turn_means(df, 'logit_rating', alpha_val=0.0)
            plot_line_with_ci(ax, per_turn['turn'], per_turn['mean'],
                              per_turn['ci_low'], per_turn['ci_high'],
                              color=color)
            sub_a0 = df[np.isclose(df['alpha'], 0.0)]
            for ci_idx in sub_a0['conversation_index'].unique():
                conv = sub_a0[sub_a0['conversation_index'] == ci_idx].sort_values('turn')
                ax.plot(conv['turn'], conv['logit_rating'],
                        color=color, alpha=0.5, linewidth=0.5)

            rho, p = stats.spearmanr(sub_a0['turn'], sub_a0['logit_rating'].dropna())
            pt = per_turn
            corrected_e = corrected_drift_stats(sub_a0, 'logit_rating', alpha_val=0.0)
            logit_drift_stats[short] = {
                'mean_first_turn': round(float(pt['mean'].iloc[0]), 4),
                'mean_last_turn': round(float(pt['mean'].iloc[-1]), 4),
                'drift_magnitude': round(float(pt['mean'].iloc[-1] - pt['mean'].iloc[0]), 4),
                'spearman_rho_vs_turn_POOLED': round(float(rho), 4),
                'spearman_p_vs_turn_POOLED': float(p),
                'n_conversations': int(sub_a0['conversation_index'].nunique()),
                'corrected_drift': corrected_e,
            }

        ax.set_xlabel('Turn')
        if i == 0:
            ax.set_ylabel('Self-Report (logit-based)')
        ax.set_title(SHORTHAND_DISPLAY[short])
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
    add_panel_label(axes[0], 'E', x=-0.18)
    fig.suptitle('Logit-Based Self-Report Across Turns', fontsize=12, y=1.02)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_E_logit_self_report_vs_turn')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_E',
        'title': 'Logit-Based Self-Report Across Turns',
        'description': ('Logit-based continuous self-reports (probability-weighted mean of '
                        'digit token logits). Shows drift aligned with probe score drift.'),
        'data_sources': {k: v for k, v in LLAMA_3B_RERUN_SELF.items()},
        'rating_type': 'logit-based (probability-weighted average)',
        'per_concept_drift': logit_drift_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel F: Informativeness — Shannon entropy of self-report
    #   Measures how much information each self-report method conveys
    #   (higher entropy = more diverse/informative responses)
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    entropy_data = {'Greedy\n(token)': [], 'Sampled\n(token)': [], 'Logit-\nbased': []}
    entropy_stats_json = {}
    n_bins_logit = 20  # bin count for continuous logit-based ratings

    for short in SHORTHANDS_ORDERED:
        concept = SHORTHAND_TO_CONCEPT[short]
        df_greedy = _load_concept_df(LLAMA_3B_GREEDY, short)
        df_rerun = _load_concept_df(LLAMA_3B_RERUN_SELF, short)
        ent_entry = {}

        def _shannon_entropy_discrete(values):
            """Shannon entropy of discrete distribution (nats → bits)."""
            values = values[~np.isnan(values)]
            if len(values) == 0:
                return 0.0
            _, counts = np.unique(values, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs + 1e-15))

        def _shannon_entropy_binned(values, n_bins=20):
            """Shannon entropy of continuous values via histogram binning."""
            values = values[~np.isnan(values)]
            if len(values) < 2:
                return 0.0
            counts, _ = np.histogram(values, bins=n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))

        # Greedy entropy
        if df_greedy is not None:
            ratings_g = df_greedy['token_rating'].dropna().values
            h_greedy = _shannon_entropy_discrete(ratings_g)
            entropy_data['Greedy\n(token)'].append(h_greedy)
            ent_entry['greedy_entropy_bits'] = round(float(h_greedy), 4)
            ent_entry['greedy_n'] = len(ratings_g)
        else:
            entropy_data['Greedy\n(token)'].append(0)

        # Sampled token / logit entropy (from rerun data, alpha=0)
        if df_rerun is not None:
            sub = df_rerun[np.isclose(df_rerun['alpha'], 0.0)]
            tok_vals = sub['token_rating'].dropna().values
            logit_vals = sub['logit_rating'].dropna().values

            h_token = _shannon_entropy_discrete(tok_vals)
            h_logit = _shannon_entropy_binned(logit_vals, n_bins=n_bins_logit)

            entropy_data['Sampled\n(token)'].append(h_token)
            entropy_data['Logit-\nbased'].append(h_logit)
            ent_entry['sampled_token_entropy_bits'] = round(float(h_token), 4)
            ent_entry['sampled_token_n'] = len(tok_vals)
            ent_entry['logit_entropy_bits'] = round(float(h_logit), 4)
            ent_entry['logit_n'] = len(logit_vals)
            ent_entry['logit_n_bins'] = n_bins_logit
        else:
            entropy_data['Sampled\n(token)'].append(0)
            entropy_data['Logit-\nbased'].append(0)

        entropy_stats_json[short] = ent_entry

    x = np.arange(len(SHORTHANDS_ORDERED))
    width = 0.25
    methods = list(entropy_data.keys())
    method_colors = ['#888888', '#5DADE2', '#2ECC71']
    for j, method in enumerate(methods):
        ax.bar(x + j * width, entropy_data[method], width, label=method,
               color=method_colors[j], edgecolor='white', linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels([SHORTHAND_DISPLAY[s] for s in SHORTHANDS_ORDERED], fontsize=8)
    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_title('Self-Report Informativeness (Entropy)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, None)
    add_panel_label(ax, 'F')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_F_informativeness_entropy')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_F',
        'title': 'Self-Report Informativeness (Shannon Entropy)',
        'description': ('Shannon entropy (bits) of self-report distributions at alpha=0. '
                        'Greedy and sampled token use discrete entropy over integer ratings; '
                        f'logit-based uses binned entropy ({n_bins_logit} bins). '
                        'Higher entropy = more diverse/informative responses. '
                        'Does not require comparison with probe (avoids spoiling Figure 3).'),
        'metric': 'shannon_entropy_bits',
        'data_sources_rerun': {k: v for k, v in LLAMA_3B_RERUN_SELF.items()},
        'data_sources_greedy': {k: v for k, v in LLAMA_3B_GREEDY.items()},
        'per_concept_entropy': entropy_stats_json,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel G: Correlation between logit and token ratings
    #   - Per-panel y-axes (no sharey)
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    corr_stats = {}
    for i, short in enumerate(SHORTHANDS_ORDERED):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]

        df = _load_concept_df(LLAMA_3B_RERUN_SELF, short)
        if df is not None:
            sub = df[np.isclose(df['alpha'], 0.0)].dropna(
                subset=['token_rating', 'logit_rating'])
            ax.scatter(sub['token_rating'], sub['logit_rating'],
                       color=color, alpha=0.3, s=15, edgecolors='none')
            if len(sub) > 2:
                reg = linear_regression_stats(sub['token_rating'].values,
                                              sub['logit_rating'].values)
                slope = reg['slope']
                intercept = reg['intercept']
                x_fit = np.array([sub['token_rating'].min(), sub['token_rating'].max()])
                ax.plot(x_fit, slope * x_fit + intercept, 'k--', linewidth=1, alpha=0.7)
                rho_s, p_s = stats.spearmanr(sub['token_rating'], sub['logit_rating'])
                r2_iso = isotonic_r2(sub['token_rating'].values, sub['logit_rating'].values)
                # Corrected: per-conversation means correlation + LMM
                conv_means_tok = sub.groupby('conversation_index')['token_rating'].mean()
                conv_means_log = sub.groupby('conversation_index')['logit_rating'].mean()
                common_idx = conv_means_tok.index.intersection(conv_means_log.index)
                rho_conv, p_conv = stats.spearmanr(conv_means_tok[common_idx], conv_means_log[common_idx])
                pearson_conv, pearson_p_conv = stats.pearsonr(conv_means_tok[common_idx].values, conv_means_log[common_idx].values)
                lmm_g = lmm_test(sub, 'logit_rating', 'token_rating', group_col='conversation_index')
                corr_stats[short] = {
                    'pearson_r_POOLED': round(float(reg['pearson_r']), 4),
                    'pearson_p_POOLED': float(reg['pearson_p']),
                    'spearman_rho_POOLED': round(float(rho_s), 4),
                    'spearman_p_POOLED': float(p_s),
                    'isotonic_r2': round(float(r2_iso), 4),
                    'n_POOLED': len(sub),
                    'slope': round(float(slope), 4),
                    'intercept': round(float(intercept), 4),
                    'linear_r2_POOLED': round(float(reg['r_squared']), 4),
                    'linear_stderr': float(reg['stderr']),
                    'per_conv_means_spearman_rho': round(float(rho_conv), 4),
                    'per_conv_means_spearman_p': float(p_conv),
                    'per_conv_means_pearson_r': round(float(pearson_conv), 4),
                    'per_conv_means_pearson_p': float(pearson_p_conv),
                    'n_conversations': len(common_idx),
                    'lmm_results': lmm_g,
                }
                ax.text(0.05, 0.95,
                        f'r = {reg["pearson_r"]:.3f}\n{format_p(reg["pearson_p"])}\nR² = {reg["r_squared"]:.3f}',
                        transform=ax.transAxes, fontsize=8, va='top')
        ax.set_xlabel('Token Rating')
        if i == 0:
            ax.set_ylabel('Logit Rating')
        ax.set_title(SHORTHAND_DISPLAY[short])
        # Auto-scale axes per panel (no forced identity line)
    add_panel_label(axes[0], 'G', x=-0.18)
    fig.suptitle('Logit vs Token Self-Report Correlation', fontsize=12, y=1.02)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_G_logit_vs_token_correlation')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_G',
        'title': 'Logit vs Token Self-Report Correlation',
        'description': ('Correlation between logit-based and token-based self-reports '
                        'at alpha=0. Black dashed line: OLS fit. Logit-based ratings are '
                        'linearly calibrated with token ratings while providing continuous, '
                        'less noisy measurements.'),
        'correlation_stats': corr_stats,
    })

    # ── Save other_stats ──
    other_stats = {
        'description': ('Figure 2 demonstrates internal state drift through conversations '
                        'and validates different self-report extraction methods. Key finding: '
                        'models show internal state drift but fail to report it with greedy '
                        'decoding. Logit-based method captures drift with continuous values. '
                        'Informativeness uses Shannon entropy of self-report distributions.'),
        'greedy_stats': greedy_stats,
        'probe_drift_stats': drift_stats,
        'logit_drift_stats': logit_drift_stats,
        'informativeness_entropy': entropy_stats_json,
        'logit_vs_token_correlation': corr_stats,
    }
    save_other_stats(fig_dir, other_stats)
    print("    Figure 2 complete.")


if __name__ == '__main__':
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rdir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    generate_figure_2(rdir)
    print(f"Output -> {rdir}")
