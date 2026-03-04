"""
Figure 2: Internal State Drift and Self-Report Methods
=======================================================
Shows that LLMs have internal state drift during conversations but fail to
report it with greedy decoding. Validates non-greedy and logit-based
self-report methods as improvements.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from shared_utils import (
    PROBES, CONCEPTS_ORDERED, CONCEPT_DISPLAY, CONCEPT_COLORS,
    SHORTHAND_TO_CONCEPT, CONCEPT_TO_SHORTHAND,
    LLAMA_3B_RERUN_SELF, LLAMA_3B_GREEDY,
    PROBE_METRIC_KEY, FLIP_CONCEPTS,
    load_results, results_to_dataframe,
    flip_if_needed, compute_per_turn_means, compute_per_turn_unique_counts,
    savefig, save_panel_json, save_other_stats, ensure_dir, add_panel_label,
    plot_line_with_ci, format_p,
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
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
    greedy_stats = {}
    for i, short in enumerate(['wellbeing', 'interest', 'focus', 'impulsivity']):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]

        df = _load_concept_df(LLAMA_3B_GREEDY, short)
        if df is not None:
            # For greedy, use token_rating
            per_turn = compute_per_turn_means(df, 'token_rating', alpha_val=0.0)
            plot_line_with_ci(ax, per_turn['turn'], per_turn['mean'],
                              per_turn['ci_low'], per_turn['ci_high'],
                              color=color)
            # Show individual conversations in light
            for ci in df['conversation_index'].unique():
                conv = df[df['conversation_index'] == ci].sort_values('turn')
                ax.plot(conv['turn'], conv['token_rating'],
                        color=color, alpha=0.08, linewidth=0.5)
            greedy_var = df.groupby('turn')['token_rating'].var().mean()
            greedy_stats[short] = {
                'mean_variance_across_turns': round(float(greedy_var), 4),
                'mean_rating': round(float(df['token_rating'].mean()), 3),
            }
        ax.set_xlabel('Turn')
        if i == 0:
            ax.set_ylabel('Self-Report (greedy)')
        ax.set_title(CONCEPT_DISPLAY[concept])
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(-0.5, 9.5)
        ax.set_xticks(range(1, 11))
    add_panel_label(axes[0], 'A', x=-0.18)
    fig.suptitle('Greedy Self-Report Across Turns', fontsize=12, y=1.02)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_A_greedy_self_report_vs_turn')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_A',
        'title': 'Greedy Self-Report Across Turns',
        'description': ('Greedy-decoded numerical self-reports (0–9) show minimal variance '
                        'across conversation turns. Individual conversations shown in light, '
                        'mean with 95% CI in bold. LLaMA-3.2-3B-Instruct, 40 conversations.'),
        'data_sources': {k: v for k, v in LLAMA_3B_GREEDY.items()},
        'model': 'LLaMA-3.2-3B-Instruct',
        'n_conversations': 40,
        'n_turns': 10,
        'rating_type': 'token (greedy)',
        'per_concept_stats': greedy_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel B: Probe score drift vs turn (4 concepts)
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    drift_stats = {}
    for i, short in enumerate(['wellbeing', 'interest', 'focus', 'impulsivity']):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]

        df = _load_concept_df(LLAMA_3B_RERUN_SELF, short)
        if df is not None:
            # Get probe scores at alpha=0 with flip
            sub = df[np.isclose(df['alpha'], 0.0)].copy()
            sub['probe_score_display'] = flip_if_needed(concept, sub['probe_score'].values)

            per_turn = compute_per_turn_means(sub.assign(
                probe_score=sub['probe_score_display']), 'probe_score', alpha_val=0.0)
            # Recompute since we modified values
            turns = sorted(sub['turn'].unique())
            means, ci_lows, ci_highs = [], [], []
            for t in turns:
                vals = sub[sub['turn'] == t]['probe_score_display'].dropna().values
                m = np.mean(vals)
                rng = np.random.RandomState(42)
                boots = [np.mean(rng.choice(vals, len(vals))) for _ in range(1000)]
                means.append(m)
                ci_lows.append(np.percentile(boots, 2.5))
                ci_highs.append(np.percentile(boots, 97.5))

            plot_line_with_ci(ax, turns, means, ci_lows, ci_highs, color=color)
            # Light traces
            for ci_idx in sub['conversation_index'].unique():
                conv = sub[sub['conversation_index'] == ci_idx].sort_values('turn')
                ax.plot(conv['turn'], conv['probe_score_display'],
                        color=color, alpha=0.06, linewidth=0.5)

            rho, p = stats.spearmanr(sub['turn'], sub['probe_score_display'])
            drift_stats[short] = {
                'mean_first_turn': round(float(means[0]), 4),
                'mean_last_turn': round(float(means[-1]), 4),
                'drift_magnitude': round(float(means[-1] - means[0]), 4),
                'spearman_rho_vs_turn': round(float(rho), 4),
                'spearman_p_vs_turn': float(p),
            }

        ax.set_xlabel('Turn')
        if i == 0:
            ax.set_ylabel('Probe Score')
        ax.set_title(CONCEPT_DISPLAY[concept])
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
    add_panel_label(axes[0], 'B', x=-0.18)
    fig.suptitle('Internal State (Probe Score) Across Turns', fontsize=12, y=1.02)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_B_probe_drift_vs_turn')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_B',
        'title': 'Internal State Drift Across Turns',
        'description': ('Probe scores (flipped for intuitive direction) show clear drift '
                        'across conversation turns, despite flat greedy self-reports. '
                        'Individual conversations in light, mean with 95% CI in bold.'),
        'data_sources': {k: v for k, v in LLAMA_3B_RERUN_SELF.items()},
        'model': 'LLaMA-3.2-3B-Instruct',
        'metric': PROBE_METRIC_KEY,
        'alpha': 0.0,
        'per_concept_drift': drift_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel C: Number of unique greedy responses per turn
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    for short in ['wellbeing', 'interest', 'focus', 'impulsivity']:
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        df = _load_concept_df(LLAMA_3B_GREEDY, short)
        if df is not None:
            uniq = compute_per_turn_unique_counts(df, 'token_rating', alpha_val=0.0)
            ax.plot(uniq['turn'], uniq['n_unique'], 'o-', color=color,
                    label=CONCEPT_DISPLAY[concept], markersize=5)
    ax.set_xlabel('Turn')
    ax.set_ylabel('Unique Responses')
    ax.set_title('Unique Greedy Responses per Turn')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.5, None)
    ax.legend(fontsize=8)
    add_panel_label(ax, 'C')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_C_unique_greedy_responses')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_C',
        'title': 'Unique Greedy Responses per Turn',
        'description': ('Number of unique discrete ratings (out of 10 possible: 0–9) '
                        'given by the model across 40 conversations at each turn. '
                        'Low counts indicate high response stereotypy.'),
        'data_sources': {k: v for k, v in LLAMA_3B_GREEDY.items()},
    })

    # ──────────────────────────────────────────────────────────────
    # Panel D: Non-greedy (sampled) token self-report vs turn
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
    for i, short in enumerate(['wellbeing', 'interest', 'focus', 'impulsivity']):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]

        df = _load_concept_df(LLAMA_3B_RERUN_SELF, short)
        if df is not None:
            per_turn = compute_per_turn_means(df, 'token_rating', alpha_val=0.0)
            plot_line_with_ci(ax, per_turn['turn'], per_turn['mean'],
                              per_turn['ci_low'], per_turn['ci_high'],
                              color=color)
            for ci_idx in df[np.isclose(df['alpha'], 0.0)]['conversation_index'].unique():
                conv = df[(df['conversation_index'] == ci_idx) &
                          np.isclose(df['alpha'], 0.0)].sort_values('turn')
                ax.plot(conv['turn'], conv['token_rating'],
                        color=color, alpha=0.06, linewidth=0.5)
        ax.set_xlabel('Turn')
        if i == 0:
            ax.set_ylabel('Self-Report (sampled token)')
        ax.set_title(CONCEPT_DISPLAY[concept])
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(-0.5, 9.5)
        ax.set_xticks(range(1, 11))
    add_panel_label(axes[0], 'D', x=-0.18)
    fig.suptitle('Non-Greedy (Sampled) Self-Report Across Turns', fontsize=12, y=1.02)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_D_sampled_token_self_report_vs_turn')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_D',
        'title': 'Non-Greedy (Sampled) Self-Report Across Turns',
        'description': ('Sampled (temperature=0.8) token-based self-reports. More variable '
                        'than greedy but still discrete (integer-valued) and relatively noisy.'),
        'data_sources': {k: v for k, v in LLAMA_3B_RERUN_SELF.items()},
        'rating_type': 'token (sampled, temp=0.8)',
    })

    # ──────────────────────────────────────────────────────────────
    # Panel E: Logit-based self-report vs turn
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
    logit_drift_stats = {}
    for i, short in enumerate(['wellbeing', 'interest', 'focus', 'impulsivity']):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]

        df = _load_concept_df(LLAMA_3B_RERUN_SELF, short)
        if df is not None:
            per_turn = compute_per_turn_means(df, 'logit_rating', alpha_val=0.0)
            plot_line_with_ci(ax, per_turn['turn'], per_turn['mean'],
                              per_turn['ci_low'], per_turn['ci_high'],
                              color=color)
            for ci_idx in df[np.isclose(df['alpha'], 0.0)]['conversation_index'].unique():
                conv = df[(df['conversation_index'] == ci_idx) &
                          np.isclose(df['alpha'], 0.0)].sort_values('turn')
                ax.plot(conv['turn'], conv['logit_rating'],
                        color=color, alpha=0.06, linewidth=0.5)

            sub = df[np.isclose(df['alpha'], 0.0)]
            rho, p = stats.spearmanr(sub['turn'], sub['logit_rating'].dropna())
            logit_drift_stats[short] = {
                'mean_first_turn': round(float(per_turn['mean'].iloc[0]), 4),
                'mean_last_turn': round(float(per_turn['mean'].iloc[-1]), 4),
                'drift_magnitude': round(float(
                    per_turn['mean'].iloc[-1] - per_turn['mean'].iloc[0]), 4),
                'spearman_rho_vs_turn': round(float(rho), 4),
                'spearman_p_vs_turn': float(p),
            }

        ax.set_xlabel('Turn')
        if i == 0:
            ax.set_ylabel('Self-Report (logit-based)')
        ax.set_title(CONCEPT_DISPLAY[concept])
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
                        'digit token logits). Shows drift aligned with probe score drift, '
                        'confirming that models track internal state changes.'),
        'data_sources': {k: v for k, v in LLAMA_3B_RERUN_SELF.items()},
        'rating_type': 'logit-based (probability-weighted average)',
        'per_concept_drift': logit_drift_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel F: Variance comparison (greedy vs sampled-token vs logit)
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    bar_data = {'Greedy\n(token)': [], 'Sampled\n(token)': [], 'Logit-\nbased': []}
    for short in ['wellbeing', 'interest', 'focus', 'impulsivity']:
        concept = SHORTHAND_TO_CONCEPT[short]
        # Greedy variance
        df_g = _load_concept_df(LLAMA_3B_GREEDY, short)
        if df_g is not None:
            bar_data['Greedy\n(token)'].append(df_g['token_rating'].var())
        else:
            bar_data['Greedy\n(token)'].append(0)
        # Sampled token variance
        df_s = _load_concept_df(LLAMA_3B_RERUN_SELF, short)
        if df_s is not None:
            sub = df_s[np.isclose(df_s['alpha'], 0.0)]
            bar_data['Sampled\n(token)'].append(sub['token_rating'].dropna().var())
            bar_data['Logit-\nbased'].append(sub['logit_rating'].dropna().var())
        else:
            bar_data['Sampled\n(token)'].append(0)
            bar_data['Logit-\nbased'].append(0)

    x = np.arange(4)
    width = 0.25
    methods = list(bar_data.keys())
    method_colors = ['#888888', '#5DADE2', '#2ECC71']
    for j, method in enumerate(methods):
        ax.bar(x + j * width, bar_data[method], width, label=method,
               color=method_colors[j], edgecolor='white', linewidth=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels([CONCEPT_DISPLAY[SHORTHAND_TO_CONCEPT[s]]
                        for s in ['wellbeing', 'interest', 'focus', 'impulsivity']],
                       fontsize=8)
    ax.set_ylabel('Variance of Self-Report')
    ax.set_title('Self-Report Variance by Method')
    ax.legend(fontsize=8)
    add_panel_label(ax, 'F')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_F_variance_comparison')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_F',
        'title': 'Self-Report Variance by Method',
        'description': ('Variance of self-reports across three methods: greedy token, '
                        'sampled token, and logit-based. Logit-based captures more variance '
                        'and produces a continuous rather than discrete measure.'),
        'variance_data': bar_data,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel G: Correlation between logit and token ratings
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    corr_stats = {}
    for i, short in enumerate(['wellbeing', 'interest', 'focus', 'impulsivity']):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]

        df = _load_concept_df(LLAMA_3B_RERUN_SELF, short)
        if df is not None:
            sub = df[np.isclose(df['alpha'], 0.0)].dropna(
                subset=['token_rating', 'logit_rating'])
            ax.scatter(sub['token_rating'], sub['logit_rating'],
                       color=color, alpha=0.3, s=15, edgecolors='none')
            # Fit line
            if len(sub) > 2:
                slope, intercept, r, p, se = stats.linregress(
                    sub['token_rating'], sub['logit_rating'])
                x_fit = np.array([sub['token_rating'].min(), sub['token_rating'].max()])
                ax.plot(x_fit, slope * x_fit + intercept, 'k--', linewidth=1, alpha=0.7)
                rho_s, p_s = stats.spearmanr(sub['token_rating'], sub['logit_rating'])
                corr_stats[short] = {
                    'pearson_r': round(float(r), 4),
                    'pearson_p': float(p),
                    'spearman_rho': round(float(rho_s), 4),
                    'spearman_p': float(p_s),
                    'n': len(sub),
                }
                ax.text(0.05, 0.95,
                        f'ρ = {rho_s:.3f}\n{format_p(p_s)}',
                        transform=ax.transAxes, fontsize=8, va='top')
        ax.set_xlabel('Token Rating')
        if i == 0:
            ax.set_ylabel('Logit Rating')
        ax.set_title(CONCEPT_DISPLAY[concept])
        ax.plot([0, 9], [0, 9], 'k:', alpha=0.3, linewidth=0.8)
    add_panel_label(axes[0], 'G', x=-0.18)
    fig.suptitle('Logit vs Token Self-Report Correlation', fontsize=12, y=1.02)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_02_G_logit_vs_token_correlation')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_02_G',
        'title': 'Logit vs Token Self-Report Correlation',
        'description': ('Correlation between logit-based and token-based self-reports '
                        'at alpha=0. Logit-based ratings are calibrated with token ratings '
                        'but provide continuous, less noisy measurements.'),
        'correlation_stats': corr_stats,
    })

    # ── Save other_stats ──
    other_stats = {
        'description': ('Figure 2 demonstrates internal state drift through conversations '
                        'and validates different self-report extraction methods. Key finding: '
                        'models show internal state drift but fail to report it with greedy '
                        'decoding. Logit-based method captures drift with continuous values.'),
        'greedy_stats': greedy_stats,
        'probe_drift_stats': drift_stats,
        'logit_drift_stats': logit_drift_stats,
        'logit_vs_token_correlation': corr_stats,
    }
    save_other_stats(fig_dir, other_stats)
    print("    Figure 2 complete.")


if __name__ == '__main__':
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rdir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    generate_figure_2(rdir)
    print(f"Output → {rdir}")
