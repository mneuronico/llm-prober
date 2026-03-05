"""
Figure 3: Introspection — Correlation, Turn-wise Analysis, and Causal Validation
==================================================================================
Demonstrates that LLMs can introspect: self-reports correlate with probe scores,
this holds turn-by-turn, and self-steering causally modulates self-reports.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from shared_utils import (
    CONCEPTS_ORDERED, CONCEPT_DISPLAY, CONCEPT_COLORS,
    SHORTHAND_TO_CONCEPT, CONCEPT_TO_SHORTHAND,
    LLAMA_3B_RERUN_SELF, PROBE_METRIC_KEY, FLIP_CONCEPTS,
    ALPHA_COLORS,
    load_results, load_summary, load_turnwise,
    results_to_dataframe, flip_if_needed, get_turnwise_stats,
    isotonic_r2, bootstrap_stat, spearman_rho, spearman_full,
    savefig, save_panel_json, save_other_stats, ensure_dir,
    add_panel_label, plot_line_with_ci, format_p,
    compute_per_turn_means,
)

SHORTHANDS = ['wellbeing', 'interest', 'focus', 'impulsivity']
ALPHAS = [-4.0, -2.0, 0.0, 2.0, 4.0]


def _load_df(short):
    """Load DataFrame for a concept from rerun experiments."""
    exp_dir = LLAMA_3B_RERUN_SELF.get(short)
    if exp_dir is None:
        return None
    concept = SHORTHAND_TO_CONCEPT[short]
    results = load_results(exp_dir)
    df = results_to_dataframe(results, probe_name=concept)
    return df


def generate_figure_3(results_dir):
    """Generate all Figure 3 panels."""
    fig_dir = ensure_dir(os.path.join(results_dir, 'Figure_3'))
    print("  Generating Figure 3: Introspection Analysis...")

    other_stats = {}

    # ──────────────────────────────────────────────────────────────
    # Panels A–D: Scatter plots — probe score vs logit self-report
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    scatter_stats = {}
    for i, short in enumerate(SHORTHANDS):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]

        df = _load_df(short)
        if df is None:
            continue
        sub = df[np.isclose(df['alpha'], 0.0)].dropna(
            subset=['probe_score', 'logit_rating']).copy()
        probe_vals = flip_if_needed(concept, sub['probe_score'].values)
        ratings = sub['logit_rating'].values

        ax.scatter(probe_vals, ratings, color=color, alpha=0.3, s=12,
                   edgecolors='none')

        # Isotonic regression fit overlay
        if len(probe_vals) > 5:
            ir = IsotonicRegression(out_of_bounds='clip')
            sort_idx = np.argsort(probe_vals)
            x_sorted = probe_vals[sort_idx]
            y_pred = ir.fit_transform(x_sorted, ratings[sort_idx])
            ax.plot(x_sorted, y_pred, color='black', linewidth=2, alpha=0.7)

            # Stats
            rho_val, p_val = stats.spearmanr(probe_vals, ratings)
            r2_iso = isotonic_r2(probe_vals, ratings)
            r2_iso_pt, r2_ci_lo, r2_ci_hi = bootstrap_stat(
                probe_vals, ratings, isotonic_r2)
            rho_pt, rho_ci_lo, rho_ci_hi = bootstrap_stat(
                probe_vals, ratings, spearman_rho)

            scatter_stats[short] = {
                'spearman_rho': round(float(rho_val), 4),
                'spearman_p': float(p_val),
                'isotonic_r2': round(float(r2_iso), 4),
                'isotonic_r2_ci': [round(r2_ci_lo, 4), round(r2_ci_hi, 4)],
                'rho_ci': [round(rho_ci_lo, 4), round(rho_ci_hi, 4)],
                'n': len(probe_vals),
            }
            ax.text(0.05, 0.95,
                    f'ρ = {rho_val:.3f}\n{format_p(p_val)}\nR²(iso) = {r2_iso:.3f}',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('Probe Score (prev. turn)')
        if i == 0:
            ax.set_ylabel('Logit Self-Report')
        ax.set_title(CONCEPT_DISPLAY[concept])

    add_panel_label(axes[0], 'A', x=-0.18)
    fig.suptitle('Introspection: Probe Score vs. Self-Report', fontsize=12, y=1.02)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_A_scatter_probe_vs_report')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_A',
        'title': 'Probe Score vs. Logit Self-Report Scatter Plots',
        'description': ('Scatter plots of probe score (prompt_assistant_last_mean, polarity-'
                        'corrected) vs. logit-based self-report at alpha=0 for all 4 concepts. '
                        'Black line: isotonic regression fit. Each dot = one conversation-turn '
                        'observation (n=400; 40 conversations × 10 turns).'),
        'data_sources': {k: v for k, v in LLAMA_3B_RERUN_SELF.items()},
        'model': 'LLaMA-3.2-3B-Instruct',
        'metric': PROBE_METRIC_KEY,
        'scatter_statistics': scatter_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel B: Bar chart — R² isotonic and Spearman rho (4 concepts)
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # R2 isotonic bars
    ax = axes[0]
    for i, short in enumerate(SHORTHANDS):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        if short in scatter_stats:
            s = scatter_stats[short]
            ax.bar(i, s['isotonic_r2'], color=color, edgecolor='white', linewidth=0.5)
            ci = s['isotonic_r2_ci']
            ax.errorbar(i, s['isotonic_r2'],
                        yerr=[[s['isotonic_r2'] - ci[0]], [ci[1] - s['isotonic_r2']]],
                        color='black', capsize=4, linewidth=1.5, capthick=1.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CONCEPT_DISPLAY[SHORTHAND_TO_CONCEPT[s]]
                        for s in SHORTHANDS], fontsize=8, rotation=15)
    ax.set_ylabel('Isotonic R²')
    ax.set_title('Introspection Accuracy')
    ax.set_ylim(0, 1)
    add_panel_label(ax, 'B')

    # Spearman rho bars
    ax = axes[1]
    for i, short in enumerate(SHORTHANDS):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        if short in scatter_stats:
            s = scatter_stats[short]
            ax.bar(i, s['spearman_rho'], color=color, edgecolor='white', linewidth=0.5)
            ci = s['rho_ci']
            ax.errorbar(i, s['spearman_rho'],
                        yerr=[[s['spearman_rho'] - ci[0]], [ci[1] - s['spearman_rho']]],
                        color='black', capsize=4, linewidth=1.5, capthick=1.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CONCEPT_DISPLAY[SHORTHAND_TO_CONCEPT[s]]
                        for s in SHORTHANDS], fontsize=8, rotation=15)
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Introspection Correlation')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_B_introspection_bars')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_B',
        'title': 'Introspection Metrics — Bar Charts',
        'description': ('Isotonic R² (left) and Spearman ρ (right) for each concept, '
                        'measuring overall introspection accuracy at alpha=0. 95% '
                        'bootstrap CIs shown as error bars.'),
        'statistics': scatter_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panels C–D: Turn-wise R² isotonic and Spearman rho
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    turnwise_data = {}

    for short in SHORTHANDS:
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        exp_dir = LLAMA_3B_RERUN_SELF.get(short)
        if exp_dir is None:
            continue

        tw_stats = get_turnwise_stats(exp_dir, concept, PROBE_METRIC_KEY,
                                      source='logit', alpha=0.0)
        if not tw_stats:
            continue

        turns = sorted([int(t) for t in tw_stats.keys()])
        r2_vals = [tw_stats[str(t)].get('isotonic_r2', np.nan) for t in turns]
        r2_lo = [tw_stats[str(t)].get('isotonic_r2_ci_low', np.nan) for t in turns]
        r2_hi = [tw_stats[str(t)].get('isotonic_r2_ci_high', np.nan) for t in turns]
        rho_vals = [tw_stats[str(t)].get('spearman_rho', np.nan) for t in turns]
        rho_lo = [tw_stats[str(t)].get('spearman_rho_ci_low', np.nan) for t in turns]
        rho_hi = [tw_stats[str(t)].get('spearman_rho_ci_high', np.nan) for t in turns]

        turnwise_data[short] = {
            'turns': turns,
            'isotonic_r2': r2_vals,
            'isotonic_r2_ci': list(zip(r2_lo, r2_hi)),
            'spearman_rho': rho_vals,
            'spearman_rho_ci': list(zip(rho_lo, rho_hi)),
            'spearman_p': [tw_stats[str(t)].get('spearman_p', np.nan) for t in turns],
        }

        # R2 plot
        plot_line_with_ci(axes[0], turns, r2_vals, r2_lo, r2_hi,
                          color=color, label=CONCEPT_DISPLAY[concept])
        # Rho plot
        plot_line_with_ci(axes[1], turns, rho_vals, rho_lo, rho_hi,
                          color=color, label=CONCEPT_DISPLAY[concept])

    axes[0].set_xlabel('Turn')
    axes[0].set_ylabel('Isotonic R²')
    axes[0].set_title('Introspection Accuracy by Turn')
    axes[0].set_xlim(0.5, 10.5)
    axes[0].set_xticks(range(1, 11))
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7, loc='best')
    add_panel_label(axes[0], 'C')

    axes[1].set_xlabel('Turn')
    axes[1].set_ylabel('Spearman ρ')
    axes[1].set_title('Introspection Correlation by Turn')
    axes[1].set_xlim(0.5, 10.5)
    axes[1].set_xticks(range(1, 11))
    axes[1].set_ylim(-0.2, 1)
    axes[1].legend(fontsize=7, loc='best')
    add_panel_label(axes[1], 'D')

    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_CD_turnwise_introspection')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_C_D',
        'title': 'Turn-wise Introspection Metrics',
        'description': ('Isotonic R² (C) and Spearman ρ (D) computed per turn (n=40 per '
                        'turn) at alpha=0, with 95% bootstrap CIs. Shows introspection '
                        'exists from the first turn and persists across the conversation.'),
        'turnwise_data': turnwise_data,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel E: Self-steering — Mean self-report vs alpha (4 concepts)
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    steering_stats = {}
    for short in SHORTHANDS:
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        df = _load_df(short)
        if df is None:
            continue

        alpha_means = []
        alpha_ci_lo = []
        alpha_ci_hi = []
        alphas_found = sorted(df['alpha'].unique())
        for a in alphas_found:
            sub = df[np.isclose(df['alpha'], a)]['logit_rating'].dropna().values
            m = np.mean(sub)
            rng = np.random.RandomState(42)
            boots = [np.mean(rng.choice(sub, len(sub))) for _ in range(1000)]
            alpha_means.append(m)
            alpha_ci_lo.append(np.percentile(boots, 2.5))
            alpha_ci_hi.append(np.percentile(boots, 97.5))

        plot_line_with_ci(ax, alphas_found, alpha_means, alpha_ci_lo, alpha_ci_hi,
                          color=color, label=CONCEPT_DISPLAY[concept])

        steering_stats[short] = {
            'alphas': [float(a) for a in alphas_found],
            'mean_ratings': [round(m, 4) for m in alpha_means],
            'ci_low': [round(c, 4) for c in alpha_ci_lo],
            'ci_high': [round(c, 4) for c in alpha_ci_hi],
        }

    ax.set_xlabel('Steering α')
    ax.set_ylabel('Mean Logit Self-Report')
    ax.set_title('Self-Steering: Self-Report vs. Alpha')
    ax.legend(fontsize=8)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    add_panel_label(ax, 'E')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_E_self_steering_mean_report')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_E',
        'title': 'Self-Steering: Mean Self-Report vs. Alpha',
        'description': ('Mean logit-based self-report as a function of steering alpha for '
                        'all 4 concepts. Monotonic increase confirms causal relationship '
                        'between internal state direction and self-reported value.'),
        'data_sources': {k: v for k, v in LLAMA_3B_RERUN_SELF.items()},
        'steering_stats': steering_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panels F–I: Self-report drift by alpha (one per concept)
    # ──────────────────────────────────────────────────────────────
    panel_labels = ['F', 'G', 'H', 'I']
    for pi, short in enumerate(SHORTHANDS):
        concept = SHORTHAND_TO_CONCEPT[short]
        df = _load_df(short)
        if df is None:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        alphas_found = sorted(df['alpha'].unique())
        drift_by_alpha = {}

        for a in alphas_found:
            sub = df[np.isclose(df['alpha'], a)]
            turns = sorted(sub['turn'].unique())
            means, ci_lo, ci_hi = [], [], []
            for t in turns:
                vals = sub[sub['turn'] == t]['logit_rating'].dropna().values
                m = np.mean(vals)
                rng = np.random.RandomState(42)
                boots = [np.mean(rng.choice(vals, len(vals))) for _ in range(1000)]
                means.append(m)
                ci_lo.append(np.percentile(boots, 2.5))
                ci_hi.append(np.percentile(boots, 97.5))

            c = ALPHA_COLORS.get(a, 'gray')
            plot_line_with_ci(ax, turns, means, ci_lo, ci_hi,
                              color=c, label=f'α = {a:+.0f}', alpha_fill=0.12)
            drift_by_alpha[str(a)] = {
                'means': [round(m, 4) for m in means],
                'ci_low': [round(c, 4) for c in ci_lo],
                'ci_high': [round(c, 4) for c in ci_hi],
            }

        ax.set_xlabel('Turn')
        ax.set_ylabel('Logit Self-Report')
        ax.set_title(f'Self-Report Drift — {CONCEPT_DISPLAY[concept]}')
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=7, loc='best')
        add_panel_label(ax, panel_labels[pi])
        plt.tight_layout()
        prefix = os.path.join(fig_dir,
                              f'Fig_03_{panel_labels[pi]}_steering_drift_{concept}')
        savefig(fig, prefix)
        save_panel_json(prefix, {
            'panel_id': f'Fig_03_{panel_labels[pi]}',
            'title': f'Self-Report Drift Under Steering — {CONCEPT_DISPLAY[concept]}',
            'description': (f'Mean logit self-report across turns for different steering '
                            f'alphas. Shows that steering shifts the baseline level while '
                            f'temporal drift persists.'),
            'concept': concept,
            'drift_by_alpha': drift_by_alpha,
        })

    # ── Save other_stats ──
    other_stats = {
        'description': ('Figure 3 establishes introspection capability. Scatter plots show '
                        'significant correlation between probe scores and self-reports. '
                        'Turn-wise analysis confirms this is not a temporal confound. '
                        'Self-steering shows causal modulation.'),
        'overall_introspection': scatter_stats,
        'turnwise_introspection': {
            short: {
                'n_significant_turns': sum(
                    1 for p in turnwise_data.get(short, {}).get('spearman_p', [])
                    if p < 0.05
                ),
                'total_turns': len(turnwise_data.get(short, {}).get('turns', [])),
            } for short in SHORTHANDS if short in turnwise_data
        },
    }
    save_other_stats(fig_dir, other_stats)
    print("    Figure 3 complete.")


if __name__ == '__main__':
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rdir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    generate_figure_3(rdir)
    print(f"Output → {rdir}")
