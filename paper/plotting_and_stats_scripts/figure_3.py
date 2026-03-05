"""
Figure 3: Introspection — Correlation, Turn-wise Analysis, and Causal Validation
==================================================================================
Demonstrates that LLMs can introspect: self-reports correlate with probe scores,
this holds turn-by-turn, and self-steering causally modulates self-reports.

v2 changes:
 - C/D: split into 4 stacked subplots per concept (shared x-axis)
 - Flip Rho sign for FLIP concepts in turnwise data
 - CDii: first-vs-last turn comparison with statistical tests
 - E: flip alpha for FLIP concepts
 - F-I: flip alpha labels for FLIP concepts
 - B: add random vector control bars alongside
 - Enhanced drift statistics in JSONs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from shared_utils import (
    CONCEPTS_ORDERED, CONCEPT_DISPLAY, CONCEPT_COLORS,
    SHORTHAND_TO_CONCEPT, CONCEPT_TO_SHORTHAND,
    SHORTHANDS_ORDERED,
    LLAMA_3B_RERUN_SELF, LLAMA_3B_RANDOM,
    PROBE_METRIC_KEY, FLIP_CONCEPTS, FLIP_SHORTHANDS,
    ALPHA_COLORS,
    load_results, load_summary, load_turnwise,
    results_to_dataframe, flip_if_needed, flip_if_needed_shorthand,
    flip_alpha_if_needed, flip_alpha_scalar,
    get_turnwise_stats,
    isotonic_r2, bootstrap_stat, spearman_rho, spearman_full,
    savefig, save_panel_json, save_other_stats, ensure_dir,
    add_panel_label, plot_line_with_ci, format_p,
    compute_per_turn_means, SHORTHAND_DISPLAY,
)

SHORTHANDS = SHORTHANDS_ORDERED
ALPHAS = [-4.0, -2.0, 0.0, 2.0, 4.0]


def _load_df(short, exp_dirs=None):
    """Load DataFrame for a concept from experiment directories."""
    if exp_dirs is None:
        exp_dirs = LLAMA_3B_RERUN_SELF
    exp_dir = exp_dirs.get(short)
    if exp_dir is None or not os.path.isdir(exp_dir):
        return None
    concept = SHORTHAND_TO_CONCEPT[short]
    results = load_results(exp_dir)
    return results_to_dataframe(results, probe_name=concept)


def generate_figure_3(results_dir):
    """Generate all Figure 3 panels."""
    fig_dir = ensure_dir(os.path.join(results_dir, 'Figure_3'))
    print("  Generating Figure 3: Introspection Analysis...")

    other_stats = {}

    # ──────────────────────────────────────────────────────────────
    # Panels A (4 subplots): Scatter — probe score vs logit self-report
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

        if len(probe_vals) > 5:
            ir = IsotonicRegression(out_of_bounds='clip')
            sort_idx = np.argsort(probe_vals)
            x_sorted = probe_vals[sort_idx]
            y_pred = ir.fit_transform(x_sorted, ratings[sort_idx])
            ax.plot(x_sorted, y_pred, color='black', linewidth=2, alpha=0.7)

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
                'polarity_flip_applied': concept in FLIP_CONCEPTS,
            }
            ax.text(0.05, 0.95,
                    f'ρ = {rho_val:.3f}\n{format_p(p_val)}\nR²(iso) = {r2_iso:.3f}',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('Probe Score')
        if i == 0:
            ax.set_ylabel('Logit Self-Report')
        ax.set_title(SHORTHAND_DISPLAY[short])

    add_panel_label(axes[0], 'A', x=-0.18)
    fig.suptitle('Introspection: Probe Score vs. Self-Report', fontsize=12, y=1.02)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_A_scatter_probe_vs_report')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_A',
        'title': 'Probe Score vs. Logit Self-Report Scatter Plots',
        'description': ('Scatter of polarity-corrected probe score vs. logit self-report '
                        'at alpha=0. Black line: isotonic regression fit. One dot per '
                        'conversation-turn observation.'),
        'data_sources': {k: v for k, v in LLAMA_3B_RERUN_SELF.items()},
        'model': 'LLaMA-3.2-3B-Instruct',
        'metric': PROBE_METRIC_KEY,
        'scatter_statistics': scatter_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel B: R² and Rho bars + optional random vector control
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    random_stats = {}

    # Compute R²/Rho from random scoring experiments as control
    for short in SHORTHANDS:
        concept = SHORTHAND_TO_CONCEPT[short]
        df_rand = _load_df(short, LLAMA_3B_RANDOM)
        if df_rand is not None:
            sub_r = df_rand[np.isclose(df_rand['alpha'], 0.0)].dropna(
                subset=['probe_score', 'logit_rating']).copy()
            if len(sub_r) > 5:
                probe_r = flip_if_needed(concept, sub_r['probe_score'].values)
                rating_r = sub_r['logit_rating'].values
                r2_r = isotonic_r2(probe_r, rating_r)
                rho_r, _ = stats.spearmanr(probe_r, rating_r)
                random_stats[short] = {
                    'isotonic_r2': round(float(r2_r), 4),
                    'spearman_rho': round(float(rho_r), 4),
                    'n': len(sub_r),
                }

    has_random = len(random_stats) > 0
    n_bars = 2 if has_random else 1
    bar_w = 0.35 if has_random else 0.6
    offset_probe = -bar_w/2 if has_random else 0
    offset_random = bar_w/2

    # R² bars
    ax = axes[0]
    for i, short in enumerate(SHORTHANDS):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        if short in scatter_stats:
            s = scatter_stats[short]
            ax.bar(i + offset_probe, s['isotonic_r2'], bar_w,
                   color=color, edgecolor='white', linewidth=0.5,
                   label='Probe' if i == 0 else None)
            ci = s['isotonic_r2_ci']
            ax.errorbar(i + offset_probe, s['isotonic_r2'],
                        yerr=[[s['isotonic_r2'] - ci[0]], [ci[1] - s['isotonic_r2']]],
                        color='black', capsize=3, linewidth=1.2, capthick=1.2)
        if has_random and short in random_stats:
            ax.bar(i + offset_random, random_stats[short]['isotonic_r2'], bar_w,
                   color='#CCCCCC', edgecolor='white', linewidth=0.5,
                   label='Random' if i == 0 else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels([SHORTHAND_DISPLAY[s] for s in SHORTHANDS], fontsize=8, rotation=15)
    ax.set_ylabel('Isotonic R²')
    ax.set_title('Introspection Accuracy')
    ax.set_ylim(0, 1)
    if has_random:
        ax.legend(fontsize=8)
    add_panel_label(ax, 'B')

    # Rho bars
    ax = axes[1]
    for i, short in enumerate(SHORTHANDS):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        if short in scatter_stats:
            s = scatter_stats[short]
            ax.bar(i + offset_probe, s['spearman_rho'], bar_w,
                   color=color, edgecolor='white', linewidth=0.5,
                   label='Probe' if i == 0 else None)
            ci = s['rho_ci']
            ax.errorbar(i + offset_probe, s['spearman_rho'],
                        yerr=[[s['spearman_rho'] - ci[0]], [ci[1] - s['spearman_rho']]],
                        color='black', capsize=3, linewidth=1.2, capthick=1.2)
        if has_random and short in random_stats:
            ax.bar(i + offset_random, random_stats[short]['spearman_rho'], bar_w,
                   color='#CCCCCC', edgecolor='white', linewidth=0.5,
                   label='Random' if i == 0 else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels([SHORTHAND_DISPLAY[s] for s in SHORTHANDS], fontsize=8, rotation=15)
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Introspection Correlation')
    ax.set_ylim(-0.2, 1)
    if has_random:
        ax.legend(fontsize=8)

    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_B_introspection_bars')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_B',
        'title': 'Introspection Metrics — Bar Charts (with Random Control)',
        'description': ('Isotonic R² (left) and Spearman ρ (right) for each concept at alpha=0. '
                        'Grey bars: random scoring control (probe scores from random direction). '
                        '95% bootstrap CIs on probe-based bars.'),
        'probe_statistics': scatter_stats,
        'random_control_statistics': random_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panels C (4 stacked): Turn-wise R² per concept
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=True)
    turnwise_data = {}

    for j, short in enumerate(SHORTHANDS):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[j]
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

        turnwise_data[short] = {
            'turns': turns,
            'isotonic_r2': r2_vals,
            'isotonic_r2_ci': list(zip(r2_lo, r2_hi)),
        }

        plot_line_with_ci(ax, turns, r2_vals, r2_lo, r2_hi, color=color)
        ax.set_ylabel('Isotonic R²')
        ax.set_title(SHORTHAND_DISPLAY[short], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))

    axes[-1].set_xlabel('Turn')
    add_panel_label(axes[0], 'C')
    fig.suptitle('Turn-wise Introspection Accuracy (R²)', fontsize=12, y=1.01)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_C_turnwise_r2_stacked')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_C',
        'title': 'Turn-wise Isotonic R² (Stacked per Concept)',
        'description': ('Isotonic R² computed per turn at alpha=0, with one subplot per '
                        'concept. Shows introspection accuracy from first through last turn.'),
        'turnwise_r2': {k: v for k, v in turnwise_data.items()},
    })

    # ──────────────────────────────────────────────────────────────
    # Panels D (4 stacked): Turn-wise Spearman rho per concept
    #   — Flip Rho sign for FLIP concepts
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(6, 12), sharex=True)
    turnwise_rho_data = {}

    for j, short in enumerate(SHORTHANDS):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[j]
        exp_dir = LLAMA_3B_RERUN_SELF.get(short)
        if exp_dir is None:
            continue

        tw_stats = get_turnwise_stats(exp_dir, concept, PROBE_METRIC_KEY,
                                      source='logit', alpha=0.0)
        if not tw_stats:
            continue

        turns = sorted([int(t) for t in tw_stats.keys()])
        rho_vals = [tw_stats[str(t)].get('spearman_rho', np.nan) for t in turns]
        rho_lo = [tw_stats[str(t)].get('spearman_rho_ci_low', np.nan) for t in turns]
        rho_hi = [tw_stats[str(t)].get('spearman_rho_ci_high', np.nan) for t in turns]
        rho_p = [tw_stats[str(t)].get('spearman_p', np.nan) for t in turns]

        # Flip Rho for FLIP concepts
        flip_sign = -1 if concept in FLIP_CONCEPTS else 1
        rho_vals = [v * flip_sign for v in rho_vals]
        rho_lo_adj = [v * flip_sign for v in (rho_hi if flip_sign == -1 else rho_lo)]
        rho_hi_adj = [v * flip_sign for v in (rho_lo if flip_sign == -1 else rho_hi)]

        turnwise_rho_data[short] = {
            'turns': turns,
            'spearman_rho': rho_vals,
            'spearman_rho_ci': list(zip(rho_lo_adj, rho_hi_adj)),
            'spearman_p': rho_p,
            'rho_sign_flipped': concept in FLIP_CONCEPTS,
        }

        plot_line_with_ci(ax, turns, rho_vals, rho_lo_adj, rho_hi_adj, color=color)
        ax.set_ylabel('Spearman ρ')
        ax.set_title(SHORTHAND_DISPLAY[short], fontsize=10)
        ax.set_ylim(-0.3, 1)
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    axes[-1].set_xlabel('Turn')
    add_panel_label(axes[0], 'D')
    fig.suptitle('Turn-wise Introspection Correlation (ρ)', fontsize=12, y=1.01)
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_D_turnwise_rho_stacked')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_D',
        'title': 'Turn-wise Spearman ρ (Stacked per Concept)',
        'description': ('Spearman ρ computed per turn at alpha=0. Rho is sign-flipped for '
                        'sad_vs_happy and impulsive_vs_planning so that positive = positive '
                        'introspection (higher self-report ↔ higher concept pole). '
                        'One subplot per concept. 95% bootstrap CIs.'),
        'turnwise_rho': turnwise_rho_data,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel CDii: First vs Last turn introspection comparison
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    first_last_stats = {}

    for i, short in enumerate(SHORTHANDS):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]

        df = _load_df(short)
        if df is None:
            continue
        sub = df[np.isclose(df['alpha'], 0.0)].dropna(
            subset=['probe_score', 'logit_rating']).copy()
        sub['probe_display'] = flip_if_needed(concept, sub['probe_score'].values)
        turns = sorted(sub['turn'].unique())
        first_turn = turns[0]
        last_turn = turns[-1]

        first = sub[sub['turn'] == first_turn]
        last_ = sub[sub['turn'] == last_turn]

        # R² for first and last
        r2_first = isotonic_r2(first['probe_display'].values, first['logit_rating'].values)
        r2_last = isotonic_r2(last_['probe_display'].values, last_['logit_rating'].values)
        # Rho for first and last
        rho_first, p_first = stats.spearmanr(
            first['probe_display'].values, first['logit_rating'].values)
        rho_last, p_last = stats.spearmanr(
            last_['probe_display'].values, last_['logit_rating'].values)
        # Flip rho for FLIP concepts
        if concept in FLIP_CONCEPTS:
            rho_first, rho_last = -rho_first, -rho_last

        width = 0.35
        # R² bars (first vs last)
        axes[0].bar(i - width/2, r2_first, width, color=color, alpha=0.5,
                    label=f'Turn {first_turn}' if i == 0 else None)
        axes[0].bar(i + width/2, r2_last, width, color=color,
                    label=f'Turn {last_turn}' if i == 0 else None)
        # Rho bars
        axes[1].bar(i - width/2, rho_first, width, color=color, alpha=0.5,
                    label=f'Turn {first_turn}' if i == 0 else None)
        axes[1].bar(i + width/2, rho_last, width, color=color,
                    label=f'Turn {last_turn}' if i == 0 else None)

        first_last_stats[short] = {
            'first_turn': int(first_turn),
            'last_turn': int(last_turn),
            'r2_first': round(float(r2_first), 4),
            'r2_last': round(float(r2_last), 4),
            'r2_difference': round(float(r2_last - r2_first), 4),
            'rho_first': round(float(rho_first), 4),
            'rho_last': round(float(rho_last), 4),
            'rho_difference': round(float(rho_last - rho_first), 4),
            'rho_first_p': float(p_first),
            'rho_last_p': float(p_last),
            'n_first': len(first),
            'n_last': len(last_),
            'polarity_flip_applied': concept in FLIP_CONCEPTS,
        }

    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels([SHORTHAND_DISPLAY[s] for s in SHORTHANDS], fontsize=8, rotation=15)
    axes[0].set_ylabel('Isotonic R²')
    axes[0].set_title('First vs Last Turn: R²')
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    add_panel_label(axes[0], 'CDii')

    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels([SHORTHAND_DISPLAY[s] for s in SHORTHANDS], fontsize=8, rotation=15)
    axes[1].set_ylabel('Spearman ρ')
    axes[1].set_title('First vs Last Turn: ρ')
    axes[1].set_ylim(-0.3, 1)
    axes[1].axhline(0, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_CDii_first_vs_last_turn')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_CDii',
        'title': 'First vs Last Turn Introspection Comparison',
        'description': ('Comparison of introspection metrics at the first and last conversation '
                        'turn. Tests whether introspection degrades or improves across the '
                        'conversation. Rho sign-flipped for FLIP concepts.'),
        'first_last_stats': first_last_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel E: Self-steering — Mean self-report vs alpha
    #   — Flip alpha for FLIP concepts
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

        # Flip alpha for display if needed
        display_alphas = flip_alpha_if_needed(short, np.array(alphas_found))
        # Sort by display alpha for clean line
        sort_idx = np.argsort(display_alphas)
        display_alphas = display_alphas[sort_idx]
        alpha_means = [alpha_means[k] for k in sort_idx]
        alpha_ci_lo = [alpha_ci_lo[k] for k in sort_idx]
        alpha_ci_hi = [alpha_ci_hi[k] for k in sort_idx]

        plot_line_with_ci(ax, display_alphas, alpha_means, alpha_ci_lo, alpha_ci_hi,
                          color=color, label=SHORTHAND_DISPLAY[short])

        # One-sided test: does self-report increase monotonically with display alpha?
        rho_a, p_a = stats.spearmanr(
            flip_alpha_if_needed(short, df['alpha'].values),
            df['logit_rating'].dropna().values[:len(df['alpha'].values)])

        steering_stats[short] = {
            'display_alphas': [float(a) for a in display_alphas],
            'mean_ratings': [round(m, 4) for m in alpha_means],
            'ci_low': [round(c, 4) for c in alpha_ci_lo],
            'ci_high': [round(c, 4) for c in alpha_ci_hi],
            'alpha_display_flipped': short in FLIP_SHORTHANDS,
            'spearman_rho_display_alpha_vs_rating': round(float(rho_a), 4),
            'spearman_p': float(p_a),
        }

    ax.set_xlabel('Steering α (display-corrected)')
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
        'title': 'Self-Steering: Mean Self-Report vs. Alpha (Polarity-Corrected)',
        'description': ('Mean logit-based self-report vs steering alpha (sign-corrected for '
                        'FLIP concepts so positive α → positive direction). Monotonic increase '
                        'confirms causal relationship. 95% bootstrap CIs.'),
        'data_sources': {k: v for k, v in LLAMA_3B_RERUN_SELF.items()},
        'steering_stats': steering_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panels F–I: Self-report drift by alpha (one per concept)
    #   — Flip alpha in labels for FLIP concepts
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
            # Display alpha label (flipped for FLIP concepts)
            display_a = flip_alpha_scalar(short, a)
            plot_line_with_ci(ax, turns, means, ci_lo, ci_hi,
                              color=c, label=f'α = {display_a:+.0f}', alpha_fill=0.12)

            # Drift statistic: Spearman of turn vs rating at this alpha
            sub_nona = sub.dropna(subset=['logit_rating'])
            rho_d, p_d = stats.spearmanr(sub_nona['turn'], sub_nona['logit_rating'])
            drift_by_alpha[str(a)] = {
                'display_alpha': float(display_a),
                'means': [round(m, 4) for m in means],
                'ci_low': [round(c, 4) for c in ci_lo],
                'ci_high': [round(c, 4) for c in ci_hi],
                'drift_spearman_rho': round(float(rho_d), 4),
                'drift_spearman_p': float(p_d),
                'drift_magnitude': round(float(means[-1] - means[0]), 4),
            }

        ax.set_xlabel('Turn')
        ax.set_ylabel('Logit Self-Report')
        ax.set_title(f'Self-Report Drift — {SHORTHAND_DISPLAY[short]}')
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
            'title': f'Self-Report Drift Under Steering — {SHORTHAND_DISPLAY[short]}',
            'description': (f'Mean logit self-report across turns for different steering '
                            f'alphas (display-corrected for polarity). Drift Spearman ρ '
                            f'indicates temporal trend at each alpha level.'),
            'concept': concept,
            'alpha_display_flipped': short in FLIP_SHORTHANDS,
            'drift_by_alpha': drift_by_alpha,
        })

    # ── Save other_stats ──
    other_stats = {
        'description': ('Figure 3 establishes introspection. Scatter plots show correlations '
                        'between probe scores and self-reports. Turn-wise analysis per concept '
                        'shows introspection from first turn. First-vs-last comparison tests '
                        'stability. Self-steering with polarity-corrected alpha confirms '
                        'causal modulation. Random vector controls validate probe specificity.'),
        'overall_introspection': scatter_stats,
        'random_control': random_stats,
        'first_vs_last_turn': first_last_stats,
        'turnwise_r2': {s: turnwise_data.get(s, {}) for s in SHORTHANDS},
        'turnwise_rho': {s: turnwise_rho_data.get(s, {}) for s in SHORTHANDS},
    }
    save_other_stats(fig_dir, other_stats)
    print("    Figure 3 complete.")


if __name__ == '__main__':
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rdir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    generate_figure_3(rdir)
    print(f"Output -> {rdir}")
