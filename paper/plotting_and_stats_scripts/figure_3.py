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
import pandas as pd
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
    cluster_bootstrap_stat, per_conversation_spearman, one_sample_test,
    lmm_test, corrected_drift_stats, corrected_steering_stats,
    corrected_correlation_stats,
    savefig, save_panel_json, save_other_stats, ensure_dir,
    add_panel_label, plot_line_with_ci, format_p,
    compute_per_turn_means, compute_grouped_cluster_means,
    paired_difference_test, SHORTHAND_DISPLAY,
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


def _per_conversation_metric_map(df, x_col, y_col, metric_fn, min_obs=3):
    """Compute one scalar metric per conversation."""
    metric_map = {}
    for conv_id in sorted(df['conversation_index'].unique()):
        sub = df[df['conversation_index'] == conv_id].dropna(subset=[x_col, y_col])
        if len(sub) < min_obs:
            continue
        try:
            value = metric_fn(sub[x_col].values, sub[y_col].values)
        except Exception:
            continue
        if np.isfinite(value):
            metric_map[int(conv_id)] = float(value)
    return metric_map


def _cluster_bootstrap_metric_delta(true_df, rand_df, x_col, y_col, metric_fn,
                                    n_bootstrap=1000, seed=42):
    """
    Cluster-bootstrap the pooled metric difference between true and random runs.
    """
    common = sorted(
        set(true_df['conversation_index'].unique()) &
        set(rand_df['conversation_index'].unique())
    )
    if len(common) < 3:
        return {
            'n_paired_conversations': len(common),
            'observed_delta': np.nan,
            'bootstrap_ci_95': [np.nan, np.nan],
            'bootstrap_p_two_sided': np.nan,
        }

    true_sub = true_df[true_df['conversation_index'].isin(common)].copy()
    rand_sub = rand_df[rand_df['conversation_index'].isin(common)].copy()

    true_cache = {}
    rand_cache = {}
    for conv_id in common:
        t_conv = true_sub[true_sub['conversation_index'] == conv_id]
        r_conv = rand_sub[rand_sub['conversation_index'] == conv_id]
        true_cache[int(conv_id)] = (
            t_conv[x_col].to_numpy(dtype=float),
            t_conv[y_col].to_numpy(dtype=float),
        )
        rand_cache[int(conv_id)] = (
            r_conv[x_col].to_numpy(dtype=float),
            r_conv[y_col].to_numpy(dtype=float),
        )

    observed_delta = float(
        metric_fn(true_sub[x_col].to_numpy(dtype=float),
                  true_sub[y_col].to_numpy(dtype=float))
        - metric_fn(rand_sub[x_col].to_numpy(dtype=float),
                    rand_sub[y_col].to_numpy(dtype=float))
    )

    rng = np.random.RandomState(seed)
    common_arr = np.array(common, dtype=int)
    boot_deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(common_arr, size=len(common_arr), replace=True)
        true_x = np.concatenate([true_cache[int(c)][0] for c in sampled])
        true_y = np.concatenate([true_cache[int(c)][1] for c in sampled])
        rand_x = np.concatenate([rand_cache[int(c)][0] for c in sampled])
        rand_y = np.concatenate([rand_cache[int(c)][1] for c in sampled])
        try:
            delta = float(metric_fn(true_x, true_y) - metric_fn(rand_x, rand_y))
        except Exception:
            continue
        if np.isfinite(delta):
            boot_deltas.append(delta)

    if not boot_deltas:
        return {
            'n_paired_conversations': len(common),
            'observed_delta': observed_delta,
            'bootstrap_ci_95': [np.nan, np.nan],
            'bootstrap_p_two_sided': np.nan,
        }

    boot_deltas = np.asarray(boot_deltas, dtype=float)
    ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])
    lower_tail = (np.sum(boot_deltas <= 0) + 1) / (len(boot_deltas) + 1)
    upper_tail = (np.sum(boot_deltas >= 0) + 1) / (len(boot_deltas) + 1)
    p_two_sided = min(1.0, 2 * min(lower_tail, upper_tail))
    return {
        'n_paired_conversations': len(common),
        'observed_delta': observed_delta,
        'bootstrap_ci_95': [float(ci_low), float(ci_high)],
        'bootstrap_p_two_sided': float(p_two_sided),
    }


def generate_figure_3(results_dir):
    """Generate all Figure 3 panels."""
    fig_dir = ensure_dir(os.path.join(results_dir, 'Figure_3'))
    appendix_dir = ensure_dir(os.path.join(results_dir, 'Appendix_Figure_3'))
    print("  Generating Figure 3: Introspection Analysis...")

    other_stats = {}

    # ──────────────────────────────────────────────────────────────
    # Panels A (4 subplots): Scatter — probe score vs logit self-report
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(4.5, 16))
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

            # Cluster bootstrap (resample at conversation level)
            conv_ids = sub['conversation_index'].values
            r2_cpt, r2_cci_lo, r2_cci_hi = cluster_bootstrap_stat(
                conv_ids, probe_vals, ratings, isotonic_r2)
            rho_cpt, rho_cci_lo, rho_cci_hi = cluster_bootstrap_stat(
                conv_ids, probe_vals, ratings, spearman_rho)

            # Corrected: per-conversation rho + LMM
            sub_corr = sub.copy()
            sub_corr['probe_display'] = probe_vals
            corrected_a = corrected_correlation_stats(
                sub_corr, 'probe_display', 'logit_rating', alpha_val=0.0)

            scatter_stats[short] = {
                'spearman_rho_POOLED': round(float(rho_val), 4),
                'spearman_p_POOLED': float(p_val),
                'isotonic_r2': round(float(r2_iso), 4),
                'observation_bootstrap_r2_ci': [round(r2_ci_lo, 4), round(r2_ci_hi, 4)],
                'observation_bootstrap_rho_ci': [round(rho_ci_lo, 4), round(rho_ci_hi, 4)],
                'cluster_bootstrap_r2_ci': [round(float(r2_cci_lo), 4), round(float(r2_cci_hi), 4)],
                'cluster_bootstrap_rho_ci': [round(float(rho_cci_lo), 4), round(float(rho_cci_hi), 4)],
                'n_POOLED': len(probe_vals),
                'polarity_flip_applied': concept in FLIP_CONCEPTS,
                'corrected_stats': corrected_a,
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
    turnwise_data = {}
    turnwise_rho_data = {}
    for short in SHORTHANDS:
        concept = SHORTHAND_TO_CONCEPT[short]
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
        rho_vals = [tw_stats[str(t)].get('spearman_rho', np.nan) for t in turns]
        rho_lo = [tw_stats[str(t)].get('spearman_rho_ci_low', np.nan) for t in turns]
        rho_hi = [tw_stats[str(t)].get('spearman_rho_ci_high', np.nan) for t in turns]
        rho_p = [tw_stats[str(t)].get('spearman_p', np.nan) for t in turns]
        flip_sign = -1 if concept in FLIP_CONCEPTS else 1
        turnwise_rho_data[short] = {
            'turns': turns,
            'spearman_rho': [v * flip_sign for v in rho_vals],
            'spearman_rho_ci': list(zip(
                [v * flip_sign for v in (rho_hi if flip_sign == -1 else rho_lo)],
                [v * flip_sign for v in (rho_lo if flip_sign == -1 else rho_hi)],
            )),
            'spearman_p': rho_p,
            'rho_sign_flipped': concept in FLIP_CONCEPTS,
        }

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    for short in SHORTHANDS:
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        if short not in turnwise_data:
            continue
        turns = turnwise_data[short]['turns']
        r2_vals = turnwise_data[short]['isotonic_r2']
        r2_ci = turnwise_data[short]['isotonic_r2_ci']
        plot_line_with_ci(
            ax,
            turns,
            r2_vals,
            [ci[0] for ci in r2_ci],
            [ci[1] for ci in r2_ci],
            color=color,
            label=SHORTHAND_DISPLAY[short],
            alpha_fill=0.08,
        )
    ax.set_xlabel('Turn')
    ax.set_ylabel('Isotonic RÂ²')
    ax.set_ylabel('Isotonic R2')
    ax.set_title('Turn-wise Introspection Accuracy')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc='best')
    add_panel_label(ax, 'E')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_E_turnwise_r2_combined')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_E',
        'title': 'Turn-wise Isotonic RÂ² (Combined)',
        'description': ('All four turn-wise isotonic RÂ² curves shown in a single axis with '
                        'lighter confidence bands to facilitate cross-concept comparison.'),
        'turnwise_r2': {k: v for k, v in turnwise_data.items()},
    })

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    for short in SHORTHANDS:
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        if short not in turnwise_rho_data:
            continue
        turns = turnwise_rho_data[short]['turns']
        rho_vals = turnwise_rho_data[short]['spearman_rho']
        rho_ci = turnwise_rho_data[short]['spearman_rho_ci']
        plot_line_with_ci(
            ax,
            turns,
            rho_vals,
            [ci[0] for ci in rho_ci],
            [ci[1] for ci in rho_ci],
            color=color,
            label=SHORTHAND_DISPLAY[short],
            alpha_fill=0.08,
        )
    ax.set_xlabel('Turn')
    ax.set_ylabel('Spearman Ï')
    ax.set_ylabel('Spearman rho')
    ax.set_title('Turn-wise Introspection Correlation')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(-0.3, 1)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.legend(fontsize=7, loc='best')
    add_panel_label(ax, 'F')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_F_turnwise_rho_combined')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_F',
        'title': 'Turn-wise Spearman Ï (Combined)',
        'description': ('All four turn-wise Spearman Ï curves shown in a single axis with '
                        'lighter confidence bands to facilitate cross-concept comparison.'),
        'turnwise_rho': turnwise_rho_data,
    })

    fig, axes = plt.subplots(1, 2, figsize=(5, 4))
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
                conv_ids_r = sub_r['conversation_index'].values
                _, r2_r_lo, r2_r_hi = cluster_bootstrap_stat(
                    conv_ids_r, probe_r, rating_r, isotonic_r2)
                _, rho_r_lo, rho_r_hi = cluster_bootstrap_stat(
                    conv_ids_r, probe_r, rating_r, spearman_rho)
                random_stats[short] = {
                    'isotonic_r2': round(float(r2_r), 4),
                    'spearman_rho': round(float(rho_r), 4),
                    'n': len(sub_r),
                    'cluster_bootstrap_r2_ci': [
                        round(float(r2_r_lo), 4), round(float(r2_r_hi), 4)
                    ],
                    'cluster_bootstrap_rho_ci': [
                        round(float(rho_r_lo), 4), round(float(rho_r_hi), 4)
                    ],
                }

    paired_true_vs_random = {}
    for short in SHORTHANDS:
        concept = SHORTHAND_TO_CONCEPT[short]
        df_true = _load_df(short)
        df_rand = _load_df(short, LLAMA_3B_RANDOM)
        if df_true is None or df_rand is None:
            continue

        sub_true = df_true[np.isclose(df_true['alpha'], 0.0)].dropna(
            subset=['probe_score', 'logit_rating']).copy()
        sub_rand = df_rand[np.isclose(df_rand['alpha'], 0.0)].dropna(
            subset=['probe_score', 'logit_rating']).copy()
        sub_true['probe_display'] = flip_if_needed(concept, sub_true['probe_score'].values)
        sub_rand['probe_display'] = flip_if_needed(concept, sub_rand['probe_score'].values)

        paired_true_vs_random[short] = {
            'isotonic_r2': {
                **_cluster_bootstrap_metric_delta(
                    sub_true, sub_rand, 'probe_display', 'logit_rating', isotonic_r2),
            },
            'spearman_rho': {
                **_cluster_bootstrap_metric_delta(
                    sub_true, sub_rand, 'probe_display', 'logit_rating', spearman_rho),
            },
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
            ci = s['cluster_bootstrap_r2_ci']
            ax.errorbar(i + offset_probe, s['isotonic_r2'],
                        yerr=[[s['isotonic_r2'] - ci[0]], [ci[1] - s['isotonic_r2']]],
                        color='black', capsize=3, linewidth=1.2, capthick=1.2)
        if has_random and short in random_stats:
            ax.bar(i + offset_random, random_stats[short]['isotonic_r2'], bar_w,
                   color='#CCCCCC', edgecolor='white', linewidth=0.5,
                   label='Random' if i == 0 else None)
            ci = random_stats[short]['cluster_bootstrap_r2_ci']
            ax.errorbar(i + offset_random, random_stats[short]['isotonic_r2'],
                        yerr=[[random_stats[short]['isotonic_r2'] - ci[0]],
                              [ci[1] - random_stats[short]['isotonic_r2']]],
                        color='black', capsize=3, linewidth=1.2, capthick=1.2)
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
            ax.bar(i + offset_probe, s['spearman_rho_POOLED'], bar_w,
                   color=color, edgecolor='white', linewidth=0.5,
                   label='Probe' if i == 0 else None)
            ci = s['cluster_bootstrap_rho_ci']
            ax.errorbar(i + offset_probe, s['spearman_rho_POOLED'],
                        yerr=[[s['spearman_rho_POOLED'] - ci[0]], [ci[1] - s['spearman_rho_POOLED']]],
                        color='black', capsize=3, linewidth=1.2, capthick=1.2)
        if has_random and short in random_stats:
            ax.bar(i + offset_random, random_stats[short]['spearman_rho'], bar_w,
                   color='#CCCCCC', edgecolor='white', linewidth=0.5,
                   label='Random' if i == 0 else None)
            ci = random_stats[short]['cluster_bootstrap_rho_ci']
            ax.errorbar(i + offset_random, random_stats[short]['spearman_rho'],
                        yerr=[[random_stats[short]['spearman_rho'] - ci[0]],
                              [ci[1] - random_stats[short]['spearman_rho']]],
                        color='black', capsize=3, linewidth=1.2, capthick=1.2)
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
        'title': 'Introspection Metrics - Bar Charts (with Random Control)',
        'description': ('Isotonic R² (left) and Spearman ρ (right) for each concept at alpha=0. '
                        'Grey bars: random-direction control. 95% cluster-bootstrap CIs are '
                        'shown for both true-probe and random-control bars, and the JSON stores '
                        'cluster-bootstrap pooled-delta tests for true minus random.'),
        'probe_statistics': scatter_stats,
        'random_control_statistics': random_stats,
        'paired_true_vs_random_per_conversation': paired_true_vs_random,
    })

    # ──────────────────────────────────────────────────────────────
    # Panels C (4 stacked): Turn-wise R² per concept
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(2, 12), sharex=True)
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
    fig, axes = plt.subplots(4, 1, figsize=(2, 12), sharex=True)
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
        n_obs = len(first)  # = number of conversations

        # R² for first and last + bootstrap CIs
        r2_first = isotonic_r2(first['probe_display'].values, first['logit_rating'].values)
        r2_last = isotonic_r2(last_['probe_display'].values, last_['logit_rating'].values)
        _, r2_first_lo, r2_first_hi = bootstrap_stat(
            first['probe_display'].values, first['logit_rating'].values, isotonic_r2)
        _, r2_last_lo, r2_last_hi = bootstrap_stat(
            last_['probe_display'].values, last_['logit_rating'].values, isotonic_r2)

        # Rho for first and last + bootstrap CIs
        rho_first, p_first = stats.spearmanr(
            first['probe_display'].values, first['logit_rating'].values)
        rho_last, p_last = stats.spearmanr(
            last_['probe_display'].values, last_['logit_rating'].values)
        _, rho_first_lo, rho_first_hi = bootstrap_stat(
            first['probe_display'].values, first['logit_rating'].values, spearman_rho)
        _, rho_last_lo, rho_last_hi = bootstrap_stat(
            last_['probe_display'].values, last_['logit_rating'].values, spearman_rho)

        # Statistical comparison: paired permutation test for R² and rho difference
        # Per-conversation R² and rho at first and last turn
        r2_diffs = []
        rho_diffs = []
        for ci_val in sorted(sub['conversation_index'].unique()):
            cv = sub[sub['conversation_index'] == ci_val]
            cv_first = cv[cv['turn'] == first_turn]
            cv_last = cv[cv['turn'] == last_turn]
            if len(cv_first) >= 1 and len(cv_last) >= 1:
                # Can't compute per-conversation R²/rho with n=1, use paired bootstrap instead
                pass
        # Bootstrap test: is R²(last) - R²(first) > 0?
        rng_cdii = np.random.RandomState(42)
        n_boot = 2000
        r2_diff_boots = []
        rho_diff_boots = []
        first_probe = first['probe_display'].values
        first_rating = first['logit_rating'].values
        last_probe = last_['probe_display'].values
        last_rating = last_['logit_rating'].values
        for _ in range(n_boot):
            idx_f = rng_cdii.choice(len(first_probe), len(first_probe), replace=True)
            idx_l = rng_cdii.choice(len(last_probe), len(last_probe), replace=True)
            r2_f_b = isotonic_r2(first_probe[idx_f], first_rating[idx_f])
            r2_l_b = isotonic_r2(last_probe[idx_l], last_rating[idx_l])
            r2_diff_boots.append(r2_l_b - r2_f_b)
            rho_f_b = spearman_rho(first_probe[idx_f], first_rating[idx_f])
            rho_l_b = spearman_rho(last_probe[idx_l], last_rating[idx_l])
            rho_diff_boots.append(rho_l_b - rho_f_b)
        r2_diff_boots = np.array(r2_diff_boots)
        rho_diff_boots = np.array(rho_diff_boots)
        r2_diff_p = float(np.mean(r2_diff_boots <= 0)) * 2  # two-sided
        r2_diff_p = min(r2_diff_p, 1.0)
        rho_diff_p = float(np.mean(rho_diff_boots <= 0)) * 2
        rho_diff_p = min(rho_diff_p, 1.0)

        width = 0.35
        # R² bars with error bars
        axes[0].bar(i - width/2, r2_first, width, color=color, alpha=0.5,
                    label=f'Turn {first_turn}' if i == 0 else None)
        axes[0].bar(i + width/2, r2_last, width, color=color,
                    label=f'Turn {last_turn}' if i == 0 else None)
        axes[0].errorbar(i - width/2, r2_first,
                         yerr=[[r2_first - r2_first_lo], [r2_first_hi - r2_first]],
                         color='black', capsize=3, linewidth=1, capthick=1)
        axes[0].errorbar(i + width/2, r2_last,
                         yerr=[[r2_last - r2_last_lo], [r2_last_hi - r2_last]],
                         color='black', capsize=3, linewidth=1, capthick=1)
        # Rho bars with error bars
        axes[1].bar(i - width/2, rho_first, width, color=color, alpha=0.5,
                    label=f'Turn {first_turn}' if i == 0 else None)
        axes[1].bar(i + width/2, rho_last, width, color=color,
                    label=f'Turn {last_turn}' if i == 0 else None)
        axes[1].errorbar(i - width/2, rho_first,
                         yerr=[[rho_first - rho_first_lo], [rho_first_hi - rho_first]],
                         color='black', capsize=3, linewidth=1, capthick=1)
        axes[1].errorbar(i + width/2, rho_last,
                         yerr=[[rho_last - rho_last_lo], [rho_last_hi - rho_last]],
                         color='black', capsize=3, linewidth=1, capthick=1)

        first_last_stats[short] = {
            'first_turn': int(first_turn),
            'last_turn': int(last_turn),
            'n_first': n_obs,
            'n_last': len(last_),
            'polarity_flip_applied': concept in FLIP_CONCEPTS,
            'r2_first': round(float(r2_first), 4),
            'r2_last': round(float(r2_last), 4),
            'r2_first_ci': [round(float(r2_first_lo), 4), round(float(r2_first_hi), 4)],
            'r2_last_ci': [round(float(r2_last_lo), 4), round(float(r2_last_hi), 4)],
            'r2_difference': round(float(r2_last - r2_first), 4),
            'r2_difference_bootstrap_p': round(float(r2_diff_p), 4),
            'r2_difference_ci': [round(float(np.percentile(r2_diff_boots, 2.5)), 4),
                                 round(float(np.percentile(r2_diff_boots, 97.5)), 4)],
            'rho_first': round(float(rho_first), 4),
            'rho_last': round(float(rho_last), 4),
            'rho_first_ci': [round(float(rho_first_lo), 4), round(float(rho_first_hi), 4)],
            'rho_last_ci': [round(float(rho_last_lo), 4), round(float(rho_last_hi), 4)],
            'rho_difference': round(float(rho_last - rho_first), 4),
            'rho_difference_bootstrap_p': round(float(rho_diff_p), 4),
            'rho_difference_ci': [round(float(np.percentile(rho_diff_boots, 2.5)), 4),
                                  round(float(np.percentile(rho_diff_boots, 97.5)), 4)],
            'rho_first_p': float(p_first),
            'rho_last_p': float(p_last),
        }

    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels([SHORTHAND_DISPLAY[s] for s in SHORTHANDS], fontsize=8, rotation=15)
    axes[0].set_ylabel('Isotonic R²')
    axes[0].set_title('First vs Last Turn: R²')
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    add_panel_label(axes[0], 'A')

    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels([SHORTHAND_DISPLAY[s] for s in SHORTHANDS], fontsize=8, rotation=15)
    axes[1].set_ylabel('Spearman ρ (polarity-corrected)')
    axes[1].set_title('First vs Last Turn: ρ')
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    prefix = os.path.join(appendix_dir, 'App_Fig_03_A_first_vs_last_turn')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'App_Fig_03_A',
        'title': 'First vs Last Turn Introspection Comparison',
        'description': ('Comparison of introspection metrics at the first and last conversation '
                        'turn with 95% bootstrap CIs and bootstrap significance test for '
                        'the difference. Rho sign-flipped for FLIP concepts.'),
        'first_last_stats': first_last_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel CDiii: Within-conversation introspection change over turns
    #   For each conversation, compute Spearman ρ(turn, concordance) and
    #   slope(turn, concordance) where concordance = z(probe) * z(rating).
    #   Also: LMM interaction test: rating ~ probe * turn + (1|conv).
    # ──────────────────────────────────────────────────────────────
    import statsmodels.formula.api as smf
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cdiii_stats = {}

    for i, short in enumerate(SHORTHANDS):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]

        df = _load_df(short)
        if df is None:
            continue
        sub = df[np.isclose(df['alpha'], 0.0)].dropna(
            subset=['probe_score', 'logit_rating']).copy()
        sub['probe_display'] = flip_if_needed(concept, sub['probe_score'].values)

        # --- Method 1: Per-conversation Spearman of concordance vs turn ---
        # For each conversation, z-score probe and rating, compute product,
        # then correlate with turn.
        conv_rhos_concordance = []
        conv_rhos_residual = []
        for ci_val in sorted(sub['conversation_index'].unique()):
            cv = sub[sub['conversation_index'] == ci_val].sort_values('turn')
            if len(cv) < 4:
                continue
            p_vals = cv['probe_display'].values
            r_vals = cv['logit_rating'].values
            t_vals = cv['turn'].values

            # z-score within conversation
            p_z = (p_vals - p_vals.mean()) / (p_vals.std() + 1e-12)
            r_z = (r_vals - r_vals.mean()) / (r_vals.std() + 1e-12)
            concordance = p_z * r_z  # positive = agreement

            rho_c, _ = stats.spearmanr(t_vals, concordance)
            if np.isfinite(rho_c):
                conv_rhos_concordance.append(rho_c)

            # Also: Spearman of |z_probe - z_rating| vs turn (residual approach)
            abs_resid = np.abs(p_z - r_z)
            rho_r, _ = stats.spearmanr(t_vals, abs_resid)
            if np.isfinite(rho_r):
                conv_rhos_residual.append(rho_r)

        conv_rhos_concordance = np.array(conv_rhos_concordance)
        conv_rhos_residual = np.array(conv_rhos_residual)

        # t-test of concordance rhos against 0
        concordance_ttest = one_sample_test(conv_rhos_concordance)
        # t-test of residual rhos against 0 (negative = improving)
        residual_ttest = one_sample_test(conv_rhos_residual)

        # --- Method 2: LMM interaction test ---
        # logit_rating ~ probe_display * turn + (1|conversation_index)
        lmm_interaction = {}
        try:
            sub_lmm = sub.copy()
            # Center turn for interpretability
            sub_lmm['turn_c'] = sub_lmm['turn'] - sub_lmm['turn'].mean()
            sub_lmm['probe_c'] = sub_lmm['probe_display'] - sub_lmm['probe_display'].mean()
            md = smf.mixedlm(
                'logit_rating ~ probe_c * turn_c',
                sub_lmm, groups=sub_lmm['conversation_index'])
            res = md.fit(reml=True, method='lbfgs')
            # The interaction term probe_c:turn_c tests if the relationship
            # between probe and rating changes with turn
            interaction_key = 'probe_c:turn_c'
            lmm_interaction = {
                'interaction_coef': round(float(res.fe_params.get(interaction_key, np.nan)), 6),
                'interaction_p': round(float(res.pvalues.get(interaction_key, np.nan)), 6),
                'probe_coef': round(float(res.fe_params.get('probe_c', np.nan)), 4),
                'probe_p': float(res.pvalues.get('probe_c', np.nan)),
                'turn_coef': round(float(res.fe_params.get('turn_c', np.nan)), 6),
                'turn_p': float(res.pvalues.get('turn_c', np.nan)),
                'n_obs': len(sub_lmm),
                'n_groups': int(sub_lmm['conversation_index'].nunique()),
                'converged': res.converged,
                'interpretation': ('Positive interaction means probe-rating relationship '
                                   'strengthens over turns; negative means it weakens.'),
            }
        except Exception as e:
            lmm_interaction = {'error': str(e)}

        # --- Plot: distribution of per-conversation concordance rhos ---
        ax.hist(conv_rhos_concordance, bins=15, color=color, alpha=0.6,
                edgecolor='white', linewidth=0.5)
        mean_rho = float(np.mean(conv_rhos_concordance))
        ax.axvline(mean_rho, color='black', linewidth=2, linestyle='--')
        ax.axvline(0, color='gray', linewidth=1, linestyle=':')
        p_str = format_p(concordance_ttest.get('t_p', 1.0))
        ax.set_title(f'{SHORTHAND_DISPLAY[short]}\nμ={mean_rho:.3f}, {p_str}',
                     fontsize=9)
        ax.set_xlabel('Within-conv ρ(turn, concordance)')
        if i == 0:
            ax.set_ylabel('Number of conversations')

        cdiii_stats[short] = {
            'n_conversations': len(conv_rhos_concordance),
            'polarity_flip_applied': concept in FLIP_CONCEPTS,
            'concordance_rho_ttest': concordance_ttest,
            'concordance_rho_mean': round(float(np.mean(conv_rhos_concordance)), 4),
            'concordance_rho_std': round(float(np.std(conv_rhos_concordance)), 4),
            'concordance_rho_values': [round(float(v), 4) for v in conv_rhos_concordance],
            'residual_rho_ttest': residual_ttest,
            'residual_rho_mean': round(float(np.mean(conv_rhos_residual)), 4),
            'lmm_interaction_test': lmm_interaction,
        }

    add_panel_label(axes[0], 'B', x=-0.18)
    fig.suptitle('Within-Conversation Introspection Change Over Turns', fontsize=12, y=1.02)
    plt.tight_layout()
    prefix = os.path.join(appendix_dir, 'App_Fig_03_B_within_conv_trend')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'App_Fig_03_B',
        'title': 'Within-Conversation Introspection Change Over Turns',
        'description': (
            'Tests whether introspection quality changes monotonically across turns '
            'within each conversation (α=0). Three methods: '
            '(1) Per-conversation Spearman ρ(turn, concordance) where concordance = '
            'z(probe) × z(rating), then one-sample t-test of 40 ρ values against 0. '
            '(2) Per-conversation Spearman ρ(turn, |z_probe − z_rating|) as a residual '
            'measure (negative = improving agreement). '
            '(3) LMM: logit_rating ~ probe × turn + (1|conversation), where the '
            'interaction term tests whether the probe-rating relationship changes '
            'with turn.'
        ),
        'cdiii_stats': cdiii_stats,
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
        valid_alphas = []
        alphas_found = sorted(df['alpha'].unique())
        for a in alphas_found:
            sub = df[np.isclose(df['alpha'], a)].dropna(
                subset=['conversation_index', 'logit_rating']).copy()
            grouped = compute_grouped_cluster_means(sub, 'alpha', 'logit_rating')
            if grouped.empty:
                continue
            row = grouped.iloc[0]
            valid_alphas.append(float(a))
            alpha_means.append(float(row['mean']))
            alpha_ci_lo.append(float(row['ci_low']))
            alpha_ci_hi.append(float(row['ci_high']))

        # Flip alpha for display if needed
        display_alphas = flip_alpha_if_needed(short, np.array(valid_alphas))
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

        # Corrected steering stats (3 methods)
        df_steer = df.copy()
        df_steer['display_alpha'] = flip_alpha_if_needed(short, df['alpha'].values)
        corrected_steer = corrected_steering_stats(
            df_steer, 'display_alpha', 'logit_rating')

        steering_stats[short] = {
            'display_alphas': [float(a) for a in display_alphas],
            'mean_ratings': [round(m, 4) for m in alpha_means],
            'ci_low': [round(c, 4) for c in alpha_ci_lo],
            'ci_high': [round(c, 4) for c in alpha_ci_hi],
            'alpha_display_flipped': short in FLIP_SHORTHANDS,
            'spearman_rho_display_alpha_vs_rating_POOLED': round(float(rho_a), 4),
            'spearman_p_POOLED': float(p_a),
            'corrected_steering': corrected_steer,
        }

    ax.set_xlabel('Steering α')
    ax.set_ylabel('Mean Logit Self-Report')
    ax.set_title('Self-Steering: Self-Report vs. Alpha')
    ax.legend(fontsize=8)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    add_panel_label(ax, 'G')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_G_self_steering_mean_report')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_G',
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
    panel_labels = ['C', 'D', 'E', 'F']
    for pi, short in enumerate(SHORTHANDS):
        concept = SHORTHAND_TO_CONCEPT[short]
        df = _load_df(short)
        if df is None:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        alphas_found = sorted(df['alpha'].unique())
        drift_by_alpha = {}

        alpha_pairs = sorted(
            [(float(flip_alpha_scalar(short, a)), float(a)) for a in alphas_found],
            key=lambda pair: pair[0],
        )
        for display_a, a in alpha_pairs:
            sub = df[np.isclose(df['alpha'], a)]
            per_turn = compute_grouped_cluster_means(
                sub.dropna(subset=['conversation_index', 'logit_rating']).copy(),
                'turn',
                'logit_rating',
            )
            if per_turn.empty:
                continue
            turns = per_turn['turn'].tolist()
            means = per_turn['mean'].tolist()
            ci_lo = per_turn['ci_low'].tolist()
            ci_hi = per_turn['ci_high'].tolist()

            c = ALPHA_COLORS.get(display_a, 'gray')
            plot_line_with_ci(ax, turns, means, ci_lo, ci_hi,
                              color=c, label=f'α = {display_a:+.0f}', alpha_fill=0.12)

            # Drift statistic: Spearman of turn vs rating at this alpha
            sub_nona = sub.dropna(subset=['logit_rating'])
            rho_d, p_d = stats.spearmanr(sub_nona['turn'], sub_nona['logit_rating'])
            # Corrected drift (per-conv slope + LMM)
            corrected_fi = corrected_drift_stats(sub, 'logit_rating', alpha_val=a)
            drift_by_alpha[str(a)] = {
                'display_alpha': float(display_a),
                'means': [round(m, 4) for m in means],
                'ci_low': [round(c, 4) for c in ci_lo],
                'ci_high': [round(c, 4) for c in ci_hi],
                'drift_spearman_rho_POOLED': round(float(rho_d), 4),
                'drift_spearman_p_POOLED': float(p_d),
                'drift_magnitude': round(float(means[-1] - means[0]), 4),
                'corrected_drift': corrected_fi,
            }

        ax.set_xlabel('Turn')
        ax.set_ylabel('Logit Self-Report')
        ax.set_title(f'Self-Report Drift — {SHORTHAND_DISPLAY[short]}')
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=7, loc='best')
        add_panel_label(ax, panel_labels[pi])
        plt.tight_layout()
        prefix = os.path.join(appendix_dir,
                              f'App_Fig_03_{panel_labels[pi]}_steering_drift_{concept}')
        savefig(fig, prefix)
        save_panel_json(prefix, {
            'panel_id': f'App_Fig_03_{panel_labels[pi]}',
            'title': f'Self-Report Drift Under Steering - {SHORTHAND_DISPLAY[short]}',
            'description': (f'Mean logit self-report across turns for different steering '
                            f'alphas (display-corrected for polarity). Drift Spearman ρ '
                            f'indicates temporal trend at each alpha level.'),
            'concept': concept,
            'alpha_display_flipped': short in FLIP_SHORTHANDS,
            'drift_by_alpha': drift_by_alpha,
        })

    # Panel J: Drift magnitude vs alpha
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    alpha_drift_stats = {}
    for short in SHORTHANDS:
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        df = _load_df(short)
        if df is None:
            continue

        drift_rows = []
        for a in sorted(df['alpha'].unique()):
            sub = df[np.isclose(df['alpha'], a)].copy()
            for conv_id in sorted(sub['conversation_index'].unique()):
                cv = sub[sub['conversation_index'] == conv_id].sort_values('turn')
                values = cv['logit_rating'].dropna()
                if len(values) >= 2:
                    drift_rows.append({
                        'conversation_index': int(conv_id),
                        'alpha': float(a),
                        'display_alpha': float(flip_alpha_scalar(short, a)),
                        'drift': float(values.iloc[-1] - values.iloc[0]),
                    })
        if not drift_rows:
            continue

        drift_df = pd.DataFrame(drift_rows)
        drift_summary = compute_grouped_cluster_means(drift_df, 'display_alpha', 'drift')
        if drift_summary.empty:
            continue
        corrected_alpha_drift = corrected_steering_stats(
            drift_df, 'display_alpha', 'drift')

        plot_line_with_ci(
            ax,
            drift_summary['display_alpha'].tolist(),
            drift_summary['mean'].tolist(),
            drift_summary['ci_low'].tolist(),
            drift_summary['ci_high'].tolist(),
            color=color,
            label=SHORTHAND_DISPLAY[short],
        )

        alpha_drift_stats[short] = {
            'display_alphas': [float(v) for v in drift_summary['display_alpha'].tolist()],
            'mean_drifts': [round(float(v), 4) for v in drift_summary['mean'].tolist()],
            'ci_low': [round(float(v), 4) for v in drift_summary['ci_low'].tolist()],
            'ci_high': [round(float(v), 4) for v in drift_summary['ci_high'].tolist()],
            'n_conversations_per_alpha': {
                str(float(alpha)): int(n)
                for alpha, n in drift_df.groupby('display_alpha')['conversation_index'].nunique().items()
            },
            'corrected_alpha_drift': corrected_alpha_drift,
        }

    ax.set_xlabel('Steering α')
    ax.set_ylabel('Drift (last - first turn)')
    ax.set_title('Self-Report Drift Magnitude vs. Alpha')
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=8)
    add_panel_label(ax, 'H')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_03_H_drift_magnitude_vs_alpha')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_03_H',
        'title': 'Self-Report Drift Magnitude vs. Alpha',
        'description': ('Per-conversation self-report drift (last - first turn) summarized '
                        'by steering alpha, with cluster-bootstrap 95% CIs and three '
                        'trend tests stored in the corrected statistics.'),
        'alpha_drift_stats': alpha_drift_stats,
    })

    # ── Save other_stats ──
    main_other_stats = {
        'description': ('Figure 3 main panels establish introspection with scatter plots, '
                        'random controls, turn-wise analyses in both stacked and combined '
                        'formats, self-steering response curves, and drift-vs-alpha.'),
        'overall_introspection': scatter_stats,
        'random_control': random_stats,
        'paired_true_vs_random': paired_true_vs_random,
        'turnwise_r2': {s: turnwise_data.get(s, {}) for s in SHORTHANDS},
        'turnwise_rho': {s: turnwise_rho_data.get(s, {}) for s in SHORTHANDS},
        'steering_stats': steering_stats,
        'alpha_drift_stats': alpha_drift_stats,
    }
    appendix_other_stats = {
        'description': ('Appendix Figure 3 keeps first-vs-last-turn tests, within-conversation '
                        'trend analyses, and the four steering-drift panels.'),
        'first_vs_last_turn': first_last_stats,
        'within_conversation_trend': cdiii_stats,
        'steering_drift_panels': {
            label: SHORTHANDS[idx] for idx, label in enumerate(['C', 'D', 'E', 'F'])
        },
    }
    save_other_stats(fig_dir, main_other_stats)
    save_other_stats(appendix_dir, appendix_other_stats)
    print("    Figure 3 complete.")


if __name__ == '__main__':
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rdir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    generate_figure_3(rdir)
    print(f"Output -> {rdir}")
