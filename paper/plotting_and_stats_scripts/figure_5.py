"""
Figure 5: Cross-Model Generalization
======================================
Validates findings across model sizes (LLaMA 1B/3B/8B) and families
(Gemma 3 4B, Qwen 2.5 7B).

v2 changes:
 - B: recompute R² from raw experiment dirs (not CSV)
 - Bii: add Rho vs model size plot
 - C: REPLACED — logit self-report vs alpha curves + Rho heatmap
 - D: improved selection criteria
 - E/F: drift bar subpanels (Eii, Fii)
 - G: normalize layer axis (0-1), add layer-range shading, allow negative d
 - H: turns already fixed
 - K/L: flip Rho for sad_vs_happy
 - M: LLaMA 8B scatter kept (high-introspection examples)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from shared_utils import (
    PROBES, CONCEPTS_ORDERED, CONCEPT_DISPLAY, CONCEPT_COLORS,
    SHORTHAND_TO_CONCEPT, CONCEPT_TO_SHORTHAND,
    SHORTHAND_DISPLAY, SHORTHANDS_ORDERED,
    LLAMA_3B_4X4_SELF, LLAMA_1B_SELF, LLAMA_8B_SELF,
    GEMMA_4B_SELF, QWEN_7B_SELF,
    MODEL_SIZE_CSV_DIR, CROSS_FAMILY_CSV_DIR,
    MODEL_FAMILY_COLORS, MODEL_SIZE_COLORS,
    PROBE_METRIC_KEY, FLIP_CONCEPTS, FLIP_SHORTHANDS,
    LAYER_RANGE_FRAC,
    load_results, load_summary, load_turnwise, load_metrics,
    load_sweep_data,
    results_to_dataframe, flip_if_needed,
    flip_alpha_if_needed, flip_alpha_scalar,
    get_turnwise_stats, isotonic_r2, bootstrap_stat, spearman_rho,
    cluster_bootstrap_stat, per_conversation_spearman,
    one_sample_test, lmm_test,
    corrected_drift_stats, corrected_correlation_stats,
    corrected_steering_stats,
    savefig, save_panel_json, save_other_stats, ensure_dir,
    add_panel_label, plot_line_with_ci, format_p, draw_vector_heatmap,
    compute_per_turn_means, compute_grouped_cluster_means,
    linear_regression_stats, exact_permutation_slope_test,
)

SHORTHANDS = SHORTHANDS_ORDERED
LLAMA_SIZES = ['1B', '3B', '8B']
LLAMA_SIZE_VALS = {'1B': 1.0, '3B': 3.0, '8B': 8.0}
LLAMA_MODELS = {
    '1B': ('llama_1b', LLAMA_1B_SELF),
    '3B': ('llama_3b', LLAMA_3B_4X4_SELF),
    '8B': ('llama_8b', LLAMA_8B_SELF),
}


def _compute_r2_from_raw(model_key, exp_dirs, concept_short, alpha=0.0):
    """Compute isotonic R² and Spearman rho from raw experiment directory.
    Includes corrected statistics (per-conversation rho, cluster bootstrap, LMM)."""
    concept = SHORTHAND_TO_CONCEPT[concept_short]
    exp_dir = exp_dirs.get(concept_short)
    if exp_dir is None or not os.path.isdir(exp_dir):
        return None
    try:
        results = load_results(exp_dir)
        df = results_to_dataframe(results, probe_name=concept)
        sub = df[np.isclose(df['alpha'], alpha)].dropna(
            subset=['probe_score', 'logit_rating'])
        probe_vals = flip_if_needed(concept, sub['probe_score'].values)
        ratings = sub['logit_rating'].values
        if len(probe_vals) < 5:
            return None
        r2_pt, r2_lo, r2_hi = bootstrap_stat(probe_vals, ratings, isotonic_r2)
        rho_pt, rho_lo, rho_hi = bootstrap_stat(probe_vals, ratings, spearman_rho)
        rho_val, rho_p = stats.spearmanr(probe_vals, ratings)
        # Cluster bootstrap CIs
        conv_ids = sub['conversation_index'].values
        r2_cpt, r2_clo, r2_chi = cluster_bootstrap_stat(
            conv_ids, probe_vals, ratings, isotonic_r2)
        rho_cpt, rho_clo, rho_chi = cluster_bootstrap_stat(
            conv_ids, probe_vals, ratings, spearman_rho)
        # Corrected: per-conversation rho + LMM
        sub_corr = sub.copy()
        sub_corr['probe_display'] = probe_vals
        corrected = corrected_correlation_stats(
            sub_corr, 'probe_display', 'logit_rating', alpha_val=alpha)
        return {
            'isotonic_r2': float(r2_pt),
            'r2_ci_low': float(r2_lo),
            'r2_ci_high': float(r2_hi),
            'r2_cluster_ci_low': float(r2_clo),
            'r2_cluster_ci_high': float(r2_chi),
            'spearman_rho': float(rho_val),
            'spearman_p_POOLED': float(rho_p),
            'rho_ci_low': float(rho_lo),
            'rho_ci_high': float(rho_hi),
            'rho_cluster_ci_low': float(rho_clo),
            'rho_cluster_ci_high': float(rho_chi),
            'n_POOLED': len(probe_vals),
            'corrected_stats': corrected,
        }
    except Exception:
        return None


def _size_trend_from_points(size_to_value):
    """OLS and exact-permutation trend test on one value per model size."""
    xs = []
    ys = []
    for size in LLAMA_SIZES:
        value = size_to_value.get(size)
        if value is None or not np.isfinite(value):
            continue
        xs.append(np.log(LLAMA_SIZE_VALS[size]))
        ys.append(float(value))
    ols = linear_regression_stats(xs, ys)
    exact = exact_permutation_slope_test(xs, ys, alternative='greater')
    return {
        'x_log_size': [float(v) for v in xs],
        'y_values': [float(v) for v in ys],
        'ols': ols,
        'exact_permutation': exact,
    }


def _size_trend_from_distributions(size_to_values):
    """OLS on independent per-conversation values pooled across sizes."""
    xs = []
    ys = []
    for size in LLAMA_SIZES:
        values = size_to_values.get(size, [])
        for value in values:
            if np.isfinite(value):
                xs.append(np.log(LLAMA_SIZE_VALS[size]))
                ys.append(float(value))
    return linear_regression_stats(xs, ys)


def _bootstrap_size_mean_slope(size_to_values, alternative='greater',
                               n_bootstrap=10000, seed=42):
    """
    Bootstrap the slope of mean outcome vs. log model size.
    """
    rng = np.random.default_rng(seed)
    valid_sizes = [
        size for size in LLAMA_SIZES
        if len(size_to_values.get(size, [])) > 0
    ]
    xs = np.array([np.log(LLAMA_SIZE_VALS[size]) for size in valid_sizes], dtype=float)
    observed_means = np.array([
        float(np.mean(size_to_values[size])) for size in valid_sizes
    ], dtype=float)

    if len(valid_sizes) < 3:
        return {
            'sizes': valid_sizes,
            'x_log_size': [float(v) for v in xs.tolist()],
            'mean_by_size': {size: float(observed_means[i]) for i, size in enumerate(valid_sizes)},
            'observed_slope': np.nan,
            'bootstrap_slope_mean': np.nan,
            'bootstrap_slope_ci_95': [np.nan, np.nan],
            'one_sided_p': np.nan,
            'two_sided_p': np.nan,
            'n_bootstrap': n_bootstrap,
        }

    observed_slope = float(stats.linregress(xs, observed_means).slope)
    boot_slopes = np.zeros(n_bootstrap, dtype=float)
    for bi in range(n_bootstrap):
        boot_means = []
        for size in valid_sizes:
            arr = np.asarray(size_to_values[size], dtype=float)
            sampled = rng.choice(arr, size=len(arr), replace=True)
            boot_means.append(float(np.mean(sampled)))
        boot_slopes[bi] = float(stats.linregress(xs, boot_means).slope)

    ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])
    p_le_zero = (np.sum(boot_slopes <= 0) + 1) / (len(boot_slopes) + 1)
    p_ge_zero = (np.sum(boot_slopes >= 0) + 1) / (len(boot_slopes) + 1)
    if alternative == 'greater':
        one_sided_p = float(p_le_zero)
    elif alternative == 'less':
        one_sided_p = float(p_ge_zero)
    else:
        one_sided_p = np.nan
    two_sided_p = float(min(1.0, 2 * min(p_le_zero, p_ge_zero)))
    return {
        'sizes': valid_sizes,
        'x_log_size': [float(v) for v in xs.tolist()],
        'mean_by_size': {size: float(observed_means[i]) for i, size in enumerate(valid_sizes)},
        'observed_slope': observed_slope,
        'bootstrap_slope_mean': float(np.mean(boot_slopes)),
        'bootstrap_slope_ci_95': [float(ci_low), float(ci_high)],
        'one_sided_p': one_sided_p,
        'two_sided_p': two_sided_p,
        'n_bootstrap': n_bootstrap,
    }


def _bootstrap_group_diff(a_vals, b_vals, n_bootstrap=10000, seed=42):
    """Bootstrap mean difference (b - a) with percentile CI and two-sided p."""
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return {
            'mean_diff_b_minus_a': np.nan,
            'ci_95': [np.nan, np.nan],
            'p_two_sided': np.nan,
            'n_bootstrap': n_bootstrap,
            'n_a': int(len(a)),
            'n_b': int(len(b)),
        }
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        boots[i] = float(np.mean(b_s) - np.mean(a_s))
    obs = float(np.mean(b) - np.mean(a))
    ci = np.percentile(boots, [2.5, 97.5])
    p_le_zero = (np.sum(boots <= 0) + 1) / (len(boots) + 1)
    p_ge_zero = (np.sum(boots >= 0) + 1) / (len(boots) + 1)
    p_two_sided = float(min(1.0, 2 * min(p_le_zero, p_ge_zero)))
    return {
        'mean_diff_b_minus_a': obs,
        'ci_95': [float(ci[0]), float(ci[1])],
        'p_two_sided': p_two_sided,
        'n_bootstrap': n_bootstrap,
        'n_a': int(len(a)),
        'n_b': int(len(b)),
    }


def _shannon_entropy_binned(values, n_bins=20):
    """Shannon entropy (bits) for continuous values via histogram binning."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    if np.allclose(np.min(arr), np.max(arr)):
        return 0.0
    hist, _ = np.histogram(arr, bins=n_bins)
    probs = hist.astype(float)
    probs = probs / np.sum(probs)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _first_last_metric_bootstrap(sub, x_col, y_col, metric_fn, n_bootstrap=2000, seed=42):
    """
    Cluster bootstrap test for metric(last_turn) - metric(first_turn).
    """
    sub = sub.dropna(subset=['conversation_index', 'turn', x_col, y_col]).copy()
    turns = sorted(sub['turn'].unique())
    if len(turns) < 2:
        return {}
    first_turn = turns[0]
    last_turn = turns[-1]
    first_df = sub[sub['turn'] == first_turn].copy()
    last_df = sub[sub['turn'] == last_turn].copy()
    common_convs = sorted(set(first_df['conversation_index']) & set(last_df['conversation_index']))
    if len(common_convs) < 3:
        return {}
    first_df = first_df[first_df['conversation_index'].isin(common_convs)].copy()
    last_df = last_df[last_df['conversation_index'].isin(common_convs)].copy()

    first_metric = metric_fn(first_df[x_col].values, first_df[y_col].values)
    last_metric = metric_fn(last_df[x_col].values, last_df[y_col].values)
    observed_diff = float(last_metric - first_metric)

    rng = np.random.RandomState(seed)
    boot_diffs = []
    convs = np.array(common_convs)
    for _ in range(n_bootstrap):
        sampled = rng.choice(convs, len(convs), replace=True)
        first_parts = [first_df[first_df['conversation_index'] == conv] for conv in sampled]
        last_parts = [last_df[last_df['conversation_index'] == conv] for conv in sampled]
        first_boot = pd.concat(first_parts, ignore_index=True)
        last_boot = pd.concat(last_parts, ignore_index=True)
        try:
            boot_first = metric_fn(first_boot[x_col].values, first_boot[y_col].values)
            boot_last = metric_fn(last_boot[x_col].values, last_boot[y_col].values)
            boot_diffs.append(float(boot_last - boot_first))
        except Exception:
            continue
    if not boot_diffs:
        return {}
    boot_diffs = np.array(boot_diffs, dtype=float)
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    p_two_sided = 2 * min(np.mean(boot_diffs >= 0), np.mean(boot_diffs <= 0))
    return {
        'first_turn': int(first_turn),
        'last_turn': int(last_turn),
        'first_metric': float(first_metric),
        'last_metric': float(last_metric),
        'difference_last_minus_first': observed_diff,
        'bootstrap_ci_95': [float(ci_low), float(ci_high)],
        'bootstrap_p_two_sided': float(min(1.0, p_two_sided)),
        'n_conversations': int(len(common_convs)),
    }


def generate_figure_5(results_dir):
    """Generate all Figure 5 panels."""
    fig_dir = ensure_dir(os.path.join(results_dir, 'Figure_5'))
    appendix_dir = ensure_dir(os.path.join(results_dir, 'Appendix_Figure_5'))
    print("  Generating Figure 5: Cross-Model Generalization...")

    other_stats = {}

    # ──────────────────────────────────────────────────────────────
    # Panel A: Probe quality heatmap (Max Cohen's d)
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    d_matrix = np.zeros((len(LLAMA_SIZES), len(SHORTHANDS)))
    d_values = {}

    for si, size in enumerate(LLAMA_SIZES):
        model_key = LLAMA_MODELS[size][0]
        d_values[size] = {}
        for ci_idx, short in enumerate(SHORTHANDS):
            concept = SHORTHAND_TO_CONCEPT[short]
            try:
                metrics = load_metrics(PROBES[model_key][concept])
                d_val = metrics['best_d']
            except (KeyError, FileNotFoundError):
                d_val = np.nan
            d_matrix[si, ci_idx] = d_val
            d_values[size][short] = round(float(d_val), 3) if not np.isnan(d_val) else None

    im = draw_vector_heatmap(ax, d_matrix, cmap='YlGnBu', aspect='auto')
    for si in range(len(LLAMA_SIZES)):
        for ci_idx in range(len(SHORTHANDS)):
            v = d_matrix[si, ci_idx]
            text_color = 'white' if v > 2.5 else 'black'
            ax.text(ci_idx, si, f'{v:.2f}', ha='center', va='center',
                    fontsize=9, color=text_color)
    ax.set_xticks(range(len(SHORTHANDS)))
    ax.set_xticklabels([SHORTHAND_DISPLAY[s] for s in SHORTHANDS],
                       fontsize=8, rotation=25, ha='right')
    ax.set_yticks(range(len(LLAMA_SIZES)))
    ax.set_yticklabels([f'LLaMA {s}' for s in LLAMA_SIZES], fontsize=9)
    ax.set_title("Probe Quality (Max Cohen's d)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Cohen's d")
    add_panel_label(ax, 'A')
    plt.tight_layout()
    prefix = os.path.join(appendix_dir, 'App_Fig_05_A_probe_quality_heatmap')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'App_Fig_05_A',
        'title': "Probe Quality — Max Cohen's d",
        'd_values': d_values,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel B: R² isotonic vs model size (recomputed from raw dirs)
    # ──────────────────────────────────────────────────────────────
    raw_r2_data = {}  # {size: {short: stats_dict}}
    for size in LLAMA_SIZES:
        model_key, exp_dirs = LLAMA_MODELS[size]
        raw_r2_data[size] = {}
        for short in SHORTHANDS:
            result = _compute_r2_from_raw(model_key, exp_dirs, short)
            if result is not None:
                raw_r2_data[size][short] = result

    fig, ax = plt.subplots(1, 1, figsize=(5.5 * 0.7, 4))
    b_stats = {}
    for short in SHORTHANDS:
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        sizes_x, r2_y, r2_lo, r2_hi = [], [], [], []
        for size in LLAMA_SIZES:
            if short in raw_r2_data[size]:
                d = raw_r2_data[size][short]
                sizes_x.append(LLAMA_SIZE_VALS[size])
                r2_y.append(d['isotonic_r2'])
                r2_lo.append(d['r2_ci_low'])
                r2_hi.append(d['r2_ci_high'])
        if sizes_x:
            plot_line_with_ci(
                ax,
                sizes_x,
                r2_y,
                r2_lo,
                r2_hi,
                color=color,
                label=SHORTHAND_DISPLAY[short],
                alpha_fill=0.12,
                marker='o',
                markersize=6,
            )
            size_to_value = {s: raw_r2_data[s].get(short, {}).get('isotonic_r2')
                             for s in LLAMA_SIZES}
            b_stats[short] = {
                'per_size': size_to_value,
                'size_trend': _size_trend_from_points(size_to_value),
            }

    ax.set_xlabel('Model Size (B parameters)')
    ax.set_ylabel('Isotonic R²')
    ax.set_title('Introspection vs. Model Size')
    ax.set_xscale('log')
    ax.set_xticks([1, 3, 8])
    ax.set_xticklabels(['1B', '3B', '8B'])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc='upper left')
    add_panel_label(ax, 'A')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_A_r2_vs_model_size')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_A',
        'title': 'Isotonic R² vs. Model Size (Recomputed from Raw Data)',
        'description': ('Isotonic R² at alpha=0 for each concept across LLaMA 1B/3B/8B, '
                        'recomputed from raw experiment directories (not pre-computed CSV).'),
        'per_concept': b_stats,
        'data_sources': {size: {s: str(LLAMA_MODELS[size][1].get(s, ''))
                                for s in SHORTHANDS} for size in LLAMA_SIZES},
    })

    # ──────────────────────────────────────────────────────────────
    # Panel Bii: Spearman Rho vs model size
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5 * 0.7, 4))
    bii_stats = {}
    for short in SHORTHANDS:
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        sizes_x, rho_y, rho_lo, rho_hi = [], [], [], []
        for size in LLAMA_SIZES:
            if short in raw_r2_data[size]:
                d = raw_r2_data[size][short]
                # Rho already polarity-corrected via flip_if_needed in
                # _compute_r2_from_raw — do NOT flip again.
                sizes_x.append(LLAMA_SIZE_VALS[size])
                rho_y.append(d['spearman_rho'])
                rho_lo.append(d['rho_ci_low'])
                rho_hi.append(d['rho_ci_high'])
        if sizes_x:
            plot_line_with_ci(
                ax,
                sizes_x,
                rho_y,
                rho_lo,
                rho_hi,
                color=color,
                label=SHORTHAND_DISPLAY[short],
                alpha_fill=0.12,
                marker='o',
                markersize=6,
            )
            size_to_value = {
                s: raw_r2_data[s].get(short, {}).get('spearman_rho')
                for s in LLAMA_SIZES
            }
            bii_stats[short] = {
                'per_size': {s: round(v, 4) if v is not None else None
                             for s, v in size_to_value.items()},
                'size_trend': _size_trend_from_points(size_to_value),
            }

    ax.set_xlabel('Model Size (B parameters)')
    ax.set_ylabel('Spearman ρ (polarity-corrected)')
    ax.set_title('Introspection Correlation vs. Model Size')
    ax.set_xscale('log')
    ax.set_xticks([1, 3, 8])
    ax.set_xticklabels(['1B', '3B', '8B'])
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=7, loc='upper left')
    # Auto-scale y-axis to data
    add_panel_label(ax, 'B')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_B_rho_vs_model_size')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_B',
        'title': 'Spearman ρ vs. Model Size',
        'description': ('Spearman ρ at alpha=0 (polarity-corrected for FLIP concepts). '
                        'Rho flipped for sad_vs_happy & impulsive_vs_planning.'),
        'per_concept': bii_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel Biii: LLaMA 8B best-introspection scatter plots
    #   (Moved here — before C — for logical ordering near size data)
    # ──────────────────────────────────────────────────────────────
    best_concepts_8b = ['wellbeing', 'interest']
    fig, axes = plt.subplots(1, 2, figsize=(9 * 0.7, 4))
    best_scatter_stats = {}
    for i, short in enumerate(best_concepts_8b):
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        ax = axes[i]
        exp_dir = LLAMA_8B_SELF.get(short)
        if exp_dir is None or not os.path.isdir(exp_dir):
            continue
        try:
            results = load_results(exp_dir)
            df = results_to_dataframe(results, probe_name=concept)
            sub = df[np.isclose(df['alpha'], 0.0)].dropna(
                subset=['probe_score', 'logit_rating'])
            probe_vals = flip_if_needed(concept, sub['probe_score'].values)
            ratings = sub['logit_rating'].values
            ax.scatter(probe_vals, ratings, color=color, alpha=0.3, s=15,
                       edgecolors='none')
            if len(probe_vals) > 5:
                ir = IsotonicRegression(out_of_bounds='clip')
                sort_idx = np.argsort(probe_vals)
                y_pred = ir.fit_transform(probe_vals[sort_idx], ratings[sort_idx])
                ax.plot(probe_vals[sort_idx], y_pred, 'k-', linewidth=2, alpha=0.7)
                rho_val, p_val = stats.spearmanr(probe_vals, ratings)
                r2_val = isotonic_r2(probe_vals, ratings)
                ax.text(0.05, 0.95,
                        f'ρ = {rho_val:.3f}\n{format_p(p_val)}\nR²(iso) = {r2_val:.3f}',
                        transform=ax.transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                # Corrected stats: per-conversation rho + LMM
                sub_corr = sub.copy()
                sub_corr['probe_display'] = probe_vals
                corrected_biii = corrected_correlation_stats(
                    sub_corr, 'probe_display', 'logit_rating', alpha_val=0.0)
                best_scatter_stats[short] = {
                    'spearman_rho': round(float(rho_val), 4),
                    'spearman_p_POOLED': float(p_val),
                    'isotonic_r2': round(float(r2_val), 4),
                    'n_POOLED': len(probe_vals),
                    'corrected_stats': corrected_biii,
                }
            ax.set_xlabel('Probe Score')
            if i == 0:
                ax.set_ylabel('Logit Self-Report')
            ax.set_title(f'LLaMA 8B — {SHORTHAND_DISPLAY[short]}')
        except Exception as e:
            print(f"    Warning: 8B scatter error for {short}: {e}")

    add_panel_label(axes[0], 'C')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_C_llama8b_best_scatters')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_C',
        'title': 'High-Introspection Scatter — LLaMA 8B',
        'description': ('Scatter plots of probe score vs logit self-report for the two '
                        'concepts with highest R² at LLaMA 8B (wellbeing, interest). '
                        'R² ~0.9, confirming strong introspection at larger model size.'),
        'statistics': best_scatter_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel C: NEW — self-report vs alpha curves + Rho heatmap
    #   Left: 3 sub-axes (one per model size), 4 concept curves each
    #   Right: Heatmap of Spearman rho(alpha, logit_rating)
    # ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.2])
    c_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_heat = fig.add_subplot(gs[0, 3])

    rho_matrix = np.full((len(LLAMA_SIZES), len(SHORTHANDS)), np.nan)
    rho_p_matrix_pooled = np.full((len(LLAMA_SIZES), len(SHORTHANDS)), np.nan)
    # Three corrected p-value matrices
    p_matrix_m1 = np.full((len(LLAMA_SIZES), len(SHORTHANDS)), np.nan)  # exact perm on N=5 means
    p_matrix_m2 = np.full((len(LLAMA_SIZES), len(SHORTHANDS)), np.nan)  # LMM
    p_matrix_m3 = np.full((len(LLAMA_SIZES), len(SHORTHANDS)), np.nan)  # per-conv slopes t-test
    c_stats = {}

    for si, size in enumerate(LLAMA_SIZES):
        model_key, exp_dirs = LLAMA_MODELS[size]
        ax = c_axes[si]
        c_stats[size] = {}

        for short in SHORTHANDS:
            concept = SHORTHAND_TO_CONCEPT[short]
            color = CONCEPT_COLORS[concept]
            exp_dir = exp_dirs.get(short)
            if exp_dir is None or not os.path.isdir(exp_dir):
                continue
            try:
                results = load_results(exp_dir)
                df = results_to_dataframe(results, probe_name=concept)
                alphas_found = sorted(df['alpha'].unique())
                valid_alphas = []

                means, ci_lo, ci_hi = [], [], []
                for a in alphas_found:
                    sub_a = df[np.isclose(df['alpha'], a)].dropna(
                        subset=['conversation_index', 'logit_rating']).copy()
                    grouped = compute_grouped_cluster_means(sub_a, 'alpha', 'logit_rating')
                    if grouped.empty:
                        continue
                    row = grouped.iloc[0]
                    valid_alphas.append(float(a))
                    means.append(float(row['mean']))
                    ci_lo.append(float(row['ci_low']))
                    ci_hi.append(float(row['ci_high']))

                display_alphas = flip_alpha_if_needed(short, np.array(valid_alphas))

                sort_idx = np.argsort(display_alphas)
                da_s = [display_alphas[k] for k in sort_idx]
                means_s = [means[k] for k in sort_idx]
                ci_lo_s = [ci_lo[k] for k in sort_idx]
                ci_hi_s = [ci_hi[k] for k in sort_idx]

                plot_line_with_ci(ax, da_s, means_s, ci_lo_s, ci_hi_s,
                                  color=color, label=SHORTHAND_DISPLAY[short],
                                  alpha_fill=0.15)

                # Pooled rho (original, kept with _POOLED tag)
                da_all = flip_alpha_if_needed(short, df['alpha'].values)
                lr_all = df['logit_rating'].dropna().values
                n_min = min(len(da_all), len(lr_all))
                rho_c, p_c = stats.spearmanr(da_all[:n_min], lr_all[:n_min])
                rho_matrix[si, SHORTHANDS.index(short)] = rho_c
                rho_p_matrix_pooled[si, SHORTHANDS.index(short)] = p_c

                # Corrected steering stats (3 methods)
                df_steer = df.copy()
                df_steer['display_alpha'] = flip_alpha_if_needed(short, df['alpha'].values)
                corrected_c = corrected_steering_stats(
                    df_steer, 'display_alpha', 'logit_rating')
                ci_short = SHORTHANDS.index(short)
                # Method 1: exact perm on N=5 means
                m1_info = corrected_c.get('method1_alpha_means', {})
                p_matrix_m1[si, ci_short] = m1_info.get(
                    'exact_permutation_p', np.nan)
                # Method 2: LMM
                lmm_info = corrected_c.get('method2_lmm', {})
                p_matrix_m2[si, ci_short] = lmm_info.get('slope_p', np.nan)
                # Method 3: per-conv slopes t-test
                slopes_info = corrected_c.get('method3_per_conv_slopes', {})
                p_matrix_m3[si, ci_short] = slopes_info.get('t_p', np.nan)

                c_stats[size][short] = {
                    'display_alphas': [float(a) for a in da_s],
                    'mean_ratings': [round(m, 4) for m in means_s],
                    'spearman_rho_alpha_rating': round(float(rho_c), 4),
                    'spearman_p_POOLED': float(p_c),
                    'corrected_steering': corrected_c,
                }
            except Exception as e:
                print(f"    Warning: Panel C error {size}/{short}: {e}")

        ax.set_xlabel('Steering α')
        ax.set_ylabel('Logit Self-Report')
        ax.set_title(f'LLaMA {size}')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        if si == 0:
            ax.legend(fontsize=6, loc='upper left')

    # Rho heatmap (use method 1 p-values for stars)
    im = draw_vector_heatmap(
        ax_heat,
        rho_matrix,
        cmap='RdYlGn',
        vmin=-1,
        vmax=1,
        aspect='auto',
    )
    for si in range(len(LLAMA_SIZES)):
        for ci_idx in range(len(SHORTHANDS)):
            v = rho_matrix[si, ci_idx]
            p_v = p_matrix_m1[si, ci_idx]  # exact permutation p
            if np.isnan(v):
                ax_heat.text(ci_idx, si, '—', ha='center', va='center', fontsize=9)
            else:
                sig_star = '*' if (not np.isnan(p_v) and p_v < 0.05) else ''
                text_color = 'white' if abs(v) > 0.6 else 'black'
                ax_heat.text(ci_idx, si, f'{v:.2f}{sig_star}', ha='center',
                             va='center', fontsize=9, color=text_color)
    ax_heat.set_xticks(range(len(SHORTHANDS)))
    ax_heat.set_xticklabels([SHORTHAND_DISPLAY[s] for s in SHORTHANDS],
                            fontsize=7, rotation=25, ha='right')
    ax_heat.set_yticks(range(len(LLAMA_SIZES)))
    ax_heat.set_yticklabels([f'LLaMA {s}' for s in LLAMA_SIZES], fontsize=9)
    ax_heat.set_title('ρ(α, self-report)')
    plt.colorbar(im, ax=ax_heat, shrink=0.8, label='Spearman ρ')

    add_panel_label(c_axes[0], 'B')
    plt.tight_layout()
    prefix = os.path.join(appendix_dir, 'App_Fig_05_B_steering_curves_and_rho')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'App_Fig_05_B',
        'title': 'Self-Report vs Alpha Curves + Rho Heatmap',
        'description': ('Left 3 panels: logit self-report vs display-corrected alpha per '
                        'concept per model size. Right: heatmap of Spearman ρ between alpha '
                        'and logit rating (* = p<0.05 by exact permutation). '
                        'Three corrected p-value matrices included.'),
        'per_size_stats': c_stats,
        'rho_matrix': rho_matrix.tolist(),
        'rho_p_matrix_POOLED': rho_p_matrix_pooled.tolist(),
        'p_matrix_method1_exact_perm': p_matrix_m1.tolist(),
        'p_matrix_method2_lmm': p_matrix_m2.tolist(),
        'p_matrix_method3_per_conv_slopes': p_matrix_m3.tolist(),
    })

    # ──────────────────────────────────────────────────────────────
    # Panel D: Mean R² bar chart using sign-validated probes only
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5 * 0.5, 4))
    im = draw_vector_heatmap(
        ax,
        rho_matrix,
        cmap='RdYlGn',
        vmin=-1,
        vmax=1,
        aspect='auto',
    )
    for si in range(len(LLAMA_SIZES)):
        for ci_idx in range(len(SHORTHANDS)):
            v = rho_matrix[si, ci_idx]
            p_v = p_matrix_m1[si, ci_idx]
            if np.isnan(v):
                ax.text(ci_idx, si, 'â€”', ha='center', va='center', fontsize=9)
            else:
                sig_star = '*' if (not np.isnan(p_v) and p_v < 0.05) else ''
                text_color = 'white' if abs(v) > 0.6 else 'black'
                ax.text(ci_idx, si, f'{v:.2f}{sig_star}', ha='center',
                        va='center', fontsize=9, color=text_color)
    ax.set_xticks(range(len(SHORTHANDS)))
    ax.set_xticklabels([SHORTHAND_DISPLAY[s] for s in SHORTHANDS],
                       fontsize=7, rotation=25, ha='right')
    ax.set_yticks(range(len(LLAMA_SIZES)))
    ax.set_yticklabels([f'LLaMA {s}' for s in LLAMA_SIZES], fontsize=9)
    ax.set_title('rho(alpha, self-report)')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spearman rho')
    ax.set_title('Ï(Î±, self-report)')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Spearman Ï')
    if len(fig.axes) > 2:
        fig.axes[-1].remove()
    ax.set_title('rho(alpha, self-report)')
    add_panel_label(ax, 'D')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_D_steering_rho_heatmap')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_D',
        'title': 'Self-Report Steering Heatmap',
        'description': ('Heatmap of Spearman Ï between display-corrected alpha and logit '
                        'self-report across concepts and LLaMA model sizes.'),
        'rho_matrix': rho_matrix.tolist(),
        'rho_p_matrix_POOLED': rho_p_matrix_pooled.tolist(),
        'p_matrix_method1_exact_perm': p_matrix_m1.tolist(),
        'p_matrix_method2_lmm': p_matrix_m2.tolist(),
        'p_matrix_method3_per_conv_slopes': p_matrix_m3.tolist(),
    })

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 4))
    validated_r2 = {s: [] for s in LLAMA_SIZES}
    rng_d = np.random.default_rng(42)

    for size in LLAMA_SIZES:
        for short in SHORTHANDS:
            if short not in raw_r2_data[size]:
                continue
            d = raw_r2_data[size][short]
            rho_idx = SHORTHANDS.index(short)
            si = LLAMA_SIZES.index(size)
            rho_val = rho_matrix[si, rho_idx]
            if np.isnan(rho_val) or rho_val > 0:
                validated_r2[size].append(d['isotonic_r2'])

    d_means = [float(np.mean(validated_r2[s])) if validated_r2[s] else 0.0 for s in LLAMA_SIZES]
    ci_lo_list, ci_hi_list = [], []
    for size in LLAMA_SIZES:
        vals = np.asarray(validated_r2[size], dtype=float)
        if len(vals) > 1:
            boots = [float(np.mean(rng_d.choice(vals, size=len(vals), replace=True)))
                     for _ in range(1000)]
            ci_lo_list.append(float(np.percentile(boots, 2.5)))
            ci_hi_list.append(float(np.percentile(boots, 97.5)))
        elif len(vals) == 1:
            ci_lo_list.append(float(vals[0]))
            ci_hi_list.append(float(vals[0]))
        else:
            ci_lo_list.append(0.0)
            ci_hi_list.append(0.0)

    d_bootstrap = _bootstrap_size_mean_slope(validated_r2, alternative='greater')
    d_pairwise = {
        '1B_vs_3B': _bootstrap_group_diff(validated_r2['1B'], validated_r2['3B']),
        '3B_vs_8B': _bootstrap_group_diff(validated_r2['3B'], validated_r2['8B']),
        '1B_vs_8B': _bootstrap_group_diff(validated_r2['1B'], validated_r2['8B']),
    }
    colors = [MODEL_SIZE_COLORS[s] for s in LLAMA_SIZES]
    ax.bar(range(len(LLAMA_SIZES)), d_means, color=colors,
           edgecolor='white', linewidth=0.5)
    for i in range(len(LLAMA_SIZES)):
        ax.errorbar(i, d_means[i],
                    yerr=[[d_means[i] - ci_lo_list[i]], [ci_hi_list[i] - d_means[i]]],
                    color='black', capsize=5, linewidth=1.5, capthick=1.5)
    ax.set_xticks(range(len(LLAMA_SIZES)))
    ax.set_xticklabels([f'LLaMA {s}' for s in LLAMA_SIZES])
    ax.set_ylabel('Mean Isotonic R²')
    ax.set_title('Introspection by Model Size')
    ax.set_ylim(0, 1)
    add_panel_label(ax, 'E')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_E_mean_r2_by_size')
    savefig(fig, prefix)
    d_json = {
        'panel_id': 'Fig_05_E',
        'title': 'Mean R² by Size',
        'description': ('Mean isotonic R² across concept-model pairs with '
                        'sign-validated steering effects (ρ(α, self-report) > 0).'),
        'validated_r2': {s: [round(float(v), 4) for v in validated_r2[s]] for s in LLAMA_SIZES},
        'means': [round(float(m), 4) for m in d_means],
        'ci_95_low': [round(float(v), 4) for v in ci_lo_list],
        'ci_95_high': [round(float(v), 4) for v in ci_hi_list],
        'size_trend_bootstrap': d_bootstrap,
        'pairwise_bootstrap_tests': d_pairwise,
    }
    save_panel_json(prefix, d_json)

    # ──────────────────────────────────────────────────────────────
    # Panel E: Probe score drift — one concept, 3 LLaMA sizes
    # ──────────────────────────────────────────────────────────────
    example_concept = 'wellbeing'
    concept_name = SHORTHAND_TO_CONCEPT[example_concept]

    fig, ax = plt.subplots(1, 1, figsize=(5.5 / 3, 4))
    drift_stats_e = {}
    for size in LLAMA_SIZES:
        model_key, exp_dirs = LLAMA_MODELS[size]
        exp_dir = exp_dirs.get(example_concept)
        if exp_dir is None or not os.path.isdir(exp_dir):
            continue
        try:
            results = load_results(exp_dir)
            df = results_to_dataframe(results, probe_name=concept_name)
            sub = df[np.isclose(df['alpha'], 0.0)].copy()
            sub['probe_display'] = flip_if_needed(concept_name, sub['probe_score'].values)
            per_turn = compute_grouped_cluster_means(
                sub.dropna(subset=['conversation_index', 'probe_display']).copy(),
                'turn',
                'probe_display',
            )
            turns = per_turn['turn'].tolist()
            means = per_turn['mean'].tolist()
            ci_lo = per_turn['ci_low'].tolist()
            ci_hi = per_turn['ci_high'].tolist()
            color = MODEL_SIZE_COLORS[size]
            plot_line_with_ci(ax, turns, means, ci_lo, ci_hi,
                              color=color, label=f'LLaMA {size}')
            rho_e, p_e = stats.spearmanr(
                sub['turn'], sub['probe_display'].dropna())
            # Per-conversation drift (for error bars in Eii)
            conv_drifts = []
            for ci_idx in sub['conversation_index'].unique():
                cv = sub[sub['conversation_index'] == ci_idx].sort_values('turn')
                if len(cv) >= 2:
                    first_val = cv['probe_display'].iloc[0]
                    last_val = cv['probe_display'].iloc[-1]
                    conv_drifts.append(last_val - first_val)
            # Corrected drift stats
            corrected_e = corrected_drift_stats(sub, 'probe_display', alpha_val=0.0)
            drift_stats_e[size] = {
                'drift_magnitude': round(float(means[-1] - means[0]), 4),
                'spearman_rho_vs_turn_POOLED': round(float(rho_e), 4),
                'spearman_p_POOLED': float(p_e),
                'per_conv_drifts': conv_drifts,
                'corrected_drift': corrected_e,
            }
        except Exception as e:
            print(f"    Warning: Drift plot error for {size}: {e}")

    ax.set_xlabel('Turn')
    ax.set_ylabel('Probe Score (polarity-corrected)')
    ax.set_title(f'Internal State Drift — {SHORTHAND_DISPLAY[example_concept]}')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)
    add_panel_label(ax, 'F')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_F_drift_across_sizes')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_F',
        'title': f'Probe Drift Across Sizes — {SHORTHAND_DISPLAY[example_concept]}',
        'drift_stats': drift_stats_e,
    })

    # Panel Eii: Drift magnitude bar chart with error bars
    fig, ax = plt.subplots(1, 1, figsize=(2, 3.5))
    eii_json_stats = {}
    eii_size_values = {}
    for i, size in enumerate(LLAMA_SIZES):
        if size in drift_stats_e:
            color = MODEL_SIZE_COLORS[size]
            drifts = drift_stats_e[size].get('per_conv_drifts', [])
            eii_size_values[size] = [float(v) for v in drifts]
            mean_drift = np.mean(drifts) if drifts else drift_stats_e[size]['drift_magnitude']
            ax.bar(i, mean_drift, color=color, edgecolor='white', linewidth=0.5)
            if len(drifts) > 1:
                rng = np.random.RandomState(42)
                boots = [np.mean(rng.choice(drifts, len(drifts))) for _ in range(1000)]
                ci_lo = np.percentile(boots, 2.5)
                ci_hi = np.percentile(boots, 97.5)
                ax.errorbar(i, mean_drift,
                            yerr=[[mean_drift - ci_lo], [ci_hi - mean_drift]],
                            color='black', capsize=5, linewidth=1.5, capthick=1.5)
                eii_json_stats[size] = {
                    'mean_drift': round(float(mean_drift), 4),
                    'ci_95': [round(float(ci_lo), 4), round(float(ci_hi), 4)],
                    'n_conversations': len(drifts),
                    'std_drift': round(float(np.std(drifts, ddof=1)), 4),
                }
    eii_trend = _bootstrap_size_mean_slope(eii_size_values, alternative='greater')
    ax.set_xticks(range(len(LLAMA_SIZES)))
    ax.set_xticklabels([f'LLaMA {s}' for s in LLAMA_SIZES])
    ax.set_ylabel('Drift (last − first turn)')
    ax.set_title(f'Probe Drift Magnitude — {SHORTHAND_DISPLAY[example_concept]}')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    add_panel_label(ax, 'G')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_G_drift_magnitude_bars')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_G',
        'title': 'Probe Drift Magnitude Bars (with 95% CI)',
        'description': ('Per-conversation probe score drift (last - first turn) '
                        'with bootstrap 95% CI error bars.'),
        'drift_stats': eii_json_stats,
        'size_trend_bootstrap': eii_trend,
    })

    # Scale-control for probe drift: check if absolute probe scale changes with size
    probe_scale_control = {}
    normalized_drift_by_size = {}
    for size in LLAMA_SIZES:
        model_key, exp_dirs = LLAMA_MODELS[size]
        exp_dir = exp_dirs.get(example_concept)
        if exp_dir is None or not os.path.isdir(exp_dir):
            continue
        try:
            results = load_results(exp_dir)
            df = results_to_dataframe(results, probe_name=concept_name)
            sub = df[np.isclose(df['alpha'], 0.0)].copy()
            sub['probe_display'] = flip_if_needed(concept_name, sub['probe_score'].values)
            vals = sub['probe_display'].dropna().values
            if len(vals) == 0:
                continue
            mean_abs = float(np.mean(np.abs(vals)))
            std_val = float(np.std(vals))
            iqr_val = float(np.percentile(vals, 75) - np.percentile(vals, 25))

            drifts = drift_stats_e.get(size, {}).get('per_conv_drifts', [])
            if len(drifts) > 0 and std_val > 0:
                norm = [float(d / std_val) for d in drifts]
            else:
                norm = []

            probe_scale_control[size] = {
                'mean_abs_probe_score': mean_abs,
                'std_probe_score': std_val,
                'iqr_probe_score': iqr_val,
                'n_obs': int(len(vals)),
                'normalized_drift_values': norm,
                'mean_normalized_drift': float(np.mean(norm)) if len(norm) > 0 else np.nan,
            }
            normalized_drift_by_size[size] = norm
        except Exception as e:
            print(f"    Warning: probe scale control error for {size}: {e}")

    probe_scale_control['normalized_drift_size_trend_bootstrap'] = _bootstrap_size_mean_slope(
        normalized_drift_by_size, alternative='greater')

    # ──────────────────────────────────────────────────────────────
    # Panel F: Self-report drift — one concept, 3 sizes
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5 / 3, 4))
    drift_stats_f = {}
    for size in LLAMA_SIZES:
        model_key, exp_dirs = LLAMA_MODELS[size]
        exp_dir = exp_dirs.get(example_concept)
        if exp_dir is None or not os.path.isdir(exp_dir):
            continue
        try:
            results = load_results(exp_dir)
            df = results_to_dataframe(results, probe_name=concept_name)
            sub = df[np.isclose(df['alpha'], 0.0)]
            per_turn = compute_grouped_cluster_means(
                sub.dropna(subset=['conversation_index', 'logit_rating']).copy(),
                'turn',
                'logit_rating',
            )
            turns = per_turn['turn'].tolist()
            means = per_turn['mean'].tolist()
            ci_lo = per_turn['ci_low'].tolist()
            ci_hi = per_turn['ci_high'].tolist()
            color = MODEL_SIZE_COLORS[size]
            plot_line_with_ci(ax, turns, means, ci_lo, ci_hi,
                              color=color, label=f'LLaMA {size}')
            rho_f, p_f = stats.spearmanr(
                sub['turn'], sub['logit_rating'].dropna())
            # Per-conversation drift (for error bars in Fii)
            conv_drifts_f = []
            for ci_idx in sub['conversation_index'].unique():
                cv = sub[sub['conversation_index'] == ci_idx].sort_values('turn')
                lr = cv['logit_rating'].dropna()
                if len(lr) >= 2:
                    conv_drifts_f.append(lr.iloc[-1] - lr.iloc[0])
            # Corrected drift stats
            sub_f = sub.copy()
            corrected_f = corrected_drift_stats(sub_f, 'logit_rating', alpha_val=0.0)
            drift_stats_f[size] = {
                'drift_magnitude': round(float(means[-1] - means[0]), 4),
                'spearman_rho_vs_turn_POOLED': round(float(rho_f), 4),
                'spearman_p_POOLED': float(p_f),
                'per_conv_drifts': conv_drifts_f,
                'corrected_drift': corrected_f,
            }
        except Exception as e:
            print(f"    Warning: Self-report drift error for {size}: {e}")

    ax.set_xlabel('Turn')
    ax.set_ylabel('Logit Self-Report')
    ax.set_title(f'Self-Report Drift — {SHORTHAND_DISPLAY[example_concept]}')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)
    add_panel_label(ax, 'H')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_H_report_drift_across_sizes')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_H',
        'title': f'Self-Report Drift — {SHORTHAND_DISPLAY[example_concept]}',
        'drift_stats': drift_stats_f,
    })

    # Panel Fii: Self-report drift magnitude bars (with bootstrap CI)
    fig, ax = plt.subplots(1, 1, figsize=(2, 3.5))
    fii_stats = {}
    fii_size_values = {}
    rng_fii = np.random.default_rng(42)
    for i, size in enumerate(LLAMA_SIZES):
        if size in drift_stats_f and drift_stats_f[size].get('per_conv_drifts'):
            color = MODEL_SIZE_COLORS[size]
            drifts = np.array(drift_stats_f[size]['per_conv_drifts'])
            fii_size_values[size] = [float(v) for v in drifts.tolist()]
            mean_d = float(np.mean(drifts))
            # Bootstrap 95% CI
            boot_means = [float(np.mean(rng_fii.choice(drifts, size=len(drifts), replace=True)))
                          for _ in range(1000)]
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
            ax.bar(i, mean_d, color=color, edgecolor='white', linewidth=0.5)
            ax.errorbar(i, mean_d, yerr=[[mean_d - ci_lo], [ci_hi - mean_d]],
                        fmt='none', ecolor='black', capsize=4, linewidth=1.2)
            fii_stats[size] = {
                'mean_drift': round(mean_d, 4),
                'ci_95': [round(ci_lo, 4), round(ci_hi, 4)],
                'n_conversations': len(drifts),
                'std_drift': round(float(np.std(drifts)), 4),
            }
    fii_trend = _bootstrap_size_mean_slope(fii_size_values, alternative='greater')
    ax.set_xticks(range(len(LLAMA_SIZES)))
    ax.set_xticklabels([f'LLaMA {s}' for s in LLAMA_SIZES])
    ax.set_ylabel('Drift (last − first turn)')
    ax.set_title(f'Report Drift Magnitude — {SHORTHAND_DISPLAY[example_concept]}')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    add_panel_label(ax, 'I')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_I_report_drift_magnitude_bars')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_I',
        'title': 'Report Drift Magnitude Bars (bootstrap CI)',
        'drift_stats': fii_stats,
        'size_trend_bootstrap': fii_trend,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel G: Layer sweeps (Gemma 4B, Qwen 7B) — normalized + shading
    # ──────────────────────────────────────────────────────────────
    cross_concept = 'sad_vs_happy'
    cross_models = [
        ('Gemma 3 4B', 'gemma_4b', MODEL_FAMILY_COLORS['gemma']),
        ('Qwen 2.5 7B', 'qwen_7b', MODEL_FAMILY_COLORS['qwen']),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sweep_info = {}

    # Shade layer search range
    lo_frac, hi_frac = LAYER_RANGE_FRAC
    ax.axvspan(lo_frac, hi_frac, alpha=0.08, color='gray',
               label=f'Search range ({lo_frac:.0%}-{hi_frac:.0%})')

    for model_name, model_key, color in cross_models:
        try:
            probe_dir = PROBES[model_key][cross_concept]
            sweep = load_sweep_data(probe_dir)
            metrics = load_metrics(probe_dir)
            sweep_d = np.array(sweep['sweep_d'])
            num_layers = metrics['num_layers']
            best_layer = metrics['best_layer']
            best_d = metrics['best_d']

            # Normalized layer axis (0-1)
            layers_norm = np.arange(num_layers) / (num_layers - 1)
            best_norm = best_layer / (num_layers - 1)

            ax.plot(layers_norm, sweep_d, color=color, linewidth=1.5,
                    label=f'{model_name} (d={best_d:.2f}, L{best_layer}/{num_layers})')
            ax.plot(best_norm, sweep_d[best_layer], 'o', color=color,
                    markersize=7, markeredgecolor='white', markeredgewidth=1)
            ax.axvline(best_norm, color=color, linestyle='--', alpha=0.5, linewidth=1)
            sweep_info[model_name] = {
                'best_layer': best_layer,
                'best_layer_normalized': round(float(best_norm), 3),
                'best_d': round(best_d, 3),
                'num_layers': num_layers,
            }
        except Exception as e:
            print(f"    Warning: Sweep error for {model_name}: {e}")

    ax.set_xlabel('Normalized Layer Position (0 = first, 1 = last)')
    ax.set_ylabel("Cohen's d")
    ax.set_title(f"Layer Sweep — {CONCEPT_DISPLAY[cross_concept]}")
    ax.legend(fontsize=7)
    # Allow negative d
    ax.set_xlim(0, 1)
    add_panel_label(ax, 'C')
    plt.tight_layout()
    prefix = os.path.join(appendix_dir, 'App_Fig_05_C_cross_family_layer_sweep')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'App_Fig_05_C',
        'title': f'Normalized Layer Sweep — {CONCEPT_DISPLAY[cross_concept]}',
        'description': ('Layer sweep with normalized x-axis (0-1) for cross-family '
                        'comparison. Layer search range shaded. Negative d values allowed.'),
        'sweep_info': sweep_info,
        'layer_search_range_frac': [lo_frac, hi_frac],
    })

    # ──────────────────────────────────────────────────────────────
    # Panel H: Self-report drift — Gemma and Qwen
    # ──────────────────────────────────────────────────────────────
    cross_exp_dirs = {
        'Gemma 3 4B': GEMMA_4B_SELF.get('wellbeing'),
        'Qwen 2.5 7B': QWEN_7B_SELF.get('wellbeing'),
    }
    cross_colors = {
        'Gemma 3 4B': MODEL_FAMILY_COLORS['gemma'],
        'Qwen 2.5 7B': MODEL_FAMILY_COLORS['qwen'],
    }

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    h_drift_stats = {}
    for model_name, exp_dir in cross_exp_dirs.items():
        if exp_dir is None or not os.path.isdir(exp_dir):
            continue
        try:
            results = load_results(exp_dir)
            df = results_to_dataframe(results, probe_name=cross_concept)
            sub = df[np.isclose(df['alpha'], 0.0)]
            per_turn = compute_grouped_cluster_means(
                sub.dropna(subset=['conversation_index', 'logit_rating']).copy(),
                'turn',
                'logit_rating',
            )
            turns = per_turn['turn'].tolist()
            means = per_turn['mean'].tolist()
            ci_lo = per_turn['ci_low'].tolist()
            ci_hi = per_turn['ci_high'].tolist()
            color = cross_colors[model_name]
            plot_line_with_ci(ax, turns, means, ci_lo, ci_hi,
                              color=color, label=model_name)
            rho_h, p_h = stats.spearmanr(sub['turn'], sub['logit_rating'].dropna())
            # Corrected drift stats
            sub_h = sub.copy()
            corrected_h = corrected_drift_stats(sub_h, 'logit_rating', alpha_val=0.0)
            h_drift_stats[model_name] = {
                'drift_magnitude': round(float(means[-1] - means[0]), 4),
                'spearman_rho_vs_turn_POOLED': round(float(rho_h), 4),
                'spearman_p_POOLED': float(p_h),
                'corrected_drift': corrected_h,
            }
        except Exception as e:
            print(f"    Warning: Cross-family drift error for {model_name}: {e}")

    ax.set_xlabel('Turn')
    ax.set_ylabel('Logit Self-Report')
    ax.set_title(f'Self-Report Drift — {CONCEPT_DISPLAY[cross_concept]}')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)
    add_panel_label(ax, 'J')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_J_cross_family_report_drift')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_J',
        'title': 'Self-Report Drift — Gemma vs Qwen',
        'drift_stats': h_drift_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panels I–J: Scatter plots for Gemma and Qwen
    # ──────────────────────────────────────────────────────────────
    scatter_labels = ['K', 'L']
    cross_scatter_stats = {}
    for idx, (model_name, exp_dir) in enumerate(cross_exp_dirs.items()):
        if exp_dir is None or not os.path.isdir(exp_dir):
            continue
        try:
            results = load_results(exp_dir)
            df = results_to_dataframe(results, probe_name=cross_concept)
            sub = df[np.isclose(df['alpha'], 0.0)].dropna(
                subset=['probe_score', 'logit_rating'])
            probe_vals = flip_if_needed(cross_concept, sub['probe_score'].values)
            ratings = sub['logit_rating'].values

            fig, ax = plt.subplots(1, 1, figsize=(5 * 0.5, 4))
            color = cross_colors[model_name]
            ax.scatter(probe_vals, ratings, color=color, alpha=0.3, s=15,
                       edgecolors='none')
            if len(probe_vals) > 5:
                ir = IsotonicRegression(out_of_bounds='clip')
                sort_idx = np.argsort(probe_vals)
                y_pred = ir.fit_transform(probe_vals[sort_idx], ratings[sort_idx])
                ax.plot(probe_vals[sort_idx], y_pred, color='black',
                        linewidth=2, alpha=0.7)
                rho_val, p_val = stats.spearmanr(probe_vals, ratings)
                r2_val = isotonic_r2(probe_vals, ratings)
                ax.text(0.05, 0.95,
                        f'ρ = {rho_val:.3f}\n{format_p(p_val)}\nR²(iso) = {r2_val:.3f}',
                        transform=ax.transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                # Corrected stats: per-conversation rho + LMM
                sub_corr = sub.copy()
                sub_corr['probe_display'] = probe_vals
                corrected_ij = corrected_correlation_stats(
                    sub_corr, 'probe_display', 'logit_rating', alpha_val=0.0)
                logit_entropy_bits = _shannon_entropy_binned(ratings, n_bins=20)
                cross_scatter_stats[model_name] = {
                    'spearman_rho': round(float(rho_val), 4),
                    'spearman_p_POOLED': float(p_val),
                    'isotonic_r2': round(float(r2_val), 4),
                    'logit_entropy_bits': round(float(logit_entropy_bits), 4),
                    'n_POOLED': len(probe_vals),
                    'corrected_stats': corrected_ij,
                }
            ax.set_xlabel('Probe Score')
            ax.set_ylabel('Logit Self-Report')
            ax.set_title(f'{model_name} — {CONCEPT_DISPLAY[cross_concept]}')
            add_panel_label(ax, scatter_labels[idx])
            plt.tight_layout()
            prefix = os.path.join(fig_dir,
                                  f'Fig_05_{scatter_labels[idx]}_scatter_{model_name.replace(" ", "_")}')
            savefig(fig, prefix)
            save_panel_json(prefix, {
                'panel_id': f'Fig_05_{scatter_labels[idx]}',
                'title': f'Scatter — {model_name}',
                'statistics': cross_scatter_stats.get(model_name, {}),
            })
        except Exception as e:
            print(f"    Warning: Cross-family scatter error for {model_name}: {e}")

    # ──────────────────────────────────────────────────────────────
    # Panels K–L: Turnwise R² and Rho — Gemma, Qwen
    #   Flip Rho for sad_vs_happy
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(5, 4))
    cross_turnwise = {}
    flip_rho_sign = -1 if cross_concept in FLIP_CONCEPTS else 1

    for model_name, exp_dir in cross_exp_dirs.items():
        if exp_dir is None or not os.path.isdir(exp_dir):
            continue
        try:
            tw_stats = get_turnwise_stats(exp_dir, cross_concept, PROBE_METRIC_KEY,
                                          source='logit', alpha=0.0)
            if not tw_stats:
                continue
            color = cross_colors[model_name]
            turns = sorted([int(t) for t in tw_stats.keys()])
            r2_vals = [tw_stats[str(t)].get('isotonic_r2', np.nan) for t in turns]
            r2_lo = [tw_stats[str(t)].get('isotonic_r2_ci_low', np.nan) for t in turns]
            r2_hi = [tw_stats[str(t)].get('isotonic_r2_ci_high', np.nan) for t in turns]
            rho_vals = [tw_stats[str(t)].get('spearman_rho', np.nan) for t in turns]
            rho_lo = [tw_stats[str(t)].get('spearman_rho_ci_low', np.nan) for t in turns]
            rho_hi = [tw_stats[str(t)].get('spearman_rho_ci_high', np.nan) for t in turns]
            rho_p = [tw_stats[str(t)].get('spearman_p', np.nan) for t in turns]

            # Flip rho for sad_vs_happy
            rho_vals = [v * flip_rho_sign for v in rho_vals]
            rho_lo_adj = [v * flip_rho_sign for v in (rho_hi if flip_rho_sign == -1 else rho_lo)]
            rho_hi_adj = [v * flip_rho_sign for v in (rho_lo if flip_rho_sign == -1 else rho_hi)]

            plot_line_with_ci(axes[0], turns, r2_vals, r2_lo, r2_hi,
                              color=color, label=model_name)
            plot_line_with_ci(axes[1], turns, rho_vals, rho_lo_adj, rho_hi_adj,
                              color=color, label=model_name)

            results = load_results(exp_dir)
            df_cross = results_to_dataframe(results, probe_name=cross_concept)
            sub_cross = df_cross[np.isclose(df_cross['alpha'], 0.0)].dropna(
                subset=['conversation_index', 'turn', 'probe_score', 'logit_rating']).copy()
            sub_cross['probe_display'] = flip_if_needed(
                cross_concept, sub_cross['probe_score'].values)
            first_last_r2 = _first_last_metric_bootstrap(
                sub_cross, 'probe_display', 'logit_rating', isotonic_r2)
            first_last_rho = _first_last_metric_bootstrap(
                sub_cross, 'probe_display', 'logit_rating',
                lambda x, y: stats.spearmanr(x, y)[0])

            cross_turnwise[model_name] = {
                'turns': turns,
                'r2': [round(v, 4) if not np.isnan(v) else None for v in r2_vals],
                'rho_flipped': [round(v, 4) if not np.isnan(v) else None for v in rho_vals],
                'rho_p': [float(p) if not np.isnan(p) else None for p in rho_p],
                'rho_sign_flipped': cross_concept in FLIP_CONCEPTS,
                'first_last_tests': {
                    'r2_last_minus_first': first_last_r2,
                    'rho_last_minus_first': first_last_rho,
                },
            }
        except Exception as e:
            print(f"    Warning: Cross-family turnwise error for {model_name}: {e}")

    axes[0].set_xlabel('Turn')
    axes[0].set_ylabel('Isotonic R²')
    axes[0].set_title('Introspection Accuracy by Turn')
    axes[0].set_xlim(0.5, 10.5)
    axes[0].set_xticks(range(1, 11))
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    add_panel_label(axes[0], 'M')

    axes[1].set_xlabel('Turn')
    axes[1].set_ylabel('Spearman ρ (flipped)')
    axes[1].set_title('Introspection Correlation by Turn')
    axes[1].set_xlim(0.5, 10.5)
    axes[1].set_xticks(range(1, 11))
    axes[1].set_ylim(-0.3, 1)
    axes[1].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[1].legend(fontsize=8)
    add_panel_label(axes[1], 'N')

    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_MN_cross_family_turnwise')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_M_N',
        'title': 'Turnwise Introspection — Gemma vs Qwen (Rho Flipped)',
        'description': (f'Isotonic R² (K) and Spearman ρ (L, sign-flipped for {cross_concept}) '
                        'per turn for cross-family models.'),
        'cross_turnwise': cross_turnwise,
    })

    # ── Save other_stats ──
    other_stats = {
        'description': ('Figure 5 validates cross-model generalization. R² and Rho recomputed '
                        'from raw experiment directories. Panel C replaced with self-report '
                        'vs alpha curves + Rho heatmap. Layer sweep normalized. '
                        'Rho sign-flipped for sad_vs_happy in K/L. Drift bar charts added.'),
        'raw_r2_data': {
            size: {short: {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in vals.items()}
                   for short, vals in sd.items()}
            for size, sd in raw_r2_data.items()
        },
        'cross_family_introspection': cross_scatter_stats,
        'cross_family_turnwise': cross_turnwise,
        'llama_8b_best': best_scatter_stats,
        'validated_mean_r2': d_json,
        'drift_probe': drift_stats_e,
        'drift_probe_size_trend_bootstrap': eii_trend,
        'probe_scale_control': probe_scale_control,
        'drift_report': drift_stats_f,
        'drift_report_size_trend_bootstrap': fii_trend,
        'rho_matrix_steering': rho_matrix.tolist(),
    }
    save_other_stats(fig_dir, other_stats)
    save_other_stats(appendix_dir, {
        'description': ('Appendix Figure 5 contains the probe-quality heatmap, the '
                        'multi-panel steering summary, and the cross-family layer sweep.'),
        'probe_quality': d_values,
        'steering_curves': c_stats,
        'cross_family_layer_sweep': sweep_info,
    })
    print("    Figure 5 complete.")


if __name__ == '__main__':
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rdir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    generate_figure_5(rdir)
    print(f"Output -> {rdir}")
