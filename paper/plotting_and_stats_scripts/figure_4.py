"""
Figure 4: 4×4 Steering Matrix and Introspection Improvement
=============================================================
Shows the full matrix of steering × measured concept combinations,
identifies conditions where steering significantly improves introspection,
and decomposes improvement into internal-state and report-quality components.

v2 changes:
 - G-H: flip alpha for FLIP steering concepts (impulsive_vs_planning)
 - G: auto y-scale (not forced 0-1)
 - I-J: flip alpha labels for FLIP steering concepts; verify probe polarity
 - K-L: flip alpha for FLIP steering concepts
 - Enhanced drift significance statistics
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from shared_utils import (
    CONCEPTS_ORDERED, CONCEPT_DISPLAY, CONCEPT_COLORS,
    SHORTHAND_TO_CONCEPT, CONCEPT_TO_SHORTHAND,
    SHORTHAND_DISPLAY, SHORTHANDS_ORDERED,
    LLAMA_3B_4X4_WITH_RERUNS, LLAMA_3B_4X4_ALL,
    MATRIX_ANALYSIS_DIR, PROBE_METRIC_KEY,
    FLIP_CONCEPTS, FLIP_SHORTHANDS,
    ALPHA_COLORS,
    load_results, load_summary, load_turnwise,
    load_matrix_r2_csv, load_matrix_increase_csv, load_matrix_summary,
    results_to_dataframe, flip_if_needed,
    flip_alpha_if_needed, flip_alpha_scalar,
    get_turnwise_stats, isotonic_r2, bootstrap_stat, spearman_rho,
    savefig, save_panel_json, save_other_stats, ensure_dir,
    add_panel_label, plot_line_with_ci, format_p,
)

SHORTHANDS = SHORTHANDS_ORDERED
ALPHAS = [-4.0, -2.0, 0.0, 2.0, 4.0]


def _concept_order_from_summary():
    """Get concept order used in matrix analysis."""
    try:
        summ = load_matrix_summary()
        return summ.get('concept_order', SHORTHANDS)
    except Exception:
        return SHORTHANDS


def generate_figure_4(results_dir):
    """Generate all Figure 4 panels."""
    fig_dir = ensure_dir(os.path.join(results_dir, 'Figure_4'))
    print("  Generating Figure 4: 4×4 Steering Matrix & Introspection Improvement...")

    other_stats = {}
    concept_order = _concept_order_from_summary()

    # Load pre-computed matrices
    r2_csv = load_matrix_r2_csv()
    inc_csv = load_matrix_increase_csv()
    try:
        mat_summ = load_matrix_summary()
        vmin_global = mat_summ.get('global_isotonic_r2_vmin', 0)
        vmax_global = mat_summ.get('global_isotonic_r2_vmax', 0.8)
    except Exception:
        vmin_global, vmax_global = 0, 0.8

    # ──────────────────────────────────────────────────────────────
    # Panels A–E: 5 heatmaps — R² isotonic for each alpha
    # ──────────────────────────────────────────────────────────────
    labels = ['A', 'B', 'C', 'D', 'E']
    for ai, alpha in enumerate(ALPHAS):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
        matrix = np.zeros((len(concept_order), len(concept_order)))

        alpha_rows = r2_csv[np.isclose(r2_csv['alpha'].astype(float), alpha)]
        for _, row in alpha_rows.iterrows():
            steer = row['steering_concept']
            meas = row['measured_concept']
            val = row['isotonic_r2']
            if pd.isna(val):
                continue
            if steer in concept_order and meas in concept_order:
                si = concept_order.index(steer)
                mi = concept_order.index(meas)
                matrix[si, mi] = float(val)

        im = ax.imshow(matrix, cmap='YlOrRd', vmin=vmin_global, vmax=vmax_global,
                       aspect='equal')
        for si in range(len(concept_order)):
            for mi in range(len(concept_order)):
                v = matrix[si, mi]
                text_color = 'white' if v > 0.6 * vmax_global else 'black'
                ax.text(mi, si, f'{v:.3f}', ha='center', va='center',
                        fontsize=8, color=text_color)

        display_labels = [SHORTHAND_DISPLAY.get(c, c) for c in concept_order]
        ax.set_xticks(range(len(concept_order)))
        ax.set_xticklabels(display_labels, fontsize=8, rotation=30, ha='right')
        ax.set_yticks(range(len(concept_order)))
        ax.set_yticklabels(display_labels, fontsize=8)
        ax.set_xlabel('Measured Concept')
        ax.set_ylabel('Steering Direction')
        ax.set_title(f'Isotonic R²  (α = {alpha:+.0f})')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Isotonic R²')
        add_panel_label(ax, labels[ai])
        plt.tight_layout()
        prefix = os.path.join(fig_dir,
                              f'Fig_04_{labels[ai]}_heatmap_alpha_{alpha:+.0f}')
        savefig(fig, prefix)
        save_panel_json(prefix, {
            'panel_id': f'Fig_04_{labels[ai]}',
            'title': f'Isotonic R² Matrix (α = {alpha:+.0f})',
            'description': (f'4×4 heatmap of isotonic R² values for each steering × '
                            f'measured concept combination at α = {alpha:+.0f}.'),
            'alpha': alpha,
            'vmin': vmin_global, 'vmax': vmax_global,
            'matrix': matrix.tolist(),
            'concept_order': concept_order,
            'data_source': os.path.join(MATRIX_ANALYSIS_DIR,
                                        'isotonic_r2_values_by_cell_alpha.csv'),
        })

    # ──────────────────────────────────────────────────────────────
    # Panel F: Max increase heatmap with significance markers
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    increase_matrix = np.zeros((len(concept_order), len(concept_order)))
    significant_cells = []

    for _, row in inc_csv.iterrows():
        steer = row['steering_concept']
        meas = row['measured_concept']
        if steer in concept_order and meas in concept_order:
            si = concept_order.index(steer)
            mi = concept_order.index(meas)
            inc_val = row.get('max_increase_vs_baseline', 0)
            increase_matrix[si, mi] = float(inc_val) if pd.notna(inc_val) else 0

            sig = False
            sig_col = row.get('significant_p_lt_threshold')
            if pd.notna(sig_col):
                sig = (str(sig_col).lower() in ['yes', 'true', '1'])
            p_col = row.get('bootstrap_p_two_sided')
            if pd.notna(p_col) and float(p_col) < 0.05:
                sig = True
            if sig:
                significant_cells.append((si, mi, float(inc_val)))

    im = ax.imshow(increase_matrix, cmap='Greens', aspect='equal')
    for si in range(len(concept_order)):
        for mi in range(len(concept_order)):
            v = increase_matrix[si, mi]
            text_color = 'white' if v > 0.15 else 'black'
            ax.text(mi, si, f'{v:.3f}', ha='center', va='center',
                    fontsize=8, color=text_color)
    for si, mi, val in significant_cells:
        rect = plt.Rectangle((mi - 0.5, si - 0.5), 1, 1,
                              fill=False, edgecolor='red', linewidth=2.5)
        ax.add_patch(rect)
        ax.text(mi, si + 0.35, '*', ha='center', va='center',
                fontsize=14, color='red', fontweight='bold')

    display_labels = [SHORTHAND_DISPLAY.get(c, c) for c in concept_order]
    ax.set_xticks(range(len(concept_order)))
    ax.set_xticklabels(display_labels, fontsize=8, rotation=30, ha='right')
    ax.set_yticks(range(len(concept_order)))
    ax.set_yticklabels(display_labels, fontsize=8)
    ax.set_xlabel('Measured Concept')
    ax.set_ylabel('Steering Direction')
    ax.set_title('Max R² Increase vs. Baseline (α = 0)')
    plt.colorbar(im, ax=ax, shrink=0.8, label='ΔR² (max)')
    add_panel_label(ax, 'F')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_04_F_max_increase_matrix')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_04_F',
        'title': 'Maximum R² Increase vs. Baseline',
        'description': ('Maximum isotonic R² increase over baseline (α=0) for each cell. '
                        'Red borders and * indicate significant improvement (p < 0.05).'),
        'significant_cells': [
            {'steering': concept_order[s], 'measured': concept_order[m],
             'increase': round(v, 4)} for s, m, v in significant_cells
        ],
        'matrix': increase_matrix.tolist(),
        'concept_order': concept_order,
    })

    # ──────────────────────────────────────────────────────────────
    # Load data for significant conditions
    # ──────────────────────────────────────────────────────────────
    sig_conditions = [
        ('focus', 'wellbeing', 'distracted_vs_focused', 'sad_vs_happy'),
        ('impulsivity', 'interest', 'impulsive_vs_planning', 'bored_vs_interested'),
    ]
    sig_cond_colors = ['#E67E22', '#27AE60']

    sig_data = {}
    for steer_short, meas_short, steer_concept, meas_concept in sig_conditions:
        exp_key = (steer_short, meas_short)
        exp_dir = LLAMA_3B_4X4_WITH_RERUNS.get(exp_key)
        if exp_dir is None or not os.path.isdir(exp_dir):
            continue

        results = load_results(exp_dir)
        df = results_to_dataframe(results, probe_name=meas_concept)

        # Determine if steering concept needs alpha flip
        steer_needs_flip = steer_concept in FLIP_CONCEPTS

        r2_by_alpha, rho_by_alpha = [], []
        r2_ci_lo_a, r2_ci_hi_a = [], []
        rho_ci_lo_a, rho_ci_hi_a = [], []
        var_by_alpha = []
        var_ci_lo_a, var_ci_hi_a = [], []

        for a in ALPHAS:
            sub = df[np.isclose(df['alpha'], a)].dropna(
                subset=['probe_score', 'logit_rating'])
            probe_vals = flip_if_needed(meas_concept, sub['probe_score'].values)
            ratings = sub['logit_rating'].values

            if len(probe_vals) > 5:
                r2_pt, r2_lo, r2_hi = bootstrap_stat(probe_vals, ratings, isotonic_r2)
                rho_pt, rho_lo, rho_hi = bootstrap_stat(probe_vals, ratings, spearman_rho)
            else:
                r2_pt = r2_lo = r2_hi = np.nan
                rho_pt = rho_lo = rho_hi = np.nan

            r2_by_alpha.append(r2_pt)
            r2_ci_lo_a.append(r2_lo)
            r2_ci_hi_a.append(r2_hi)
            rho_by_alpha.append(rho_pt)
            rho_ci_lo_a.append(rho_lo)
            rho_ci_hi_a.append(rho_hi)

            v = np.var(ratings) if len(ratings) > 0 else 0
            rng = np.random.RandomState(42)
            if len(ratings) > 1:
                boot_v = [np.var(rng.choice(ratings, len(ratings))) for _ in range(1000)]
                var_ci_lo_a.append(np.percentile(boot_v, 2.5))
                var_ci_hi_a.append(np.percentile(boot_v, 97.5))
            else:
                var_ci_lo_a.append(0)
                var_ci_hi_a.append(0)
            var_by_alpha.append(v)

        label = f'{SHORTHAND_DISPLAY[steer_short]} → {SHORTHAND_DISPLAY[meas_short]}'

        # Compute display alphas (flip for FLIP steering concept)
        display_alphas = flip_alpha_if_needed(steer_short, np.array(ALPHAS))

        sig_data[exp_key] = {
            'label': label,
            'steer_short': steer_short,
            'meas_short': meas_short,
            'steer_concept': steer_concept,
            'meas_concept': meas_concept,
            'steer_needs_flip': steer_needs_flip,
            'r2': r2_by_alpha,
            'r2_ci': list(zip(r2_ci_lo_a, r2_ci_hi_a)),
            'rho': rho_by_alpha,
            'rho_ci': list(zip(rho_ci_lo_a, rho_ci_hi_a)),
            'variance': var_by_alpha,
            'variance_ci': list(zip(var_ci_lo_a, var_ci_hi_a)),
            'display_alphas': display_alphas.tolist(),
            'raw_alphas': ALPHAS,
            'df': df,
        }

    # ──────────────────────────────────────────────────────────────
    # Panel G: R² isotonic vs display alpha
    #   — Auto y-scale, alpha flipped for FLIP steering concepts
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    g_stats = {}
    for idx, (key, data) in enumerate(sig_data.items()):
        color = sig_cond_colors[idx]
        da = data['display_alphas']
        sort_idx = np.argsort(da)
        da_sorted = [da[k] for k in sort_idx]
        r2_sorted = [data['r2'][k] for k in sort_idx]
        r2_lo_sorted = [data['r2_ci'][k][0] for k in sort_idx]
        r2_hi_sorted = [data['r2_ci'][k][1] for k in sort_idx]
        plot_line_with_ci(ax, da_sorted, r2_sorted, r2_lo_sorted, r2_hi_sorted,
                          color=color, label=data['label'])
        # Test: Spearman of R² vs display alpha
        rho_r2a, p_r2a = stats.spearmanr(da_sorted, r2_sorted)
        g_stats[str(key)] = {
            'display_alphas': da_sorted,
            'r2_values': [round(v, 4) for v in r2_sorted],
            'spearman_rho_r2_vs_alpha': round(float(rho_r2a), 4),
            'spearman_p': float(p_r2a),
            'alpha_display_flipped': data['steer_needs_flip'],
        }
    ax.set_xlabel('Steering α (display-corrected)')
    ax.set_ylabel('Isotonic R²')
    ax.set_title('Introspection vs. Steering Strength')
    ax.legend(fontsize=8)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    # Auto y-scale (do NOT force 0-1)
    add_panel_label(ax, 'G')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_04_G_r2_vs_alpha_significant')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_04_G',
        'title': 'Isotonic R² vs. Steering Alpha (Display-Corrected)',
        'description': ('Isotonic R² as a function of display-corrected steering alpha. '
                        'For impulsive_vs_planning steering, alpha is sign-flipped so positive '
                        'α = more impulsive. Auto y-scale.'),
        'conditions': g_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel H: Spearman rho vs display alpha
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    h_stats = {}
    for idx, (key, data) in enumerate(sig_data.items()):
        color = sig_cond_colors[idx]
        da = data['display_alphas']
        sort_idx = np.argsort(da)
        da_sorted = [da[k] for k in sort_idx]
        rho_sorted = [data['rho'][k] for k in sort_idx]
        rho_lo_sorted = [data['rho_ci'][k][0] for k in sort_idx]
        rho_hi_sorted = [data['rho_ci'][k][1] for k in sort_idx]
        plot_line_with_ci(ax, da_sorted, rho_sorted, rho_lo_sorted, rho_hi_sorted,
                          color=color, label=data['label'])
        rho_rha, p_rha = stats.spearmanr(da_sorted, rho_sorted)
        h_stats[str(key)] = {
            'display_alphas': da_sorted,
            'rho_values': [round(v, 4) for v in rho_sorted],
            'spearman_rho_change_vs_alpha': round(float(rho_rha), 4),
            'spearman_p': float(p_rha),
        }
    ax.set_xlabel('Steering α (display-corrected)')
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Introspection Correlation vs. Steering Strength')
    ax.legend(fontsize=8)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    # Tighten y-axis to data with some margin
    all_rho_vals = []
    for key, data in sig_data.items():
        all_rho_vals.extend([v for v in data['rho'] if not np.isnan(v)])
        for lo, hi in data['rho_ci']:
            if not np.isnan(lo):
                all_rho_vals.append(lo)
            if not np.isnan(hi):
                all_rho_vals.append(hi)
    if all_rho_vals:
        y_lo = min(all_rho_vals)
        y_hi = max(all_rho_vals)
        margin = (y_hi - y_lo) * 0.15
        ax.set_ylim(y_lo - margin, y_hi + margin)
    add_panel_label(ax, 'H')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_04_H_rho_vs_alpha_significant')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_04_H',
        'title': 'Spearman ρ vs. Steering Alpha (Display-Corrected)',
        'description': ('Spearman ρ vs display-corrected steering alpha for the two '
                        'significant conditions. Enhanced with Spearman test of rho-change '
                        'vs alpha.'),
        'conditions': h_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panels I–J: Probe score drift — alpha labels flipped
    # ──────────────────────────────────────────────────────────────
    panel_labels_drift = ['I', 'J']
    drift_decomposition = {}
    for ci, (key, data) in enumerate(sig_data.items()):
        steer_short = data['steer_short']
        meas_short = data['meas_short']
        meas_concept = data['meas_concept']
        df = data['df']
        label_name = data['label']

        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
        drift_by_alpha = {}

        for a in ALPHAS:
            sub = df[np.isclose(df['alpha'], a)].copy()
            sub['probe_display'] = flip_if_needed(meas_concept, sub['probe_score'].values)
            turns = sorted(sub['turn'].unique())
            means, ci_lo_list, ci_hi_list = [], [], []
            for t in turns:
                vals = sub[sub['turn'] == t]['probe_display'].dropna().values
                m = np.mean(vals)
                rng = np.random.RandomState(42)
                boots = [np.mean(rng.choice(vals, len(vals))) for _ in range(1000)]
                means.append(m)
                ci_lo_list.append(np.percentile(boots, 2.5))
                ci_hi_list.append(np.percentile(boots, 97.5))

            c = ALPHA_COLORS.get(a, 'gray')
            display_a = flip_alpha_scalar(steer_short, a)
            plot_line_with_ci(ax, turns, means, ci_lo_list, ci_hi_list,
                              color=c, label=f'α = {display_a:+.0f}', alpha_fill=0.1)

            # Drift significance
            sub_nona = sub.dropna(subset=['probe_display'])
            if len(sub_nona) > 3:
                rho_d, p_d = stats.spearmanr(sub_nona['turn'], sub_nona['probe_display'])
            else:
                rho_d, p_d = np.nan, np.nan
            # Per-conversation drift for error bars
            per_conv_drifts_l = []
            for ci_l in sub['conversation_index'].unique():
                cv_l = sub[sub['conversation_index'] == ci_l].sort_values('turn')
                pv_l = cv_l['probe_display'].dropna()
                if len(pv_l) >= 2:
                    per_conv_drifts_l.append(float(pv_l.iloc[-1] - pv_l.iloc[0]))

            drift_by_alpha[str(a)] = {
                'display_alpha': float(display_a),
                'drift_magnitude': round(float(means[-1] - means[0]), 4),
                'first_turn_mean': round(float(means[0]), 4),
                'last_turn_mean': round(float(means[-1]), 4),
                'drift_spearman_rho': round(float(rho_d), 4),
                'drift_spearman_p': float(p_d),
                'per_conv_drifts': per_conv_drifts_l,
            }

        ax.set_xlabel('Turn')
        ax.set_ylabel('Probe Score (polarity-corrected)')
        ax.set_title(f'Internal State Drift — {label_name}')
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=7, loc='best')
        add_panel_label(ax, panel_labels_drift[ci])
        plt.tight_layout()
        prefix = os.path.join(fig_dir,
                              f'Fig_04_{panel_labels_drift[ci]}_drift_{steer_short}_{meas_short}')
        savefig(fig, prefix)
        save_panel_json(prefix, {
            'panel_id': f'Fig_04_{panel_labels_drift[ci]}',
            'title': f'Probe Score Drift — {label_name}',
            'description': (f'Mean probe score ({meas_concept}, polarity-corrected) across turns '
                            f'for different steering alphas (steering: {steer_short}). '
                            f'Alpha labels display-corrected (flipped={data["steer_needs_flip"]}). '
                            f'Drift significance: Spearman ρ of turn vs probe score at each α.'),
            'steer_concept': steer_short,
            'meas_concept': meas_short,
            'alpha_display_flipped': data['steer_needs_flip'],
            'probe_polarity_flipped': meas_concept in FLIP_CONCEPTS,
            'drift_by_alpha': drift_by_alpha,
        })
        drift_decomposition[f'{steer_short}→{meas_short}'] = drift_by_alpha

    # ──────────────────────────────────────────────────────────────
    # Panel K: Report variance vs display alpha
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    k_stats = {}
    for idx, (key, data) in enumerate(sig_data.items()):
        color = sig_cond_colors[idx]
        da = data['display_alphas']
        sort_idx = np.argsort(da)
        da_sorted = [da[k] for k in sort_idx]
        var_sorted = [data['variance'][k] for k in sort_idx]
        var_lo_sorted = [data['variance_ci'][k][0] for k in sort_idx]
        var_hi_sorted = [data['variance_ci'][k][1] for k in sort_idx]
        plot_line_with_ci(ax, da_sorted, var_sorted, var_lo_sorted, var_hi_sorted,
                          color=color, label=data['label'])
        k_stats[str(key)] = {
            'display_alphas': da_sorted,
            'variance_values': [round(v, 4) for v in var_sorted],
        }
    ax.set_xlabel('Steering α (display-corrected)')
    ax.set_ylabel('Self-Report Variance')
    ax.set_title('Report Informativeness vs. Steering')
    ax.legend(fontsize=8)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    add_panel_label(ax, 'K')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_04_K_report_variance_vs_alpha')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_04_K',
        'title': 'Self-Report Variance vs. Display-Corrected Alpha',
        'description': ('Variance of logit self-reports vs display-corrected alpha. '
                        'Alpha sign-flipped for impulsive_vs_planning steering.'),
        'conditions': k_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel L: Drift magnitude vs display alpha
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    l_stats = {}
    rng_l = np.random.default_rng(42)
    for idx, (key, data) in enumerate(sig_data.items()):
        steer_short = data['steer_short']
        meas_short = data['meas_short']
        color = sig_cond_colors[idx]
        dk = f'{steer_short}→{meas_short}'
        if dk in drift_decomposition:
            da = data['display_alphas']
            sort_idx = np.argsort(da)
            da_sorted = [da[k] for k in sort_idx]
            drift_vals = []
            ci_lo_l = []
            ci_hi_l = []
            for k in sort_idx:
                entry = drift_decomposition[dk][str(ALPHAS[k])]
                pcl = np.array(entry.get('per_conv_drifts', []))
                mean_d = float(np.mean(pcl)) if len(pcl) > 0 else entry['drift_magnitude']
                drift_vals.append(mean_d)
                if len(pcl) >= 2:
                    boots = [float(np.mean(rng_l.choice(pcl, size=len(pcl), replace=True)))
                             for _ in range(1000)]
                    ci_lo_l.append(np.percentile(boots, 2.5))
                    ci_hi_l.append(np.percentile(boots, 97.5))
                else:
                    ci_lo_l.append(mean_d)
                    ci_hi_l.append(mean_d)
            yerr_lo = [drift_vals[j] - ci_lo_l[j] for j in range(len(drift_vals))]
            yerr_hi = [ci_hi_l[j] - drift_vals[j] for j in range(len(drift_vals))]
            ax.errorbar(da_sorted, drift_vals, yerr=[yerr_lo, yerr_hi],
                        fmt='o-', color=color, label=data['label'],
                        linewidth=1.5, markersize=6, capsize=3)
            # Trend test: Spearman of drift magnitude vs display alpha
            rho_dl, p_dl = stats.spearmanr(da_sorted, drift_vals)
            l_stats[dk] = {
                'display_alphas': da_sorted,
                'drift_values': [round(v, 4) for v in drift_vals],
                'ci_95_low': [round(v, 4) for v in ci_lo_l],
                'ci_95_high': [round(v, 4) for v in ci_hi_l],
                'spearman_rho_drift_vs_alpha': round(float(rho_dl), 4),
                'spearman_p': float(p_dl),
            }
    ax.set_xlabel('Steering α (display-corrected)')
    ax.set_ylabel('Drift (Δ Probe Score, last − first)')
    ax.set_title('Internal State Drift vs. Steering')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=8)
    add_panel_label(ax, 'L')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_04_L_drift_magnitude_vs_alpha')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_04_L',
        'title': 'Drift Magnitude vs. Display-Corrected Alpha',
        'description': ('Probe score drift (last−first turn) vs display-corrected alpha. '
                        'Alpha flipped for impulsive_vs_planning steering. Includes Spearman '
                        'test of drift-vs-alpha trend.'),
        'drift_vs_alpha_stats': l_stats,
    })

    # ── Save other_stats ──
    other_stats = {
        'description': ('Figure 4 presents the 4×4 steering matrix. Two conditions show '
                        'significant R² improvement: focus→wellbeing and impulsivity→interest. '
                        'Alpha is display-corrected (negated for impulsive_vs_planning steering) '
                        'so positive α consistently means "more of the concept." '
                        'Drift statistics include per-alpha Spearman rho of turn vs probe score.'),
        'significant_conditions': [],
        'drift_decomposition': drift_decomposition,
    }
    for idx, (key, data) in enumerate(sig_data.items()):
        alpha0_idx = ALPHAS.index(0.0)
        best_idx = int(np.nanargmax(data['r2']))
        other_stats['significant_conditions'].append({
            'steering': data['steer_short'],
            'measured': data['meas_short'],
            'baseline_r2': round(float(data['r2'][alpha0_idx]), 4),
            'best_r2': round(float(data['r2'][best_idx]), 4),
            'best_alpha_raw': ALPHAS[best_idx],
            'best_alpha_display': data['display_alphas'][best_idx],
            'alpha_flipped': data['steer_needs_flip'],
            'probe_meas_flipped': data['meas_concept'] in FLIP_CONCEPTS,
        })

    save_other_stats(fig_dir, other_stats)
    print("    Figure 4 complete.")


if __name__ == '__main__':
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rdir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    generate_figure_4(rdir)
    print(f"Output -> {rdir}")
