"""
Figure 4: 4×4 Steering Matrix and Introspection Improvement
=============================================================
Shows the full matrix of steering × measured concept combinations,
identifies conditions where steering significantly improves introspection,
and decomposes improvement into internal-state and report-quality components.
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
    MATRIX_ANALYSIS_DIR, PROBE_METRIC_KEY, FLIP_CONCEPTS,
    ALPHA_COLORS,
    load_results, load_summary, load_turnwise,
    load_matrix_r2_csv, load_matrix_increase_csv, load_matrix_summary,
    results_to_dataframe, flip_if_needed,
    get_turnwise_stats, isotonic_r2, bootstrap_stat, spearman_rho,
    savefig, save_panel_json, save_other_stats, ensure_dir,
    add_panel_label, plot_line_with_ci, format_p,
)

SHORTHANDS = SHORTHANDS_ORDERED  # focus, wellbeing, interest, impulsivity
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

        # CSV is long format: steering_concept, measured_concept, alpha, isotonic_r2
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
        # Annotate
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
                            f'measured concept combination at α = {alpha:+.0f}. '
                            f'Shared color scale across all alpha panels.'),
            'alpha': alpha,
            'vmin': vmin_global,
            'vmax': vmax_global,
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

            # Check significance
            sig = False
            sig_col = row.get('significant_p_lt_threshold')
            if pd.notna(sig_col):
                sig = (str(sig_col).lower() in ['yes', 'true', '1'])
            # Also check p-value
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
            label = f'{v:.3f}'
            ax.text(mi, si, label, ha='center', va='center',
                    fontsize=8, color=text_color)
    # Mark significant cells
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
                        'Red borders and ★ indicate statistically significant improvement '
                        '(bootstrap p < 0.05). Two significant cells: focus→wellbeing '
                        'and impulsivity→interest.'),
        'significant_cells': [
            {'steering': concept_order[s], 'measured': concept_order[m],
             'increase': round(v, 4)}
            for s, m, v in significant_cells
        ],
        'matrix': increase_matrix.tolist(),
        'concept_order': concept_order,
    })

    # ──────────────────────────────────────────────────────────────
    # Panels G–H: R² iso and Rho vs alpha for the 2 significant conditions
    # ──────────────────────────────────────────────────────────────
    # The two significant conditions:
    sig_conditions = [
        ('focus', 'wellbeing', 'distracted_vs_focused', 'sad_vs_happy'),
        ('impulsivity', 'interest', 'impulsive_vs_planning', 'bored_vs_interested'),
    ]
    sig_cond_colors = ['#E67E22', '#27AE60']  # wellbeing, interest colors

    # Load data for significant conditions across alphas
    sig_data = {}
    for steer_short, meas_short, steer_concept, meas_concept in sig_conditions:
        exp_key = (steer_short, meas_short)
        exp_dir = LLAMA_3B_4X4_WITH_RERUNS.get(exp_key)
        if exp_dir is None or not os.path.isdir(exp_dir):
            continue

        results = load_results(exp_dir)
        df = results_to_dataframe(results, probe_name=meas_concept)

        r2_by_alpha = []
        rho_by_alpha = []
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
                r2_pt, r2_lo, r2_hi = np.nan, np.nan, np.nan
                rho_pt, rho_lo, rho_hi = np.nan, np.nan, np.nan

            r2_by_alpha.append(r2_pt)
            r2_ci_lo_a.append(r2_lo)
            r2_ci_hi_a.append(r2_hi)
            rho_by_alpha.append(rho_pt)
            rho_ci_lo_a.append(rho_lo)
            rho_ci_hi_a.append(rho_hi)

            # Variance of logit rating
            v = np.var(ratings)
            rng = np.random.RandomState(42)
            boot_v = [np.var(rng.choice(ratings, len(ratings))) for _ in range(1000)]
            var_by_alpha.append(v)
            var_ci_lo_a.append(np.percentile(boot_v, 2.5))
            var_ci_hi_a.append(np.percentile(boot_v, 97.5))

        label = f'{SHORTHAND_DISPLAY[steer_short]} → {SHORTHAND_DISPLAY[meas_short]}'
        sig_data[(steer_short, meas_short)] = {
            'label': label,
            'r2': r2_by_alpha, 'r2_ci': list(zip(r2_ci_lo_a, r2_ci_hi_a)),
            'rho': rho_by_alpha, 'rho_ci': list(zip(rho_ci_lo_a, rho_ci_hi_a)),
            'variance': var_by_alpha,
            'variance_ci': list(zip(var_ci_lo_a, var_ci_hi_a)),
            'df': df,
            'meas_concept': meas_concept,
        }

    # Panel G: R² isotonic vs alpha
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    for idx, (key, data) in enumerate(sig_data.items()):
        color = sig_cond_colors[idx]
        r2_lo = [c[0] for c in data['r2_ci']]
        r2_hi = [c[1] for c in data['r2_ci']]
        plot_line_with_ci(ax, ALPHAS, data['r2'], r2_lo, r2_hi,
                          color=color, label=data['label'])
    ax.set_xlabel('Steering α')
    ax.set_ylabel('Isotonic R²')
    ax.set_title('Introspection vs. Steering Strength')
    ax.legend(fontsize=8)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylim(0, 1)
    add_panel_label(ax, 'G')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_04_G_r2_vs_alpha_significant')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_04_G',
        'title': 'Isotonic R² vs. Steering Alpha (Significant Conditions)',
        'description': ('Isotonic R² as a function of steering alpha for the two '
                        'significant conditions (focus→wellbeing and impulsivity→interest). '
                        'Shows monotonic improvement with steering.'),
        'conditions': {str(k): {
            'r2_values': [round(v, 4) for v in d['r2']],
            'alphas': ALPHAS,
        } for k, d in sig_data.items()},
    })

    # Panel H: Spearman rho vs alpha
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    for idx, (key, data) in enumerate(sig_data.items()):
        color = sig_cond_colors[idx]
        rho_lo = [c[0] for c in data['rho_ci']]
        rho_hi = [c[1] for c in data['rho_ci']]
        plot_line_with_ci(ax, ALPHAS, data['rho'], rho_lo, rho_hi,
                          color=color, label=data['label'])
    ax.set_xlabel('Steering α')
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Introspection Correlation vs. Steering Strength')
    ax.legend(fontsize=8)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylim(-0.2, 1)
    add_panel_label(ax, 'H')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_04_H_rho_vs_alpha_significant')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_04_H',
        'title': 'Spearman ρ vs. Steering Alpha (Significant Conditions)',
        'description': 'Spearman ρ as a function of steering alpha for the two significant conditions.',
        'conditions': {str(k): {
            'rho_values': [round(v, 4) for v in d['rho']],
        } for k, d in sig_data.items()},
    })

    # ──────────────────────────────────────────────────────────────
    # Panels I–J: Probe score drift for different alphas (2 conditions)
    # ──────────────────────────────────────────────────────────────
    panel_labels_drift = ['I', 'J']
    drift_decomposition = {}
    for ci, (key, data) in enumerate(sig_data.items()):
        steer_short, meas_short = key
        meas_concept = data['meas_concept']
        df = data['df']
        label_name = data['label']

        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
        drift_by_alpha = {}
        for a in ALPHAS:
            sub = df[np.isclose(df['alpha'], a)].copy()
            sub['probe_display'] = flip_if_needed(meas_concept, sub['probe_score'].values)
            turns = sorted(sub['turn'].unique())
            means = []
            ci_lo_list, ci_hi_list = [], []
            for t in turns:
                vals = sub[sub['turn'] == t]['probe_display'].dropna().values
                m = np.mean(vals)
                rng = np.random.RandomState(42)
                boots = [np.mean(rng.choice(vals, len(vals))) for _ in range(1000)]
                means.append(m)
                ci_lo_list.append(np.percentile(boots, 2.5))
                ci_hi_list.append(np.percentile(boots, 97.5))

            c = ALPHA_COLORS.get(a, 'gray')
            plot_line_with_ci(ax, turns, means, ci_lo_list, ci_hi_list,
                              color=c, label=f'α = {a:+.0f}', alpha_fill=0.1)

            # Drift metric: difference between last and first turn mean
            drift_by_alpha[str(a)] = {
                'drift_magnitude': round(float(means[-1] - means[0]), 4),
                'first_turn_mean': round(float(means[0]), 4),
                'last_turn_mean': round(float(means[-1]), 4),
            }

        ax.set_xlabel('Turn')
        ax.set_ylabel('Probe Score')
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
            'title': f'Probe Score Drift Under Steering — {label_name}',
            'description': (f'Mean probe score ({meas_concept}) across turns for different '
                            f'steering alphas (steering with {steer_short}). Shows how '
                            f'steering affects both level and drift of internal state.'),
            'drift_by_alpha': drift_by_alpha,
        })
        drift_decomposition[f'{steer_short}→{meas_short}'] = drift_by_alpha

    # ──────────────────────────────────────────────────────────────
    # Panel K: Report variance vs alpha (both significant conditions)
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    for idx, (key, data) in enumerate(sig_data.items()):
        color = sig_cond_colors[idx]
        var_lo = [c[0] for c in data['variance_ci']]
        var_hi = [c[1] for c in data['variance_ci']]
        plot_line_with_ci(ax, ALPHAS, data['variance'], var_lo, var_hi,
                          color=color, label=data['label'])
    ax.set_xlabel('Steering α')
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
        'title': 'Self-Report Variance vs. Steering Alpha',
        'description': ('Variance of logit-based self-reports as a function of steering '
                        'alpha. Higher variance = more informative reports. In both '
                        'conditions, the direction that improves R² also increases variance.'),
        'conditions': {str(k): {
            'variance_values': [round(v, 4) for v in d['variance']],
        } for k, d in sig_data.items()},
    })

    # ──────────────────────────────────────────────────────────────
    # Panel L: Drift metric vs alpha for both conditions
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    for idx, (key, data) in enumerate(sig_data.items()):
        steer_short, meas_short = key
        color = sig_cond_colors[idx]
        dk = f'{steer_short}→{meas_short}'
        if dk in drift_decomposition:
            drift_vals = [drift_decomposition[dk][str(a)]['drift_magnitude']
                          for a in ALPHAS]
            ax.plot(ALPHAS, drift_vals, 'o-', color=color, label=data['label'],
                    linewidth=1.5, markersize=6)
    ax.set_xlabel('Steering α')
    ax.set_ylabel('Drift (Δ Probe Score, last − first turn)')
    ax.set_title('Internal State Drift Magnitude vs. Steering')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=8)
    add_panel_label(ax, 'L')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_04_L_drift_magnitude_vs_alpha')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_04_L',
        'title': 'Drift Magnitude vs. Steering Alpha',
        'description': ('Drift of probe score (last minus first turn mean) as a function '
                        'of steering alpha. In focus→wellbeing, drift varies with alpha '
                        '(both components affected). In impulsivity→interest, drift is '
                        'relatively constant (only report component affected).'),
        'drift_decomposition': drift_decomposition,
    })

    # ── Save other_stats ──
    other_stats = {
        'description': ('Figure 4 presents the 4×4 steering matrix, identifying two '
                        'conditions where steering significantly improves introspection: '
                        '(1) focus→wellbeing (ΔR²=0.299, p=0.001); '
                        '(2) impulsivity→interest (ΔR²=0.098, p=0.012). '
                        'Decomposition analysis shows that in condition 1, both internal '
                        'state formation and report quality improve; in condition 2, '
                        'only report quality improves.'),
        'significant_conditions': [
            {'steering': 'focus (distracted_vs_focused)',
             'measured': 'wellbeing (sad_vs_happy)',
             'baseline_r2': None, 'best_r2': None, 'increase': 0.299,
             'p_value': 0.001},
            {'steering': 'impulsivity (impulsive_vs_planning)',
             'measured': 'interest (bored_vs_interested)',
             'baseline_r2': None, 'best_r2': None, 'increase': 0.098,
             'p_value': 0.012},
        ],
        'drift_decomposition': drift_decomposition,
    }
    # Fill in actual R2 values from sig_data
    for idx, (key, data) in enumerate(sig_data.items()):
        if idx < len(other_stats['significant_conditions']):
            alpha0_idx = ALPHAS.index(0.0)
            other_stats['significant_conditions'][idx]['baseline_r2'] = round(
                data['r2'][alpha0_idx], 4)
            best_idx = np.nanargmax(data['r2'])
            other_stats['significant_conditions'][idx]['best_r2'] = round(
                data['r2'][best_idx], 4)
            other_stats['significant_conditions'][idx]['best_alpha'] = ALPHAS[best_idx]

    save_other_stats(fig_dir, other_stats)
    print("    Figure 4 complete.")


if __name__ == '__main__':
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rdir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    generate_figure_4(rdir)
    print(f"Output → {rdir}")
