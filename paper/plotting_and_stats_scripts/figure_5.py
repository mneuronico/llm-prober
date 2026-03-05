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
    savefig, save_panel_json, save_other_stats, ensure_dir,
    add_panel_label, plot_line_with_ci, format_p,
    compute_per_turn_means,
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
    """Compute isotonic R² and Spearman rho from raw experiment directory."""
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
        return {
            'isotonic_r2': float(r2_pt),
            'r2_ci_low': float(r2_lo),
            'r2_ci_high': float(r2_hi),
            'spearman_rho': float(rho_val),
            'spearman_p': float(rho_p),
            'rho_ci_low': float(rho_lo),
            'rho_ci_high': float(rho_hi),
            'n': len(probe_vals),
        }
    except Exception:
        return None


def generate_figure_5(results_dir):
    """Generate all Figure 5 panels."""
    fig_dir = ensure_dir(os.path.join(results_dir, 'Figure_5'))
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

    im = ax.imshow(d_matrix, cmap='YlGnBu', aspect='auto')
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
    prefix = os.path.join(fig_dir, 'Fig_05_A_probe_quality_heatmap')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_A',
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

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
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
            ax.errorbar(sizes_x, r2_y,
                        yerr=[np.array(r2_y) - np.array(r2_lo),
                              np.array(r2_hi) - np.array(r2_y)],
                        fmt='o-', color=color, label=SHORTHAND_DISPLAY[short],
                        capsize=4, linewidth=1.5, markersize=7)
            b_stats[short] = {s: raw_r2_data[s].get(short, {}).get('isotonic_r2')
                              for s in LLAMA_SIZES}

    ax.set_xlabel('Model Size (B parameters)')
    ax.set_ylabel('Isotonic R²')
    ax.set_title('Introspection vs. Model Size')
    ax.set_xscale('log')
    ax.set_xticks([1, 3, 8])
    ax.set_xticklabels(['1B', '3B', '8B'])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc='upper left')
    add_panel_label(ax, 'B')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_B_r2_vs_model_size')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_B',
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
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    bii_stats = {}
    for short in SHORTHANDS:
        concept = SHORTHAND_TO_CONCEPT[short]
        color = CONCEPT_COLORS[concept]
        sizes_x, rho_y, rho_lo, rho_hi = [], [], [], []
        for size in LLAMA_SIZES:
            if short in raw_r2_data[size]:
                d = raw_r2_data[size][short]
                # Flip rho for FLIP concepts
                flip = -1 if concept in FLIP_CONCEPTS else 1
                sizes_x.append(LLAMA_SIZE_VALS[size])
                rho_y.append(d['spearman_rho'] * flip)
                rho_lo.append(d['rho_ci_low'] * flip if flip > 0 else d['rho_ci_high'] * flip)
                rho_hi.append(d['rho_ci_high'] * flip if flip > 0 else d['rho_ci_low'] * flip)
        if sizes_x:
            sort_idx = np.argsort(rho_lo)  # just for proper errorbars
            ax.errorbar(sizes_x, rho_y,
                        yerr=[np.array(rho_y) - np.array(rho_lo),
                              np.array(rho_hi) - np.array(rho_y)],
                        fmt='o-', color=color, label=SHORTHAND_DISPLAY[short],
                        capsize=4, linewidth=1.5, markersize=7)
            bii_stats[short] = {s: round(raw_r2_data[s].get(short, {}).get('spearman_rho', 0) * flip, 4)
                                for s in LLAMA_SIZES}

    ax.set_xlabel('Model Size (B parameters)')
    ax.set_ylabel('Spearman ρ (polarity-corrected)')
    ax.set_title('Introspection Correlation vs. Model Size')
    ax.set_xscale('log')
    ax.set_xticks([1, 3, 8])
    ax.set_xticklabels(['1B', '3B', '8B'])
    ax.set_ylim(-0.3, 1)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=7, loc='upper left')
    add_panel_label(ax, 'Bii')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_Bii_rho_vs_model_size')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_Bii',
        'title': 'Spearman ρ vs. Model Size',
        'description': ('Spearman ρ at alpha=0 (polarity-corrected for FLIP concepts). '
                        'Rho flipped for sad_vs_happy & impulsive_vs_planning.'),
        'per_concept': bii_stats,
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
    rho_p_matrix = np.full((len(LLAMA_SIZES), len(SHORTHANDS)), np.nan)
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
                display_alphas = flip_alpha_if_needed(short, np.array(alphas_found))

                means, ci_lo, ci_hi = [], [], []
                for a in alphas_found:
                    vals = df[np.isclose(df['alpha'], a)]['logit_rating'].dropna().values
                    m = np.mean(vals)
                    rng = np.random.RandomState(42)
                    boots = [np.mean(rng.choice(vals, len(vals))) for _ in range(1000)]
                    means.append(m)
                    ci_lo.append(np.percentile(boots, 2.5))
                    ci_hi.append(np.percentile(boots, 97.5))

                sort_idx = np.argsort(display_alphas)
                da_s = [display_alphas[k] for k in sort_idx]
                means_s = [means[k] for k in sort_idx]
                ci_lo_s = [ci_lo[k] for k in sort_idx]
                ci_hi_s = [ci_hi[k] for k in sort_idx]

                plot_line_with_ci(ax, da_s, means_s, ci_lo_s, ci_hi_s,
                                  color=color, label=SHORTHAND_DISPLAY[short],
                                  alpha_fill=0.15)

                # Compute rho(display_alpha, rating)
                da_all = flip_alpha_if_needed(short, df['alpha'].values)
                lr_all = df['logit_rating'].dropna().values
                n_min = min(len(da_all), len(lr_all))
                rho_c, p_c = stats.spearmanr(da_all[:n_min], lr_all[:n_min])
                rho_matrix[si, SHORTHANDS.index(short)] = rho_c
                rho_p_matrix[si, SHORTHANDS.index(short)] = p_c
                c_stats[size][short] = {
                    'display_alphas': [float(a) for a in da_s],
                    'mean_ratings': [round(m, 4) for m in means_s],
                    'spearman_rho_alpha_rating': round(float(rho_c), 4),
                    'spearman_p': float(p_c),
                }
            except Exception as e:
                print(f"    Warning: Panel C error {size}/{short}: {e}")

        ax.set_xlabel('Steering α (display-corrected)')
        ax.set_ylabel('Logit Self-Report')
        ax.set_title(f'LLaMA {size}')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        if si == 0:
            ax.legend(fontsize=6, loc='upper left')

    # Rho heatmap
    im = ax_heat.imshow(rho_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    for si in range(len(LLAMA_SIZES)):
        for ci_idx in range(len(SHORTHANDS)):
            v = rho_matrix[si, ci_idx]
            p_v = rho_p_matrix[si, ci_idx]
            if np.isnan(v):
                ax_heat.text(ci_idx, si, '—', ha='center', va='center', fontsize=9)
            else:
                sig_star = '*' if p_v < 0.05 else ''
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

    add_panel_label(c_axes[0], 'C')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_C_steering_curves_and_rho')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_C',
        'title': 'Self-Report vs Alpha Curves + Rho Heatmap',
        'description': ('Left 3 panels: logit self-report vs display-corrected alpha per '
                        'concept per model size. Right: heatmap of Spearman ρ between alpha '
                        'and logit rating (* = p<0.05). Replaces previous signed-R² panel.'),
        'per_size_stats': c_stats,
        'rho_matrix': rho_matrix.tolist(),
        'rho_p_matrix': rho_p_matrix.tolist(),
    })

    # ──────────────────────────────────────────────────────────────
    # Panel D: Mean R² bar chart (good-quality probes)
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    good_r2 = {s: [] for s in LLAMA_SIZES}

    for size in LLAMA_SIZES:
        for short in SHORTHANDS:
            if short in raw_r2_data[size]:
                d = raw_r2_data[size][short]
                # Exclude if rho(alpha, rating) is negative (inverted steering)
                concept = SHORTHAND_TO_CONCEPT[short]
                rho_idx = SHORTHANDS.index(short)
                si = LLAMA_SIZES.index(size)
                rho_val = rho_matrix[si, rho_idx]
                if np.isnan(rho_val) or rho_val >= 0:
                    good_r2[size].append(d['isotonic_r2'])

    means = [np.mean(good_r2[s]) if good_r2[s] else 0 for s in LLAMA_SIZES]
    ci_lo_list, ci_hi_list = [], []
    for s in LLAMA_SIZES:
        vals = good_r2[s]
        if len(vals) > 1:
            rng = np.random.RandomState(42)
            boots = [np.mean(rng.choice(vals, len(vals))) for _ in range(1000)]
            ci_lo_list.append(np.percentile(boots, 2.5))
            ci_hi_list.append(np.percentile(boots, 97.5))
        else:
            ci_lo_list.append(means[LLAMA_SIZES.index(s)])
            ci_hi_list.append(means[LLAMA_SIZES.index(s)])

    colors = [MODEL_SIZE_COLORS[s] for s in LLAMA_SIZES]
    ax.bar(range(len(LLAMA_SIZES)), means, color=colors,
           edgecolor='white', linewidth=0.5)
    for i in range(len(LLAMA_SIZES)):
        ax.errorbar(i, means[i],
                    yerr=[[means[i] - ci_lo_list[i]], [ci_hi_list[i] - means[i]]],
                    color='black', capsize=5, linewidth=1.5, capthick=1.5)
    ax.set_xticks(range(len(LLAMA_SIZES)))
    ax.set_xticklabels([f'LLaMA {s}' for s in LLAMA_SIZES])
    ax.set_ylabel('Mean Isotonic R²')
    ax.set_title('Introspection by Model Size\n(validated probes only)')
    ax.set_ylim(0, 1)
    add_panel_label(ax, 'D')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_D_mean_r2_by_size')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_D',
        'title': 'Mean R² by Size (Validated Probes)',
        'description': ('Mean isotonic R² across concepts with validated probes '
                        '(ρ(α,rating) ≥ 0), excluding inverted-steering cases.'),
        'good_r2': {s: [round(v, 4) for v in good_r2[s]] for s in LLAMA_SIZES},
        'means': [round(m, 4) for m in means],
    })

    # ──────────────────────────────────────────────────────────────
    # Panel E: Probe score drift — one concept, 3 LLaMA sizes
    # ──────────────────────────────────────────────────────────────
    example_concept = 'wellbeing'
    concept_name = SHORTHAND_TO_CONCEPT[example_concept]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
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
            turns = sorted(sub['turn'].unique())
            means, ci_lo, ci_hi = [], [], []
            for t in turns:
                vals = sub[sub['turn'] == t]['probe_display'].dropna().values
                m = np.mean(vals)
                rng = np.random.RandomState(42)
                boots = [np.mean(rng.choice(vals, len(vals))) for _ in range(1000)]
                means.append(m)
                ci_lo.append(np.percentile(boots, 2.5))
                ci_hi.append(np.percentile(boots, 97.5))
            color = MODEL_SIZE_COLORS[size]
            plot_line_with_ci(ax, turns, means, ci_lo, ci_hi,
                              color=color, label=f'LLaMA {size}')
            rho_e, p_e = stats.spearmanr(
                sub['turn'], sub['probe_display'].dropna())
            drift_stats_e[size] = {
                'drift_magnitude': round(float(means[-1] - means[0]), 4),
                'spearman_rho_vs_turn': round(float(rho_e), 4),
                'spearman_p': float(p_e),
            }
        except Exception as e:
            print(f"    Warning: Drift plot error for {size}: {e}")

    ax.set_xlabel('Turn')
    ax.set_ylabel('Probe Score (polarity-corrected)')
    ax.set_title(f'Internal State Drift — {SHORTHAND_DISPLAY[example_concept]}')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)
    add_panel_label(ax, 'E')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_E_drift_across_sizes')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_E',
        'title': f'Probe Drift Across Sizes — {SHORTHAND_DISPLAY[example_concept]}',
        'drift_stats': drift_stats_e,
    })

    # Panel Eii: Drift magnitude bar chart
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    for i, size in enumerate(LLAMA_SIZES):
        if size in drift_stats_e:
            color = MODEL_SIZE_COLORS[size]
            ax.bar(i, drift_stats_e[size]['drift_magnitude'], color=color,
                   edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(LLAMA_SIZES)))
    ax.set_xticklabels([f'LLaMA {s}' for s in LLAMA_SIZES])
    ax.set_ylabel('Drift (last − first turn)')
    ax.set_title(f'Probe Drift Magnitude — {SHORTHAND_DISPLAY[example_concept]}')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    add_panel_label(ax, 'Eii')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_Eii_drift_magnitude_bars')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_Eii',
        'title': 'Probe Drift Magnitude Bars',
        'drift_stats': drift_stats_e,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel F: Self-report drift — one concept, 3 sizes
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
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
            color = MODEL_SIZE_COLORS[size]
            plot_line_with_ci(ax, turns, means, ci_lo, ci_hi,
                              color=color, label=f'LLaMA {size}')
            rho_f, p_f = stats.spearmanr(
                sub['turn'], sub['logit_rating'].dropna())
            drift_stats_f[size] = {
                'drift_magnitude': round(float(means[-1] - means[0]), 4),
                'spearman_rho_vs_turn': round(float(rho_f), 4),
                'spearman_p': float(p_f),
            }
        except Exception as e:
            print(f"    Warning: Self-report drift error for {size}: {e}")

    ax.set_xlabel('Turn')
    ax.set_ylabel('Logit Self-Report')
    ax.set_title(f'Self-Report Drift — {SHORTHAND_DISPLAY[example_concept]}')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)
    add_panel_label(ax, 'F')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_F_report_drift_across_sizes')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_F',
        'title': f'Self-Report Drift — {SHORTHAND_DISPLAY[example_concept]}',
        'drift_stats': drift_stats_f,
    })

    # Panel Fii: Self-report drift magnitude bars
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    for i, size in enumerate(LLAMA_SIZES):
        if size in drift_stats_f:
            color = MODEL_SIZE_COLORS[size]
            ax.bar(i, drift_stats_f[size]['drift_magnitude'], color=color,
                   edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(LLAMA_SIZES)))
    ax.set_xticklabels([f'LLaMA {s}' for s in LLAMA_SIZES])
    ax.set_ylabel('Drift (last − first turn)')
    ax.set_title(f'Report Drift Magnitude — {SHORTHAND_DISPLAY[example_concept]}')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    add_panel_label(ax, 'Fii')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_Fii_report_drift_magnitude_bars')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_Fii',
        'title': 'Report Drift Magnitude Bars',
        'drift_stats': drift_stats_f,
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
    add_panel_label(ax, 'G')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_G_cross_family_layer_sweep')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_G',
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
            color = cross_colors[model_name]
            plot_line_with_ci(ax, turns, means, ci_lo, ci_hi,
                              color=color, label=model_name)
            rho_h, p_h = stats.spearmanr(sub['turn'], sub['logit_rating'].dropna())
            h_drift_stats[model_name] = {
                'drift_magnitude': round(float(means[-1] - means[0]), 4),
                'spearman_rho_vs_turn': round(float(rho_h), 4),
                'spearman_p': float(p_h),
            }
        except Exception as e:
            print(f"    Warning: Cross-family drift error for {model_name}: {e}")

    ax.set_xlabel('Turn')
    ax.set_ylabel('Logit Self-Report')
    ax.set_title(f'Self-Report Drift — {CONCEPT_DISPLAY[cross_concept]}')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)
    add_panel_label(ax, 'H')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_H_cross_family_report_drift')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_H',
        'title': 'Self-Report Drift — Gemma vs Qwen',
        'drift_stats': h_drift_stats,
    })

    # ──────────────────────────────────────────────────────────────
    # Panels I–J: Scatter plots for Gemma and Qwen
    # ──────────────────────────────────────────────────────────────
    scatter_labels = ['I', 'J']
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

            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
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
                cross_scatter_stats[model_name] = {
                    'spearman_rho': round(float(rho_val), 4),
                    'spearman_p': float(p_val),
                    'isotonic_r2': round(float(r2_val), 4),
                    'n': len(probe_vals),
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
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

            cross_turnwise[model_name] = {
                'turns': turns,
                'r2': [round(v, 4) if not np.isnan(v) else None for v in r2_vals],
                'rho_flipped': [round(v, 4) if not np.isnan(v) else None for v in rho_vals],
                'rho_p': [float(p) if not np.isnan(p) else None for p in rho_p],
                'rho_sign_flipped': cross_concept in FLIP_CONCEPTS,
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
    add_panel_label(axes[0], 'K')

    axes[1].set_xlabel('Turn')
    axes[1].set_ylabel('Spearman ρ (flipped)')
    axes[1].set_title('Introspection Correlation by Turn')
    axes[1].set_xlim(0.5, 10.5)
    axes[1].set_xticks(range(1, 11))
    axes[1].set_ylim(-0.3, 1)
    axes[1].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[1].legend(fontsize=8)
    add_panel_label(axes[1], 'L')

    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_KL_cross_family_turnwise')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_K_L',
        'title': 'Turnwise Introspection — Gemma vs Qwen (Rho Flipped)',
        'description': (f'Isotonic R² (K) and Spearman ρ (L, sign-flipped for {cross_concept}) '
                        'per turn for cross-family models.'),
        'cross_turnwise': cross_turnwise,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel M: LLaMA 8B best introspection scatter
    # ──────────────────────────────────────────────────────────────
    best_concepts = ['wellbeing', 'interest']
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    best_scatter_stats = {}
    for i, short in enumerate(best_concepts):
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
                best_scatter_stats[short] = {
                    'spearman_rho': round(float(rho_val), 4),
                    'spearman_p': float(p_val),
                    'isotonic_r2': round(float(r2_val), 4),
                }
            ax.set_xlabel('Probe Score')
            if i == 0:
                ax.set_ylabel('Logit Self-Report')
            ax.set_title(f'LLaMA 8B — {SHORTHAND_DISPLAY[short]}')
        except Exception as e:
            print(f"    Warning: 8B scatter error for {short}: {e}")

    add_panel_label(axes[0], 'M')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_M_llama8b_best_scatters')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_M',
        'title': 'High-Introspection — LLaMA 8B',
        'statistics': best_scatter_stats,
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
        'drift_probe': drift_stats_e,
        'drift_report': drift_stats_f,
        'rho_matrix_steering': rho_matrix.tolist(),
    }
    save_other_stats(fig_dir, other_stats)
    print("    Figure 5 complete.")


if __name__ == '__main__':
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rdir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    generate_figure_5(rdir)
    print(f"Output -> {rdir}")
