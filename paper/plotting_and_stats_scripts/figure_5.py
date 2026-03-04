"""
Figure 5: Cross-Model Generalization
======================================
Validates findings across model sizes (LLaMA 1B/3B/8B) and families
(Gemma 3 4B, Qwen 2.5 7B). Shows scaling laws for introspection,
probe quality variation, and cross-family transferability.
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
    PROBE_METRIC_KEY, FLIP_CONCEPTS,
    load_results, load_summary, load_turnwise, load_metrics,
    load_sweep_data,
    load_model_size_r2_csv, load_model_size_trend_csv,
    load_model_size_steering_csv,
    results_to_dataframe, flip_if_needed,
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


def generate_figure_5(results_dir):
    """Generate all Figure 5 panels."""
    fig_dir = ensure_dir(os.path.join(results_dir, 'Figure_5'))
    print("  Generating Figure 5: Cross-Model Generalization...")

    other_stats = {}

    # ──────────────────────────────────────────────────────────────
    # Panel A: Probe quality heatmap (Max Cohen's d) — models × concepts
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
        'title': "Probe Quality — Max Cohen's d by Model Size and Concept",
        'description': ("Heatmap of best Cohen's d for each probe across LLaMA model "
                        "sizes. Shows that probe quality varies across concepts and "
                        "models, and identical training procedures do not guarantee "
                        "equal-quality probes."),
        'd_values': d_values,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel B: R² isotonic vs model size (4 concept curves)
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    try:
        r2_csv = load_model_size_r2_csv()
        for short in SHORTHANDS:
            concept = SHORTHAND_TO_CONCEPT[short]
            color = CONCEPT_COLORS[concept]
            sub = r2_csv[r2_csv['concept'] == short].copy()
            if sub.empty:
                continue
            # Map model names to sizes
            size_map = {'1B': 1.0, '3B': 3.0, '8B': 8.0}
            sub = sub.copy()
            sub['size'] = sub['model'].map(size_map)
            sub = sub.sort_values('size')
            ax.errorbar(sub['size'], sub['isotonic_r2_alpha0'],
                        yerr=[sub['isotonic_r2_alpha0'] - sub['ci_low'],
                              sub['ci_high'] - sub['isotonic_r2_alpha0']],
                        fmt='o-', color=color, label=SHORTHAND_DISPLAY[short],
                        capsize=4, linewidth=1.5, markersize=7)
    except Exception as e:
        print(f"    Warning: Could not load model size R2 CSV: {e}")

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
        'title': 'Isotonic R² vs. Model Size',
        'description': ('Isotonic R² at alpha=0 for each concept across LLaMA 1B/3B/8B. '
                        'Three of four concepts show monotonically increasing introspection '
                        'with model size. impulsivity shows an anomalous decrease at 8B '
                        'due to poor probe quality at that scale.'),
        'data_source': os.path.join(MODEL_SIZE_CSV_DIR,
                                    'isotonic_r2_alpha0_vs_model_size_self_steering.csv'),
    })

    # ──────────────────────────────────────────────────────────────
    # Panel C: Self-steering validation heatmap (R² × sign of slope)
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    steer_matrix = np.zeros((len(LLAMA_SIZES), len(SHORTHANDS)))
    steer_details = {}

    for si, size in enumerate(LLAMA_SIZES):
        model_key, exp_dirs = LLAMA_MODELS[size]
        steer_details[size] = {}
        for ci_idx, short in enumerate(SHORTHANDS):
            concept = SHORTHAND_TO_CONCEPT[short]
            exp_dir = exp_dirs.get(short)
            if exp_dir is None or not os.path.isdir(exp_dir):
                steer_matrix[si, ci_idx] = np.nan
                continue

            try:
                results = load_results(exp_dir)
                df = results_to_dataframe(results, probe_name=concept)

                # Compute mean rating per alpha
                alphas_found = sorted(df['alpha'].unique())
                mean_ratings = []
                for a in alphas_found:
                    sub = df[np.isclose(df['alpha'], a)]['logit_rating'].dropna()
                    mean_ratings.append(sub.mean())

                if len(alphas_found) >= 2:
                    slope, _, r, p, _ = stats.linregress(alphas_found, mean_ratings)
                    # Get R2 iso at alpha=0
                    sub0 = df[np.isclose(df['alpha'], 0.0)].dropna(
                        subset=['probe_score', 'logit_rating'])
                    probe_vals = flip_if_needed(concept, sub0['probe_score'].values)
                    r2_val = isotonic_r2(probe_vals, sub0['logit_rating'].values)
                    signed_r2 = r2_val * np.sign(slope)
                    steer_matrix[si, ci_idx] = signed_r2
                    steer_details[size][short] = {
                        'r2_iso': round(float(r2_val), 4),
                        'slope_sign': int(np.sign(slope)),
                        'signed_r2': round(float(signed_r2), 4),
                        'steering_slope': round(float(slope), 4),
                        'steering_p': float(p),
                    }
                else:
                    steer_matrix[si, ci_idx] = np.nan
            except Exception as e:
                steer_matrix[si, ci_idx] = np.nan
                print(f"    Warning: Error processing {size}/{short}: {e}")

    vmax = max(0.5, np.nanmax(np.abs(steer_matrix)))
    im = ax.imshow(steer_matrix, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
    for si in range(len(LLAMA_SIZES)):
        for ci_idx in range(len(SHORTHANDS)):
            v = steer_matrix[si, ci_idx]
            if np.isnan(v):
                ax.text(ci_idx, si, '—', ha='center', va='center', fontsize=9)
            else:
                text_color = 'white' if abs(v) > 0.6 * vmax else 'black'
                ax.text(ci_idx, si, f'{v:+.2f}', ha='center', va='center',
                        fontsize=9, color=text_color)

    ax.set_xticks(range(len(SHORTHANDS)))
    ax.set_xticklabels([SHORTHAND_DISPLAY[s] for s in SHORTHANDS],
                       fontsize=8, rotation=25, ha='right')
    ax.set_yticks(range(len(LLAMA_SIZES)))
    ax.set_yticklabels([f'LLaMA {s}' for s in LLAMA_SIZES], fontsize=9)
    ax.set_title('Self-Steering Validation\n(R² × sign(slope))')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Signed R²')
    add_panel_label(ax, 'C')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_C_self_steering_validation')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_C',
        'title': 'Self-Steering Validation Heatmap',
        'description': ('Isotonic R² multiplied by the sign of the steering slope. '
                        'Positive values: steering in the expected direction changes '
                        'self-report correctly. Negative values: inverted behavior, '
                        'indicating poor probe quality. Key finding: impulsivity at 8B '
                        'is inverted.'),
        'steer_details': steer_details,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel D: Mean R² bar chart (good-quality probes only)
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    good_r2 = {s: [] for s in LLAMA_SIZES}

    try:
        r2_csv = load_model_size_r2_csv()
        for _, row in r2_csv.iterrows():
            model = row['model']
            short = row['concept']
            r2_val = row['isotonic_r2_alpha0']
            # Check if probe is "good quality" (not inverted in steering)
            is_good = True
            if model in steer_details and short in steer_details.get(model, {}):
                if steer_details[model][short].get('slope_sign', 1) < 0:
                    is_good = False
            if is_good:
                good_r2[model].append(r2_val)
    except Exception:
        pass

    means = [np.mean(good_r2[s]) if good_r2[s] else 0 for s in LLAMA_SIZES]
    # Bootstrap CIs
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
    bars = ax.bar(range(len(LLAMA_SIZES)), means, color=colors,
                  edgecolor='white', linewidth=0.5)
    for i in range(len(LLAMA_SIZES)):
        ax.errorbar(i, means[i],
                    yerr=[[means[i] - ci_lo_list[i]], [ci_hi_list[i] - means[i]]],
                    color='black', capsize=5, linewidth=1.5, capthick=1.5)
    ax.set_xticks(range(len(LLAMA_SIZES)))
    ax.set_xticklabels([f'LLaMA {s}' for s in LLAMA_SIZES])
    ax.set_ylabel('Mean Isotonic R²')
    ax.set_title('Introspection by Model Size\n(good-quality probes only)')
    ax.set_ylim(0, 1)
    add_panel_label(ax, 'D')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_D_mean_r2_by_size')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_D',
        'title': 'Mean Isotonic R² by Model Size (Good-Quality Probes)',
        'description': ('Mean isotonic R² across concepts with validated probes only '
                        '(excluding inverted steering). Shows clear scaling trend.'),
        'good_r2_values': {s: [round(v, 4) for v in good_r2[s]] for s in LLAMA_SIZES},
        'means': [round(m, 4) for m in means],
    })

    # ──────────────────────────────────────────────────────────────
    # Panel E: Probe score drift — one concept, 3 LLaMA sizes
    # ──────────────────────────────────────────────────────────────
    example_concept = 'wellbeing'
    concept_name = SHORTHAND_TO_CONCEPT[example_concept]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
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
        except Exception as e:
            print(f"    Warning: Drift plot error for {size}: {e}")

    ax.set_xlabel('Turn')
    ax.set_ylabel('Probe Score')
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
        'title': f'Probe Score Drift Across Model Sizes — {SHORTHAND_DISPLAY[example_concept]}',
        'description': (f'Mean probe score ({concept_name}, flipped) across turns for '
                        f'LLaMA 1B/3B/8B at alpha=0. Drift is visible in all sizes '
                        f'but cleaner in larger models.'),
        'concept': concept_name,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel F: Self-report drift — one concept, 3 LLaMA sizes
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
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
        except Exception as e:
            print(f"    Warning: Self-report drift error for {size}: {e}")

    ax.set_xlabel('Turn')
    ax.set_ylabel('Logit Self-Report')
    ax.set_title(f'Self-Report Drift Across Model Sizes — {SHORTHAND_DISPLAY[example_concept]}')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)
    add_panel_label(ax, 'F')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_F_report_drift_across_sizes')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_F',
        'title': f'Self-Report Drift Across Model Sizes — {SHORTHAND_DISPLAY[example_concept]}',
        'description': ('Mean logit-based self-report across turns for LLaMA 1B/3B/8B. '
                        'Drift effect generalizes across model sizes.'),
        'concept': concept_name,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel G: Layer sweeps for Gemma 4B and Qwen 7B (sad_vs_happy)
    # ──────────────────────────────────────────────────────────────
    cross_concept = 'sad_vs_happy'
    cross_models = [
        ('Gemma 3 4B', 'gemma_4b', MODEL_FAMILY_COLORS['gemma']),
        ('Qwen 2.5 7B', 'qwen_7b', MODEL_FAMILY_COLORS['qwen']),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sweep_info = {}
    for model_name, model_key, color in cross_models:
        try:
            probe_dir = PROBES[model_key][cross_concept]
            sweep = load_sweep_data(probe_dir)
            metrics = load_metrics(probe_dir)
            sweep_d = np.array(sweep['sweep_d'])
            num_layers = metrics['num_layers']
            best_layer = metrics['best_layer']
            best_d = metrics['best_d']

            layers = np.arange(num_layers)
            ax.plot(layers, sweep_d, color=color, linewidth=1.5,
                    label=f'{model_name} (d={best_d:.2f}, L{best_layer})')
            ax.plot(best_layer, sweep_d[best_layer], 'o', color=color,
                    markersize=7, markeredgecolor='white', markeredgewidth=1)
            sweep_info[model_name] = {
                'best_layer': best_layer,
                'best_d': round(best_d, 3),
                'num_layers': num_layers,
            }
        except Exception as e:
            print(f"    Warning: Sweep error for {model_name}: {e}")

    ax.set_xlabel('Layer')
    ax.set_ylabel("Cohen's d")
    ax.set_title(f"Layer Sweep — {CONCEPT_DISPLAY[cross_concept]}")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    add_panel_label(ax, 'G')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_G_cross_family_layer_sweep')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_G',
        'title': f'Layer Sweep — Gemma 3 4B vs Qwen 2.5 7B ({CONCEPT_DISPLAY[cross_concept]})',
        'description': ("Cohen's d layer sweep for sad_vs_happy in two non-LLaMA "
                        "families. Qwen shows much stronger separation (d≈3.5) vs "
                        "Gemma (d≈1.8)."),
        'sweep_info': sweep_info,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel H: Self-report drift for Gemma and Qwen
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
        'title': f'Self-Report Drift — Gemma vs Qwen',
        'description': 'Self-report drift across turns for non-LLaMA models (sad_vs_happy).',
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
                'title': f'Introspection Scatter — {model_name}',
                'description': (f'Probe score vs logit self-report for {model_name} '
                                f'({cross_concept}). Shows introspection capability in '
                                f'non-LLaMA families.'),
                'statistics': cross_scatter_stats.get(model_name, {}),
            })
        except Exception as e:
            print(f"    Warning: Cross-family scatter error for {model_name}: {e}")

    # ──────────────────────────────────────────────────────────────
    # Panels K–L: Turn-wise R² and Rho for Gemma and Qwen
    # ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cross_turnwise = {}

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

            plot_line_with_ci(axes[0], turns, r2_vals, r2_lo, r2_hi,
                              color=color, label=model_name)
            plot_line_with_ci(axes[1], turns, rho_vals, rho_lo, rho_hi,
                              color=color, label=model_name)

            cross_turnwise[model_name] = {
                'turns': turns,
                'r2': [round(v, 4) if not np.isnan(v) else None for v in r2_vals],
                'rho': [round(v, 4) if not np.isnan(v) else None for v in rho_vals],
                'rho_p': [float(p) if not np.isnan(p) else None for p in rho_p],
                'n_significant_turns': sum(1 for p in rho_p if p < 0.05),
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
    axes[1].set_ylabel('Spearman ρ')
    axes[1].set_title('Introspection Correlation by Turn')
    axes[1].set_xlim(0.5, 10.5)
    axes[1].set_xticks(range(1, 11))
    axes[1].set_ylim(-0.3, 1)
    axes[1].legend(fontsize=8)
    add_panel_label(axes[1], 'L')

    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_KL_cross_family_turnwise')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_K_L',
        'title': 'Turn-wise Introspection — Gemma vs Qwen',
        'description': ('Isotonic R² (K) and Spearman ρ (L) per turn for Gemma 3 4B '
                        'and Qwen 2.5 7B (sad_vs_happy). Qwen shows strong, significant '
                        'introspection at every turn with R²(iso) starting at ~0.9 and '
                        'decreasing. Gemma shows minimal effect due to low probe and '
                        'report quality.'),
        'cross_turnwise': cross_turnwise,
    })

    # ──────────────────────────────────────────────────────────────
    # Panel M: High-introspection scatter plots (LLaMA 8B best cases)
    # ──────────────────────────────────────────────────────────────
    best_concepts = ['wellbeing', 'interest']  # R2 ~0.9 at 8B
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
                ax.plot(probe_vals[sort_idx], y_pred, color='black',
                        linewidth=2, alpha=0.7)
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
            print(f"    Warning: LLaMA 8B scatter error for {short}: {e}")

    add_panel_label(axes[0], 'M')
    plt.tight_layout()
    prefix = os.path.join(fig_dir, 'Fig_05_M_llama8b_best_scatters')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'Fig_05_M',
        'title': 'High-Introspection Examples — LLaMA 8B',
        'description': ('Scatter plots for the two concepts with highest R²(iso) at '
                        'LLaMA 8B (wellbeing and interest, both ≈0.9). Demonstrates '
                        'that introspection can be excellent in larger models.'),
        'statistics': best_scatter_stats,
    })

    # ── Save other_stats ──
    # Random scoring control comparison
    random_stats = {}
    try:
        from shared_utils import LLAMA_3B_RANDOM
        for short in SHORTHANDS:
            concept = SHORTHAND_TO_CONCEPT[short]
            rand_dir = LLAMA_3B_RANDOM.get(short)
            real_dir = LLAMA_3B_4X4_SELF.get(short)
            if rand_dir and os.path.isdir(rand_dir) and real_dir and os.path.isdir(real_dir):
                # Random probe
                rand_results = load_results(rand_dir)
                rand_df = results_to_dataframe(rand_results, probe_name=concept)
                rand_sub = rand_df[np.isclose(rand_df['alpha'], 0.0)].copy()
                rand_sub['probe_display'] = flip_if_needed(
                    concept, rand_sub['probe_score'].values)
                rand_rho, rand_p = stats.spearmanr(
                    rand_sub['turn'], rand_sub['probe_display'].dropna())
                rand_drift = float(rand_sub.groupby('turn')['probe_display'].mean().iloc[-1] -
                                   rand_sub.groupby('turn')['probe_display'].mean().iloc[0])

                # Real probe
                real_results = load_results(real_dir)
                real_df = results_to_dataframe(real_results, probe_name=concept)
                real_sub = real_df[np.isclose(real_df['alpha'], 0.0)].copy()
                real_sub['probe_display'] = flip_if_needed(
                    concept, real_sub['probe_score'].values)
                real_rho, real_p = stats.spearmanr(
                    real_sub['turn'], real_sub['probe_display'].dropna())
                real_drift = float(real_sub.groupby('turn')['probe_display'].mean().iloc[-1] -
                                   real_sub.groupby('turn')['probe_display'].mean().iloc[0])

                random_stats[short] = {
                    'real_probe_drift': round(real_drift, 4),
                    'real_probe_rho_vs_turn': round(float(real_rho), 4),
                    'real_probe_p_vs_turn': float(real_p),
                    'random_probe_drift': round(rand_drift, 4),
                    'random_probe_rho_vs_turn': round(float(rand_rho), 4),
                    'random_probe_p_vs_turn': float(rand_p),
                    'drift_ratio': round(abs(real_drift / rand_drift), 2) if rand_drift != 0 else None,
                }
    except Exception as e:
        print(f"    Warning: Random scoring comparison error: {e}")

    # Trend significance
    try:
        trend_csv = load_model_size_trend_csv()
        trend_stats = trend_csv.to_dict('records')
    except Exception:
        trend_stats = []

    other_stats = {
        'description': ('Figure 5 validates generalization. Key findings: '
                        '(1) Introspection scales with model size for 3/4 concepts. '
                        '(2) The anomalous concept (impulsivity at 8B) has inverted self-steering. '
                        '(3) Qwen 2.5 7B shows excellent introspection (R²(iso)~0.9 at turn 1). '
                        '(4) Gemma 3 4B shows weak effects due to low probe and report quality. '
                        '(5) Random probe controls show much smaller drift than trained probes.'),
        'random_vs_trained_drift': random_stats,
        'scaling_trend_significance': trend_stats,
        'cross_family_introspection': cross_scatter_stats,
        'cross_family_turnwise': cross_turnwise,
        'llama_8b_best_introspection': best_scatter_stats,
    }
    save_other_stats(fig_dir, other_stats)
    print("    Figure 5 complete.")


if __name__ == '__main__':
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rdir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    generate_figure_5(rdir)
    print(f"Output → {rdir}")
