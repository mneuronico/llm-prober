import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from shared_utils import (
    CONCEPT_DISPLAY,
    GEMMA_4B_SELF,
    QWEN_7B_SELF,
    MODEL_FAMILY_COLORS,
    load_results,
    results_to_dataframe,
    flip_if_needed,
    compute_grouped_cluster_means,
    corrected_drift_stats,
    plot_line_with_ci,
    add_panel_label,
    savefig,
    save_panel_json,
    ensure_dir,
)


def generate_cross_family_probe_drift(results_dir, concept_key='wellbeing',
                                      probe_concept='sad_vs_happy'):
    fig_dir = os.path.join(results_dir, 'Appendix_Figure_5')
    ensure_dir(fig_dir)

    cross_exp_dirs = {
        'Gemma 3 4B': GEMMA_4B_SELF.get(concept_key),
        'Qwen 2.5 7B': QWEN_7B_SELF.get(concept_key),
    }
    cross_colors = {
        'Gemma 3 4B': MODEL_FAMILY_COLORS['gemma'],
        'Qwen 2.5 7B': MODEL_FAMILY_COLORS['qwen'],
    }

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4))
    drift_stats = {}

    for model_name, exp_dir in cross_exp_dirs.items():
        if exp_dir is None or not os.path.isdir(exp_dir):
            continue

        results = load_results(exp_dir)
        df = results_to_dataframe(results, probe_name=probe_concept)
        sub = df[np.isclose(df['alpha'], 0.0)].dropna(
            subset=['conversation_index', 'turn', 'probe_score']).copy()
        sub['probe_display'] = flip_if_needed(probe_concept, sub['probe_score'].values)

        probe_mean = float(np.mean(sub['probe_display']))
        probe_sd = float(np.std(sub['probe_display'], ddof=1))
        if not np.isfinite(probe_sd) or probe_sd <= 0:
            continue

        sub['probe_display_z'] = (sub['probe_display'] - probe_mean) / probe_sd
        per_turn = compute_grouped_cluster_means(
            sub.dropna(subset=['conversation_index', 'probe_display_z']).copy(),
            'turn',
            'probe_display_z',
        )

        turns = per_turn['turn'].tolist()
        means = per_turn['mean'].tolist()
        ci_lo = per_turn['ci_low'].tolist()
        ci_hi = per_turn['ci_high'].tolist()
        color = cross_colors[model_name]
        plot_line_with_ci(ax, turns, means, ci_lo, ci_hi, color=color, label=model_name)

        rho_turn, p_turn = stats.spearmanr(sub['turn'], sub['probe_display_z'])
        corrected = corrected_drift_stats(sub, 'probe_display_z', alpha_val=0.0)
        drift_stats[model_name] = {
            'normalization_mean_raw_probe': round(probe_mean, 4),
            'normalization_sd_raw_probe': round(probe_sd, 4),
            'drift_magnitude_z': round(float(means[-1] - means[0]), 4),
            'spearman_rho_vs_turn_POOLED': round(float(rho_turn), 4),
            'spearman_p_POOLED': float(p_turn),
            'corrected_drift': corrected,
        }

    ax.set_xlabel('Turn')
    ax.set_ylabel('Normalized Probe Score')
    ax.set_title(f'Normalized Probe Drift — {CONCEPT_DISPLAY[probe_concept]}')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)
    add_panel_label(ax, 'D')
    plt.tight_layout()

    prefix = os.path.join(fig_dir, 'App_Fig_05_D_cross_family_probe_drift')
    savefig(fig, prefix)
    save_panel_json(prefix, {
        'panel_id': 'App_Fig_05_D',
        'title': 'Normalized Probe Drift — Gemma vs Qwen',
        'description': ('Probe-score drift across turns for the cross-family wellbeing '
                        'comparison. Scores are sign-corrected and z-scored within model '
                        'so Gemma and Qwen can be compared on a common scale.'),
        'drift_stats': drift_stats,
    })
    return prefix + '.png', drift_stats


if __name__ == '__main__':
    if len(sys.argv) > 1:
        out_dir = sys.argv[1]
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    panel_path, _ = generate_cross_family_probe_drift(out_dir)
    print(panel_path)
