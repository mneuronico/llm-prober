"""
Figure 1: Probe Training Validation
====================================
Panels A-H: Layer sweeps (Cohen's d) and score distributions for all 4 concept
pairs on LLaMA-3.2-3B-Instruct, validating that trained linear probes
successfully identify interpretable directions.

v2 changes: boxplots from individual token scores, layer range shading,
flipped probe scores for FLIP concepts, concept example info, enhanced stats.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.stats import mannwhitneyu

from shared_utils import (
    REPO_ROOT, PROBES, CONCEPTS_ORDERED, CONCEPT_DISPLAY, CONCEPT_COLORS,
    load_sweep_data, load_metrics, load_concept, load_individual_eval_scores,
    savefig, save_panel_json, save_other_stats, ensure_dir, add_panel_label,
    FLIP_CONCEPTS, LAYER_RANGE_FRAC,
)


def _load_config(probe_dir):
    """Load the config event from log.jsonl."""
    log_path = os.path.join(probe_dir, 'log.jsonl')
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get('event') == 'config':
                    return entry.get('config', {})
            except json.JSONDecodeError:
                continue
    return {}


def _plot_layer_sweep(ax, sweep_d, best_layer, num_layers, concept_name, color):
    """Plot Cohen's d across layers with best layer highlighted and range shading."""
    layers = np.arange(num_layers)
    sweep_d = np.array(sweep_d)

    # Shade the layer search range
    lo_frac, hi_frac = LAYER_RANGE_FRAC
    lo_layer = int(np.floor(num_layers * lo_frac))
    hi_layer = int(np.ceil(num_layers * hi_frac))
    ax.axvspan(lo_layer, hi_layer, alpha=0.08, color='gray',
               label=f'Search range ({lo_frac:.0%}-{hi_frac:.0%})')

    ax.plot(layers, sweep_d, color=color, linewidth=1.5, zorder=2)
    ax.axvline(best_layer, color=color, linestyle='--', alpha=0.5, linewidth=1)
    ax.plot(best_layer, sweep_d[best_layer], 'o', color=color, markersize=8,
            zorder=3, markeredgecolor='white', markeredgewidth=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel("Cohen's d")
    ax.set_title(CONCEPT_DISPLAY[concept_name])
    ax.text(best_layer + 0.5, sweep_d[best_layer],
            f'  d = {sweep_d[best_layer]:.2f}\n  Layer {best_layer}',
            fontsize=8, va='center', color=color)
    ax.set_xlim(-0.5, num_layers - 0.5)
    # Allow negative d values (do not force ylim bottom=0)


def _plot_score_dist_boxplot(ax, probe_dir, concept_name, color, n_eval):
    """Plot boxplots of individual token scores for pos/neg evaluation prompts."""
    concept_info = load_concept(probe_dir)
    eval_data = load_individual_eval_scores(probe_dir)
    needs_flip = concept_name in FLIP_CONCEPTS

    if eval_data and (eval_data['pos_scores'] or eval_data['neg_scores']):
        all_pos = np.concatenate(eval_data['pos_scores']) if eval_data['pos_scores'] else np.array([])
        all_neg = np.concatenate(eval_data['neg_scores']) if eval_data['neg_scores'] else np.array([])

        if needs_flip:
            all_pos, all_neg = -all_pos, -all_neg
            high_label = concept_info['neg_label'].capitalize()
            low_label = concept_info['pos_label'].capitalize()
            high_data, low_data = all_neg, all_pos
        else:
            high_label = concept_info['pos_label'].capitalize()
            low_label = concept_info['neg_label'].capitalize()
            high_data, low_data = all_pos, all_neg

        bp = ax.boxplot([low_data, high_data], vert=True,
                        labels=[low_label, high_label],
                        patch_artist=True, widths=0.6,
                        showfliers=False, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='black',
                                       markeredgecolor='black', markersize=4))
        bp['boxes'][0].set_facecolor(to_rgba('#555555', 0.4))
        bp['boxes'][1].set_facecolor(to_rgba(color, 0.4))
        for element in ['whiskers', 'caps']:
            for item in bp[element]:
                item.set_color('gray')
        bp['medians'][0].set_color('#333333')
        bp['medians'][1].set_color(color)

        ax.set_ylabel('Probe Score')
        ax.set_title(CONCEPT_DISPLAY[concept_name])

        if len(high_data) > 0 and len(low_data) > 0:
            u_stat, u_p = mannwhitneyu(high_data, low_data, alternative='two-sided')
            return {
                'high_label': high_label, 'low_label': low_label,
                'high_n_eval_texts': len(high_data),
                'low_n_eval_texts': len(low_data),
                'note': ('Each eval text produces one independent score at the best layer. '
                         'Mann-Whitney U compares n_high vs n_low independent scores.'),
                'high_median': round(float(np.median(high_data)), 4),
                'low_median': round(float(np.median(low_data)), 4),
                'high_mean': round(float(np.mean(high_data)), 4),
                'low_mean': round(float(np.mean(low_data)), 4),
                'mann_whitney_U': float(u_stat), 'mann_whitney_p': float(u_p),
            }
    else:
        # Fallback: use sweep mean/std for Gaussian
        sweep = load_sweep_data(probe_dir)
        metrics = load_metrics(probe_dir)
        best_layer = metrics['best_layer']
        pos_mean = sweep['eval_pos_mean'][best_layer]
        neg_mean = sweep['eval_neg_mean'][best_layer]
        pos_std = sweep['eval_pos_std'][best_layer]
        neg_std = sweep['eval_neg_std'][best_layer]

        if needs_flip:
            pos_mean, neg_mean = -neg_mean, -pos_mean
            high_label = concept_info['neg_label'].capitalize()
            low_label = concept_info['pos_label'].capitalize()
        else:
            high_label = concept_info['pos_label'].capitalize()
            low_label = concept_info['neg_label'].capitalize()

        from scipy.stats import norm
        lo_val = min(neg_mean - 4*neg_std, pos_mean - 4*pos_std)
        hi_val = max(neg_mean + 4*neg_std, pos_mean + 4*pos_std)
        x = np.linspace(lo_val, hi_val, 300)
        ax.fill_between(x, norm.pdf(x, pos_mean, pos_std), alpha=0.35, color=color,
                        label=f'{high_label} (Gaussian)')
        ax.fill_between(x, norm.pdf(x, neg_mean, neg_std), alpha=0.25, color='#555555',
                        label=f'{low_label} (Gaussian)')
        ax.legend(fontsize=7)
        ax.set_ylabel('Density')
        ax.set_title(CONCEPT_DISPLAY[concept_name] + ' (Gaussian approx.)')
        ax.set_yticks([])
    return {}


def generate_figure_1(results_dir):
    """Generate all Figure 1 panels."""
    fig_dir = ensure_dir(os.path.join(results_dir, 'Figure_1'))
    print("  Generating Figure 1: Probe Training Validation...")
    model = 'llama_3b'
    other_stats = {}
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    panel_idx = 0

    for concept in CONCEPTS_ORDERED:
        probe_dir = PROBES[model][concept]
        sweep = load_sweep_data(probe_dir)
        metrics = load_metrics(probe_dir)
        concept_info = load_concept(probe_dir)
        config = _load_config(probe_dir)

        sweep_d = np.array(sweep['sweep_d'])
        sweep_p = np.array(sweep['sweep_p'])
        num_layers = metrics['num_layers']
        best_layer = metrics['best_layer']
        best_d = metrics['best_d']
        best_p = metrics['best_p']
        color = CONCEPT_COLORS[concept]

        lo_frac, hi_frac = LAYER_RANGE_FRAC
        lo_layer = int(np.floor(num_layers * lo_frac))
        hi_layer = int(np.ceil(num_layers * hi_frac))

        # ---- Panel: Layer sweep ----
        label = panel_labels[panel_idx]
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
        _plot_layer_sweep(ax, sweep_d, best_layer, num_layers, concept, color)
        add_panel_label(ax, label)
        prefix = os.path.join(fig_dir, f'Fig_01_{label}_layer_sweep_{concept}')
        savefig(fig, prefix)

        sig_layers = int(np.sum(np.array(sweep_p) < 0.05))
        save_panel_json(prefix, {
            'panel_id': f'Fig_01_{label}',
            'title': f"Cohen's d Layer Sweep - {CONCEPT_DISPLAY[concept]}",
            'description': (f"Cohen's d for the {concept} linear probe across all "
                            f"{num_layers} layers of LLaMA-3.2-3B-Instruct. "
                            f"Best layer {best_layer} highlighted (d = {best_d:.3f}, "
                            f"p = {best_p:.2e}). Layer search range shaded: "
                            f"layers {lo_layer}-{hi_layer} ({lo_frac:.0%}-{hi_frac:.0%})."),
            'data_source': os.path.join(probe_dir, 'log.jsonl'),
            'model': 'LLaMA-3.2-3B-Instruct',
            'concept': concept,
            'polarity_note': ('Probe direction inverted (positive = negative pole). '
                              'Scores are flipped for analysis.' if concept in FLIP_CONCEPTS
                              else 'Probe direction matches intuitive pole.'),
            'best_layer': best_layer,
            'best_d': round(best_d, 4),
            'best_p': best_p,
            'num_layers': num_layers,
            'layer_search_range': [lo_layer, hi_layer],
            'layer_search_frac': [lo_frac, hi_frac],
            'n_significant_layers_p005': sig_layers,
            'sweep_d': [round(d, 4) for d in sweep_d.tolist()],
            'sweep_p': [float(f'{p:.4e}') for p in sweep_p.tolist()],
        })
        panel_idx += 1

        # ---- Panel: Score distribution (boxplot) ----
        label = panel_labels[panel_idx]
        n_eval = metrics['n_eval']
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
        dist_stats = _plot_score_dist_boxplot(ax, probe_dir, concept, color, n_eval)
        add_panel_label(ax, label)
        prefix = os.path.join(fig_dir, f'Fig_01_{label}_score_dist_{concept}')
        savefig(fig, prefix)
        save_panel_json(prefix, {
            'panel_id': f'Fig_01_{label}',
            'title': f"Score Distribution - {CONCEPT_DISPLAY[concept]}",
            'description': (f"Boxplot of probe scores for positive "
                            f"({concept_info['pos_label']}) and negative "
                            f"({concept_info['neg_label']}) evaluation texts at "
                            f"best layer {best_layer}. One score per eval text. "
                            f"{'Scores flipped for intuitive direction. ' if concept in FLIP_CONCEPTS else ''}"
                            f"Mann-Whitney U test on {dist_stats.get('high_n_eval_texts', '?')} "
                            f"vs {dist_stats.get('low_n_eval_texts', '?')} independent scores."),
            'data_source': os.path.join(probe_dir, 'scores/'),
            'model': 'LLaMA-3.2-3B-Instruct',
            'concept': concept,
            'best_layer': best_layer,
            'pos_label': concept_info['pos_label'],
            'neg_label': concept_info['neg_label'],
            'n_eval_prompts': n_eval,
            'statistics': dist_stats,
        })
        panel_idx += 1

        # Gather example info from config
        prompts_cfg = config.get('prompts', {})
        concept_cfg = config.get('concept', {})
        other_stats[concept] = {
            'best_layer': best_layer,
            'best_d': round(best_d, 4),
            'best_p': best_p,
            'n_train': metrics['n_train'],
            'n_eval': metrics['n_eval'],
            'num_layers': num_layers,
            'proj_std_best_layer': metrics.get('proj_std_best_layer'),
            'pos_label': concept_info['pos_label'],
            'neg_label': concept_info['neg_label'],
            'polarity_flip': concept in FLIP_CONCEPTS,
            'pos_system_prompt': concept_cfg.get('pos_system',
                                                 concept_info.get('pos_system', '')),
            'neg_system_prompt': concept_cfg.get('neg_system',
                                                 concept_info.get('neg_system', '')),
            'train_questions_sample': prompts_cfg.get('train_questions', [])[:5],
            'eval_questions': prompts_cfg.get('eval_questions', []),
        }

    save_other_stats(fig_dir, {
        'description': ("Figure 1 validates linear probe training for 4 concept pairs "
                        "on LLaMA-3.2-3B-Instruct. Polarity note: sad_vs_happy and "
                        "impulsive_vs_planning probes have inverted polarity (positive "
                        "probe direction = sad/planning, not happy/impulsive). Signs are "
                        "flipped for visualization and analysis. Layer search range "
                        f"shading: {LAYER_RANGE_FRAC[0]:.0%} to {LAYER_RANGE_FRAC[1]:.0%}."),
        'model': 'LLaMA-3.2-3B-Instruct (meta-llama/llama-3.2-3b-instruct)',
        'concepts': other_stats,
    })
    print("    Figure 1 complete.")


if __name__ == '__main__':
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rdir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    generate_figure_1(rdir)
    print(f"Output -> {rdir}")
