"""
Figure 1: Probe Training Validation
====================================
Panels A–H: Layer sweeps (Cohen's d) and score distributions for all 4 concept
pairs on LLaMA-3.2-3B-Instruct, validating that trained linear probes
successfully identify interpretable directions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from shared_utils import (
    PROBES, CONCEPTS_ORDERED, CONCEPT_DISPLAY, CONCEPT_COLORS,
    load_sweep_data, load_metrics, load_concept,
    savefig, save_panel_json, save_other_stats, ensure_dir, add_panel_label,
    FLIP_CONCEPTS,
)


def _plot_layer_sweep(ax, sweep_d, best_layer, num_layers, concept_name, color):
    """Plot Cohen's d across layers with best layer highlighted."""
    layers = np.arange(num_layers)
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
    ax.set_ylim(bottom=0)


def _plot_score_dist(ax, pos_mean, pos_std, neg_mean, neg_std, concept_name,
                     n_pos, n_neg, color):
    """Plot overlapping Gaussian distributions at best layer."""
    # Determine needs-flip status for label ordering
    if concept_name in FLIP_CONCEPTS:
        # After flip: positive direction is the neg_label
        # Load concept to get labels
        high_label_mean, high_label_std = neg_mean, neg_std
        low_label_mean, low_label_std = pos_mean, pos_std
        concept = load_concept(PROBES['llama_3b'][concept_name])
        high_label = concept['neg_label'].capitalize()
        low_label = concept['pos_label'].capitalize()
    else:
        high_label_mean, high_label_std = pos_mean, pos_std
        low_label_mean, low_label_std = neg_mean, neg_std
        concept = load_concept(PROBES['llama_3b'][concept_name])
        high_label = concept['pos_label'].capitalize()
        low_label = concept['neg_label'].capitalize()

    # Generate smooth distributions
    x_min = min(low_label_mean - 4 * low_label_std,
                high_label_mean - 4 * high_label_std)
    x_max = max(low_label_mean + 4 * low_label_std,
                high_label_mean + 4 * high_label_std)
    x = np.linspace(x_min, x_max, 300)

    from matplotlib.colors import to_rgba
    c_high = to_rgba(color, alpha=0.7)
    c_low = to_rgba('#555555', alpha=0.7)

    dist_high = norm.pdf(x, high_label_mean, high_label_std)
    dist_low = norm.pdf(x, low_label_mean, low_label_std)

    ax.fill_between(x, dist_high, alpha=0.35, color=color, label=f'{high_label} (n={n_pos})')
    ax.plot(x, dist_high, color=color, linewidth=1.5)
    ax.fill_between(x, dist_low, alpha=0.25, color='#555555', label=f'{low_label} (n={n_neg})')
    ax.plot(x, dist_low, color='#555555', linewidth=1.5)

    ax.set_xlabel('Probe Score')
    ax.set_ylabel('Density')
    ax.set_title(CONCEPT_DISPLAY[concept_name])
    ax.legend(fontsize=8, loc='upper right')
    ax.set_yticks([])


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

        sweep_d = np.array(sweep['sweep_d'])
        sweep_p = np.array(sweep['sweep_p'])
        num_layers = metrics['num_layers']
        best_layer = metrics['best_layer']
        best_d = metrics['best_d']
        best_p = metrics['best_p']
        color = CONCEPT_COLORS[concept]

        # ── Panel: Layer sweep ──
        label = panel_labels[panel_idx]
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
        _plot_layer_sweep(ax, sweep_d, best_layer, num_layers, concept, color)
        add_panel_label(ax, label)
        prefix = os.path.join(fig_dir, f'Fig_01_{label}_layer_sweep_{concept}')
        savefig(fig, prefix)
        save_panel_json(prefix, {
            'panel_id': f'Fig_01_{label}',
            'title': f"Cohen's d Layer Sweep — {CONCEPT_DISPLAY[concept]}",
            'description': (f"Cohen's d for the {concept} linear probe across all "
                            f"{num_layers} layers of LLaMA-3.2-3B-Instruct. "
                            f"Best layer {best_layer} highlighted (d = {best_d:.3f}, "
                            f"p = {best_p:.2e})."),
            'data_source': os.path.join(probe_dir, 'log.jsonl'),
            'model': 'LLaMA-3.2-3B-Instruct',
            'concept': concept,
            'best_layer': best_layer,
            'best_d': round(best_d, 4),
            'best_p': best_p,
            'num_layers': num_layers,
            'sweep_d': [round(d, 4) for d in sweep_d.tolist()],
            'sweep_p': [float(f'{p:.4e}') for p in sweep_p.tolist()],
        })
        panel_idx += 1

        # ── Panel: Score distribution ──
        label = panel_labels[panel_idx]
        pos_mean = sweep['eval_pos_mean'][best_layer]
        neg_mean = sweep['eval_neg_mean'][best_layer]
        pos_std = sweep['eval_pos_std'][best_layer]
        neg_std = sweep['eval_neg_std'][best_layer]
        n_eval = metrics['n_eval']

        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
        _plot_score_dist(ax, pos_mean, pos_std, neg_mean, neg_std,
                         concept, n_eval, n_eval, color)
        add_panel_label(ax, label)
        prefix = os.path.join(fig_dir, f'Fig_01_{label}_score_dist_{concept}')
        savefig(fig, prefix)
        save_panel_json(prefix, {
            'panel_id': f'Fig_01_{label}',
            'title': f"Score Distribution — {CONCEPT_DISPLAY[concept]}",
            'description': (f"Distribution of probe scores for positive ({concept_info['pos_label']}) "
                            f"and negative ({concept_info['neg_label']}) evaluation texts at "
                            f"best layer {best_layer}. Gaussian approximation from mean/std. "
                            f"n = {n_eval} evaluation prompts per condition."),
            'data_source': os.path.join(probe_dir, 'log.jsonl'),
            'model': 'LLaMA-3.2-3B-Instruct',
            'concept': concept,
            'best_layer': best_layer,
            'pos_label': concept_info['pos_label'],
            'neg_label': concept_info['neg_label'],
            'pos_mean': round(pos_mean, 4),
            'pos_std': round(pos_std, 4),
            'neg_mean': round(neg_mean, 4),
            'neg_std': round(neg_std, 4),
            'n_eval': n_eval,
        })
        panel_idx += 1

        # Store for other_stats
        other_stats[concept] = {
            'best_layer': best_layer,
            'best_d': round(best_d, 4),
            'best_p': best_p,
            'n_train': metrics['n_train'],
            'n_eval': metrics['n_eval'],
            'num_layers': num_layers,
            'pos_label': concept_info['pos_label'],
            'neg_label': concept_info['neg_label'],
        }

    save_other_stats(fig_dir, {
        'description': ("Figure 1 validates linear probe training for 4 concept pairs "
                        "on LLaMA-3.2-3B-Instruct. Polarity note: sad_vs_happy and "
                        "impulsive_vs_planning probes have inverted polarity (positive "
                        "probe direction = sad/planning, not happy/impulsive). Signs are "
                        "flipped for visualization and analysis."),
        'model': 'LLaMA-3.2-3B-Instruct (meta-llama/llama-3.2-3b-instruct)',
        'concepts': other_stats,
    })
    print("    Figure 1 complete.")


if __name__ == '__main__':
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    rdir = os.path.join(os.path.dirname(__file__), '..', f'results_{ts}')
    generate_figure_1(rdir)
    print(f"Output → {rdir}")
