"""
Shared utilities for paper figure generation.
Provides data loading, statistical computation, plotting helpers, and constants.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType for editable PDF text
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# ──────────────────── PATHS ────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ──────── Probe output directories ────────
PROBES = {
    'llama_3b': {
        'sad_vs_happy': os.path.join(REPO_ROOT, 'outputs', 'sad_vs_happy', '20260106_181019'),
        'bored_vs_interested': os.path.join(REPO_ROOT, 'outputs', 'bored_vs_interested', '20260122_112208'),
        'distracted_vs_focused': os.path.join(REPO_ROOT, 'outputs', 'distracted_vs_focused', '20260122_095601'),
        'impulsive_vs_planning': os.path.join(REPO_ROOT, 'outputs', 'impulsive_vs_planning', '20260120_181954'),
    },
    'llama_1b': {
        'sad_vs_happy': os.path.join(REPO_ROOT, 'outputs', 'sad_vs_happy', '20260222_114144'),
        'bored_vs_interested': os.path.join(REPO_ROOT, 'outputs', 'bored_vs_interested', '20260222_113022'),
        'distracted_vs_focused': os.path.join(REPO_ROOT, 'outputs', 'distracted_vs_focused', '20260222_113736'),
        'impulsive_vs_planning': os.path.join(REPO_ROOT, 'outputs', 'impulsive_vs_planning', '20260222_113350'),
    },
    'llama_8b': {
        'sad_vs_happy': os.path.join(REPO_ROOT, 'outputs', 'sad_vs_happy', '20260223_114426'),
        'bored_vs_interested': os.path.join(REPO_ROOT, 'outputs', 'bored_vs_interested', '20260223_121414'),
        'distracted_vs_focused': os.path.join(REPO_ROOT, 'outputs', 'distracted_vs_focused', '20260223_122930'),
        'impulsive_vs_planning': os.path.join(REPO_ROOT, 'outputs', 'impulsive_vs_planning', '20260223_122141'),
    },
    'gemma_4b': {
        'sad_vs_happy': os.path.join(REPO_ROOT, 'outputs', 'sad_vs_happy_gemma_3_4b-it', '20260225_105047'),
    },
    'qwen_7b': {
        'sad_vs_happy': os.path.join(REPO_ROOT, 'outputs', 'sad_vs_happy_qwen25_7b-instruct', '20260227_180321'),
    },
}

# ──────── Experiment directories ────────
# LLaMA 3B reruns (preferred for deep analysis)
LLAMA_3B_RERUN_SELF = {
    'wellbeing': os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4-reruns',
                              'wellbeing_self_steering_alpha_m4_to_p4_20260224_163740'),
    'interest': os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4-reruns',
                             'interest_self_steering_alpha_m4_to_p4_20260224_163740'),
    'focus': os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4-reruns',
                          'focus_self_steering_alpha_m4_to_p4_20260224_163740'),
    'impulsivity': os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4-reruns',
                                'impulsivity_self_steering_alpha_m4_to_p4_20260224_163740'),
}

# LLaMA 3B 4x4 self-steering (for cross-size comparison with 1B/8B)
LLAMA_3B_4X4_SELF = {
    'wellbeing': os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
                              'wellbeing_self_steering_alpha_m4_to_p4_20260220_142358'),
    'interest': os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
                             'interest_self_steering_alpha_m4_to_p4_20260221_134238'),
    'focus': os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
                          'focus_self_steering_alpha_m4_to_p4_20260220_142358'),
    'impulsivity': os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
                                'impulsivity_self_steering_alpha_m4_to_p4_20260221_134238'),
}

# LLaMA 3B 4x4 matrix cross-steering experiments (all 16)
LLAMA_3B_4X4_ALL = {
    ('focus', 'focus'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'focus_self_steering_alpha_m4_to_p4_20260220_142358'),
    ('focus', 'impulsivity'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'focus_steers_impulsivity_cross_alpha_m4_to_p4_20260220_142358'),
    ('focus', 'interest'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'focus_steers_interest_cross_alpha_m4_to_p4_20260220_142358'),
    ('focus', 'wellbeing'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'focus_steers_wellbeing_focus_wellbeing_cross_20260214_121258'),
    ('impulsivity', 'impulsivity'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'impulsivity_self_steering_alpha_m4_to_p4_20260221_134238'),
    ('impulsivity', 'focus'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'impulsivity_steers_focus_cross_alpha_m4_to_p4_20260221_134238'),
    ('impulsivity', 'interest'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'impulsivity_steers_interest_cross_alpha_m4_to_p4_20260221_134238'),
    ('impulsivity', 'wellbeing'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'impulsivity_steers_wellbeing_cross_alpha_m4_to_p4_20260221_134238'),
    ('interest', 'interest'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'interest_self_steering_alpha_m4_to_p4_20260221_134238'),
    ('interest', 'focus'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'interest_steers_focus_cross_alpha_m4_to_p4_20260221_134238'),
    ('interest', 'impulsivity'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'interest_steers_impulsivity_cross_alpha_m4_to_p4_20260221_134238'),
    ('interest', 'wellbeing'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'interest_steers_wellbeing_cross_alpha_m4_to_p4_20260221_134238'),
    ('wellbeing', 'wellbeing'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'wellbeing_self_steering_alpha_m4_to_p4_20260220_142358'),
    ('wellbeing', 'focus'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'wellbeing_steers_focus_focus_wellbeing_cross_20260214_121258'),
    ('wellbeing', 'impulsivity'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'wellbeing_steers_impulsivity_cross_alpha_m4_to_p4_20260221_134238'),
    ('wellbeing', 'interest'): os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
        'wellbeing_steers_interest_cross_alpha_m4_to_p4_20260221_134238'),
}

# Rerun-updated map: for the 6 reruns, override the 4x4 entries
LLAMA_3B_4X4_WITH_RERUNS = dict(LLAMA_3B_4X4_ALL)
LLAMA_3B_4X4_WITH_RERUNS[('wellbeing', 'wellbeing')] = LLAMA_3B_RERUN_SELF['wellbeing']
LLAMA_3B_4X4_WITH_RERUNS[('interest', 'interest')] = LLAMA_3B_RERUN_SELF['interest']
LLAMA_3B_4X4_WITH_RERUNS[('focus', 'focus')] = LLAMA_3B_RERUN_SELF['focus']
LLAMA_3B_4X4_WITH_RERUNS[('impulsivity', 'impulsivity')] = LLAMA_3B_RERUN_SELF['impulsivity']
LLAMA_3B_4X4_WITH_RERUNS[('wellbeing', 'impulsivity')] = os.path.join(
    REPO_ROOT, 'analysis', 'llama-3b-4by4-reruns',
    'wellbeing_steers_impulsivity_cross_alpha_m4_to_p4_20260224_163740')
LLAMA_3B_4X4_WITH_RERUNS[('wellbeing', 'interest')] = os.path.join(
    REPO_ROOT, 'analysis', 'llama-3b-4by4-reruns',
    'wellbeing_steers_interest_cross_alpha_m4_to_p4_20260224_163740')

# Matrix analysis CSV directory
MATRIX_ANALYSIS_DIR = os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4',
                                   'matrix_analysis_with_rerun6_20260225_105534')

# Greedy experiments for LLaMA 3B
LLAMA_3B_GREEDY = {}
_greedy_base = os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4-reruns-token-primary-greedy')
for _short in ['wellbeing', 'interest', 'focus', 'impulsivity']:
    # Prefer later timestamp
    for _ts in ['20260303_172502', '20260303_172030']:
        _cand = os.path.join(_greedy_base,
                             f'{_short}_no_steering_3b_alpha0_greedy_token_primary_{_ts}')
        if os.path.isdir(_cand):
            LLAMA_3B_GREEDY[_short] = _cand
            break

# Random scoring controls for LLaMA 3B
LLAMA_3B_RANDOM = {}
_rand_base = os.path.join(REPO_ROOT, 'analysis', 'llama-3b-4by4-reruns-random-scoring')
for _short in ['wellbeing', 'interest', 'focus', 'impulsivity']:
    _cand = os.path.join(_rand_base,
                         f'{_short}_no_steering_3b_alpha0_random_scoring_20260227_232941')
    if os.path.isdir(_cand):
        LLAMA_3B_RANDOM[_short] = _cand


def _discover_self_steering_dirs(base_dir, model_tag=None):
    """Discover self-steering experiment directories within a base directory."""
    mapping = {}
    if not os.path.isdir(base_dir):
        return mapping
    shorthand_map = {
        'wellbeing': 'wellbeing', 'interest': 'interest',
        'focus': 'focus', 'impulsivity': 'impulsivity',
    }
    for d in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, d)
        if not os.path.isdir(full):
            continue
        for short in shorthand_map:
            if d.startswith(f'{short}_') and 'self_steering' in d:
                mapping[short] = full
                break
            elif d.startswith(f'{short}_') and 'no_steering' in d and 'alpha0' in d:
                mapping[short] = full
                break
    return mapping


# LLaMA 1B / 8B self-steering
LLAMA_1B_SELF = _discover_self_steering_dirs(
    os.path.join(REPO_ROOT, 'analysis', 'llama-1b-self-steering'))
LLAMA_8B_SELF = _discover_self_steering_dirs(
    os.path.join(REPO_ROOT, 'analysis', 'llama-8b-self-steering'))

# Gemma 4B
GEMMA_4B_SELF = _discover_self_steering_dirs(
    os.path.join(REPO_ROOT, 'analysis', 'gemma-4b-self-steering'))

# Qwen 7B
QWEN_7B_DIR = os.path.join(REPO_ROOT, 'analysis',
                            'wellbeing_no_steering_qwen25_7b_alpha0_20260227_212109')
QWEN_7B_SELF = {'wellbeing': QWEN_7B_DIR} if os.path.isdir(QWEN_7B_DIR) else {}

# Model-size comparison CSVs
MODEL_SIZE_CSV_DIR = os.path.join(REPO_ROOT, 'analysis', 'llama-model_size_comparison')
CROSS_FAMILY_CSV_DIR = os.path.join(REPO_ROOT, 'analysis', 'cross-family-comparison')


# ──────────────────── CONCEPT CONSTANTS ────────────────────
# Mapping: shorthand → probe concept name
SHORTHAND_TO_CONCEPT = {
    'wellbeing': 'sad_vs_happy',
    'interest': 'bored_vs_interested',
    'focus': 'distracted_vs_focused',
    'impulsivity': 'impulsive_vs_planning',
}
CONCEPT_TO_SHORTHAND = {v: k for k, v in SHORTHAND_TO_CONCEPT.items()}

# Display names for plots
CONCEPT_DISPLAY = {
    'sad_vs_happy': 'Happy / Sad',
    'bored_vs_interested': 'Interested / Bored',
    'distracted_vs_focused': 'Focused / Distracted',
    'impulsive_vs_planning': 'Impulsive / Planning',
}
SHORTHAND_DISPLAY = {
    'wellbeing': 'Happy / Sad',
    'interest': 'Interested / Bored',
    'focus': 'Focused / Distracted',
    'impulsivity': 'Impulsive / Planning',
}

# Concepts that need probe score sign flip (probe positive ≠ self-report positive)
FLIP_CONCEPTS = {'sad_vs_happy', 'impulsive_vs_planning'}
FLIP_SHORTHANDS = {'wellbeing', 'impulsivity'}

# Main ordered concept list
CONCEPTS_ORDERED = ['sad_vs_happy', 'bored_vs_interested',
                    'distracted_vs_focused', 'impulsive_vs_planning']
SHORTHANDS_ORDERED = ['wellbeing', 'interest', 'focus', 'impulsivity']

# Probe metric used for introspection analysis
PROBE_METRIC_KEY = 'prompt_assistant_last_mean'

# Layer range fraction used for best-layer search shading (lo, hi)
LAYER_RANGE_FRAC = (0.20, 0.80)


# ──────────────────── COLORS ────────────────────
CONCEPT_COLORS = {
    'sad_vs_happy': '#E67E22',
    'bored_vs_interested': '#27AE60',
    'distracted_vs_focused': '#2980B9',
    'impulsive_vs_planning': '#C0392B',
}
SHORTHAND_COLORS = {CONCEPT_TO_SHORTHAND[k]: v for k, v in CONCEPT_COLORS.items()}

# Alpha color scale (dark→light for -4→+4)
ALPHA_COLORS = {
    -4.0: '#1a1a6c',
    -2.0: '#4a4aad',
    0.0:  '#2c2c2c',
    2.0:  '#ad4a4a',
    4.0:  '#c91818',
}
ALPHA_CMAP_NAME = 'RdBu_r'

MODEL_FAMILY_COLORS = {
    'gemma': '#8E44AD',
    'qwen': '#16A085',
}
MODEL_SIZE_COLORS = {
    '1B': '#6BAED6',
    '3B': '#2171B5',
    '8B': '#08306B',
}


# ──────────────────── STYLE SETUP ────────────────────
def setup_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.framealpha': 0.8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
    })

setup_style()


def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=14):
    """Add a bold panel label (A, B, C...) to an axis."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left')


# ──────────────────── DATA LOADING ────────────────────
_json_cache = {}


def _load_json_cached(path):
    """Load JSON with caching."""
    path = os.path.normpath(path)
    if path not in _json_cache:
        with open(path, 'r', encoding='utf-8') as f:
            _json_cache[path] = json.load(f)
    return _json_cache[path]


def load_results(experiment_dir):
    """Load results.json from an experiment directory → list of dicts."""
    return _load_json_cached(os.path.join(experiment_dir, 'results.json'))


def load_summary(experiment_dir):
    """Load summary.json → dict."""
    return _load_json_cached(os.path.join(experiment_dir, 'summary.json'))


def load_turnwise(experiment_dir):
    """Load turnwise_relationship_vs_alpha.json → dict."""
    return _load_json_cached(os.path.join(experiment_dir,
                                          'turnwise_relationship_vs_alpha.json'))


def load_metrics(probe_dir):
    """Load metrics.json from a probe output directory → dict."""
    return _load_json_cached(os.path.join(probe_dir, 'metrics.json'))


def load_concept(probe_dir):
    """Load concept.json → dict."""
    return _load_json_cached(os.path.join(probe_dir, 'concept.json'))


def load_sweep_data(probe_dir):
    """Parse log.jsonl to extract sweep_full event data."""
    log_path = os.path.join(probe_dir, 'log.jsonl')
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get('event') == 'sweep_full':
                    return entry
            except json.JSONDecodeError:
                continue
    raise FileNotFoundError(f"No sweep_full event in {log_path}")


def load_matrix_r2_csv():
    """Load the 4x4 R2 isotonic matrix CSV."""
    path = os.path.join(MATRIX_ANALYSIS_DIR, 'isotonic_r2_values_by_cell_alpha.csv')
    return pd.read_csv(path)


def load_matrix_increase_csv():
    """Load the max-increase matrix CSV."""
    path = os.path.join(MATRIX_ANALYSIS_DIR, 'max_increase_vs_alpha0_by_cell.csv')
    return pd.read_csv(path)


def load_matrix_summary():
    """Load matrix analysis summary JSON."""
    path = os.path.join(MATRIX_ANALYSIS_DIR, 'matrix_analysis_summary.json')
    return _load_json_cached(path)


def load_model_size_r2_csv():
    """Load isotonic R2 vs model size CSV."""
    path = os.path.join(MODEL_SIZE_CSV_DIR,
                        'isotonic_r2_alpha0_vs_model_size_self_steering.csv')
    return pd.read_csv(path)


def load_model_size_trend_csv():
    """Load isotonic R2 trend significance CSV."""
    path = os.path.join(MODEL_SIZE_CSV_DIR,
                        'isotonic_r2_alpha0_trend_significance.csv')
    return pd.read_csv(path)


def load_model_size_steering_csv():
    """Load self-steering max increase by model size CSV."""
    path = os.path.join(MODEL_SIZE_CSV_DIR,
                        'self_steering_max_isotonic_r2_increase_vs_alpha0_by_model_size.csv')
    return pd.read_csv(path)


def load_individual_eval_scores(probe_dir, best_layer=None):
    """
    Load individual per-token probe scores from npz files in the scores/ directory.
    Returns dict with keys 'pos_scores' and 'neg_scores', each a list of arrays
    (one array of generation-token scores per evaluation prompt).
    If best_layer is None, reads it from metrics.json.
    """
    if best_layer is None:
        metrics = load_metrics(probe_dir)
        best_layer = metrics['best_layer']

    scores_dir = os.path.join(probe_dir, 'scores')
    if not os.path.isdir(scores_dir):
        return None

    # Read the config to get alpha values and determine pos/neg
    log_path = os.path.join(probe_dir, 'log.jsonl')
    score_prompts = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get('event') == 'score_prompt':
                    score_prompts.append(entry)
            except json.JSONDecodeError:
                continue

    # Group by alpha: alpha > 0 → "positive" condition; alpha == 0 / baseline
    # Actually the structure is: score_prompt events with alpha and npz_path
    # We want the alpha=0 evaluation scores, split by system prompt direction
    concept_info = load_concept(probe_dir)
    pos_system = concept_info.get('pos_system', '')
    neg_system = concept_info.get('neg_system', '')

    pos_scores = []
    neg_scores = []
    for sp in score_prompts:
        npz_path = sp.get('npz_path')
        if not npz_path:
            continue
        full_npz = os.path.join(REPO_ROOT, npz_path) if not os.path.isabs(npz_path) else npz_path
        if not os.path.exists(full_npz):
            continue
        try:
            npz = np.load(full_npz, allow_pickle=True)
            prompt_len = int(npz['prompt_len'][0])
            all_scores = npz['scores_agg']
            gen_scores = all_scores[prompt_len:]  # generation tokens only
            alpha = sp.get('alpha', 0.0)
            system = sp.get('system_prompt', '')

            if system == pos_system and np.isclose(alpha, 0.0):
                pos_scores.append(gen_scores)
            elif system == neg_system and np.isclose(alpha, 0.0):
                neg_scores.append(gen_scores)
        except Exception:
            continue

    # If pos_system == neg_system (neutral system), split by alpha sign
    if pos_system == neg_system:
        pos_scores = []
        neg_scores = []
        for sp in score_prompts:
            npz_path = sp.get('npz_path')
            if not npz_path:
                continue
            full_npz = os.path.join(REPO_ROOT, npz_path) if not os.path.isabs(npz_path) else npz_path
            if not os.path.exists(full_npz):
                continue
            try:
                npz = np.load(full_npz, allow_pickle=True)
                prompt_len = int(npz['prompt_len'][0])
                gen_scores = npz['scores_agg'][prompt_len:]
                alpha = sp.get('alpha', 0.0)
                if alpha > 0:
                    pos_scores.append(gen_scores)
                elif alpha < 0:
                    neg_scores.append(gen_scores)
            except Exception:
                continue

    return {'pos_scores': pos_scores, 'neg_scores': neg_scores}


# ──────────────────── DATA EXTRACTION ────────────────────
def flip_if_needed(concept_name, values):
    """Flip probe scores by -1 for concepts where polarity is reversed."""
    values = np.asarray(values, dtype=float)
    if concept_name in FLIP_CONCEPTS:
        return -values
    return values


def flip_alpha_if_needed(concept_or_shorthand, alpha_values):
    """Negate alpha values for display when concept has FLIP polarity.
    For FLIP concepts, positive alpha steers toward the NEGATIVE pole,
    so we negate alpha for intuitive display."""
    alpha_values = np.asarray(alpha_values, dtype=float)
    if concept_or_shorthand in FLIP_CONCEPTS or concept_or_shorthand in FLIP_SHORTHANDS:
        return -alpha_values
    return alpha_values


def flip_alpha_scalar(concept_or_shorthand, alpha):
    """Negate a single alpha value for display if concept has FLIP polarity."""
    if concept_or_shorthand in FLIP_CONCEPTS or concept_or_shorthand in FLIP_SHORTHANDS:
        return -alpha
    return alpha


def flip_if_needed_shorthand(shorthand, values):
    """Flip probe scores by -1 using shorthand name."""
    values = np.asarray(values, dtype=float)
    if shorthand in FLIP_SHORTHANDS:
        return -values
    return values


def _find_probe_key(pm_dict, probe_name):
    """Find the matching probe key in probe_metrics, allowing model-suffixed names."""
    if probe_name in pm_dict:
        return probe_name
    # Try fuzzy match: e.g. 'sad_vs_happy' matches 'sad_vs_happy_gemma_3_4b-it'
    for k in pm_dict:
        if k.startswith(probe_name) or probe_name.startswith(k):
            return k
    # Fallback to first key
    return list(pm_dict.keys())[0] if pm_dict else None


def results_to_dataframe(results, probe_name=None, metric_key=PROBE_METRIC_KEY):
    """
    Convert results list to a pandas DataFrame with key columns.
    probe_name: which probe to extract scores for (e.g., 'sad_vs_happy').
                If None, tries to find the first available probe.
    Handles model-suffixed probe names (e.g. 'sad_vs_happy_gemma_3_4b-it').
    """
    rows = []
    resolved_name = None
    for r in results:
        row = {
            'conversation_index': r['conversation_index'],
            'turn_index': r['turn_index'],
            'turn': r['turn_index'],  # already 1-indexed in data
            'alpha': r['alpha'],
            'token_rating': r.get('token_rating'),
            'logit_rating': r.get('logit_rating'),
            'rating': r.get('rating'),
        }
        # Extract probe score
        pm = r.get('probe_metrics', {})
        if resolved_name is None:
            pname = probe_name if probe_name else None
            resolved_name = _find_probe_key(pm, pname) if pname else (list(pm.keys())[0] if pm else None)
        if resolved_name and resolved_name in pm:
            row['probe_score'] = pm[resolved_name].get(metric_key)
            row['probe_completion_mean'] = pm[resolved_name].get('completion_assistant_mean')
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def get_turnwise_stats(experiment_dir, probe_name, metric_key=PROBE_METRIC_KEY,
                       source='logit', alpha=0.0):
    """
    Extract per-turn correlation stats from turnwise_relationship_vs_alpha.json.
    Returns dict {turn_number_str: stats_dict}.
    The raw data stored is alpha -> list-of-dicts (each has 'turn' key).
    """
    tw = load_turnwise(experiment_dir)
    try:
        per_probe = tw['per_source'][source]['per_probe']
        # Try exact match first, then fuzzy match for model-suffixed names
        if probe_name in per_probe:
            data = per_probe[probe_name][metric_key]
        else:
            # Try to find a key that starts with the concept name
            matched = None
            for k in per_probe:
                if k.startswith(probe_name.split('_')[0]) or probe_name.startswith(k.split('_')[0]):
                    matched = k
                    break
            if matched is None:
                matched = list(per_probe.keys())[0]  # fallback to first
            data = per_probe[matched][metric_key]
    except KeyError:
        return {}
    alpha_key = str(float(alpha))
    raw = data.get('stats_by_alpha_turn', {}).get(alpha_key)
    if raw is None:
        alpha_key = str(int(alpha)) if alpha == int(alpha) else alpha_key
        raw = data.get('stats_by_alpha_turn', {}).get(alpha_key)
    if raw is None:
        return {}
    # Convert list format to dict keyed by turn number string
    if isinstance(raw, list):
        return {str(entry['turn']): entry for entry in raw}
    return raw


def get_summary_stats(experiment_dir, probe_name, alpha=0.0, source='logit'):
    """
    Extract aggregated stats from summary.json for a given probe/alpha/source.
    Returns the rating_vs_turn_trend_bootstrap dict.
    """
    summ = load_summary(experiment_dir)
    try:
        alpha_key = str(float(alpha))
        probe_data = summ['per_probe'][probe_name]['per_alpha'][alpha_key]
        return probe_data['rating_sources'][source].get(
            'rating_vs_turn_trend_bootstrap', {})
    except KeyError:
        return {}


def get_rating_vs_alpha(experiment_dir, probe_name, source='logit'):
    """
    Get mean rating for each alpha from summary.json.
    Returns dict {alpha: mean_rating}.
    """
    summ = load_summary(experiment_dir)
    result = {}
    try:
        for alpha_key, adata in summ['per_probe'][probe_name]['per_alpha'].items():
            bootstrap = adata['rating_sources'][source].get(
                'rating_vs_turn_trend_bootstrap', {})
            # Use intercept + slope * mean_turn as approximate mean
            # Or just compute from results
            result[float(alpha_key)] = bootstrap
    except KeyError:
        pass
    return result


# ──────────────────── STATISTICS ────────────────────
def isotonic_r2(x, y):
    """Compute R² from isotonic regression."""
    from sklearn.isotonic import IsotonicRegression
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    ir = IsotonicRegression(out_of_bounds='clip')
    y_pred = ir.fit_transform(x, y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def bootstrap_stat(x, y, stat_fn, n_bootstrap=1000, ci=0.95, seed=42):
    """
    Bootstrap a statistic computed on paired (x, y) data.
    stat_fn(x, y) → scalar.
    Returns (point_estimate, ci_low, ci_high).
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    point = stat_fn(x, y)
    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        try:
            boot_vals.append(stat_fn(x[idx], y[idx]))
        except Exception:
            continue
    boot_vals = np.array(boot_vals)
    alpha = (1 - ci) / 2
    ci_low = np.nanpercentile(boot_vals, 100 * alpha)
    ci_high = np.nanpercentile(boot_vals, 100 * (1 - alpha))
    return point, ci_low, ci_high


def spearman_rho(x, y):
    """Compute Spearman rho (just the coefficient)."""
    r, _ = stats.spearmanr(x, y, nan_policy='omit')
    return r


def spearman_full(x, y):
    """Compute Spearman rho and p-value."""
    return stats.spearmanr(x, y, nan_policy='omit')


# ──────────────────── SAVE HELPERS ────────────────────
def savefig(fig, path_prefix, close=True):
    """Save figure as both PNG and PDF."""
    fig.savefig(f'{path_prefix}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{path_prefix}.pdf', bbox_inches='tight')
    if close:
        plt.close(fig)


def save_panel_json(path_prefix, metadata):
    """Save panel metadata JSON."""
    with open(f'{path_prefix}.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)


def save_other_stats(fig_dir, stats_dict):
    """Save other_stats.json for a figure directory."""
    path = os.path.join(fig_dir, 'other_stats.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2, default=str, ensure_ascii=False)


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


# ──────────────────── PLOT HELPERS ────────────────────
def plot_line_with_ci(ax, x, y_mean, y_ci_low, y_ci_high, color, label=None,
                      alpha_fill=0.2, linewidth=1.5, **kwargs):
    """Plot a line with filled confidence interval band."""
    ax.plot(x, y_mean, color=color, label=label, linewidth=linewidth, **kwargs)
    ax.fill_between(x, y_ci_low, y_ci_high, color=color, alpha=alpha_fill)


def compute_per_turn_means(df, value_col, alpha_val=0.0):
    """
    Compute mean, CI of a value column per turn from a results DataFrame.
    Returns DataFrame with columns: turn, mean, ci_low, ci_high, std.
    """
    sub = df[np.isclose(df['alpha'], alpha_val)]
    grouped = sub.groupby('turn')[value_col]
    result = grouped.agg(['mean', 'std', 'count']).reset_index()
    # Bootstrap CI
    ci_low_list, ci_high_list = [], []
    for t in result['turn']:
        vals = sub[sub['turn'] == t][value_col].dropna().values
        if len(vals) < 2:
            ci_low_list.append(np.nan)
            ci_high_list.append(np.nan)
            continue
        boots = []
        rng = np.random.RandomState(42)
        for _ in range(1000):
            boots.append(np.mean(rng.choice(vals, len(vals), replace=True)))
        ci_low_list.append(np.percentile(boots, 2.5))
        ci_high_list.append(np.percentile(boots, 97.5))
    result['ci_low'] = ci_low_list
    result['ci_high'] = ci_high_list
    return result


def compute_per_turn_unique_counts(df, value_col, alpha_val=0.0):
    """Count number of unique discrete responses per turn."""
    sub = df[np.isclose(df['alpha'], alpha_val)]
    grouped = sub.groupby('turn')[value_col]
    return grouped.nunique().reset_index().rename(columns={value_col: 'n_unique'})


def get_scatter_data(df, probe_concept, alpha_val=0.0, rating_col='logit_rating'):
    """
    Extract (probe_score, rating) pairs for scatter plot.
    Applies polarity flip if needed.
    Returns (probe_scores, ratings) arrays.
    """
    sub = df[np.isclose(df['alpha'], alpha_val)].copy()
    sub = sub.dropna(subset=['probe_score', rating_col])
    probe_scores = flip_if_needed(probe_concept, sub['probe_score'].values)
    ratings = sub[rating_col].values
    return probe_scores, ratings


def format_p(p):
    """Format p-value for display."""
    if p < 0.001:
        return f"p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


if __name__ == '__main__':
    # Quick validation
    print("=== Shared Utils Validation ===")
    print(f"Repo root: {REPO_ROOT}")
    print(f"\nLLaMA 3B probes:")
    for c, p in PROBES['llama_3b'].items():
        exists = os.path.isdir(p)
        print(f"  {c}: {'OK' if exists else 'MISSING'} — {p}")
    print(f"\nLLaMA 3B rerun experiments:")
    for c, p in LLAMA_3B_RERUN_SELF.items():
        exists = os.path.isdir(p)
        print(f"  {c}: {'OK' if exists else 'MISSING'}")
    print(f"\nLLaMA 3B greedy experiments:")
    for c, p in LLAMA_3B_GREEDY.items():
        exists = os.path.isdir(p)
        print(f"  {c}: {'OK' if exists else 'MISSING'}")
    print(f"\nLLaMA 1B experiments: {list(LLAMA_1B_SELF.keys())}")
    print(f"LLaMA 8B experiments: {list(LLAMA_8B_SELF.keys())}")
    print(f"Gemma 4B experiments: {list(GEMMA_4B_SELF.keys())}")
    print(f"Qwen 7B experiments: {list(QWEN_7B_SELF.keys())}")
    print(f"Random scoring controls: {list(LLAMA_3B_RANDOM.keys())}")
