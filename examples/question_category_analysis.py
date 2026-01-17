import os
from typing import Dict, List, Optional

import numpy as np

from concept_probe.utils import ensure_dir, json_dump

try:
    from scipy.stats import ttest_ind
except Exception:
    ttest_ind = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _mean_completion_score(npz_path: str) -> float:
    data = np.load(npz_path)
    scores = data["scores_agg"]
    prompt_len = int(data["prompt_len"][0]) if "prompt_len" in data else 0
    if prompt_len < scores.shape[0]:
        span = scores[prompt_len:]
    else:
        span = scores
    if span.size == 0:
        return float("nan")
    return float(np.mean(span))


def _sem(values: List[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    nx, ny = x.size, y.size
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return float((x.mean() - y.mean()) / (np.sqrt(pooled) + 1e-12))


def _alpha_slug(alpha: float) -> str:
    if abs(alpha) < 1e-12:
        return "zero"
    sign = "neg" if alpha < 0 else "pos"
    value = f"{abs(alpha):g}".replace(".", "p")
    return f"{sign}{value}"


def _plot_score_histogram(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    label_a: str,
    label_b: str,
    alpha: float,
    out_path: str,
) -> bool:
    if plt is None:
        return False
    scores_a = scores_a[np.isfinite(scores_a)]
    scores_b = scores_b[np.isfinite(scores_b)]
    if scores_a.size == 0 and scores_b.size == 0:
        return False
    combined = np.concatenate([scores_a, scores_b]) if scores_a.size and scores_b.size else (
        scores_a if scores_a.size else scores_b
    )
    bins = np.histogram_bin_edges(combined, bins="auto")
    fig, ax = plt.subplots(figsize=(6, 4))
    if scores_a.size:
        ax.hist(scores_a, bins=bins, alpha=0.6, color="#2c8d5b", label=label_a)
    if scores_b.size:
        ax.hist(scores_b, bins=bins, alpha=0.6, color="#b24a3b", label=label_b)
    ax.set_xlabel("Mean probe score (completion)")
    ax.set_ylabel("Count")
    ax.set_title(f"Probe score distribution (alpha={alpha:g})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def analyze_question_category_scores(
    results_a: List[Dict[str, object]],
    results_b: List[Dict[str, object]],
    alphas: List[float],
    *,
    label_a: str = "category_a",
    label_b: str = "category_b",
    analysis_dir: Optional[str] = None,
) -> Dict[str, object]:
    if not results_a or not results_b:
        return {"per_sample": [], "stats": {}, "analysis_dir": None, "plots_dir": None}

    base_a = os.path.dirname(results_a[0]["npz_path"])
    base_b = os.path.dirname(results_b[0]["npz_path"])
    common_base = os.path.commonpath([base_a, base_b])
    if analysis_dir is None:
        analysis_dir = os.path.join(common_base, "analysis")
    plots_dir = os.path.join(analysis_dir, "plots")
    ensure_dir(plots_dir)

    per_sample: List[Dict[str, object]] = []
    for rec in results_a:
        per_sample.append(
            {
                "category": label_a,
                "alpha": float(rec["alpha"]),
                "prompt": rec["prompt"],
                "completion": rec["completion"],
                "score_mean": _mean_completion_score(rec["npz_path"]),
                "npz_path": rec["npz_path"],
                "html_path": rec["html_path"],
            }
        )
    for rec in results_b:
        per_sample.append(
            {
                "category": label_b,
                "alpha": float(rec["alpha"]),
                "prompt": rec["prompt"],
                "completion": rec["completion"],
                "score_mean": _mean_completion_score(rec["npz_path"]),
                "npz_path": rec["npz_path"],
                "html_path": rec["html_path"],
            }
        )

    json_dump(os.path.join(analysis_dir, "per_sample.json"), {"items": per_sample})

    stats_by_alpha = []
    alpha_vals = [float(a) for a in alphas]
    for alpha in alpha_vals:
        scores_a = np.array(
            [float(r["score_mean"]) for r in per_sample if r["alpha"] == alpha and r["category"] == label_a],
            dtype=np.float32,
        )
        scores_b = np.array(
            [float(r["score_mean"]) for r in per_sample if r["alpha"] == alpha and r["category"] == label_b],
            dtype=np.float32,
        )
        scores_a = scores_a[np.isfinite(scores_a)]
        scores_b = scores_b[np.isfinite(scores_b)]
        mean_a = float(np.mean(scores_a)) if scores_a.size else float("nan")
        mean_b = float(np.mean(scores_b)) if scores_b.size else float("nan")
        sem_a = _sem(scores_a.tolist())
        sem_b = _sem(scores_b.tolist())
        p_value = None
        if ttest_ind is not None and scores_a.size > 1 and scores_b.size > 1:
            _, p_value = ttest_ind(scores_a, scores_b, equal_var=False)
            p_value = float(p_value)

        _plot_score_histogram(
            scores_a,
            scores_b,
            label_a,
            label_b,
            alpha,
            os.path.join(plots_dir, f"score_hist_alpha_{_alpha_slug(alpha)}.png"),
        )

        stats_by_alpha.append(
            {
                "alpha": float(alpha),
                "label_a": label_a,
                "label_b": label_b,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "sem_a": sem_a,
                "sem_b": sem_b,
                "n_a": int(scores_a.size),
                "n_b": int(scores_b.size),
                "cohen_d": _cohen_d(scores_a, scores_b),
                "p_value": p_value,
            }
        )

    stats = {"by_alpha": stats_by_alpha, "label_a": label_a, "label_b": label_b}
    json_dump(os.path.join(analysis_dir, "stats.json"), stats)

    return {"per_sample": per_sample, "stats": stats, "analysis_dir": analysis_dir, "plots_dir": plots_dir}
