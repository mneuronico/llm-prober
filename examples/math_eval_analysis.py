import os
from typing import Dict, List, Optional

import numpy as np

from concept_probe.math_eval import evaluate_answer
from concept_probe.utils import ensure_dir, json_dump

try:
    from scipy.stats import linregress, pearsonr, ttest_ind
except Exception:
    linregress = None
    pearsonr = None
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


def _plot_accuracy_by_alpha(alpha_vals: List[float], acc_vals: List[float], n_vals: List[int], out_path: str) -> bool:
    if plt is None:
        return False
    se = [np.sqrt(a * (1 - a) / n) if n > 0 else float("nan") for a, n in zip(acc_vals, n_vals)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(alpha_vals, acc_vals, yerr=se, marker="o", color="#2b6aa6", linewidth=2)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs steering alpha")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_score_by_alpha(alpha_vals: List[float], mean_scores: List[float], sem_scores: List[float], out_path: str) -> bool:
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(alpha_vals, mean_scores, yerr=sem_scores, marker="o", color="#b24a3b", linewidth=2)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Mean probe score")
    ax.set_title("Mean probe score vs steering alpha")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_score_by_correctness(
    mean_correct: float,
    mean_incorrect: float,
    sem_correct: float,
    sem_incorrect: float,
    out_path: str,
) -> bool:
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([0, 1], [mean_correct, mean_incorrect], yerr=[sem_correct, sem_incorrect], color=["#2c8d5b", "#b24a3b"])
    ax.set_xticks([0, 1], ["correct", "incorrect"])
    ax.set_ylabel("Mean probe score")
    ax.set_title("Probe score by correctness")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_score_by_correctness_by_alpha(
    alpha_vals: List[float],
    mean_correct: List[float],
    mean_incorrect: List[float],
    sem_correct: List[float],
    sem_incorrect: List[float],
    out_path: str,
) -> bool:
    if plt is None:
        return False
    x = np.arange(len(alpha_vals))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, mean_correct, width, yerr=sem_correct, color="#2c8d5b", label="correct")
    ax.bar(x + width / 2, mean_incorrect, width, yerr=sem_incorrect, color="#b24a3b", label="incorrect")
    ax.set_xticks(x, [str(a) for a in alpha_vals])
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Mean probe score")
    ax.set_title("Probe score by correctness per alpha")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def analyze_math_eval_results(
    results: List[Dict[str, object]],
    problems: List[Dict[str, object]],
    alphas: List[float],
    *,
    analysis_dir: Optional[str] = None,
    marker: str = "ANSWER:",
    require_marker: bool = True,
) -> Dict[str, object]:
    if not results:
        return {"per_sample": [], "stats": {}, "analysis_dir": None, "plots_dir": None}

    if analysis_dir is None:
        base_dir = os.path.dirname(results[0]["npz_path"])
        analysis_dir = os.path.join(base_dir, "analysis")
    plots_dir = os.path.join(analysis_dir, "plots")
    ensure_dir(plots_dir)

    alpha_count = len(alphas)
    per_sample: List[Dict[str, object]] = []
    for idx, rec in enumerate(results):
        prompt_idx = idx // alpha_count
        if prompt_idx >= len(problems):
            break
        problem = problems[prompt_idx]
        eval_info = evaluate_answer(
            rec["completion"], problem["answer"], marker=marker, require_marker=require_marker
        )
        score_mean = _mean_completion_score(rec["npz_path"])
        per_sample.append(
            {
                "problem_id": int(problem["id"]),
                "expression": problem["expression"],
                "expected": int(problem["answer"]),
                "prompt": rec["prompt"],
                "alpha": float(rec["alpha"]),
                "completion": rec["completion"],
                "parsed_answer": eval_info["parsed"],
                "used_marker": bool(eval_info["used_marker"]),
                "correct": bool(eval_info["correct"]),
                "score_mean": score_mean,
                "npz_path": rec["npz_path"],
                "html_path": rec["html_path"],
            }
        )

    json_dump(os.path.join(analysis_dir, "per_sample.json"), {"items": per_sample})

    by_alpha: Dict[float, List[Dict[str, object]]] = {}
    for row in per_sample:
        by_alpha.setdefault(float(row["alpha"]), []).append(row)

    alpha_vals = sorted(by_alpha.keys())
    accuracy_vals = []
    mean_score_vals = []
    sem_score_vals = []
    counts = []
    for alpha in alpha_vals:
        rows = by_alpha[alpha]
        counts.append(len(rows))
        acc = sum(1 for r in rows if r["correct"]) / len(rows)
        accuracy_vals.append(float(acc))
        scores = [float(r["score_mean"]) for r in rows if not np.isnan(r["score_mean"])]
        mean_score_vals.append(float(np.mean(scores)) if scores else float("nan"))
        sem_score_vals.append(_sem(scores))

    _plot_accuracy_by_alpha(
        alpha_vals, accuracy_vals, counts, os.path.join(plots_dir, "accuracy_vs_alpha.png")
    )
    _plot_score_by_alpha(
        alpha_vals, mean_score_vals, sem_score_vals, os.path.join(plots_dir, "score_vs_alpha.png")
    )

    correct_scores = np.array([r["score_mean"] for r in per_sample if r["correct"]], dtype=np.float32)
    incorrect_scores = np.array([r["score_mean"] for r in per_sample if not r["correct"]], dtype=np.float32)
    mean_correct = float(np.mean(correct_scores)) if correct_scores.size else float("nan")
    mean_incorrect = float(np.mean(incorrect_scores)) if incorrect_scores.size else float("nan")
    sem_correct = _sem(correct_scores.tolist())
    sem_incorrect = _sem(incorrect_scores.tolist())
    _plot_score_by_correctness(
        mean_correct,
        mean_incorrect,
        sem_correct,
        sem_incorrect,
        os.path.join(plots_dir, "score_by_correctness.png"),
    )

    mean_correct_by_alpha = []
    mean_incorrect_by_alpha = []
    sem_correct_by_alpha = []
    sem_incorrect_by_alpha = []
    per_alpha_correctness = []
    for alpha in alpha_vals:
        rows = by_alpha[alpha]
        correct_scores_alpha = [float(r["score_mean"]) for r in rows if r["correct"]]
        incorrect_scores_alpha = [float(r["score_mean"]) for r in rows if not r["correct"]]
        mean_c = float(np.mean(correct_scores_alpha)) if correct_scores_alpha else float("nan")
        mean_i = float(np.mean(incorrect_scores_alpha)) if incorrect_scores_alpha else float("nan")
        sem_c = _sem(correct_scores_alpha)
        sem_i = _sem(incorrect_scores_alpha)
        mean_correct_by_alpha.append(mean_c)
        mean_incorrect_by_alpha.append(mean_i)
        sem_correct_by_alpha.append(sem_c)
        sem_incorrect_by_alpha.append(sem_i)
        per_alpha_correctness.append(
            {
                "alpha": float(alpha),
                "mean_correct": mean_c,
                "mean_incorrect": mean_i,
                "sem_correct": sem_c,
                "sem_incorrect": sem_i,
                "n_correct": int(len(correct_scores_alpha)),
                "n_incorrect": int(len(incorrect_scores_alpha)),
            }
        )

    _plot_score_by_correctness_by_alpha(
        alpha_vals,
        mean_correct_by_alpha,
        mean_incorrect_by_alpha,
        sem_correct_by_alpha,
        sem_incorrect_by_alpha,
        os.path.join(plots_dir, "score_by_correctness_by_alpha.png"),
    )

    alpha_arr = np.array([float(r["alpha"]) for r in per_sample], dtype=np.float32)
    correct_arr = np.array([1.0 if r["correct"] else 0.0 for r in per_sample], dtype=np.float32)
    score_arr = np.array([float(r["score_mean"]) for r in per_sample], dtype=np.float32)

    stats = {
        "accuracy_by_alpha": [
            {"alpha": float(a), "accuracy": float(acc), "n": int(n)}
            for a, acc, n in zip(alpha_vals, accuracy_vals, counts)
        ],
        "score_by_alpha": [
            {"alpha": float(a), "mean_score": float(m), "sem": float(s), "n": int(n)}
            for a, m, s, n in zip(alpha_vals, mean_score_vals, sem_score_vals, counts)
        ],
        "correct_vs_incorrect": {
            "mean_correct": mean_correct,
            "mean_incorrect": mean_incorrect,
            "sem_correct": sem_correct,
            "sem_incorrect": sem_incorrect,
            "cohen_d": _cohen_d(correct_scores, incorrect_scores),
        },
        "correct_vs_incorrect_by_alpha": per_alpha_correctness,
    }

    if alpha_vals:
        alpha_vec = np.array(alpha_vals, dtype=np.float64)
        acc_vec = np.array(accuracy_vals, dtype=np.float64)
        if alpha_vec.size >= 2 and np.isfinite(acc_vec).all():
            slope, intercept = np.polyfit(alpha_vec, acc_vec, 1)
            trend = {"slope": float(slope), "intercept": float(intercept), "p_value": None}
            if linregress is not None:
                reg = linregress(alpha_vec, acc_vec)
                trend["p_value"] = float(reg.pvalue)
            stats["accuracy_linear_trend"] = trend
        else:
            stats["accuracy_linear_trend"] = {"slope": None, "intercept": None, "p_value": None}
    else:
        stats["accuracy_linear_trend"] = {"slope": None, "intercept": None, "p_value": None}

    if pearsonr is not None:
        corr_acc, p_acc = pearsonr(alpha_arr, correct_arr)
        corr_score, p_score = pearsonr(alpha_arr, score_arr)
        stats["correlations"] = {
            "alpha_vs_accuracy": {"r": float(corr_acc), "p": float(p_acc)},
            "alpha_vs_score": {"r": float(corr_score), "p": float(p_score)},
        }
    else:
        stats["correlations"] = {
            "alpha_vs_accuracy": {"r": None, "p": None},
            "alpha_vs_score": {"r": None, "p": None},
        }

    if ttest_ind is not None and correct_scores.size > 1 and incorrect_scores.size > 1:
        _, p = ttest_ind(correct_scores, incorrect_scores, equal_var=False)
        stats["correct_vs_incorrect"]["p_value"] = float(p)
    else:
        stats["correct_vs_incorrect"]["p_value"] = None

    json_dump(os.path.join(analysis_dir, "stats.json"), stats)
    return {"per_sample": per_sample, "stats": stats, "analysis_dir": analysis_dir, "plots_dir": plots_dir}
