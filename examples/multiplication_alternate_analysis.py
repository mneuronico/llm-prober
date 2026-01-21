import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _plot_percent_error_by_alpha(
    alpha_vals: List[float], mean_vals: List[float], sem_vals: List[float], out_path: Path
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(alpha_vals, mean_vals, yerr=sem_vals, marker="o", color="#6b3fa0", linewidth=2)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Mean percentage error")
    ax.set_title("Percentage error vs steering alpha")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_score_by_alpha(
    alpha_vals: List[float], mean_scores: List[float], sem_scores: List[float], out_path: Path
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(alpha_vals, mean_scores, yerr=sem_scores, marker="o", color="#b24a3b", linewidth=2)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Mean probe score")
    ax.set_title("Mean probe score vs steering alpha")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_score_by_correctness(
    mean_correct: float,
    mean_incorrect: float,
    sem_correct: float,
    sem_incorrect: float,
    out_path: Path,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([0, 1], [mean_correct, mean_incorrect], yerr=[sem_correct, sem_incorrect], color=["#2c8d5b", "#b24a3b"])
    ax.set_xticks([0, 1], ["correct", "incorrect"])
    ax.set_ylabel("Mean probe score")
    ax.set_title("Probe score by correctness")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_score_by_correctness_by_alpha(
    alpha_vals: List[float],
    mean_correct: List[float],
    mean_incorrect: List[float],
    sem_correct: List[float],
    sem_incorrect: List[float],
    out_path: Path,
) -> None:
    if plt is None:
        return
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


def _percent_error(parsed: Optional[float], expected: Optional[float]) -> Optional[float]:
    if parsed is None or expected is None:
        return None
    if float(expected) == 0.0:
        return None
    return abs(float(parsed) - float(expected)) / abs(float(expected)) * 100.0


def analyze_alternate(batch_dir: str) -> Dict[str, str]:
    batch_path = Path(batch_dir).resolve()
    base_analysis_dir = batch_path / "analysis"
    per_sample_path = base_analysis_dir / "per_sample.json"
    if not per_sample_path.exists():
        raise FileNotFoundError(f"Missing per_sample.json at {per_sample_path}")

    per_sample = _load_json(per_sample_path).get("items", [])
    alpha_vals = sorted({float(item["alpha"]) for item in per_sample})

    alternate_dir = batch_path / "alternate_analysis"
    plots_dir = alternate_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_sample_alt: List[Dict[str, object]] = []
    for item in per_sample:
        parsed = item.get("parsed_answer")
        expected = item.get("expected")
        pct = _percent_error(parsed, expected)
        updated = dict(item)
        updated["percent_error"] = pct
        per_sample_alt.append(updated)

    (alternate_dir / "per_sample.json").write_text(
        json.dumps({"items": per_sample_alt}, indent=2), encoding="utf-8"
    )

    by_alpha: Dict[float, List[Dict[str, object]]] = {}
    for row in per_sample_alt:
        by_alpha.setdefault(float(row["alpha"]), []).append(row)

    pct_means: List[float] = []
    pct_sems: List[float] = []
    counts: List[int] = []
    for alpha in alpha_vals:
        rows = by_alpha.get(alpha, [])
        vals = [float(r["percent_error"]) for r in rows if r.get("percent_error") is not None]
        counts.append(len(rows))
        pct_means.append(float(np.mean(vals)) if vals else float("nan"))
        pct_sems.append(_sem(vals))

    mean_score_vals: List[float] = []
    sem_score_vals: List[float] = []
    for alpha in alpha_vals:
        rows = by_alpha.get(alpha, [])
        scores = [float(r["score_mean"]) for r in rows if np.isfinite(r["score_mean"])]
        mean_score_vals.append(float(np.mean(scores)) if scores else float("nan"))
        sem_score_vals.append(_sem(scores))

    correct_scores = np.array([r["score_mean"] for r in per_sample_alt if r["correct"]], dtype=np.float32)
    incorrect_scores = np.array([r["score_mean"] for r in per_sample_alt if not r["correct"]], dtype=np.float32)
    mean_correct = float(np.mean(correct_scores)) if correct_scores.size else float("nan")
    mean_incorrect = float(np.mean(incorrect_scores)) if incorrect_scores.size else float("nan")
    sem_correct = _sem(correct_scores.tolist())
    sem_incorrect = _sem(incorrect_scores.tolist())

    mean_correct_by_alpha: List[float] = []
    mean_incorrect_by_alpha: List[float] = []
    sem_correct_by_alpha: List[float] = []
    sem_incorrect_by_alpha: List[float] = []
    per_alpha_correctness: List[Dict[str, object]] = []
    for alpha in alpha_vals:
        rows = by_alpha.get(alpha, [])
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

    stats = {
        "percent_error_by_alpha": [
            {"alpha": float(a), "mean_percent_error": float(m), "sem": float(s), "n": int(n)}
            for a, m, s, n in zip(alpha_vals, pct_means, pct_sems, counts)
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

    alpha_arr = np.array([float(r["alpha"]) for r in per_sample_alt], dtype=np.float32)
    percent_arr = np.array(
        [float(r["percent_error"]) for r in per_sample_alt if r.get("percent_error") is not None],
        dtype=np.float32,
    )
    score_arr = np.array([float(r["score_mean"]) for r in per_sample_alt], dtype=np.float32)

    if alpha_vals and percent_arr.size == len(per_sample_alt):
        alpha_vec = np.array(alpha_vals, dtype=np.float64)
        pct_vec = np.array(pct_means, dtype=np.float64)
        if alpha_vec.size >= 2 and np.isfinite(pct_vec).all():
            slope, intercept = np.polyfit(alpha_vec, pct_vec, 1)
            stats["percent_error_linear_trend"] = {"slope": float(slope), "intercept": float(intercept)}
        else:
            stats["percent_error_linear_trend"] = {"slope": None, "intercept": None}
    else:
        stats["percent_error_linear_trend"] = {"slope": None, "intercept": None}

    if percent_arr.size == len(per_sample_alt):
        corr_pct = np.corrcoef(alpha_arr, percent_arr)[0, 1] if alpha_arr.size else float("nan")
        corr_score = np.corrcoef(alpha_arr, score_arr)[0, 1] if alpha_arr.size else float("nan")
        stats["correlations"] = {
            "alpha_vs_percent_error": {"r": float(corr_pct)},
            "alpha_vs_score": {"r": float(corr_score)},
        }
    else:
        stats["correlations"] = {
            "alpha_vs_percent_error": {"r": None},
            "alpha_vs_score": {"r": None},
        }

    (alternate_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    _plot_percent_error_by_alpha(alpha_vals, pct_means, pct_sems, plots_dir / "percent_error_vs_alpha.png")
    _plot_score_by_alpha(alpha_vals, mean_score_vals, sem_score_vals, plots_dir / "score_vs_alpha.png")
    _plot_score_by_correctness(
        mean_correct, mean_incorrect, sem_correct, sem_incorrect, plots_dir / "score_by_correctness.png"
    )
    _plot_score_by_correctness_by_alpha(
        alpha_vals,
        mean_correct_by_alpha,
        mean_incorrect_by_alpha,
        sem_correct_by_alpha,
        sem_incorrect_by_alpha,
        plots_dir / "score_by_correctness_by_alpha.png",
    )

    return {
        "analysis_dir": str(alternate_dir),
        "plots_dir": str(plots_dir),
        "stats_path": str(alternate_dir / "stats.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Alternate analysis with percent error.")
    parser.add_argument("batch_dir", help="Path to batch directory.")
    args = parser.parse_args()
    result = analyze_alternate(args.batch_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
