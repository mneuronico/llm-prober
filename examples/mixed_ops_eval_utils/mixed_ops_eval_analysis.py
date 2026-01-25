# Analyze mixed-ops eval batch plots.

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


RATING_COLORS = {
    "COHERENT": "#33b233",
    "PARTIALLY_COHERENT": "#f2cc33",
    "NONSENSE": "#d93434",
}


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


def _plot_accuracy_by_alpha(alpha_vals: List[float], acc_vals: List[float], n_vals: List[int], out_path: Path) -> None:
    if plt is None:
        return
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


def analyze_mixed_ops_batch(batch_dir: str) -> Dict[str, str]:
    batch_path = Path(batch_dir).resolve()
    per_sample_path = batch_path / "analysis" / "per_sample.json"
    ratings_path = batch_path / "coherence_rating.json"
    if not per_sample_path.exists() or not ratings_path.exists():
        raise FileNotFoundError("Missing per_sample.json or coherence_rating.json in batch directory.")

    per_sample = _load_json(per_sample_path).get("items", [])
    ratings_raw = _load_json(ratings_path)
    ratings: Dict[str, str] = {
        entry["example"]: entry["rating"]
        for entry in ratings_raw
        if isinstance(entry, dict) and "example" in entry and "rating" in entry
    }

    alpha_vals = sorted({float(item["alpha"]) for item in per_sample})
    counts_by_rating: Dict[str, List[int]] = {k: [0] * len(alpha_vals) for k in RATING_COLORS}
    accuracy_by_rating: Dict[str, List[float]] = {
        k: [float("nan")] * len(alpha_vals) for k in RATING_COLORS
    }

    by_alpha: Dict[float, List[Dict[str, object]]] = {}
    for row in per_sample:
        by_alpha.setdefault(float(row["alpha"]), []).append(row)

    accuracy_vals: List[float] = []
    mean_score_vals: List[float] = []
    sem_score_vals: List[float] = []
    counts: List[int] = []
    for alpha in alpha_vals:
        rows = by_alpha.get(alpha, [])
        counts.append(len(rows))
        acc = sum(1 for r in rows if r["correct"]) / len(rows) if rows else float("nan")
        accuracy_vals.append(float(acc))
        scores = [float(r["score_mean"]) for r in rows if np.isfinite(r["score_mean"])]
        mean_score_vals.append(float(np.mean(scores)) if scores else float("nan"))
        sem_score_vals.append(_sem(scores))

    correct_scores = np.array([r["score_mean"] for r in per_sample if r["correct"]], dtype=np.float32)
    incorrect_scores = np.array([r["score_mean"] for r in per_sample if not r["correct"]], dtype=np.float32)
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

    for i, alpha in enumerate(alpha_vals):
        alpha_items = [item for item in per_sample if float(item["alpha"]) == float(alpha)]
        for rating in RATING_COLORS:
            rows = []
            for item in alpha_items:
                example = Path(item["npz_path"]).name
                if ratings.get(example) == rating:
                    rows.append(item)
            counts_by_rating[rating][i] = len(rows)
            if rows:
                accuracy_by_rating[rating][i] = sum(1 for r in rows if r["correct"]) / len(rows)

    plots_dir = batch_path / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_accuracy = plots_dir / "accuracy_vs_alpha_by_coherence.png"
    out_counts = plots_dir / "coherence_counts_by_alpha.png"
    out_accuracy_base = plots_dir / "accuracy_vs_alpha.png"
    out_score = plots_dir / "score_vs_alpha.png"
    out_score_correctness = plots_dir / "score_by_correctness.png"
    out_score_correctness_alpha = plots_dir / "score_by_correctness_by_alpha.png"
    stats_path = batch_path / "analysis" / "stats.json"

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
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    if plt is None:
        return {
            "accuracy_plot": str(out_accuracy),
            "counts_plot": str(out_counts),
            "stats_path": str(stats_path),
        }

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for rating, color in RATING_COLORS.items():
        ax.plot(alpha_vals, accuracy_by_rating[rating], marker="o", linewidth=2, color=color, label=rating)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs steering alpha (by coherence)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_accuracy, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for rating, color in RATING_COLORS.items():
        ax.plot(alpha_vals, counts_by_rating[rating], marker="o", linewidth=2, color=color, label=rating)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Count")
    ax.set_title("Coherence rating count by alpha")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_counts, dpi=160)
    plt.close(fig)

    _plot_accuracy_by_alpha(alpha_vals, accuracy_vals, counts, out_accuracy_base)
    _plot_score_by_alpha(alpha_vals, mean_score_vals, sem_score_vals, out_score)
    _plot_score_by_correctness(
        mean_correct, mean_incorrect, sem_correct, sem_incorrect, out_score_correctness
    )
    _plot_score_by_correctness_by_alpha(
        alpha_vals,
        mean_correct_by_alpha,
        mean_incorrect_by_alpha,
        sem_correct_by_alpha,
        sem_incorrect_by_alpha,
        out_score_correctness_alpha,
    )

    return {
        "accuracy_plot": str(out_accuracy),
        "counts_plot": str(out_counts),
        "stats_path": str(stats_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mixed-ops eval batch plots.")
    parser.add_argument("batch_dir", help="Path to batch directory.")
    args = parser.parse_args()
    result = analyze_mixed_ops_batch(args.batch_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
