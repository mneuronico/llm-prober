# Combine mixed-ops batch results into a single analysis folder.

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ROOT_DIR = Path(__file__).resolve().parents[1]

RATING_COLORS = {
    "COHERENT": "#33b233",
    "PARTIALLY_COHERENT": "#f2cc33",
    "NONSENSE": "#d93434",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    ax.set_title("Accuracy vs steering alpha (combined)")
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
    ax.set_title("Mean probe score vs steering alpha (combined)")
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
    ax.set_title("Probe score by correctness (combined)")
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
    ax.set_title("Probe score by correctness per alpha (combined)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_accuracy_by_coherence(
    alpha_vals: List[float],
    accuracy_by_rating: Dict[str, List[float]],
    out_path: Path,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for rating, color in RATING_COLORS.items():
        ax.plot(alpha_vals, accuracy_by_rating[rating], marker="o", linewidth=2, color=color, label=rating)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs steering alpha (by coherence, combined)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_coherence_counts(
    alpha_vals: List[float],
    counts_by_rating: Dict[str, List[int]],
    out_path: Path,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for rating, color in RATING_COLORS.items():
        ax.plot(alpha_vals, counts_by_rating[rating], marker="o", linewidth=2, color=color, label=rating)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Count")
    ax.set_title("Coherence rating count by alpha (combined)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _normalize_npz_path(path_str: str) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        return raw.resolve()
    return (ROOT_DIR / raw).resolve()


def _find_batches(eval_dir: Path) -> List[Path]:
    return sorted(
        [p for p in eval_dir.iterdir() if p.is_dir() and p.name.startswith("batch_")],
        key=lambda p: p.name,
    )


def _load_batch_per_sample(batch_dir: Path) -> List[Dict[str, object]]:
    per_sample_path = batch_dir / "analysis" / "per_sample.json"
    if not per_sample_path.exists():
        print(f"Warning: missing per_sample.json in {batch_dir}")
        return []
    data = _load_json(per_sample_path)
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    rel_batch = os.path.relpath(batch_dir, batch_dir.parent).replace("\\", "/")
    for item in items:
        if isinstance(item, dict):
            item["batch_dir"] = rel_batch
            item["batch_name"] = batch_dir.name
    return [item for item in items if isinstance(item, dict)]


def _load_batch_coherence(batch_dir: Path) -> Dict[str, str]:
    ratings_path = batch_dir / "coherence_rating.json"
    if not ratings_path.exists():
        return {}
    try:
        ratings_raw = _load_json(ratings_path)
    except json.JSONDecodeError:
        return {}
    ratings: Dict[str, str] = {}
    for entry in ratings_raw:
        if not isinstance(entry, dict):
            continue
        example = entry.get("example")
        rating = entry.get("rating")
        if isinstance(example, str) and isinstance(rating, str):
            ratings[str((batch_dir / example).resolve())] = rating
    return ratings


def _compute_stats(per_sample: List[Dict[str, object]]) -> Tuple[Dict[str, object], List[float]]:
    alpha_vals = sorted({float(item["alpha"]) for item in per_sample})
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
        acc = sum(1 for r in rows if r.get("correct")) / len(rows) if rows else float("nan")
        accuracy_vals.append(float(acc))
        scores = [float(r["score_mean"]) for r in rows if np.isfinite(r.get("score_mean", float("nan")))]
        mean_score_vals.append(float(np.mean(scores)) if scores else float("nan"))
        sem_score_vals.append(_sem(scores))

    correct_scores = np.array([r["score_mean"] for r in per_sample if r.get("correct")], dtype=np.float32)
    incorrect_scores = np.array([r["score_mean"] for r in per_sample if not r.get("correct")], dtype=np.float32)
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
        correct_scores_alpha = [float(r["score_mean"]) for r in rows if r.get("correct")]
        incorrect_scores_alpha = [float(r["score_mean"]) for r in rows if not r.get("correct")]
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
    return stats, alpha_vals


def _compute_coherence(
    per_sample: List[Dict[str, object]], alpha_vals: List[float], ratings: Dict[str, str]
) -> Tuple[Dict[str, List[int]], Dict[str, List[float]], List[Dict[str, str]]]:
    counts_by_rating: Dict[str, List[int]] = {k: [0] * len(alpha_vals) for k in RATING_COLORS}
    accuracy_by_rating: Dict[str, List[float]] = {
        k: [float("nan")] * len(alpha_vals) for k in RATING_COLORS
    }
    coherence_records: List[Dict[str, str]] = []

    by_alpha: Dict[float, List[Dict[str, object]]] = {}
    for row in per_sample:
        by_alpha.setdefault(float(row["alpha"]), []).append(row)

    for i, alpha in enumerate(alpha_vals):
        rows = by_alpha.get(alpha, [])
        for rating in RATING_COLORS:
            rated_rows = []
            for item in rows:
                npz_path = item.get("npz_path")
                if not isinstance(npz_path, str):
                    continue
                key = str(_normalize_npz_path(npz_path))
                item_rating = ratings.get(key)
                if item_rating == rating:
                    rated_rows.append(item)
                    coherence_records.append({"npz_path": npz_path, "rating": item_rating})
            counts_by_rating[rating][i] = len(rated_rows)
            if rated_rows:
                accuracy_by_rating[rating][i] = sum(1 for r in rated_rows if r.get("correct")) / len(rated_rows)

    return counts_by_rating, accuracy_by_rating, coherence_records


def analyze_eval_dir(eval_dir: Path, output_name: str, batch_dirs: Optional[List[Path]] = None) -> None:
    if batch_dirs is None:
        batch_dirs = _find_batches(eval_dir)
    if not batch_dirs:
        print(f"Warning: no batch directories found under {eval_dir}")
        return

    per_sample: List[Dict[str, object]] = []
    ratings: Dict[str, str] = {}
    for batch_dir in batch_dirs:
        per_sample.extend(_load_batch_per_sample(batch_dir))
        ratings.update(_load_batch_coherence(batch_dir))

    if not per_sample:
        print(f"Warning: no per-sample items found under {eval_dir}")
        return

    analysis_dir = eval_dir / output_name
    plots_dir = analysis_dir / "plots"
    _ensure_dir(plots_dir)

    (analysis_dir / "per_sample.json").write_text(
        json.dumps({"items": per_sample}, indent=2), encoding="utf-8"
    )

    stats, alpha_vals = _compute_stats(per_sample)
    (analysis_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    _plot_accuracy_by_alpha(alpha_vals, [r["accuracy"] for r in stats["accuracy_by_alpha"]],
                            [r["n"] for r in stats["accuracy_by_alpha"]],
                            plots_dir / "accuracy_vs_alpha.png")
    _plot_score_by_alpha(alpha_vals, [r["mean_score"] for r in stats["score_by_alpha"]],
                         [r["sem"] for r in stats["score_by_alpha"]],
                         plots_dir / "score_vs_alpha.png")

    correct_stats = stats["correct_vs_incorrect"]
    _plot_score_by_correctness(
        correct_stats["mean_correct"],
        correct_stats["mean_incorrect"],
        correct_stats["sem_correct"],
        correct_stats["sem_incorrect"],
        plots_dir / "score_by_correctness.png",
    )

    per_alpha_correctness = stats["correct_vs_incorrect_by_alpha"]
    _plot_score_by_correctness_by_alpha(
        alpha_vals,
        [r["mean_correct"] for r in per_alpha_correctness],
        [r["mean_incorrect"] for r in per_alpha_correctness],
        [r["sem_correct"] for r in per_alpha_correctness],
        [r["sem_incorrect"] for r in per_alpha_correctness],
        plots_dir / "score_by_correctness_by_alpha.png",
    )

    if ratings:
        counts_by_rating, accuracy_by_rating, coherence_records = _compute_coherence(
            per_sample, alpha_vals, ratings
        )
        _plot_accuracy_by_coherence(alpha_vals, accuracy_by_rating, plots_dir / "accuracy_vs_alpha_by_coherence.png")
        _plot_coherence_counts(alpha_vals, counts_by_rating, plots_dir / "coherence_counts_by_alpha.png")
        (analysis_dir / "coherence_rating.json").write_text(
            json.dumps(coherence_records, indent=2), encoding="utf-8"
        )
    else:
        print(f"Note: no coherence ratings found for {eval_dir}; skipping coherence plots.")

    print(f"Combined analysis written to {analysis_dir}")


def _resolve_eval_and_batches(path: Path) -> Tuple[Path, Optional[List[Path]]]:
    if path.name.startswith("batch_") and (path / "analysis" / "per_sample.json").exists():
        return path.parent, [path]
    return path, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine mixed-ops batch results into a single analysis folder."
    )
    parser.add_argument("eval_dirs", nargs="+", help="Eval folders to combine (math_eval_mixed_ops_5x2).")
    parser.add_argument(
        "--output-name",
        default="analysis_all",
        help="Name of the combined analysis folder inside each eval dir.",
    )
    args = parser.parse_args()

    for raw_path in args.eval_dirs:
        path = Path(raw_path).resolve()
        if not path.exists():
            print(f"Warning: path not found: {path}")
            continue
        eval_dir, batch_dirs = _resolve_eval_and_batches(path)
        analyze_eval_dir(eval_dir, args.output_name, batch_dirs=batch_dirs)


if __name__ == "__main__":
    main()
