# Analyze coherence vs accuracy for math eval batches.

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except Exception:
    plt = None
    Patch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.coherence_rater import rate_batch_coherence


RATING_COLORS = {
    "COHERENT": (0.20, 0.70, 0.20),
    "PARTIALLY_COHERENT": (0.95, 0.80, 0.20),
    "NONSENSE": (0.85, 0.20, 0.20),
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_per_sample(batch_dir: Path) -> List[Dict[str, object]]:
    per_sample_path = batch_dir / "analysis" / "per_sample.json"
    if not per_sample_path.exists():
        raise FileNotFoundError(f"Missing per_sample.json at {per_sample_path}")
    data = _load_json(per_sample_path)
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError("per_sample.json is missing an items list.")
    return items


def _load_ratings(batch_dir: Path) -> Dict[str, str]:
    ratings_path = batch_dir / "coherence_rating.json"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing coherence_rating.json at {ratings_path}")
    data = _load_json(ratings_path)
    if not isinstance(data, list):
        raise ValueError("coherence_rating.json must be a list.")
    ratings: Dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        example = entry.get("example")
        rating = entry.get("rating")
        if isinstance(example, str) and isinstance(rating, str):
            ratings[example] = rating
    return ratings


def _attach_ratings(
    items: List[Dict[str, object]], ratings: Dict[str, str]
) -> List[Dict[str, object]]:
    out = []
    missing = []
    for item in items:
        npz_path = item.get("npz_path")
        if not isinstance(npz_path, str):
            missing.append("<unknown>")
            continue
        example = Path(npz_path).name
        rating = ratings.get(example)
        if rating is None:
            missing.append(example)
            continue
        updated = dict(item)
        updated["rating"] = rating
        updated["example"] = example
        out.append(updated)
    if missing:
        raise RuntimeError(
            "Missing coherence ratings for some items. "
            f"First missing: {missing[0]} (total missing: {len(missing)})"
        )
    return out


def _accuracy(points: List[Dict[str, object]]) -> float:
    if not points:
        return float("nan")
    return float(sum(1 for p in points if p.get("correct")) / len(points))


def _accuracy_se(acc: float, n: int) -> float:
    if n <= 0 or not np.isfinite(acc):
        return float("nan")
    return float(np.sqrt(acc * (1.0 - acc) / n))


def _series_by_alpha(
    items: List[Dict[str, object]], alpha_vals: List[float]
) -> List[Tuple[float, float, int]]:
    series = []
    for alpha in alpha_vals:
        rows = [r for r in items if float(r.get("alpha")) == float(alpha)]
        acc = _accuracy(rows)
        series.append((float(alpha), acc, len(rows)))
    return series


def _plot_accuracy_curves(
    alpha_vals: List[float],
    series_by_label: Dict[str, List[Tuple[float, float, int]]],
    out_path: Path,
    title: str,
) -> bool:
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, series in series_by_label.items():
        points = [(a, acc, n) for a, acc, n in series if n > 0 and np.isfinite(acc)]
        if not points:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        se = [_accuracy_se(p[1], p[2]) for p in points]
        ax.errorbar(x, y, yerr=se, marker="o", linewidth=2, label=label)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _enough_points(series: List[Tuple[float, float, int]]) -> bool:
    return sum(1 for _, _, n in series if n > 0) >= 2


def _apply_brightness(color: Tuple[float, float, float], correct: bool) -> Tuple[float, float, float]:
    base = np.array(color, dtype=np.float32)
    if correct:
        return tuple((base * 0.35 + 0.65).clip(0.0, 1.0))
    return tuple((base * 0.35).clip(0.0, 1.0))


def _plot_heatmap(
    items: List[Dict[str, object]],
    alpha_vals: List[float],
    out_path: Path,
) -> bool:
    if plt is None:
        return False
    alpha_vals = sorted(alpha_vals)
    problem_ids = sorted({int(r.get("problem_id")) for r in items})
    lookup: Dict[Tuple[int, float], Dict[str, object]] = {}
    for item in items:
        key = (int(item.get("problem_id")), float(item.get("alpha")))
        lookup[key] = item

    grid = np.zeros((len(problem_ids), len(alpha_vals), 3), dtype=np.float32)
    for i, pid in enumerate(problem_ids):
        for j, alpha in enumerate(alpha_vals):
            cell = lookup.get((pid, float(alpha)))
            if cell is None:
                grid[i, j] = (0.5, 0.5, 0.5)
                continue
            rating = str(cell.get("rating"))
            correct = bool(cell.get("correct"))
            base = RATING_COLORS.get(rating, (0.5, 0.5, 0.5))
            grid[i, j] = _apply_brightness(base, correct)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(alpha_vals)), max(4.0, 0.25 * len(problem_ids))))
    ax.imshow(grid, aspect="auto")
    ax.set_xticks(range(len(alpha_vals)), [str(a) for a in alpha_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(problem_ids)), [str(pid) for pid in problem_ids])
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Problem id")
    ax.set_title("Coherence (hue) and correctness (brightness)")
    if Patch is not None:
        legend_items = [
            Patch(color=RATING_COLORS["COHERENT"], label="COHERENT"),
            Patch(color=RATING_COLORS["PARTIALLY_COHERENT"], label="PARTIALLY_COHERENT"),
            Patch(color=RATING_COLORS["NONSENSE"], label="NONSENSE"),
        ]
        ax.legend(handles=legend_items, title="Coherence", loc="upper right", frameon=True)
    fig.text(0.5, 0.01, "Brightness: light = correct, dark = incorrect", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def analyze_coherence_accuracy(
    batch_dir: str,
    *,
    max_elements_per_request: int = 8,
    rate_if_missing: bool = False,
    model: str = "openai/gpt-oss-20b",
) -> Dict[str, str]:
    batch_path = Path(batch_dir).resolve()
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

    ratings_path = batch_path / "coherence_rating.json"
    if not ratings_path.exists() and rate_if_missing:
        rate_batch_coherence(
            str(batch_path),
            max_elements_per_request=max_elements_per_request,
            model=model,
            env_path=r"C:\Users\Nico\Documents\GitHub\llm-prober\.env"
        )

    per_sample_items = _load_per_sample(batch_path)
    ratings = _load_ratings(batch_path)
    items = _attach_ratings(per_sample_items, ratings)

    alpha_vals = sorted({float(item.get("alpha")) for item in items})
    plots_dir = batch_path / "analysis" / "plots"
    _ensure_dir(plots_dir)

    by_rating: Dict[str, List[Dict[str, object]]] = {k: [] for k in RATING_COLORS}
    for item in items:
        rating = str(item.get("rating"))
        if rating in by_rating:
            by_rating[rating].append(item)

    series_three = {label: _series_by_alpha(rows, alpha_vals) for label, rows in by_rating.items()}
    if all(_enough_points(series) for series in series_three.values()):
        out_path = plots_dir / "accuracy_vs_alpha_by_coherence.png"
        _plot_accuracy_curves(
            alpha_vals, series_three, out_path, "Accuracy vs steering alpha (by coherence)"
        )
        return {"plot": str(out_path)}

    coherent = by_rating["COHERENT"]
    not_coherent = by_rating["PARTIALLY_COHERENT"] + by_rating["NONSENSE"]
    series_two = {
        "COHERENT": _series_by_alpha(coherent, alpha_vals),
        "NOT_COHERENT": _series_by_alpha(not_coherent, alpha_vals),
    }
    if all(_enough_points(series) for series in series_two.values()):
        out_path = plots_dir / "accuracy_vs_alpha_by_coherence_binary.png"
        _plot_accuracy_curves(
            alpha_vals, series_two, out_path, "Accuracy vs steering alpha (coherent vs not)"
        )
        return {"plot": str(out_path)}

    heatmap_path = plots_dir / "coherence_correctness_heatmap.png"
    _plot_heatmap(items, alpha_vals, heatmap_path)
    return {"plot": str(heatmap_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze coherence vs accuracy for math eval batches.")
    parser.add_argument("batch_dir", help="Path to batch directory.")
    parser.add_argument("--max-elements-per-request", type=int, default=8)
    parser.add_argument("--rate-if-missing", action="store_true")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    args = parser.parse_args()

    result = analyze_coherence_accuracy(
        args.batch_dir,
        max_elements_per_request=args.max_elements_per_request,
        rate_if_missing=args.rate_if_missing,
        model=args.model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
