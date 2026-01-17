import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make sure the project root is on sys.path so we can import the library.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from concept_probe.utils import ensure_dir, json_dump, safe_slug

try:
    from scipy.stats import pearsonr, ttest_ind
except Exception:
    pearsonr = None
    ttest_ind = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


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


def _load_ratings(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ratings: Dict[str, float] = {}
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            for item in data["items"]:
                if not isinstance(item, dict):
                    continue
                if "id" in item and "correctness" in item:
                    ratings[str(item["id"])] = float(item["correctness"])
        else:
            for key, value in data.items():
                ratings[str(key)] = float(value)
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if "id" in item and "correctness" in item:
                ratings[str(item["id"])] = float(item["correctness"])
    else:
        raise ValueError("Unsupported ratings format. Use a dict or list of {id, correctness}.")
    return ratings


def _plot_correctness_by_category(
    mean_adv: float,
    mean_non: float,
    sem_adv: float,
    sem_non: float,
    out_path: str,
) -> bool:
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([0, 1], [mean_adv, mean_non], yerr=[sem_adv, sem_non], color=["#b24a3b", "#2c8d5b"])
    ax.set_xticks([0, 1], ["adversarial", "non_adversarial"])
    ax.set_ylabel("Correctness rating")
    ax.set_title("Correctness by question type")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_metric_scatter(
    xs: np.ndarray,
    ys: np.ndarray,
    categories: List[str],
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> bool:
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"adversarial": "#b24a3b", "non_adversarial": "#2c8d5b"}
    for cat in ("adversarial", "non_adversarial"):
        idx = [i for i, c in enumerate(categories) if c == cat]
        if not idx:
            continue
        ax.scatter(xs[idx], ys[idx], label=cat, color=colors[cat], alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _metric_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_f = x[mask]
    y_f = y[mask]
    if x_f.size < 2:
        return {"n": int(x_f.size), "r": None, "p_value": None, "slope": None, "intercept": None}
    slope, intercept = np.polyfit(x_f, y_f, 1)
    r_val, p_val = (None, None)
    if pearsonr is not None:
        r_val, p_val = pearsonr(x_f, y_f)
        r_val = float(r_val)
        p_val = float(p_val)
    return {
        "n": int(x_f.size),
        "r": r_val,
        "p_value": p_val,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def main(
    eval_items_path: str = r"C:\Users\Nico\Documents\GitHub\llm-prober\outputs_multi\truthfulqa_multi_probe\20260116_155358\truthfulqa_eval_items.json",
    ratings_path: str = r"C:\Users\Nico\Documents\GitHub\llm-prober\outputs_multi\truthfulqa_multi_probe\20260116_155358\truthfulqa_ratings.json",
    output_dir: Optional[str] = None,
) -> None:
    with open(eval_items_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    items = eval_data["items"]
    ratings = _load_ratings(ratings_path)

    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(eval_items_path))
        output_dir = os.path.join(base_dir, "analysis-v2")
    plots_dir = os.path.join(output_dir, "plots")
    ensure_dir(plots_dir)

    per_item = []
    for item in items:
        item_id = str(item["id"])
        if item_id not in ratings:
            continue
        per_item.append(
            {
                "id": item_id,
                "type": item["type"],
                "question": item["question"],
                "correctness": float(ratings[item_id]),
                "npz_path": item["npz_path"],
            }
        )

    if not per_item:
        raise ValueError("No items matched the provided ratings.")

    adv_scores = np.array([p["correctness"] for p in per_item if p["type"] == "adversarial"], dtype=np.float32)
    non_scores = np.array([p["correctness"] for p in per_item if p["type"] == "non_adversarial"], dtype=np.float32)
    mean_adv = float(np.mean(adv_scores)) if adv_scores.size else float("nan")
    mean_non = float(np.mean(non_scores)) if non_scores.size else float("nan")
    sem_adv = _sem(adv_scores.tolist())
    sem_non = _sem(non_scores.tolist())
    p_value = None
    if ttest_ind is not None and adv_scores.size > 1 and non_scores.size > 1:
        _, p_value = ttest_ind(adv_scores, non_scores, equal_var=False)
        p_value = float(p_value)

    _plot_correctness_by_category(
        mean_adv,
        mean_non,
        sem_adv,
        sem_non,
        os.path.join(plots_dir, "correctness_by_category.png"),
    )

    metrics_by_probe: Dict[str, Dict[str, List[float]]] = {}
    categories: List[str] = []
    correctness: List[float] = []
    for entry in per_item:
        data = np.load(entry["npz_path"])
        scores_agg = data["scores_agg"]
        prompt_len = int(data["prompt_len"][0]) if "prompt_len" in data else 0
        probe_names = [str(n) for n in data["probe_names"].tolist()]
        categories.append(entry["type"])
        correctness.append(float(entry["correctness"]))

        for idx, name in enumerate(probe_names):
            scores = scores_agg[idx]
            if prompt_len < scores.shape[0]:
                span = scores[prompt_len:]
            else:
                span = scores
            span = span[np.isfinite(span)]
            if span.size == 0:
                mean_val = float("nan")
                max_val = float("nan")
                prop_pos = float("nan")
            else:
                mean_val = float(np.mean(span))
                max_val = float(np.max(span))
                prop_pos = float(np.mean(span > 0))

            metrics_by_probe.setdefault(name, {"mean": [], "max": [], "prop_pos": []})
            metrics_by_probe[name]["mean"].append(mean_val)
            metrics_by_probe[name]["max"].append(max_val)
            metrics_by_probe[name]["prop_pos"].append(prop_pos)

    x = np.array(correctness, dtype=np.float32)
    stats: Dict[str, object] = {
        "correctness_by_category": {
            "mean_adversarial": mean_adv,
            "mean_non_adversarial": mean_non,
            "sem_adversarial": sem_adv,
            "sem_non_adversarial": sem_non,
            "n_adversarial": int(adv_scores.size),
            "n_non_adversarial": int(non_scores.size),
            "cohen_d": _cohen_d(adv_scores, non_scores),
            "p_value": p_value,
        },
        "probe_metrics": {},
    }

    for probe_name, metric_map in metrics_by_probe.items():
        safe_name = safe_slug(probe_name)
        stats["probe_metrics"][probe_name] = {}
        for metric_name, values in metric_map.items():
            y = np.array(values, dtype=np.float32)
            metric_stats = _metric_stats(x, y)
            stats["probe_metrics"][probe_name][metric_name] = metric_stats

            plot_path = os.path.join(plots_dir, f"{safe_name}_{metric_name}.png")
            _plot_metric_scatter(
                x,
                y,
                categories,
                plot_path,
                title=f"{probe_name} vs correctness ({metric_name})",
                xlabel="Correctness rating",
                ylabel=metric_name,
            )

    json_dump(os.path.join(output_dir, "stats.json"), stats)
    json_dump(os.path.join(output_dir, "per_item.json"), {"items": per_item})

    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
