# Coherence rating count by alpha

import argparse
import json
from pathlib import Path
from typing import Dict, List

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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot coherence rating counts per alpha.")
    parser.add_argument("batch_dir", help="Path to batch directory.")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    per_sample_path = batch_dir / "analysis" / "per_sample.json"
    ratings_path = batch_dir / "coherence_rating.json"
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
    counts: Dict[str, List[int]] = {k: [0] * len(alpha_vals) for k in RATING_COLORS}

    for item in per_sample:
        example = Path(item["npz_path"]).name
        rating = ratings.get(example)
        if rating not in counts:
            continue
        alpha = float(item["alpha"])
        idx = alpha_vals.index(alpha)
        counts[rating][idx] += 1

    if plt is None:
        print("matplotlib not available; counts:")
        print(json.dumps({"alphas": alpha_vals, "counts": counts}, indent=2))
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for rating, color in RATING_COLORS.items():
        ax.plot(alpha_vals, counts[rating], marker="o", linewidth=2, color=color, label=rating)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Count")
    ax.set_title("Coherence rating count by alpha")
    ax.grid(True, alpha=0.25)
    ax.legend()

    plots_dir = batch_dir / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "coherence_counts_by_alpha.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
