import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _load_stats(path: Path) -> Dict[str, object]:
    if path.is_dir():
        stats_path = path / "analysis" / "stats.json"
    else:
        stats_path = path
    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json not found at {stats_path}")
    return json.loads(stats_path.read_text(encoding="utf-8"))


def _extract_series(stats: Dict[str, object]) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    rows = stats.get("by_alpha", [])
    if not isinstance(rows, list):
        raise ValueError("stats.json missing by_alpha list.")
    rows_sorted = sorted(rows, key=lambda r: float(r.get("alpha", 0.0)))
    alphas = [float(r["alpha"]) for r in rows_sorted]
    mean_a = [float(r["mean_a"]) for r in rows_sorted]
    mean_b = [float(r["mean_b"]) for r in rows_sorted]
    sem_a = [float(r["sem_a"]) for r in rows_sorted]
    sem_b = [float(r["sem_b"]) for r in rows_sorted]
    return alphas, mean_a, sem_a, mean_b, sem_b


def _plot_series(
    ax,
    alphas: List[float],
    means: List[float],
    sems: List[float],
    *,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    ax.errorbar(
        alphas,
        means,
        yerr=sems,
        marker="o",
        linewidth=2,
        markersize=4,
        capsize=3,
        color=color,
        linestyle=linestyle,
        label=label,
    )


def plot_alpha_probe_comparison(
    fabrication_path: Path,
    lying_path: Path,
    out_path: Path,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available.")

    fabrication_stats = _load_stats(fabrication_path)
    lying_stats = _load_stats(lying_path)

    f_alphas, f_mean_true, f_sem_true, f_mean_made, f_sem_made = _extract_series(fabrication_stats)
    l_alphas, l_mean_true, l_sem_true, l_mean_made, l_sem_made = _extract_series(lying_stats)

    fig, ax = plt.subplots(figsize=(7, 4))

    _plot_series(
        ax,
        f_alphas,
        f_mean_true,
        f_sem_true,
        color="#2c8d5b",
        linestyle="-",
        label="Fabrication vs truthfulness — true questions",
    )
    _plot_series(
        ax,
        f_alphas,
        f_mean_made,
        f_sem_made,
        color="#b24a3b",
        linestyle="-",
        label="Fabrication vs truthfulness — made up questions",
    )
    _plot_series(
        ax,
        l_alphas,
        l_mean_true,
        l_sem_true,
        color="#2c8d5b",
        linestyle="--",
        label="Lying vs truthfulness — true questions",
    )
    _plot_series(
        ax,
        l_alphas,
        l_mean_made,
        l_sem_made,
        color="#b24a3b",
        linestyle="--",
        label="Lying vs truthfulness — made up questions",
    )

    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Mean probe score")
    ax.set_title("Probe score vs steering alpha")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot probe score vs alpha for fabrication vs truthfulness and lying vs truthfulness runs."
    )
    parser.add_argument(
        "--fabrication",
        type=Path,
        default=Path("outputs/fabrication_vs_truthfulness/20260115_180659/factuality_eval"),
        help="Path to fabrication-vs-truthfulness run (batch dir or stats.json path).",
    )
    parser.add_argument(
        "--lying",
        type=Path,
        default=Path("outputs/lying_vs_truthfulness/20260116_095854/lying_eval"),
        help="Path to lying-vs-truthfulness run (batch dir or stats.json path).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/factuality_vs_lying_alpha_probe_scores.png"),
        help="Output image path.",
    )
    args = parser.parse_args()
    plot_alpha_probe_comparison(args.fabrication, args.lying, args.out)


if __name__ == "__main__":
    main()
