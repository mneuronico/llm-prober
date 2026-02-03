import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _load_json(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Results file must be a list.")
    return data


def _mean_sem(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if len(values) < 2:
        return (mean, 0.0)
    sem = float(arr.std(ddof=1) / math.sqrt(len(values)))
    return (mean, sem)


def _plot_series(
    xs: List[int],
    means: List[float],
    sems: List[float],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    plt.figure(figsize=(7.2, 4.2))
    plt.errorbar(xs, means, yerr=sems, marker="o", linewidth=1.6, capsize=3)
    plt.xlabel("Turno del asistente")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean last-assistant probe score vs assistant turn (interest run)."
    )
    parser.add_argument("--results", required=True, help="Path to interest_results.json or posthoc results.")
    parser.add_argument("--output", default="interest_last_assistant_probe_vs_turn.png")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results_path = Path(args.results).resolve()
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    results = _load_json(results_path)
    by_turn: Dict[int, List[float]] = {}
    for rec in results:
        if not isinstance(rec, dict):
            continue
        turn = rec.get("turn_index")
        score = rec.get("last_assistant_probe_mean")
        if isinstance(turn, int) and isinstance(score, (int, float)):
            by_turn.setdefault(turn, []).append(float(score))

    if not by_turn:
        raise ValueError("No last_assistant_probe_mean values found in results.")

    turns = sorted(by_turn)
    means: List[float] = []
    sems: List[float] = []
    for turn in turns:
        mean, sem = _mean_sem(by_turn[turn])
        means.append(mean)
        sems.append(sem)

    output_path = Path(args.output).resolve()
    _plot_series(
        turns,
        means,
        sems,
        title="Probe score (último mensaje asistente) vs turno",
        ylabel="Probe score (último mensaje asistente)",
        out_path=output_path,
    )
    print(f"Wrote plot to {output_path}")


if __name__ == "__main__":
    main()
