import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.stats import linregress
except Exception:
    linregress = None

def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _linear_slope_p(xs: List[float], ys: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(xs) < 2 or len(ys) < 2:
        return (None, None)
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.allclose(x, x[0]):
        return (None, None)
    if linregress is None:
        slope, _ = np.polyfit(x, y, 1)
        return (float(slope), None)
    res = linregress(x, y)
    return (float(res.slope), float(res.pvalue))


def _plot(alpha_vals: List[float], ys_in: List[Optional[float]], out_path: Path, *, ylabel: str, title: str) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    xs: List[float] = []
    ys: List[float] = []
    for a, s in zip(alpha_vals, ys_in):
        if s is None or np.isnan(s):
            continue
        xs.append(a)
        ys.append(s)
    if not xs:
        raise ValueError("No valid values to plot.")
    plt.figure(figsize=(6.6, 4.2))
    plt.plot(xs, ys, marker="o", linewidth=1.6)
    plt.xlabel("Alpha (fabrication vs truthfulness steering)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot slope and p-values of sad_vs_happy assistant last mean vs wellbeing rating across alphas."
    )
    parser.add_argument("--analysis-dir", required=True, help="Analysis folder with results.json")
    parser.add_argument(
        "--probe-name",
        default="sad_vs_happy",
        help="Probe name key inside results.json probe_metrics",
    )
    parser.add_argument(
        "--metric-key",
        default="prompt_assistant_last_mean",
        help="Metric key inside probe_metrics[probe_name]",
    )
    parser.add_argument(
        "--rating-key",
        default="rating",
        help="Rating key in results.json (default: rating)",
    )
    parser.add_argument(
        "--out-name",
        default="alignment_slope_vs_alpha.png",
        help="Output slope plot filename (saved inside analysis-dir)",
    )
    parser.add_argument(
        "--out-name-p",
        default="alignment_slope_pvalue_vs_alpha.png",
        help="Output p-value plot filename (saved inside analysis-dir)",
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir).resolve()
    results_path = analysis_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {analysis_dir}")

    results = _load_json(results_path)
    if not isinstance(results, list):
        raise ValueError("results.json must be a list of records.")

    by_alpha: Dict[float, List[Tuple[float, float]]] = {}
    for rec in results:
        alpha = rec.get("alpha")
        if not isinstance(alpha, (int, float)):
            continue
        rating = rec.get(args.rating_key)
        if not isinstance(rating, (int, float)):
            continue
        probe_metrics = rec.get("probe_metrics", {}) or {}
        probe_block = probe_metrics.get(args.probe_name, {})
        if not isinstance(probe_block, dict):
            continue
        metric_val = probe_block.get(args.metric_key)
        if not isinstance(metric_val, (int, float)):
            continue
        by_alpha.setdefault(float(alpha), []).append((float(rating), float(metric_val)))

    alpha_vals = sorted(by_alpha.keys())
    slopes: List[Optional[float]] = []
    pvals: List[Optional[float]] = []
    for alpha in alpha_vals:
        pairs = by_alpha.get(alpha, [])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        slope, pval = _linear_slope_p(xs, ys)
        slopes.append(slope)
        pvals.append(pval)

    out_path = analysis_dir / args.out_name
    _plot(
        alpha_vals,
        slopes,
        out_path,
        ylabel="Slope: sad_vs_happy prompt_assistant_last_mean vs wellbeing rating",
        title="Alignment Slope vs Steering Alpha",
    )

    out_path_p = analysis_dir / args.out_name_p
    if any(p is not None for p in pvals):
        _plot(
            alpha_vals,
            pvals,
            out_path_p,
            ylabel="P-value (slope != 0)",
            title="Alignment Slope P-value vs Steering Alpha",
        )

    summary = {
        "analysis_dir": str(analysis_dir),
        "probe_name": args.probe_name,
        "metric_key": args.metric_key,
        "rating_key": args.rating_key,
        "alpha_vals": alpha_vals,
        "slopes": slopes,
        "pvalues": pvals,
        "plot_path": str(out_path),
        "plot_path_pvalues": str(out_path_p) if any(p is not None for p in pvals) else None,
    }
    (analysis_dir / "alignment_slope_vs_alpha.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
