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


def _r_squared(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    if linregress is not None:
        res = linregress(x, y)
        return float(res.rvalue ** 2)
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr ** 2)


def _plot_line(
    xs: List[float],
    ys_in: List[Optional[float]],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    xs_plot: List[float] = []
    ys_plot: List[float] = []
    for x, y in zip(xs, ys_in):
        if y is None or np.isnan(y):
            continue
        xs_plot.append(float(x))
        ys_plot.append(float(y))
    if not xs_plot:
        raise ValueError("No valid points to plot.")
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(xs_plot, ys_plot, marker="o", linewidth=1.6)
    plt.xlabel("Alpha")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _collect_by_alpha(
    results: List[Dict[str, object]],
    *,
    probe_name: Optional[str],
    metric_key: str,
    rating_key: str,
) -> Dict[float, List[Tuple[float, float]]]:
    by_alpha: Dict[float, List[Tuple[float, float]]] = {}
    for rec in results:
        alpha = rec.get("alpha")
        if not isinstance(alpha, (int, float)):
            continue
        rating = rec.get(rating_key)
        if not isinstance(rating, (int, float)):
            continue
        metric_val: Optional[float] = None
        if probe_name is not None:
            probe_metrics = rec.get("probe_metrics", {}) or {}
            probe_block = probe_metrics.get(probe_name, {})
            if isinstance(probe_block, dict):
                raw_metric = probe_block.get(metric_key)
                if isinstance(raw_metric, (int, float)):
                    metric_val = float(raw_metric)
        else:
            raw_metric = rec.get(metric_key)
            if isinstance(raw_metric, (int, float)):
                metric_val = float(raw_metric)
        if not isinstance(metric_val, (int, float)):
            continue
        by_alpha.setdefault(float(alpha), []).append((float(rating), float(metric_val)))
    return by_alpha


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot R^2-vs-alpha and report-variance-vs-alpha from analysis results.json."
    )
    parser.add_argument("--analysis-dir", required=True, help="Analysis folder with results JSON")
    parser.add_argument(
        "--results-file",
        default=None,
        help="Optional explicit results filename (default: auto-detect).",
    )
    parser.add_argument(
        "--probe-name",
        default=None,
        help="Probe name in probe_metrics. Omit for flat result files with top-level metric keys.",
    )
    parser.add_argument(
        "--metric-key",
        default="prompt_assistant_last_mean",
        help="Metric key inside probe_metrics[probe_name]",
    )
    parser.add_argument("--rating-key", default="rating", help="Rating key in results.json")
    parser.add_argument("--out-r2", default="r2_vs_alpha.png")
    parser.add_argument("--out-var", default="report_variance_vs_alpha.png")
    parser.add_argument("--out-json", default="r2_and_variance_vs_alpha.json")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir).resolve()
    if args.results_file:
        results_path = analysis_dir / args.results_file
    else:
        candidates = [
            "results.json",
            "truthfulness_confidence_results.json",
            "interest_results.json",
            "wellbeing_results.json",
        ]
        results_path = None
        for name in candidates:
            path = analysis_dir / name
            if path.exists():
                results_path = path
                break
        if results_path is None:
            raise FileNotFoundError(
                f"No known results file found in {analysis_dir}. Tried: {', '.join(candidates)}"
            )
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {analysis_dir}")

    results = _load_json(results_path)
    if not isinstance(results, list):
        raise ValueError("results.json must contain a list.")

    by_alpha = _collect_by_alpha(
        results,
        probe_name=args.probe_name,
        metric_key=args.metric_key,
        rating_key=args.rating_key,
    )
    alpha_vals = sorted(by_alpha.keys())

    r2_vals: List[Optional[float]] = []
    variance_vals: List[Optional[float]] = []
    counts: List[int] = []
    for alpha in alpha_vals:
        pairs = by_alpha.get(alpha, [])
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        r2_vals.append(_r_squared(xs, ys))
        counts.append(len(xs))
        if len(xs) < 2:
            variance_vals.append(None)
        else:
            variance_vals.append(float(np.var(np.array(xs, dtype=float), ddof=1)))

    r2_path = analysis_dir / args.out_r2
    _plot_line(
        alpha_vals,
        r2_vals,
        title=f"R^2 vs alpha: {args.probe_name} {args.metric_key} vs {args.rating_key}",
        ylabel="R^2",
        out_path=r2_path,
    )

    var_path = analysis_dir / args.out_var
    _plot_line(
        alpha_vals,
        variance_vals,
        title=f"Report variance vs alpha ({args.rating_key})",
        ylabel="Variance",
        out_path=var_path,
    )

    out = {
        "analysis_dir": str(analysis_dir),
        "results_file": str(results_path.name),
        "probe_name": args.probe_name,
        "metric_key": args.metric_key,
        "rating_key": args.rating_key,
        "alphas": alpha_vals,
        "r2_values": r2_vals,
        "report_variance_values": variance_vals,
        "counts": counts,
        "plot_r2_path": str(r2_path),
        "plot_variance_path": str(var_path),
    }
    (analysis_dir / args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {r2_path}")
    print(f"Wrote {var_path}")


if __name__ == "__main__":
    main()
