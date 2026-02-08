import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.stats import linregress, pearsonr, spearmanr
except Exception:
    linregress = None
    pearsonr = None
    spearmanr = None


DEFAULT_STATS = [
    "count",
    "slope",
    "r2",
    "linreg_p",
    "pearson_r",
    "pearson_p",
    "spearman_rho",
    "spearman_p",
]

STAT_YLABEL = {
    "count": "Sample count",
    "slope": "Slope (metric vs rating)",
    "intercept": "Intercept (metric vs rating)",
    "linreg_r": "Linear regression r",
    "linreg_p": "Linear regression p-value",
    "r2": "R^2",
    "pearson_r": "Pearson r",
    "pearson_p": "Pearson p-value",
    "spearman_rho": "Spearman rho",
    "spearman_p": "Spearman p-value",
}

STAT_ALIASES = {
    "p": "linreg_p",
    "p_value": "linreg_p",
    "r": "linreg_r",
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


def _sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    slug = slug.strip("_")
    return slug or "value"


def _resolve_results_path(analysis_dir: Path, results_file: Optional[str]) -> Path:
    if results_file:
        path = analysis_dir / results_file
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        return path
    candidates = [
        "results.json",
        "truthfulness_confidence_results.json",
        "interest_results.json",
        "wellbeing_results.json",
    ]
    for name in candidates:
        path = analysis_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No known results file found in {analysis_dir}. Tried: {', '.join(candidates)}"
    )


def _discover_probe_names(results: List[Dict[str, object]]) -> List[str]:
    names: Dict[str, bool] = {}
    for rec in results:
        if not isinstance(rec, dict):
            continue
        block = rec.get("probe_metrics")
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            if isinstance(key, str) and isinstance(value, dict):
                names[key] = True
    return sorted(names.keys())


def _select_probe_name(
    results: List[Dict[str, object]],
    requested_probe_name: Optional[str],
) -> Optional[str]:
    probe_names = _discover_probe_names(results)
    if requested_probe_name:
        if not probe_names:
            return requested_probe_name
        if requested_probe_name not in probe_names:
            raise ValueError(
                f"Probe '{requested_probe_name}' not found. Available probes: {probe_names}"
            )
        return requested_probe_name
    if not probe_names:
        return requested_probe_name
    if len(probe_names) == 1:
        return probe_names[0]
    raise ValueError(
        f"Multiple probes found ({probe_names}). Pass --probe-name to disambiguate."
    )


def _parse_alpha_filter(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None or raw.strip() == "":
        return None
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values or None


def _parse_stats(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(DEFAULT_STATS)
    parsed: List[str] = []
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        key = STAT_ALIASES.get(key, key)
        parsed.append(key)
    if not parsed:
        return list(DEFAULT_STATS)
    unknown = [k for k in parsed if k not in STAT_YLABEL]
    if unknown:
        raise ValueError(
            f"Unknown stats: {unknown}. Allowed: {sorted(STAT_YLABEL.keys())} or 'all'."
        )
    seen: Dict[str, bool] = {}
    deduped: List[str] = []
    for key in parsed:
        if key in seen:
            continue
        seen[key] = True
        deduped.append(key)
    return deduped


def _extract_metric(
    rec: Dict[str, object],
    *,
    probe_name: Optional[str],
    metric_key: str,
) -> Optional[float]:
    probe_metrics = rec.get("probe_metrics")
    if isinstance(probe_metrics, dict) and probe_name is not None:
        block = probe_metrics.get(probe_name)
        if isinstance(block, dict):
            from_probe = _safe_float(block.get(metric_key))
            if from_probe is not None:
                return from_probe
    return _safe_float(rec.get(metric_key))


def _infer_rating_key(results: List[Dict[str, object]], requested: str) -> str:
    if requested != "auto":
        return requested
    candidates = [
        "rating",
        "truthfulness_confidence_score",
        "interest_score",
        "wellbeing_score",
        "focus_score",
        "empathy_score",
    ]
    for key in candidates:
        for rec in results:
            if _safe_float(rec.get(key)) is not None:
                return key
    raise ValueError(
        "Could not infer rating key. Pass --rating-key explicitly."
    )


def _infer_metric_key(
    results: List[Dict[str, object]],
    *,
    probe_name: Optional[str],
    requested: str,
) -> str:
    if requested != "auto":
        return requested

    # Preferred keys for "last assistant turn probe mean score".
    preferred = [
        "prompt_assistant_last_mean",
        "last_assistant_probe_mean",
        "probe_score_mean",
        "completion_assistant_mean",
    ]

    for key in preferred:
        for rec in results:
            if _extract_metric(rec, probe_name=probe_name, metric_key=key) is not None:
                return key

    # Last resort: first numeric key in the selected probe block.
    if probe_name is not None:
        for rec in results:
            block = (rec.get("probe_metrics") or {}).get(probe_name)
            if isinstance(block, dict):
                for key, value in block.items():
                    if _safe_float(value) is not None and isinstance(key, str):
                        return key

    # Last resort: first numeric top-level key.
    for rec in results:
        for key, value in rec.items():
            if isinstance(key, str) and _safe_float(value) is not None:
                if key in ("alpha", "turn_index", "assistant_message_index", "conversation_index"):
                    continue
                return key

    raise ValueError(
        "Could not infer metric key. Pass --metric-key explicitly."
    )


def _linreg_stats(xs: List[float], ys: List[float]) -> Dict[str, Optional[float]]:
    if len(xs) < 2 or len(ys) < 2:
        return {
            "slope": None,
            "intercept": None,
            "linreg_r": None,
            "linreg_p": None,
            "r2": None,
        }
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.allclose(x, x[0]):
        return {
            "slope": None,
            "intercept": None,
            "linreg_r": None,
            "linreg_p": None,
            "r2": None,
        }
    if linregress is not None:
        try:
            res = linregress(x, y)
            r_val = float(res.rvalue)
            p_val = float(res.pvalue)
            return {
                "slope": float(res.slope),
                "intercept": float(res.intercept),
                "linreg_r": r_val if math.isfinite(r_val) else None,
                "linreg_p": p_val if math.isfinite(p_val) else None,
                "r2": float(r_val**2) if math.isfinite(r_val) else None,
            }
        except Exception:
            pass

    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return {
            "slope": None,
            "intercept": None,
            "linreg_r": None,
            "linreg_p": None,
            "r2": None,
        }

    corr = np.corrcoef(x, y)[0, 1]
    corr_f = float(corr) if math.isfinite(float(corr)) else None
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "linreg_r": corr_f,
        "linreg_p": None,
        "r2": float(corr_f**2) if corr_f is not None else None,
    }


def _corr_stats(xs: List[float], ys: List[float]) -> Dict[str, Optional[float]]:
    out = {
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
    }
    if len(xs) < 2 or len(ys) < 2:
        return out

    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return out

    if pearsonr is not None:
        try:
            pr = pearsonr(x, y)
            pr_r = float(pr.statistic)
            pr_p = float(pr.pvalue)
            out["pearson_r"] = pr_r if math.isfinite(pr_r) else None
            out["pearson_p"] = pr_p if math.isfinite(pr_p) else None
        except Exception:
            pass
    else:
        pr = np.corrcoef(x, y)[0, 1]
        pr_f = float(pr) if math.isfinite(float(pr)) else None
        out["pearson_r"] = pr_f

    if spearmanr is not None:
        try:
            sr = spearmanr(x, y)
            sr_rho = float(sr.correlation)
            sr_p = float(sr.pvalue)
            out["spearman_rho"] = sr_rho if math.isfinite(sr_rho) else None
            out["spearman_p"] = sr_p if math.isfinite(sr_p) else None
        except Exception:
            pass
    return out


def _compute_stats_for_pairs(xs: List[float], ys: List[float]) -> Dict[str, Optional[float]]:
    stats: Dict[str, Optional[float]] = {"count": float(len(xs))}
    stats.update(_linreg_stats(xs, ys))
    stats.update(_corr_stats(xs, ys))
    return stats


def _plot_stat_vs_turn(
    *,
    turns: Sequence[int],
    series_by_alpha: Dict[float, List[Optional[float]]],
    stat_key: str,
    title: str,
    out_path: Path,
) -> bool:
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    plt.figure(figsize=(8.0, 4.8))
    plotted = False
    for alpha in sorted(series_by_alpha.keys()):
        ys_raw = series_by_alpha[alpha]
        xs_plot: List[int] = []
        ys_plot: List[float] = []
        for turn, value in zip(turns, ys_raw):
            if value is None or not math.isfinite(value):
                continue
            xs_plot.append(int(turn))
            ys_plot.append(float(value))
        if not xs_plot:
            continue
        plotted = True
        plt.plot(xs_plot, ys_plot, marker="o", linewidth=1.5, label=f"alpha={alpha:g}")
    if not plotted:
        plt.close()
        return False
    plt.xlabel("Turn")
    plt.ylabel(STAT_YLABEL.get(stat_key, stat_key))
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute turn-wise relationship stats between rating and a probe metric, "
            "with one curve per alpha, from an existing conversation analysis folder."
        )
    )
    parser.add_argument("--analysis-dir", required=True, help="Folder containing results.json")
    parser.add_argument(
        "--results-file",
        default=None,
        help="Optional results filename inside analysis-dir. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--probe-name",
        required=True,
        help=(
            "Probe name inside probe_metrics (e.g. bored_vs_interested)."
        ),
    )
    parser.add_argument(
        "--metric-key",
        default="auto",
        help=(
            "Metric key from probe_metrics[probe_name] or top-level metric. "
            "Default 'auto' infers a last-assistant metric."
        ),
    )
    parser.add_argument(
        "--rating-key",
        default="auto",
        help="Key for self-report value in each record. Default 'auto' infers common keys.",
    )
    parser.add_argument(
        "--turn-key",
        default="turn_index",
        help="Key for turn index in each record.",
    )
    parser.add_argument(
        "--alphas",
        default=None,
        help="Optional comma-separated alpha filter (e.g. -6,-3,0,3,6).",
    )
    parser.add_argument(
        "--stats",
        default="all",
        help=(
            "Comma-separated stats to plot. Use 'all' for defaults. "
            "Available: count,slope,intercept,linreg_r,linreg_p,r2,pearson_r,pearson_p,spearman_rho,spearman_p"
        ),
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Optional filename prefix. Default is derived from probe+metric.",
    )
    parser.add_argument(
        "--out-json",
        default="turnwise_relationship_vs_alpha.json",
        help="Output JSON summary filename (inside analysis-dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analysis_dir = Path(args.analysis_dir).resolve()
    results_path = _resolve_results_path(analysis_dir, args.results_file)
    payload = _load_json(results_path)
    if not isinstance(payload, list):
        raise ValueError(f"{results_path.name} must contain a list of records.")
    results: List[Dict[str, object]] = [r for r in payload if isinstance(r, dict)]
    if not results:
        raise ValueError("No valid records found in results file.")

    probe_name = _select_probe_name(results, args.probe_name)
    resolved_rating_key = _infer_rating_key(results, args.rating_key)
    resolved_metric_key = _infer_metric_key(
        results,
        probe_name=probe_name,
        requested=args.metric_key,
    )
    stats_to_plot = _parse_stats(args.stats)
    alpha_filter = _parse_alpha_filter(args.alphas)
    alpha_filter_set = set(alpha_filter) if alpha_filter is not None else None

    by_alpha_turn: Dict[float, Dict[int, List[Tuple[float, float]]]] = {}
    for rec in results:
        alpha = _safe_float(rec.get("alpha"))
        if alpha is None:
            continue
        if alpha_filter_set is not None and float(alpha) not in alpha_filter_set:
            continue

        turn_raw = rec.get(args.turn_key)
        if not isinstance(turn_raw, int):
            continue
        turn = int(turn_raw)
        rating = _safe_float(rec.get(resolved_rating_key))
        metric = _extract_metric(rec, probe_name=probe_name, metric_key=resolved_metric_key)
        if rating is None or metric is None:
            continue
        by_alpha_turn.setdefault(alpha, {}).setdefault(turn, []).append((rating, metric))

    if not by_alpha_turn:
        raise ValueError(
            "No valid (rating, metric) pairs found after filtering. "
            "Check --probe-name/--metric-key/--rating-key/--turn-key/--alphas."
        )

    alphas = sorted(by_alpha_turn.keys())
    turn_set = {t for turns_block in by_alpha_turn.values() for t in turns_block.keys()}
    turns = sorted(turn_set)

    stats_by_alpha_turn: Dict[str, List[Dict[str, Optional[float]]]] = {}
    series_by_stat: Dict[str, Dict[float, List[Optional[float]]]] = {
        key: {a: [] for a in alphas} for key in stats_to_plot
    }

    for alpha in alphas:
        rows_for_alpha: List[Dict[str, Optional[float]]] = []
        for turn in turns:
            pairs = by_alpha_turn.get(alpha, {}).get(turn, [])
            xs = [x for x, _ in pairs]
            ys = [y for _, y in pairs]
            stats = _compute_stats_for_pairs(xs, ys)
            row: Dict[str, Optional[float]] = {"turn": int(turn)}
            if stats.get("count") is not None:
                stats["count"] = int(stats["count"])
            row.update(stats)
            rows_for_alpha.append(row)
            for stat_key in stats_to_plot:
                series_by_stat[stat_key][alpha].append(stats.get(stat_key))
        stats_by_alpha_turn[str(alpha)] = rows_for_alpha

    prefix = args.out_prefix
    if not prefix:
        probe_tag = _sanitize_slug(probe_name) if probe_name else "flat"
        metric_tag = _sanitize_slug(resolved_metric_key)
        prefix = f"turnwise_{probe_tag}_{metric_tag}"

    plot_paths: Dict[str, Optional[str]] = {}
    for stat_key in stats_to_plot:
        out_path = analysis_dir / f"{prefix}_{stat_key}_vs_turn_by_alpha.png"
        made_plot = _plot_stat_vs_turn(
            turns=turns,
            series_by_alpha=series_by_stat[stat_key],
            stat_key=stat_key,
            title=f"{stat_key} vs turn (curves by alpha)",
            out_path=out_path,
        )
        plot_paths[stat_key] = str(out_path) if made_plot else None

    out = {
        "analysis_dir": str(analysis_dir),
        "results_file": results_path.name,
        "probe_name": probe_name,
        "metric_key": resolved_metric_key,
        "rating_key": resolved_rating_key,
        "turn_key": args.turn_key,
        "alphas": alphas,
        "turns": turns,
        "stats": stats_to_plot,
        "stats_by_alpha_turn": stats_by_alpha_turn,
        "plot_paths": plot_paths,
    }
    out_json_path = analysis_dir / args.out_json
    out_json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_json_path}")
    for stat_key in stats_to_plot:
        path = plot_paths.get(stat_key)
        if path:
            print(f"Wrote {path}")
        else:
            print(f"Skipped {stat_key}: no plottable values")


if __name__ == "__main__":
    main()
