import argparse
import json
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

try:
    import statsmodels.api as sm
except Exception:
    sm = None


@dataclass
class AssistantTokens:
    token_indices: List[int]
    token_texts: List[str]


class _AssistantTokenParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._current_role: Optional[str] = None
        self._in_role_div = False
        self._in_tokens_div = False
        self._in_span = False
        self._span_idx: Optional[int] = None
        self._span_text_parts: List[str] = []
        self.tokens: List[Tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "div":
            cls = _attr_value(attrs, "class")
            if cls == "role":
                self._in_role_div = True
            elif cls == "tokens" and self._current_role == "assistant":
                self._in_tokens_div = True
        elif tag == "span" and self._in_tokens_div:
            cls = _attr_value(attrs, "class") or ""
            if "tok" in cls.split():
                idx_val = _attr_value(attrs, "data-idx")
                if idx_val is not None:
                    try:
                        self._span_idx = int(idx_val)
                        self._in_span = True
                        self._span_text_parts = []
                    except ValueError:
                        self._span_idx = None

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            if self._in_role_div:
                self._in_role_div = False
            elif self._in_tokens_div:
                self._in_tokens_div = False
        elif tag == "span" and self._in_span:
            text = "".join(self._span_text_parts)
            if self._span_idx is not None:
                self.tokens.append((self._span_idx, text))
            self._in_span = False
            self._span_idx = None
            self._span_text_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_role_div:
            role = data.strip().lower()
            if role:
                self._current_role = role
        elif self._in_span:
            self._span_text_parts.append(data)


def _attr_value(attrs: List[Tuple[str, Optional[str]]], name: str) -> Optional[str]:
    for key, value in attrs:
        if key == name:
            return value
    return None


def _parse_assistant_tokens(html_path: Path) -> AssistantTokens:
    parser = _AssistantTokenParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    parser.close()
    if not parser.tokens:
        return AssistantTokens([], [])
    parser.tokens.sort(key=lambda t: t[0])
    token_indices = [idx for idx, _ in parser.tokens]
    token_texts = [text for _, text in parser.tokens]
    return AssistantTokens(token_indices, token_texts)


def _find_probe_indices(probe_names: np.ndarray) -> Dict[str, int]:
    names = [str(n) for n in probe_names.tolist()]
    indices = {}
    for name in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
        if name not in names:
            raise ValueError(f"Probe {name} not found in npz probe_names.")
        indices[name] = names.index(name)
    return indices


def _sem(values: List[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _collect_html_files(batch_dir: Path) -> List[Path]:
    html_files = []
    for path in batch_dir.glob("prompt_*.html"):
        if path.name == "batch_prompts.html":
            continue
        html_files.append(path)
    return sorted(html_files)


def _linear_stats(points: List[Tuple[int, float]]) -> Dict[str, float]:
    if len(points) < 3:
        return {"n": len(points), "slope": float("nan"), "intercept": float("nan"), "r": float("nan"), "p": float("nan")}
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    if np.std(x) < 1e-12:
        return {"n": len(points), "slope": float("nan"), "intercept": float("nan"), "r": float("nan"), "p": float("nan")}
    if scipy_stats is None:
        slope, intercept = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        return {"n": len(points), "slope": float(slope), "intercept": float(intercept), "r": float(r), "p": float("nan")}
    result = scipy_stats.linregress(x, y)
    return {
        "n": int(len(points)),
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r": float(result.rvalue),
        "p": float(result.pvalue),
    }


def _clustered_stats_from_slopes(slopes: List[float]) -> Dict[str, float]:
    if not slopes:
        return {"n": 0, "mean_slope": float("nan"), "t": float("nan"), "p": float("nan")}
    arr = np.array(slopes, dtype=np.float64)
    mean = float(np.mean(arr))
    if arr.size < 2 or scipy_stats is None:
        return {"n": int(arr.size), "mean_slope": mean, "t": float("nan"), "p": float("nan")}
    t_stat, p_val = scipy_stats.ttest_1samp(arr, 0.0)
    return {"n": int(arr.size), "mean_slope": mean, "t": float(t_stat), "p": float(p_val)}


def _compute_clustered_stats(
    html_files: List[Path],
    type_by_npz: Dict[Path, str],
    base_types: List[str],
    *,
    include_prompt_prefix: bool,
) -> Dict[str, object]:
    slopes: Dict[str, Dict[str, List[float]]] = {
        "fabrication_vs_truthfulness": {t: [] for t in base_types},
        "lying_vs_truthfulness": {t: [] for t in base_types},
    }

    for html_path in html_files:
        npz_path = html_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        assistant = _parse_assistant_tokens(html_path)
        if not assistant.token_indices:
            continue

        with np.load(npz_path) as data:
            scores = data["scores_agg"]
            probe_indices = _find_probe_indices(data["probe_names"])
            prompt_len = int(data["prompt_len"][0]) if "prompt_len" in data else None
        if prompt_len is None:
            continue

        if base_types == ["adversarial", "non_adversarial"]:
            item_type = type_by_npz.get(npz_path.resolve())
            if item_type not in ("adversarial", "non_adversarial"):
                continue
        else:
            item_type = "all"

        if include_prompt_prefix:
            completion_indices = assistant.token_indices
        else:
            completion_indices = [idx for idx in assistant.token_indices if idx >= prompt_len]

        if len(completion_indices) < 3:
            continue

        x = np.arange(len(completion_indices), dtype=np.float64)
        for probe_name, probe_idx in probe_indices.items():
            y_vals = []
            for global_idx in completion_indices:
                if global_idx >= scores.shape[1]:
                    continue
                y_vals.append(float(scores[probe_idx, global_idx]))
            if len(y_vals) != len(x):
                min_len = min(len(y_vals), len(x))
                if min_len < 3:
                    continue
                y = np.array(y_vals[:min_len], dtype=np.float64)
                x_use = x[:min_len]
            else:
                y = np.array(y_vals, dtype=np.float64)
                x_use = x
            if np.std(x_use) < 1e-12:
                continue
            slope, _ = np.polyfit(x_use, y, 1)
            slopes[probe_name][item_type].append(float(slope))

    stats: Dict[str, object] = {"probes": {}}
    for probe_name in slopes:
        stats["probes"][probe_name] = {}
        for item_type, vals in slopes[probe_name].items():
            stats["probes"][probe_name][item_type] = _clustered_stats_from_slopes(vals)
    return stats


def _print_clustered_summary(stats: Dict[str, object]) -> None:
    for probe_name in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
        for item_type, s in stats.get("probes", {}).get(probe_name, {}).items():
            label = probe_name if item_type == "all" else f"{probe_name} ({item_type})"
            print(
                f"{label} clustered: n={s.get('n')} mean_slope={s.get('mean_slope'):.4g} t={s.get('t'):.4g} p={s.get('p'):.4g}"
            )


def _fit_mixedlm(x_vals: List[float], y_vals: List[float], groups: List[str]) -> Dict[str, float]:
    if sm is None:
        return {"n": len(y_vals), "groups": len(set(groups)), "slope": float("nan"), "p": float("nan"), "converged": False}
    if len(y_vals) < 3 or len(set(groups)) < 2:
        return {"n": len(y_vals), "groups": len(set(groups)), "slope": float("nan"), "p": float("nan"), "converged": False}
    x = np.array(x_vals, dtype=np.float64)
    y = np.array(y_vals, dtype=np.float64)
    x_center = x - np.mean(x)
    exog = sm.add_constant(x_center)
    try:
        model = sm.MixedLM(y, exog, groups=groups)
        result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        slope = float(result.params[1]) if len(result.params) > 1 else float("nan")
        pval = float(result.pvalues[1]) if hasattr(result, "pvalues") and len(result.pvalues) > 1 else float("nan")
        return {
            "n": int(len(y_vals)),
            "groups": int(len(set(groups))),
            "slope": slope,
            "p": pval,
            "converged": bool(getattr(result, "converged", False)),
        }
    except Exception:
        return {"n": int(len(y_vals)), "groups": int(len(set(groups))), "slope": float("nan"), "p": float("nan"), "converged": False}


def _print_mixed_summary(stats: Dict[str, object]) -> None:
    all_stats = stats.get("all", {})
    for probe_name in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
        s = all_stats.get(probe_name, {})
        print(
            f"{probe_name} mixed (all): n={s.get('n')} groups={s.get('groups')} slope={s.get('slope'):.4g} p={s.get('p'):.4g} converged={s.get('converged')}"
        )
    for item_type, probes in stats.get("by_type", {}).items():
        for probe_name in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
            s = probes.get(probe_name, {})
            print(
                f"{probe_name} mixed ({item_type}): n={s.get('n')} groups={s.get('groups')} slope={s.get('slope'):.4g} p={s.get('p'):.4g} converged={s.get('converged')}"
            )


def _fit_cluster_robust(x_vals: List[float], y_vals: List[float], groups: List[str]) -> Dict[str, float]:
    if sm is None:
        return {
            "n": len(y_vals),
            "groups": len(set(groups)),
            "slope": float("nan"),
            "se": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
        }
    if len(y_vals) < 3 or len(set(groups)) < 2:
        return {
            "n": len(y_vals),
            "groups": len(set(groups)),
            "slope": float("nan"),
            "se": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
        }
    x = np.array(x_vals, dtype=np.float64)
    y = np.array(y_vals, dtype=np.float64)
    x_center = x - np.mean(x)
    exog = sm.add_constant(x_center)
    try:
        model = sm.OLS(y, exog)
        result = model.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
        slope = float(result.params[1]) if len(result.params) > 1 else float("nan")
        se = float(result.bse[1]) if len(result.bse) > 1 else float("nan")
        t_val = float(result.tvalues[1]) if len(result.tvalues) > 1 else float("nan")
        p_val = float(result.pvalues[1]) if len(result.pvalues) > 1 else float("nan")
        return {
            "n": int(len(y_vals)),
            "groups": int(len(set(groups))),
            "slope": slope,
            "se": se,
            "t": t_val,
            "p": p_val,
        }
    except Exception:
        return {
            "n": int(len(y_vals)),
            "groups": int(len(set(groups))),
            "slope": float("nan"),
            "se": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
        }


def _print_cluster_robust_summary(stats: Dict[str, object]) -> None:
    all_stats = stats.get("all", {})
    for probe_name in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
        s = all_stats.get(probe_name, {})
        print(
            f"{probe_name} cluster-robust (all): n={s.get('n')} groups={s.get('groups')} slope={s.get('slope'):.4g} p={s.get('p'):.4g}"
        )
    for item_type, probes in stats.get("by_type", {}).items():
        for probe_name in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
            s = probes.get(probe_name, {})
            print(
                f"{probe_name} cluster-robust ({item_type}): n={s.get('n')} groups={s.get('groups')} slope={s.get('slope'):.4g} p={s.get('p'):.4g}"
            )


def _compute_slope_diff_cluster_robust(
    mixed_by_type: Dict[str, Dict[str, Dict[str, List[float]]]],
    *,
    include_prompt_prefix: bool,
) -> Dict[str, object]:
    if sm is None:
        return {"error": "statsmodels not available"}

    def _fit_for_probe(probe_name: str) -> Dict[str, float]:
        x_adv = mixed_by_type["adversarial"][probe_name]["x"]
        y_adv = mixed_by_type["adversarial"][probe_name]["y"]
        g_adv = mixed_by_type["adversarial"][probe_name]["group"]
        x_non = mixed_by_type["non_adversarial"][probe_name]["x"]
        y_non = mixed_by_type["non_adversarial"][probe_name]["y"]
        g_non = mixed_by_type["non_adversarial"][probe_name]["group"]

        x = np.array(x_adv + x_non, dtype=np.float64)
        y = np.array(y_adv + y_non, dtype=np.float64)
        groups = np.array(g_adv + g_non, dtype=object)
        t_ind = np.array([1] * len(x_adv) + [0] * len(x_non), dtype=np.float64)
        if x.size < 3:
            return {"n": int(x.size), "groups": int(len(set(groups))), "interaction": float("nan"), "p": float("nan")}

        x_center = x - np.mean(x)
        exog = np.column_stack([np.ones_like(x_center), x_center, t_ind, x_center * t_ind])
        model = sm.OLS(y, exog)
        result = model.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
        interaction = float(result.params[3]) if len(result.params) > 3 else float("nan")
        p_val = float(result.pvalues[3]) if len(result.pvalues) > 3 else float("nan")
        return {
            "n": int(x.size),
            "groups": int(len(set(groups))),
            "interaction": interaction,
            "p": p_val,
        }

    return {
        "settings": {"include_prompt_prefix": bool(include_prompt_prefix)},
        "probes": {
            "fabrication_vs_truthfulness": _fit_for_probe("fabrication_vs_truthfulness"),
            "lying_vs_truthfulness": _fit_for_probe("lying_vs_truthfulness"),
        },
    }


def _print_slope_diff_summary(stats: Dict[str, object]) -> None:
    probes = stats.get("probes", {})
    for probe_name in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
        s = probes.get(probe_name, {})
        print(
            f"{probe_name} slope diff (adv - non): interaction={s.get('interaction'):.4g} p={s.get('p'):.4g} n={s.get('n')} groups={s.get('groups')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze probe score trend over token position across all responses."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs_multi/truthfulqa_multi_probe/20260116_155358"),
        help="Run directory containing truthfulqa_eval batch outputs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs_multi/truthfulqa_multi_probe/20260116_155358/analysis/plots/token_position_trend.png"
        ),
        help="Output plot path.",
    )
    parser.add_argument(
        "--eval-items",
        type=Path,
        default=None,
        help="Path to truthfulqa_eval_items.json for adversarial/non-adversarial split.",
    )
    parser.add_argument(
        "--split-type",
        action="store_true",
        help="Split curves by question type (adversarial vs non-adversarial).",
    )
    parser.add_argument(
        "--clustered-stats",
        action="store_true",
        help="Compute per-response slopes and test mean slope vs 0 (accounts for non-independence).",
    )
    parser.add_argument(
        "--mixed-effects",
        action="store_true",
        help="Fit mixed-effects model with random intercept per response (accounts for non-independence).",
    )
    parser.add_argument(
        "--cluster-robust",
        action="store_true",
        help="Fit OLS with cluster-robust SEs by response (accounts for non-independence).",
    )
    parser.add_argument(
        "--slope-diff",
        action="store_true",
        help="Test whether adversarial vs non-adversarial slopes differ (interaction term).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum number of samples at a token position to include in the plot.",
    )
    parser.add_argument(
        "--include-prompt-prefix",
        action="store_true",
        help="Include assistant tokens that occur before prompt_len (prompt prefix).",
    )
    args = parser.parse_args()

    if plt is None:
        raise RuntimeError("matplotlib is not available.")

    run_dir = args.run_dir
    eval_items_path = args.eval_items or (run_dir / "truthfulqa_eval_items.json")
    type_by_npz: Dict[Path, str] = {}
    if args.split_type:
        if not eval_items_path.exists():
            raise FileNotFoundError(f"truthfulqa_eval_items.json not found at {eval_items_path}")
        data = json.loads(eval_items_path.read_text(encoding="utf-8"))
        for item in data.get("items", []):
            npz_path = item.get("npz_path")
            item_type = item.get("type")
            if not npz_path or not item_type:
                continue
            try:
                type_by_npz[Path(npz_path).resolve()] = str(item_type)
            except Exception:
                continue
    batch_dirs = list((run_dir / "truthfulqa_eval").glob("batch_*"))
    if not batch_dirs:
        raise FileNotFoundError(f"No batch_* directories found under {run_dir / 'truthfulqa_eval'}")
    batch_dir = batch_dirs[0]

    html_files = _collect_html_files(batch_dir)
    if not html_files:
        raise FileNotFoundError(f"No prompt_*.html files found in {batch_dir}")

    base_types = ["all"]
    if args.split_type:
        base_types = ["adversarial", "non_adversarial"]

    per_pos: Dict[str, Dict[str, Dict[int, List[float]]]] = {
        "fabrication_vs_truthfulness": {t: {} for t in base_types},
        "lying_vs_truthfulness": {t: {} for t in base_types},
    }
    points: Dict[str, Dict[str, List[Tuple[int, float]]]] = {
        "fabrication_vs_truthfulness": {t: [] for t in base_types},
        "lying_vs_truthfulness": {t: [] for t in base_types},
    }
    mixed_all: Dict[str, Dict[str, List[float]]] = {
        "fabrication_vs_truthfulness": {"x": [], "y": [], "group": []},
        "lying_vs_truthfulness": {"x": [], "y": [], "group": []},
    }
    mixed_by_type: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        "adversarial": {
            "fabrication_vs_truthfulness": {"x": [], "y": [], "group": []},
            "lying_vs_truthfulness": {"x": [], "y": [], "group": []},
        },
        "non_adversarial": {
            "fabrication_vs_truthfulness": {"x": [], "y": [], "group": []},
            "lying_vs_truthfulness": {"x": [], "y": [], "group": []},
        },
    }
    missing_type = 0

    for html_path in html_files:
        npz_path = html_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        assistant = _parse_assistant_tokens(html_path)
        if not assistant.token_indices:
            continue

        with np.load(npz_path) as data:
            scores = data["scores_agg"]
            probe_indices = _find_probe_indices(data["probe_names"])
            prompt_len = int(data["prompt_len"][0]) if "prompt_len" in data else None
        if prompt_len is None:
            continue

        if args.split_type:
            item_type = type_by_npz.get(npz_path.resolve())
            if item_type not in ("adversarial", "non_adversarial"):
                missing_type += 1
                continue
        else:
            item_type = "all"

        if args.include_prompt_prefix:
            completion_indices = assistant.token_indices
        else:
            completion_indices = [idx for idx in assistant.token_indices if idx >= prompt_len]

        response_id = html_path.stem
        for pos, global_idx in enumerate(completion_indices):
            if global_idx >= scores.shape[1]:
                continue
            for probe_name, probe_idx in probe_indices.items():
                value = float(scores[probe_idx, global_idx])
                per_pos[probe_name][item_type].setdefault(pos, []).append(value)
                points[probe_name][item_type].append((pos, value))
                mixed_all[probe_name]["x"].append(float(pos))
                mixed_all[probe_name]["y"].append(value)
                mixed_all[probe_name]["group"].append(response_id)
                if args.split_type:
                    mixed_by_type[item_type][probe_name]["x"].append(float(pos))
                    mixed_by_type[item_type][probe_name]["y"].append(value)
                    mixed_by_type[item_type][probe_name]["group"].append(response_id)

    stats = {
        "settings": {
            "min_count": int(args.min_count),
            "split_type": bool(args.split_type),
            "include_prompt_prefix": bool(args.include_prompt_prefix),
            "missing_type": int(missing_type),
        },
        "probes": {},
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {
        "fabrication_vs_truthfulness": "#2c8d5b",
        "lying_vs_truthfulness": "#b24a3b",
    }
    linestyles = {
        "adversarial": "-",
        "non_adversarial": "--",
        "all": "-",
    }

    for probe_name, color in colors.items():
        stats["probes"][probe_name] = {}
        for item_type in base_types:
            probe_points = points.get(probe_name, {}).get(item_type, [])
            stats["probes"][probe_name][item_type] = _linear_stats(probe_points)
            stats["probes"][probe_name][item_type]["n_positions"] = len(
                per_pos.get(probe_name, {}).get(item_type, {})
            )

            xs = []
            means = []
            sems = []
            for pos in sorted(per_pos.get(probe_name, {}).get(item_type, {}).keys()):
                vals = per_pos[probe_name][item_type][pos]
                if len(vals) < args.min_count:
                    continue
                xs.append(pos)
                means.append(float(np.mean(vals)))
                sems.append(_sem(vals))

            if xs:
                label = probe_name if item_type == "all" else f"{probe_name} ({item_type})"
                ax.plot(
                    xs,
                    means,
                    color=color,
                    linewidth=2,
                    linestyle=linestyles[item_type],
                    label=label,
                )
                sems_arr = np.array(sems, dtype=np.float64)
                if np.isfinite(sems_arr).any():
                    ax.fill_between(
                        xs,
                        np.array(means) - sems_arr,
                        np.array(means) + sems_arr,
                        color=color,
                        alpha=0.15,
                    )

    ax.set_xlabel("Token position in completion")
    ax.set_ylabel("Mean probe score")
    ax.set_title("Probe score vs token position")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    for probe_name in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
        for item_type in base_types:
            s = stats["probes"].get(probe_name, {}).get(item_type, {})
            label = probe_name if item_type == "all" else f"{probe_name} ({item_type})"
            print(
                f"{label}: n={s.get('n')} slope={s.get('slope'):.4g} r={s.get('r'):.4g} p={s.get('p'):.4g}"
            )

    if args.clustered_stats:
        clustered = _compute_clustered_stats(
            html_files,
            type_by_npz,
            base_types,
            include_prompt_prefix=args.include_prompt_prefix,
        )
        clustered_path = args.out.with_suffix(".clustered.stats.json")
        clustered_path.write_text(json.dumps(clustered, indent=2), encoding="utf-8")
        _print_clustered_summary(clustered)

    if args.mixed_effects:
        mixed_stats: Dict[str, object] = {"all": {}, "by_type": {}}
        for probe_name in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
            mixed_stats["all"][probe_name] = _fit_mixedlm(
                mixed_all[probe_name]["x"],
                mixed_all[probe_name]["y"],
                mixed_all[probe_name]["group"],
            )
        if args.split_type:
            for item_type in ("adversarial", "non_adversarial"):
                mixed_stats["by_type"][item_type] = {}
                for probe_name in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
                    mixed_stats["by_type"][item_type][probe_name] = _fit_mixedlm(
                        mixed_by_type[item_type][probe_name]["x"],
                        mixed_by_type[item_type][probe_name]["y"],
                        mixed_by_type[item_type][probe_name]["group"],
                    )
        mixed_path = args.out.with_suffix(".mixed.stats.json")
        mixed_path.write_text(json.dumps(mixed_stats, indent=2), encoding="utf-8")
        _print_mixed_summary(mixed_stats)

    if args.cluster_robust:
        robust_stats: Dict[str, object] = {"all": {}, "by_type": {}}
        for probe_name in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
            robust_stats["all"][probe_name] = _fit_cluster_robust(
                mixed_all[probe_name]["x"],
                mixed_all[probe_name]["y"],
                mixed_all[probe_name]["group"],
            )
        if args.split_type:
            for item_type in ("adversarial", "non_adversarial"):
                robust_stats["by_type"][item_type] = {}
                for probe_name in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
                    robust_stats["by_type"][item_type][probe_name] = _fit_cluster_robust(
                        mixed_by_type[item_type][probe_name]["x"],
                        mixed_by_type[item_type][probe_name]["y"],
                        mixed_by_type[item_type][probe_name]["group"],
                    )
        robust_path = args.out.with_suffix(".cluster_robust.stats.json")
        robust_path.write_text(json.dumps(robust_stats, indent=2), encoding="utf-8")
        _print_cluster_robust_summary(robust_stats)

    if args.slope_diff:
        if not args.split_type:
            raise ValueError("--slope-diff requires --split-type.")
        slope_diff = _compute_slope_diff_cluster_robust(
            mixed_by_type, include_prompt_prefix=args.include_prompt_prefix
        )
        slope_diff_path = args.out.with_suffix(".slope_diff.cluster_robust.stats.json")
        slope_diff_path.write_text(json.dumps(slope_diff, indent=2), encoding="utf-8")
        _print_slope_diff_summary(slope_diff)


if __name__ == "__main__":
    main()
