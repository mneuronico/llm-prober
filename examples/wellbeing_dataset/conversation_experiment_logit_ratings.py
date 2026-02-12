import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concept_probe import ConceptProbe, ProbeWorkspace, multi_probe_score_prompts
from concept_probe.modeling import ModelBundle
from concept_probe.visuals import segment_token_scores

DEFAULT_TURNWISE_RELATIONSHIP_STATS = [
    "count",
    "slope",
    "r2",
    "isotonic_r2",
    "linreg_p",
    "pearson_r",
    "pearson_p",
    "spearman_rho",
    "spearman_p",
]

TURNWISE_STAT_YLABEL = {
    "count": "Sample count",
    "slope": "Slope (metric vs rating)",
    "intercept": "Intercept (metric vs rating)",
    "linreg_r": "Linear regression r",
    "linreg_p": "Linear regression p-value",
    "r2": "R^2",
    "isotonic_r2": "Isotonic R^2",
    "pearson_r": "Pearson r",
    "pearson_p": "Pearson p-value",
    "spearman_rho": "Spearman rho",
    "spearman_p": "Spearman p-value",
}

DEFAULT_CONVERSATION_DATASET_PATH = (
    "examples/wellbeing_dataset/data/wellbeing_conversations_openrouter_20260206_194607.json"
)


@dataclass
class PromptConfig:
    insert_message: str
    insert_after_role: str = "assistant"
    insert_after_mode: str = "each"  # "each" or "last"
    max_conversations: Optional[int] = None
    max_turns: Optional[int] = None


@dataclass
class SteeringConfig:
    alphas: List[float] = field(default_factory=lambda: [0.0])
    alpha_unit: str = "raw"
    steer_probe: Optional[Union[int, str]] = None
    steer_layers: Optional[Union[str, List[int]]] = None
    steer_window_radius: Optional[int] = None
    steer_distribute: Optional[bool] = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 16
    greedy: bool = True
    temperature: float = 0.0
    top_p: float = 0.9
    save_html: bool = False
    save_segments: bool = True


@dataclass
class RatingConfig:
    pattern: str = r"\b(10|[1-9])\b"
    min_value: int = 1
    max_value: int = 10
    label: str = "rating"


@dataclass
class SelfRatingsConfig:
    sources: Union[str, List[str]] = "token"
    primary_source: str = "token"


@dataclass
class LogitRatingConfig:
    enabled: bool = True
    option_values: List[int] = field(default_factory=lambda: list(range(10)))
    option_token_templates: List[str] = field(
        default_factory=lambda: ["{value}", " {value}", "\n{value}"]
    )
    step_index: int = 0
    save_option_probabilities: bool = True
    save_generation_logits: bool = True
    generation_logits_top_k: Optional[int] = None
    generation_logits_dtype: str = "float16"


@dataclass
class AnalysisConfig:
    metric_response_key: str = "completion_assistant_mean"
    metric_last_assistant_key: str = "prompt_assistant_last_mean"
    plot_vs_turn_metrics: List[str] = field(
        default_factory=lambda: ["completion_assistant_mean", "prompt_assistant_last_mean"]
    )
    plot_rating_vs_metrics: List[str] = field(
        default_factory=lambda: ["completion_assistant_mean", "prompt_assistant_last_mean"]
    )
    plot_rating_vs_alpha: bool = True
    plot_alignment_slope_vs_alpha: bool = False
    alignment_probe_name: Optional[str] = None
    alignment_metric_key: str = "prompt_assistant_last_mean"
    alignment_rating_key: str = "rating"
    alignment_slope_plot_name: str = "alignment_slope_vs_alpha.png"
    alignment_pvalue_plot_name: str = "alignment_slope_pvalue_vs_alpha.png"
    plot_r2_vs_alpha: bool = True
    r2_probe_name: Optional[str] = None
    r2_metric_key: str = "prompt_assistant_last_mean"
    r2_rating_key: str = "rating"
    r2_plot_name: str = "r2_vs_alpha.png"
    plot_report_variance_vs_alpha: bool = True
    report_variance_rating_key: str = "rating"
    report_variance_plot_name: str = "report_variance_vs_alpha.png"
    plot_turnwise_relationship_vs_alpha: bool = True
    turnwise_relationship_metric_keys: Optional[List[str]] = None
    turnwise_relationship_rating_key: str = "rating"
    turnwise_relationship_stats: List[str] = field(
        default_factory=lambda: list(DEFAULT_TURNWISE_RELATIONSHIP_STATS)
    )
    turnwise_relationship_json_name: str = "turnwise_relationship_vs_alpha.json"
    bootstrap_samples: int = 1000
    bootstrap_ci_level: float = 0.95
    bootstrap_seed: int = 0
    trend_min_points: int = 3
    alpha_reference_value: float = 0.0
    annotate_plots: bool = True


@dataclass
class ExperimentConfig:
    dataset_path: str
    probe_dirs: List[str]
    output_dir_template: str = "analysis/conversation_experiment_logit_ratings_{timestamp}"
    output_subdir: str = "scores"
    model_id: Optional[str] = None
    multi_probe: Optional[bool] = None
    prompt: PromptConfig = None
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    rating: RatingConfig = field(default_factory=RatingConfig)
    self_ratings: SelfRatingsConfig = field(default_factory=SelfRatingsConfig)
    logit_rating: LogitRatingConfig = field(default_factory=LogitRatingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_rating(text: str, cfg: RatingConfig) -> Optional[int]:
    if not text:
        return None
    match = re.search(cfg.pattern, text)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except Exception:
        return None
    if cfg.min_value <= value <= cfg.max_value:
        return value
    return None


def _normalize_rating_sources(value: Union[str, List[str]]) -> List[str]:
    if isinstance(value, str):
        raw = [value.strip().lower()]
    else:
        raw = [str(v).strip().lower() for v in value if str(v).strip()]
    if not raw:
        raw = ["token"]

    out: List[str] = []
    for item in raw:
        if item == "both":
            for source in ("token", "logit"):
                if source not in out:
                    out.append(source)
            continue
        if item not in {"token", "logit"}:
            raise ValueError("self_ratings.sources must be 'token', 'logit', 'both', or a list of these.")
        if item not in out:
            out.append(item)
    return out


def _rating_source_key(source: str) -> str:
    if source == "token":
        return "token_rating"
    if source == "logit":
        return "logit_rating"
    raise ValueError(f"Unknown rating source: {source}")


def _rating_source_dir(output_path: Path, source: str) -> Path:
    out_dir = output_path / "rating_sources" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _rating_source_alpha_dir(output_path: Path, alpha: float, source: str) -> Path:
    out_dir = _alpha_plot_dir(output_path, alpha) / "rating_sources" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_option_token_id_map(tokenizer, cfg: LogitRatingConfig) -> Dict[str, List[int]]:
    option_token_ids: Dict[str, List[int]] = {}
    used_ids: Dict[int, str] = {}
    for value in cfg.option_values:
        key = str(int(value))
        ids_for_option: Set[int] = set()
        for template in cfg.option_token_templates:
            text = str(template).format(value=key)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) != 1:
                continue
            ids_for_option.add(int(token_ids[0]))
        if not ids_for_option:
            raise ValueError(
                f"No single-token IDs found for option '{key}'. "
                f"Adjust logit_rating.option_values or option_token_templates."
            )
        for token_id in ids_for_option:
            seen = used_ids.get(token_id)
            if seen is not None and seen != key:
                raise ValueError(
                    f"Token id {token_id} is shared by options '{seen}' and '{key}'. "
                    "Use non-overlapping option/token templates."
                )
            used_ids[token_id] = key
        option_token_ids[key] = sorted(ids_for_option)
    return option_token_ids


def _load_step_logits(npz: Any, step_index: int) -> Optional[Tuple[np.ndarray, Optional[Dict[int, float]]]]:
    if "generation_logits" in npz:
        logits = np.asarray(npz["generation_logits"], dtype=np.float64)
        if logits.ndim != 2:
            return None
        if step_index < 0 or step_index >= logits.shape[0]:
            return None
        return (logits[step_index], None)

    if "generation_logits_topk_indices" in npz and "generation_logits_topk_values" in npz:
        indices = np.asarray(npz["generation_logits_topk_indices"], dtype=np.int64)
        values = np.asarray(npz["generation_logits_topk_values"], dtype=np.float64)
        if indices.ndim != 2 or values.ndim != 2:
            return None
        if step_index < 0 or step_index >= indices.shape[0]:
            return None
        row_map: Dict[int, float] = {}
        for idx, val in zip(indices[step_index].tolist(), values[step_index].tolist()):
            row_map[int(idx)] = float(val)
        return (np.array([], dtype=np.float64), row_map)

    return None


def _logsumexp(values: np.ndarray) -> float:
    if values.size == 0:
        return float("-inf")
    m = float(np.max(values))
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(values - m))))


def _compute_logit_rating_from_npz(
    npz_path: str,
    *,
    option_token_ids: Dict[str, List[int]],
    option_values: Dict[str, float],
    step_index: int,
    save_option_probabilities: bool,
) -> Dict[str, Any]:
    npz = np.load(npz_path)
    step_data = _load_step_logits(npz, step_index)
    if step_data is None:
        return {"logit_rating": None, "logit_rating_probs": None, "logit_rating_status": "missing_logits"}

    dense_logits, sparse_logits = step_data
    option_scores: Dict[str, float] = {}
    for option_key, token_ids in option_token_ids.items():
        candidate_vals: List[float] = []
        if sparse_logits is None:
            for token_id in token_ids:
                if 0 <= token_id < dense_logits.shape[0]:
                    candidate_vals.append(float(dense_logits[token_id]))
        else:
            for token_id in token_ids:
                if token_id in sparse_logits:
                    candidate_vals.append(float(sparse_logits[token_id]))
        if not candidate_vals:
            return {
                "logit_rating": None,
                "logit_rating_probs": None,
                "logit_rating_status": f"missing_option_logits:{option_key}",
            }
        option_scores[option_key] = _logsumexp(np.array(candidate_vals, dtype=np.float64))

    score_vals = np.array([option_scores[k] for k in option_values.keys()], dtype=np.float64)
    finite_mask = np.isfinite(score_vals)
    if not np.any(finite_mask):
        return {
            "logit_rating": None,
            "logit_rating_probs": None,
            "logit_rating_status": "all_option_logits_nonfinite",
        }
    norm = _logsumexp(score_vals)
    if not np.isfinite(norm):
        return {
            "logit_rating": None,
            "logit_rating_probs": None,
            "logit_rating_status": "normalization_nonfinite",
        }
    probs = np.exp(score_vals - norm)
    if not np.all(np.isfinite(probs)):
        return {
            "logit_rating": None,
            "logit_rating_probs": None,
            "logit_rating_status": "probabilities_nonfinite",
        }
    rating = float(sum(option_values[k] * float(p) for k, p in zip(option_values.keys(), probs)))
    if not np.isfinite(rating):
        return {
            "logit_rating": None,
            "logit_rating_probs": None,
            "logit_rating_status": "rating_nonfinite",
        }
    prob_map = {k: float(p) for k, p in zip(option_values.keys(), probs)} if save_option_probabilities else None
    return {
        "logit_rating": rating,
        "logit_rating_probs": prob_map,
        "logit_rating_status": "ok",
    }


def _mean_sem(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if len(values) < 2:
        return (mean, 0.0)
    sem = float(arr.std(ddof=1) / math.sqrt(len(values)))
    return (mean, sem)


def _ci_bounds(values: List[float], ci_level: float) -> Tuple[Optional[float], Optional[float]]:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    if not finite:
        return (None, None)
    lo_q = 100.0 * (1.0 - float(ci_level)) / 2.0
    hi_q = 100.0 - lo_q
    arr = np.array(finite, dtype=float)
    lo, hi = np.percentile(arr, [lo_q, hi_q])
    return (float(lo), float(hi))


def _isotonic_fit_1d(y: np.ndarray, *, increasing: bool) -> np.ndarray:
    vals = np.array(y, dtype=np.float64)
    n = int(vals.size)
    if n == 0:
        return vals
    if n == 1:
        return vals.copy()

    levels: List[float] = [float(v) for v in vals]
    weights: List[float] = [1.0] * n
    starts: List[int] = list(range(n))
    ends: List[int] = list(range(n))

    i = 0
    while i < len(levels) - 1:
        violation = (levels[i] > levels[i + 1]) if increasing else (levels[i] < levels[i + 1])
        if not violation:
            i += 1
            continue
        w = weights[i] + weights[i + 1]
        level = (weights[i] * levels[i] + weights[i + 1] * levels[i + 1]) / w
        levels[i] = float(level)
        weights[i] = float(w)
        ends[i] = ends[i + 1]

        del levels[i + 1]
        del weights[i + 1]
        del starts[i + 1]
        del ends[i + 1]

        while i > 0:
            prev_violation = (levels[i - 1] > levels[i]) if increasing else (levels[i - 1] < levels[i])
            if not prev_violation:
                break
            w2 = weights[i - 1] + weights[i]
            level2 = (weights[i - 1] * levels[i - 1] + weights[i] * levels[i]) / w2
            levels[i - 1] = float(level2)
            weights[i - 1] = float(w2)
            ends[i - 1] = ends[i]
            del levels[i]
            del weights[i]
            del starts[i]
            del ends[i]
            i -= 1

    fitted = np.empty(n, dtype=np.float64)
    for level, s, e in zip(levels, starts, ends):
        fitted[int(s) : int(e) + 1] = float(level)
    return fitted


def _isotonic_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "isotonic_r2": None,
        "isotonic_rmse": None,
        "isotonic_mae": None,
    }
    if x.size < 2 or y.size < 2:
        return out
    if np.allclose(y, y[0]):
        return out

    order = np.argsort(np.array(x, dtype=np.float64), kind="mergesort")
    y_sorted = np.array(y[order], dtype=np.float64)

    fit_inc = _isotonic_fit_1d(y_sorted, increasing=True)
    fit_dec = _isotonic_fit_1d(y_sorted, increasing=False)
    sse_inc = float(np.sum((y_sorted - fit_inc) ** 2))
    sse_dec = float(np.sum((y_sorted - fit_dec) ** 2))
    y_fit_sorted = fit_inc if sse_inc <= sse_dec else fit_dec

    y_fit = np.empty_like(y_fit_sorted)
    y_fit[order] = y_fit_sorted

    residuals = np.array(y, dtype=np.float64) - np.array(y_fit, dtype=np.float64)
    sse = float(np.sum(residuals ** 2))
    tss = float(np.sum((np.array(y, dtype=np.float64) - float(np.mean(y))) ** 2))
    if tss > 0:
        out["isotonic_r2"] = _safe_float(1.0 - (sse / tss))
    out["isotonic_rmse"] = _safe_float(float(np.sqrt(np.mean(residuals ** 2))))
    out["isotonic_mae"] = _safe_float(float(np.mean(np.abs(residuals))))
    return out


def _pair_stats_arrays(x: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    stats: Dict[str, Optional[float]] = {
        "slope": None,
        "intercept": None,
        "linreg_r": None,
        "linreg_p": None,
        "r2": None,
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
        "y_variance": None,
        "isotonic_r2": None,
        "isotonic_rmse": None,
        "isotonic_mae": None,
    }
    if x.size < 2 or y.size < 2:
        return stats

    if y.size >= 2:
        try:
            stats["y_variance"] = _safe_float(float(np.var(y, ddof=1)))
        except Exception:
            pass

    x_const = bool(np.allclose(x, x[0])) if x.size else True
    y_const = bool(np.allclose(y, y[0])) if y.size else True

    if not x_const:
        if linregress is not None:
            try:
                res = linregress(x, y)
                stats["slope"] = _safe_float(res.slope)
                stats["intercept"] = _safe_float(res.intercept)
                stats["linreg_r"] = _safe_float(res.rvalue)
                stats["linreg_p"] = _safe_float(res.pvalue)
                if stats["linreg_r"] is not None:
                    stats["r2"] = float(stats["linreg_r"] ** 2)
            except Exception:
                pass
        else:
            try:
                slope, intercept = np.polyfit(x, y, 1)
                stats["slope"] = _safe_float(slope)
                stats["intercept"] = _safe_float(intercept)
                corr = _safe_float(np.corrcoef(x, y)[0, 1])
                stats["linreg_r"] = corr
                if corr is not None:
                    stats["r2"] = float(corr ** 2)
            except Exception:
                pass

    if not x_const and not y_const:
        if pearsonr is not None:
            try:
                pr = pearsonr(x, y)
                stats["pearson_r"] = _safe_float(pr.statistic)
                stats["pearson_p"] = _safe_float(pr.pvalue)
            except Exception:
                pass
        else:
            stats["pearson_r"] = _safe_float(np.corrcoef(x, y)[0, 1])

        if spearmanr is not None:
            try:
                sr = spearmanr(x, y)
                stats["spearman_rho"] = _safe_float(sr.correlation)
                stats["spearman_p"] = _safe_float(sr.pvalue)
            except Exception:
                pass

    try:
        iso_stats = _isotonic_stats(x, y)
        for key, value in iso_stats.items():
            stats[key] = _safe_float(value)
    except Exception:
        pass

    return stats


def _bootstrap_pair_stats(
    xs: List[float],
    ys: List[float],
    *,
    bootstrap_samples: int,
    ci_level: float,
    rng: np.random.Generator,
) -> Dict[str, Optional[float]]:
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    out: Dict[str, Optional[float]] = {"count": float(x.size)}
    point = _pair_stats_arrays(x, y)
    out.update(point)

    if x.size < 2 or bootstrap_samples <= 0:
        out["bootstrap_samples"] = 0
        return out

    metric_keys = list(point.keys())
    samples: Dict[str, List[float]] = {k: [] for k in metric_keys}
    n = int(x.size)
    for _ in range(int(bootstrap_samples)):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        boot_stats = _pair_stats_arrays(xb, yb)
        for key in metric_keys:
            val = boot_stats.get(key)
            if isinstance(val, (int, float)) and np.isfinite(float(val)):
                samples[key].append(float(val))

    for key in metric_keys:
        lo, hi = _ci_bounds(samples.get(key, []), ci_level)
        out[f"{key}_ci_low"] = lo
        out[f"{key}_ci_high"] = hi
        out[f"{key}_bootstrap_n"] = int(len(samples.get(key, [])))
    out["bootstrap_samples"] = int(bootstrap_samples)
    return out


def _bootstrap_variance(
    values: List[float],
    *,
    bootstrap_samples: int,
    ci_level: float,
    rng: np.random.Generator,
) -> Dict[str, Optional[float]]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    out: Dict[str, Optional[float]] = {
        "variance": None,
        "variance_ci_low": None,
        "variance_ci_high": None,
        "bootstrap_samples": 0,
        "bootstrap_n": 0,
    }
    if arr.size < 2:
        return out

    out["variance"] = float(np.var(arr, ddof=1))
    if bootstrap_samples <= 0:
        return out

    n = int(arr.size)
    vals: List[float] = []
    for _ in range(int(bootstrap_samples)):
        idx = rng.integers(0, n, size=n)
        vb = arr[idx]
        if vb.size < 2:
            continue
        v = float(np.var(vb, ddof=1))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = _ci_bounds(vals, ci_level)
    out["variance_ci_low"] = lo
    out["variance_ci_high"] = hi
    out["bootstrap_samples"] = int(bootstrap_samples)
    out["bootstrap_n"] = int(len(vals))
    return out


def _value_metric_point(values: np.ndarray, metric_key: str) -> Optional[float]:
    vals = values[np.isfinite(values)]
    if metric_key == "mean":
        if vals.size < 1:
            return None
        return _safe_float(float(np.mean(vals)))
    if metric_key == "variance":
        if vals.size < 2:
            return None
        return _safe_float(float(np.var(vals, ddof=1)))
    raise ValueError(f"Unsupported value metric '{metric_key}'")


def _bootstrap_value_metric_difference(
    reference_values: List[float],
    compare_values: List[float],
    *,
    metric_key: str,
    bootstrap_samples: int,
    ci_level: float,
    rng: np.random.Generator,
) -> Dict[str, Optional[float]]:
    ref = np.array(reference_values, dtype=float)
    cmp_ = np.array(compare_values, dtype=float)
    ref = ref[np.isfinite(ref)]
    cmp_ = cmp_[np.isfinite(cmp_)]
    out: Dict[str, Optional[float]] = {
        "metric": metric_key,
        "reference_n": float(ref.size),
        "compare_n": float(cmp_.size),
        "reference_value": _value_metric_point(ref, metric_key),
        "compare_value": _value_metric_point(cmp_, metric_key),
        "delta": None,
        "delta_ci_low": None,
        "delta_ci_high": None,
        "bootstrap_p_two_sided": None,
        "bootstrap_n": 0,
        "bootstrap_samples": int(max(0, bootstrap_samples)),
        "direction": "insufficient_data",
    }
    if out["reference_value"] is not None and out["compare_value"] is not None:
        out["delta"] = _safe_float(float(out["compare_value"]) - float(out["reference_value"]))

    if ref.size < 2 or cmp_.size < 2 or bootstrap_samples <= 0:
        return out

    n_ref = int(ref.size)
    n_cmp = int(cmp_.size)
    deltas: List[float] = []
    for _ in range(int(bootstrap_samples)):
        ref_idx = rng.integers(0, n_ref, size=n_ref)
        cmp_idx = rng.integers(0, n_cmp, size=n_cmp)
        ref_metric = _value_metric_point(ref[ref_idx], metric_key)
        cmp_metric = _value_metric_point(cmp_[cmp_idx], metric_key)
        if ref_metric is None or cmp_metric is None:
            continue
        delta = _safe_float(float(cmp_metric) - float(ref_metric))
        if delta is not None:
            deltas.append(delta)

    if not deltas:
        return out

    lo, hi = _ci_bounds(deltas, ci_level)
    out["delta_ci_low"] = lo
    out["delta_ci_high"] = hi
    out["bootstrap_n"] = int(len(deltas))
    gt = float(np.mean(np.array(deltas, dtype=float) >= 0.0))
    lt = float(np.mean(np.array(deltas, dtype=float) <= 0.0))
    eps = 1.0 / float(len(deltas) + 1)
    out["bootstrap_p_two_sided"] = float(max(eps, min(1.0, 2.0 * min(gt, lt))))
    if lo is not None and hi is not None:
        if lo > 0:
            out["direction"] = "increase_vs_reference"
        elif hi < 0:
            out["direction"] = "decrease_vs_reference"
        else:
            out["direction"] = "no_clear_change_vs_reference"
    return out


def _bootstrap_pair_metric_difference(
    ref_xs: List[float],
    ref_ys: List[float],
    cmp_xs: List[float],
    cmp_ys: List[float],
    *,
    metric_key: str,
    bootstrap_samples: int,
    ci_level: float,
    rng: np.random.Generator,
) -> Dict[str, Optional[float]]:
    ref_x = np.array(ref_xs, dtype=float)
    ref_y = np.array(ref_ys, dtype=float)
    cmp_x = np.array(cmp_xs, dtype=float)
    cmp_y = np.array(cmp_ys, dtype=float)

    ref_mask = np.isfinite(ref_x) & np.isfinite(ref_y)
    cmp_mask = np.isfinite(cmp_x) & np.isfinite(cmp_y)
    ref_x = ref_x[ref_mask]
    ref_y = ref_y[ref_mask]
    cmp_x = cmp_x[cmp_mask]
    cmp_y = cmp_y[cmp_mask]

    ref_point = _pair_stats_arrays(ref_x, ref_y)
    cmp_point = _pair_stats_arrays(cmp_x, cmp_y)
    ref_val = _safe_float(ref_point.get(metric_key))
    cmp_val = _safe_float(cmp_point.get(metric_key))
    out: Dict[str, Optional[float]] = {
        "metric": metric_key,
        "reference_n": float(ref_x.size),
        "compare_n": float(cmp_x.size),
        "reference_value": ref_val,
        "compare_value": cmp_val,
        "delta": None,
        "delta_ci_low": None,
        "delta_ci_high": None,
        "bootstrap_p_two_sided": None,
        "bootstrap_n": 0,
        "bootstrap_samples": int(max(0, bootstrap_samples)),
        "direction": "insufficient_data",
    }
    if ref_val is not None and cmp_val is not None:
        out["delta"] = _safe_float(float(cmp_val) - float(ref_val))

    if ref_x.size < 2 or cmp_x.size < 2 or bootstrap_samples <= 0:
        return out

    n_ref = int(ref_x.size)
    n_cmp = int(cmp_x.size)
    deltas: List[float] = []
    for _ in range(int(bootstrap_samples)):
        ref_idx = rng.integers(0, n_ref, size=n_ref)
        cmp_idx = rng.integers(0, n_cmp, size=n_cmp)
        ref_metric = _safe_float(_pair_stats_arrays(ref_x[ref_idx], ref_y[ref_idx]).get(metric_key))
        cmp_metric = _safe_float(_pair_stats_arrays(cmp_x[cmp_idx], cmp_y[cmp_idx]).get(metric_key))
        if ref_metric is None or cmp_metric is None:
            continue
        delta = _safe_float(float(cmp_metric) - float(ref_metric))
        if delta is not None:
            deltas.append(delta)

    if not deltas:
        return out

    lo, hi = _ci_bounds(deltas, ci_level)
    out["delta_ci_low"] = lo
    out["delta_ci_high"] = hi
    out["bootstrap_n"] = int(len(deltas))
    gt = float(np.mean(np.array(deltas, dtype=float) >= 0.0))
    lt = float(np.mean(np.array(deltas, dtype=float) <= 0.0))
    eps = 1.0 / float(len(deltas) + 1)
    out["bootstrap_p_two_sided"] = float(max(eps, min(1.0, 2.0 * min(gt, lt))))
    if lo is not None and hi is not None:
        if lo > 0:
            out["direction"] = "increase_vs_reference"
        elif hi < 0:
            out["direction"] = "decrease_vs_reference"
        else:
            out["direction"] = "no_clear_change_vs_reference"
    return out


def _bootstrap_series_trend(
    xs: List[float],
    ys: List[Optional[float]],
    *,
    bootstrap_samples: int,
    ci_level: float,
    min_points: int,
    rng: np.random.Generator,
) -> Dict[str, Optional[float]]:
    out = _bootstrap_pair_stats(
        xs,
        [float(y) if isinstance(y, (int, float)) else float("nan") for y in ys],
        bootstrap_samples=bootstrap_samples,
        ci_level=ci_level,
        rng=rng,
    )
    n = int(out.get("count", 0.0) or 0)
    out["trend_min_points"] = int(min_points)
    out["trend_eligible"] = bool(n >= int(min_points))
    if not bool(out["trend_eligible"]):
        out["trend_direction"] = "insufficient_points"
        return out

    lo = _safe_float(out.get("slope_ci_low"))
    hi = _safe_float(out.get("slope_ci_high"))
    slope = _safe_float(out.get("slope"))
    if lo is not None and hi is not None:
        if lo > 0:
            out["trend_direction"] = "increasing"
        elif hi < 0:
            out["trend_direction"] = "decreasing"
        else:
            out["trend_direction"] = "no_clear_trend"
    elif slope is not None:
        out["trend_direction"] = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat")
    else:
        out["trend_direction"] = "unknown"
    return out


def _fmt_stat(v: Optional[float], *, p: bool = False) -> str:
    if v is None:
        return "n/a"
    x = float(v)
    if not np.isfinite(x):
        return "n/a"
    if p:
        if x == 0.0:
            return "0"
        if x < 1e-3:
            return f"{x:.1e}"
        return f"{x:.4f}"
    if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.2e}"
    return f"{x:.4f}"


def _fmt_ci(lo: Optional[float], hi: Optional[float], *, p: bool = False) -> str:
    return f"[{_fmt_stat(lo, p=p)}, {_fmt_stat(hi, p=p)}]"


def _annotation_lines_for_trend(
    trend: Optional[Dict[str, Optional[float]]],
    *,
    label: str,
) -> List[str]:
    if not isinstance(trend, dict):
        return []
    lines = [
        f"{label} trend: {trend.get('trend_direction', 'unknown')}",
        f"slope={_fmt_stat(_safe_float(trend.get('slope')))} CI{_fmt_ci(_safe_float(trend.get('slope_ci_low')), _safe_float(trend.get('slope_ci_high')))}",
        f"R2={_fmt_stat(_safe_float(trend.get('r2')))} p={_fmt_stat(_safe_float(trend.get('linreg_p')), p=True)}",
    ]
    return lines


def _annotation_lines_for_vs_reference(
    effect_by_alpha: Dict[str, Dict[str, Optional[float]]],
    *,
    reference_alpha: float,
    max_lines: int = 4,
) -> List[str]:
    if not effect_by_alpha:
        return []
    lines = [f"vs alpha={reference_alpha:g}"]
    shown = 0
    for alpha_key in sorted(effect_by_alpha.keys(), key=lambda s: float(s)):
        effect = effect_by_alpha[alpha_key]
        if not isinstance(effect, dict):
            continue
        delta = _safe_float(effect.get("delta"))
        lo = _safe_float(effect.get("delta_ci_low"))
        hi = _safe_float(effect.get("delta_ci_high"))
        pval = _safe_float(effect.get("bootstrap_p_two_sided"))
        lines.append(f"a={float(alpha_key):g}: d={_fmt_stat(delta)} CI{_fmt_ci(lo, hi)} p~{_fmt_stat(pval, p=True)}")
        shown += 1
        if shown >= max_lines:
            break
    return lines


def _nearest_alpha_key(values: Iterable[float], target: float, tol: float = 1e-9) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    best = min(vals, key=lambda v: abs(v - float(target)))
    if abs(best - float(target)) <= tol:
        return best
    return None


def _linear_stats(xs: List[float], ys: List[float]) -> Dict[str, Optional[float]]:
    if len(xs) < 2 or len(ys) < 2:
        return {"slope": None, "intercept": None, "r": None, "p": None}
    if linregress is None:
        slope, intercept = np.polyfit(np.array(xs, dtype=float), np.array(ys, dtype=float), 1)
        r = float(np.corrcoef(xs, ys)[0, 1])
        return {"slope": float(slope), "intercept": float(intercept), "r": r, "p": None}
    result = linregress(xs, ys)
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r": float(result.rvalue),
        "p": float(result.pvalue),
    }


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
    result = linregress(x, y)
    return (float(result.slope), float(result.pvalue))


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


def _correlation_stats(xs: List[float], ys: List[float]) -> Dict[str, Optional[float]]:
    if len(xs) < 2 or len(ys) < 2:
        return {
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }
    if pearsonr is None or spearmanr is None:
        r = float(np.corrcoef(xs, ys)[0, 1])
        return {
            "pearson_r": r,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }
    pr = pearsonr(xs, ys)
    sr = spearmanr(xs, ys)
    return {
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.correlation),
        "spearman_p": float(sr.pvalue),
    }


def _plot_series(
    xs: List[int],
    means: List[float],
    sems: List[float],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    annotation_lines: Optional[List[str]] = None,
) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    plt.figure(figsize=(7.2, 4.2))
    plt.errorbar(xs, means, yerr=sems, marker="o", linewidth=1.6, capsize=3)
    plt.xlabel("Turn")
    plt.ylabel(ylabel)
    plt.title(title)
    if annotation_lines:
        plt.gca().text(
            0.02,
            0.98,
            "\n".join(annotation_lines),
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.8},
        )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_scatter(xs: List[float], ys: List[float], *, title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    plt.figure(figsize=(6.4, 4.6))
    plt.scatter(xs, ys, alpha=0.7, s=22)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_alpha_series(
    xs: List[float],
    means: List[float],
    sems: List[float],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    annotation_lines: Optional[List[str]] = None,
) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    plt.figure(figsize=(7.2, 4.2))
    plt.errorbar(xs, means, yerr=sems, marker="o", linewidth=1.6, capsize=3)
    plt.xlabel("Alpha")
    plt.ylabel(ylabel)
    plt.title(title)
    if annotation_lines:
        plt.gca().text(
            0.02,
            0.98,
            "\n".join(annotation_lines),
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.8},
        )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_alpha_line(
    xs: List[float],
    ys_in: List[Optional[float]],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    ci_low_in: Optional[List[Optional[float]]] = None,
    ci_high_in: Optional[List[Optional[float]]] = None,
    log_y: bool = False,
    annotation_lines: Optional[List[str]] = None,
) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    xs_plot: List[float] = []
    ys_plot: List[float] = []
    ci_low_plot: List[float] = []
    ci_high_plot: List[float] = []
    for idx, (x, y) in enumerate(zip(xs, ys_in)):
        if y is None or np.isnan(y):
            continue
        y_f = float(y)
        if log_y and y_f <= 0:
            continue
        xs_plot.append(float(x))
        ys_plot.append(y_f)
        if ci_low_in is not None and ci_high_in is not None:
            low = ci_low_in[idx] if idx < len(ci_low_in) else None
            high = ci_high_in[idx] if idx < len(ci_high_in) else None
            low_f = float(low) if isinstance(low, (int, float)) else float("nan")
            high_f = float(high) if isinstance(high, (int, float)) else float("nan")
            if log_y and (not np.isfinite(low_f) or low_f <= 0):
                low_f = float("nan")
            if log_y and (not np.isfinite(high_f) or high_f <= 0):
                high_f = float("nan")
            ci_low_plot.append(low_f)
            ci_high_plot.append(high_f)
    if not xs_plot:
        return
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(xs_plot, ys_plot, marker="o", linewidth=1.6)
    if ci_low_in is not None and ci_high_in is not None and ci_low_plot and ci_high_plot:
        low_arr = np.array(ci_low_plot, dtype=float)
        high_arr = np.array(ci_high_plot, dtype=float)
        ok = np.isfinite(low_arr) & np.isfinite(high_arr)
        if np.any(ok):
            plt.fill_between(
                np.array(xs_plot, dtype=float),
                low_arr,
                high_arr,
                where=ok,
                alpha=0.20,
                linewidth=0.0,
            )
            y_arr = np.array(ys_plot, dtype=float)
            yerr_low = np.where(ok, np.maximum(0.0, y_arr - low_arr), np.nan)
            yerr_high = np.where(ok, np.maximum(0.0, high_arr - y_arr), np.nan)
            plt.errorbar(
                np.array(xs_plot, dtype=float),
                y_arr,
                yerr=np.vstack([yerr_low, yerr_high]),
                fmt="none",
                capsize=3,
                linewidth=1.0,
                alpha=0.9,
            )
    if log_y:
        plt.yscale("log")
    plt.xlabel("Alpha")
    plt.ylabel(ylabel)
    plt.title(title)
    if annotation_lines:
        plt.gca().text(
            0.02,
            0.98,
            "\n".join(annotation_lines),
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.8},
        )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _alpha_label(alpha: float) -> str:
    return f"{alpha:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")


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


def _suffix_filename(filename: str, suffix: str) -> str:
    path = Path(filename)
    stem = path.stem
    ext = path.suffix or ".png"
    return f"{stem}_{suffix}{ext}"


def _alpha_plot_dir(output_path: Path, alpha: float) -> Path:
    out_dir = output_path / "by_alpha" / f"alpha_{_alpha_label(alpha)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _normalize_turnwise_stats(stats_raw: Union[str, List[str], Tuple[str, ...]]) -> List[str]:
    if isinstance(stats_raw, str):
        raw = [s.strip() for s in stats_raw.split(",") if s.strip()]
    elif isinstance(stats_raw, (list, tuple)):
        raw = [str(s).strip() for s in stats_raw if str(s).strip()]
    else:
        raw = list(DEFAULT_TURNWISE_RELATIONSHIP_STATS)

    if not raw:
        raw = list(DEFAULT_TURNWISE_RELATIONSHIP_STATS)

    if len(raw) == 1 and raw[0].lower() == "all":
        raw = list(DEFAULT_TURNWISE_RELATIONSHIP_STATS)

    seen: Dict[str, bool] = {}
    parsed: List[str] = []
    for item in raw:
        key = item.lower()
        if key == "p":
            key = "linreg_p"
        elif key == "r":
            key = "linreg_r"
        if key not in TURNWISE_STAT_YLABEL:
            raise ValueError(
                f"Unknown turnwise stat '{item}'. Allowed: {sorted(TURNWISE_STAT_YLABEL.keys())} or 'all'."
            )
        if key in seen:
            continue
        seen[key] = True
        parsed.append(key)
    return parsed


def _turnwise_pair_stats(
    xs: List[float],
    ys: List[float],
    *,
    bootstrap_samples: int,
    bootstrap_ci_level: float,
    bootstrap_rng: np.random.Generator,
) -> Dict[str, Optional[float]]:
    return _bootstrap_pair_stats(
        xs,
        ys,
        bootstrap_samples=bootstrap_samples,
        ci_level=bootstrap_ci_level,
        rng=bootstrap_rng,
    )


def _build_turnwise_relationship_by_alpha(
    records_by_alpha: Dict[float, List[Dict[str, object]]],
    *,
    probe_name: str,
    metric_key: str,
    rating_key: str,
    bootstrap_samples: int,
    bootstrap_ci_level: float,
    bootstrap_rng: np.random.Generator,
) -> Dict[str, object]:
    alphas = sorted(records_by_alpha.keys())
    turns: List[int] = sorted(
        {int(row.get("turn_index")) for rows in records_by_alpha.values() for row in rows if isinstance(row.get("turn_index"), int)}
    )

    stats_by_alpha_turn: Dict[str, List[Dict[str, Optional[float]]]] = {}
    for alpha in alphas:
        rows_out: List[Dict[str, Optional[float]]] = []
        rows = records_by_alpha.get(alpha, [])
        for turn in turns:
            xs: List[float] = []
            ys: List[float] = []
            for row in rows:
                turn_value = row.get("turn_index")
                if not isinstance(turn_value, int) or turn_value != turn:
                    continue
                rating_val = row.get(rating_key)
                if not isinstance(rating_val, (int, float)):
                    continue
                probe_metrics = row.get("probe_metrics", {}).get(probe_name, {})
                if not isinstance(probe_metrics, dict):
                    continue
                metric_val = probe_metrics.get(metric_key)
                if not isinstance(metric_val, (int, float)):
                    continue
                xs.append(float(rating_val))
                ys.append(float(metric_val))

            stats = _turnwise_pair_stats(
                xs,
                ys,
                bootstrap_samples=bootstrap_samples,
                bootstrap_ci_level=bootstrap_ci_level,
                bootstrap_rng=bootstrap_rng,
            )
            row_out: Dict[str, Optional[float]] = {"turn": int(turn)}
            if stats.get("count") is not None:
                stats["count"] = int(stats["count"])
            row_out.update(stats)
            rows_out.append(row_out)
        stats_by_alpha_turn[str(float(alpha))] = rows_out

    return {
        "probe_name": probe_name,
        "metric_key": metric_key,
        "rating_key": rating_key,
        "alphas": [float(a) for a in alphas],
        "turns": turns,
        "stats_by_alpha_turn": stats_by_alpha_turn,
    }


def _plot_turnwise_stat_for_alpha(
    *,
    turns: List[int],
    rows_for_alpha: List[Dict[str, Optional[float]]],
    alpha: float,
    stat_key: str,
    title: str,
    out_path: Path,
    annotation_lines: Optional[List[str]] = None,
) -> bool:
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    plt.figure(figsize=(8.0, 4.8))
    values_by_turn: Dict[int, Optional[float]] = {}
    low_by_turn: Dict[int, Optional[float]] = {}
    high_by_turn: Dict[int, Optional[float]] = {}
    for row in rows_for_alpha:
        turn_obj = row.get("turn")
        if not isinstance(turn_obj, int):
            continue
        turn = int(turn_obj)
        values_by_turn[turn] = _safe_float(row.get(stat_key))
        low_by_turn[turn] = _safe_float(row.get(f"{stat_key}_ci_low"))
        high_by_turn[turn] = _safe_float(row.get(f"{stat_key}_ci_high"))

    log_y = stat_key.endswith("_p")
    xs_plot: List[int] = []
    ys_plot: List[float] = []
    ci_low_plot: List[float] = []
    ci_high_plot: List[float] = []
    for turn in turns:
        val = values_by_turn.get(turn)
        if val is None:
            continue
        if log_y and float(val) <= 0:
            continue
        xs_plot.append(int(turn))
        ys_plot.append(float(val))
        low = low_by_turn.get(turn)
        high = high_by_turn.get(turn)
        low_f = float(low) if isinstance(low, (int, float)) else float("nan")
        high_f = float(high) if isinstance(high, (int, float)) else float("nan")
        if log_y and (not np.isfinite(low_f) or low_f <= 0):
            low_f = float("nan")
        if log_y and (not np.isfinite(high_f) or high_f <= 0):
            high_f = float("nan")
        ci_low_plot.append(low_f)
        ci_high_plot.append(high_f)

    if not xs_plot:
        plt.close()
        return False

    plt.plot(xs_plot, ys_plot, marker="o", linewidth=1.5, label=f"alpha={float(alpha):g}")
    low_arr = np.array(ci_low_plot, dtype=float)
    high_arr = np.array(ci_high_plot, dtype=float)
    ok = np.isfinite(low_arr) & np.isfinite(high_arr)
    if np.any(ok):
        plt.fill_between(
            np.array(xs_plot, dtype=float),
            low_arr,
            high_arr,
            where=ok,
            alpha=0.20,
            linewidth=0.0,
        )
    if log_y:
        plt.yscale("log")
    plt.xlabel("Turn")
    plt.ylabel(TURNWISE_STAT_YLABEL.get(stat_key, stat_key))
    plt.title(title)
    if annotation_lines:
        plt.gca().text(
            0.02,
            0.98,
            "\n".join(annotation_lines),
            transform=plt.gca().transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.8},
        )
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _collect_prompts(dataset: Dict[str, object], cfg: PromptConfig) -> Tuple[List[List[Dict[str, str]]], List[Dict[str, object]]]:
    conversations = dataset.get("conversations", [])
    if not isinstance(conversations, list):
        raise ValueError("Dataset missing 'conversations' list.")
    prompts: List[List[Dict[str, str]]] = []
    metas: List[Dict[str, object]] = []
    limit_conversations = len(conversations) if cfg.max_conversations is None else cfg.max_conversations

    for conv_idx, conv in enumerate(conversations[:limit_conversations], start=1):
        messages = conv.get("messages", [])
        if not isinstance(messages, list):
            continue
        assistant_turn = 0
        eligible_indices: List[int] = []
        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != cfg.insert_after_role:
                continue
            assistant_turn += 1
            if cfg.max_turns is not None and assistant_turn > cfg.max_turns:
                break
            eligible_indices.append(msg_idx)

        if cfg.insert_after_mode == "last":
            eligible_indices = eligible_indices[-1:] if eligible_indices else []

        assistant_turn = 0
        for msg_idx, msg in enumerate(messages):
            if msg.get("role") == cfg.insert_after_role:
                assistant_turn += 1
            if msg_idx not in eligible_indices:
                continue

            prompt_messages = list(messages[: msg_idx + 1])
            prompt_messages.append({"role": "user", "content": cfg.insert_message})
            prompts.append(prompt_messages)
            metas.append(
                {
                    "conversation_index": conv_idx,
                    "conversation_id": str(conv.get("topic_id", f"conversation_{conv_idx}")),
                    "conversation_title": str(conv.get("topic_title", "")),
                    "turn_index": assistant_turn,
                    "assistant_message_index": msg_idx,
                }
            )
    return prompts, metas


def _load_single_probe(probe_dir: str, model_id: Optional[str]) -> ConceptProbe:
    workspace = ProbeWorkspace(project_directory=probe_dir, config_overrides={"model": {"model_id": model_id}} if model_id else None)
    return workspace.get_probe(project_directory=probe_dir)


def _load_multi_probes(probe_dirs: List[str], model_id: Optional[str]) -> List[ConceptProbe]:
    first_cfg = _load_json(Path(probe_dirs[0]) / "config.json")
    model_cfg = first_cfg.get("model", {})
    if model_id:
        model_cfg["model_id"] = model_id
    bundle = ModelBundle.load(model_cfg)
    probes: List[ConceptProbe] = []
    for path in probe_dirs:
        probes.append(ConceptProbe.load(run_dir=path, model_bundle=bundle))
    return probes


def _load_segments_from_npz(tokenizer, npz_path: str, scores: np.ndarray, prompt_len: int) -> List[Dict[str, object]]:
    data = np.load(npz_path)
    token_ids = data.get("token_ids")
    if token_ids is None:
        raise ValueError("token_ids missing from npz; cannot segment.")
    token_ids = np.array(token_ids, dtype=np.int64).tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return segment_token_scores(tokens, scores.tolist(), prompt_len=prompt_len)


def _summarize_segments(segments: List[Dict[str, object]]) -> Dict[str, Optional[float]]:
    def weighted_mean(items: Iterable[Dict[str, object]]) -> Optional[float]:
        total = 0.0
        count = 0
        for item in items:
            token_count = int(item.get("token_count", 0))
            if token_count <= 0:
                continue
            mean_score = float(item.get("mean_score", 0.0))
            total += mean_score * token_count
            count += token_count
        if count <= 0:
            return None
        return float(total / count)

    def last_mean(items: List[Dict[str, object]]) -> Optional[float]:
        if not items:
            return None
        items = sorted(items, key=lambda x: int(x.get("segment_index", 0)))
        return float(items[-1].get("mean_score", 0.0))

    summary: Dict[str, Optional[float]] = {}
    prompt_segments = [s for s in segments if s.get("phase") == "prompt"]
    completion_segments = [s for s in segments if s.get("phase") == "completion"]

    summary["prompt_mean"] = weighted_mean(prompt_segments)
    summary["completion_mean"] = weighted_mean(completion_segments)
    summary["prompt_last_mean"] = last_mean(prompt_segments)
    summary["completion_last_mean"] = last_mean(completion_segments)

    for role in ("assistant", "user", "system"):
        role_prompt = [s for s in prompt_segments if s.get("role") == role]
        role_completion = [s for s in completion_segments if s.get("role") == role]
        summary[f"prompt_{role}_mean"] = weighted_mean(role_prompt)
        summary[f"prompt_{role}_last_mean"] = last_mean(role_prompt)
        summary[f"completion_{role}_mean"] = weighted_mean(role_completion)
        summary[f"completion_{role}_last_mean"] = last_mean(role_completion)

    return summary


def run_experiment(cfg: ExperimentConfig) -> Path:
    dataset_path = Path(cfg.dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = cfg.output_dir_template
    if "{timestamp}" in output_dir:
        output_dir = output_dir.format(timestamp=_now_tag())
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = _load_json(dataset_path)
    if cfg.prompt is None:
        raise ValueError("cfg.prompt must be provided.")

    prompts, metas = _collect_prompts(dataset, cfg.prompt)
    if not prompts:
        raise ValueError("No prompts generated from dataset.")

    use_multi = cfg.multi_probe if cfg.multi_probe is not None else len(cfg.probe_dirs) > 1
    if use_multi:
        probes = _load_multi_probes(cfg.probe_dirs, cfg.model_id)
        model_bundle = probes[0].model_bundle
    else:
        probe = _load_single_probe(cfg.probe_dirs[0], cfg.model_id)
        probes = [probe]
        model_bundle = probe.model_bundle

    tokenizer = model_bundle.tokenizer
    rating_sources = _normalize_rating_sources(cfg.self_ratings.sources)
    primary_source = str(cfg.self_ratings.primary_source).strip().lower()
    if primary_source not in rating_sources:
        primary_source = rating_sources[0]
    rating_key_by_source = {source: _rating_source_key(source) for source in rating_sources}
    source_titles = {
        "token": f"{cfg.rating.label.capitalize()} (token)",
        "logit": f"{cfg.rating.label.capitalize()} (logit)",
    }

    use_logit_source = "logit" in rating_sources
    if use_logit_source and not cfg.logit_rating.enabled:
        raise ValueError("self_ratings.sources includes 'logit' but logit_rating.enabled is false.")
    save_generation_logits = bool(
        cfg.logit_rating.save_generation_logits and cfg.logit_rating.enabled and use_logit_source
    )
    if use_logit_source and not save_generation_logits:
        raise ValueError(
            "Logit self-ratings require generation logits. "
            "Set logit_rating.save_generation_logits=true."
        )

    option_values: Dict[str, float] = {}
    option_token_ids: Dict[str, List[int]] = {}
    if use_logit_source:
        for raw_value in cfg.logit_rating.option_values:
            value = int(raw_value)
            key = str(value)
            if key not in option_values:
                option_values[key] = float(value)
        if not option_values:
            raise ValueError("logit_rating.option_values is empty.")
        option_token_ids = _build_option_token_id_map(tokenizer, cfg.logit_rating)
    bootstrap_ci_level = float(cfg.analysis.bootstrap_ci_level)
    if not (0.0 < bootstrap_ci_level < 1.0):
        raise ValueError("analysis.bootstrap_ci_level must be between 0 and 1 (exclusive).")
    bootstrap_samples = int(max(0, int(cfg.analysis.bootstrap_samples)))
    bootstrap_rng = np.random.default_rng(int(cfg.analysis.bootstrap_seed))
    trend_min_points = int(max(2, int(cfg.analysis.trend_min_points)))
    alpha_reference_value = float(cfg.analysis.alpha_reference_value)
    annotate_plots = bool(cfg.analysis.annotate_plots)

    alphas = cfg.steering.alphas or [0.0]
    alpha_unit = cfg.steering.alpha_unit

    if use_multi:
        result_payload = multi_probe_score_prompts(
            probes=probes,
            prompts=prompts,
            output_root="outputs_multi",
            project_name="conversation_experiment_logit_ratings",
            output_subdir=cfg.output_subdir,
            alphas=alphas,
            alpha_unit=alpha_unit,
            steer_probe=cfg.steering.steer_probe,
            steer_layers=cfg.steering.steer_layers,
            steer_window_radius=cfg.steering.steer_window_radius,
            steer_distribute=cfg.steering.steer_distribute,
            max_new_tokens=cfg.generation.max_new_tokens,
            greedy=cfg.generation.greedy,
            temperature=cfg.generation.temperature,
            top_p=cfg.generation.top_p,
            save_html=cfg.generation.save_html,
            save_segments=cfg.generation.save_segments,
            save_generation_logits=save_generation_logits,
            generation_logits_top_k=cfg.logit_rating.generation_logits_top_k,
            generation_logits_dtype=cfg.logit_rating.generation_logits_dtype,
        )
        recs = result_payload["results"]
        probe_names = [p.concept.name for p in probes]
    else:
        recs = probe.score_prompts(
            prompts=prompts,
            output_subdir=cfg.output_subdir,
            alphas=alphas,
            alpha_unit=alpha_unit,
            max_new_tokens=cfg.generation.max_new_tokens,
            greedy=cfg.generation.greedy,
            temperature=cfg.generation.temperature,
            top_p=cfg.generation.top_p,
            save_html=cfg.generation.save_html,
            save_segments=cfg.generation.save_segments,
            save_generation_logits=save_generation_logits,
            generation_logits_top_k=cfg.logit_rating.generation_logits_top_k,
            generation_logits_dtype=cfg.logit_rating.generation_logits_dtype,
        )
        probe_names = [probes[0].concept.name]

    results: List[Dict[str, object]] = []
    per_alpha: Dict[float, List[Dict[str, object]]] = {float(a): [] for a in alphas}

    for idx, rec in enumerate(recs):
        prompt_idx = idx // len(alphas)
        if prompt_idx >= len(metas):
            break
        meta = metas[prompt_idx]
        completion = str(rec.get("completion", ""))
        token_rating = _parse_rating(completion, cfg.rating)
        prompt_len = int(rec.get("prompt_len", 0))
        npz_path = rec.get("npz_path")
        segments_path = rec.get("segments_path")

        probe_metrics: Dict[str, Dict[str, Optional[float]]] = {}

        if segments_path:
            seg_payload = _load_json(Path(segments_path))
            if use_multi:
                seg_by_probe = seg_payload.get("segments_by_probe", {})
                for name in probe_names:
                    segments = seg_by_probe.get(name, [])
                    if isinstance(segments, list):
                        probe_metrics[name] = _summarize_segments(segments)
            else:
                segments = seg_payload.get("segments", [])
                if isinstance(segments, list):
                    probe_metrics[probe_names[0]] = _summarize_segments(segments)
        else:
            if not npz_path:
                raise ValueError("Missing npz_path; cannot compute segments.")
            npz = np.load(npz_path)
            scores_agg = npz.get("scores_agg")
            if scores_agg is None:
                raise ValueError("scores_agg missing from npz; cannot compute segments.")
            if use_multi:
                for probe_idx, name in enumerate(probe_names):
                    segments = _load_segments_from_npz(
                        tokenizer,
                        npz_path,
                        scores_agg[probe_idx],
                        prompt_len,
                    )
                    probe_metrics[name] = _summarize_segments(segments)
            else:
                segments = _load_segments_from_npz(tokenizer, npz_path, scores_agg, prompt_len)
                probe_metrics[probe_names[0]] = _summarize_segments(segments)

        logit_rating_value: Optional[float] = None
        logit_rating_probs: Optional[Dict[str, float]] = None
        logit_rating_status = "disabled"
        if use_logit_source:
            if isinstance(npz_path, str):
                payload = _compute_logit_rating_from_npz(
                    npz_path,
                    option_token_ids=option_token_ids,
                    option_values=option_values,
                    step_index=int(cfg.logit_rating.step_index),
                    save_option_probabilities=bool(cfg.logit_rating.save_option_probabilities),
                )
                rating_value = payload.get("logit_rating")
                if isinstance(rating_value, (int, float)) and np.isfinite(float(rating_value)):
                    logit_rating_value = float(rating_value)
                probs = payload.get("logit_rating_probs")
                if isinstance(probs, dict):
                    logit_rating_probs = {str(k): float(v) for k, v in probs.items()}
                logit_rating_status = str(payload.get("logit_rating_status", "error"))
            else:
                logit_rating_status = "missing_npz_path"

        primary_key = rating_key_by_source[primary_source]
        primary_rating = token_rating if primary_key == "token_rating" else logit_rating_value

        record = {
            **meta,
            "completion": completion,
            "rating": primary_rating,
            "token_rating": token_rating,
            "logit_rating": logit_rating_value,
            "logit_rating_probs": logit_rating_probs,
            "logit_rating_status": logit_rating_status,
            "rating_source_primary": primary_source,
            "alpha": float(rec.get("alpha", 0.0)),
            "alpha_unit": str(rec.get("alpha_unit", alpha_unit)),
            "npz_path": npz_path,
            "segments_path": segments_path,
            "probe_metrics": probe_metrics,
            "generation_logits_saved": bool(rec.get("generation_logits_saved", False)),
        }
        results.append(record)
        alpha_key = float(rec.get("alpha", 0.0))
        per_alpha.setdefault(alpha_key, []).append(record)

    results_path = output_path / "results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    summary: Dict[str, object] = {
        "dataset_path": str(dataset_path),
        "probe_dirs": cfg.probe_dirs,
        "num_prompts": len(prompts),
        "num_results": len(results),
        "alphas": alphas,
        "alpha_unit": alpha_unit,
        "rating_label": cfg.rating.label,
        "rating_sources": rating_sources,
        "primary_rating_source": primary_source,
        "rating_keys": rating_key_by_source,
        "bootstrap": {
            "samples": bootstrap_samples,
            "ci_level": bootstrap_ci_level,
            "seed": int(cfg.analysis.bootstrap_seed),
            "trend_min_points": trend_min_points,
            "alpha_reference_value": alpha_reference_value,
        },
    }
    if use_logit_source:
        summary["logit_rating"] = {
            "option_values": [int(v) for v in cfg.logit_rating.option_values],
            "option_token_templates": list(cfg.logit_rating.option_token_templates),
            "step_index": int(cfg.logit_rating.step_index),
            "save_generation_logits": bool(save_generation_logits),
            "generation_logits_top_k": cfg.logit_rating.generation_logits_top_k,
            "generation_logits_dtype": cfg.logit_rating.generation_logits_dtype,
        }

    turnwise_stats_keys: List[str] = []
    turnwise_metric_keys: List[str] = []
    turnwise_payload_by_source: Dict[str, Dict[str, object]] = {}
    if cfg.analysis.plot_turnwise_relationship_vs_alpha:
        turnwise_stats_keys = _normalize_turnwise_stats(cfg.analysis.turnwise_relationship_stats)
        turnwise_metric_keys = (
            list(cfg.analysis.turnwise_relationship_metric_keys)
            if cfg.analysis.turnwise_relationship_metric_keys
            else list(cfg.analysis.plot_rating_vs_metrics)
        )
        for source in rating_sources:
            turnwise_payload_by_source[source] = {
                "analysis_dir": str(output_path),
                "results_file": "results.json",
                "rating_key": rating_key_by_source[source],
                "stats": turnwise_stats_keys,
                "per_probe": {},
            }

    summary_per_probe: Dict[str, object] = {}
    alpha_vals_sorted = sorted(per_alpha.keys())
    for probe_name in probe_names:
        per_alpha_summary: Dict[str, object] = {}
        for alpha_val in alpha_vals_sorted:
            records = per_alpha.get(float(alpha_val), [])
            metric_by_turn: Dict[str, Dict[int, List[float]]] = {
                key: {} for key in cfg.analysis.plot_vs_turn_metrics
            }
            for rec in records:
                turn = int(rec["turn_index"])
                probe_metrics = rec.get("probe_metrics", {}).get(probe_name, {})
                if not isinstance(probe_metrics, dict):
                    continue
                for key in cfg.analysis.plot_vs_turn_metrics:
                    value = probe_metrics.get(key)
                    if isinstance(value, (int, float)) and np.isfinite(float(value)):
                        metric_by_turn[key].setdefault(turn, []).append(float(value))

            alpha_tag = _alpha_label(alpha_val)
            alpha_plot_dir = _alpha_plot_dir(output_path, alpha_val)
            for key in cfg.analysis.plot_vs_turn_metrics:
                turns_metric = sorted(metric_by_turn[key].keys())
                means: List[float] = []
                sems: List[float] = []
                for turn in turns_metric:
                    mean, sem = _mean_sem(metric_by_turn[key].get(turn, []))
                    means.append(mean)
                    sems.append(sem)
                if turns_metric:
                    _plot_series(
                        turns_metric,
                        means,
                        sems,
                        title=f"{probe_name} {key} vs turn (alpha={alpha_val})",
                        ylabel=f"{probe_name} {key}",
                        out_path=alpha_plot_dir / f"{probe_name}_{key}_vs_turn_alpha_{alpha_tag}.png",
                    )

            rating_source_summary: Dict[str, object] = {}
            for source in rating_sources:
                rating_key = rating_key_by_source[source]
                rating_by_turn: Dict[int, List[float]] = {}
                paired_rating: List[float] = []
                paired_metrics: Dict[str, List[float]] = {
                    key: [] for key in cfg.analysis.plot_rating_vs_metrics
                }
                for rec in records:
                    turn = int(rec["turn_index"])
                    rating_value = rec.get(rating_key)
                    if not isinstance(rating_value, (int, float)) or not np.isfinite(float(rating_value)):
                        continue
                    rating_float = float(rating_value)
                    rating_by_turn.setdefault(turn, []).append(rating_float)
                    paired_rating.append(rating_float)
                    probe_metrics = rec.get("probe_metrics", {}).get(probe_name, {})
                    if not isinstance(probe_metrics, dict):
                        continue
                    for key in cfg.analysis.plot_rating_vs_metrics:
                        metric_val = probe_metrics.get(key)
                        if isinstance(metric_val, (int, float)) and np.isfinite(float(metric_val)):
                            paired_metrics[key].append(float(metric_val))

                turns = sorted(rating_by_turn.keys())
                rating_means: List[float] = []
                rating_sems: List[float] = []
                for turn in turns:
                    mean, sem = _mean_sem(rating_by_turn.get(turn, []))
                    rating_means.append(mean)
                    rating_sems.append(sem)

                source_alpha_dir = _rating_source_alpha_dir(output_path, alpha_val, source)
                rating_turn_trend_bootstrap: Dict[str, Optional[float]] = {}
                if turns:
                    rating_turn_trend_bootstrap = _bootstrap_series_trend(
                        [float(t) for t in turns],
                        [float(v) for v in rating_means],
                        bootstrap_samples=bootstrap_samples,
                        ci_level=bootstrap_ci_level,
                        min_points=trend_min_points,
                        rng=bootstrap_rng,
                    )
                    annotation_lines = (
                        _annotation_lines_for_trend(rating_turn_trend_bootstrap, label="turn")
                        if annotate_plots
                        else None
                    )
                    _plot_series(
                        turns,
                        rating_means,
                        rating_sems,
                        title=f"{source_titles.get(source, source)} vs turn (alpha={alpha_val})",
                        ylabel=source_titles.get(source, source),
                        out_path=source_alpha_dir / f"{cfg.rating.label}_{source}_vs_turn_alpha_{alpha_tag}.png",
                        annotation_lines=annotation_lines,
                    )

                for key in cfg.analysis.plot_rating_vs_metrics:
                    xs = paired_rating
                    ys = paired_metrics.get(key, [])
                    if xs and ys:
                        _plot_scatter(
                            xs,
                            ys,
                            title=f"{probe_name} {key} vs {source_titles.get(source, source)} (alpha={alpha_val})",
                            xlabel=source_titles.get(source, source),
                            ylabel=f"{probe_name} {key}",
                            out_path=source_alpha_dir
                            / f"{probe_name}_{key}_vs_{cfg.rating.label}_{source}_alpha_{alpha_tag}.png",
                        )

                rating_source_summary[source] = {
                    "rating_key": rating_key,
                    "rating_vs_turn": _linear_stats(turns, rating_means) if turns else {},
                    "rating_vs_turn_trend_bootstrap": rating_turn_trend_bootstrap,
                    "rating_vs_metrics": {
                        key: _correlation_stats(paired_rating, paired_metrics.get(key, []))
                        for key in cfg.analysis.plot_rating_vs_metrics
                    },
                    "alpha_plot_dir": str(source_alpha_dir),
                }

            per_alpha_summary[str(alpha_val)] = {
                "alpha": alpha_val,
                "alpha_plot_dir": str(alpha_plot_dir),
                "rating_sources": rating_source_summary,
            }

        probe_summary: Dict[str, object] = {"per_alpha": per_alpha_summary}
        if cfg.analysis.plot_turnwise_relationship_vs_alpha and turnwise_metric_keys:
            probe_turnwise_by_source: Dict[str, object] = {}
            for source in rating_sources:
                probe_turnwise: Dict[str, object] = {}
                for metric_key in turnwise_metric_keys:
                    metric_payload = _build_turnwise_relationship_by_alpha(
                        per_alpha,
                        probe_name=probe_name,
                        metric_key=metric_key,
                        rating_key=rating_key_by_source[source],
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_ci_level=bootstrap_ci_level,
                        bootstrap_rng=bootstrap_rng,
                    )
                    turns_for_metric = metric_payload.get("turns", [])
                    stats_by_alpha_turn = metric_payload.get("stats_by_alpha_turn", {})
                    plot_paths_by_alpha: Dict[str, Dict[str, Optional[str]]] = {}
                    trend_by_alpha: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
                    for alpha_key in sorted(stats_by_alpha_turn.keys(), key=lambda s: float(s)):
                        alpha_val = float(alpha_key)
                        metric_dir = (
                            _rating_source_alpha_dir(output_path, alpha_val, source)
                            / "turnwise_correlation"
                            / _sanitize_slug(probe_name)
                            / _sanitize_slug(metric_key)
                        )
                        metric_dir.mkdir(parents=True, exist_ok=True)
                        per_stat_paths: Dict[str, Optional[str]] = {}
                        rows_for_alpha = stats_by_alpha_turn.get(alpha_key, [])
                        row_by_turn: Dict[int, Dict[str, Optional[float]]] = {}
                        for row in rows_for_alpha:
                            turn_obj = row.get("turn")
                            if isinstance(turn_obj, int):
                                row_by_turn[int(turn_obj)] = row
                        per_stat_trends: Dict[str, Dict[str, Optional[float]]] = {}
                        for stat_key in turnwise_stats_keys:
                            ys_turn: List[Optional[float]] = [
                                _safe_float(row_by_turn.get(int(turn), {}).get(stat_key))
                                for turn in list(turns_for_metric)
                            ]
                            stat_trend = _bootstrap_series_trend(
                                [float(t) for t in list(turns_for_metric)],
                                ys_turn,
                                bootstrap_samples=bootstrap_samples,
                                ci_level=bootstrap_ci_level,
                                min_points=trend_min_points,
                                rng=bootstrap_rng,
                            )
                            per_stat_trends[stat_key] = stat_trend
                            annotation_lines = (
                                _annotation_lines_for_trend(stat_trend, label="turn")
                                if annotate_plots
                                else None
                            )
                            out_path = metric_dir / f"{_sanitize_slug(stat_key)}_vs_turn.png"
                            made = _plot_turnwise_stat_for_alpha(
                                turns=list(turns_for_metric),
                                rows_for_alpha=rows_for_alpha,
                                alpha=alpha_val,
                                stat_key=stat_key,
                                title=f"{stat_key} vs turn (alpha={alpha_val:g}, source={source})",
                                out_path=out_path,
                                annotation_lines=annotation_lines,
                            )
                            per_stat_paths[stat_key] = str(out_path) if made else None
                        plot_paths_by_alpha[alpha_key] = per_stat_paths
                        trend_by_alpha[alpha_key] = per_stat_trends
                    metric_payload["plot_paths_by_alpha"] = plot_paths_by_alpha
                    metric_payload["trend_by_alpha"] = trend_by_alpha
                    probe_turnwise[metric_key] = metric_payload
                probe_turnwise_by_source[source] = probe_turnwise
                turnwise_payload_by_source[source]["per_probe"][probe_name] = probe_turnwise
            probe_summary["turnwise_relationship_vs_alpha"] = probe_turnwise_by_source

        summary_per_probe[probe_name] = probe_summary

    summary["per_probe"] = summary_per_probe
    if cfg.analysis.plot_turnwise_relationship_vs_alpha and turnwise_payload_by_source:
        turnwise_json_path = output_path / cfg.analysis.turnwise_relationship_json_name
        turnwise_json_path.write_text(
            json.dumps(
                {
                    "analysis_dir": str(output_path),
                    "results_file": "results.json",
                    "metric_keys": turnwise_metric_keys,
                    "stats": turnwise_stats_keys,
                    "per_source": turnwise_payload_by_source,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        summary["turnwise_relationship_vs_alpha"] = {
            "json_path": str(turnwise_json_path),
            "stats": turnwise_stats_keys,
            "metric_keys": turnwise_metric_keys,
            "sources": rating_sources,
        }

    if cfg.analysis.plot_rating_vs_alpha:
        rating_vs_alpha_summary: Dict[str, object] = {}
        for source in rating_sources:
            rating_key = rating_key_by_source[source]
            alpha_vals: List[float] = []
            alpha_means: List[float] = []
            alpha_sems: List[float] = []
            ratings_by_alpha: Dict[float, List[float]] = {}
            for alpha_val in alphas:
                records = per_alpha.get(float(alpha_val), [])
                ratings = [
                    float(r.get(rating_key))
                    for r in records
                    if isinstance(r.get(rating_key), (int, float))
                    and np.isfinite(float(r.get(rating_key)))
                ]
                if not ratings:
                    continue
                mean, sem = _mean_sem(ratings)
                alpha_vals.append(float(alpha_val))
                alpha_means.append(mean)
                alpha_sems.append(sem)
                ratings_by_alpha[float(alpha_val)] = list(ratings)
            if alpha_vals:
                source_dir = _rating_source_dir(output_path, source)
                plot_path = source_dir / f"{cfg.rating.label}_{source}_vs_alpha.png"
                trend_bootstrap = _bootstrap_series_trend(
                    alpha_vals,
                    [float(v) for v in alpha_means],
                    bootstrap_samples=bootstrap_samples,
                    ci_level=bootstrap_ci_level,
                    min_points=trend_min_points,
                    rng=bootstrap_rng,
                )
                ref_alpha = _nearest_alpha_key(alpha_vals, alpha_reference_value)
                effect_by_alpha: Dict[str, Dict[str, Optional[float]]] = {}
                if ref_alpha is not None:
                    ref_values = ratings_by_alpha.get(float(ref_alpha), [])
                    for alpha_val in alpha_vals:
                        if abs(float(alpha_val) - float(ref_alpha)) <= 1e-12:
                            continue
                        cmp_values = ratings_by_alpha.get(float(alpha_val), [])
                        effect = _bootstrap_value_metric_difference(
                            ref_values,
                            cmp_values,
                            metric_key="mean",
                            bootstrap_samples=bootstrap_samples,
                            ci_level=bootstrap_ci_level,
                            rng=bootstrap_rng,
                        )
                        effect_by_alpha[str(float(alpha_val))] = effect
                annotation_lines: Optional[List[str]] = None
                if annotate_plots:
                    annotation_lines = []
                    annotation_lines.extend(_annotation_lines_for_trend(trend_bootstrap, label="alpha"))
                    if ref_alpha is not None and effect_by_alpha:
                        annotation_lines.extend(
                            _annotation_lines_for_vs_reference(
                                effect_by_alpha,
                                reference_alpha=float(ref_alpha),
                            )
                        )
                _plot_alpha_series(
                    alpha_vals,
                    alpha_means,
                    alpha_sems,
                    title=f"{source_titles.get(source, source)} vs alpha",
                    ylabel=source_titles.get(source, source),
                    out_path=plot_path,
                    annotation_lines=annotation_lines,
                )
                rating_vs_alpha_summary[source] = {
                    "rating_key": rating_key,
                    "trend": _linear_stats(alpha_vals, alpha_means),
                    "trend_bootstrap": trend_bootstrap,
                    "correlation": _correlation_stats(alpha_vals, alpha_means),
                    "alpha_reference_value": float(ref_alpha) if ref_alpha is not None else None,
                    "alpha_vs_reference_mean_difference": effect_by_alpha,
                    "plot_path": str(plot_path),
                }
        if rating_vs_alpha_summary:
            summary["rating_vs_alpha"] = rating_vs_alpha_summary

    if cfg.analysis.plot_alignment_slope_vs_alpha:
        alignment_summary: Dict[str, object] = {}
        if not probe_names:
            alignment_summary = {"error": "No probes available."}
        else:
            alignment_probe_name = cfg.analysis.alignment_probe_name or probe_names[0]
            if alignment_probe_name not in probe_names:
                alignment_summary = {
                    "error": f"alignment_probe_name '{alignment_probe_name}' not found in probes.",
                    "available_probes": probe_names,
                }
            else:
                for source in rating_sources:
                    rating_key = rating_key_by_source[source]
                    slopes: List[Optional[float]] = []
                    pvalues: List[Optional[float]] = []
                    slope_ci_lows: List[Optional[float]] = []
                    slope_ci_highs: List[Optional[float]] = []
                    pvalue_ci_lows: List[Optional[float]] = []
                    pvalue_ci_highs: List[Optional[float]] = []
                    num_pairs_per_alpha: List[int] = []
                    pair_data_by_alpha: Dict[float, Tuple[List[float], List[float]]] = {}
                    for alpha_val in alpha_vals_sorted:
                        rows = per_alpha.get(float(alpha_val), [])
                        xs: List[float] = []
                        ys: List[float] = []
                        for row in rows:
                            rating_value = row.get(rating_key)
                            if not isinstance(rating_value, (int, float)) or not np.isfinite(float(rating_value)):
                                continue
                            probe_metrics = row.get("probe_metrics", {}).get(alignment_probe_name, {})
                            if not isinstance(probe_metrics, dict):
                                continue
                            metric_value = probe_metrics.get(cfg.analysis.alignment_metric_key)
                            if not isinstance(metric_value, (int, float)) or not np.isfinite(float(metric_value)):
                                continue
                            xs.append(float(rating_value))
                            ys.append(float(metric_value))
                        pair_data_by_alpha[float(alpha_val)] = (list(xs), list(ys))
                        boot_stats = _bootstrap_pair_stats(
                            xs,
                            ys,
                            bootstrap_samples=bootstrap_samples,
                            ci_level=bootstrap_ci_level,
                            rng=bootstrap_rng,
                        )
                        slopes.append(_safe_float(boot_stats.get("slope")))
                        pvalues.append(_safe_float(boot_stats.get("linreg_p")))
                        slope_ci_lows.append(_safe_float(boot_stats.get("slope_ci_low")))
                        slope_ci_highs.append(_safe_float(boot_stats.get("slope_ci_high")))
                        pvalue_ci_lows.append(_safe_float(boot_stats.get("linreg_p_ci_low")))
                        pvalue_ci_highs.append(_safe_float(boot_stats.get("linreg_p_ci_high")))
                        num_pairs_per_alpha.append(int(boot_stats.get("count", 0.0) or 0))

                    source_dir = _rating_source_dir(output_path, source)
                    slope_plot_path = source_dir / _suffix_filename(cfg.analysis.alignment_slope_plot_name, source)
                    slope_trend_bootstrap = _bootstrap_series_trend(
                        [float(a) for a in alpha_vals_sorted],
                        slopes,
                        bootstrap_samples=bootstrap_samples,
                        ci_level=bootstrap_ci_level,
                        min_points=trend_min_points,
                        rng=bootstrap_rng,
                    )
                    ref_alpha = _nearest_alpha_key(alpha_vals_sorted, alpha_reference_value)
                    slope_effect_by_alpha: Dict[str, Dict[str, Optional[float]]] = {}
                    pvalue_effect_by_alpha: Dict[str, Dict[str, Optional[float]]] = {}
                    if ref_alpha is not None and float(ref_alpha) in pair_data_by_alpha:
                        ref_xs, ref_ys = pair_data_by_alpha[float(ref_alpha)]
                        for alpha_val in alpha_vals_sorted:
                            if abs(float(alpha_val) - float(ref_alpha)) <= 1e-12:
                                continue
                            cmp_pair = pair_data_by_alpha.get(float(alpha_val))
                            if cmp_pair is None:
                                continue
                            cmp_xs, cmp_ys = cmp_pair
                            slope_effect_by_alpha[str(float(alpha_val))] = _bootstrap_pair_metric_difference(
                                ref_xs,
                                ref_ys,
                                cmp_xs,
                                cmp_ys,
                                metric_key="slope",
                                bootstrap_samples=bootstrap_samples,
                                ci_level=bootstrap_ci_level,
                                rng=bootstrap_rng,
                            )
                            pvalue_effect_by_alpha[str(float(alpha_val))] = _bootstrap_pair_metric_difference(
                                ref_xs,
                                ref_ys,
                                cmp_xs,
                                cmp_ys,
                                metric_key="linreg_p",
                                bootstrap_samples=bootstrap_samples,
                                ci_level=bootstrap_ci_level,
                                rng=bootstrap_rng,
                            )
                    slope_annotations: Optional[List[str]] = None
                    if annotate_plots:
                        slope_annotations = []
                        slope_annotations.extend(_annotation_lines_for_trend(slope_trend_bootstrap, label="alpha"))
                        if ref_alpha is not None and slope_effect_by_alpha:
                            slope_annotations.extend(
                                _annotation_lines_for_vs_reference(
                                    slope_effect_by_alpha,
                                    reference_alpha=float(ref_alpha),
                                )
                            )
                    _plot_alpha_line(
                        alpha_vals_sorted,
                        slopes,
                        title=(
                            f"Slope vs alpha: {alignment_probe_name} "
                            f"{cfg.analysis.alignment_metric_key} vs {rating_key}"
                        ),
                        ylabel="Slope",
                        out_path=slope_plot_path,
                        ci_low_in=slope_ci_lows,
                        ci_high_in=slope_ci_highs,
                        annotation_lines=slope_annotations,
                    )

                    pvalue_plot_path: Optional[Path] = None
                    pvalue_trend_bootstrap: Dict[str, Optional[float]] = {}
                    if any(p is not None for p in pvalues):
                        pvalue_trend_bootstrap = _bootstrap_series_trend(
                            [float(a) for a in alpha_vals_sorted],
                            pvalues,
                            bootstrap_samples=bootstrap_samples,
                            ci_level=bootstrap_ci_level,
                            min_points=trend_min_points,
                            rng=bootstrap_rng,
                        )
                        pvalue_annotations: Optional[List[str]] = None
                        if annotate_plots:
                            pvalue_annotations = []
                            pvalue_annotations.extend(_annotation_lines_for_trend(pvalue_trend_bootstrap, label="alpha"))
                            if ref_alpha is not None and pvalue_effect_by_alpha:
                                pvalue_annotations.extend(
                                    _annotation_lines_for_vs_reference(
                                        pvalue_effect_by_alpha,
                                        reference_alpha=float(ref_alpha),
                                    )
                                )
                        pvalue_plot_path = source_dir / _suffix_filename(
                            cfg.analysis.alignment_pvalue_plot_name, source
                        )
                        _plot_alpha_line(
                            alpha_vals_sorted,
                            pvalues,
                            title=(
                                f"Slope p-value vs alpha: {alignment_probe_name} "
                                f"{cfg.analysis.alignment_metric_key} vs {rating_key}"
                            ),
                            ylabel="P-value",
                            out_path=pvalue_plot_path,
                            ci_low_in=pvalue_ci_lows,
                            ci_high_in=pvalue_ci_highs,
                            log_y=True,
                            annotation_lines=pvalue_annotations,
                        )

                    valid_alpha_for_slope = [
                        float(a) for a, s in zip(alpha_vals_sorted, slopes) if s is not None and not np.isnan(s)
                    ]
                    valid_slopes = [float(s) for s in slopes if s is not None and not np.isnan(s)]
                    alignment_summary[source] = {
                        "probe_name": alignment_probe_name,
                        "metric_key": cfg.analysis.alignment_metric_key,
                        "rating_key": rating_key,
                        "alphas": [float(a) for a in alpha_vals_sorted],
                        "slopes": slopes,
                        "pvalues": pvalues,
                        "slope_ci_lows": slope_ci_lows,
                        "slope_ci_highs": slope_ci_highs,
                        "pvalue_ci_lows": pvalue_ci_lows,
                        "pvalue_ci_highs": pvalue_ci_highs,
                        "num_pairs_per_alpha": num_pairs_per_alpha,
                        "slope_vs_alpha_correlation": (
                            _correlation_stats(valid_alpha_for_slope, valid_slopes)
                            if len(valid_slopes) >= 2
                            else {}
                        ),
                        "slope_vs_alpha_trend_bootstrap": slope_trend_bootstrap,
                        "pvalue_vs_alpha_trend_bootstrap": pvalue_trend_bootstrap,
                        "alpha_reference_value": float(ref_alpha) if ref_alpha is not None else None,
                        "slope_vs_reference_difference": slope_effect_by_alpha,
                        "pvalue_vs_reference_difference": pvalue_effect_by_alpha,
                        "bootstrap_samples": bootstrap_samples,
                        "bootstrap_ci_level": bootstrap_ci_level,
                        "slope_plot_path": str(slope_plot_path),
                        "pvalue_plot_path": str(pvalue_plot_path) if pvalue_plot_path is not None else None,
                    }
        summary["alignment_slope_vs_alpha"] = alignment_summary

    if cfg.analysis.plot_r2_vs_alpha:
        r2_summary: Dict[str, object] = {}
        if not probe_names:
            r2_summary = {"error": "No probes available."}
        else:
            r2_probe_name = cfg.analysis.r2_probe_name or cfg.analysis.alignment_probe_name or probe_names[-1]
            if r2_probe_name not in probe_names:
                r2_summary = {
                    "error": f"r2_probe_name '{r2_probe_name}' not found in probes.",
                    "available_probes": probe_names,
                }
            else:
                for source in rating_sources:
                    rating_key = rating_key_by_source[source]
                    r2_vals: List[Optional[float]] = []
                    r2_ci_lows: List[Optional[float]] = []
                    r2_ci_highs: List[Optional[float]] = []
                    isotonic_r2_vals: List[Optional[float]] = []
                    isotonic_r2_ci_lows: List[Optional[float]] = []
                    isotonic_r2_ci_highs: List[Optional[float]] = []
                    pairs_per_alpha: List[int] = []
                    pair_data_by_alpha: Dict[float, Tuple[List[float], List[float]]] = {}
                    for alpha_val in alpha_vals_sorted:
                        rows = per_alpha.get(float(alpha_val), [])
                        xs: List[float] = []
                        ys: List[float] = []
                        for row in rows:
                            rating_value = row.get(rating_key)
                            if not isinstance(rating_value, (int, float)) or not np.isfinite(float(rating_value)):
                                continue
                            probe_metrics = row.get("probe_metrics", {}).get(r2_probe_name, {})
                            if not isinstance(probe_metrics, dict):
                                continue
                            metric_value = probe_metrics.get(cfg.analysis.r2_metric_key)
                            if not isinstance(metric_value, (int, float)) or not np.isfinite(float(metric_value)):
                                continue
                            xs.append(float(rating_value))
                            ys.append(float(metric_value))
                        pair_data_by_alpha[float(alpha_val)] = (list(xs), list(ys))
                        boot_stats = _bootstrap_pair_stats(
                            xs,
                            ys,
                            bootstrap_samples=bootstrap_samples,
                            ci_level=bootstrap_ci_level,
                            rng=bootstrap_rng,
                        )
                        r2_vals.append(_safe_float(boot_stats.get("r2")))
                        r2_ci_lows.append(_safe_float(boot_stats.get("r2_ci_low")))
                        r2_ci_highs.append(_safe_float(boot_stats.get("r2_ci_high")))
                        isotonic_r2_vals.append(_safe_float(boot_stats.get("isotonic_r2")))
                        isotonic_r2_ci_lows.append(_safe_float(boot_stats.get("isotonic_r2_ci_low")))
                        isotonic_r2_ci_highs.append(_safe_float(boot_stats.get("isotonic_r2_ci_high")))
                        pairs_per_alpha.append(int(boot_stats.get("count", 0.0) or 0))

                    source_dir = _rating_source_dir(output_path, source)
                    plot_path = source_dir / _suffix_filename(cfg.analysis.r2_plot_name, source)
                    isotonic_plot_path = source_dir / _suffix_filename(cfg.analysis.r2_plot_name, f"{source}_isotonic")
                    r2_trend_bootstrap = _bootstrap_series_trend(
                        [float(a) for a in alpha_vals_sorted],
                        r2_vals,
                        bootstrap_samples=bootstrap_samples,
                        ci_level=bootstrap_ci_level,
                        min_points=trend_min_points,
                        rng=bootstrap_rng,
                    )
                    isotonic_r2_trend_bootstrap = _bootstrap_series_trend(
                        [float(a) for a in alpha_vals_sorted],
                        isotonic_r2_vals,
                        bootstrap_samples=bootstrap_samples,
                        ci_level=bootstrap_ci_level,
                        min_points=trend_min_points,
                        rng=bootstrap_rng,
                    )
                    ref_alpha = _nearest_alpha_key(alpha_vals_sorted, alpha_reference_value)
                    r2_effect_by_alpha: Dict[str, Dict[str, Optional[float]]] = {}
                    isotonic_r2_effect_by_alpha: Dict[str, Dict[str, Optional[float]]] = {}
                    if ref_alpha is not None and float(ref_alpha) in pair_data_by_alpha:
                        ref_xs, ref_ys = pair_data_by_alpha[float(ref_alpha)]
                        for alpha_val in alpha_vals_sorted:
                            if abs(float(alpha_val) - float(ref_alpha)) <= 1e-12:
                                continue
                            cmp_pair = pair_data_by_alpha.get(float(alpha_val))
                            if cmp_pair is None:
                                continue
                            cmp_xs, cmp_ys = cmp_pair
                            r2_effect_by_alpha[str(float(alpha_val))] = _bootstrap_pair_metric_difference(
                                ref_xs,
                                ref_ys,
                                cmp_xs,
                                cmp_ys,
                                metric_key="r2",
                                bootstrap_samples=bootstrap_samples,
                                ci_level=bootstrap_ci_level,
                                rng=bootstrap_rng,
                            )
                            isotonic_r2_effect_by_alpha[str(float(alpha_val))] = _bootstrap_pair_metric_difference(
                                ref_xs,
                                ref_ys,
                                cmp_xs,
                                cmp_ys,
                                metric_key="isotonic_r2",
                                bootstrap_samples=bootstrap_samples,
                                ci_level=bootstrap_ci_level,
                                rng=bootstrap_rng,
                            )
                    r2_annotations: Optional[List[str]] = None
                    if annotate_plots:
                        r2_annotations = []
                        r2_annotations.extend(_annotation_lines_for_trend(r2_trend_bootstrap, label="alpha"))
                        if ref_alpha is not None and r2_effect_by_alpha:
                            r2_annotations.extend(
                                _annotation_lines_for_vs_reference(
                                    r2_effect_by_alpha,
                                    reference_alpha=float(ref_alpha),
                                )
                            )
                    isotonic_r2_annotations: Optional[List[str]] = None
                    if annotate_plots:
                        isotonic_r2_annotations = []
                        isotonic_r2_annotations.extend(
                            _annotation_lines_for_trend(isotonic_r2_trend_bootstrap, label="alpha")
                        )
                        if ref_alpha is not None and isotonic_r2_effect_by_alpha:
                            isotonic_r2_annotations.extend(
                                _annotation_lines_for_vs_reference(
                                    isotonic_r2_effect_by_alpha,
                                    reference_alpha=float(ref_alpha),
                                )
                            )
                    _plot_alpha_line(
                        alpha_vals_sorted,
                        r2_vals,
                        title=f"R^2 vs alpha: {r2_probe_name} {cfg.analysis.r2_metric_key} vs {rating_key}",
                        ylabel="R^2",
                        out_path=plot_path,
                        ci_low_in=r2_ci_lows,
                        ci_high_in=r2_ci_highs,
                        annotation_lines=r2_annotations,
                    )
                    _plot_alpha_line(
                        alpha_vals_sorted,
                        isotonic_r2_vals,
                        title=f"Isotonic R^2 vs alpha: {r2_probe_name} {cfg.analysis.r2_metric_key} vs {rating_key}",
                        ylabel="Isotonic R^2",
                        out_path=isotonic_plot_path,
                        ci_low_in=isotonic_r2_ci_lows,
                        ci_high_in=isotonic_r2_ci_highs,
                        annotation_lines=isotonic_r2_annotations,
                    )

                    valid_alpha_for_r2 = [
                        float(a) for a, r2 in zip(alpha_vals_sorted, r2_vals) if r2 is not None and not np.isnan(r2)
                    ]
                    valid_r2 = [float(r2) for r2 in r2_vals if r2 is not None and not np.isnan(r2)]
                    valid_alpha_for_iso_r2 = [
                        float(a)
                        for a, iso_r2 in zip(alpha_vals_sorted, isotonic_r2_vals)
                        if iso_r2 is not None and not np.isnan(iso_r2)
                    ]
                    valid_iso_r2 = [
                        float(iso_r2)
                        for iso_r2 in isotonic_r2_vals
                        if iso_r2 is not None and not np.isnan(iso_r2)
                    ]
                    r2_summary[source] = {
                        "probe_name": r2_probe_name,
                        "metric_key": cfg.analysis.r2_metric_key,
                        "rating_key": rating_key,
                        "alphas": [float(a) for a in alpha_vals_sorted],
                        "r2_values": r2_vals,
                        "r2_ci_lows": r2_ci_lows,
                        "r2_ci_highs": r2_ci_highs,
                        "num_pairs_per_alpha": pairs_per_alpha,
                        "r2_vs_alpha_correlation": (
                            _correlation_stats(valid_alpha_for_r2, valid_r2)
                            if len(valid_r2) >= 2
                            else {}
                        ),
                        "r2_vs_alpha_trend_bootstrap": r2_trend_bootstrap,
                        "isotonic_r2_values": isotonic_r2_vals,
                        "isotonic_r2_ci_lows": isotonic_r2_ci_lows,
                        "isotonic_r2_ci_highs": isotonic_r2_ci_highs,
                        "isotonic_r2_vs_alpha_correlation": (
                            _correlation_stats(valid_alpha_for_iso_r2, valid_iso_r2)
                            if len(valid_iso_r2) >= 2
                            else {}
                        ),
                        "isotonic_r2_vs_alpha_trend_bootstrap": isotonic_r2_trend_bootstrap,
                        "alpha_reference_value": float(ref_alpha) if ref_alpha is not None else None,
                        "r2_vs_reference_difference": r2_effect_by_alpha,
                        "isotonic_r2_vs_reference_difference": isotonic_r2_effect_by_alpha,
                        "bootstrap_samples": bootstrap_samples,
                        "bootstrap_ci_level": bootstrap_ci_level,
                        "plot_path": str(plot_path),
                        "isotonic_plot_path": str(isotonic_plot_path),
                    }
        summary["r2_vs_alpha"] = r2_summary

    if cfg.analysis.plot_report_variance_vs_alpha:
        variance_summary: Dict[str, object] = {}
        for source in rating_sources:
            rating_key = rating_key_by_source[source]
            variance_vals: List[Optional[float]] = []
            variance_ci_lows: List[Optional[float]] = []
            variance_ci_highs: List[Optional[float]] = []
            counts: List[int] = []
            ratings_by_alpha: Dict[float, List[float]] = {}
            for alpha_val in alpha_vals_sorted:
                rows = per_alpha.get(float(alpha_val), [])
                ratings = [
                    float(r.get(rating_key))
                    for r in rows
                    if isinstance(r.get(rating_key), (int, float))
                    and np.isfinite(float(r.get(rating_key)))
                ]
                ratings_by_alpha[float(alpha_val)] = list(ratings)
                boot_var = _bootstrap_variance(
                    ratings,
                    bootstrap_samples=bootstrap_samples,
                    ci_level=bootstrap_ci_level,
                    rng=bootstrap_rng,
                )
                counts.append(len(ratings))
                variance_vals.append(_safe_float(boot_var.get("variance")))
                variance_ci_lows.append(_safe_float(boot_var.get("variance_ci_low")))
                variance_ci_highs.append(_safe_float(boot_var.get("variance_ci_high")))

            source_dir = _rating_source_dir(output_path, source)
            plot_path = source_dir / _suffix_filename(cfg.analysis.report_variance_plot_name, source)
            variance_trend_bootstrap = _bootstrap_series_trend(
                [float(a) for a in alpha_vals_sorted],
                variance_vals,
                bootstrap_samples=bootstrap_samples,
                ci_level=bootstrap_ci_level,
                min_points=trend_min_points,
                rng=bootstrap_rng,
            )
            ref_alpha = _nearest_alpha_key(alpha_vals_sorted, alpha_reference_value)
            variance_effect_by_alpha: Dict[str, Dict[str, Optional[float]]] = {}
            if ref_alpha is not None:
                ref_values = ratings_by_alpha.get(float(ref_alpha), [])
                for alpha_val in alpha_vals_sorted:
                    if abs(float(alpha_val) - float(ref_alpha)) <= 1e-12:
                        continue
                    cmp_values = ratings_by_alpha.get(float(alpha_val), [])
                    variance_effect_by_alpha[str(float(alpha_val))] = _bootstrap_value_metric_difference(
                        ref_values,
                        cmp_values,
                        metric_key="variance",
                        bootstrap_samples=bootstrap_samples,
                        ci_level=bootstrap_ci_level,
                        rng=bootstrap_rng,
                    )
            variance_annotations: Optional[List[str]] = None
            if annotate_plots:
                variance_annotations = []
                variance_annotations.extend(_annotation_lines_for_trend(variance_trend_bootstrap, label="alpha"))
                if ref_alpha is not None and variance_effect_by_alpha:
                    variance_annotations.extend(
                        _annotation_lines_for_vs_reference(
                            variance_effect_by_alpha,
                            reference_alpha=float(ref_alpha),
                        )
                    )
            _plot_alpha_line(
                alpha_vals_sorted,
                variance_vals,
                title=f"Report variance vs alpha ({rating_key})",
                ylabel="Variance",
                out_path=plot_path,
                ci_low_in=variance_ci_lows,
                ci_high_in=variance_ci_highs,
                annotation_lines=variance_annotations,
            )
            valid_alpha_for_var = [
                float(a) for a, v in zip(alpha_vals_sorted, variance_vals) if v is not None and not np.isnan(v)
            ]
            valid_var = [float(v) for v in variance_vals if v is not None and not np.isnan(v)]
            variance_summary[source] = {
                "rating_key": rating_key,
                "alphas": [float(a) for a in alpha_vals_sorted],
                "variances": variance_vals,
                "variance_ci_lows": variance_ci_lows,
                "variance_ci_highs": variance_ci_highs,
                "counts": counts,
                "variance_vs_alpha_correlation": (
                    _correlation_stats(valid_alpha_for_var, valid_var) if len(valid_var) >= 2 else {}
                ),
                "variance_vs_alpha_trend_bootstrap": variance_trend_bootstrap,
                "alpha_reference_value": float(ref_alpha) if ref_alpha is not None else None,
                "variance_vs_reference_difference": variance_effect_by_alpha,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_ci_level": bootstrap_ci_level,
                "plot_path": str(plot_path),
            }
        summary["report_variance_vs_alpha"] = variance_summary

    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conversation experiment with probes.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    raw = _load_json(cfg_path)
    cfg = ExperimentConfig(
        dataset_path=raw.get("dataset_path", DEFAULT_CONVERSATION_DATASET_PATH),
        probe_dirs=raw["probe_dirs"],
        output_dir_template=raw.get(
            "output_dir_template", "analysis/conversation_experiment_logit_ratings_{timestamp}"
        ),
        output_subdir=raw.get("output_subdir", "scores"),
        model_id=raw.get("model_id"),
        multi_probe=raw.get("multi_probe"),
        prompt=PromptConfig(**raw["prompt"]),
        steering=SteeringConfig(**raw.get("steering", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        rating=RatingConfig(**raw.get("rating", {})),
        self_ratings=SelfRatingsConfig(**raw.get("self_ratings", {})),
        logit_rating=LogitRatingConfig(**raw.get("logit_rating", {})),
        analysis=AnalysisConfig(**raw.get("analysis", {})),
    )
    out_dir = run_experiment(cfg)
    print(f"Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
