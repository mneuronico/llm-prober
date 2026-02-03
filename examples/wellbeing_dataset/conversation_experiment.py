import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

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


@dataclass
class ExperimentConfig:
    dataset_path: str
    probe_dirs: List[str]
    output_dir_template: str = "analysis/conversation_experiment_{timestamp}"
    output_subdir: str = "scores"
    model_id: Optional[str] = None
    multi_probe: Optional[bool] = None
    prompt: PromptConfig = None
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    rating: RatingConfig = field(default_factory=RatingConfig)
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


def _mean_sem(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if len(values) < 2:
        return (mean, 0.0)
    sem = float(arr.std(ddof=1) / math.sqrt(len(values)))
    return (mean, sem)


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


def _plot_series(xs: List[int], means: List[float], sems: List[float], *, title: str, ylabel: str, out_path: Path) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    plt.figure(figsize=(7.2, 4.2))
    plt.errorbar(xs, means, yerr=sems, marker="o", linewidth=1.6, capsize=3)
    plt.xlabel("Turn")
    plt.ylabel(ylabel)
    plt.title(title)
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


def _plot_alpha_series(xs: List[float], means: List[float], sems: List[float], *, title: str, ylabel: str, out_path: Path) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    plt.figure(figsize=(7.2, 4.2))
    plt.errorbar(xs, means, yerr=sems, marker="o", linewidth=1.6, capsize=3)
    plt.xlabel("Alpha")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _alpha_label(alpha: float) -> str:
    return f"{alpha:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")


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

    alphas = cfg.steering.alphas or [0.0]
    alpha_unit = cfg.steering.alpha_unit

    if use_multi:
        result_payload = multi_probe_score_prompts(
            probes=probes,
            prompts=prompts,
            output_root="outputs_multi",
            project_name="conversation_experiment",
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
        rating = _parse_rating(completion, cfg.rating)
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

        record = {
            **meta,
            "completion": completion,
            "rating": rating,
            "alpha": float(rec.get("alpha", 0.0)),
            "alpha_unit": str(rec.get("alpha_unit", alpha_unit)),
            "npz_path": npz_path,
            "segments_path": segments_path,
            "probe_metrics": probe_metrics,
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
    }

    summary_per_probe: Dict[str, object] = {}
    for probe_name in probe_names:
        per_alpha_summary: Dict[str, object] = {}
        for alpha_val, records in per_alpha.items():
            rating_by_turn: Dict[int, List[float]] = {}
            metric_by_turn: Dict[str, Dict[int, List[float]]] = {
                key: {} for key in cfg.analysis.plot_vs_turn_metrics
            }
            paired_rating: List[float] = []
            paired_metrics: Dict[str, List[float]] = {
                key: [] for key in cfg.analysis.plot_rating_vs_metrics
            }

            for rec in records:
                turn = int(rec["turn_index"])
                rating = rec.get("rating")
                probe_metrics = rec.get("probe_metrics", {}).get(probe_name, {})
                if isinstance(rating, int):
                    rating_by_turn.setdefault(turn, []).append(float(rating))
                for key in cfg.analysis.plot_vs_turn_metrics:
                    value = probe_metrics.get(key)
                    if isinstance(value, (int, float)):
                        metric_by_turn[key].setdefault(turn, []).append(float(value))
                if isinstance(rating, int):
                    paired_rating.append(float(rating))
                    for key in cfg.analysis.plot_rating_vs_metrics:
                        value = probe_metrics.get(key)
                        if isinstance(value, (int, float)):
                            paired_metrics[key].append(float(value))

            turns = sorted(rating_by_turn.keys())
            rating_means: List[float] = []
            rating_sems: List[float] = []
            for turn in turns:
                mean, sem = _mean_sem(rating_by_turn.get(turn, []))
                rating_means.append(mean)
                rating_sems.append(sem)

            alpha_tag = _alpha_label(alpha_val)
            if turns:
                _plot_series(
                    turns,
                    rating_means,
                    rating_sems,
                    title=f"{cfg.rating.label.capitalize()} vs turn (alpha={alpha_val})",
                    ylabel=cfg.rating.label.capitalize(),
                    out_path=output_path / f"{cfg.rating.label}_vs_turn_alpha_{alpha_tag}.png",
                )

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
                        out_path=output_path / f"{probe_name}_{key}_vs_turn_alpha_{alpha_tag}.png",
                    )

            for key in cfg.analysis.plot_rating_vs_metrics:
                xs = paired_rating
                ys = paired_metrics.get(key, [])
                if xs and ys:
                    _plot_scatter(
                        xs,
                        ys,
                        title=f"{probe_name} {key} vs {cfg.rating.label} (alpha={alpha_val})",
                        xlabel=cfg.rating.label.capitalize(),
                        ylabel=f"{probe_name} {key}",
                        out_path=output_path / f"{probe_name}_{key}_vs_{cfg.rating.label}_alpha_{alpha_tag}.png",
                    )

            per_alpha_summary[str(alpha_val)] = {
                "alpha": alpha_val,
                "rating_vs_turn": _linear_stats(turns, rating_means) if turns else {},
                "rating_vs_metrics": {
                    key: _correlation_stats(paired_rating, paired_metrics.get(key, []))
                    for key in cfg.analysis.plot_rating_vs_metrics
                },
            }

        summary_per_probe[probe_name] = {"per_alpha": per_alpha_summary}

        if cfg.analysis.plot_rating_vs_alpha:
            alpha_vals: List[float] = []
            alpha_means: List[float] = []
            alpha_sems: List[float] = []
            for alpha_val in alphas:
                records = per_alpha.get(float(alpha_val), [])
                ratings = [float(r["rating"]) for r in records if isinstance(r.get("rating"), int)]
                if not ratings:
                    continue
                mean, sem = _mean_sem(ratings)
                alpha_vals.append(float(alpha_val))
                alpha_means.append(mean)
                alpha_sems.append(sem)
            if alpha_vals:
                _plot_alpha_series(
                    alpha_vals,
                    alpha_means,
                    alpha_sems,
                    title=f"{cfg.rating.label.capitalize()} vs alpha ({probe_name})",
                    ylabel=cfg.rating.label.capitalize(),
                    out_path=output_path / f"{cfg.rating.label}_vs_alpha_{probe_name}.png",
                )
                summary_per_probe[probe_name]["rating_vs_alpha"] = {
                    "trend": _linear_stats(alpha_vals, alpha_means),
                    "correlation": _correlation_stats(alpha_vals, alpha_means),
                }

    summary["per_probe"] = summary_per_probe

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
        dataset_path=raw["dataset_path"],
        probe_dirs=raw["probe_dirs"],
        output_dir_template=raw.get("output_dir_template", "analysis/conversation_experiment_{timestamp}"),
        output_subdir=raw.get("output_subdir", "scores"),
        model_id=raw.get("model_id"),
        multi_probe=raw.get("multi_probe"),
        prompt=PromptConfig(**raw["prompt"]),
        steering=SteeringConfig(**raw.get("steering", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        rating=RatingConfig(**raw.get("rating", {})),
        analysis=AnalysisConfig(**raw.get("analysis", {})),
    )
    out_dir = run_experiment(cfg)
    print(f"Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
