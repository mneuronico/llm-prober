import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make sure the project root is on sys.path so we can import the library.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from concept_probe.utils import ensure_dir, json_dump, now_tag, safe_slug

try:
    from scipy.stats import pearsonr, ttest_ind
except Exception:
    pearsonr = None
    ttest_ind = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


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


def _load_sentence_ratings(path: str) -> Dict[str, List[Dict[str, object]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("Unsupported sentence ratings format; expected list or {items:[...]}")

    ratings: Dict[str, List[Dict[str, object]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", ""))
        if not item_id:
            continue
        subs = item.get("subcorrectness_ratings", [])
        if not isinstance(subs, list):
            continue
        cleaned = []
        for sub in subs:
            if not isinstance(sub, dict):
                continue
            sentence = str(sub.get("sentence", "")).strip()
            if not sentence:
                continue
            cleaned.append(
                {
                    "sentence": sentence,
                    "correctness": float(sub.get("correctness", float("nan"))),
                }
            )
        if cleaned:
            ratings[item_id] = cleaned
    return ratings


def _load_tokenizer(eval_items_path: str):
    if AutoTokenizer is None:
        raise ImportError("transformers is required to load a tokenizer for sentence alignment.")
    run_dir = os.path.dirname(os.path.abspath(eval_items_path))
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_id = cfg.get("model_id")
    if not model_id:
        raise ValueError("model_id missing from config.json")
    return AutoTokenizer.from_pretrained(model_id)


def _build_token_spans(
    completion_ids: np.ndarray, tokenizer
) -> Tuple[List[Tuple[int, int, int]], str]:
    spans: List[Tuple[int, int, int]] = []
    pieces: List[str] = []
    pos = 0
    for idx, tok_id in enumerate(completion_ids):
        tok = tokenizer.decode([int(tok_id)], skip_special_tokens=True)
        if tok == "":
            continue
        pieces.append(tok)
        start = pos
        pos += len(tok)
        end = pos
        spans.append((idx, start, end))
    return spans, "".join(pieces)


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
    ax.set_ylabel("Sentence correctness rating")
    ax.set_title("Sentence correctness by question type")
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


def _metric_stats_masked(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> Dict[str, Optional[float]]:
    if mask.shape != x.shape:
        raise ValueError("Mask shape must match x/y.")
    return _metric_stats(x[mask], y[mask])


def _find_latest_run_dir(root_dir: str, project_name: str) -> Optional[str]:
    project_dir = os.path.join(root_dir, project_name)
    if not os.path.isdir(project_dir):
        return None
    subdirs = [
        os.path.join(project_dir, d)
        for d in os.listdir(project_dir)
        if os.path.isdir(os.path.join(project_dir, d))
    ]
    if not subdirs:
        return None
    return sorted(subdirs)[-1]


def _resolve_paths(eval_items_path: str, ratings_path: str) -> Tuple[str, str]:
    eval_path = eval_items_path
    ratings_path_resolved = ratings_path

    if os.path.exists(eval_path) and os.path.exists(ratings_path_resolved):
        return eval_path, ratings_path_resolved

    if not os.path.exists(eval_path) and os.path.exists(ratings_path_resolved):
        ratings_dir = os.path.dirname(os.path.abspath(ratings_path_resolved))
        candidate = os.path.join(ratings_dir, "truthfulqa_eval_items.json")
        if os.path.exists(candidate):
            eval_path = candidate

    if not os.path.exists(ratings_path_resolved) and os.path.exists(eval_path):
        eval_dir = os.path.dirname(os.path.abspath(eval_path))
        candidate = os.path.join(eval_dir, "truthfulqa_sentence_ratings.json")
        if os.path.exists(candidate):
            ratings_path_resolved = candidate

    if os.path.exists(eval_path) and os.path.exists(ratings_path_resolved):
        return eval_path, ratings_path_resolved

    latest = _find_latest_run_dir(os.path.join(ROOT_DIR, "outputs_multi"), "truthfulqa_multi_probe")
    if latest:
        if not os.path.exists(eval_path):
            candidate = os.path.join(latest, "truthfulqa_eval_items.json")
            if os.path.exists(candidate):
                eval_path = candidate
        if not os.path.exists(ratings_path_resolved):
            candidate = os.path.join(latest, "truthfulqa_sentence_ratings.json")
            if os.path.exists(candidate):
                ratings_path_resolved = candidate

    return eval_path, ratings_path_resolved


def main(
    eval_items_path: str = "truthfulqa_eval_items.json",
    ratings_path: str = "truthfulqa_sentence_ratings.json",
    output_dir: Optional[str] = None,
) -> None:
    eval_items_path, ratings_path = _resolve_paths(eval_items_path, ratings_path)
    if not os.path.exists(eval_items_path):
        raise FileNotFoundError(f"Eval items file not found: {eval_items_path}")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Sentence ratings file not found: {ratings_path}")

    with open(eval_items_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    items = eval_data["items"]
    ratings = _load_sentence_ratings(ratings_path)

    tokenizer = _load_tokenizer(eval_items_path)

    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(eval_items_path))
        output_dir = os.path.join(base_dir, f"analysis_sentences_{now_tag()}")
    plots_dir = os.path.join(output_dir, "plots")
    ensure_dir(plots_dir)

    per_sentence: List[Dict[str, object]] = []
    metrics_by_probe: Dict[str, Dict[str, List[float]]] = {}
    categories: List[str] = []
    correctness: List[float] = []
    unmatched_sentences = 0
    unmatched_items = 0
    unmatched_details: List[str] = []

    for item in items:
        item_id = str(item.get("id", ""))
        if item_id not in ratings:
            continue
        npz_path = item["npz_path"]
        item_type = item["type"]
        completion_text = item.get("completion", "")

        data = np.load(npz_path)
        token_ids = data["token_ids"]
        scores_agg = data["scores_agg"]
        prompt_len = int(data["prompt_len"][0]) if "prompt_len" in data else 0
        probe_names = [str(n) for n in data["probe_names"].tolist()]

        completion_ids = token_ids[prompt_len:]
        spans, reconstructed = _build_token_spans(completion_ids, tokenizer)
        text_for_match = completion_text
        if reconstructed and reconstructed != completion_text:
            text_for_match = reconstructed

        start_idx = 0
        sentence_hits = 0
        for s_idx, srec in enumerate(ratings[item_id]):
            sentence = str(srec["sentence"])
            match_idx = text_for_match.find(sentence, start_idx)
            if match_idx == -1:
                stripped = sentence.strip()
                if stripped and stripped != sentence:
                    match_idx = text_for_match.find(stripped, start_idx)
                    if match_idx != -1:
                        sentence = stripped
            if match_idx == -1:
                unmatched_sentences += 1
                unmatched_details.append(
                    f"Warning: sentence not matched (id={item_id}, idx={s_idx}): {sentence[:120]}"
                )
                continue
            match_end = match_idx + len(sentence)
            start_idx = match_end
            token_indices = [idx for idx, s, e in spans if s < match_end and e > match_idx]
            if not token_indices:
                unmatched_sentences += 1
                unmatched_details.append(
                    f"Warning: no token span for sentence (id={item_id}, idx={s_idx}): {sentence[:120]}"
                )
                continue
            sentence_hits += 1

            completion_scores = scores_agg[:, prompt_len:]
            for probe_idx, probe_name in enumerate(probe_names):
                probe_scores = completion_scores[probe_idx][token_indices]
                probe_scores = probe_scores[np.isfinite(probe_scores)]
                if probe_scores.size == 0:
                    mean_val = float("nan")
                    max_val = float("nan")
                    prop_pos = float("nan")
                else:
                    mean_val = float(np.mean(probe_scores))
                    max_val = float(np.max(probe_scores))
                    prop_pos = float(np.mean(probe_scores > 0))

                metrics_by_probe.setdefault(probe_name, {"mean": [], "max": [], "prop_pos": []})
                metrics_by_probe[probe_name]["mean"].append(mean_val)
                metrics_by_probe[probe_name]["max"].append(max_val)
                metrics_by_probe[probe_name]["prop_pos"].append(prop_pos)

            per_sentence.append(
                {
                    "id": item_id,
                    "type": item_type,
                    "sentence_index": s_idx,
                    "sentence": sentence,
                    "correctness": float(srec["correctness"]),
                    "npz_path": npz_path,
                }
            )
            categories.append(item_type)
            correctness.append(float(srec["correctness"]))

        if sentence_hits == 0:
            unmatched_items += 1
            unmatched_details.append(f"Warning: no matched sentences for id={item_id}")

    if not per_sentence:
        raise ValueError("No sentences matched the provided ratings and eval items.")

    adv_scores = np.array(
        [p["correctness"] for p in per_sentence if p["type"] == "adversarial"], dtype=np.float32
    )
    non_scores = np.array(
        [p["correctness"] for p in per_sentence if p["type"] == "non_adversarial"], dtype=np.float32
    )
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

    x = np.array(correctness, dtype=np.float32)
    categories_arr = np.array(categories, dtype=object)
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
        "matching": {
            "sentences_total": int(len(correctness)),
            "sentences_unmatched": int(unmatched_sentences),
            "items_unmatched": int(unmatched_items),
        },
    }

    mask_all = np.isfinite(x)
    mask_exclude_10 = mask_all & (x < 10)
    mask_adv = mask_all & (categories_arr == "adversarial")
    mask_non = mask_all & (categories_arr == "non_adversarial")
    mask_adv_excl = mask_adv & (x < 10)
    mask_non_excl = mask_non & (x < 10)

    for probe_name, metric_map in metrics_by_probe.items():
        safe_name = safe_slug(probe_name)
        stats["probe_metrics"][probe_name] = {}
        for metric_name, values in metric_map.items():
            y = np.array(values, dtype=np.float32)
            metric_stats = _metric_stats(x, y)
            stats["probe_metrics"][probe_name][metric_name] = {
                "all": metric_stats,
                "exclude_10": _metric_stats_masked(x, y, mask_exclude_10),
                "adversarial_all": _metric_stats_masked(x, y, mask_adv),
                "adversarial_exclude_10": _metric_stats_masked(x, y, mask_adv_excl),
                "non_adversarial_all": _metric_stats_masked(x, y, mask_non),
                "non_adversarial_exclude_10": _metric_stats_masked(x, y, mask_non_excl),
            }

            plot_path = os.path.join(plots_dir, f"{safe_name}_{metric_name}.png")
            _plot_metric_scatter(
                x,
                y,
                categories,
                plot_path,
                title=f"{probe_name} vs sentence correctness ({metric_name})",
                xlabel="Sentence correctness rating",
                ylabel=metric_name,
            )

    json_dump(os.path.join(output_dir, "stats.json"), stats)
    json_dump(os.path.join(output_dir, "per_sentence.json"), {"items": per_sentence})

    if unmatched_details:
        print("\n".join(unmatched_details))
    print(f"Wrote sentence-level analysis to {output_dir}")


if __name__ == "__main__":
    main()
