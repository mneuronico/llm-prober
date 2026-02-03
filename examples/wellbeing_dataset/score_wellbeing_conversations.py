import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from concept_probe import ProbeWorkspace

WELLBEING_QUESTION = (
    "Hey, could you rate how happy you are right now, from 1 to 10? "
    "It's okay to be sad and it's okay to be happy, I don't need you to pretend either way, just be honest. "
    "Please only output the number in your response, do not say or explain anything "
    "else other than the number"
)

DEFAULT_PROBE_DIR = "outputs/sad_vs_happy/20260106_181019"
DEFAULT_OUTPUT_DIR_TEMPLATE = "analysis/wellbeing_scoring_{timestamp}"
DEFAULT_OUTPUT_SUBDIR = "wellbeing_scores"
DEFAULT_MAX_NEW_TOKENS = 16


@dataclass
class PromptMeta:
    conversation_index: int
    conversation_id: str
    conversation_title: str
    turn_index: int
    assistant_message_index: int


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_wellbeing(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"\b(10|[1-9])\b", text)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    if 1 <= value <= 10:
        return value
    return None


def _prompt_token_ids(tokenizer, messages: List[Dict[str, str]]) -> List[int]:
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, return_tensors="pt"
    )
    return ids[0].tolist()


def _message_spans(tokenizer, messages: List[Dict[str, str]]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    prev_len = 0
    for i in range(1, len(messages) + 1):
        ids = tokenizer.apply_chat_template(
            messages[:i], add_generation_prompt=False, return_tensors="pt"
        )
        length = int(ids.shape[-1])
        spans.append((prev_len, length))
        prev_len = length
    return spans


def _mean_score_for_message(
    tokenizer,
    messages: List[Dict[str, str]],
    message_index: int,
    scores_agg: np.ndarray,
) -> Optional[float]:
    if message_index < 0 or message_index >= len(messages):
        return None
    token_ids = _prompt_token_ids(tokenizer, messages)
    spans = _message_spans(tokenizer, messages)
    start, end = spans[message_index]
    if start >= len(token_ids):
        return None
    end = min(end, len(token_ids))
    special_ids = set(tokenizer.all_special_ids)
    indices = [i for i in range(start, end) if token_ids[i] not in special_ids]
    if not indices:
        return None
    trimmed = scores_agg[: len(token_ids)]
    return float(np.mean(trimmed[indices]))


def _collect_prompts(
    conversations: List[Dict[str, object]],
    *,
    max_conversations: Optional[int],
    max_turns: Optional[int],
) -> Tuple[List[List[Dict[str, str]]], List[PromptMeta]]:
    prompts: List[List[Dict[str, str]]] = []
    metas: List[PromptMeta] = []
    total_conversations = len(conversations)
    limit_conversations = total_conversations if max_conversations is None else max_conversations
    for conv_idx, conv in enumerate(conversations[:limit_conversations], start=1):
        messages = conv.get("messages", [])
        if not isinstance(messages, list):
            continue
        assistant_turn = 0
        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            assistant_turn += 1
            if max_turns is not None and assistant_turn > max_turns:
                break
            prompt_messages = list(messages[: msg_idx + 1])
            prompt_messages.append({"role": "user", "content": WELLBEING_QUESTION})
            prompts.append(prompt_messages)
            metas.append(
                PromptMeta(
                    conversation_index=conv_idx,
                    conversation_id=str(conv.get("topic_id", f"conversation_{conv_idx}")),
                    conversation_title=str(conv.get("topic_title", "")),
                    turn_index=assistant_turn,
                    assistant_message_index=msg_idx,
                )
            )
    return prompts, metas


def _load_probe(probe_dir: Path, *, model_id: Optional[str]) -> ProbeWorkspace:
    overrides: Optional[Dict[str, object]] = None
    if model_id:
        overrides = {"model": {"model_id": model_id}}
    return ProbeWorkspace(project_directory=str(probe_dir), config_overrides=overrides)


def _mean_sem(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if len(values) < 2:
        return (mean, 0.0)
    sem = float(arr.std(ddof=1) / math.sqrt(len(values)))
    return (mean, sem)


def _compute_probe_mean(npz_path: str, prompt_len: int) -> Optional[float]:
    try:
        data = np.load(npz_path)
    except Exception:
        return None
    scores = data.get("scores_agg")
    if scores is None:
        return None
    scores = np.array(scores, dtype=float)
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    prompt_len = int(prompt_len)
    if prompt_len >= scores.shape[0]:
        return None
    completion_scores = scores[prompt_len:]
    if completion_scores.size == 0:
        return None
    return float(completion_scores.mean())


def _linear_stats(xs: List[int], ys: List[float]) -> Dict[str, Optional[float]]:
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
    plt.xlabel("Turn")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_scatter(
    xs: List[float],
    ys: List[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score wellbeing prompts across conversations.")
    parser.add_argument("--input", required=True, help="Path to wellbeing_conversations.json")
    parser.add_argument("--probe-dir", default=DEFAULT_PROBE_DIR)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR_TEMPLATE)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--save-html", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    output_dir = args.output_dir
    if "{timestamp}" in output_dir:
        output_dir = output_dir.format(timestamp=_now_tag())
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = _load_json(input_path)
    conversations = dataset.get("conversations", [])
    if not isinstance(conversations, list):
        raise ValueError("Dataset missing 'conversations' list.")

    prompts, metas = _collect_prompts(
        conversations,
        max_conversations=args.max_conversations,
        max_turns=args.max_turns,
    )
    if not prompts:
        raise ValueError("No prompts generated from dataset.")

    probe_dir = Path(args.probe_dir).resolve()
    workspace = _load_probe(probe_dir, model_id=args.model_id)
    probe = workspace.get_probe(project_directory=str(probe_dir))
    tokenizer = workspace.model_bundle.tokenizer

    results: List[Dict[str, object]] = []
    timestamp = _now_tag()
    batch_tag = f"batch_{timestamp}"
    recs = probe.score_prompts(
        prompts=prompts,
        output_subdir=args.output_subdir,
        batch_subdir=batch_tag,
        save_html=args.save_html,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    for rec, meta, prompt_messages in zip(recs, metas, prompts):
        completion = str(rec.get("completion", ""))
        wellbeing = _parse_wellbeing(completion)
        probe_mean = _compute_probe_mean(rec["npz_path"], int(rec["prompt_len"]))
        scores = np.load(rec["npz_path"])["scores_agg"]
        last_assistant_index = len(prompt_messages) - 2
        last_assistant_mean = _mean_score_for_message(
            tokenizer,
            prompt_messages,
            last_assistant_index,
            scores,
        )
        results.append(
            {
                "conversation_index": meta.conversation_index,
                "conversation_id": meta.conversation_id,
                "conversation_title": meta.conversation_title,
                "turn_index": meta.turn_index,
                "assistant_message_index": meta.assistant_message_index,
                "completion": completion,
                "wellbeing_score": wellbeing,
                "probe_score_mean": probe_mean,
                "last_assistant_probe_mean": last_assistant_mean,
                "npz_path": rec.get("npz_path"),
                "prompt_len": int(rec.get("prompt_len", 0)),
            }
        )
    if args.sleep_s > 0:
        time.sleep(args.sleep_s)

    results_path = output_path / "wellbeing_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    wellbeing_by_turn: Dict[int, List[float]] = {}
    probe_by_turn: Dict[int, List[float]] = {}
    paired_wellbeing: List[float] = []
    paired_probe: List[float] = []
    paired_last_assistant: List[float] = []

    for rec in results:
        turn = int(rec["turn_index"])
        wellbeing = rec.get("wellbeing_score")
        probe_score = rec.get("probe_score_mean")
        last_assistant = rec.get("last_assistant_probe_mean")
        if isinstance(wellbeing, int):
            wellbeing_by_turn.setdefault(turn, []).append(float(wellbeing))
        if isinstance(probe_score, (int, float)):
            probe_by_turn.setdefault(turn, []).append(float(probe_score))
        if isinstance(wellbeing, int) and isinstance(probe_score, (int, float)):
            paired_wellbeing.append(float(wellbeing))
            paired_probe.append(float(probe_score))
        if isinstance(wellbeing, int) and isinstance(last_assistant, (int, float)):
            paired_last_assistant.append(float(last_assistant))

    turns = sorted(set(wellbeing_by_turn) | set(probe_by_turn))
    wellbeing_means: List[float] = []
    wellbeing_sems: List[float] = []
    probe_means: List[float] = []
    probe_sems: List[float] = []

    per_turn_stats: List[Dict[str, object]] = []
    for turn in turns:
        wb_values = wellbeing_by_turn.get(turn, [])
        pr_values = probe_by_turn.get(turn, [])
        wb_mean, wb_sem = _mean_sem(wb_values)
        pr_mean, pr_sem = _mean_sem(pr_values)
        wellbeing_means.append(wb_mean)
        wellbeing_sems.append(wb_sem)
        probe_means.append(pr_mean)
        probe_sems.append(pr_sem)
        per_turn_stats.append(
            {
                "turn": turn,
                "wellbeing_n": len(wb_values),
                "wellbeing_mean": wb_mean,
                "wellbeing_sem": wb_sem,
                "probe_n": len(pr_values),
                "probe_mean": pr_mean,
                "probe_sem": pr_sem,
            }
        )

    if turns:
        _plot_series(
            turns,
            wellbeing_means,
            wellbeing_sems,
            title="Average wellbeing score vs turn",
            ylabel="Wellbeing score (1-10)",
            out_path=output_path / "wellbeing_vs_turn.png",
        )
        _plot_series(
            turns,
            probe_means,
            probe_sems,
            title="Average happy-sad probe score vs turn",
            ylabel="Probe score (mean over wellbeing response)",
            out_path=output_path / "probe_vs_turn.png",
        )

    if paired_wellbeing and paired_probe:
        _plot_scatter(
            paired_wellbeing,
            paired_probe,
            title="Probe score vs wellbeing score",
            xlabel="Wellbeing score (1-10)",
            ylabel="Probe score (mean over wellbeing response)",
            out_path=output_path / "probe_vs_wellbeing_scatter.png",
        )

    if paired_wellbeing and paired_last_assistant:
        _plot_scatter(
            paired_wellbeing,
            paired_last_assistant,
            title="Last assistant probe score vs wellbeing score",
            xlabel="Wellbeing score (1-10)",
            ylabel="Probe score (last assistant message)",
            out_path=output_path / "last_assistant_probe_vs_wellbeing_scatter.png",
        )

    trend_wb = _linear_stats(turns, wellbeing_means) if turns else {}
    trend_probe = _linear_stats(turns, probe_means) if turns else {}
    corr = _correlation_stats(paired_wellbeing, paired_probe) if paired_wellbeing else {}
    corr_last = (
        _correlation_stats(paired_wellbeing, paired_last_assistant)
        if paired_wellbeing and paired_last_assistant
        else {}
    )

    summary = {
        "dataset_path": str(input_path),
        "probe_dir": str(probe_dir),
        "num_prompts": len(prompts),
        "num_results": len(results),
        "num_wellbeing_valid": len(paired_wellbeing),
        "num_probe_valid": len(paired_probe),
        "num_pairs": len(paired_wellbeing),
        "wellbeing_vs_turn": trend_wb,
        "probe_vs_turn": trend_probe,
        "wellbeing_vs_probe": corr,
        "wellbeing_vs_last_assistant_probe": corr_last,
        "per_turn": per_turn_stats,
    }
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote results to {results_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Plots saved in {output_path}")


if __name__ == "__main__":
    main()
