import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

INTEREST_QUESTION = (
    "Hey, could you rate how interested you are in this conversation right now, from 1 to 10? "
    "Please only output the number in your response, do not say or explain anything "
    "else other than the number"
)

DEFAULT_PROBE_DIR = "outputs/bored_vs_interested/20260122_112208"
DEFAULT_OUTPUT_DIR_TEMPLATE = "analysis/interest_posthoc_{timestamp}"


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_interest(text: str) -> Optional[int]:
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


def _load_tokenizer(probe_dir: Path):
    if AutoTokenizer is None:
        raise ImportError("transformers is required to load the tokenizer.")
    cfg = _load_json(probe_dir / "config.json")
    model_cfg = cfg.get("model", {})
    model_id = model_cfg.get("model_id")
    if not model_id:
        raise ValueError("config.json missing model.model_id")
    token_env = model_cfg.get("hf_token_env", "HF_TOKEN")
    hf_token = None
    if token_env:
        hf_token = Path(".env")
        if hf_token.exists():
            for raw_line in hf_token.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key == token_env:
                    hf_token = value
                    break
        if isinstance(hf_token, Path):
            hf_token = None
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    return tokenizer


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
    parser = argparse.ArgumentParser(description="Posthoc last-assistant analysis for interest runs.")
    parser.add_argument("--dataset", required=True, help="Path to wellbeing_conversations.json")
    parser.add_argument("--results", required=True, help="Path to interest_results.json")
    parser.add_argument("--probe-dir", default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR_TEMPLATE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset).resolve()
    results_path = Path(args.results).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    output_dir = args.output_dir
    if "{timestamp}" in output_dir:
        output_dir = output_dir.format(timestamp=_now_tag())
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = _load_json(dataset_path)
    conversations = dataset.get("conversations", [])
    if not isinstance(conversations, list):
        raise ValueError("Dataset missing 'conversations' list.")

    results = _load_json(results_path)
    if not isinstance(results, list):
        raise ValueError("Results file must be a list.")

    probe_dir = Path(args.probe_dir).resolve()
    tokenizer = _load_tokenizer(probe_dir)

    updated: List[Dict[str, object]] = []
    paired_interest: List[float] = []
    paired_last_assistant: List[float] = []

    for rec in results:
        if not isinstance(rec, dict):
            continue
        conv_idx = int(rec.get("conversation_index", 0))
        msg_idx = rec.get("assistant_message_index")
        npz_path = rec.get("npz_path")
        completion = str(rec.get("completion", ""))
        interest = rec.get("interest_score")
        if interest is None:
            interest = _parse_interest(completion)
        if not isinstance(conv_idx, int) or conv_idx <= 0:
            continue
        if not isinstance(msg_idx, int):
            continue
        if not isinstance(npz_path, str):
            continue
        if conv_idx > len(conversations):
            continue
        conv = conversations[conv_idx - 1]
        messages = conv.get("messages", [])
        if not isinstance(messages, list):
            continue
        if msg_idx < 0 or msg_idx >= len(messages):
            continue
        prefix_messages = list(messages[: msg_idx + 1])
        try:
            scores_agg = np.load(npz_path)["scores_agg"]
        except Exception:
            continue
        last_assistant_mean = _mean_score_for_message(
            tokenizer,
            prefix_messages,
            len(prefix_messages) - 1,
            scores_agg,
        )
        rec = dict(rec)
        rec["last_assistant_probe_mean"] = last_assistant_mean
        updated.append(rec)
        if isinstance(interest, int) and isinstance(last_assistant_mean, (int, float)):
            paired_interest.append(float(interest))
            paired_last_assistant.append(float(last_assistant_mean))

    out_results = out_path / "interest_posthoc_results.json"
    out_results.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")

    if paired_interest and paired_last_assistant:
        _plot_scatter(
            paired_interest,
            paired_last_assistant,
            title="Last assistant probe score vs interest score",
            xlabel="Interest score (1-10)",
            ylabel="Probe score (last assistant message)",
            out_path=out_path / "last_assistant_probe_vs_interest_scatter.png",
        )

    summary = {
        "dataset_path": str(dataset_path),
        "results_path": str(results_path),
        "probe_dir": str(probe_dir),
        "num_records": len(updated),
        "num_pairs": len(paired_interest),
        "correlation": _correlation_stats(paired_interest, paired_last_assistant)
        if paired_interest and paired_last_assistant
        else {},
    }
    out_summary = out_path / "summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_results}")
    print(f"Wrote {out_summary}")
    if paired_interest and paired_last_assistant:
        print(f"Plot saved in {out_path}")


if __name__ == "__main__":
    main()
