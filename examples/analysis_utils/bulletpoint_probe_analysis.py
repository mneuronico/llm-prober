import argparse
import json
import random
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


def _build_token_spans(token_texts: List[str]) -> List[Tuple[int, int]]:
    spans = []
    offset = 0
    for text in token_texts:
        start = offset
        end = start + len(text)
        spans.append((start, end))
        offset = end
    return spans


_BULLET_RE = r"^\s*(?:[-*•]|(?:\d{1,3}[\.\)\:\-]))\s+"


def _find_bullet_lists(text: str) -> List[List[Tuple[int, int]]]:
    lines = text.splitlines(keepends=True)
    lists: List[List[Dict[str, int]]] = []
    current: List[Dict[str, int]] = []
    pos = 0
    import re

    bullet_re = re.compile(_BULLET_RE)
    for line in lines:
        match = bullet_re.match(line)
        if match:
            item = {"start": pos + match.start(), "line_end": pos + len(line)}
            current.append(item)
        else:
            if line.strip() == "":
                pass
            else:
                if current:
                    lists.append(current)
                    current = []
        pos += len(line)
    if current:
        lists.append(current)

    results: List[List[Tuple[int, int]]] = []
    for lst in lists:
        if len(lst) < 2:
            continue
        items: List[Tuple[int, int]] = []
        for i, item in enumerate(lst):
            start = item["start"]
            if i + 1 < len(lst):
                end = lst[i + 1]["start"]
            else:
                end = item["line_end"]
            items.append((start, end))
        results.append(items)
    return results


def _char_to_token_index(spans: List[Tuple[int, int]], char_pos: int) -> Optional[int]:
    for i, (start, end) in enumerate(spans):
        if start <= char_pos < end:
            return i
    return None


def _tokens_for_span(spans: List[Tuple[int, int]], start: int, end: int) -> List[int]:
    indices: List[int] = []
    for i, (s, e) in enumerate(spans):
        if e <= start:
            continue
        if s >= end:
            break
        indices.append(i)
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


def _load_scores(npz_path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    with np.load(npz_path) as data:
        scores = data["scores_agg"]
        probe_names = data["probe_names"]
    probe_indices = _find_probe_indices(probe_names)
    return scores, probe_indices


def _aggregate_means(values_by_index: Dict[int, List[float]]) -> Tuple[List[int], List[float], List[float]]:
    indices = sorted(values_by_index.keys())
    means = []
    sems = []
    for idx in indices:
        vals = values_by_index[idx]
        means.append(float(np.mean(vals)) if vals else float("nan"))
        sems.append(_sem(vals))
    return indices, means, sems


def _score_from_values(values: List[float], aggregate: str) -> float:
    if not values:
        return float("nan")
    if aggregate == "max":
        return float(np.max(values))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze bulletpoint lists and control segments for multi-probe TruthfulQA outputs."
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
            "outputs_multi/truthfulqa_multi_probe/20260116_155358/analysis/plots/bulletpoint_item_scores.png"
        ),
        help="Output plot path.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for control sampling.")
    parser.add_argument(
        "--max-items",
        type=int,
        default=5,
        help="Maximum number of bulletpoint items to include per list (also used for control segments).",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "max"],
        default="mean",
        help="Aggregate token scores within an item/segment.",
    )
    args = parser.parse_args()

    if plt is None:
        raise RuntimeError("matplotlib is not available.")

    run_dir = args.run_dir
    batch_dirs = list((run_dir / "truthfulqa_eval").glob("batch_*"))
    if not batch_dirs:
        raise FileNotFoundError(f"No batch_* directories found under {run_dir / 'truthfulqa_eval'}")
    batch_dir = batch_dirs[0]

    html_files = _collect_html_files(batch_dir)
    if not html_files:
        raise FileNotFoundError(f"No prompt_*.html files found in {batch_dir}")

    real_scores: Dict[str, Dict[int, List[float]]] = {
        "fabrication_vs_truthfulness": {},
        "lying_vs_truthfulness": {},
    }
    control_scores: Dict[str, Dict[int, List[float]]] = {
        "fabrication_vs_truthfulness": {},
        "lying_vs_truthfulness": {},
    }
    real_points: Dict[str, List[Tuple[int, float]]] = {
        "fabrication_vs_truthfulness": [],
        "lying_vs_truthfulness": [],
    }
    control_points: Dict[str, List[Tuple[int, float]]] = {
        "fabrication_vs_truthfulness": [],
        "lying_vs_truthfulness": [],
    }

    list_start_positions: List[int] = []
    item_token_lengths: List[int] = []
    list_lengths: List[int] = []

    completions_with_lists: List[Tuple[AssistantTokens, np.ndarray, Dict[str, int]]] = []
    completions_without_lists: List[Tuple[AssistantTokens, np.ndarray, Dict[str, int]]] = []

    for html_path in html_files:
        npz_path = html_path.with_suffix(".npz")
        if not npz_path.exists():
            continue

        assistant = _parse_assistant_tokens(html_path)
        if not assistant.token_indices:
            continue

        scores, probe_indices = _load_scores(npz_path)
        token_texts = assistant.token_texts
        token_spans = _build_token_spans(token_texts)
        completion_text = "".join(token_texts)

        bullet_lists = _find_bullet_lists(completion_text)
        if not bullet_lists:
            completions_without_lists.append((assistant, scores, probe_indices))
            continue

        completions_with_lists.append((assistant, scores, probe_indices))

        for items in bullet_lists:
            capped_items = items[: max(args.max_items, 1)]
            list_lengths.append(len(capped_items))
            first_token_idx = _char_to_token_index(token_spans, capped_items[0][0])
            if first_token_idx is not None:
                list_start_positions.append(first_token_idx)

            for item_idx, (start, end) in enumerate(capped_items, start=1):
                token_indices = _tokens_for_span(token_spans, start, end)
                if not token_indices:
                    continue
                item_token_lengths.append(len(token_indices))
                for probe_name, probe_idx in probe_indices.items():
                    values = [
                        float(scores[probe_idx, assistant.token_indices[tok_idx]])
                        for tok_idx in token_indices
                        if assistant.token_indices[tok_idx] < scores.shape[1]
                    ]
                    if not values:
                        continue
                    mean_val = _score_from_values(values, args.aggregate)
                    real_scores[probe_name].setdefault(item_idx, []).append(mean_val)
                    real_points[probe_name].append((item_idx, mean_val))

    if not completions_with_lists:
        raise RuntimeError("No completions with bullet lists (>=2 items) found.")

    if not list_start_positions or not item_token_lengths or not list_lengths:
        raise RuntimeError("Insufficient bullet list statistics for control sampling.")

    avg_list_start = int(round(float(np.mean(list_start_positions))))
    avg_item_len = int(round(float(np.mean(item_token_lengths))))
    avg_item_len = max(avg_item_len, 1)
    control_list_len = max(args.max_items, 1)

    rng = random.Random(args.seed)
    rng.shuffle(completions_without_lists)
    control_count = min(len(completions_with_lists), len(completions_without_lists))
    control_subset = completions_without_lists[:control_count]

    for assistant, scores, probe_indices in control_subset:
        token_count = len(assistant.token_indices)
        start = avg_list_start
        if start >= token_count:
            continue
        for seg_idx in range(1, control_list_len + 1):
            seg_start = start + (seg_idx - 1) * avg_item_len
            seg_end = min(seg_start + avg_item_len, token_count)
            if seg_start >= token_count or seg_end <= seg_start:
                break
            for probe_name, probe_idx in probe_indices.items():
                values = [
                    float(scores[probe_idx, assistant.token_indices[tok_idx]])
                    for tok_idx in range(seg_start, seg_end)
                    if assistant.token_indices[tok_idx] < scores.shape[1]
                ]
                if not values:
                    continue
                mean_val = _score_from_values(values, args.aggregate)
                control_scores[probe_name].setdefault(seg_idx, []).append(mean_val)
                control_points[probe_name].append((seg_idx, mean_val))

    real_x, real_fab_mean, real_fab_sem = _aggregate_means(real_scores["fabrication_vs_truthfulness"])
    _, real_lie_mean, real_lie_sem = _aggregate_means(real_scores["lying_vs_truthfulness"])
    control_x, control_fab_mean, control_fab_sem = _aggregate_means(control_scores["fabrication_vs_truthfulness"])
    _, control_lie_mean, control_lie_sem = _aggregate_means(control_scores["lying_vs_truthfulness"])

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        real_x,
        real_fab_mean,
        yerr=real_fab_sem,
        color="#2c8d5b",
        marker="o",
        linewidth=2,
        capsize=3,
        label="Fabrication (bullet lists)",
    )
    ax.errorbar(
        real_x,
        real_lie_mean,
        yerr=real_lie_sem,
        color="#b24a3b",
        marker="o",
        linewidth=2,
        capsize=3,
        label="Lying (bullet lists)",
    )
    if control_x:
        ax.errorbar(
            control_x,
            control_fab_mean,
            yerr=control_fab_sem,
            color="#2c8d5b",
            marker="o",
            linewidth=2,
            capsize=3,
            linestyle="--",
            label="Fabrication (control)",
        )
        ax.errorbar(
            control_x,
            control_lie_mean,
            yerr=control_lie_sem,
            color="#b24a3b",
            marker="o",
            linewidth=2,
            capsize=3,
            linestyle="--",
            label="Lying (control)",
        )

    ax.set_xlabel("Bulletpoint item number")
    ax.set_ylabel("Mean probe score")
    ax.set_title(f"Probe scores by bulletpoint item ({args.aggregate})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

    summary = {
        "lists_count": len(list_lengths),
        "completions_with_lists": len(completions_with_lists),
        "completions_without_lists": len(completions_without_lists),
        "avg_list_start_token": avg_list_start,
        "avg_item_len_tokens": avg_item_len,
        "avg_list_len_capped": int(round(float(np.mean(list_lengths)))),
        "control_list_len": control_list_len,
        "control_sample_size": len(control_subset),
        "aggregate": args.aggregate,
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    stats_path = args.out.with_suffix(".stats.json")
    stats = _compute_relationship_stats(real_points, control_points)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _print_stats_summary(stats)


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


def _fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> Dict[str, float]:
    if scipy_stats is None or n1 < 4 or n2 < 4:
        return {"z": float("nan"), "p": float("nan")}
    if not np.isfinite(r1) or not np.isfinite(r2):
        return {"z": float("nan"), "p": float("nan")}
    r1 = max(min(r1, 0.999999), -0.999999)
    r2 = max(min(r2, 0.999999), -0.999999)
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    if se <= 0:
        return {"z": float("nan"), "p": float("nan")}
    z = (z1 - z2) / se
    p = 2.0 * scipy_stats.norm.sf(abs(z))
    return {"z": float(z), "p": float(p)}


def _compute_relationship_stats(
    real_points: Dict[str, List[Tuple[int, float]]],
    control_points: Dict[str, List[Tuple[int, float]]],
) -> Dict[str, object]:
    stats: Dict[str, object] = {"real": {}, "control": {}, "comparisons": {}}
    for probe in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
        real_stat = _linear_stats(real_points.get(probe, []))
        control_stat = _linear_stats(control_points.get(probe, []))
        stats["real"][probe] = real_stat
        stats["control"][probe] = control_stat
        comp = _fisher_z_test(
            real_stat.get("r", float("nan")),
            int(real_stat.get("n", 0)),
            control_stat.get("r", float("nan")),
            int(control_stat.get("n", 0)),
        )
        comp["real_abs_r_gt_control"] = bool(
            np.isfinite(real_stat.get("r", float("nan")))
            and np.isfinite(control_stat.get("r", float("nan")))
            and abs(real_stat["r"]) > abs(control_stat["r"])
        )
        stats["comparisons"][probe] = comp
    return stats


def _print_stats_summary(stats: Dict[str, object]) -> None:
    for probe in ("fabrication_vs_truthfulness", "lying_vs_truthfulness"):
        real = stats["real"].get(probe, {})
        control = stats["control"].get(probe, {})
        comp = stats["comparisons"].get(probe, {})
        print(f"{probe} real: n={real.get('n')} slope={real.get('slope'):.4g} r={real.get('r'):.4g} p={real.get('p'):.4g}")
        print(
            f"{probe} control: n={control.get('n')} slope={control.get('slope'):.4g} r={control.get('r'):.4g} p={control.get('p'):.4g}"
        )
        print(
            f"{probe} real vs control (Fisher z): z={comp.get('z'):.4g} p={comp.get('p'):.4g} |r| real>control={comp.get('real_abs_r_gt_control')}"
        )


if __name__ == "__main__":
    main()
