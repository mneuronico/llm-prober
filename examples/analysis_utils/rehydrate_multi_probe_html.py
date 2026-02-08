import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concept_probe.visuals import render_batch_heatmap_multi, render_token_heatmap_multi


def _decode_probe_names(raw: np.ndarray) -> List[str]:
    names: List[str] = []
    for item in raw:
        if isinstance(item, bytes):
            names.append(item.decode("utf-8", errors="ignore"))
        else:
            names.append(str(item))
    return names


def _resolve_model_id(batch_dir: Path, explicit_model_id: str | None) -> str:
    if explicit_model_id:
        return explicit_model_id
    run_dir = batch_dir.parent.parent
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config.json at {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model = cfg.get("model_id")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"config.json at {cfg_path} missing model_id")
    return model


def _load_npz(npz_path: Path) -> Dict[str, object]:
    data = np.load(str(npz_path))
    token_ids = data.get("token_ids")
    scores_agg = data.get("scores_agg")
    probe_names_raw = data.get("probe_names")
    if token_ids is None or scores_agg is None or probe_names_raw is None:
        raise ValueError(f"Missing required keys in {npz_path}")
    return {
        "token_ids": np.array(token_ids, dtype=np.int32),
        "scores_agg": np.array(scores_agg, dtype=np.float32),
        "probe_names": _decode_probe_names(np.array(probe_names_raw)),
    }


def rehydrate_batch_html(batch_dir: Path, *, model_id: str | None) -> Dict[str, object]:
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
    npz_paths = sorted(batch_dir.glob("*.npz"))
    if not npz_paths:
        raise ValueError(f"No .npz files found in {batch_dir}")

    resolved_model_id = _resolve_model_id(batch_dir, model_id)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)

    batch_entries: List[tuple[str, List[str], Dict[str, List[float]]]] = []
    html_count = 0
    for npz_path in npz_paths:
        payload = _load_npz(npz_path)
        token_ids = payload["token_ids"]
        scores_agg = payload["scores_agg"]
        probe_names = payload["probe_names"]

        if scores_agg.ndim != 2:
            raise ValueError(f"Expected 2D scores_agg in {npz_path}, got {scores_agg.shape}")
        if scores_agg.shape[0] != len(probe_names):
            raise ValueError(
                f"probe_names length ({len(probe_names)}) does not match scores_agg rows "
                f"({scores_agg.shape[0]}) in {npz_path}"
            )

        # Match the library's token rendering behavior: decode one token id at a time
        # so we get readable whitespace/newline tokens instead of raw BPE markers.
        tokens = [tokenizer.decode([int(t)], skip_special_tokens=False) for t in token_ids.tolist()]
        scores_by_probe = {
            probe_names[i]: scores_agg[i].astype(float).tolist()
            for i in range(len(probe_names))
        }

        html_path = npz_path.with_suffix(".html")
        title = npz_path.stem
        render_token_heatmap_multi(tokens, scores_by_probe, str(html_path), title=title)
        batch_entries.append((title, tokens, scores_by_probe))
        html_count += 1

    batch_html = batch_dir / "batch_prompts.html"
    render_batch_heatmap_multi(batch_entries, str(batch_html), title="Batch prompt scores")

    return {
        "batch_dir": str(batch_dir),
        "model_id": resolved_model_id,
        "npz_count": len(npz_paths),
        "html_count": html_count,
        "batch_html": str(batch_html),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate multi-probe token-highlight HTML files from existing .npz outputs."
    )
    parser.add_argument(
        "--batch-dir",
        required=True,
        help="Path to a multi-probe batch directory containing .npz files.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional HF model id. If omitted, resolves from run config.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = rehydrate_batch_html(Path(args.batch_dir).resolve(), model_id=args.model_id)
    print(
        "Rehydrated HTML for "
        f"{result['html_count']}/{result['npz_count']} prompts. "
        f"Batch view: {result['batch_html']}"
    )


if __name__ == "__main__":
    main()
