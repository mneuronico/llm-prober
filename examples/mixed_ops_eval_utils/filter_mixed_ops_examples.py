# Filter mixed-ops examples by coherence rating and correctness.

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
VALID_RATINGS = {"COHERENT", "PARTIALLY_COHERENT", "NONSENSE"}


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_batch_dir(path: Path) -> bool:
    return path.name.startswith("batch_") and (path / "analysis" / "per_sample.json").exists()


def _find_batches(eval_dir: Path) -> List[Path]:
    return sorted([p for p in eval_dir.iterdir() if _is_batch_dir(p)], key=lambda p: p.name)


def _load_per_sample(batch_dir: Path) -> List[Dict[str, object]]:
    per_sample_path = batch_dir / "analysis" / "per_sample.json"
    if not per_sample_path.exists():
        return []
    data = _load_json(per_sample_path)
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    for item in items:
        if isinstance(item, dict):
            item["batch_name"] = batch_dir.name
            item["batch_dir"] = str(batch_dir)
    return [item for item in items if isinstance(item, dict)]


def _load_ratings(batch_dir: Path) -> Dict[str, str]:
    ratings_path = batch_dir / "coherence_rating.json"
    if not ratings_path.exists():
        return {}
    try:
        data = _load_json(ratings_path)
    except json.JSONDecodeError:
        return {}
    ratings: Dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        example = entry.get("example")
        rating = entry.get("rating")
        if isinstance(example, str) and isinstance(rating, str):
            ratings[example] = rating
    return ratings


def _normalize_rating(value: str) -> str:
    return value.strip().upper()


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _collect_matches(
    batch_dir: Path,
    rating_filter: str,
    correct_filter: bool,
) -> Tuple[List[Dict[str, object]], int, int]:
    per_sample = _load_per_sample(batch_dir)
    ratings = _load_ratings(batch_dir)
    if not ratings:
        return [], len(per_sample), 0

    matched: List[Dict[str, object]] = []
    for item in per_sample:
        npz_path = item.get("npz_path")
        if not isinstance(npz_path, str):
            continue
        example_name = Path(npz_path).name
        rating = ratings.get(example_name)
        if rating != rating_filter:
            continue
        if bool(item.get("correct")) != correct_filter:
            continue
        matched.append(
            {
                "prompt": item.get("prompt"),
                "completion": item.get("completion"),
                "alpha": item.get("alpha"),
                "correct": bool(item.get("correct")),
                "rating": rating,
                "batch_name": item.get("batch_name"),
                "npz_path": item.get("npz_path"),
                "html_path": item.get("html_path"),
            }
        )
    return matched, len(per_sample), len(ratings)


def _write_output(
    out_dir: Path,
    output_name: str,
    rating: str,
    correct: bool,
    sources: List[Dict[str, object]],
    total_items: int,
    total_rated: int,
    matches: List[Dict[str, object]],
) -> None:
    payload = {
        "meta": {
            "rating": rating,
            "correct": correct,
            "total_items": total_items,
            "total_rated": total_rated,
            "matched": len(matches),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sources": sources,
        },
        "items": matches,
    }
    out_path = out_dir / output_name
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


def process_path(path: Path, rating: str, correct: bool, output_name: str) -> None:
    if _is_batch_dir(path):
        matched, total_items, total_rated = _collect_matches(path, rating, correct)
        sources = [{"type": "batch", "path": str(path), "batch": path.name}]
        _write_output(path, output_name, rating, correct, sources, total_items, total_rated, matched)
        return

    if not path.exists():
        print(f"Warning: path not found: {path}")
        return

    batch_dirs = _find_batches(path)
    if not batch_dirs:
        print(f"Warning: no batch folders found under {path}")
        return

    all_matches: List[Dict[str, object]] = []
    total_items = 0
    total_rated = 0
    sources: List[Dict[str, object]] = []
    for batch_dir in batch_dirs:
        matches, items_count, rated_count = _collect_matches(batch_dir, rating, correct)
        total_items += items_count
        total_rated += rated_count
        all_matches.extend(matches)
        sources.append(
            {
                "type": "batch",
                "path": str(batch_dir),
                "batch": batch_dir.name,
                "items": items_count,
                "rated": rated_count,
                "matched": len(matches),
            }
        )

    _write_output(path, output_name, rating, correct, sources, total_items, total_rated, all_matches)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter mixed-ops examples by coherence rating and correctness."
    )
    parser.add_argument("paths", nargs="+", help="Batch folder or eval folder(s).")
    parser.add_argument("--rating", required=True, help="COHERENT, PARTIALLY_COHERENT, or NONSENSE.")
    parser.add_argument("--correct", required=True, help="true or false.")
    parser.add_argument(
        "--output-name",
        default="filtered_examples.json",
        help="Output JSON filename to write in each target directory.",
    )
    args = parser.parse_args()

    rating = _normalize_rating(args.rating)
    if rating not in VALID_RATINGS:
        raise ValueError(f"Unknown rating: {args.rating}")
    correct = _parse_bool(args.correct)

    for raw in args.paths:
        process_path(Path(raw).resolve(), rating, correct, args.output_name)


if __name__ == "__main__":
    main()
