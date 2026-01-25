# Backfill missing mixed-ops coherence ratings and plots.

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from examples.mixed_ops_eval_utils.mixed_ops_multi_concept_eval import ENV_PATH, OUTPUT_SUBDIR, PROJECTS
from examples.mixed_ops_eval_utils.mixed_ops_eval_analysis import analyze_mixed_ops_batch
from examples.main_utils.coherence_rater import rate_batch_coherence


def _find_batch_dirs(output_dir: Path, batch_name: Optional[str], all_batches: bool) -> List[Path]:
    if not output_dir.exists():
        return []
    if batch_name:
        target = output_dir / batch_name
        return [target] if target.exists() else []
    candidates = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("batch_")]
    if not candidates:
        return []
    if all_batches:
        return sorted(candidates, key=lambda p: p.name)
    return [sorted(candidates, key=lambda p: p.name)[-1]]


def _iter_projects(names: Optional[List[str]]) -> Iterable:
    if not names:
        return PROJECTS
    requested = {name.strip().lower() for name in names if name.strip()}
    return [p for p in PROJECTS if p.name.lower() in requested]


def _rehydrate_batch(
    batch_dir: Path,
    *,
    env_path: str,
    model: str,
    max_elements: int,
) -> None:
    per_sample_path = batch_dir / "analysis" / "per_sample.json"
    if not per_sample_path.exists():
        print(f"Warning: missing per_sample.json in {batch_dir}")
        return

    coherence_path = batch_dir / "coherence_rating.json"
    if not coherence_path.exists():
        try:
            rate_batch_coherence(
                str(batch_dir),
                max_elements_per_request=max_elements,
                model=model,
                env_path=env_path,
            )
        except Exception as exc:
            print(f"Warning: coherence rating failed for {batch_dir.name}: {exc}")
            return

    try:
        analyze_mixed_ops_batch(str(batch_dir))
    except Exception as exc:
        print(f"Warning: mixed-ops coherence plots failed for {batch_dir.name}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing mixed-ops coherence ratings and plots."
    )
    parser.add_argument(
        "--project",
        action="append",
        help="Project name to process (repeatable). Defaults to all projects.",
    )
    parser.add_argument(
        "--batch",
        help="Specific batch directory name (e.g., batch_20260125_122612). Defaults to latest.",
    )
    parser.add_argument(
        "--all-batches",
        action="store_true",
        help="Process every batch directory under the eval output.",
    )
    parser.add_argument("--env-path", default=ENV_PATH, help="Path to .env for GROQ_API_KEY.")
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="Groq model to use for coherence ratings.",
    )
    parser.add_argument(
        "--max-per-request",
        type=int,
        default=8,
        help="Max completions per coherence request.",
    )
    args = parser.parse_args()

    projects = list(_iter_projects(args.project))
    if not projects:
        print("Warning: no matching projects found.")
        return

    for project in projects:
        output_dir = project.project_dir / OUTPUT_SUBDIR
        batch_dirs = _find_batch_dirs(output_dir, args.batch, args.all_batches)
        if not batch_dirs:
            print(f"Warning: no batches found for {project.name} in {output_dir}")
            continue
        for batch_dir in batch_dirs:
            _rehydrate_batch(
                batch_dir,
                env_path=args.env_path,
                model=args.model,
                max_elements=args.max_per_request,
            )


if __name__ == "__main__":
    main()
