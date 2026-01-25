# Rate coherence for an existing mixed-ops batch and regenerate plots.

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from examples.mixed_ops_eval_utils.mixed_ops_eval_analysis import analyze_mixed_ops_batch
from examples.main_utils.coherence_rater import rate_batch_coherence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rate coherence for an existing mixed-ops batch and regenerate plots."
    )
    parser.add_argument("batch_dir", help="Path to batch directory with prompt_*.npz files.")
    parser.add_argument(
        "--env-path",
        default=str(ROOT_DIR / ".env"),
        help="Path to .env containing GROQ_API_KEY (default: repo .env).",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="Groq model name to use for coherence rating.",
    )
    parser.add_argument(
        "--max-elements-per-request",
        type=int,
        default=8,
        help="Max completions per request to the rater.",
    )
    args = parser.parse_args()

    batch_path = Path(args.batch_dir).resolve()
    if not batch_path.exists():
        raise FileNotFoundError(f"Batch directory not found: {batch_path}")

    rate_batch_coherence(
        str(batch_path),
        max_elements_per_request=args.max_elements_per_request,
        model=args.model,
        env_path=args.env_path,
    )
    analyze_mixed_ops_batch(str(batch_path))


if __name__ == "__main__":
    main()
