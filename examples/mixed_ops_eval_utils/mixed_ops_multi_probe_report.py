import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from social_iqa_eval_utils.social_iqa_multi_probe_report import (
    _build_arrays,
    _extract_probe_names,
    _load_items,
    render_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mixed-ops multi-probe HTML report.")
    parser.add_argument("path", help="Batch dir or analysis/per_sample.json path.")
    parser.add_argument("--output", default="report.html", help="Output HTML filename.")
    args = parser.parse_args()

    items, analysis_dir = _load_items(Path(args.path))
    probe_names = _extract_probe_names(items)
    if not probe_names:
        raise ValueError("No probe names found in per_sample.json.")
    rows = _build_arrays(items, probe_names)
    if not rows:
        raise ValueError("No valid rows with correct labels and scores.")
    out_path = render_report(
        rows,
        probe_names,
        analysis_dir,
        title="Mixed Ops Multi-Probe Report",
        output_name=args.output,
    )
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
