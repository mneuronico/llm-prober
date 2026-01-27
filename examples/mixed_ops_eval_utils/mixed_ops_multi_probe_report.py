import argparse

from concept_probe import generate_multi_probe_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mixed-ops multi-probe HTML report.")
    parser.add_argument("path", help="Batch dir or analysis/per_sample.json path.")
    parser.add_argument("--output", default="report.html", help="Output HTML filename.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for train/test split.")
    args = parser.parse_args()

    out_path = generate_multi_probe_report(
        args.path,
        title="Mixed Ops Multi-Probe Report",
        output_name=args.output,
        seed=args.seed,
    )
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
