#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "analysis"
DEFAULT_FAMILY_GLOB = "gemma-*-self-steering"
DEFAULT_EXPERIMENT_SCRIPT = REPO_ROOT / "examples" / "wellbeing_dataset" / "conversation_experiment_logit_ratings.py"


def _discover_run_dirs(analysis_dir: Path, family_glob: str) -> List[Path]:
    runs: List[Path] = []
    for family_dir in sorted(analysis_dir.glob(family_glob)):
        if not family_dir.is_dir():
            continue
        for run_dir in sorted(family_dir.iterdir()):
            if run_dir.is_dir():
                runs.append(run_dir.resolve())
    return runs


def _find_generated_config(run_dir: Path) -> Path:
    configs = sorted(run_dir.glob("*.generated_config.json"))
    if not configs:
        raise FileNotFoundError(f"No *.generated_config.json found in {run_dir}")
    if len(configs) == 1:
        return configs[0].resolve()
    # Prefer config file that starts with run directory name when multiple are present.
    preferred = [p for p in configs if p.name.startswith(run_dir.name)]
    if preferred:
        return sorted(preferred)[0].resolve()
    return configs[0].resolve()


def _run_one(
    *,
    python_exe: str,
    experiment_script: Path,
    run_dir: Path,
    config_path: Path,
    bootstrap_samples: int,
    dry_run: bool,
) -> int:
    cmd = [
        python_exe,
        str(experiment_script),
        "--config",
        str(config_path),
        "--reanalyze-only",
        "--existing-output-dir",
        str(run_dir),
        "--bootstrap-samples-override",
        str(int(bootstrap_samples)),
    ]
    print(" ".join(cmd))
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(completed.returncode)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch reanalysis for Gemma self-steering runs. "
            "Runs conversation_experiment_logit_ratings.py in --reanalyze-only mode."
        )
    )
    parser.add_argument(
        "--analysis-dir",
        default=str(DEFAULT_ANALYSIS_DIR),
        help=f"Root analysis directory (default: {DEFAULT_ANALYSIS_DIR}).",
    )
    parser.add_argument(
        "--family-glob",
        default=DEFAULT_FAMILY_GLOB,
        help=f"Glob for model-family folders under analysis-dir (default: {DEFAULT_FAMILY_GLOB}).",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable to use for child runs (default: current interpreter).",
    )
    parser.add_argument(
        "--experiment-script",
        default=str(DEFAULT_EXPERIMENT_SCRIPT),
        help=f"Path to conversation experiment script (default: {DEFAULT_EXPERIMENT_SCRIPT}).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap samples override for reanalysis (default: 1000).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional run directory names to include (exact name match).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Optional run directory names to skip (exact name match).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing remaining runs if one run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analysis_dir = Path(args.analysis_dir).resolve()
    experiment_script = Path(args.experiment_script).resolve()

    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
    if not experiment_script.exists():
        raise FileNotFoundError(f"Experiment script not found: {experiment_script}")

    only = set(str(x).strip() for x in args.only if str(x).strip())
    skip = set(str(x).strip() for x in args.skip if str(x).strip())

    runs = _discover_run_dirs(analysis_dir, args.family_glob)
    if only:
        runs = [r for r in runs if r.name in only]
    if skip:
        runs = [r for r in runs if r.name not in skip]

    if not runs:
        print("No runs matched the requested filters.")
        return

    print(f"[info] repo={REPO_ROOT}")
    print(f"[info] runs={len(runs)} bootstrap_samples={int(args.bootstrap_samples)}")

    failures: List[str] = []
    started = time.time()
    for idx, run_dir in enumerate(runs, start=1):
        print(f"[{idx}/{len(runs)}] {run_dir.name}")
        try:
            cfg = _find_generated_config(run_dir)
            rc = _run_one(
                python_exe=str(args.python_exe),
                experiment_script=experiment_script,
                run_dir=run_dir,
                config_path=cfg,
                bootstrap_samples=int(args.bootstrap_samples),
                dry_run=bool(args.dry_run),
            )
            if rc != 0:
                failures.append(f"{run_dir.name} (exit={rc})")
                print(f"[error] {run_dir.name} failed with exit code {rc}")
                if not args.continue_on_error:
                    break
            else:
                print(f"[ok] {run_dir.name}")
        except Exception as exc:
            failures.append(f"{run_dir.name} ({exc})")
            print(f"[error] {run_dir.name}: {exc}")
            if not args.continue_on_error:
                break

    elapsed = time.time() - started
    print(f"[done] elapsed_sec={elapsed:.1f}")
    if failures:
        print("[done] failures:")
        for item in failures:
            print(f" - {item}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
