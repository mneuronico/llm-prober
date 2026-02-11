import argparse
import copy
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERSATION_EXPERIMENT_SCRIPT = (
    REPO_ROOT / "examples" / "wellbeing_dataset" / "conversation_experiment_logit_ratings.py"
)

DEFAULT_CONFIGS = [
    "examples/wellbeing_dataset/empathy_truthfulness_multi_probe_config.json",
    "examples/wellbeing_dataset/focus_truthfulness_multi_probe_config.json",
    "examples/wellbeing_dataset/wellbeing_truthfulness_multi_probe_config.json",
    "examples/wellbeing_dataset/interest_truthfulness_multi_probe_config.json",
    "examples/wellbeing_dataset/impulsivity_truthfulness_multi_probe_config.json",
]

DEFAULT_SPLIT_DEFAULTS_CONFIG = (
    "examples/wellbeing_dataset/conversation_experiment_defaults.json"
)
DEFAULT_SPLIT_EXPERIMENTS_CONFIG = (
    "examples/wellbeing_dataset/conversation_experiment_concepts.json"
)


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_configs(config_args: Optional[List[str]]) -> List[Path]:
    raw_paths = config_args if config_args else DEFAULT_CONFIGS
    resolved: List[Path] = []
    for raw in raw_paths:
        p = Path(raw)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        resolved.append(p)
    return resolved


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: object, override: object) -> object:
    if isinstance(base, dict) and isinstance(override, dict):
        merged: Dict[object, object] = {k: copy.deepcopy(v) for k, v in base.items()}
        for k, v in override.items():
            if k in merged:
                merged[k] = _deep_merge(merged[k], v)
            else:
                merged[k] = copy.deepcopy(v)
        return merged
    return copy.deepcopy(override)


def _sanitize_filename(text: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in text.strip())
    safe = safe.strip("_")
    return safe or "experiment"


def _extract_experiments(payload: object) -> List[Dict[str, object]]:
    if isinstance(payload, dict) and isinstance(payload.get("experiments"), list):
        items = payload["experiments"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Experiments config must be a list or contain an 'experiments' list.")

    out: List[Dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "name" not in item:
            raise ValueError("Every experiment entry must include 'name'.")
        name = str(item["name"]).strip()
        if not name:
            raise ValueError("Experiment 'name' cannot be empty.")
        if "overrides" in item:
            overrides = item.get("overrides")
            if not isinstance(overrides, dict):
                raise ValueError(f"Experiment '{name}' has non-dict 'overrides'.")
        else:
            overrides = {k: v for k, v in item.items() if k != "name"}
        out.append({"name": name, "overrides": overrides})
    return out


def _build_configs_from_split(
    *,
    defaults_config_path: Path,
    experiments_config_path: Path,
    selected_experiment_names: Optional[List[str]],
    generated_configs_dir: Optional[Path],
) -> List[Path]:
    if not defaults_config_path.exists():
        raise FileNotFoundError(f"Defaults config not found: {defaults_config_path}")
    if not experiments_config_path.exists():
        raise FileNotFoundError(f"Experiments config not found: {experiments_config_path}")

    defaults_payload = _load_json(defaults_config_path)
    defaults_block = defaults_payload.get("defaults", defaults_payload) if isinstance(defaults_payload, dict) else defaults_payload
    if not isinstance(defaults_block, dict):
        raise ValueError("Defaults config must be a JSON object or contain a top-level 'defaults' object.")

    experiments_payload = _load_json(experiments_config_path)
    experiments = _extract_experiments(experiments_payload)
    if not experiments:
        raise ValueError("No experiments found in experiments config.")

    selected_set = set(selected_experiment_names) if selected_experiment_names else None
    if selected_set is not None:
        found_names = {str(item["name"]) for item in experiments}
        missing = [name for name in selected_set if name not in found_names]
        if missing:
            raise ValueError(f"Requested experiment names not found: {missing}")
        experiments = [item for item in experiments if str(item["name"]) in selected_set]

    if generated_configs_dir is not None:
        generated_configs_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []
    for item in experiments:
        name = str(item["name"])
        overrides = item.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Experiment '{name}' overrides must be a dict.")
        merged = _deep_merge(defaults_block, overrides)
        if not isinstance(merged, dict):
            raise ValueError(f"Experiment '{name}' merged config is not an object.")
        output_dir_template = str(
            merged.get("output_dir_template", "analysis/conversation_experiment_logit_ratings_{timestamp}")
        )
        if "{timestamp}" in output_dir_template:
            output_dir_template = output_dir_template.format(timestamp=_now_tag())
            merged["output_dir_template"] = output_dir_template
        output_dir_path = _resolve_path(output_dir_template)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if generated_configs_dir is not None:
            out_path = generated_configs_dir / f"{_sanitize_filename(name)}.json"
        else:
            out_path = output_dir_path / f"{_sanitize_filename(name)}.generated_config.json"
        out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        out_paths.append(out_path)
    return out_paths


def _default_logs_dir() -> Path:
    return REPO_ROOT / "analysis" / f"conversation_experiment_logit_ratings_batch_{_now_tag()}" / "logs"


def _resolve_output_dir_from_config(config_path: Path) -> Path:
    raw = _load_json(config_path)
    output_dir_template = str(raw.get("output_dir_template", "analysis/conversation_experiment_logit_ratings"))
    if "{timestamp}" in output_dir_template:
        output_dir_template = output_dir_template.format(timestamp=_now_tag())
    return _resolve_path(output_dir_template)


def _run_one(
    *,
    python_exec: str,
    config_path: Path,
    log_path: Path,
    dry_run: bool,
) -> int:
    cmd = [
        python_exec,
        str(CONVERSATION_EXPERIMENT_SCRIPT),
        "--config",
        str(config_path),
    ]
    print(f"[START] {config_path.name}")
    print(f"        cmd: {' '.join(cmd)}")
    print(f"        log: {log_path}")

    if dry_run:
        return 0

    start_time = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb", buffering=0) as log_fh:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["CP_PROGRESS_INLINE"] = env.get("CP_PROGRESS_INLINE", "1")
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            env=env,
        )
        if proc.stdout is None:
            raise RuntimeError("Failed to capture child process stdout.")
        while True:
            chunk = proc.stdout.read(1)
            if chunk == b"":
                break
            # Keep carriage returns intact so child inline progress bars stay on one line.
            if hasattr(sys.stdout, "buffer"):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            else:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            log_fh.write(chunk)
        proc.stdout.close()
        result_code = proc.wait()
    elapsed = time.time() - start_time
    if result_code == 0:
        print(f"[DONE] {config_path.name} ({elapsed:.1f}s)")
    else:
        print(f"[FAIL] {config_path.name} rc={result_code} ({elapsed:.1f}s)")
        print(f"       log: {log_path}")
    return int(result_code)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run wellbeing conversation experiments (token/logit self-ratings) for multiple configs."
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help=(
            "Optional list of config paths. "
            "Default: empathy/focus/wellbeing/interest/impulsivity multi-probe configs."
        ),
    )
    parser.add_argument(
        "--defaults-config",
        default=None,
        help=(
            "Defaults JSON path for split-config mode. "
            "Use with --experiments-config."
        ),
    )
    parser.add_argument(
        "--experiments-config",
        default=None,
        help=(
            "Experiments JSON path for split-config mode. "
            "Use with --defaults-config."
        ),
    )
    parser.add_argument(
        "--experiment-names",
        nargs="*",
        default=None,
        help="Optional subset of experiment names from --experiments-config.",
    )
    parser.add_argument(
        "--generated-configs-dir",
        default=None,
        help="Where to write merged configs for split-config mode (default: sibling folder next to logs).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child runs (default: current interpreter).",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Directory for per-config logs (default: analysis/conversation_experiment_logit_ratings_batch_<timestamp>/logs).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining configs even if one fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and logs without launching processes.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    using_split_mode = bool(args.defaults_config or args.experiments_config)
    split_inline_artifacts = bool(using_split_mode and not args.logs_dir and not args.generated_configs_dir)
    logs_dir = Path(args.logs_dir).resolve() if args.logs_dir else (
        None if split_inline_artifacts else _default_logs_dir()
    )
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)

    if using_split_mode and args.configs is not None:
        raise ValueError("Use either --configs or split mode (--defaults-config + --experiments-config), not both.")

    if using_split_mode:
        if not args.defaults_config or not args.experiments_config:
            raise ValueError("Split mode requires both --defaults-config and --experiments-config.")
        defaults_path = _resolve_path(args.defaults_config)
        experiments_path = _resolve_path(args.experiments_config)
        generated_dir = Path(args.generated_configs_dir).resolve() if args.generated_configs_dir else None
        config_paths = _build_configs_from_split(
            defaults_config_path=defaults_path,
            experiments_config_path=experiments_path,
            selected_experiment_names=args.experiment_names,
            generated_configs_dir=generated_dir,
        )
        print(f"Split mode defaults: {defaults_path}")
        print(f"Split mode experiments: {experiments_path}")
        if generated_dir is not None:
            print(f"Generated configs: {generated_dir}")
        else:
            print("Generated configs: per-experiment output folders")
    else:
        config_paths = _resolve_configs(args.configs)

    print(f"Repo root: {REPO_ROOT}")
    print(f"Python: {args.python}")
    print(f"Config count: {len(config_paths)}")
    print("Mode: sequential")
    if logs_dir is not None:
        print(f"Logs dir: {logs_dir}")
    else:
        print("Logs dir: per-experiment output folders")

    failures = 0
    completed = 0

    try:
        for config_path in config_paths:
            if logs_dir is not None:
                log_path = logs_dir / f"{config_path.stem}.log"
            else:
                out_dir = _resolve_output_dir_from_config(config_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                log_path = out_dir / "run_all.log"
            rc = _run_one(
                python_exec=args.python,
                config_path=config_path,
                log_path=log_path,
                dry_run=args.dry_run,
            )
            completed += 1
            if rc != 0:
                failures += 1
                if not args.continue_on_error:
                    break
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)

    print(f"Completed: {completed}/{len(config_paths)}")
    if failures > 0:
        print(f"Failures: {failures}")
        raise SystemExit(1)
    print("All experiments finished successfully.")


if __name__ == "__main__":
    main()
