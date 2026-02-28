import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "examples" / "wellbeing_dataset" / "run_all_conversation_experiments_logit_ratings.py"
DEFAULTS_CONFIG = REPO_ROOT / "examples" / "wellbeing_dataset" / "conversation_experiment_defaults.json"
EXPERIMENTS_CONFIG = (
    REPO_ROOT
    / "examples"
    / "wellbeing_dataset"
    / "conversation_experiment_concepts_self_qwen25_7b_wellbeing_alpha0_random_scoring_0_to_9.json"
)


def main() -> int:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--defaults-config",
        str(DEFAULTS_CONFIG),
        "--experiments-config",
        str(EXPERIMENTS_CONFIG),
    ]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    return int(subprocess.call(cmd, cwd=str(REPO_ROOT)))


if __name__ == "__main__":
    raise SystemExit(main())
