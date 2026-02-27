import os
import sys
import gc
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from concept_probe import train_concept_from_json

try:
    import torch
except Exception:
    torch = None


JSON_FILES = [
    #"sad_vs_happy.json",
    "bored_vs_interested.json",
    "impulsive_vs_planning.json",
    "distracted_vs_focused.json",
]


def _cleanup_cuda() -> None:
    """Best-effort memory cleanup between sequential trainings."""
    gc.collect()
    if torch is None:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1] / "experiment_configs" / "central_pairs_llama31_8b"
    for idx, json_name in enumerate(JSON_FILES, start=1):
        _cleanup_cuda()
        json_path = base_dir / json_name
        if not json_path.exists():
            raise FileNotFoundError(f"Missing concept JSON: {json_path}")
        print(f"[train {idx}/{len(JSON_FILES)}] {json_path}")
        train_concept_from_json(str(json_path))
        _cleanup_cuda()


if __name__ == "__main__":
    main()
