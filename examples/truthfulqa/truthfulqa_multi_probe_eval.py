import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

# Make sure the project root is on sys.path so we can import the library.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from concept_probe import ConceptProbe, multi_probe_score_prompts
from concept_probe.modeling import ModelBundle
from concept_probe.utils import json_dump
from truthfulqa import read_truthfulqa


PROBE_PROJECTS = {
    "fabrication_vs_truthfulness": r"outputs\fabrication_vs_truthfulness\20260115_180659",
    "lying_vs_truthfulness": r"outputs\lying_vs_truthfulness\20260116_095854",
}


OUTPUT_ROOT = "outputs_multi"
PROJECT_NAME = "truthfulqa_multi_probe"
OUTPUT_SUBDIR = "truthfulqa_eval"
SAMPLE_PER_TYPE = 20
SAMPLE_SEED = 7

SYSTEM_PROMPT = "You are a helpful assistant."
EVAL_INSTRUCTION = "Answer the question in 4-6 sentences."
USE_INSTRUCTION = False

STEER_PROBE = None  # e.g. "lying_vs_truthfulness"
ALPHAS = None       # e.g. [0.0] or [-5.0, 0.0, 5.0]
ALPHA_UNIT = "raw"


def _find_latest_run(concept_name: str, outputs_root: str = "outputs") -> Optional[str]:
    concept_dir = os.path.join(outputs_root, concept_name)
    if not os.path.isdir(concept_dir):
        return None
    subdirs = [
        os.path.join(concept_dir, d)
        for d in os.listdir(concept_dir)
        if os.path.isdir(os.path.join(concept_dir, d))
    ]
    if not subdirs:
        return None
    return sorted(subdirs)[-1]


def _load_model_bundle(run_dir: str) -> ModelBundle:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return ModelBundle.load(cfg["model"])


def _sample_truthfulqa(
    n_per_type: int,
    *,
    seed: int,
    dir_path: str = ".",
    csv_path: Optional[str] = None,
) -> List[Dict[str, object]]:
    if csv_path is None:
        csv_path = os.path.join(EXAMPLES_DIR, "TruthfulQA.csv")
    items = read_truthfulqa(dir_path=dir_path, csv_path=csv_path, n=None, seed=seed)
    adv = [item for item in items if item["type"] == "adversarial"]
    non = [item for item in items if item["type"] == "non_adversarial"]
    if len(adv) < n_per_type or len(non) < n_per_type:
        raise ValueError("Not enough items in one of the categories to sample the requested size.")

    rng = np.random.default_rng(seed)
    adv_idx = rng.choice(len(adv), size=n_per_type, replace=False)
    non_idx = rng.choice(len(non), size=n_per_type, replace=False)
    sampled = [adv[i] for i in adv_idx] + [non[i] for i in non_idx]
    rng.shuffle(sampled)
    return sampled


def _build_prompts(items: List[Dict[str, object]]) -> List[str]:
    prompts = []
    for item in items:
        question = str(item["prompt"])
        if USE_INSTRUCTION and EVAL_INSTRUCTION:
            prompts.append(f"{EVAL_INSTRUCTION}\nQuestion: {question}")
        else:
            prompts.append(question)
    return prompts


def main() -> None:
    probe_paths: List[str] = []
    for name, path in PROBE_PROJECTS.items():
        resolved = path or _find_latest_run(name)
        if resolved is None:
            raise FileNotFoundError(f"Could not find a run directory for probe '{name}'.")
        probe_paths.append(resolved)

    model_bundle = _load_model_bundle(probe_paths[0])
    probes = [ConceptProbe.load(run_dir, model_bundle=model_bundle) for run_dir in probe_paths]

    items = _sample_truthfulqa(SAMPLE_PER_TYPE, seed=SAMPLE_SEED)
    prompts = _build_prompts(items)

    eval_result = multi_probe_score_prompts(
        probes,
        prompts,
        system_prompt=SYSTEM_PROMPT,
        output_root=OUTPUT_ROOT,
        project_name=PROJECT_NAME,
        output_subdir=OUTPUT_SUBDIR,
        alphas=ALPHAS,
        alpha_unit=ALPHA_UNIT,
        steer_probe=STEER_PROBE,
    )

    results = eval_result["results"]
    run_dir = eval_result["run_dir"]

    if len(results) != len(items):
        raise ValueError("Unexpected number of results; steering with alphas may require one result per prompt.")

    output_items = []
    for item, rec in zip(items, results):
        output_items.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["prompt"],
                "prompt": rec["prompt"],
                "correct_answers": item["correct_answers"],
                "incorrect_answers": item["incorrect_answers"],
                "completion": rec["completion"],
                "alpha": rec["alpha"],
                "npz_path": rec["npz_path"],
                "html_path": rec["html_path"],
            }
        )

    out_path = os.path.join(run_dir, "truthfulqa_eval_items.json")
    json_dump(
        out_path,
        {
            "items": output_items,
            "multi_probe_run_dir": run_dir,
            "probes": eval_result["probes"],
            "settings": {
                "sample_per_type": SAMPLE_PER_TYPE,
                "seed": SAMPLE_SEED,
                "system_prompt": SYSTEM_PROMPT,
                "instruction": EVAL_INSTRUCTION if USE_INSTRUCTION else None,
                "alpha_unit": ALPHA_UNIT,
                "alphas": ALPHAS,
                "steer_probe": STEER_PROBE,
            },
        },
    )

    print(f"Wrote eval items to {out_path}")


if __name__ == "__main__":
    main()
