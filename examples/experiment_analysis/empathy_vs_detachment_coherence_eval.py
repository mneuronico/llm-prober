import json
import os
import sys
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ROOT_DIR = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from concept_probe import ProbeWorkspace
from concept_probe.math_eval import generate_addition_problems
from examples.math_eval_utils.math_eval_analysis import analyze_math_eval_results
from examples.main_utils.coherence_rater import rate_batch_coherence


PROJECT_DIR = "outputs/empathy_vs_detachment/20260109_150734"
ENV_PATH = r"C:\Users\Nico\Documents\GitHub\llm-prober\.env"
ALPHAS = [-100.0, -50.0, -40.0, -20.0, -5.0, 0.0, 5.0, 20.0, 50.0, 100.0]
ALPHA_UNIT = "sigma"

EVAL_INSTRUCTION = (
    "Solve the arithmetic problem. Show brief working, then end with the final line "
    "\"ANSWER: <number>\" using only digits for the number."
)

RATING_COLORS = {
    "COHERENT": "#33b233",
    "PARTIALLY_COHERENT": "#f2cc33",
    "NONSENSE": "#d93434",
}


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_accuracy_by_coherence(
    alpha_vals: List[float],
    accuracy_by_rating: Dict[str, List[float]],
    out_path: Path,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for rating, color in RATING_COLORS.items():
        ax.plot(alpha_vals, accuracy_by_rating[rating], marker="o", linewidth=2, color=color, label=rating)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs steering alpha (by coherence)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_counts_by_coherence(
    alpha_vals: List[float],
    counts_by_rating: Dict[str, List[int]],
    out_path: Path,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for rating, color in RATING_COLORS.items():
        ax.plot(alpha_vals, counts_by_rating[rating], marker="o", linewidth=2, color=color, label=rating)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Count")
    ax.set_title("Coherence rating count by alpha")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    if not PROJECT_DIR:
        raise ValueError("Set PROJECT_DIR before running.")

    workspace = ProbeWorkspace(project_directory=PROJECT_DIR)
    probe = workspace.get_probe(name="empathy_vs_detachment")

    problems = generate_addition_problems(
        20,
        seed=7,
        min_terms=2,
        max_terms=4,
        min_value=10,
        max_value=999,
        allow_negative=False,
    )

    eval_prompts = [f"{EVAL_INSTRUCTION}\nProblem: {p['expression']}" for p in problems]

    results = probe.score_prompts(
        prompts=eval_prompts,
        system_prompt=workspace.config["prompts"]["neutral_system"],
        alphas=ALPHAS,
        alpha_unit=ALPHA_UNIT,
        steer_layers="window",
        steer_window_radius=2,
        steer_distribute=True,
        max_new_tokens=512,
        output_subdir="math_eval",
    )

    analyze_math_eval_results(results, problems, ALPHAS)

    batch_dir = Path(results[0]["npz_path"]).resolve().parent
    rate_batch_coherence(
        str(batch_dir),
        max_elements_per_request=8,
        model="openai/gpt-oss-20b",
        env_path=ENV_PATH,
    )

    per_sample_path = batch_dir / "analysis" / "per_sample.json"
    ratings_path = batch_dir / "coherence_rating.json"
    per_sample = _load_json(per_sample_path).get("items", [])
    ratings_raw = _load_json(ratings_path)
    ratings = {
        entry["example"]: entry["rating"]
        for entry in ratings_raw
        if isinstance(entry, dict) and "example" in entry and "rating" in entry
    }

    counts_by_rating: Dict[str, List[int]] = {k: [0] * len(ALPHAS) for k in RATING_COLORS}
    accuracy_by_rating: Dict[str, List[float]] = {
        k: [float("nan")] * len(ALPHAS) for k in RATING_COLORS
    }

    for i, alpha in enumerate(ALPHAS):
        alpha_items = [item for item in per_sample if float(item["alpha"]) == float(alpha)]
        for rating in RATING_COLORS:
            rows = []
            for item in alpha_items:
                example = Path(item["npz_path"]).name
                if ratings.get(example) == rating:
                    rows.append(item)
            counts_by_rating[rating][i] = len(rows)
            if rows:
                accuracy_by_rating[rating][i] = sum(1 for r in rows if r["correct"]) / len(rows)

    plots_dir = batch_dir / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_accuracy_by_coherence(
        ALPHAS, accuracy_by_rating, plots_dir / "accuracy_vs_alpha_by_coherence.png"
    )
    _plot_counts_by_coherence(
        ALPHAS, counts_by_rating, plots_dir / "coherence_counts_by_alpha.png"
    )


if __name__ == "__main__":
    main()
