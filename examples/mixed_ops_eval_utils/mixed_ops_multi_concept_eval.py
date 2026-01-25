# Run mixed-ops evals for multiple concepts.


import argparse
import html
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EXAMPLES_DIR))

from concept_probe import ProbeWorkspace
from examples.math_eval_utils.math_eval_analysis import analyze_math_eval_results
from examples.mixed_ops_eval_utils.mixed_ops_eval import build_mixed_ops_prompts, generate_mixed_ops_problems
from examples.mixed_ops_eval_utils.mixed_ops_eval_analysis import analyze_mixed_ops_batch
from examples.main_utils.coherence_rater import rate_batch_coherence


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    project_dir: Path


PROJECTS: Sequence[ProjectSpec] = [
    ProjectSpec("bored_vs_interested", ROOT_DIR / "outputs/bored_vs_interested/20260122_112208"),
    ProjectSpec("distracted_vs_focused", ROOT_DIR / "outputs/distracted_vs_focused/20260122_095601"),
    ProjectSpec("dumb_vs_smart", ROOT_DIR / "outputs/dumb_vs_smart/20260122_113829"),
    ProjectSpec("impulsive_vs_planning", ROOT_DIR / "outputs/impulsive_vs_planning/20260120_181954"),
    ProjectSpec("introvert_vs_extrovert", ROOT_DIR / "outputs/introvert_vs_extrovert/20260120_180218"),
    ProjectSpec(
        "rough_messy_vs_detailed_ordered",
        ROOT_DIR / "outputs/rough_messy_vs_detailed_ordered/20260122_112948",
    ),
]

ENV_PATH = r"C:\Users\Nico\Documents\GitHub\llm-prober\.env"
ALPHAS = [-100, -50, 50, 100]
ALPHA_UNIT = "sigma"
PROBLEM_COUNT = 20
PROBLEM_SEED = 19
OUTPUT_SUBDIR = "math_eval_mixed_ops_5x2"
NUM_TERMS = 5
DIGITS = 2

DASHBOARD_DIR = ROOT_DIR / "outputs/mixed_ops_dashboard"
DASHBOARD_FILENAME = "index.html"


def run_project(project: ProjectSpec, problems: List[Dict[str, object]], prompts: List[str]) -> Path:
    workspace = ProbeWorkspace(project_directory=str(project.project_dir))
    probe = workspace.get_probe(name=project.name)

    results = probe.score_prompts(
        prompts=prompts,
        system_prompt=workspace.config["prompts"]["neutral_system"],
        alphas=ALPHAS,
        alpha_unit=ALPHA_UNIT,
        steer_layers="window",
        steer_window_radius=2,
        steer_distribute=True,
        max_new_tokens=512,
        output_subdir=OUTPUT_SUBDIR,
    )

    analyze_math_eval_results(results, problems, ALPHAS)

    batch_dir = Path(results[0]["npz_path"]).resolve().parent
    coherence_ok = True
    try:
        rate_batch_coherence(
            str(batch_dir),
            max_elements_per_request=8,
            model="openai/gpt-oss-20b",
            env_path=ENV_PATH,
        )
    except Exception as exc:
        coherence_ok = False
        print(
            f"Warning: coherence rating failed for {project.name} ({batch_dir.name}): {exc}"
        )

    if coherence_ok:
        analyze_mixed_ops_batch(str(batch_dir))
    return batch_dir


def _find_latest_batch_dir(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    candidates = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("batch_")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def _relpath(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path, base_dir).replace("\\", "/")


def _load_stats(stats_path: Path) -> Dict[str, object]:
    if not stats_path.exists():
        return {}
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _summarize_stats(stats: Dict[str, object]) -> Dict[str, object]:
    accuracy_by_alpha = stats.get("accuracy_by_alpha", [])
    score_by_alpha = stats.get("score_by_alpha", [])

    def _find_alpha(rows: List[Dict[str, object]], alpha: float, key: str) -> Optional[float]:
        for row in rows:
            if float(row.get("alpha", 0.0)) == float(alpha):
                val = row.get(key)
                return float(val) if val is not None else None
        return None

    best_acc = None
    best_alpha = None
    for row in accuracy_by_alpha:
        acc = row.get("accuracy")
        if acc is None:
            continue
        if best_acc is None or float(acc) > float(best_acc):
            best_acc = float(acc)
            best_alpha = float(row.get("alpha", 0.0))

    return {
        "acc_at_zero": _find_alpha(accuracy_by_alpha, 0.0, "accuracy"),
        "score_at_zero": _find_alpha(score_by_alpha, 0.0, "mean_score"),
        "best_acc": best_acc,
        "best_alpha": best_alpha,
    }


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if not isinstance(value, (int, float)) or not (value == value):
        return "n/a"
    return f"{value:.3f}"


def build_dashboard(projects: Sequence[ProjectSpec]) -> Path:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DASHBOARD_DIR / DASHBOARD_FILENAME

    cards: List[str] = []
    for project in projects:
        output_dir = project.project_dir / OUTPUT_SUBDIR
        batch_dir = _find_latest_batch_dir(output_dir)
        display_name = project.name.replace("_", " ")

        if batch_dir is None:
            cards.append(
                f"""
                <article class="card missing">
                  <header>
                    <h2>{html.escape(display_name)}</h2>
                    <div class="meta">No batch found under {html.escape(str(output_dir))}</div>
                  </header>
                </article>
                """
            )
            continue

        stats_path = batch_dir / "analysis" / "stats.json"
        stats = _load_stats(stats_path)
        summary = _summarize_stats(stats)

        batch_prompts = batch_dir / "batch_prompts.html"
        per_sample = batch_dir / "analysis" / "per_sample.json"
        plots_dir = batch_dir / "analysis" / "plots"

        plot_names = [
            "accuracy_vs_alpha.png",
            "score_vs_alpha.png",
            "score_by_correctness.png",
            "score_by_correctness_by_alpha.png",
            "accuracy_vs_alpha_by_coherence.png",
            "coherence_counts_by_alpha.png",
        ]
        plot_items = []
        for plot_name in plot_names:
            plot_path = plots_dir / plot_name
            if plot_path.exists():
                plot_items.append(
                    f"""
                    <figure class="plot">
                      <img src="{html.escape(_relpath(plot_path, DASHBOARD_DIR))}" alt="{html.escape(plot_name)}">
                      <figcaption>{html.escape(plot_name.replace('_', ' '))}</figcaption>
                    </figure>
                    """
                )

        links = [
            (batch_prompts, "Batch prompts"),
            (stats_path, "Stats JSON"),
            (per_sample, "Per-sample JSON"),
        ]
        link_items = []
        for path, label in links:
            if path.exists():
                link_items.append(
                    f'<a href="{html.escape(_relpath(path, DASHBOARD_DIR))}" target="_blank">{html.escape(label)}</a>'
                )

        cards.append(
            f"""
            <article class="card">
              <header>
                <h2>{html.escape(display_name)}</h2>
                <div class="meta">{html.escape(batch_dir.name)}</div>
              </header>
              <section class="stats">
                <div><span>acc @ 0</span><strong>{_format_metric(summary.get("acc_at_zero"))}</strong></div>
                <div><span>score @ 0</span><strong>{_format_metric(summary.get("score_at_zero"))}</strong></div>
                <div>
                  <span>best acc</span>
                  <strong>{_format_metric(summary.get("best_acc"))}</strong>
                  <em>alpha {summary.get("best_alpha") if summary.get("best_alpha") is not None else "n/a"}</em>
                </div>
              </section>
              <nav class="links">
                {"".join(link_items) if link_items else "<span class=\"muted\">No links yet</span>"}
              </nav>
              <details>
                <summary>Plots</summary>
                <div class="plot-grid">
                  {"".join(plot_items) if plot_items else "<div class=\"muted\">Plots not found yet.</div>"}
                </div>
              </details>
            </article>
            """
        )

    page = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Mixed Ops Experiment Dashboard</title>
    <style>
      :root {{
        --bg: #f6f1ea;
        --bg-alt: #efe7dd;
        --ink: #1d1c1a;
        --muted: #6c635a;
        --accent: #1f6b4f;
        --accent-2: #d3753b;
        --card: #ffffff;
        --stroke: rgba(29, 28, 26, 0.12);
        --shadow: 0 16px 40px rgba(29, 28, 26, 0.12);
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        font-family: "Space Grotesk", "Segoe UI", "Trebuchet MS", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at 10% 10%, #f0e3d4 0%, transparent 50%),
          radial-gradient(circle at 80% 0%, #f7d8c4 0%, transparent 45%),
          linear-gradient(140deg, var(--bg), var(--bg-alt));
        min-height: 100vh;
      }}

      header.hero {{
        padding: 48px 8vw 24px;
      }}

      header.hero h1 {{
        margin: 0 0 8px;
        font-size: clamp(28px, 4vw, 44px);
        letter-spacing: -0.02em;
      }}

      header.hero p {{
        margin: 0;
        color: var(--muted);
        max-width: 780px;
        line-height: 1.5;
      }}

      main {{
        padding: 24px 8vw 64px;
      }}

      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
      }}

      .card {{
        background: var(--card);
        border: 1px solid var(--stroke);
        border-radius: 18px;
        padding: 20px;
        box-shadow: var(--shadow);
        animation: rise 450ms ease forwards;
        opacity: 0;
      }}

      .card:nth-child(2) {{ animation-delay: 60ms; }}
      .card:nth-child(3) {{ animation-delay: 120ms; }}
      .card:nth-child(4) {{ animation-delay: 180ms; }}
      .card:nth-child(5) {{ animation-delay: 240ms; }}
      .card:nth-child(6) {{ animation-delay: 300ms; }}

      @keyframes rise {{
        from {{
          opacity: 0;
          transform: translateY(12px);
        }}
        to {{
          opacity: 1;
          transform: translateY(0);
        }}
      }}

      .card h2 {{
        margin: 0 0 6px;
        font-size: 20px;
        text-transform: capitalize;
      }}

      .meta {{
        font-size: 13px;
        color: var(--muted);
      }}

      .stats {{
        margin: 16px 0;
        display: grid;
        gap: 10px;
      }}

      .stats div {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        padding: 10px 12px;
        border-radius: 12px;
        background: #f7f3ee;
      }}

      .stats span {{
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }}

      .stats strong {{
        font-size: 18px;
        color: var(--accent);
      }}

      .stats em {{
        font-size: 12px;
        color: var(--accent-2);
        margin-left: 8px;
      }}

      .links {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 12px;
      }}

      .links a {{
        text-decoration: none;
        color: var(--ink);
        border: 1px solid var(--stroke);
        padding: 6px 12px;
        border-radius: 999px;
        background: #faf7f3;
        font-size: 13px;
      }}

      details {{
        border-top: 1px dashed var(--stroke);
        padding-top: 10px;
      }}

      summary {{
        cursor: pointer;
        font-weight: 600;
        color: var(--accent);
        list-style: none;
      }}

      summary::-webkit-details-marker {{
        display: none;
      }}

      .plot-grid {{
        margin-top: 16px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
      }}

      .plot {{
        margin: 0;
        padding: 10px;
        border-radius: 12px;
        background: #f4ede5;
        border: 1px solid var(--stroke);
      }}

      .plot img {{
        width: 100%;
        height: auto;
        display: block;
        border-radius: 8px;
      }}

      .plot figcaption {{
        margin-top: 6px;
        font-size: 12px;
        color: var(--muted);
      }}

      .missing {{
        border-style: dashed;
        background: #fdf9f5;
      }}

      .muted {{
        color: var(--muted);
        font-size: 13px;
      }}
    </style>
  </head>
  <body>
    <header class="hero">
      <h1>Mixed Ops Experiments</h1>
      <p>
        Consolidated view for the six concept steering runs. Each card links to the batch HTML,
        stats, and plots inside the latest batch folder.
      </p>
      <p class="muted">Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </header>
    <main>
      <div class="grid">
        {"".join(cards)}
      </div>
    </main>
  </body>
</html>
"""

    out_path.write_text(page, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mixed-ops evals for multiple concepts.")
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Only rebuild the dashboard without running experiments.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Do not write the dashboard after running experiments.",
    )
    args = parser.parse_args()

    if args.dashboard_only:
        path = build_dashboard(PROJECTS)
        print(f"Dashboard written to {path}")
        return

    problems = generate_mixed_ops_problems(
        PROBLEM_COUNT,
        seed=PROBLEM_SEED,
        num_terms=NUM_TERMS,
        digits=DIGITS,
        operators=["+", "-", "*"],
        exclude_trailing_zero=False,
    )
    prompts = build_mixed_ops_prompts(problems)

    for project in PROJECTS:
        run_project(project, problems, prompts)

    if not args.no_dashboard:
        path = build_dashboard(PROJECTS)
        print(f"Dashboard written to {path}")


if __name__ == "__main__":
    main()
