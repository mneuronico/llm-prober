import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None


REPO_ROOT = Path(__file__).resolve().parents[2]


def _alpha_tag(alpha: float) -> str:
    return f"{alpha:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run_dir(path_arg: str) -> Path:
    p = Path(path_arg).resolve()
    if p.is_file() and p.name == "self_model_eval_items.json":
        return p.parent
    if p.is_dir():
        return p
    raise FileNotFoundError(f"Expected a run dir or self_model_eval_items.json path, got: {path_arg}")


def _resolve_npz(npz_path: str) -> Path:
    p = Path(npz_path)
    if p.is_absolute() and p.exists():
        return p
    p2 = (REPO_ROOT / p).resolve()
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not resolve npz path: {npz_path}")


def _completion_means_by_probe(npz_path: Path) -> Dict[str, float]:
    data = np.load(str(npz_path))
    scores = data.get("scores_agg")
    prompt_len_arr = data.get("prompt_len")
    probe_names_raw = data.get("probe_names")
    if scores is None or prompt_len_arr is None or probe_names_raw is None:
        raise ValueError(f"Missing required arrays in {npz_path}")

    scores = np.array(scores, dtype=float)
    prompt_len = int(np.array(prompt_len_arr, dtype=int).reshape(-1)[0])
    probe_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in probe_names_raw]

    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
        if not probe_names:
            probe_names = ["probe_0"]

    if scores.ndim != 2:
        raise ValueError(f"Unexpected scores_agg shape in {npz_path}: {scores.shape}")
    if len(probe_names) != scores.shape[0]:
        probe_names = [f"probe_{i}" for i in range(scores.shape[0])]

    out: Dict[str, float] = {}
    for i, name in enumerate(probe_names):
        row = scores[i]
        span = row[prompt_len:] if prompt_len < row.shape[0] else row
        out[name] = float(np.mean(span)) if span.size else float("nan")
    return out


def _plot_box(true_vals: List[float], rand_vals: List[float], *, title: str, out_path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(6.8, 4.6))
    try:
        plt.boxplot(
            [true_vals, rand_vals],
            tick_labels=["True pairs", "Random pairs"],
            showfliers=False,
        )
    except TypeError:
        plt.boxplot([true_vals, rand_vals], labels=["True pairs", "Random pairs"], showfliers=False)
    x_true = np.full(len(true_vals), 1.0) + np.random.uniform(-0.06, 0.06, size=len(true_vals))
    plt.scatter(x_true, true_vals, s=18, alpha=0.8)
    plt.ylabel("Absolute completion-mean distance")
    plt.title(title)
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_perm_means(
    true_mean: float,
    random_perm_means: List[float],
    *,
    title: str,
    out_path: Path,
) -> None:
    if plt is None:
        return
    arr = np.array(random_perm_means, dtype=float).reshape(-1)
    plt.figure(figsize=(6.8, 4.6))
    plt.hist(arr, bins=40, alpha=0.8)
    plt.axvline(true_mean, color="red", linestyle="--", linewidth=2, label=f"True mean={true_mean:.4f}")
    plt.xlabel("Mean distance per random pairing assignment")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _euclidean_distance(a: Dict[str, float], b: Dict[str, float], probe_names: List[str]) -> float:
    return float(math.sqrt(sum((float(a[p]) - float(b[p])) ** 2 for p in probe_names)))


def _one_sided_empirical_p(true_mean: float, random_means: List[float]) -> float:
    vals = np.array(random_means, dtype=float)
    return float((1 + np.sum(vals >= true_mean)) / (1 + vals.size))


def _run_analysis(
    items: List[Dict[str, object]],
    *,
    num_random: int,
    seed: int,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    probe_names_master: Optional[List[str]] = None

    for item in items:
        if not isinstance(item, dict):
            continue
        pair_id = item.get("pair_id")
        polarity = item.get("polarity")
        alpha = item.get("alpha")
        npz_path = item.get("npz_path")
        if not isinstance(pair_id, str) or not isinstance(polarity, str):
            continue
        if polarity not in {"positive", "negative"}:
            continue
        if not isinstance(alpha, (int, float)):
            continue
        if not isinstance(npz_path, str):
            continue
        means = _completion_means_by_probe(_resolve_npz(npz_path))
        if probe_names_master is None:
            probe_names_master = sorted(means.keys())
        rows.append(
            {
                "pair_id": pair_id,
                "polarity": polarity,
                "alpha": float(alpha),
                "means": means,
                "question_id": item.get("question_id"),
            }
        )

    if not rows:
        raise ValueError("No usable items found. Expected pair_id/polarity/alpha/npz_path entries.")
    if not probe_names_master:
        raise ValueError("No probe names found in npz files.")

    rng = np.random.default_rng(seed)
    alphas = sorted({float(r["alpha"]) for r in rows})
    out: Dict[str, object] = {"probe_names": probe_names_master, "num_random": num_random, "seed": seed, "alphas": {}}

    for alpha in alphas:
        rows_alpha = [r for r in rows if float(r["alpha"]) == float(alpha)]
        pos_rows = [r for r in rows_alpha if r["polarity"] == "positive"]
        neg_rows = [r for r in rows_alpha if r["polarity"] == "negative"]
        neg_by_pair = {r["pair_id"]: r for r in neg_rows}

        true_pairs: List[Tuple[Dict[str, object], Dict[str, object]]] = []
        for pr in pos_rows:
            nr = neg_by_pair.get(pr["pair_id"])
            if nr is None:
                continue
            true_pairs.append((pr, nr))
        if not true_pairs:
            continue

        true_dist_by_probe: Dict[str, List[float]] = {p: [] for p in probe_names_master}
        true_dist_joint: List[float] = []
        for pr, nr in true_pairs:
            for p in probe_names_master:
                dv = abs(float(pr["means"][p]) - float(nr["means"][p]))
                true_dist_by_probe[p].append(float(dv))
            true_dist_joint.append(
                _euclidean_distance(pr["means"], nr["means"], probe_names_master)
            )

        neg_pool = list(neg_rows)
        if len(neg_pool) != len(pos_rows):
            raise ValueError(
                f"Alpha {alpha}: positives ({len(pos_rows)}) and negatives ({len(neg_pool)}) differ. "
                "Random 1:1 pairing expects balanced sets."
            )

        rand_dist_by_probe: Dict[str, List[float]] = {p: [] for p in probe_names_master}
        rand_dist_joint: List[float] = []
        rand_mean_by_probe: Dict[str, List[float]] = {p: [] for p in probe_names_master}
        rand_mean_joint: List[float] = []

        for _ in range(num_random):
            perm = rng.permutation(len(neg_pool))
            probe_vals_this_perm: Dict[str, List[float]] = {p: [] for p in probe_names_master}
            joint_vals_this_perm: List[float] = []
            for i, pr in enumerate(pos_rows):
                nr = neg_pool[int(perm[i])]
                for p in probe_names_master:
                    dv = abs(float(pr["means"][p]) - float(nr["means"][p]))
                    probe_vals_this_perm[p].append(float(dv))
                    rand_dist_by_probe[p].append(float(dv))
                joint_d = _euclidean_distance(pr["means"], nr["means"], probe_names_master)
                joint_vals_this_perm.append(float(joint_d))
                rand_dist_joint.append(float(joint_d))
            for p in probe_names_master:
                rand_mean_by_probe[p].append(float(np.mean(probe_vals_this_perm[p])))
            rand_mean_joint.append(float(np.mean(joint_vals_this_perm)))

        alpha_block: Dict[str, object] = {"n_true_pairs": len(true_pairs), "per_probe": {}, "joint": {}}
        for p in probe_names_master:
            true_vals = true_dist_by_probe[p]
            rand_vals = rand_dist_by_probe[p]
            true_mean = float(np.mean(true_vals))
            rand_mean = float(np.mean(rand_vals))
            p_emp = _one_sided_empirical_p(true_mean, rand_mean_by_probe[p])
            mw_p = None
            mw_u = None
            if mannwhitneyu is not None:
                mw = mannwhitneyu(true_vals, rand_vals, alternative="greater")
                mw_u = float(mw.statistic)
                mw_p = float(mw.pvalue)
            alpha_block["per_probe"][p] = {
                "true_pair_distances": true_vals,
                "random_pair_distances": rand_vals,
                "true_mean_distance": true_mean,
                "random_mean_distance": rand_mean,
                "random_assignment_mean_distances": rand_mean_by_probe[p],
                "empirical_p_true_mean_ge_random_assignment_mean": p_emp,
                "mannwhitney_u_greater": mw_u,
                "mannwhitney_p_greater": mw_p,
            }

        true_mean_joint = float(np.mean(true_dist_joint))
        rand_pair_mean_joint = float(np.mean(rand_dist_joint))
        rand_assignment_means_joint = list(rand_mean_joint)
        p_emp_joint = _one_sided_empirical_p(true_mean_joint, rand_assignment_means_joint)
        mw_joint_p = None
        mw_joint_u = None
        if mannwhitneyu is not None:
            mw = mannwhitneyu(true_dist_joint, rand_dist_joint, alternative="greater")
            mw_joint_u = float(mw.statistic)
            mw_joint_p = float(mw.pvalue)
        alpha_block["joint"] = {
            "true_pair_distances": true_dist_joint,
            "random_pair_distances": rand_dist_joint,
            "true_mean_distance": true_mean_joint,
            "random_mean_distance": rand_pair_mean_joint,
            "random_assignment_mean_distances": rand_assignment_means_joint,
            "empirical_p_true_mean_ge_random_assignment_mean": p_emp_joint,
            "mannwhitney_u_greater": mw_joint_u,
            "mannwhitney_p_greater": mw_joint_p,
        }
        out["alphas"][str(alpha)] = alpha_block

    return out


def _save_plots(result: Dict[str, object], out_dir: Path) -> None:
    if plt is None:
        return
    probe_names = result.get("probe_names", [])
    alpha_blocks = result.get("alphas", {})
    if not isinstance(alpha_blocks, dict):
        return

    for alpha_key, alpha_block in alpha_blocks.items():
        alpha_val = float(alpha_key)
        tag = _alpha_tag(alpha_val)
        per_probe = alpha_block.get("per_probe", {})
        if isinstance(per_probe, dict):
            for probe in probe_names:
                block = per_probe.get(probe, {})
                if not isinstance(block, dict):
                    continue
                true_vals = block.get("true_pair_distances", [])
                rand_vals = block.get("random_pair_distances", [])
                rand_means = block.get("random_assignment_mean_distances", [])
                true_mean = block.get("true_mean_distance")
                if not true_vals or not rand_vals or true_mean is None:
                    continue
                _plot_box(
                    true_vals,
                    rand_vals,
                    title=f"{probe}: true vs random pair distances (alpha={alpha_val})",
                    out_path=out_dir / f"pair_distance_box_alpha_{tag}_{probe}.png",
                )
                _plot_perm_means(
                    float(true_mean),
                    rand_means,
                    title=f"{probe}: true mean vs random pairing means (alpha={alpha_val})",
                    out_path=out_dir / f"pair_distance_permmean_alpha_{tag}_{probe}.png",
                )

        joint = alpha_block.get("joint", {})
        if isinstance(joint, dict):
            true_vals = joint.get("true_pair_distances", [])
            rand_vals = joint.get("random_pair_distances", [])
            rand_means = joint.get("random_assignment_mean_distances", [])
            true_mean = joint.get("true_mean_distance")
            if true_vals and rand_vals and true_mean is not None:
                _plot_box(
                    true_vals,
                    rand_vals,
                    title=f"joint probes: true vs random pair distances (alpha={alpha_val})",
                    out_path=out_dir / f"pair_distance_box_alpha_{tag}_joint.png",
                )
                _plot_perm_means(
                    float(true_mean),
                    rand_means,
                    title=f"joint probes: true mean vs random pairing means (alpha={alpha_val})",
                    out_path=out_dir / f"pair_distance_permmean_alpha_{tag}_joint.png",
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare true pair completion-mean distances against random pairings."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to run dir containing self_model_eval_items.json, or the JSON file itself.",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=5000,
        help="Number of random one-to-one pairing assignments.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-name",
        default="pair_distance_vs_random.json",
        help="Summary JSON filename written in run dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    items_path = run_dir / "self_model_eval_items.json"
    if not items_path.exists():
        raise FileNotFoundError(f"Missing self_model_eval_items.json in {run_dir}")
    payload = _load_json(items_path)
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("self_model_eval_items.json missing items list")

    result = _run_analysis(items, num_random=int(args.num_random), seed=int(args.seed))
    out_path = run_dir / args.output_name
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_plots(result, run_dir)
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
