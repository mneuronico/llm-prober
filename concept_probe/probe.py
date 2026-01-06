import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from .config import ConceptSpec, resolve_config
from .logger import JsonlLogger
from .modeling import ModelBundle, apply_chat, attention_mask_from_ids
from .steering import MultiLayerSteererLayerwise
from .utils import deep_merge, ensure_dir, json_dump, jsonl_to_pretty, now_tag, safe_slug, set_seed
from .visuals import plot_score_hist, plot_sweep, render_batch_heatmap, render_token_heatmap
from .console import ConsoleLogger

try:
    from scipy.stats import ttest_ind
except Exception:
    ttest_ind = None


def normed_np(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cohen_d_np(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = x.size, y.size
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return float((x.mean() - y.mean()) / (np.sqrt(pooled) + 1e-12))


def pick_layer(num_layers: int, layer_idx: Optional[int], layer_frac: float) -> int:
    if layer_idx is not None:
        return int(layer_idx)
    return int(layer_frac * (num_layers - 1))


def select_layers(selection: Dict[str, Any], best_layer: int, num_layers: int) -> List[int]:
    mode = selection.get("mode", "best")
    if mode == "best":
        return [best_layer]
    if mode == "window":
        radius = int(selection.get("window_radius", 0))
        lo = max(0, best_layer - radius)
        hi = min(num_layers - 1, best_layer + radius)
        return list(range(lo, hi + 1))
    if mode == "explicit":
        return list(map(int, selection.get("layers", [])))
    if mode == "all":
        return list(range(num_layers))
    raise ValueError(f"Unknown layer selection mode: {mode}")


def _decode_tokens(tokenizer, token_ids: Iterable[int]) -> List[str]:
    return [tokenizer.decode([int(t)], skip_special_tokens=False) for t in token_ids]


def _prompt_slug(text: str, max_words: int = 8, max_len: int = 48) -> str:
    words = text.strip().split()
    snippet = " ".join(words[:max_words])
    slug = safe_slug(snippet)[:max_len]
    return slug or "prompt"


def _alpha_label(alpha: float, unit: str) -> str:
    if unit == "sigma":
        label = f"sigma{alpha:+.2f}"
    else:
        label = f"alpha{alpha:+.4f}"
    label = label.replace("+", "p").replace("-", "m").replace(".", "p")
    return safe_slug(label)


@torch.no_grad()
def forward_hidden_states_all_layers(model, input_ids: torch.Tensor) -> List[torch.Tensor]:
    attn = attention_mask_from_ids(input_ids).to(model.device)
    out = model(
        input_ids=input_ids.to(model.device),
        attention_mask=attn,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hs = out.hidden_states
    num_layers = len(hs) - 1
    return [hs[l + 1][0].detach().float().cpu() for l in range(num_layers)]


def reps_from_hs_layers(
    hs_layers: List[torch.Tensor],
    prompt_len: Optional[int],
    mode: str,
    last_k: int,
) -> np.ndarray:
    num_layers = len(hs_layers)
    dim = int(hs_layers[0].shape[1])
    reps = np.zeros((num_layers, dim), dtype=np.float32)

    for l in range(num_layers):
        hs = hs_layers[l]
        total = hs.shape[0]

        if mode == "sequence_last":
            reps[l] = hs[-1].numpy()
        elif mode == "sequence_last_k_mean":
            k = min(last_k, total)
            reps[l] = hs[-k:].mean(dim=0).numpy()
        elif mode == "sequence_all_mean":
            reps[l] = hs.mean(dim=0).numpy()
        elif mode == "assistant_last":
            reps[l] = hs[-1].numpy()
        elif mode == "assistant_last_k_mean":
            if prompt_len is None or prompt_len >= total:
                k = min(last_k, total)
                reps[l] = hs[-k:].mean(dim=0).numpy()
            else:
                span = hs[prompt_len:]
                if span.shape[0] == 0:
                    reps[l] = hs.mean(dim=0).numpy()
                else:
                    k = min(last_k, span.shape[0])
                    reps[l] = span[-k:].mean(dim=0).numpy()
        elif mode == "assistant_all_mean":
            if prompt_len is None or prompt_len >= total:
                reps[l] = hs.mean(dim=0).numpy()
            else:
                span = hs[prompt_len:]
                reps[l] = (span.mean(dim=0) if span.shape[0] > 0 else hs.mean(dim=0)).numpy()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return reps


def token_scores_from_hs_layers(
    hs_layers: List[torch.Tensor],
    concept_vectors: np.ndarray,
    layer_indices: List[int],
    aggregate: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if not layer_indices:
        raise ValueError("layer_indices must be non-empty")
    total = int(hs_layers[0].shape[0])
    scores_per_layer = np.zeros((total, len(layer_indices)), dtype=np.float32)

    for i, li in enumerate(layer_indices):
        hs = hs_layers[li]
        v = torch.tensor(concept_vectors[li], dtype=hs.dtype)
        scores_per_layer[:, i] = (hs @ v).numpy()

    if aggregate == "mean":
        scores_agg = scores_per_layer.mean(axis=1)
    elif aggregate == "sum":
        scores_agg = scores_per_layer.sum(axis=1)
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")

    return scores_per_layer, scores_agg


@torch.no_grad()
def generate_once(
    model,
    tokenizer,
    system: str,
    user: str,
    max_new_tokens: int,
    greedy: bool,
    temperature: Optional[float],
    top_p: Optional[float],
) -> Tuple[torch.Tensor, int]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    input_ids = apply_chat(tokenizer, messages, add_generation_prompt=True)
    prompt_len = int(input_ids.shape[-1])
    attn = attention_mask_from_ids(input_ids).to(model.device)

    gen_ids = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        temperature=temperature if not greedy else None,
        top_p=top_p if not greedy else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return gen_ids[0].detach().cpu(), prompt_len


@dataclass
class ConceptProbe:
    model_bundle: ModelBundle
    config: Dict[str, Any]
    concept: ConceptSpec
    run_dir: str
    logger: JsonlLogger
    console: Optional[ConsoleLogger] = None

    concept_vectors: Optional[np.ndarray] = None
    best_layer: Optional[int] = None
    proj_std_best_layer: Optional[float] = None
    sweep_d: Optional[np.ndarray] = None
    sweep_p: Optional[np.ndarray] = None

    def train(self) -> "ConceptProbe":
        if ttest_ind is None:
            self._warn("scipy not available; sweep_p will be NaN.")
        set_seed(int(self.config["random"]["seed"]))
        ensure_dir(self.run_dir)

        config_path = os.path.join(self.run_dir, "config.json")
        concept_path = os.path.join(self.run_dir, "concept.json")
        json_dump(config_path, self.config)
        json_dump(concept_path, self.concept.to_dict())

        self.logger.log("config", {"config": self.config, "concept": self.concept.to_dict()})

        train_prompts = list(self.config["prompts"]["train_questions"])
        readout = self.config["readout"]
        train_cfg = self.config["training"]
        eval_cfg = self.config["evaluation"]
        out_cfg = self.config["output"]
        plot_cfg = self.config["plots"]

        tokenizer = self.model_bundle.tokenizer
        model = self.model_bundle.model
        num_layers = self.model_bundle.num_layers

        self._status(f"Training concept '{self.concept.name}' with model {self.model_bundle.model_id}")
        self._status("[Phase 1/4] Generating training completions")
        train_pos_ids, train_neg_ids = [], []
        train_pos_plens, train_neg_plens = [], []

        for i, q in enumerate(train_prompts):
            ids_pos, plen_pos = generate_once(
                model,
                tokenizer,
                self.concept.pos_system,
                q,
                train_cfg["train_max_new_tokens"],
                train_cfg["train_greedy"],
                temperature=train_cfg.get("train_temperature", 0.7),
                top_p=train_cfg.get("train_top_p", 0.9),
            )
            ids_neg, plen_neg = generate_once(
                model,
                tokenizer,
                self.concept.neg_system,
                q,
                train_cfg["train_max_new_tokens"],
                train_cfg["train_greedy"],
                temperature=train_cfg.get("train_temperature", 0.7),
                top_p=train_cfg.get("train_top_p", 0.9),
            )
            train_pos_ids.append(ids_pos)
            train_neg_ids.append(ids_neg)
            train_pos_plens.append(plen_pos)
            train_neg_plens.append(plen_neg)

            pos_completion = tokenizer.decode(ids_pos[plen_pos:], skip_special_tokens=True)
            neg_completion = tokenizer.decode(ids_neg[plen_neg:], skip_special_tokens=True)
            self.logger.log(
                "train_gen",
                {
                    "i": i,
                    "question": q,
                    "pos_label": self.concept.pos_label,
                    "neg_label": self.concept.neg_label,
                    "pos_system": self.concept.pos_system,
                    "neg_system": self.concept.neg_system,
                    "pos_text": tokenizer.decode(ids_pos, skip_special_tokens=False),
                    "neg_text": tokenizer.decode(ids_neg, skip_special_tokens=False),
                    "pos_completion": pos_completion,
                    "neg_completion": neg_completion,
                },
            )

        self._status("[Phase 2/4] Forward passes for training reps")
        reps_pos = np.stack(
            [
                reps_from_hs_layers(
                    forward_hidden_states_all_layers(model, train_pos_ids[i].unsqueeze(0)),
                    prompt_len=train_pos_plens[i],
                    mode=readout["train_mode"],
                    last_k=readout["train_last_k"],
                )
                for i in range(len(train_prompts))
            ],
            axis=0,
        ).astype(np.float32)

        reps_neg = np.stack(
            [
                reps_from_hs_layers(
                    forward_hidden_states_all_layers(model, train_neg_ids[i].unsqueeze(0)),
                    prompt_len=train_neg_plens[i],
                    mode=readout["train_mode"],
                    last_k=readout["train_last_k"],
                )
                for i in range(len(train_prompts))
            ],
            axis=0,
        ).astype(np.float32)

        num_layers_check = reps_pos.shape[1]
        if num_layers_check != num_layers:
            raise ValueError("Layer count mismatch between model and reps.")

        concept_vectors = np.zeros((num_layers, reps_pos.shape[2]), dtype=np.float32)
        for l in range(num_layers):
            mu_pos = reps_pos[:, l, :].mean(axis=0)
            mu_neg = reps_neg[:, l, :].mean(axis=0)
            concept_vectors[l] = normed_np(mu_pos - mu_neg)

        eval_pos_scores_mat = np.zeros((0, num_layers), dtype=np.float32)
        eval_neg_scores_mat = np.zeros((0, num_layers), dtype=np.float32)
        sweep_d = np.zeros((num_layers,), dtype=np.float32)
        sweep_p = np.full((num_layers,), np.nan, dtype=np.float32)
        eval_pos_mean = np.zeros((num_layers,), dtype=np.float32)
        eval_neg_mean = np.zeros((num_layers,), dtype=np.float32)
        eval_pos_std = np.zeros((num_layers,), dtype=np.float32)
        eval_neg_std = np.zeros((num_layers,), dtype=np.float32)

        eval_prompts = list(self.config["prompts"]["eval_questions"])
        did_eval = False
        eval_mode = eval_cfg.get("eval_mode", "auto")
        if eval_cfg.get("do_eval", True):
            if eval_mode == "auto":
                if len(self.concept.eval_pos_texts) > 0 and len(self.concept.eval_neg_texts) > 0:
                    eval_mode = "read"
                else:
                    eval_mode = "generate"

            if eval_mode == "read":
                eval_pos_texts = list(self.concept.eval_pos_texts)
                eval_neg_texts = list(self.concept.eval_neg_texts)
                eval_system = eval_cfg.get("eval_system", self.config["prompts"]["neutral_system"])
                if len(eval_pos_texts) > 0 and len(eval_neg_texts) > 0:
                    self._status("[Phase 3/4] Reading eval texts")
                    eval_pos_reps = np.stack(
                        [
                            reps_from_hs_layers(
                                forward_hidden_states_all_layers(
                                    model,
                                    apply_chat(
                                        tokenizer,
                                        [{"role": "system", "content": eval_system}, {"role": "user", "content": s}],
                                        add_generation_prompt=False,
                                    ),
                                ),
                                prompt_len=None,
                                mode=readout["read_mode"],
                                last_k=readout["read_last_k"],
                            )
                            for s in eval_pos_texts
                        ],
                        axis=0,
                    ).astype(np.float32)

                    eval_neg_reps = np.stack(
                        [
                            reps_from_hs_layers(
                                forward_hidden_states_all_layers(
                                    model,
                                    apply_chat(
                                        tokenizer,
                                        [{"role": "system", "content": eval_system}, {"role": "user", "content": s}],
                                        add_generation_prompt=False,
                                    ),
                                ),
                                prompt_len=None,
                                mode=readout["read_mode"],
                                last_k=readout["read_last_k"],
                            )
                            for s in eval_neg_texts
                        ],
                        axis=0,
                    ).astype(np.float32)

                    eval_pos_scores_mat = np.zeros((eval_pos_reps.shape[0], num_layers), dtype=np.float32)
                    eval_neg_scores_mat = np.zeros((eval_neg_reps.shape[0], num_layers), dtype=np.float32)

                    for l in range(num_layers):
                        v = concept_vectors[l]
                        pos_scores = eval_pos_reps[:, l, :] @ v
                        neg_scores = eval_neg_reps[:, l, :] @ v
                        eval_pos_scores_mat[:, l] = pos_scores
                        eval_neg_scores_mat[:, l] = neg_scores
                        eval_pos_mean[l] = pos_scores.mean()
                        eval_neg_mean[l] = neg_scores.mean()
                        eval_pos_std[l] = pos_scores.std(ddof=1)
                        eval_neg_std[l] = neg_scores.std(ddof=1)
                        sweep_d[l] = cohen_d_np(pos_scores, neg_scores)
                        if ttest_ind is not None:
                            _, p = ttest_ind(pos_scores, neg_scores, equal_var=False)
                            sweep_p[l] = float(p)
                    did_eval = True
                else:
                    self._warn("eval_mode=read but eval_pos_texts/eval_neg_texts are empty.")
            else:
                if len(eval_prompts) > 0:
                    self._status("[Phase 3/4] Generating eval completions")
                    eval_pos_ids, eval_neg_ids = [], []
                    eval_pos_plens, eval_neg_plens = [], []

                    for i, q in enumerate(eval_prompts):
                        ids_pos, plen_pos = generate_once(
                            model,
                            tokenizer,
                            self.concept.pos_system,
                            q,
                            eval_cfg["eval_max_new_tokens"],
                            eval_cfg["eval_greedy"],
                            temperature=eval_cfg.get("eval_temperature", 0.7),
                            top_p=eval_cfg.get("eval_top_p", 0.9),
                        )
                        ids_neg, plen_neg = generate_once(
                            model,
                            tokenizer,
                            self.concept.neg_system,
                            q,
                            eval_cfg["eval_max_new_tokens"],
                            eval_cfg["eval_greedy"],
                            temperature=eval_cfg.get("eval_temperature", 0.7),
                            top_p=eval_cfg.get("eval_top_p", 0.9),
                        )
                        eval_pos_ids.append(ids_pos)
                        eval_neg_ids.append(ids_neg)
                        eval_pos_plens.append(plen_pos)
                        eval_neg_plens.append(plen_neg)

                        self.logger.log(
                            "eval_gen",
                            {
                                "i": i,
                                "question": q,
                                "pos_text": tokenizer.decode(ids_pos, skip_special_tokens=False),
                                "neg_text": tokenizer.decode(ids_neg, skip_special_tokens=False),
                                "pos_completion": tokenizer.decode(ids_pos[plen_pos:], skip_special_tokens=True),
                                "neg_completion": tokenizer.decode(ids_neg[plen_neg:], skip_special_tokens=True),
                            },
                        )

                    eval_pos_reps = np.stack(
                        [
                            reps_from_hs_layers(
                                forward_hidden_states_all_layers(model, eval_pos_ids[i].unsqueeze(0)),
                                prompt_len=eval_pos_plens[i],
                                mode=readout["train_mode"],
                                last_k=readout["train_last_k"],
                            )
                            for i in range(len(eval_prompts))
                        ],
                        axis=0,
                    ).astype(np.float32)

                    eval_neg_reps = np.stack(
                        [
                            reps_from_hs_layers(
                                forward_hidden_states_all_layers(model, eval_neg_ids[i].unsqueeze(0)),
                                prompt_len=eval_neg_plens[i],
                                mode=readout["train_mode"],
                                last_k=readout["train_last_k"],
                            )
                            for i in range(len(eval_prompts))
                        ],
                        axis=0,
                    ).astype(np.float32)

                    eval_pos_scores_mat = np.zeros((eval_pos_reps.shape[0], num_layers), dtype=np.float32)
                    eval_neg_scores_mat = np.zeros((eval_neg_reps.shape[0], num_layers), dtype=np.float32)

                    for l in range(num_layers):
                        v = concept_vectors[l]
                        pos_scores = eval_pos_reps[:, l, :] @ v
                        neg_scores = eval_neg_reps[:, l, :] @ v
                        eval_pos_scores_mat[:, l] = pos_scores
                        eval_neg_scores_mat[:, l] = neg_scores
                        eval_pos_mean[l] = pos_scores.mean()
                        eval_neg_mean[l] = neg_scores.mean()
                        eval_pos_std[l] = pos_scores.std(ddof=1)
                        eval_neg_std[l] = neg_scores.std(ddof=1)
                        sweep_d[l] = cohen_d_np(pos_scores, neg_scores)
                        if ttest_ind is not None:
                            _, p = ttest_ind(pos_scores, neg_scores, equal_var=False)
                            sweep_p[l] = float(p)
                    did_eval = True

        if not did_eval:
            reason = "evaluation disabled"
            if eval_cfg.get("do_eval", True):
                if eval_mode == "read":
                    reason = "missing eval_pos_texts/eval_neg_texts"
                elif len(eval_prompts) == 0:
                    reason = "empty eval_questions"
            self._warn(reason)
            self.logger.log("eval_skipped", {"reason": reason})

        if train_cfg["do_layer_sweep"] and did_eval:
            if train_cfg["sweep_select"] == "effect_size":
                best_layer = int(np.argmax(sweep_d))
            else:
                best_layer = int(np.nanargmin(sweep_p))
        else:
            best_layer = pick_layer(num_layers, train_cfg.get("layer_idx"), train_cfg.get("layer_frac", 0.75))

        if eval_pos_scores_mat.size > 0:
            combined = np.concatenate(
                [eval_pos_scores_mat[:, best_layer], eval_neg_scores_mat[:, best_layer]], axis=0
            )
            proj_std = float(np.std(combined, ddof=1) + 1e-12)
        else:
            proj_std = float("nan")

        self.concept_vectors = concept_vectors
        self.best_layer = best_layer
        self.proj_std_best_layer = proj_std
        self.sweep_d = sweep_d
        self.sweep_p = sweep_p

        self._status("[Phase 4/4] Sweep + artifact saving")
        self.logger.log(
            "sweep_full",
            {
                "best_layer": best_layer,
                "best_d": float(sweep_d[best_layer]),
                "best_p": None if np.isnan(sweep_p[best_layer]) else float(sweep_p[best_layer]),
                "sweep_d": sweep_d.tolist(),
                "sweep_p": [None if np.isnan(x) else float(x) for x in sweep_p],
                "eval_pos_mean": eval_pos_mean.tolist(),
                "eval_neg_mean": eval_neg_mean.tolist(),
                "eval_pos_std": eval_pos_std.tolist(),
                "eval_neg_std": eval_neg_std.tolist(),
                "proj_std_best_layer": proj_std,
            },
        )

        plots_dir = os.path.join(self.run_dir, "plots")
        ensure_dir(plots_dir)
        if plot_cfg.get("enabled", True) and eval_pos_scores_mat.size > 0:
            if plot_cfg.get("sweep_plot", True):
                ok = plot_sweep(sweep_d, sweep_p, os.path.join(plots_dir, "sweep.png"))
                self.logger.log("plot_sweep", {"ok": ok})
            if plot_cfg.get("score_hist", True):
                ok = plot_score_hist(
                    eval_pos_scores_mat[:, best_layer],
                    eval_neg_scores_mat[:, best_layer],
                    os.path.join(plots_dir, "score_hist.png"),
                )
                self.logger.log("plot_score_hist", {"ok": ok})

        tensors = {
            "best_layer": np.array([best_layer], dtype=np.int32),
            "best_v": concept_vectors[best_layer].astype(np.float32),
            "concept_vectors": concept_vectors.astype(np.float32),
            "sweep_d": sweep_d.astype(np.float32),
            "sweep_p": sweep_p.astype(np.float32),
            "eval_pos_mean": eval_pos_mean.astype(np.float32),
            "eval_neg_mean": eval_neg_mean.astype(np.float32),
            "eval_pos_std": eval_pos_std.astype(np.float32),
            "eval_neg_std": eval_neg_std.astype(np.float32),
            "eval_pos_scores_mat": eval_pos_scores_mat.astype(np.float32),
            "eval_neg_scores_mat": eval_neg_scores_mat.astype(np.float32),
        }
        if out_cfg.get("save_reps_to_npz", True):
            tensors.update(
                {
                    "train_pos_reps": reps_pos.astype(np.float32),
                    "train_neg_reps": reps_neg.astype(np.float32),
                }
            )
        npz_path = os.path.join(self.run_dir, "tensors.npz")
        np.savez_compressed(npz_path, **tensors)

        metrics = {
            "best_layer": best_layer,
            "best_d": float(sweep_d[best_layer]),
            "best_p": None if np.isnan(sweep_p[best_layer]) else float(sweep_p[best_layer]),
            "proj_std_best_layer": proj_std,
            "num_layers": num_layers,
            "n_train": len(train_prompts),
            "n_eval": len(eval_prompts),
        }
        json_dump(os.path.join(self.run_dir, "metrics.json"), metrics)
        self.logger.log("artifacts_saved", {"npz_path": npz_path})
        if self.config["output"].get("pretty_json_log", False):
            pretty_path = os.path.join(self.run_dir, "log.pretty.json")
            jsonl_to_pretty(os.path.join(self.run_dir, "log.jsonl"), pretty_path)
        self._status(f"Saved artifacts to {self.run_dir}")
        return self

    def _require_trained(self) -> None:
        if self.concept_vectors is None or self.best_layer is None:
            raise ValueError("ConceptProbe is not trained or loaded.")

    def _status(self, message: str) -> None:
        if self.console is not None:
            self.console.info(message)

    def _warn(self, message: str) -> None:
        if self.console is not None:
            self.console.warn(message)
        self.logger.log("warning", {"message": message})

    def _layer_selection(self, override: Optional[Dict[str, Any]] = None) -> List[int]:
        self._require_trained()
        selection = dict(self.config["evaluation"]["layer_selection"])
        if override:
            selection = deep_merge(selection, override)
        return select_layers(selection, int(self.best_layer), self.model_bundle.num_layers)

    def score_texts(
        self,
        texts: List[str],
        system_prompt: Optional[str] = None,
        layer_selection: Optional[Dict[str, Any]] = None,
        output_subdir: str = "scores",
        save_html: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        self._require_trained()
        tokenizer = self.model_bundle.tokenizer
        model = self.model_bundle.model
        aggregate = self.config["evaluation"]["score_aggregate"]
        layers = self._layer_selection(layer_selection)
        sys_prompt = system_prompt or self.config["prompts"]["neutral_system"]
        save_html = self.config["plots"].get("heatmap_html", True) if save_html is None else save_html

        out_dir = os.path.join(self.run_dir, output_subdir)
        ensure_dir(out_dir)
        results = []
        batch_entries = []

        for i, text in enumerate(texts):
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}]
            input_ids = apply_chat(tokenizer, messages, add_generation_prompt=False)
            hs_layers = forward_hidden_states_all_layers(model, input_ids)
            scores_per_layer, scores_agg = token_scores_from_hs_layers(
                hs_layers, self.concept_vectors, layers, aggregate=aggregate
            )
            token_ids = input_ids[0].detach().cpu().numpy().astype(np.int32)
            tokens = _decode_tokens(tokenizer, token_ids)

            slug = _prompt_slug(text)
            base = f"text_{i:03d}_{slug}"
            npz_path = os.path.join(out_dir, f"{base}.npz")
            np.savez_compressed(
                npz_path,
                token_ids=token_ids,
                scores_per_layer=scores_per_layer,
                scores_agg=scores_agg,
                layer_indices=np.array(layers, dtype=np.int32),
            )

            if save_html:
                html_path = os.path.join(out_dir, f"{base}.html")
                render_token_heatmap(tokens, scores_agg.tolist(), html_path, title=slug)
                batch_entries.append((f"text {i:03d}: {slug}", tokens, scores_agg.tolist()))
            else:
                html_path = None

            rec = {
                "text": text,
                "system_prompt": sys_prompt,
                "layers": layers,
                "npz_path": npz_path,
                "html_path": html_path,
            }
            results.append(rec)
            self.logger.log("score_text", rec)

        if save_html and batch_entries:
            batch_path = os.path.join(out_dir, "batch_texts.html")
            render_batch_heatmap(batch_entries, batch_path, title="Batch text scores")
            self.logger.log("score_text_batch", {"html_path": batch_path, "count": len(batch_entries)})

        return results

    def score_prompts(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        greedy: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        layer_selection: Optional[Dict[str, Any]] = None,
        output_subdir: str = "scores",
        alphas: Optional[List[float]] = None,
        alpha_unit: str = "raw",
        steer_layers: Optional[Union[str, List[int]]] = None,
        steer_window_radius: Optional[int] = None,
        steer_distribute: Optional[bool] = None,
        save_html: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        self._require_trained()
        tokenizer = self.model_bundle.tokenizer
        model = self.model_bundle.model
        aggregate = self.config["evaluation"]["score_aggregate"]
        layers = self._layer_selection(layer_selection)
        sys_prompt = system_prompt or self.config["steering"]["steer_system"]
        save_html = self.config["plots"].get("heatmap_html", True) if save_html is None else save_html

        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config["steering"]["steer_max_new_tokens"]
        greedy = greedy if greedy is not None else (not self.config["steering"]["steer_sampling"])
        temperature = temperature if temperature is not None else self.config["steering"]["steer_temperature"]
        top_p = top_p if top_p is not None else self.config["steering"]["steer_top_p"]

        if alphas is None:
            alphas = [0.0]

        out_dir = os.path.join(self.run_dir, output_subdir)
        ensure_dir(out_dir)
        results = []
        batch_entries = []

        if steer_layers is None:
            steer_layers = self.config["steering"]["steer_layers"]
        if steer_window_radius is None:
            steer_window_radius = self.config["steering"]["steer_window_radius"]
        if steer_distribute is None:
            steer_distribute = self.config["steering"]["steer_distribute"]

        if steer_layers == "probe":
            steer_layer_list = [int(self.best_layer)]
        elif steer_layers == "window":
            lo = max(0, int(self.best_layer) - int(steer_window_radius))
            hi = min(self.model_bundle.num_layers - 1, int(self.best_layer) + int(steer_window_radius))
            steer_layer_list = list(range(lo, hi + 1))
        elif isinstance(steer_layers, list):
            steer_layer_list = list(map(int, steer_layers))
        else:
            steer_layer_list = list(map(int, steer_layers))

        for i, prompt in enumerate(prompts):
            prompt_slug = _prompt_slug(prompt)
            for j, alpha in enumerate(alphas):
                alpha_val = float(alpha)
                if alpha_unit == "sigma":
                    if self.proj_std_best_layer is None or np.isnan(self.proj_std_best_layer):
                        raise ValueError("proj_std_best_layer is not available for sigma scaling.")
                    alpha_val = alpha_val * float(self.proj_std_best_layer)

                messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
                input_ids = apply_chat(tokenizer, messages, add_generation_prompt=True).to(model.device)
                attn = attention_mask_from_ids(input_ids).to(model.device)
                prompt_len = int(input_ids.shape[-1])

                with MultiLayerSteererLayerwise(
                    model, steer_layer_list, self.concept_vectors, alpha_val, distribute=steer_distribute
                ):
                    gen_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_new_tokens=max_new_tokens,
                        do_sample=not greedy,
                        temperature=temperature if not greedy else None,
                        top_p=top_p if not greedy else None,
                        pad_token_id=tokenizer.eos_token_id,
                    )[0].detach().cpu()

                hs_layers = forward_hidden_states_all_layers(model, gen_ids.unsqueeze(0))
                scores_per_layer, scores_agg = token_scores_from_hs_layers(
                    hs_layers, self.concept_vectors, layers, aggregate=aggregate
                )

                token_ids = gen_ids.numpy().astype(np.int32)
                tokens = _decode_tokens(tokenizer, token_ids)
                completion_ids = token_ids[prompt_len:]
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

                alpha_label = _alpha_label(float(alpha), alpha_unit)
                base = f"prompt_{i:03d}_{prompt_slug}_{alpha_label}"
                npz_path = os.path.join(out_dir, f"{base}.npz")
                np.savez_compressed(
                    npz_path,
                    token_ids=token_ids,
                    prompt_len=np.array([prompt_len], dtype=np.int32),
                    scores_per_layer=scores_per_layer,
                    scores_agg=scores_agg,
                    layer_indices=np.array(layers, dtype=np.int32),
                )

                if save_html:
                    html_path = os.path.join(out_dir, f"{base}.html")
                    title = f"prompt {i:03d} | {prompt_slug} | {alpha_label}"
                    render_token_heatmap(tokens, scores_agg.tolist(), html_path, title=title)
                    batch_entries.append((title, tokens, scores_agg.tolist()))
                else:
                    html_path = None

                rec = {
                    "prompt": prompt,
                    "system_prompt": sys_prompt,
                    "alpha": float(alpha),
                    "alpha_unit": alpha_unit,
                    "alpha_value": float(alpha_val),
                    "steer_layers": steer_layer_list,
                    "completion": completion_text,
                    "prompt_len": prompt_len,
                    "layers": layers,
                    "npz_path": npz_path,
                    "html_path": html_path,
                }
                results.append(rec)
                self.logger.log("score_prompt", rec)

        if save_html and batch_entries:
            batch_path = os.path.join(out_dir, "batch_prompts.html")
            render_batch_heatmap(batch_entries, batch_path, title="Batch prompt scores")
            self.logger.log("score_prompt_batch", {"html_path": batch_path, "count": len(batch_entries)})

        return results

    @classmethod
    def load(cls, run_dir: str, model_bundle: ModelBundle) -> "ConceptProbe":
        config_path = os.path.join(run_dir, "config.json")
        concept_path = os.path.join(run_dir, "concept.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.loads(f.read())
        with open(concept_path, "r", encoding="utf-8") as f:
            c = ConceptSpec.from_config({"concept": json.loads(f.read())})

        logger = JsonlLogger(os.path.join(run_dir, "log.jsonl"))
        obj = cls(model_bundle=model_bundle, config=cfg, concept=c, run_dir=run_dir, logger=logger)

        tensors = np.load(os.path.join(run_dir, "tensors.npz"))
        obj.concept_vectors = tensors["concept_vectors"]
        obj.best_layer = int(tensors["best_layer"][0])
        obj.sweep_d = tensors["sweep_d"]
        obj.sweep_p = tensors["sweep_p"]
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.loads(f.read())
                obj.proj_std_best_layer = metrics.get("proj_std_best_layer")
        return obj


class ProbeWorkspace:
    def __init__(
        self,
        root_dir: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        defaults_path: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        cfg = resolve_config(config_overrides, defaults_path=defaults_path)
        if model_id:
            cfg["model"]["model_id"] = model_id
        if root_dir:
            cfg["output"]["root_dir"] = root_dir
        self.config = cfg
        self.root_dir = cfg["output"]["root_dir"]
        ensure_dir(self.root_dir)
        self.console = ConsoleLogger(cfg["output"].get("console", False))
        self.console.info(f"Loading model {cfg['model']['model_id']}...")
        self.model_bundle = ModelBundle.load(cfg["model"])
        self.console.info(f"Model loaded ({self.model_bundle.num_layers} layers).")

    def train_concept(
        self,
        concept: Optional[ConceptSpec] = None,
        concept_overrides: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> ConceptProbe:
        cfg = deep_merge(self.config, config_overrides or {})
        if concept is None:
            concept = ConceptSpec.from_config(cfg, concept_overrides)
        elif concept_overrides:
            concept = ConceptSpec.from_config({"concept": concept.to_dict()}, concept_overrides)

        run_tag = run_name or now_tag()
        concept_dir = os.path.join(self.root_dir, safe_slug(concept.name), run_tag)
        ensure_dir(concept_dir)
        log_path = os.path.join(concept_dir, "log.jsonl")
        logger = JsonlLogger(log_path)

        probe = ConceptProbe(
            model_bundle=self.model_bundle,
            config=cfg,
            concept=concept,
            run_dir=concept_dir,
            logger=logger,
            console=self.console,
        )
        return probe.train()
