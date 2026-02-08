# concept-probe

A small, config-driven library for training, evaluating, and using concept probes in open-weights LLMs. It supports:
- Training concept directions from two behaviors (pos vs neg).
- Layer sweep metrics (effect size + p-values).
- Token-level scoring for read or generated text.
- Steering with learned concept vectors.
- Structured outputs (logs, plots, tensors, HTML heatmaps).

This README is the full documentation: quickstart, configuration, usage, outputs, and best practices.

---

## Installation

Install from a GitHub repo:

```bash
pip install git+https://github.com/<you>/<repo>.git
```

Minimal deps:

```bash
pip install torch transformers numpy
```

Optional but recommended:

```bash
pip install scipy matplotlib bitsandbytes
```

Notes:
- `scipy` is required for p-values in the sweep (otherwise p-values are NaN).
- `matplotlib` is required for sweep/hist plots.
- `bitsandbytes` is required for 4-bit loading. If unavailable, the library falls back to non-4bit loading with a warning.

---

## Quickstart (lowest friction)

Only define the concept. Everything else uses defaults.

```python
from concept_probe import ConceptSpec, ProbeWorkspace

SAD_SYSTEM = "You are a helpful assistant... (sad tone)"
HAPPY_SYSTEM = "You are a helpful assistant... (happy tone)"

SAD_SENTENCES = [
    "I feel a heavy emptiness that I can't explain.",
    "Nothing seems to matter much today.",
]
HAPPY_SENTENCES = [
    "I feel light and excited about today.",
    "I'm grateful and full of energy.",
]

workspace = ProbeWorkspace(model_id="meta-llama/Llama-3.2-3B-Instruct")

concept = ConceptSpec(
    name="sad_vs_happy",
    pos_label="sad",
    neg_label="happy",
    pos_system=SAD_SYSTEM,
    neg_system=HAPPY_SYSTEM,
    eval_pos_texts=SAD_SENTENCES,
    eval_neg_texts=HAPPY_SENTENCES,
)

probe = workspace.train_concept(concept)
probe.score_prompts(
    prompts=["Write a short paragraph about the ocean."],
    alphas=[0.0, 6.0, -6.0],
    alpha_unit="sigma",
)
```

Outputs are written under `outputs/<concept_name>/<timestamp>/`.

---

## Core Concepts

### `ProbeWorkspace`

Holds model + config and spawns concept runs.

```python
workspace = ProbeWorkspace(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    root_dir="outputs",                # optional
    config_overrides={...},            # optional
    defaults_path=".../defaults.json", # optional
)
```

You can also load an existing run directory (no retraining) by pointing `project_directory` at a previously created run folder:

```python
workspace = ProbeWorkspace(
    project_directory="outputs/sad_vs_happy/20260109_150734",
)

probe = workspace.get_probe(name="sad_vs_happy")
```

### `ConceptSpec`

Defines a concept and optional eval texts.

```python
concept = ConceptSpec(
    name="sad_vs_happy",
    pos_label="sad",
    neg_label="happy",
    pos_system="...sad system prompt...",
    neg_system="...happy system prompt...",
    eval_pos_texts=[...],  # optional
    eval_neg_texts=[...],  # optional
)
```

If `eval_pos_texts` and `eval_neg_texts` are provided, evaluation defaults to read mode. If not provided, evaluation defaults to generate mode.

### `ConceptProbe`

Returned by `workspace.train_concept(...)`. Provides:
- `score_texts(...)` for read-only scoring.
- `score_prompts(...)` for generation + optional steering.

---

## Prompt Formats (Text or Conversation)

Most places that accept a prompt can take either:

- A plain string ("text prompt")
- A conversation, represented as a list of dicts: `[{"role": "system"|"user"|"assistant", "content": "..."}, ...]`

Supported locations:
- `prompts.train_questions` (training questions)
- `prompts.train_questions_pos` / `prompts.train_questions_neg` (opposed training questions)
- `prompts.eval_questions` (evaluation questions)
- `ConceptSpec.eval_pos_texts` / `ConceptSpec.eval_neg_texts` (read-mode eval texts)
- `ConceptProbe.score_prompts(prompts=...)`

### System prompt precedence

When a prompt is provided as a conversation:

- If the conversation contains a `system` message, that system is used.
    The system prompt passed elsewhere (e.g. `pos_system`, `neg_system`, `system_prompt=...`, or `evaluation.eval_system`) is not injected.
    A warning is logged to make this explicit.
- If the conversation contains no `system` message, the relevant system prompt is prepended as a system message.

Example conversation prompt:

```python
prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I feel stuck and anxious about my deadline."},
        {"role": "assistant", "content": "Tell me what the deadline is and what you have so far."},
        {"role": "user", "content": "Tomorrow morning, and my draft is half done."},
]

probe.score_prompts(prompts=[prompt], alphas=[0.0])
```

---

## Defaults and Configuration

All defaults live in `concept_probe/defaults.json`. You can override any field by passing `config_overrides` to `ProbeWorkspace` or `train_concept`.

Example override:

```python
workspace = ProbeWorkspace(
    config_overrides={
        "training": {"train_max_new_tokens": 128},
        "plots": {"heatmap_html": False},
    }
)
```

### Default Sections

- `model`: model id, dtype, 4-bit, device map, token env.
- `prompts`: training/eval questions and neutral system prompt.
- `readout`: how to compute representations for training and reading.
- `training`: generation params and sweep settings.
- `evaluation`: eval mode and layer selection.
- `steering`: steering defaults.
- `plots`: plots + HTML toggles.
- `output`: output folder + logging behavior.
- `random`: seed.

---

## Readout Modes

Readout modes define which hidden states are used to compute representations.

Supported modes:
- `assistant_all_mean`: mean over all assistant tokens (completion only).
- `assistant_last`: last assistant token.
- `assistant_last_k_mean`: mean over last `k` assistant tokens.
- `sequence_last`: last token of the full sequence.
- `sequence_last_k_mean`: mean over last `k` tokens of the full sequence.
- `sequence_all_mean`: mean over all tokens in the full sequence.

Defaults:
- Training: `assistant_all_mean`
- Reading: `sequence_all_mean`

`train_last_k` and `read_last_k` are only used for `*_last_k_mean`.

---

## Evaluation Modes

`evaluation.eval_mode` can be:
- `auto` (default): if concept eval texts exist, uses read mode; else generate mode.
- `read`: reads fixed texts with a neutral system prompt.
- `generate`: generates completions using pos/neg systems.

### Read Mode
Uses:
- `ConceptSpec.eval_pos_texts`
- `ConceptSpec.eval_neg_texts`
- `evaluation.eval_system` (default: neutral system)
- `readout.read_mode`

### Generate Mode
Uses:
- `prompts.eval_questions`
- `ConceptSpec.pos_system` and `neg_system`
- `readout.train_mode`

---

## Layer Selection

`evaluation.layer_selection` supports:
- `best`: best layer from sweep.
- `window`: a window around best layer (`window_radius`).
- `explicit`: explicit list in `layers`.
- `all`: all layers.

Used when scoring or plotting probe values.

---

## How Scores Are Computed

This section documents the exact score definitions used in plots, `.npz` files, and HTML heatmaps.

### Concept vectors (per layer)
For each layer `l`:
1. Run the model on training prompts and collect hidden states for all layers.
2. Convert hidden states into a single representation per layer using `readout.train_mode` (see Readout Modes).
3. Compute the mean vector for the positive and negative training sets at layer `l`.
4. Define the concept vector as a normalized difference:
   `v_l = normalize(mean_pos_l - mean_neg_l)`.

This yields one concept vector per layer (same dimensionality as the hidden state for that layer).

### `score_hist.png`
`score_hist.png` is a histogram of evaluation scores at a single layer:
1. During evaluation, compute per-layer representations for each eval item:
   - Read mode uses `readout.read_mode`.
   - Generate mode uses `readout.train_mode`.
2. For each layer `l`, compute a scalar score per item:
   `score_l = rep_l dot v_l`.
3. Choose `best_layer`:
   - If `training.do_layer_sweep` is true and eval data exists, pick the layer with the best sweep metric
     (`effect_size` uses max Cohen's d; `p_value` uses min p). If `training.best_layer_search`
     is set, the sweep is still computed for all layers but selection is restricted to the specified
     interval(s).
   - Otherwise use `training.layer_idx` or `training.layer_frac`.
4. Plot histograms of `eval_pos_scores_mat[:, best_layer]` and
   `eval_neg_scores_mat[:, best_layer]` (20 bins).

So `score_hist.png` always uses one layer: the selected `best_layer`.

### Token-level scores in `.npz` and HTML heatmaps
For `score_texts(...)` and `score_prompts(...)`:
1. Run the model and collect hidden states for every token at every layer.
2. Select layers using `evaluation.layer_selection`.
3. For each selected layer `l` and token `t`, compute:
   `score_{t,l} = h_{t,l} dot v_l`.
   These per-layer scores are saved as `scores_per_layer` with shape
   `(num_tokens, num_selected_layers)`.
4. Aggregate across selected layers using `evaluation.score_aggregate`:
   - `mean`: `scores_agg[t] = mean_l score_{t,l}`
   - `sum`: `scores_agg[t] = sum_l score_{t,l}`

`scores_agg` is what is shown in HTML heatmaps and saved in `.npz` files.
If `layer_selection.mode` is `best` (default), `scores_agg` is just the single
best-layer score. If `window`/`explicit`/`all`, `scores_agg` combines layers via
`score_aggregate`.

For multi-probe runs, the same computation happens per probe and `scores_agg` is
stacked into shape `(num_probes, num_tokens)`, which drives the HTML dropdown.

### Turn-level segmentation (save_segments)
You can optionally save **token scores segmented by role/turn** for chat-style prompts.
This is useful for multi-turn conversations where you want per-message averages,
or to cleanly separate prompt vs completion tokens.

Enable this by passing `save_segments=True` to `score_prompts(...)` (single-probe)
or `multi_probe_score_prompts(...)` (multi-probe). This writes an extra JSON file
next to each `.npz`:

- `prompt_..._segments.json` (single-probe)
- `prompt_..._segments.json` with `segments_by_probe` (multi-probe)

Segmentation behavior:
- Uses the same chat markers as the HTML heatmaps (e.g. Llama-3 `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`).
- **Special tokens and role markers are excluded**; only text tokens are kept.
- If `prompt_len` is known, each segment is split into `phase: "prompt"` and
  `phase: "completion"` when applicable.

Single-probe schema:

```json
{
  "prompt_len": 1234,
  "segments": [
    {
      "segment_index": 0,
      "message_index": 0,
      "role": "system|user|assistant",
      "phase": "prompt|completion",
      "token_indices": [ ... ],
      "tokens": [ ... ],
      "scores": [ ... ],
      "mean_score": 0.1234,
      "token_count": 57
    }
  ]
}
```

Multi-probe schema:

```json
{
  "prompt_len": 1234,
  "segments_by_probe": {
    "probe_name": [ ... same segment schema ... ]
  }
}
```

---

## Training Behavior

Training uses:
- `training.train_prompt_mode`
- `prompts.train_questions` (shared mode)
- `prompts.train_questions_pos` / `prompts.train_questions_neg` (opposed mode)
- `ConceptSpec.pos_system` / `neg_system`
- `readout.train_mode`
- `training.train_max_new_tokens`, `training.train_greedy`, etc

It computes:
- Per-layer concept vectors (`concept_vectors`)
- Sweep metrics (`sweep_d`, `sweep_p`)
- Best layer (effect size or p-value)
- Best-layer search intervals (if configured)
- Projection std at best layer (for sigma-scaled steering)

### Training prompt modes

`training.train_prompt_mode` controls how prompts are paired during training:
- `shared` (default): use `prompts.train_questions` for both sides; systems differ.
- `opposed`: use `prompts.train_questions_pos` and `prompts.train_questions_neg`; systems can be the same or different.

This enables training where the system prompt is fixed and only the user prompts differ.

### Best-layer search intervals (`training.best_layer_search`)

When `training.do_layer_sweep` is enabled and eval data exists, the library selects a single
`best_layer` from the sweep. By default it searches all layers. You can restrict this search
to one or more intervals without changing the sweep computation itself.

Key behavior:
- The sweep metrics (`sweep_d`, `sweep_p`) are still computed for every layer.
- Only the *selection* of `best_layer` is restricted to the specified interval(s).
- `plots/sweep.png` still shows all layers, but the search interval(s) are shaded and the chosen
  layer is marked with a dotted vertical line.

Accepted formats:
- `null` (default): search all layers.
- A single interval: `[lo, hi]`
- Multiple intervals: `[[lo, hi], [lo2, hi2], ...]`

Layer values:
- Integers are interpreted as layer indices (0-based, inclusive).
- Floats in `[0, 1]` are treated as proportions of depth, converted with
  `int(frac * (num_layers - 1))`. This matches `training.layer_frac` behavior.

Examples (28-layer model: layers `0..27`):
- `[4, 23]` searches layers 4 through 23 inclusive.
- `[[4, 10], [15, 20]]` searches two disjoint bands.
- `[0.15, 0.85]` maps to `[4, 22]` because `int(0.15 * 27) = 4` and `int(0.85 * 27) = 22`.

Config example:

```python
workspace = ProbeWorkspace(
    config_overrides={
        "training": {
            "best_layer_search": [[4, 10], [15, 20]],
            "sweep_select": "effect_size",
        }
    }
)
```

Additional details:
- Intervals are inclusive; reversed endpoints are swapped.
- Values are clamped to the valid layer range.
- Overlapping or adjacent intervals are merged internally.
- If `sweep_select="p_value"` and all p-values are NaN in the search interval, selection fails;
  use `sweep_select="effect_size"` or install `scipy`.

## Scoring API

### `score_texts(texts, ...)`
Scores a list of texts or conversations as read-only.

```python
probe.score_texts(
    texts=[
        "I feel okay.",
        [
            {"role": "user", "content": "I feel stuck."},
            {"role": "assistant", "content": "What feels hardest right now?"},
            {"role": "user", "content": "Starting."},
        ],
    ],
    system_prompt="You are a helpful assistant.",
)
```

Outputs:
- Each call writes to a fresh subfolder under `scores/` (see Output Structure).
- `.npz` per text with token ids and scores.
- `.html` per text (if enabled).
- `batch_texts.html` with a shared color scale.

### `score_prompts(prompts, ...)`
Generates completions, optionally steered, and scores all tokens.

```python
probe.score_prompts(
    prompts=["Write a paragraph about the ocean."],
    alphas=[0.0, 6.0, -6.0],
    alpha_unit="sigma",
    steer_layers="window",
)
```

Outputs:
- Each call writes to a fresh subfolder under `scores/` (see Output Structure).
- `.npz` per prompt/alpha.
- `.html` per prompt/alpha.
- `*_segments.json` per prompt/alpha (if `save_segments=True`).
- `batch_prompts.html` with shared scale.

Names include prompt snippets and alpha labels for clarity.

---

## Generic Auto-Graded Eval Pipeline

For evals that have:
- a prompt per item (or a shared prompt prefix + per-item variable text), and
- an expected answer for each item,

you can run a fully automated eval that:
- generates completions (optionally steered),
- computes correctness via a user-supplied evaluator,
- optionally rates coherence via Groq (warning-only on failure),
- writes `analysis/` with stats + plots for the batch.

Coherence ratings require `GROQ_API_KEY` (from `.env` or environment). If the API call fails or rate limits,
the eval still completes and only coherence-specific outputs are missing.

You can also call the coherence rater directly:

```python
from concept_probe import rate_batch_coherence

rate_batch_coherence(
    "outputs/bored_vs_interested/20260122_112208/new_math_eval_mixed_ops_5x2/batch_20260125_181957",
)
```

This is implemented in `concept_probe.coherence`.

### `run_scored_eval(...)`

```python
from concept_probe import ProbeWorkspace, run_scored_eval

def generate_item() -> dict:
    # Return a single item each time; the library handles how many to generate.
    i = 2
    return {"question": f"{i}+{i}=?", "expected": 2 * i}

def eval_answer(completion: str, item: dict) -> dict:
    # Customize parsing to your task. Return at least {"correct": bool}.
    expected = item["expected"]
    parsed = int("".join(ch for ch in completion if ch.isdigit()) or -1)
    return {"correct": parsed == expected, "parsed": parsed, "expected": expected}

workspace = ProbeWorkspace(project_directory="outputs/empathy_vs_detachment/20260109_150734")
probe = workspace.get_probe(name="empathy_vs_detachment")

run_scored_eval(
    probe,
    generator=generate_item,
    num_items=20,
    evaluator=eval_answer,
    prompt_prefix="Solve: ",      # or prompt_template="Solve: {question}"
    variable_key="question",      # field used with prompt_prefix
    output_subdir="scored_eval_custom",
    alphas=[-8, 0, 8],
    alpha_unit="sigma",
    steer_layers="window",
    steer_window_radius=2,
    max_new_tokens=128,
    rate_coherence=True,          # optional (warning-only on failure)
)
```

What it writes (inside the batch folder):
- `analysis/per_sample.json`: items + completions + correctness + score means
- `analysis/stats.json`: accuracy by alpha, score by alpha, correct vs incorrect
- `analysis/plots/*`: accuracy/score plots (and coherence plots if ratings exist)

### `run_multi_scored_eval(...)`

Same auto-graded eval pipeline, but **one generation per prompt** scored across multiple probes.
This is the eval counterpart to `multi_probe_score_prompts(...)` and is ideal for comparing
score distributions across multiple concepts on the same completions.

Example (no steering, shared items):

```python
import json
from concept_probe import ConceptProbe, run_multi_scored_eval
from concept_probe.modeling import ModelBundle

probe_runs = [
    "outputs/empathy_vs_detachment/20260109_150734",
    "outputs/bored_vs_interested/20260122_112208",
]

# Load a single model bundle for all probes.
with open(f"{probe_runs[0]}/config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
bundle = ModelBundle.load(cfg["model"])
probes = [ConceptProbe.load(run_dir=p, model_bundle=bundle) for p in probe_runs]

def generate_item() -> dict:
    return {"question": "2+2=?", "expected": 4}

def eval_answer(completion: str, item: dict) -> dict:
    return {"correct": "4" in completion}

run_multi_scored_eval(
    probes,
    generator=generate_item,
    num_items=20,
    evaluator=eval_answer,
    prompt_prefix="Solve: ",
    variable_key="question",
    output_root="outputs_multi",
    project_name="math_eval_multi_probe",
    output_subdir="eval",
    alphas=[0.0],              # no steering
    alpha_unit="sigma",
    max_new_tokens=128,
    rate_coherence=True,
)
```

Outputs are written under:

```
outputs_multi/<project_name>/<timestamp>/<output_subdir>/batch_.../analysis/
```

Multi-probe eval analysis differs slightly:
- **Accuracy is global** (shared across probes).
- **Score plots and stats are per-probe** (since scores differ by probe).

### `generate_multi_probe_report(...)`

Generate an interactive HTML report (bar charts + 2D/3D scatter + a simple classifier) from
a multi-probe eval batch.

```python
from concept_probe import generate_multi_probe_report

generate_multi_probe_report(
    "outputs_multi/social_iqa_multi_probe/20260127_154629/social_iqa_eval/batch_20260127_154629",
    title="SocialIQA Multi-Probe Report",
)
```

This writes `report.html` into the batch's `analysis/` directory. The classifier uses a 75/25
train/test split on the mean probe scores; p-values are included if `statsmodels` is installed.

#### Generator contract

`run_scored_eval` can build items in two ways:
- **Provide `items=[...]`**: you control the full list yourself.
- **Provide `generator=...`**: the generator is called `num_items` times (if set).

The generator can return:
- a single item (recommended), or
- a list of items (only when `num_items` is omitted).

If `num_items` is set and your generator returns a list, the call fails so you don't accidentally
over-generate.

If you need parameters (e.g., seeds, difficulty), pass them with `generator_kwargs` and they will
be forwarded on each generator call.

#### Prompt construction options
`run_scored_eval` can build prompts in several ways:
- Use `prompt_builder(item)` for full control (can return conversation format).
- Use `prompt_template="Instruction: {question}"` (formatted with item fields).
- Use `prompt_prefix="Instruction: "` + `variable_key="question"`.
- Use a prebuilt `item["prompt"]`.

At least one of these must succeed for each item. The prompt can be a string or a list of
`{"role": ..., "content": ...}` messages.

#### Required item fields

What each item must contain depends on how you build prompts and evaluate correctness:
- **Prompt content** is required in *some form*:
  - either `item["prompt"]`,
  - or fields used by `prompt_template` / `prompt_prefix`,
  - or handled by `prompt_builder`.
- **Expected answer** is required **if you do not pass a custom evaluator**:
  - By default, the library expects `item["expected"]`.
  - You can change this field name with `expected_key=...`.

Any extra fields you include in the item are preserved in `per_sample.json`
when `include_item=True`.

#### Evaluator contract
The evaluator is called as:

```python
def evaluator(completion: str, item: dict) -> dict:
    return {"correct": bool, ...}
```

Additional keys you return (e.g., `parsed`, `expected`) are included in `per_sample.json`.

If you don't supply an evaluator, the library uses a basic equality check:
- By default, `completion.strip()` is compared to `str(item[expected_key]).strip()`.
- If you pass `marker="ANSWER:"`, the evaluator extracts the substring
  **after the first occurrence** of the marker (or **before** it if you set
  `marker_position="before"`), then compares that to the expected answer.
- If the marker is provided but not found, the item is marked incorrect.

For most tasks you will want a custom evaluator, but marker extraction is often
enough for simple auto-graded formats.

---

## Batch Maintenance Utilities

These utilities operate on existing batches and auto-detect single-probe vs multi-probe analyses.

### `rehydrate_batch_analysis(batch_dir, ...)`
Rebuilds `analysis/` for an existing batch (recomputes stats + plots).
Optionally re-runs coherence rating first.

```python
from concept_probe import rehydrate_batch_analysis

rehydrate_batch_analysis(
    "outputs/empathy_vs_detachment/20260109_150734/mixed_ops_eval_5x2/batch_20260125_122612",
    rate_coherence=True,
)
```

### `aggregate_eval_batches(eval_dir, ...)`
Combines all batches under a single eval folder into a unified analysis folder.

```python
from concept_probe import aggregate_eval_batches

aggregate_eval_batches(
    "outputs/empathy_vs_detachment/20260109_150734/mixed_ops_eval_5x2",
    output_name="analysis_all",
)
```

---

## JSON-Driven Training (`train_concept_from_json`)

If you prefer a JSON spec, you can train a concept and run an eval sweep directly from a file.

```python
from concept_probe import train_concept_from_json

train_concept_from_json(
    "examples/configs/lying_vs_truthfulness.json",
    alphas=[-6, 0, 6],
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    root_dir="outputs",
)
```

Expected JSON shape (top-level keys):

```json
{
  "concept": {
    "name": "lying_vs_truthfulness",
    "pos_label": "lying",
    "neg_label": "truthful",
    "pos_system": "...",
    "neg_system": "..."
  },
  "prompts": {
    "train_questions": ["..."],
    "eval_questions": ["..."],
    "neutral_system": "You are a helpful assistant."
  }
}
```

Notes:
- Any extra fields in the JSON can be overridden via `config_overrides`.
- If `alphas` is omitted, it runs a single `alpha=0.0` sweep.

---

### `multi_probe_score_prompts(...)`
Generate each prompt once (or once per alpha if steering) and score the same generation with multiple probes.
This is useful when you want a shared set of activations that can be compared across probes.

Key behavior:
- Uses a single model forward pass per prompt/alpha, then scores all probes on the resulting tokens.
- Optionally steers using a single probe (choose which probe to steer with).
- Writes outputs to a dedicated multi-probe run directory (not inside any single-probe run).
- HTML heatmaps include a dropdown to switch between probes.

Example (no steering):

```python
import json
from concept_probe import ConceptProbe, multi_probe_score_prompts
from concept_probe.modeling import ModelBundle

probe_runs = [
    "outputs/fabrication_vs_truthfulness/20260115_180659",
    "outputs/lying_vs_truthfulness/20260116_095854",
]

# Load a single model bundle for all probes.
with open(f"{probe_runs[0]}/config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
bundle = ModelBundle.load(cfg["model"])
probes = [ConceptProbe.load(run_dir=p, model_bundle=bundle) for p in probe_runs]

results = multi_probe_score_prompts(
    probes,
    prompts=["Why does the Moon have phases?"],
    system_prompt="You are a helpful assistant.",
    output_root="outputs_multi",
    project_name="truthfulqa_multi_probe",
)
```

Example (steer with one probe and multiple alphas):

```python
results = multi_probe_score_prompts(
    probes,
    prompts=["Explain how vaccines work."],
    steer_probe="lying_vs_truthfulness",  # or index 1, or the ConceptProbe object
    alphas=[-5.0, 0.0, 5.0],
    alpha_unit="raw",
)
```

Notes:
- All probes must be trained on the same `model_id`.
- If `steer_probe` is `None`, steering is disabled and only a single alpha (`0.0`) is used.
- `scores_agg` in the `.npz` is shaped `(num_probes, num_tokens)`.

### Keeping outputs together (batch folders)

By default, each call to `score_prompts(...)` or `score_texts(...)` writes into a unique subfolder:

```
scores/
    batch_YYYYMMDD_HHMMSS/
        ...
```

This prevents batch HTML and per-item files from mixing across multiple scoring runs.

If you explicitly want to write directly into `scores/` (not recommended for repeated runs), pass `batch_subdir=""`.

---

## Steering

Steering is applied during generation:
- `alphas` are raw or sigma-scaled.
- `alpha_unit="sigma"` multiplies by projection std at best layer.

### Alpha units

`alpha_unit` controls how the numeric `alpha` values are interpreted:
- `raw`: use `alpha` directly (no normalization).
- `sigma`: use `alpha * proj_std_best_layer`, where `proj_std_best_layer` is the
  standard deviation of eval projections at the best layer.

Single-probe and multi-probe use the same rule; the only difference is that
multi-probe uses the `steer_probe`'s `proj_std_best_layer` when `alpha_unit="sigma"`.

Layer options:
- `steer_layers="probe"`: best layer only.
- `steer_layers="window"`: window around best layer.
- `steer_layers=[...]`: explicit list.

When steering across a window, each layer uses its own concept vector.

---

## Visualizations

HTML heatmaps:
- Tokens are grouped into system/user/assistant blocks.
- Special marker tokens are hidden.
- Each token shows its score on hover.
- A shared color scale is shown.

Batch HTML:
- `batch_texts.html` and `batch_prompts.html` compare multiple runs using a shared scale.
- Multi-probe batch HTML includes a probe dropdown to switch highlights.

Plots (if `matplotlib` installed):
- `plots/sweep.png`
- `plots/score_hist.png`

`sweep.png` details:
- Always plots every layer's sweep metrics.
- Shaded bands show `training.best_layer_search` intervals (when used).
- A dotted vertical line marks the chosen `best_layer`.

---

## Output Structure

Each run writes to:

```
outputs/
  <concept_name>/
    <timestamp>/
      config.json
      concept.json
      log.jsonl
      log.pretty.json
      metrics.json
      tensors.npz
      plots/
        sweep.png
        score_hist.png
      scores/
        batch_YYYYMMDD_HHMMSS/
          prompt_000_<slug>_<alpha>.npz
          prompt_000_<slug>_<alpha>.html
          batch_prompts.html
          text_000_<slug>.npz
          text_000_<slug>.html
          batch_texts.html
      <eval_name>/
        batch_YYYYMMDD_HHMMSS/
          prompt_000_<slug>_<alpha>.npz
          prompt_000_<slug>_<alpha>.html
          batch_prompts.html
          analysis/
            per_sample.json
            stats.json
            plots/
              accuracy_vs_alpha.png
              score_vs_alpha.png
              score_by_correctness.png
              score_by_correctness_by_alpha.png
              accuracy_vs_alpha_by_coherence.png
              coherence_counts_by_alpha.png
```

Example with multiple batches inside the same eval folder:

```
outputs/
  empathy_vs_detachment/
    20260109_150734/
      mixed_ops_eval_5x2/
        batch_20260125_122612/
          ...
        batch_20260125_153045/
          ...
        analysis_all/
          per_sample.json
          stats.json
          plots/
            accuracy_vs_alpha.png
            ...
```

Multi-probe runs write to a separate root:

```
outputs_multi/
  <project_name>/
    <timestamp>/
      config.json
      probes.json
      log.jsonl
      scores/
        batch_YYYYMMDD_HHMMSS/
          prompt_000_<slug>_<alpha>.npz
          prompt_000_<slug>_<alpha>.html
          batch_prompts.html
```

Multi-probe `.npz` contents:
- `token_ids`: full token ids for the prompt + completion.
- `prompt_len`: length of the prompt portion.
- `scores_agg`: array of shape `(num_probes, num_tokens)`.
- `probe_names`: list of probe names aligned with `scores_agg`.
- `prompt_..._segments.json` per prompt/alpha (if `save_segments=True`), with per-probe segments.

Key files:
- `config.json`: full resolved config.
- `concept.json`: concept definition.
- `metrics.json`: best layer, effect size, projection std, and best-layer search metadata.
- `tensors.npz`: concept vectors + sweep arrays.
- `log.jsonl`: event log.
- `log.pretty.json`: pretty log for reading.

---

## Logging and Console Output

The library logs to `log.jsonl` when `output.log_jsonl` is true. It also:
- writes `log.pretty.json` if both `output.log_jsonl` and `output.pretty_json_log` are true.
- prints progress to terminal if `output.console` is true.

Disable console output:

```python
workspace = ProbeWorkspace(config_overrides={"output": {"console": False}})
```

Train/eval events include assistant-only fields:
- `pos_completion`
- `neg_completion`

---

## Multi-Concept Training

You can train multiple concepts in a single workspace:

```python
for name, pos_sys, neg_sys in concepts:
    concept = ConceptSpec(name=name, pos_label="pos", neg_label="neg",
                          pos_system=pos_sys, neg_system=neg_sys)
    workspace.train_concept(concept)
```

Each concept gets its own folder under `outputs/`.

---

## Best Practices

- Use behaviorally specific system prompts.
- Provide eval texts to stabilize sweep selection.
- Keep training prompts general to avoid concept leakage.
- Use `alpha_unit="sigma"` for comparable steering across runs.
- Inspect `metrics.json` and sweep plots before trusting a probe.

---

## Troubleshooting

- If `sweep_p` is NaN: install `scipy`.
- If plots are missing: install `matplotlib` or disable plots.
- If 4-bit loading fails: disable `model.use_4bit` or install `bitsandbytes`.
