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
- `bitsandbytes` is required for 4-bit loading.

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

## Training Behavior

Training uses:
- `prompts.train_questions`
- `ConceptSpec.pos_system` / `neg_system`
- `readout.train_mode`
- `training.train_max_new_tokens`, `training.train_greedy`, etc

It computes:
- Per-layer concept vectors (`concept_vectors`)
- Sweep metrics (`sweep_d`, `sweep_p`)
- Best layer (effect size or p-value)
- Projection std at best layer (for sigma-scaled steering)

---

## Scoring API

### `score_texts(texts, ...)`
Scores a list of texts or conversations as read-only.

```python
probe.score_texts(
    texts=["I feel okay.", "I feel awful."],
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
- `batch_prompts.html` with shared scale.

Names include prompt snippets and alpha labels for clarity.

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

Plots (if `matplotlib` installed):
- `plots/sweep.png`
- `plots/score_hist.png`

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
```

Key files:
- `config.json`: full resolved config.
- `concept.json`: concept definition.
- `metrics.json`: best layer, effect size, projection std.
- `tensors.npz`: concept vectors + sweep arrays.
- `log.jsonl`: event log.
- `log.pretty.json`: pretty log for reading.

---

## Logging and Console Output

The library always logs to `log.jsonl`. It also:
- writes `log.pretty.json` if `output.pretty_json_log` is true.
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

---

## License

Add your license here.
