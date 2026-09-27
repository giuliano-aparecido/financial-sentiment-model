# Project Overview

## What this is

Dataset generation, LoRA fine-tuning, and evaluation scripts for the LLM
that powers [`financial-sentiment-api`](https://github.com/giuliano-aparecido/financial-sentiment-api)'s
news-sentiment reasoning. See [`README.md`](README.md) for the full
training pipeline walkthrough, serving options, and required secrets -
this file covers structure and the "why" behind the less obvious design
choices.

## Architecture

```
generate_synthetic_dataset.py   Offline, deterministic synthetic data
generate_real_dataset.py         Real headlines/prices/fundamentals,
                                Gemini-written reasoning text
fusion_rules.py                  fuse(): the deterministic BUY/SELL/HOLD
                                rule, ported byte-identically to
                                financial-sentiment-api as
                                app/services/fusion.py
convert_existing_to_taskb.py     One-off migration script (old single-
                                task dataset shape -> two-stage Task
                                A/B shape)
calibrate_reaction_thresholds.py  Tunes the price-move thresholds
                                news_reaction labels are derived from
build_review_list.py, spot_check.py, generate_valuation_report.py
                                Manual dataset-quality review tooling
bump_model_version.py            Bumps the -vN suffix across every
                                reference in the repo at once
notebooks/train/
  gpu/                            unsloth/bitsandbytes-based training +
                                eval (T4 default) - train_model.py and
                                evaluate_model.py also run standalone,
                                outside a notebook (RunPod/any GPU box)
  tpu/                            Separate plain transformers/peft/trl
                                implementation (unsloth/bitsandbytes are
                                CUDA-only) for Colab's free TPU v5e-1
serve/
  notebooks/                       Colab/Kaggle/RunPod + ngrok tunnel
                                serving path (default, free)
  modal/                           Modal scale-to-zero serverless GPU
                                serving path (serve.py)
tests/                          Pure-logic unit tests (dataset
                                generation helpers, recommendation-
                                accuracy detector) - the only part of
                                this repo with real CI-runnable tests
docs/                           Design-decision writeups for larger
                                changes (Task A/B redesign, etc.)
```

## Key design decisions

- **Two independently-generated datasets, deliberately mixed.**
  `generate_synthetic_dataset.py` gives clean, verified-by-construction
  labels but a fixed vocabulary a model could memorize;
  `generate_real_dataset.py` gives real headlines with noisier *proxy*
  labels derived from actual subsequent price movement. Both write the
  same schema so `train_model.py` can concatenate them directly.
- **Real data is undersampled to balance classes, never duplicated** —
  duplicating would teach the model to memorize repeated rows.
- **Validation is held out by ticker AND by template**, not a random row
  split, since a random split would leak near-duplicate phrasing into
  validation and produce a flattering, meaningless accuracy number.
- **Direction accuracy is the metric, not loss.** The first training run's
  loss curves looked like overfitting but weren't measuring what actually
  mattered.
- **Completion-only loss masking needs exact tokenizer-matched markers**,
  including incidental whitespace — a mismatched marker silently masks
  100% of the training signal rather than erroring loudly.
- **Real data's `reasoning` text is LLM-written and headline-grounded**,
  not a fixed template — an earlier fixed template caused the model to
  reproduce a memorized answer per ticker instead of reading the headline.
  Falls back to the template on an API failure.
- **The model reasons over data, not just headlines, and answers the user
  directly.** `market_data`/`valuation`/`earnings` are in the prompt and
  `answer` is in the output; valuation is always a deterministic Graham
  Number computed in code, never LLM-generated, so the model learns to
  read a real number instead of hallucinating one. A block that couldn't
  be fetched renders as exactly `Data unavailable.` in both training data
  and production.
- **GPU and TPU paths are separate implementations, not a device-name
  swap.** `unsloth`/`bitsandbytes` are CUDA-only, so `notebooks/train/tpu/`
  reimplements training on plain `transformers`/`peft`/`trl` in bf16, no
  quantization - see README's "GPU vs TPU" section for the practical
  consequences (smaller `MODEL_REGISTRY`, `-tpu`-suffixed model names,
  not yet run end-to-end on real hardware).
- **The eval prompt/output schema must stay byte-identical to
  `financial-sentiment-api`'s copy**, since that's what it's actually
  served against in production - see CONTRIBUTING.md's sync rule. A drift
  here trains one prompt and serves another.
