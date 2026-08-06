# Financial Sentiment Model — Training Pipeline

Dataset generation, LoRA fine-tuning, and evaluation scripts for the LLM
that powers [`financial-sentiment-api`](https://github.com/GiulianoAparecido/financial-sentiment-api)'s
news-sentiment reasoning. Everything here is designed to be pasted into
Google Colab cells and run on Colab's free GPU tier — there's no local
training path, and no CI, since nothing here runs outside a notebook.

Built as a portfolio/curriculum project — hardened and documented for the
practice of doing it properly, not because it needs to scale.

## What this produces

A LoRA-fine-tuned instruction model (Llama 3.2 3B by default; a few other
open models are supported via `MODEL_REGISTRY` in `gpu/train_model.py` /
`tpu/train_model.py`) that reads a stock ticker, an optional user
question, and a block of recent news headlines, and outputs structured
JSON:

```json
{
  "impacted_stocks": [
    {
      "ticker": "AAPL",
      "reasoning": "...",
      "direction": "BULLISH",
      "confidence": 0.91
    }
  ]
}
```

`financial-sentiment-api`'s `app/services/inference.py` calls the resulting
model via a Hugging Face Inference endpoint and parses this exact shape —
if you change the output schema or the prompt structure here, that repo
needs a matching change.

## Pipeline (run each of these as its own Colab cell, in order)

Steps 1-3 are hardware-agnostic and identical either way. Steps 4-5 branch
depending on which free Colab accelerator you're using — pick **one** of
`gpu/` or `tpu/`, not both, for a given training run.

1. **`!pip install -q yfinance httpx feedparser google-genai`** —
   dependencies for the real-data generator (step 3). The training script
   for whichever accelerator you pick installs its own dependencies at the
   top of that file, so nothing extra is needed for steps 4-5.
2. **`generate_synthetic_dataset.py`** — offline, deterministic, no
   dependencies beyond the standard library. Writes `dataset_train.jsonl` /
   `dataset_val.jsonl`. Takes a few seconds.
3. **`generate_real_dataset.py`** — pulls real historical headlines from
   Google News RSS and real subsequent price moves from `yfinance`, and
   derives BULLISH/BEARISH/NEUTRAL labels from what the stock actually did
   afterward. Writes `dataset_train_real.jsonl` / `dataset_val_real.jsonl`.
   Not deterministic (depends on what Google's index currently returns, and
   Gemini's reasoning text varies run to run for the same headline), and
   noticeably slower than step 2 — expect several minutes given the number
   of tickers and historical windows it scans, plus one Gemini call per
   kept headline; this is expected, not a hang. Requires a `GEMINI_API_KEY`
   secret — see below.
4. **`gpu/train_model.py`** (T4) or **`tpu/train_model.py`** (v5e-1) —
   loads the base model, adds a LoRA adapter, mixes both datasets from
   steps 2-3, fine-tunes with early stopping, and pushes the result to
   your Hugging Face account.
5. **`gpu/evaluate_model.py`** or **`tpu/evaluate_model.py`** (match
   whichever you used for step 4) — must run in the **same Colab session**
   immediately after step 4 (it reuses `model`/`tokenizer`/`alpaca_prompt`
   still in memory). Reports direction accuracy — not loss, see
   `docs/training-results-analysis.md` for why that distinction matters —
   split by dataset source and by class, plus a base-model (untrained)
   comparison so you know how much the fine-tune actually helped.

`evaluate_base_model_only.py` (in the matching `gpu/` or `tpu/` directory)
is a standalone fallback: if you need just the base-model comparison on
its own (e.g. the tuned pass already ran in an earlier session), it
reloads the pushed model straight from Hugging Face rather than requiring
the training cell's variables still in memory.

### GPU (`gpu/`) vs TPU (`tpu/`)

The two paths are **not** just a device-name swap. `unsloth` (fast LoRA
loading/training) and `bitsandbytes` (4-bit quantization) are both
CUDA-only — Colab's free TPU v5e-1 tier has no support for either, so
`tpu/`'s scripts are a separate implementation on plain `transformers` +
`peft` + `trl`, training in bf16 with no quantization instead.

Practical consequences:

- `tpu/`'s `MODEL_REGISTRY` only has working entries for `llama-3.2-3b`
  and `apertus-0.5b` — the default `llama-3.2-3b` repo is swapped to a
  non-quantized bf16 mirror. `apertus-8b`, `qwen-2.5-7b`, and `mistral-7b`
  are listed but blocked with a clear error if selected: bf16 with no
  quantization makes 7B/8B a tight-to-unsafe fit on a single v5e-1's 16GB
  HBM, and qwen/mistral have no confirmed non-quantized mirror.
- `tpu/train_model.py` installs no `unsloth`/`bitsandbytes` — just
  `transformers peft trl accelerate datasets` (plus whatever `torch_xla`
  build Colab's TPU runtime already ships).
- Both paths push an **adapter-only** model to the same naming scheme
  (`{HF_USER}/{model}-financial-reasoner-v3`), except the TPU path adds a
  `-tpu` suffix so a TPU run never overwrites a GPU-trained adapter at the
  same name, or vice versa.
- The TPU path hasn't been run end-to-end on real TPU hardware yet — the
  GPU path is the proven one. If you hit an issue running `tpu/`'s
  scripts, that's expected first-run friction, not necessarily something
  you did wrong.

## Required Colab Secrets (environment variables)

Set these once via the key icon in Colab's left sidebar — **Secrets**, not
hardcoded anywhere in these scripts. Grant each secret access to the
notebook when the permission prompt appears; if that grant hasn't happened
yet, `userdata.get(...)` will block waiting for it, which matters if you're
running the whole notebook unattended via "Run all."

| Name | What it is |
|---|---|
| `HF_TOKEN` | A Hugging Face **write**-access token, used to push the fine-tuned model. |
| `HF_USER` | Your Hugging Face username, used to build the target repo name (`{HF_USER}/{model}-financial-reasoner-v3`). |
| `GEMINI_API_KEY` | A free key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey), used by `generate_real_dataset.py` to write headline-grounded `reasoning` text (`gemini-3.5-flash-lite` — cost for the whole real dataset is well under $1). |

None of these values are ever written into any file in this repo — that's
the whole point of pulling them from Colab/Kaggle Secrets instead.

## Key design decisions

- **Two independently-generated datasets, deliberately mixed, not one.**
  `generate_synthetic_dataset.py` produces hand-authored, template-based
  examples with verified-by-construction labels — good coverage and clean
  signal, but a fixed vocabulary a model could in principle memorize.
  `generate_real_dataset.py` produces real headlines with *proxy* labels
  (derived from actual subsequent price movement, not a human judgment) —
  noisier, but real language the synthetic templates can't fully capture.
  Both write the identical `{ticker, user_query, news, output}` schema so
  either `train_model.py` (`gpu/` or `tpu/`) can concatenate them with no
  reconciliation step.
- **Real data is undersampled to balance classes, never duplicated**, to
  avoid teaching the model to memorize repeated rows. The cost is fewer
  total real-data rows; see `generate_real_dataset.py`'s docstring for the
  reasoning, and `docs/dataset-fix-plan.md` for why the raw pool (tickers x
  lookback window) needs to be grown rather than the sampling relaxed.
- **Validation is held out by ticker AND by template**, not a random row
  split — with this much template repetition, a random split would leak
  near-duplicate phrasing into validation and produce a flattering,
  meaningless accuracy number.
- **Loss is not the metric that matters here.** The first full training
  run's loss curves looked like classic overfitting, but the real
  explanation was more specific — see `docs/training-results-analysis.md`.
  Evaluate on direction accuracy (`evaluate_model.py`), not loss.
- **Completion-only loss masking has a real tokenizer gotcha.** The
  instruction/response markers (passed to unsloth's
  `train_on_responses_only` in `gpu/train_model.py`, or to trl's
  `DataCollatorForCompletionOnlyLM` in `tpu/train_model.py`) must match
  the *exact* tokenization of the prompt template, including incidental
  whitespace — a mismatched marker silently masks 100% of the training
  signal rather than erroring loudly. See the comment above that call in
  either file for the specific bug this project hit.
- **The trained model has a measured NEUTRAL-hedging bias** on real,
  ambiguous headlines (below-random-chance accuracy on real validation data
  in the first trained model). `docs/dataset-fix-plan.md` documents the
  diagnosis and the dataset changes made in response.
- **Real data's `reasoning` text is LLM-written and headline-grounded, not
  a fixed template.** The original template (`ticker moved X% -> DIRECTION`)
  never referenced the headline at all — confirmed live as the cause of a
  second failure mode (a *different* trained model reproducing an
  identical memorized answer per ticker regardless of what headline it was
  given, rather than reading it). `generate_real_dataset.py`'s
  `generate_grounded_reasoning` calls Gemini (`gemini-3.5-flash-lite`) with
  the headline and the already-decided direction, explicitly telling it not
  to reference the future price move it doesn't have — direction/confidence
  stay purely proxy-derived, only the reasoning text changes. Falls back to
  the old template on an API failure so one bad call doesn't abort a
  multi-hundred-row run.

## docs/

- `llm-training-primer.md` — a from-zero explanation of what every part of
  `gpu/train_model.py` does, for anyone reading this without an ML
  background. Written against the GPU/unsloth path; `tpu/train_model.py`
  swaps the same conceptual steps onto a different toolchain (see the
  "GPU vs TPU" section above).
- `training-results-analysis.md` — why the first training run's loss
  curves were misleading, and what to measure instead.
- `dataset-fix-plan.md` — the diagnosis and fix plan for the real-data
  accuracy gap found during evaluation.
