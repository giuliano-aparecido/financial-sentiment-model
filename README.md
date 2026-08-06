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
open models are supported via `MODEL_REGISTRY` in `train_model.py`) that
reads a stock ticker, an optional user question, and a block of recent news
headlines, and outputs structured JSON:

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

1. **`!pip install -q yfinance httpx feedparser`** — dependencies for the
   real-data generator (step 3). `train_model.py` installs its own
   dependencies (`unsloth`, `trl`, `peft`, `accelerate`, `bitsandbytes`) at
   the top of that file, so nothing extra is needed for steps 4-5.
2. **`generate_synthetic_dataset.py`** — offline, deterministic, no
   dependencies beyond the standard library. Writes `dataset_train.jsonl` /
   `dataset_val.jsonl`. Takes a few seconds.
3. **`generate_real_dataset.py`** — pulls real historical headlines from
   Google News RSS and real subsequent price moves from `yfinance`, and
   derives BULLISH/BEARISH/NEUTRAL labels from what the stock actually did
   afterward. Writes `dataset_train_real.jsonl` / `dataset_val_real.jsonl`.
   Not deterministic (depends on what Google's index currently returns),
   and noticeably slower than step 2 — expect several minutes given the
   number of tickers and historical windows it scans; this is expected, not
   a hang.
4. **`train_model.py`** — loads the base model, adds a LoRA adapter, mixes
   both datasets from steps 2-3, fine-tunes with early stopping, and pushes
   the merged result to your Hugging Face account.
5. **`evaluate_model.py`** — must run in the **same Colab session**
   immediately after step 4 (it reuses `model`/`tokenizer`/`alpaca_prompt`
   still in memory). Reports direction accuracy — not loss, see
   `docs/training-results-analysis.md` for why that distinction matters —
   split by dataset source and by class, plus a base-model (untrained)
   comparison so you know how much the fine-tune actually helped.

`evaluate_base_model_only.py` is a standalone fallback: if you need just
the base-model comparison on its own (e.g. the tuned pass already ran in an
earlier session), it reloads the pushed model straight from Hugging Face
rather than requiring the training cell's variables still in memory.

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

Neither value is ever written into any file in this repo — that's the
whole point of pulling them from Colab Secrets instead.

## Key design decisions

- **Two independently-generated datasets, deliberately mixed, not one.**
  `generate_synthetic_dataset.py` produces hand-authored, template-based
  examples with verified-by-construction labels — good coverage and clean
  signal, but a fixed vocabulary a model could in principle memorize.
  `generate_real_dataset.py` produces real headlines with *proxy* labels
  (derived from actual subsequent price movement, not a human judgment) —
  noisier, but real language the synthetic templates can't fully capture.
  Both write the identical `{ticker, user_query, news, output}` schema so
  `train_model.py` can concatenate them with no reconciliation step.
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
  instruction/response markers passed to `train_on_responses_only` must
  match the *exact* tokenization of the prompt template, including
  incidental whitespace — a mismatched marker silently masks 100% of the
  training signal rather than erroring loudly. See the comment above that
  call in `train_model.py` for the specific bug this project hit.
- **The trained model has a measured NEUTRAL-hedging bias** on real,
  ambiguous headlines (below-random-chance accuracy on real validation data
  in the first trained model). `docs/dataset-fix-plan.md` documents the
  diagnosis and the dataset changes made in response.

## docs/

- `llm-training-primer.md` — a from-zero explanation of what every part of
  `train_model.py` does, for anyone reading this without an ML background.
- `training-results-analysis.md` — why the first training run's loss
  curves were misleading, and what to measure instead.
- `dataset-fix-plan.md` — the diagnosis and fix plan for the real-data
  accuracy gap found during evaluation.
