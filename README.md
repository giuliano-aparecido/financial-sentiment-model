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
open models are supported via `MODEL_REGISTRY` in `colab/train/gpu/train_model.py` /
`colab/train/tpu/train_model.py`), trained on TWO tasks rather than one -
the two-stage pipeline redesign: the model itself only ever reasons about
the unpredictable input (the news); a deterministic rule
(`fusion_rules.py`) decides BUY/SELL/HOLD, never the model.

**Task A** (`task_a_prompt`) reads a ticker, its recent 3-day price move,
and a block of recent news headlines, and classifies how the market has
reacted to that news:

```json
{"news_reaction": "overreaction_down"}
```

news_reaction is one of `good`/`bad`/`neutral`/`overreaction_down`/
`overreaction_up` - see `fusion_rules.py`'s own docstring for how this
combines with a numeric DCF valuation gap to produce a recommendation.

**Task B** (`task_b_prompt`) is given a ticker, an optional user question,
current market data/valuation/earnings, the news, AND a news_reaction +
Recommended Action that are ALREADY DECIDED (by `fusion_rules.fuse()`,
never by the model) - its only job is to explain that given recommendation
and answer the user's question consistently with it:

```json
{
  "reasoning": "...",
  "answer": "Yes, the current signals lean favorably enough that AAPL looks like a reasonable buy here."
}
```

`financial-sentiment-api`'s `app/services/inference.py` calls the model
over HTTP TWICE per request (a Colab/ngrok tunnel by default - see
"Serving the model" below for a Modal alternative) - once per task - and
parses these two exact shapes, then calls the SAME `fuse()` function
(ported byte-identically as `app/services/fusion.py`) to turn Task A's
classification into the recommendation Task B is given. If you change
either output schema or either prompt structure here, that repo needs a
matching change (see CONTRIBUTING.md's sync rule).

## Pipeline (run each of these as its own Colab cell, in order)

Steps 1-3 are hardware-agnostic and identical either way. Steps 4-5 branch
depending on which free Colab accelerator you're using — pick **one** of
`colab/train/gpu/` or `colab/train/tpu/`, not both, for a given training run.

1. **`!pip install -q yfinance httpx feedparser google-genai pandas`** —
   dependencies for the real-data generator (step 3; `pandas` is also a
   transitive `yfinance` dependency, usually already present). The training
   script for whichever accelerator you pick installs its own dependencies
   at the top of that file, so nothing extra is needed for steps 4-5.
2. **`generate_synthetic_dataset.py`** — offline, deterministic, no
   dependencies beyond the standard library. Writes `dataset_train.jsonl` /
   `dataset_val.jsonl`. Each row includes fabricated-but-internally-
   consistent `market_data`/`valuation`/`earnings` blocks (the valuation
   figure is a real Graham Number computation on the row's own synthetic
   price/EPS/book-value, never LLM-generated) alongside the news headlines.
   Takes a few seconds.
3. **`generate_real_dataset.py`** — pulls real historical headlines from
   Google News RSS, real subsequent price moves from `yfinance`, and real
   fundamentals/earnings **as of each headline's own publish date** (not
   today's figures — see `fetch_ticker_fundamentals_history`'s docstring for
   the specific, documented approximations where yfinance can't go back far
   enough), and derives BULLISH/BEARISH/NEUTRAL labels from what the stock
   actually did afterward. Writes `dataset_train_real.jsonl` /
   `dataset_val_real.jsonl`. Not deterministic (depends on what Google's
   index currently returns, and Gemini's reasoning/answer text varies run to
   run for the same headline), and noticeably slower than step 2 — expect
   several minutes given the number of tickers and historical windows it
   scans, plus one Gemini call per kept headline; this is expected, not a
   hang. Requires a `GEMINI_API_KEY` secret — see below.
4. **`colab/train/gpu/train_model.py`** (T4) or **`colab/train/tpu/train_model.py`** (v5e-1) —
   loads the base model, adds a LoRA adapter, mixes both datasets from
   steps 2-3, fine-tunes with early stopping, and pushes the result to
   your Hugging Face account.
5. **`colab/train/gpu/evaluate_model.py`** or **`colab/train/tpu/evaluate_model.py`** (match
   whichever you used for step 4) — self-contained: reuses
   `model`/`tokenizer`/`task_a_prompt`/`task_b_prompt` if run immediately after step 4 in
   the same session, or reloads the already-pushed model straight from
   Hugging Face if run in a fresh session (e.g. the previous one expired,
   or this same script crashed partway through on a prior run — training
   already finished and pushed by that point, so there's nothing to
   retrain, just this cell to re-run). Either way needs the
   `dataset_val*.jsonl` files present on disk. Reports direction accuracy
   — not loss, see `docs/training-results-analysis.md` for why that
   distinction matters — split by dataset source and by class, plus a
   base-model (untrained) comparison so you know how much the fine-tune
   actually helped.

`evaluate_base_model_only.py` (in the matching `colab/train/gpu/` or `colab/train/tpu/` directory)
is a standalone fallback for when you need *just* the base-model
comparison on its own (e.g. you already have `evaluate_model.py`'s
fine-tuned numbers from an earlier run and don't want to redo that pass) —
same self-contained reload-or-reuse behavior as `evaluate_model.py` above,
just skipping the tuned pass entirely.

### GPU (`colab/train/gpu/`) vs TPU (`colab/train/tpu/`)

The two paths are **not** just a device-name swap. `unsloth` (fast LoRA
loading/training) and `bitsandbytes` (4-bit quantization) are both
CUDA-only — Colab's free TPU v5e-1 tier has no support for either, so
`colab/train/tpu/`'s scripts are a separate implementation on plain `transformers` +
`peft` + `trl`, training in bf16 with no quantization instead.

Practical consequences:

- `colab/train/tpu/`'s `MODEL_REGISTRY` only has working entries for `llama-3.2-3b`
  and `apertus-0.5b` — the default `llama-3.2-3b` repo is swapped to a
  non-quantized bf16 mirror. `apertus-8b`, `qwen-2.5-7b`, and `mistral-7b`
  are listed but blocked with a clear error if selected: bf16 with no
  quantization makes 7B/8B a tight-to-unsafe fit on a single v5e-1's 16GB
  HBM, and qwen/mistral have no confirmed non-quantized mirror.
- `colab/train/tpu/train_model.py` installs no `unsloth`/`bitsandbytes` — just
  `transformers peft trl accelerate datasets` (plus whatever `torch_xla`
  build Colab's TPU runtime already ships).
- Both paths push an **adapter-only** model to the same naming scheme
  (`{HF_USER}/{model}-financial-reasoner-v1`), except the TPU path adds a
  `-tpu` suffix so a TPU run never overwrites a GPU-trained adapter at the
  same name, or vice versa. The trailing number is bumped by hand each time
  a new training attempt is pushed (v4 → v7 so far), so it tracks
  individual pushes, not the prompt/output schema - the schema itself has
  stayed the "analyst pipeline" generation (market data/valuation/earnings
  inputs plus the `answer` output field, introduced at v4) across all of
  them. Older numbered repos remain on Hugging Face, untouched, for
  comparison - **run `python bump_model_version.py v8`** (substituting
  whatever the new number actually is) to bump every reference across the
  whole repo in one shot instead of hand-editing each one; confirmed live
  that hand-editing misses files that aren't in the "obvious" gpu/tpu set -
  `run/run_model.py` and `docs/llm-training-primer.md` both drifted for
  multiple version bumps before this script existed specifically to catch
  that. For a quick, no-code-edit comparison in a single session (e.g.
  "does v6 actually do worse than v7?") instead of a permanent bump, set
  the optional `MODEL_VERSION` Secret instead - see "Required Colab
  Secrets" below. The two are for different situations: the Secret is a
  session-local override with no git trace, the script changes the
  committed default everyone gets when they haven't set that Secret.
- The TPU path hasn't been run end-to-end on real TPU hardware yet — the
  GPU path is the proven one. If you hit an issue running `colab/train/tpu/`'s
  scripts, that's expected first-run friction, not necessarily something
  you did wrong.

## Serving the model

Two ways to expose the trained model to `financial-sentiment-api` over
HTTP - pick one, both speak the exact same `{"inputs": ..., "parameters":
{...}}` request / `[{"generated_text": ...}]` response shape, so switching
between them is just repointing `HF_INFERENCE_URL` (its runtime-mutable
`/api/update-inference-url` endpoint exists for exactly this):

- **`colab/run/run_model.py`** (default) - paste into a Colab cell, loads the
  model on Colab's free GPU, exposes it through an ngrok tunnel. Free, but
  the tunnel dies with the Colab session (90-minute idle timeout, 12-hour
  hard cap - see `colab/run/keep_running.py`), and needs a browser tab open.
- **`modal/serve_model.py`** - deploys to [Modal](https://modal.com) as a
  scale-to-zero serverless GPU function: no browser tab, no session limit,
  but you pay per-second of actual GPU use. For occasional personal-project
  traffic (a handful of requests a day) this lands well under Modal's
  $30/month free credit - see the file's own docstring for the exact
  autoscaling config (`min_containers` deliberately unset, `max_containers
  =1`, `scaledown_window=120`) and the one-time `modal setup` / `modal
  secret create` steps needed before `modal deploy modal/serve_model.py`
  will work.

## Required Colab Secrets (environment variables)

Set these once via the key icon in Colab's left sidebar — **Secrets**, not
hardcoded anywhere in these scripts. Grant each secret access to the
notebook when the permission prompt appears; if that grant hasn't happened
yet, `userdata.get(...)` will block waiting for it, which matters if you're
running the whole notebook unattended via "Run all."

| Name | What it is |
|---|---|
| `HF_TOKEN` | A Hugging Face **write**-access token, used to push the fine-tuned model. |
| `HF_USER` | Your Hugging Face username, used to build the target repo name (`{HF_USER}/{model}-financial-reasoner-v1`). |
| `GEMINI_API_KEY` | A free key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey), used by `generate_real_dataset.py` to write headline-grounded `reasoning` and `answer` text (`gemini-3.5-flash-lite` — cost for the whole real dataset is well under $1). |
| `MODEL_VERSION` *(optional)* | Overrides the `-vN` suffix in the target repo name for this session only, without editing any file - e.g. set to `v6` to reload/evaluate an older push for comparison. Every script falls back to the git-committed `MODEL_VERSION_DEFAULT` (bumped via `python bump_model_version.py vN`) if this isn't set, so it's safe to leave unset entirely. |
| `MODEL_CHOICE` *(optional)* | Overrides which `MODEL_REGISTRY` entry (base model family, e.g. `apertus-8b`) to train/reload for this session only, without editing any file. Falls back to the git-committed `MODEL_CHOICE_DEFAULT` - `"llama-3.1-8b"` on the GPU/RunPod paths, `"llama-3.2-3b"` on the TPU path (8B doesn't fit there - see "GPU vs TPU" below) - if unset. When reloading a pushed model in an eval script, this must match whatever `MODEL_CHOICE` that specific push was actually trained under, not whatever you'd like to try next. |

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
  Both write the identical `{ticker, user_query, market_data, valuation,
  earnings, news, output}` schema so either `train_model.py` (`colab/train/gpu/` or
  `colab/train/tpu/`) can concatenate them with no reconciliation step.
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
  `train_on_responses_only` in `colab/train/gpu/train_model.py`, or to trl's
  `DataCollatorForCompletionOnlyLM` in `colab/train/tpu/train_model.py`) must match
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

- **The model reasons over data, not just headlines, and answers the
  user directly.** v4 added `market_data`/`valuation`/`earnings` to the
  prompt and `answer` to the output. Valuation is always a deterministic
  Graham Number (`sqrt(22.5 x EPS x book value/share)`) computed in code
  from the row's own price/EPS/book-value — never LLM-generated or
  hand-waved — so the model learns to read a real number, not to
  hallucinate one. Any block that couldn't be fetched/computed renders as
  exactly `Data unavailable.` in both training data and production, so the
  model is trained on, not just hoped to handle, partial data gaps. See the
  canonical prompt template comments above `task_a_prompt`/`task_b_prompt`
  in `colab/train/gpu/train_model.py` and CONTRIBUTING.md's sync rule
  before changing any of this.

## docs/

- `llm-training-primer.md` — a from-zero explanation of what every part of
  `colab/train/gpu/train_model.py` does, for anyone reading this without an ML
  background. Written against the GPU/unsloth path; `colab/train/tpu/train_model.py`
  swaps the same conceptual steps onto a different toolchain (see the
  "GPU vs TPU" section above).
- `training-results-analysis.md` — why the first training run's loss
  curves were misleading, and what to measure instead.
- `dataset-fix-plan.md` — the diagnosis and fix plan for the real-data
  accuracy gap found during evaluation.
