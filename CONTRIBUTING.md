# Contributing

This is a personal project, but the workflow below applies to any change,
human or AI-assisted.

## Branching

- Never commit directly to `main`. Every change goes on a feature branch
  cut from an up-to-date `main`.
- Branch prefixes: `feature/`, `fix/`, `docs/`, `refactor/`, `test/`.
- Only the repository owner pushes to `main` — that happens by merging a
  reviewed pull request, not by pushing directly.

```bash
git checkout main
git pull origin main
git checkout -b fix/short-description
```

## Before opening a pull request

Most of this repo runs by hand in Colab, against live external services
(Google News RSS, yfinance, Hugging Face), so there's no CI and most of it
can't be usefully mocked. There IS a real `tests/` suite for the pure-logic
pieces that don't touch those services (the recommendation-accuracy regex
used by eval's fuse()-agreement check, `generate_real_dataset.py`'s
filter/parsing helpers) — run `python -m pytest tests/` before opening a
PR; it's cheap and catches real regressions. For everything else:

- If you change a generator script, run it locally (both are plain Python;
  `generate_synthetic_dataset.py` needs no dependencies beyond the standard
  library, `generate_real_dataset.py` needs `pip install yfinance httpx
  feedparser google-genai pandas` and a `GEMINI_API_KEY` env var) and sanity-check
  the output — row counts, a few sample rows, and that
  `python -m py_compile <file>` passes. Without a Gemini key you can still
  `py_compile` and read through the diff, but can't confirm
  `generate_grounded_reasoning` actually produces sensible text end to end
  — say so explicitly in the PR description rather than claiming it was
  tested if it wasn't.
- If you change either prompt template (`task_a_prompt` for news_reaction
  classification, `task_b_prompt` for reasoning/answer generation - see
  the two-stage pipeline redesign in `generate_real_dataset.py`'s history)
  or the output JSON schema, keep NINE files in sync, all carrying BOTH
  templates byte-identically: `colab/train/gpu/train_model.py`,
  `colab/train/tpu/train_model.py`, their own `evaluate_model.py`/
  `evaluate_base_model_only.py` copies (4 more copies, 6 total),
  `runpod/train_model.py` and `runpod/evaluate_model.py` (2 more, 8
  total), and `financial-sentiment-api`'s `app/services/inference.py` (9
  total) — a mismatch anywhere in that set silently trains or serves a
  different shape than the others expect. This also covers the
  `market_data`/`valuation`/`earnings`/`price_context` block FORMATTING
  (not just the outer templates) — the block renderers in
  `generate_synthetic_dataset.py`, `generate_real_dataset.py`, and
  `financial-sentiment-api`'s `app/services/{fundamentals,valuation,
  earnings}.py` need to produce byte-compatible shapes (e.g. "Data
  unavailable." exactly, the same "$X.XXT"/"$X.XB" market-cap notation,
  the same "{ticker} moved {pct:+.1f}% over the last 3 trading days."
  price-context phrasing), since the model is trained on one shape and
  served against whatever these renderers actually produce.
- Separately, `fusion_rules.py` (the deterministic news_reaction +
  valuation-gap -> BUY/SELL/HOLD table - the only place a recommendation
  is ever decided, in either training data generation or production) must
  stay byte-identical between this repo's own `fusion_rules.py` and
  `financial-sentiment-api`'s `app/services/fusion.py`. A drift here means
  training data and production compute different recommendations for the
  same (news_reaction, valuation gap) pair - a correctness bug, not just a
  formatting one.
- If you bump the Hugging Face model repo version (the `-financial-
  reasoner-vN` suffix, a different concept from the prompt/schema version
  above), run `python bump_model_version.py vN` instead of hand-editing
  each `HF_REPO =`/`MODEL_NAME =` line - it scans the whole repo tree
  rather than a fixed file list, specifically because hand-editing has
  already missed `run/run_model.py` and `docs/llm-training-primer.md`
  (both outside the "obvious" `gpu/`/`tpu/` set) across multiple real
  version bumps.
- If you change anything that affects the instruction/response markers
  used for completion-only loss masking (`train_on_responses_only`'s
  `instruction_part`/`response_part` in `colab/train/gpu/train_model.py`, or
  `DataCollatorForCompletionOnlyLM`'s `instruction_template`/
  `response_template` in `colab/train/tpu/train_model.py`), re-verify the tokenization
  match locally (load the target model's tokenizer, tokenize the marker in
  isolation and embedded in a real formatted example, confirm the token
  sequence actually appears) — see the comment above that call in either
  file for why this is a real, silent-failure-prone gotcha, not a
  hypothetical one.

## What a good PR description covers

- What changed and why (the "why" matters more than the "what" — the diff
  already shows what changed).
- If the change affects the trained model's behavior (prompt, schema,
  hyperparameters, dataset composition), what eval results (if any) support
  the change.

## Secrets

Never hardcode a Hugging Face token, username, or any other credential
into a script in this repo — pull it from Colab Secrets (`userdata.get(...)`)
instead. See the README's "Required Colab Secrets" section for the current
list.

## Code style

- No comments explaining *what* code does — names should already make that
  clear. Comments are for *why*, especially non-obvious constraints, a
  workaround for a specific bug, or a decision that would otherwise look
  arbitrary (see the existing `# CHANGED:` comments throughout for the
  pattern this project uses).
- Don't add speculative abstractions, config flags, or error handling for
  cases that can't occur given how a script is actually run. This is a
  small, single-purpose pipeline, not a library.
