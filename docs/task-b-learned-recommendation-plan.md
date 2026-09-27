# Plan: Task B reasons its own recommendation, instead of being told one

## Where this actually lives today (read this first)

If you've only read `financial-sentiment-api/app/routers/analyze.py`, you
won't find anything called "reaction" or "recommendation" in it — that
router only gathers *inputs* (fundamentals, earnings, live news, the
DCF valuation gap) and hands them to one function call,
`analyze_two_stage(...)`, at the very end. All the actual reasoning
happens one layer down, invisible from `analyze.py`:

- **`app/services/inference.py::analyze_two_stage`** — the orchestrator.
  Calls Task A, then `fuse()`, then Task B, in that order.
- **`app/services/inference.py::classify_news`** — Task A. One LLM call,
  given the news headline + that day's price move, classifies
  `news_reaction` ∈ `{good, bad, neutral, overreaction_down,
  overreaction_up}`. This already works the way this plan wants: the
  model reasons the classification itself from the inputs.
- **`app/services/fusion.py::fuse()`** — **not an LLM call at all.** A
  fixed Python lookup table over `(news_reaction, valuation_bucket)` that
  outputs the recommendation (BUY/SELL/HOLD) and a confidence score. This
  is the *only* place a recommendation is decided anywhere in the
  pipeline today.
- **`app/services/inference.py::generate_analysis`** — Task B. A second
  LLM call, given `news_reaction` **and** `fuse()`'s recommendation as
  already-decided facts. Its system instruction is explicit: *"your job
  is to explain it, not decide it... must never advise the opposite."*
  Task B never sees or produces a recommendation of its own — it writes
  prose consistent with the one it's handed.

**Is `fuse()` necessary?** Today, yes — it's the only thing that ever
computes a recommendation; delete it and nothing does. Under this plan
it stops being the sole production decision-maker (Task B takes that
over), but it doesn't go away: it becomes the training-label generator
Task B is taught to reproduce, and stays at inference as a logged
consistency check. See "Fuse()'s new role" below.

Byte-identical copies of `fuse()` (`financial-sentiment-model/
fusion_rules.py`) and the Task B prompt template exist for the same
reason the sibling repo's copies do — train and inference must compute
identical things the identical way. Confirmed by grep, every one of
these carries its own copy of `task_b_prompt` and needs the identical
edit:
- `runpod/train_model.py`, `runpod/evaluate_model.py`
- `colab/train/gpu/train_model.py`, `colab/train/gpu/evaluate_model.py`,
  `colab/train/gpu/evaluate_base_model_only.py`
- `colab/train/tpu/train_model.py`, `colab/train/tpu/evaluate_model.py`,
  `colab/train/tpu/evaluate_base_model_only.py`
- `financial-sentiment-api/app/services/inference.py`
  (`_build_analysis_prompt`) — the production-serving copy

See this repo's `CONTRIBUTING.md` sync-rule section.

## What changes

Task A is untouched — it already reasons the classification itself from
inputs, which is exactly what's being asked of Task B here.

**1. Task B's prompt** (`financial-sentiment-api/app/services/
inference.py::_build_analysis_prompt`, and its byte-identical copies —
see above):
- Drop `Recommended Action: {recommendation}` from `### Input:`.
- Rewrite the instruction: from *"you are given a recommended action...
  explain it, not decide it"* to *"decide the recommended action (BUY,
  SELL, or HOLD) yourself from the news reaction, valuation, earnings,
  and market data below, then explain your reasoning and answer the
  user's question."*
- Output JSON shape grows one field:
  `{"recommendation": "BUY|SELL|HOLD", "reasoning": "...", "answer": "..."}`.
- `news_reaction` stays as a given input — Task A still produces it
  separately and unchanged.

**2. Training data regeneration** (`generate_real_dataset.py` /
`generate_synthetic_dataset.py`): `fuse()` keeps running exactly as now
when a "analysis" row is built, but its output moves from an *input*
field into the *target completion* JSON. Gemini's writing prompt
(`generate_grounded_reasoning`) still tells Gemini the correct
recommendation so its prose stays consistent — that doesn't change, only
where the value is saved in the row. Requires a **full regeneration of
the 348 existing "analysis" rows** (218 train + 130 val) — real Gemini
cost, same order of spend as the last full Task B regen, not new/extra
beyond a normal regen.

**3. Training script**: confirmed a plain `SFTTrainer` text-completion
fine-tune (`format_prompts` in `runpod/train_model.py` and the Colab
copies) — no multi-head/structured-output machinery to add. This is a
template-string and one-fewer-format-argument change, not an
infrastructure change.

**4. `fuse()`'s new role**: code unchanged in both repos. At inference,
keep computing it **alongside** the model's own recommendation rather
than replacing it outright — log divergence, don't cut over to trusting
the model's field until eval says so (step 6 below). `fuse()`'s own
docstring (`fusion.py`, and byte-identically in `fusion_rules.py`)
currently states *"neither Task A nor Task B's LLM call ever produces a
recommendation itself"* — that sentence becomes false and needs
rewriting to describe the new distillation relationship (`fuse()`
generates the label Task B is trained to reproduce, and remains a
shadow/consistency check, not the production source of truth).

**5. Eval** (`evaluate_model.py`, both copies): add a real
recommendation-accuracy metric —
`model_recommendation == fuse(row.news_reaction, row.gap_pct).recommendation`
— replacing the current check, which only verifies the model didn't
*contradict* the recommendation it was handed. That check becomes
meaningless once nothing is handed to it anymore.

## The risk this plan has to manage

This reverses part of a deliberate, documented fix: the old single-call
design had the model learn the sentiment→recommendation mapping
implicitly, unreliably. `fuse()` was extracted specifically to make that
mapping a fixed, testable table instead of something trained into the
model. This plan only undoes the *last* step of that (Task B deciding,
not Task A), and leans on the one thing the old design didn't have:
`fuse()` as a **measurable ground truth**. Recommendation-accuracy vs.
`fuse()` on held-out eval turns "did the model learn this" from an
unmeasured guess into a hard number — which is the actual gate below.

## Order of work

1. **Don't spend budget yet.** Bundle this with any other pending
   Task B/training fixes into a single retrain cycle rather than testing
   it in isolation.
2. Prompt/template edit across all synced copies (see the file list
   above) + `fusion.py`'s and `fusion_rules.py`'s docstrings.
3. Regenerate the 348 Task B rows (the real Gemini cost).
4. Retrain (the real GPU cost) — one bundled cycle.
5. Eval: recommendation-accuracy vs. `fuse()` on held-out real + synthetic
   rows, split by `(news_reaction, valuation_bucket)` cell (5 reactions ×
   4 buckets = 20 cells) so a systematically weak cell is visible rather
   than averaged away.
6. **Gate**: ship only if accuracy vs. `fuse()` clears a bar worth
   trusting (suggest not shipping below ~90% agreement — disagreement
   below that is noise, not learned nuance). Below the bar, this stays
   shelved, same as the Task A redesign plan's precedent
   (`docs/two-stage-task-a-redesign-plan.md`) — not neglected, just not
   worth the retrain cost yet.

## Open question

What should production do when the model's recommendation and `fuse()`'s
disagree, even after this ships and clears the gate? Trust the model
silently, trust `fuse()` silently, or surface the disagreement in the
response? This changes what step 6's gate is actually gating, so it's
worth deciding before shipping, not after.

## Verification (once resumed)

- Recommendation-accuracy vs. `fuse()`, overall and per-cell, on held-out
  real + synthetic Task B rows.
- Spot-check regenerated rows: the saved `output` JSON's `recommendation`
  matches what `fuse()` computes from that row's own `news_reaction` +
  `gap_pct` — a mechanical check that the data-shape migration didn't
  introduce a mismatch between the label and the inputs it was derived
  from.
- Live smoke test against a real ticker query: confirm Task B's response
  includes a `recommendation` field, and log `fuse()`'s shadow value
  alongside it for a first look at real-world divergence before the gate
  decision.
