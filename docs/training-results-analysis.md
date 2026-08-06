# Training run analysis — why the loss curves look wrong (and what they actually tell us)

*Written after the first full run with the mixed synthetic+real dataset,
completion-only masking, and early stopping.*

## The numbers

| step | ~epoch | train loss | val loss | notes |
|------|--------|-----------|----------|-------|
| 50   | 0.25   | 0.12      | **0.50** | best val loss of the whole run |
| 100  | 0.5    | 0.049     | 0.55     | strike 1 |
| 150  | 0.75   | 0.058     | 0.52     | strike 2 |
| 200  | ~1.0   | 0.057     | 0.57     | strike 3 → early stopping fired |

Dataset composition for that run (synthetic generator is deterministic,
counts verified locally):

- Train: 1403 synthetic + 285 real = **1688 rows** → ~211 steps/epoch at batch 8
- Val: 597 synthetic + 60 real = **657 rows** (91% synthetic, 9% real)

## What's suspicious

Read as a normal training run, this says: "the model peaked after seeing a
quarter of the data once, and got steadily worse from there." Models don't
usually peak before finishing even one epoch. Something else is going on.

Three facts that don't fit the simple "it overfit" story:

1. **Train loss hit 0.049 at step 100 — before the model had seen each
   training example even once.** You cannot memorize examples you haven't
   seen. What it memorized is the *templates*: the scenario templates
   behind the synthetic rows repeat every few examples, so by step 100 the
   model has seen every template dozens of times. Train loss ≈ "how well
   do I know the templates," not "how well do I judge sentiment."

2. **Val loss has a high floor built into it by design — it can't go much
   below ~0.5 no matter how good the model gets.** Two reasons:
   - The synthetic val split (91% of val) was deliberately held out **by
     template**: val examples use reasoning phrasings the model never saw.
     Loss measures exact-next-token prediction — a model that outputs a
     *perfectly correct* direction with slightly different wording than the
     unseen template still scores a high loss. The val set was built to be
     unpredictable at the token level, then measured for token-level
     predictability.
   - The real val rows (9%) are proxy-labeled from 3-day forward price
     moves. Headlines genuinely don't determine short-term price moves most
     of the time, so a chunk of those labels is effectively noise — no
     model can predict noise, and that loss is irreducible.

3. **No baseline.** No eval was run at step 0, so there's no reading of
   what val loss the *untrained* base model gets. If base Llama scores
   ~0.6 and the best checkpoint scores 0.50, training barely moved the
   needle. If base scores 2.5, step 50 represents a huge improvement.
   Without this number, these two very different outcomes are
   indistinguishable — and they'd imply completely different next steps.

## The deeper problem: loss is measuring the wrong thing entirely

The response the model is graded on is a ~150-token JSON blob. The tokens
that actually matter — `BULLISH`/`BEARISH`/`NEUTRAL` and the confidence
number — are maybe 3-5 of those tokens. The other ~145 are JSON braces,
key names, and reasoning prose. Cross-entropy loss weights every token
equally, so **~97% of the loss signal is about reproducing boilerplate and
phrasing, ~3% is about the actual classification decision.**

Concrete consequence: the model could get the direction wrong on a third of
val examples and the loss would barely register it; or get every direction
right and still show 0.5 loss because its reasoning prose doesn't match the
held-out templates word-for-word. **The metric this run early-stopped on is
nearly uncorrelated with the thing that actually matters.** The early stop
at step 200 may have been correct or may have thrown away a still-improving
model — the loss curve genuinely cannot tell us which.

So: the training *mechanics* all worked (masking, checkpointing, early
stopping, best-model reload). What's "wrong" is that the whole control loop
is steering by an instrument that doesn't point at the destination.

## Is the training data the problem?

Partly, but not in the "we need more rows" sense:

- **Template repetition on the output side** is what makes train loss
  collapse instantly and makes it meaningless. More rows from the same
  templates would change nothing.
- **The real data is drowned out**: 285 real rows vs 1403 synthetic (17% of
  train, in that run). Whatever signal real headlines carry, the gradient
  is dominated by template completion.
- But before generating anything new, the actual question is whether
  there's even a problem — which loss can't answer. Hence:

## What to do next, in order

1. **Measure direction accuracy — the metric that matters.** Load the
   fine-tuned model, run every val row through it, parse the JSON output,
   compare `direction` against the label. Report accuracy overall AND
   split by (a) synthetic vs real val, (b) direction class (confusion
   matrix — is it biased toward one answer?), (c) JSON parse-failure rate.
   This turns "0.50 loss, shrug" into "83% on synthetic val, 41% on real
   val" — something actionable. (`evaluate_model.py`)

2. **Run the same accuracy eval on the *base* model (no fine-tune).**
   That's the missing baseline. The delta between base and fine-tuned is
   the true value of the whole training pipeline. Cheap to run, and it's
   the single most informative number otherwise missing.

3. **Decide based on those numbers, not on loss:**
   - High accuracy on both val sources → ship it; the loss curve was a
     false alarm and the data is fine.
   - High on synthetic, ~chance on real → the model learned template
     patterns, not transferable judgment → dataset work is justified:
     paraphrase-diversify the synthetic reasoning text (break the
     template↔wording lock), and/or raise the real-data share.
   - Specific weakness (e.g. bad on NEUTRAL, or always BULLISH) → targeted
     data additions for that case, not blanket "more data."

4. **If/when retraining, fix the instrumentation:**
   - Run `trainer.evaluate()` once *before* `trainer.train()` for a step-0
     baseline.
   - Report the two val sources as separate metrics (HF Trainer accepts a
     dict of eval datasets) so a noisy-real floor can't mask synthetic
     progress or vice versa.
   - Consider early-stopping on direction accuracy via a custom metric
     instead of token loss — or at least lengthen patience and lower the
     learning rate (2e-4 → 5e-5) so the decision points are less noisy.
   - Optionally restructure the output JSON to put `direction` and
     `confidence` *first*, before the reasoning prose, so the tokens that
     matter contribute loss before the model commits to boilerplate — and
     so truncation can never eat them.

## Bottom line

The run didn't fail — but it also didn't tell us what we needed to know.
The loss gap that looked like overfitting is mostly an artifact of grading
exact wording against deliberately-unseen phrasings plus partially-random
labels. Don't touch the training data based on loss alone: build the
accuracy eval, get the base-vs-tuned comparison, and let that decide
whether dataset work is needed and where. (This is exactly what the actual
eval run turned up — see `dataset-fix-plan.md` for the follow-on diagnosis
once real accuracy numbers were in hand.)
