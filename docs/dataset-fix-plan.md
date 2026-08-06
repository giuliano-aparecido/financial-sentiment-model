# Plan: closing the real-data accuracy gap

*Based on the first full eval run: 63.1% overall direction accuracy, 86.0%
on synthetic val, 25.0% on real val (below the 33% you'd get from random
3-way guessing), plus a confusion matrix showing the model defaults to
NEUTRAL whenever it's not confident. See `training-results-analysis.md` for
the loss-curve analysis that preceded this.*

## Diagnosis

Three compounding causes, not one:

1. **Real data is 17% of the training mix** (285 of 1688 train rows, in the
   run these numbers came from). The gradient is dominated by synthetic
   templates; real headlines barely get a vote.
2. **Domain gap between the two sources.** Synthetic templates use
   deliberately on-the-nose signal language ("guidance cut," "beat
   expectations"). Real headlines are naturally ambiguous, and the
   3-day-forward-return proxy label is a genuinely noisy stand-in for
   "what a reader would conclude." The model learned the synthetic style's
   shortcuts and doesn't have an equivalent for real phrasing.
3. **Learned NEUTRAL-hedging.** Not a raw frequency artifact - NEUTRAL is
   actually *underrepresented* in training (~22-24% combined, not the ~33%
   an even split would give it) - so this looks like a calibration habit
   ("if the signal isn't textbook-clear, say NEUTRAL") rather than the
   model just parroting the majority class. It costs the most on real val
   specifically because real val's natural distribution is BEARISH-heavy
   (31/60) - hedging to NEUTRAL there is wrong most of the time.

## Dataset changes made in response

### 1. Split the eval's confusion matrix by source
`evaluate_model.py` now tracks confusion separately for synthetic and real
val, instead of pooling both - confirms whether NEUTRAL-hedging is a
real-data-only problem or shows up in synthetic too.

### 2. Grew the real dataset's raw pool
`generate_real_dataset.py` deliberately undersamples-to-balance rather than
duplicates (to avoid memorization), which means the only way to grow
real's usable share without reintroducing that risk is to grow the raw
pool it's drawn from:
- `TICKERS`: 20 → 40 (more liquid, well-covered names so Google News RSS
  has more to return)
- `LOOKBACK_WEEKS`: 8 → 18 (more historical windows per ticker)
- `MAX_HEADLINES_PER_TICKER`: 30 → 50

### 3. Made the labeling threshold a visible label-quality knob
`BULLISH_THRESHOLD`/`BEARISH_THRESHOLD` (±2% over a 3-day window) is a
fairly loose bar that lets ambiguous, low-conviction price moves into the
training set as confident labels. `OUTPUT_TRAIN_FILE`/`OUTPUT_VAL_FILE`
were broken out as their own constants specifically so a stricter-threshold
comparison run (±3-4%) can write to different filenames instead of
overwriting the default - fewer real examples, but each one a cleaner
signal. Still an open experiment to actually run and compare.

### 4. Diversified synthetic templates toward "subtler but still decisive" cases
`generate_synthetic_dataset.py` had every BULLISH/BEARISH template resolve
from obvious language. Added a second tier of templates where the signal is
real but softer - closer to how an actual headline reads (options
positioning, analyst estimate trims, investor-day reception, insider
filings, a competitor's stumble) - while still resolving cleanly to
BULLISH/BEARISH, never to NEUTRAL. This directly targets the hedging bias:
it teaches the model that "not blatantly obvious" doesn't mean "give up and
say NEUTRAL." True-NEUTRAL templates stay genuinely flat/ambiguous so the
boundary between "subtle-but-real signal" and "no signal" stays clean and
learnable.

## Still open

### 5. Rebalance the training MIX ratio, not just each source internally
Once (2) grows real's raw pool, revisit what fraction of final `train`
should be real vs synthetic. The original mix was roughly 83/17
synthetic/real just from each source's natural size after rebalancing.
Consider capping synthetic's contribution (subsample it down) so real makes
up a meaningfully larger share - maybe 30-40% - without inflating real via
duplication. Depends on knowing the new real dataset's actual size once
`generate_real_dataset.py` is run for real with the expanded ticker/lookback
scope.

## What needs a GPU

6. Get the base-model baseline (`evaluate_base_model_only.py`, or the
   base-model pass built into `evaluate_model.py`) so the value training
   added is known independent of the dataset changes above.
7. Retrain on the updated dataset mix - no structural changes needed to
   `train_model.py`, the schema stays `{ticker, user_query, news, output}`
   regardless of how the generators' internals change.
8. Re-run the eval and compare: the number that matters most is real-val
   accuracy moving meaningfully above 33% (random), and the source-split
   confusion matrix showing less NEUTRAL pile-up specifically on
   BEARISH/BULLISH-truth rows.

## What "done" looks like

Not a specific target number (this is a curriculum project, not a
production SLA) - but concretely: real-val accuracy clearly above the 33%
chance baseline (say 50%+), and the source-split confusion matrix no longer
showing the lopsided "everything uncertain becomes NEUTRAL" pattern on the
real subset specifically.
