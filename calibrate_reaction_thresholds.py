"""Empirically measures the single-day (move_1d) and 21-trading-day
forward-return distributions for the real dataset's 40 tickers, to pick
thresholds for the 5-class news_reaction labeling (good/bad/neutral/
overreaction_down/overreaction_up) used by generate_real_dataset.py's
label_news_reaction (Task A of the two-stage pipeline - see the plan file
for the full architecture).

Re-run 2026-08-20 against move_1d (day-0/publish-day close vs. the
preceding trading day's close) instead of the original move_3d
(3-trading-day forward cumulative window) - see generate_real_dataset.py's
history item 12 for why: a 3-day cumulative window was found to dilute
fast-reverting crashes (a >10% same-day drop that mostly reverses within 2
days read as only ~-6% under the old window), and wasn't reproducible at
real inference time for a brand-new headline anyway. The candidate grids
below are widened downward from the original 2026-08-19 run's, since
single-day return volatility is smaller in magnitude than 3-day
cumulative for the same underlying "genuinely large move" - re-verify
against this run's actual measured distribution, don't assume the old
grids still bracket the right answer.

Zero Gemini cost: only touches yfinance price history, same pattern as
the now-deleted diagnose_return_distribution.py (which settled the
earlier +/-8% BUY/SELL threshold the same way, then went dead on this
branch once label_from_forward_return was retired). Read-only: doesn't
write dataset_train_real.jsonl/
dataset_val_real.jsonl.

Imports measure_reaction_windows/classify_reaction directly from
generate_real_dataset.py rather than keeping its own copies, same "reuse
the exact production function" precedent diagnose_return_distribution.py
established for the (now retired) label_from_forward_return - classify_
reaction takes its thresholds as keyword args specifically so this script
can sweep a grid without a second copy of the classification logic to
silently drift out of sync with what generate_real_dataset.py actually
ships.

Usage:
    python calibrate_reaction_thresholds.py
"""

import datetime
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

from generate_real_dataset import (
    TICKERS,
    weekly_windows,
    measure_reaction_windows,
    classify_reaction,
)

SAMPLES_PER_TICKER = 18  # matches LOOKBACK_WEEKS - one anchor date per weekly window

# Candidate threshold grids swept below - widened downward from the
# original 3-day-window run's (0.02-0.05 / 0.05-0.08) since single-day
# return volatility is structurally smaller in magnitude than 3-day
# cumulative for the same z-score; this run's own printed distribution
# (percentiles/stdev) is the actual source of truth, these are just a
# generous starting bracket. Retracement fraction is scale-invariant (a
# FRACTION of move_1d, not an absolute return) so unchanged from before -
# the original run found it barely moved class counts across 0.4-0.6.
GOOD_BAD_CANDIDATES = (0.01, 0.015, 0.02, 0.025, 0.03)
OVERREACTION_MOVE_CANDIDATES = (0.03, 0.04, 0.05, 0.06, 0.07)
RETRACEMENT_FRACTION_CANDIDATES = (0.4, 0.5, 0.6)

# Acceptance band from the plan: combined overreaction classes should be a
# real minority (rare enough to be a genuine "distinct from routine
# good/bad" phenomenon) but not so rare training can't learn them; neutral
# should be a healthy plurality, not a byword for "everything else".
OVERREACTION_FRACTION_BAND = (0.05, 0.12)
NEUTRAL_FRACTION_BAND = (0.25, 0.45)


class Sample:
    __slots__ = ("move_1d", "move_21d", "overreaction_assessable")

    def __init__(self, move_1d, move_21d, overreaction_assessable):
        self.move_1d = move_1d
        self.move_21d = move_21d
        self.overreaction_assessable = overreaction_assessable


def sample_ticker(ticker, name):
    ticker_obj = yf.Ticker(ticker)
    try:
        earnings_dates = ticker_obj.earnings_dates
    except Exception as e:
        print(f"    Warning: earnings_dates fetch failed for {ticker!r}: {e}", flush=True)
        earnings_dates = None

    samples = []
    skip_reasons = []
    for after_date, before_date in weekly_windows():
        anchor = datetime.datetime.combine(after_date, datetime.time(12, 0), tzinfo=datetime.timezone.utc)
        move_1d, move_21d, overreaction_assessable, skip_reason = measure_reaction_windows(
            ticker_obj, anchor, earnings_dates,
        )
        if skip_reason is not None:
            skip_reasons.append(skip_reason)
            continue
        samples.append(Sample(move_1d, move_21d, overreaction_assessable))
    print(f"  [{ticker}] {len(samples)}/{SAMPLES_PER_TICKER} samples collected", flush=True)
    return samples, skip_reasons


def percentile(sorted_values, p):
    n = len(sorted_values)
    if n == 0:
        return None
    idx = min(n - 1, max(0, round(p / 100 * (n - 1))))
    return sorted_values[idx]


def print_distribution(label, values):
    values = sorted(values)
    n = len(values)
    print(f"\n{label} (n={n}):")
    if n == 0:
        print("  no samples")
        return
    for p in (5, 10, 25, 50, 75, 90, 95):
        print(f"  p{p}: {percentile(values, p) * 100:+.1f}%")
    print(f"  mean: {statistics.mean(values) * 100:+.1f}%")
    if n > 1:
        print(f"  stdev: {statistics.stdev(values) * 100:.1f}%")


def main():
    all_samples = []
    all_skip_reasons = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(sample_ticker, ticker, name): ticker for ticker, name in TICKERS}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                samples, skip_reasons = future.result()
                all_samples.extend(samples)
                all_skip_reasons.extend(skip_reasons)
            except Exception as e:
                print(f"  Warning: {ticker!r} failed entirely: {e}", flush=True)

    n = len(all_samples)
    print(f"\nTotal usable samples: {n} (across {len(TICKERS)} tickers, up to {SAMPLES_PER_TICKER} each)")
    if all_skip_reasons:
        from collections import Counter
        print(f"Skipped {len(all_skip_reasons)} rows: {dict(Counter(all_skip_reasons))}")
    if n == 0:
        print("No samples collected - aborting.")
        return

    print_distribution("single-day move", [s.move_1d for s in all_samples])

    assessable_with_21d = [s for s in all_samples if s.overreaction_assessable and s.move_21d is not None]
    print(f"\nOverreaction-assessable samples with a full 21-day window: {len(assessable_with_21d)}/{n}")

    # Retracement distribution for large movers only (|move_1d| >= the
    # smallest overreaction-move candidate) - the population any
    # OVERREACTION_MOVE_THRESHOLD choice would actually draw from.
    large_movers = [s for s in assessable_with_21d if abs(s.move_1d) >= min(OVERREACTION_MOVE_CANDIDATES)]
    down_retracements = [
        (s.move_21d - s.move_1d) / abs(s.move_1d) for s in large_movers if s.move_1d < 0
    ]
    up_retracements = [
        (s.move_1d - s.move_21d) / s.move_1d for s in large_movers if s.move_1d > 0
    ]
    print_distribution(f"Retracement fraction, down-movers (|move_1d| >= {min(OVERREACTION_MOVE_CANDIDATES)*100:.0f}%)", down_retracements)
    print_distribution(f"Retracement fraction, up-movers (|move_1d| >= {min(OVERREACTION_MOVE_CANDIDATES)*100:.0f}%)", up_retracements)

    print("\nClass-count grid (X=good/bad threshold, Y=overreaction move threshold, Z=retracement fraction):")
    print(f"Acceptance band: overreaction {OVERREACTION_FRACTION_BAND[0]*100:.0f}-{OVERREACTION_FRACTION_BAND[1]*100:.0f}%, "
          f"neutral {NEUTRAL_FRACTION_BAND[0]*100:.0f}-{NEUTRAL_FRACTION_BAND[1]*100:.0f}% (marked with ** below)")

    passing = []
    for x in GOOD_BAD_CANDIDATES:
        for y in OVERREACTION_MOVE_CANDIDATES:
            if y <= x:
                continue  # overreaction threshold must exceed the good/bad threshold
            for z in RETRACEMENT_FRACTION_CANDIDATES:
                counts = {"good": 0, "bad": 0, "neutral": 0, "overreaction_down": 0, "overreaction_up": 0}
                for s in all_samples:
                    label = classify_reaction(s.move_1d, s.move_21d, s.overreaction_assessable, x, y, z)
                    counts[label] += 1
                overreaction_n = counts["overreaction_down"] + counts["overreaction_up"]
                overreaction_frac = overreaction_n / n
                neutral_frac = counts["neutral"] / n
                meets_band = (
                    OVERREACTION_FRACTION_BAND[0] <= overreaction_frac <= OVERREACTION_FRACTION_BAND[1]
                    and NEUTRAL_FRACTION_BAND[0] <= neutral_frac <= NEUTRAL_FRACTION_BAND[1]
                )
                marker = " **" if meets_band else ""
                if meets_band:
                    passing.append((x, y, z, counts))
                print(
                    f"  X=+/-{x*100:.0f}% Y=+/-{y*100:.0f}% Z={z:.1f}: "
                    f"good={counts['good']} ({100*counts['good']/n:.1f}%) "
                    f"bad={counts['bad']} ({100*counts['bad']/n:.1f}%) "
                    f"neutral={counts['neutral']} ({100*neutral_frac:.1f}%) "
                    f"over_down={counts['overreaction_down']} over_up={counts['overreaction_up']} "
                    f"(overreaction total {100*overreaction_frac:.1f}%){marker}"
                )

    print(f"\n{len(passing)}/{len(GOOD_BAD_CANDIDATES)*len(OVERREACTION_MOVE_CANDIDATES)*len(RETRACEMENT_FRACTION_CANDIDATES)} combos meet the acceptance band.")
    if passing:
        print("Pick the combo above whose class balance looks most useful for training and record it in generate_real_dataset.py's label_news_reaction docstring.")
    else:
        print("No combo met the band - widen GOOD_BAD_CANDIDATES/OVERREACTION_MOVE_CANDIDATES/RETRACEMENT_FRACTION_CANDIDATES and rerun.")


if __name__ == "__main__":
    main()
