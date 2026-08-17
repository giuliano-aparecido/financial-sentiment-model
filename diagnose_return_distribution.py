"""Empirically measures the actual forward-return distribution for the
real dataset's 40 tickers, using the EXACT production label_from_forward_
return() function (not a reimplementation) - so BUY_THRESHOLD/SELL_
THRESHOLD can be picked from real data instead of a formula or a guess.

Zero Gemini cost: label_from_forward_return() only touches yfinance price
history. Gemini is a separate function (generate_grounded_reasoning) never
called here. Read-only: doesn't write dataset_train_real.jsonl/
dataset_val_real.jsonl or touch generate_real_dataset.py.

Usage:
    python diagnose_return_distribution.py
"""

import datetime
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

from generate_real_dataset import (
    TICKERS,
    weekly_windows,
    label_from_forward_return,
)

SAMPLES_PER_TICKER = 18  # matches LOOKBACK_WEEKS - one anchor date per weekly window


def sample_ticker(ticker, name):
    ticker_obj = yf.Ticker(ticker)
    try:
        earnings_dates = ticker_obj.earnings_dates
    except Exception as e:
        print(f"    Warning: earnings_dates fetch failed for {ticker!r}: {e}", flush=True)
        earnings_dates = None

    pct_changes = []
    for after_date, before_date in weekly_windows():
        anchor = datetime.datetime.combine(after_date, datetime.time(12, 0), tzinfo=datetime.timezone.utc)
        direction, pct_change, actual_window_days, skip_reason = label_from_forward_return(
            ticker_obj, anchor, earnings_dates,
        )
        if pct_change is not None:
            pct_changes.append(pct_change)
    print(f"  [{ticker}] {len(pct_changes)}/{SAMPLES_PER_TICKER} samples collected", flush=True)
    return pct_changes


def main():
    all_pct_changes = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(sample_ticker, ticker, name): ticker for ticker, name in TICKERS}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                all_pct_changes.extend(future.result())
            except Exception as e:
                print(f"  Warning: {ticker!r} failed entirely: {e}", flush=True)

    all_pct_changes.sort()
    n = len(all_pct_changes)
    print(f"\nTotal samples: {n} (across {len(TICKERS)} tickers, up to {SAMPLES_PER_TICKER} each)")
    if n == 0:
        print("No samples collected - aborting.")
        return

    def pct(p):
        idx = min(n - 1, max(0, round(p / 100 * (n - 1))))
        return all_pct_changes[idx]

    print("\nPercentiles of forward return (63 trading days, earnings-truncated):")
    for p in (5, 10, 25, 50, 75, 90, 95):
        print(f"  p{p}: {pct(p) * 100:+.1f}%")
    print(f"  mean: {statistics.mean(all_pct_changes) * 100:+.1f}%")
    print(f"  stdev: {statistics.stdev(all_pct_changes) * 100:.1f}%")

    print("\nHOLD/BUY/SELL split at candidate thresholds:")
    for threshold in (0.03, 0.05, 0.06, 0.08, 0.10, 0.12):
        buy = sum(1 for x in all_pct_changes if x >= threshold)
        sell = sum(1 for x in all_pct_changes if x <= -threshold)
        hold = n - buy - sell
        print(f"  +/-{threshold*100:.0f}%: HOLD={hold} ({100*hold/n:.1f}%), "
              f"BUY={buy} ({100*buy/n:.1f}%), SELL={sell} ({100*sell/n:.1f}%)")


if __name__ == "__main__":
    main()
