"""Spot-check tool for verifying dataset_train_real.jsonl/dataset_val_real.jsonl
rows against a fresh re-derivation, using the EXACT same code path
generate_real_dataset.py used to build the row in the first place (not a
reimplementation that could itself disagree) - fetches real historical
fundamentals for a ticker as of a specific date via yfinance (free, no
Gemini cost) and prints both the raw inputs and the recomputed
market_data/valuation/earnings blocks, so you can diff them by eye against
whatever a specific dataset row claims.

Usage:
    python spot_check.py TICKER YYYY-MM-DD

Example:
    python spot_check.py QCOM 2026-07-15

Note: if the numbers here DON'T match what's in the dataset file for the
same ticker/date, that's a real discrepancy worth investigating (yfinance
data can itself be revised after the fact - e.g. analyst estimates get
updated - so an exact mismatch isn't automatically a code bug, but it's
the first thing to explain). If they DO match, that specific row's
valuation math is confirmed correct against live-refetched data.
"""

import datetime
import sys

import yfinance as yf

from generate_real_dataset import (
    build_fundamentals_blocks,
    fetch_ticker_fundamentals_history,
)


def main():
    if len(sys.argv) != 3:
        print("Usage: python spot_check.py TICKER YYYY-MM-DD")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    try:
        as_of_date = datetime.datetime.strptime(sys.argv[2], "%Y-%m-%d")
    except ValueError:
        print(f"Bad date {sys.argv[2]!r} - expected YYYY-MM-DD")
        sys.exit(1)

    print(f"=== {ticker} as of {as_of_date.date()} ===\n")

    ticker_obj = yf.Ticker(ticker)
    fundamentals_history = fetch_ticker_fundamentals_history(ticker_obj)

    print("--- Raw fetched fundamentals (as of this call, not point-in-time except where noted) ---")
    for key in (
        "sector", "payout_ratio", "free_cash_flow", "dividend_rate",
        "growth_0y", "growth_1y", "growth_0y_low", "growth_0y_high",
        "book_value_per_share", "operating_margin", "currency",
        "financial_currency", "pe_forward",
    ):
        print(f"  {key}: {fundamentals_history.get(key)}")

    print()
    market_data_block, valuation_block_text, earnings_block = build_fundamentals_blocks(
        ticker_obj, fundamentals_history, as_of_date,
    )

    print("--- Recomputed market_data block ---")
    print(market_data_block)
    print()
    print("--- Recomputed valuation block ---")
    print(valuation_block_text)
    print()
    print("--- Recomputed earnings block ---")
    print(earnings_block)
    print()
    print("Compare the valuation block above against whatever the dataset row for this")
    print("ticker/date claims. A mismatch is worth digging into; a match confirms the")
    print("math for that row.")


if __name__ == "__main__":
    main()
