"""Builds a markdown report of all 40 real-dataset tickers' CURRENT
valuation, using the exact production build_fundamentals_blocks() /
build_scenarios() code path (not a reimplementation) - so this reflects
whatever the latest valuation.py-equivalent logic actually produces,
including today's consensus-offset-cap fix.

Zero Gemini cost: yfinance only. Read-only: doesn't touch
dataset_train_real.jsonl/dataset_val_real.jsonl or any generator script.

Usage:
    python generate_valuation_report.py

Writes valuation_report.md in the current directory.
"""

import datetime
import re

import yfinance as yf

from generate_real_dataset import (
    TICKERS,
    fetch_ticker_fundamentals_history,
    build_fundamentals_blocks,
)

GAP_RE = re.compile(r"Intrinsic Value \(([^)]+)\): \$([\d,.]+)\nvs Current Price: (overvalued|undervalued) by [~>](\d+)%")
PRICE_RE = re.compile(r"Price: \$([\d,.]+)")
PE_RE = re.compile(r"P/E \(trailing\): ([\d.]+|N/A) \| P/E \(forward\): ([\d.]+|N/A)")
SECTOR_RE = re.compile(r"Sector Median P/E: [\d.]+ \(([^)]+)\)")


def report_row(ticker, name):
    ticker_obj = yf.Ticker(ticker)
    as_of_date = datetime.date.today()
    fundamentals_history = fetch_ticker_fundamentals_history(ticker_obj)
    market_data, valuation, earnings = build_fundamentals_blocks(ticker_obj, fundamentals_history, as_of_date)

    price_m = PRICE_RE.search(market_data)
    pe_m = PE_RE.search(market_data)
    sector_m = SECTOR_RE.search(market_data)
    gap_m = GAP_RE.search(valuation)

    price = price_m.group(1) if price_m else "N/A"
    pe_trailing = pe_m.group(1) if pe_m else "N/A"
    pe_forward = pe_m.group(2) if pe_m else "N/A"
    sector = sector_m.group(1) if sector_m else "N/A"

    if gap_m:
        basis, intrinsic, verdict, pct = gap_m.groups()
        # Spelled out ("overvalued 56%"), not a signed number - a signed
        # +/-% is genuinely ambiguous (two opposite real conventions exist:
        # "mispricing direction" where + = overvalued, vs "upside/downside
        # to fair value" where + = undervalued) and was misread as the
        # wrong one twice. Words can't be misread either way.
        gap_label = f"{verdict} {pct}%"
    else:
        basis, intrinsic, verdict, pct, gap_label = "N/A", "N/A", "N/A", "N/A", "N/A"

    return {
        "ticker": ticker, "name": name, "price": price, "pe_trailing": pe_trailing,
        "pe_forward": pe_forward, "sector": sector, "basis": basis, "intrinsic": intrinsic,
        "verdict": verdict, "pct": pct, "gap_label": gap_label,
        "market_data": market_data, "valuation": valuation, "earnings": earnings,
    }


def main():
    rows = []
    for ticker, name in TICKERS:
        print(f"  Fetching {ticker} ({name})...", flush=True)
        try:
            rows.append(report_row(ticker, name))
        except Exception as e:
            print(f"    Warning: {ticker!r} failed: {e}", flush=True)
            rows.append({
                "ticker": ticker, "name": name, "price": "ERROR", "pe_trailing": "-",
                "pe_forward": "-", "sector": "-", "basis": "-", "intrinsic": "-",
                "verdict": "-", "pct": "-", "signed_pct": "-",
                "market_data": f"Fetch failed: {e}", "valuation": "-", "earnings": "-",
            })

    def sort_key(r):
        try:
            return -abs(int(r["pct"]))
        except (ValueError, TypeError):
            return 0

    sorted_rows = sorted(rows, key=sort_key)

    with open("valuation_report.md", "w", encoding="utf-8") as out:
        out.write(f"# Valuation report - all {len(TICKERS)} real-dataset tickers\n\n")
        out.write(f"Generated {datetime.date.today().isoformat()}, using the current "
                   "build_fundamentals_blocks()/build_scenarios() code (includes the "
                   "2026-08-17 consensus-offset-cap fix). Sorted by |mispricing %|, "
                   "largest first - the tickers most worth sanity-checking are at the top.\n\n")
        out.write("| Ticker | Name | Price | Intrinsic ($basis) | Gap | Sector | P/E (T/F) |\n")
        out.write("|---|---|---|---|---|---|---|\n")
        for r in sorted_rows:
            out.write(f"| {r['ticker']} | {r['name']} | {r['price']} | "
                       f"{r['intrinsic']} ({r['basis']}) | {r['gap_label']} | "
                       f"{r['sector']} | {r['pe_trailing']}/{r['pe_forward']} |\n")

        out.write("\n---\n\n## Full detail per ticker\n\n")
        for r in sorted_rows:
            out.write(f"### {r['ticker']} ({r['name']})\n\n")
            out.write(f"**Market Data:**\n```\n{r['market_data']}\n```\n\n")
            out.write(f"**Valuation:**\n```\n{r['valuation']}\n```\n\n")
            out.write(f"**Earnings:**\n```\n{r['earnings']}\n```\n\n")

    print(f"\nWrote valuation_report.md: {len(rows)} tickers.")


if __name__ == "__main__":
    main()
