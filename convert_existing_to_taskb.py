"""Converts the EXISTING checked-in dataset_train_real.jsonl/dataset_val_
real.jsonl (old single-task BUY/SELL/HOLD schema, written under the
retired label_from_forward_return scheme - 63-trading-day, earnings-
truncated, +/-8% thresholds) into Task B (task="analysis") rows for the
two-stage pipeline, at ZERO Gemini cost: the Gemini-written `reasoning`/
`answer` prose in these rows describes the HEADLINE itself, which hasn't
changed - it's reused as-is. Only `news_reaction` (re-labeled via the
current label_news_reaction - single-day move as of 2026-08-20, see
generate_real_dataset.py's module docstring history item 12) and
`recommendation` (recomputed via fusion_rules.fuse(), never reused from
the old row) are new.

Real headline recovery: each old row's `news` block mixes the one real
(signal) headline in with 1-3 fixed NOISE_HEADLINES entries, shuffled
with no marker for which is which. Recovered by elimination - whichever
line's headline text is NOT a known NOISE_HEADLINES entry is the real
one, and its `[date]` prefix is the timestamp label_news_reaction needs.
Rows where this can't be uniquely resolved are dropped, not guessed at.

Consistency filter (the actual reason this needs care, not just a
mechanical field rename): the measurement window changed completely (63
trading days -> now a single day, via two intermediate redesigns), so the
OLD row's Gemini-written reasoning - written to justify a conclusion drawn
from a 63-day price move - has no guarantee of still matching what fuse()
computes from a freshly re-labeled single-day reaction. Rather than
pairing possibly-inconsistent prose with a new label, this script KEEPS a
converted row only when the old row's original recommendation happens to
already agree with fuse()'s fresh output - report the match rate; a low
rate is real, informative signal about how much the redesign actually
changed labeling, not a bug to "fix" by forcing a keep.

Zero Gemini cost, zero new dataset regeneration - read-only against
yfinance (for label_news_reaction's price history) and the two existing
checked-in files. Doesn't touch generate_real_dataset.py's own output
files.

Usage:
    python convert_existing_to_taskb.py
"""

import datetime
import json
import re
from collections import Counter

import yfinance as yf

import fusion_rules
from generate_real_dataset import (
    NOISE_HEADLINES,
    fetch_ticker_fundamentals_history,
    label_news_reaction,
    price_context_block,
    _signed_gap_pct,
)

_NEWS_LINE_RE = re.compile(r"^- \[(?P<date>[A-Za-z]{3}, \d{2} [A-Za-z]{3} \d{4})\] (?P<title>.+) - [^-]+$")


def _parse_news_block_date(news_block):
    """Returns the real (non-noise) headline's published date, or None if
    it can't be uniquely resolved (zero or more than one non-noise line -
    the latter would mean a genuinely ambiguous row, since normal
    generation only ever embeds exactly one real headline per row)."""
    candidates = []
    for line in news_block.splitlines():
        m = _NEWS_LINE_RE.match(line)
        if not m:
            continue
        if m.group("title") in NOISE_HEADLINES:
            continue
        candidates.append(m.group("date"))
    if len(candidates) != 1:
        return None
    return datetime.datetime.strptime(candidates[0], "%a, %d %b %Y").date()


def convert_file(in_path, out_path):
    with open(in_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    dropped = Counter()
    converted = []
    fundamentals_cache = {}
    ticker_obj_cache = {}

    for row in rows:
        try:
            impacted = json.loads(row["output"])["impacted_stocks"][0]
            old_recommendation = impacted["recommendation"]
            reasoning = impacted["reasoning"]
            answer = impacted["answer"]
        except (KeyError, IndexError, json.JSONDecodeError):
            dropped["unparseable_output"] += 1
            continue

        published_date = _parse_news_block_date(row["news"])
        if published_date is None:
            dropped["unresolvable_headline"] += 1
            continue

        ticker = row["ticker"]
        if ticker not in ticker_obj_cache:
            ticker_obj_cache[ticker] = yf.Ticker(ticker)
            fundamentals_cache[ticker] = fetch_ticker_fundamentals_history(ticker_obj_cache[ticker])
        ticker_obj = ticker_obj_cache[ticker]
        earnings_dates = fundamentals_cache[ticker].get("earnings_dates")

        published_at = datetime.datetime.combine(published_date, datetime.time(12, 0), tzinfo=datetime.timezone.utc)
        reaction, move_1d, _move_21d, skip_reason = label_news_reaction(ticker_obj, published_at, earnings_dates)
        if reaction is None:
            dropped[f"relabel_failed:{skip_reason}"] += 1
            continue

        gap_pct = _signed_gap_pct(row["valuation"])
        fusion_result = fusion_rules.fuse(reaction, gap_pct)

        if fusion_result.recommendation != old_recommendation:
            dropped["recommendation_mismatch"] += 1
            continue

        converted.append({
            "task": "analysis",
            "ticker": ticker,
            "user_query": row["user_query"],
            "price_context": price_context_block(ticker, move_1d),
            "market_data": row["market_data"],
            "valuation": row["valuation"],
            "earnings": row["earnings"],
            "news": row["news"],
            "news_reaction": reaction,
            "recommendation": fusion_result.recommendation,
            "output": json.dumps({"reasoning": reasoning, "answer": answer}, indent=2),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in converted:
            f.write(json.dumps(r) + "\n")

    print(f"{in_path} -> {out_path}: {len(converted)}/{len(rows)} converted, dropped {dict(dropped)}", flush=True)
    return len(converted), len(rows)


def main():
    convert_file("dataset_train_real.jsonl", "dataset_train_real_taskb.jsonl")
    convert_file("dataset_val_real.jsonl", "dataset_val_real_taskb.jsonl")


if __name__ == "__main__":
    main()
