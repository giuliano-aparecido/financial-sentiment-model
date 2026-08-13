"""
Financial sentiment training dataset generator (synthetic)

Generates synthetic (news, user_query, output) examples for fine-tuning a
model to predict a stock's directional sentiment (BULLISH/BEARISH/NEUTRAL)
from recent news headlines and financial results. Designed to run standalone
in Google Colab - no dependencies beyond the standard library.

This is a rewrite of an earlier 1000-example generator. Changes 1-8 below
predate this repo's history; change 9 is the most recent, added after the
first trained model's eval results came back at 86.0% direction accuracy on
this dataset's own held-out val split but only 25.0% on the real-headline
supplement (generate_real_dataset.py) - a real, below-random-chance gap,
not just "real data is harder." A likely contributor: this dataset's
BULLISH/BEARISH templates all use blatant, textbook-obvious signal language
("beat consensus by X%", "lowered guidance"), so the model may have learned
"if the signal isn't textbook-clear, default to NEUTRAL" - a habit that
costs the most on real headlines, which are almost never phrased that
plainly. See docs/training-results-analysis.md and docs/dataset-fix-plan.md
for the full diagnosis.

1. News is formatted as multi-headline blocks - `- [date] headline - Publisher`,
   one per line, several headlines per example - matching exactly what the
   production API's fetch_live_news_rag() actually feeds the model (Google
   News RSS entries), instead of one flowing paragraph. Most examples mix in
   1-3 headlines that are irrelevant market noise, since real news feeds
   always contain some and the model needs to learn to weigh the relevant
   headline(s), not average across everything present.
2. Every example now includes a `user_query` field, matching the
   `User Question:` field the inference prompt gained after
   financial-sentiment-api PR #2. Training without this field means the
   model never learns what to do with it at inference time.
3. Added a MIXED_SIGNAL scenario category (beat-but-cut-guidance, miss-but-
   buyback, etc.) - the original dataset had zero examples where the model
   needs to weigh conflicting signals, which is exactly the case the
   inference prompt's remaining CRITICAL SENTIMENT RULE tries to hard-code
   ("weigh guidance cuts/revenue misses higher than minor operational
   wins"). Once the model has learned this from real examples, that rule
   can be dropped from the prompt too.
4. Confidence is sampled per ambiguity tier (clean single-signal / mixed /
   neutral) instead of one flat random range - previously it was pure noise
   uncorrelated with anything in the input.
5. Company list expanded (10 -> 26, including 6 invented tickers) with
   per-company plausible revenue bands, so a $10B quarter isn't generated
   for both Apple and a small-cap, and the model can't memorize "NVDA is
   usually bullish" instead of reading the actual news.
6. NEUTRAL raised from 10% (2 templates) to ~28% (8 templates) to match how
   often real news is genuinely non-eventful.
7. The eval split is held out by TEMPLATE and TICKER, not a random row
   split - with this much template repetition a random split leaks near-
   duplicates into validation and produces a flattering, meaningless
   accuracy number.
8. Template count roughly doubled per category (8->16 BULLISH/BEARISH/
   NEUTRAL, 6->12 MIXED) after a first training run showed a real train/
   eval loss gap (0.12 vs ~0.4) - with only 6-8 templates repeated across
   3 epochs, the model had a real opportunity to memorize specific
   template phrasings rather than the underlying news-pattern concept.
   More templates at the same NUM_EXAMPLES halves the average repetition
   per template without needing a bigger dataset.
9. Added a "subtle" tier of 8 more BULLISH and 8 more BEARISH templates
   alongside the existing 16 ("clear" tier, unchanged) each - phrased the
   way real headlines actually read (options positioning, analyst estimate
   trims, investor-day reception, insider filings, channel checks, a
   competitor's stumble) rather than declaring the outcome outright ("beat
   by X%", "lowered guidance"). These still resolve decisively to
   BULLISH/BEARISH - never to NEUTRAL - with the reasoning text explicitly
   naming the softer nature of the signal, so the model learns "less
   obvious does not mean give up and hedge," the opposite of the pattern
   the eval results suggest it picked up. Subtle examples get their own,
   lower confidence range (see CONFIDENCE_RANGES) since a softer signal
   genuinely warrants less certainty than a blatant one, without collapsing
   all the way to NEUTRAL-range confidence. held_out_template_indices was
   also changed from a tail-slice (last 20% of each list) to an
   evenly-spaced sample across the whole list, so the held-out val split
   for BULLISH/BEARISH now includes a representative mix of both tiers
   instead of accidentally holding out only whichever tier happens to be
   appended last.
10. v4: added `market_data`, `valuation`, `earnings` fields to each row and
    `answer` to the output JSON - the "full analyst pipeline" expansion.
    Each company's synthetic fundamentals (price, P/E, EPS, dividend yield,
    52-week range, book value/share) are derived from a single per-ticker
    random price draw (PRICE_RANGES) so a row's market_data/valuation/
    earnings blocks stay internally consistent rather than being
    independently-rolled numbers that could contradict each other. Valuation
    is a real Graham Number computation (sqrt(22.5 x EPS x book value/share))
    against the same synthetic price - never LLM-generated or hand-waved.
    Earnings beat/miss direction matches the example's resolved sentiment
    direction. Each block independently has a DATA_UNAVAILABLE_PROB chance
    of rendering as "Data unavailable." so the model is trained on, not just
    hoped to handle, partial production data gaps. `answer` is a templated
    1-sentence direct response to `user_query`, looked up by (question type,
    resolved direction) from ANSWER_TEMPLATES.
11. Replaced the Graham Number valuation formula with the same scenario-DCF
    model financial-sentiment-api's valuation.py actually serves (ported,
    not imported - see that block's own comment). Two real, confirmed-live
    problems this fixes: (a) production's Valuation block had drifted to a
    different label/methodology ("Revenue-based"/"EPS-based"/"FCF-based"/
    "Dividend-based") than every training example ever showed ("Graham
    Number") - the model was reasoning about a block shape it had never
    seen trained; (b) more fundamentally, render_valuation's inputs were
    rolled completely independently of `direction` in 100% of prior
    training data, so the model had no basis to ever learn to weigh
    valuation at all (confirmed live in production: a FLUT query with a
    heavily-undervalued reading still resolved non-bullish). Added the new
    VALUATION_SIGNAL scenario category (see its own comment) specifically
    to correlate the two in a controlled way - teaching signal-weighing,
    not "big gap always wins," the same principle MIXED_SIGNAL_SCENARIOS
    already applies to conflicting headlines.

Output schema (v4) - {"ticker", "user_query", "market_data", "valuation",
"earnings", "news", "output"} where output is
{"impacted_stocks": [{"ticker", "reasoning", "direction", "confidence",
"answer"}]}. Matches the canonical prompt template shared with
generate_real_dataset.py, colab/train/gpu/train_model.py,
colab/train/tpu/train_model.py, and financial-sentiment-api's
app/services/inference.py - keep all in sync
(see CONTRIBUTING.md's 4-way sync rule).

Output: two JSONL files (one JSON object per line), train and val.
"""

import datetime
import json
import random

random.seed(42)  # reproducible across Colab runs

NUM_EXAMPLES = 2000  # up from 1000 - more scenario categories need more rows
                      # to keep per-scenario-per-ticker counts reasonable
VAL_HOLDOUT_TICKERS = {"META", "BA", "QRNL", "HRZN"}          # never seen in training
VAL_HOLDOUT_TEMPLATE_INDEX_FRACTION = 0.2                       # ~20% of each
                                                                  # category's templates
                                                                  # are validation-only

# VALUATION_SIGNAL is new - carved proportionally out of the other four
# rather than added on top, so NUM_EXAMPLES still yields roughly the same
# per-category row counts as before for everything else.
SENTIMENT_WEIGHTS = {"BULLISH": 0.32, "BEARISH": 0.32, "NEUTRAL": 0.20, "MIXED": 0.08, "VALUATION_SIGNAL": 0.08}

# ---------------------------------------------------------------------------
# Companies - (ticker, name, sector, (quarterly revenue low, high in $B))
# ---------------------------------------------------------------------------
COMPANIES = [
    ("AAPL", "Apple Inc.", "tech", (75.0, 120.0)),
    ("TSLA", "Tesla Inc.", "auto", (20.0, 30.0)),
    ("NVDA", "Nvidia Corp.", "semiconductors", (18.0, 35.0)),
    ("AMZN", "Amazon.com Inc.", "e-commerce", (130.0, 170.0)),
    ("MSFT", "Microsoft Corp.", "software", (55.0, 70.0)),
    ("GOOGL", "Alphabet Inc.", "tech", (75.0, 90.0)),
    ("META", "Meta Platforms", "social-media", (32.0, 42.0)),
    ("AMD", "Advanced Micro Devices", "semiconductors", (5.0, 8.0)),
    ("JPM", "JPMorgan Chase", "banking", (38.0, 44.0)),
    ("DIS", "Walt Disney Co.", "entertainment", (20.0, 24.0)),
    ("NFLX", "Netflix Inc.", "streaming", (8.5, 10.5)),
    ("INTC", "Intel Corp.", "semiconductors", (11.0, 15.0)),
    ("CRM", "Salesforce Inc.", "software", (8.0, 9.5)),
    ("BA", "Boeing Co.", "aerospace", (15.0, 20.0)),
    ("PYPL", "PayPal Holdings", "fintech", (7.0, 8.0)),
    ("SHOP", "Shopify Inc.", "e-commerce", (1.5, 2.2)),
    ("UBER", "Uber Technologies", "gig-economy", (9.0, 11.5)),
    ("SBUX", "Starbucks Corp.", "retail", (8.5, 9.5)),
    ("COIN", "Coinbase Global", "crypto", (0.6, 1.8)),
    ("PLTR", "Palantir Technologies", "software", (0.5, 0.75)),
    # XOM/PEP were already in CURATED_SCENARIOS below (that comment claimed
    # they were "in COMPANIES above" - they weren't, a stale/wrong claim,
    # not just missing data). Meant zero synthetic training exposure for
    # either ticker while XOM specifically dominated real-eval misses
    # throughout this project's debugging history. Added for real now.
    ("XOM", "Exxon Mobil Corp.", "energy", (80.0, 115.0)),
    ("PEP", "PepsiCo Inc.", "consumer-staples", (20.0, 25.0)),
]

# Invented tickers so the model can't fall back on per-ticker priors learned
# from real-company name recognition - it has to actually read the news.
INVENTED_COMPANIES = [
    ("ZVEX", "Zenvex Dynamics", "robotics", (2.0, 4.0)),
    ("QRNL", "Quorinal Biotech", "biotech", (0.3, 1.2)),
    ("FLTX", "Flotix Logistics", "logistics", (3.0, 6.0)),
    ("NMBS", "Numbus Cloud", "software", (1.0, 3.0)),
    ("VLTR", "Voltrix Energy", "energy", (4.0, 9.0)),
    ("HRZN", "Horizon Materials", "industrials", (2.5, 5.5)),
]

ALL_COMPANIES = COMPANIES + INVENTED_COMPANIES

# Maps each company's informal sector label (COMPANIES/INVENTED_COMPANIES'
# 3rd tuple field, e.g. "e-commerce") to the yfinance Ticker.info["sector"]
# string classify_valuation_basis below actually branches on (see that
# function's docstring - REIT_SECTORS/ASSET_HEAVY_SECTORS are yfinance
# sector strings, not this dataset's own informal labels). No company here
# maps to "Real Estate" - this dataset has no REIT, so that branch is
# untested by synthetic data (a known, documented gap, not an oversight).
SECTOR_MAP = {
    "tech": "Technology",
    "auto": "Consumer Cyclical",
    "semiconductors": "Technology",
    "e-commerce": "Consumer Cyclical",
    "software": "Technology",
    "social-media": "Communication Services",
    "banking": "Financial Services",
    "entertainment": "Communication Services",
    "streaming": "Communication Services",
    "fintech": "Financial Services",
    "gig-economy": "Technology",
    "retail": "Consumer Cyclical",
    "crypto": "Financial Services",
    "robotics": "Industrials",
    "biotech": "Healthcare",
    "logistics": "Industrials",
    "energy": "Energy",
    "industrials": "Industrials",
    "aerospace": "Industrials",
    "consumer-staples": "Consumer Defensive",
}

# Per-ticker plausible price bands ($) - the single anchor each company's
# other synthetic fundamentals (EPS, book value/share, market cap, 52-week
# range) are derived from, so a given example's market_data/valuation/
# earnings blocks stay internally consistent with each other instead of
# being independently rolled numbers that could contradict.
PRICE_RANGES = {
    "AAPL": (150.0, 220.0), "TSLA": (180.0, 280.0), "NVDA": (90.0, 160.0),
    "AMZN": (140.0, 220.0), "MSFT": (350.0, 470.0), "GOOGL": (140.0, 200.0),
    "META": (400.0, 600.0), "AMD": (100.0, 180.0), "JPM": (180.0, 260.0),
    "DIS": (85.0, 130.0), "NFLX": (550.0, 750.0), "INTC": (18.0, 35.0),
    "CRM": (230.0, 330.0), "BA": (150.0, 220.0), "PYPL": (55.0, 90.0),
    "SHOP": (60.0, 100.0), "UBER": (60.0, 95.0), "SBUX": (75.0, 110.0),
    "COIN": (150.0, 280.0), "PLTR": (25.0, 45.0),
    "XOM": (100.0, 160.0), "PEP": (150.0, 185.0),
    "ZVEX": (15.0, 40.0), "QRNL": (5.0, 20.0), "FLTX": (20.0, 45.0),
    "NMBS": (10.0, 30.0), "VLTR": (25.0, 60.0), "HRZN": (15.0, 35.0),
}

# Chance a given block is rendered as "Data unavailable." instead of real
# content, applied independently per block - mirrors production, where
# fundamentals/earnings/news are three separate fetches that can each fail
# on their own (see app/services/fundamentals.py and earnings.py in
# financial-sentiment-api). Trained on and served identically so the model
# learns to degrade gracefully rather than only being hoped to.
DATA_UNAVAILABLE_PROB = 0.15

# Chance a company's synthetic trailing EPS is negative this example -
# forces the valuation basis classifier into its "revenue" path (see
# classify_valuation_basis below), matching the real valuation.py's
# behavior for loss-making companies. Kept low since most real large/
# mid-caps are profitable most quarters.
LOSS_MAKING_PROB = 0.08

# ---------------------------------------------------------------------------
# Scenario-DCF valuation model - ported from financial-sentiment-api's
# app/services/valuation.py (see that module's own history-of-rejected-
# approaches comment for why it looks like this), NOT the old Graham Number
# formula this generator used through v4. That mismatch was a real, live
# bug: production's Valuation block has rendered "Intrinsic Value
# (Revenue-based/EPS-based/FCF-based/Dividend-based)" since valuation.py's
# scenario-DCF rewrite, but every training example still showed "Intrinsic
# Value (Graham Number)" - a label and methodology the model never saw
# trained, on a block it was nonetheless supposed to be able to reason
# about. Ported (not imported) because this repo and financial-sentiment-api
# are separate repos with no shared package - CONTRIBUTING.md's 4-way sync
# rule already requires the market_data/earnings block renderers to be
# hand-kept-identical the same way; this extends that to valuation. Keep
# this block byte-for-byte in step with valuation.py's own constants/
# functions on any future change there.
STAGE_1_YEARS = 5
STAGE_2_YEARS = 5

ASSET_HEAVY_SECTORS = {"Energy", "Industrials", "Basic Materials", "Utilities"}
DIVIDEND_PAYOUT_THRESHOLD = 0.40
DIVIDEND_PAYOUT_CEILING = 1.20
REIT_SECTORS = {"Real Estate"}

BASIS_LABELS = {
    "revenue": "Revenue-based",
    "eps": "EPS-based",
    "fcf": "FCF-based",
    "dividends": "Dividend-based",
}

DISCOUNT_RATE = 0.10
SCENARIO_PROBABILITY = 1 / 3

# Same curated per-ticker DCF assumptions as valuation.py - NVDA/MSFT/PEP/
# NFLX/XOM are all in COMPANIES above, so synthetic examples for those five
# tickers get the exact same scenario assumptions production would use for
# them, instead of the generic/derived fallback every other ticker gets.
CURATED_SCENARIOS = {
    "NVDA": {
        "normal": {"g1": 0.30, "g2": 0.10, "exit_multiple": 20.0},
        "best": {"g1": 0.30, "g2": 0.15, "exit_multiple": 25.0},
        "worst": {"g1": 0.05, "g2": 0.05, "exit_multiple": 10.0},
    },
    "MSFT": {
        "normal": {"g1": 0.15, "g2": 0.10, "exit_multiple": 20.0},
        "best": {"g1": 0.20, "g2": 0.10, "exit_multiple": 25.0},
        "worst": {"g1": 0.05, "g2": 0.05, "exit_multiple": 12.0},
    },
    "PEP": {
        "normal": {"g1": 0.03, "g2": 0.03, "exit_multiple": 20.0},
        "best": {"g1": 0.05, "g2": 0.05, "exit_multiple": 25.0},
        "worst": {"g1": 0.03, "g2": -0.05, "exit_multiple": 15.0},
    },
    "NFLX": {
        "normal": {"g1": 0.12, "g2": 0.10, "exit_multiple": 20.0},
        "best": {"g1": 0.15, "g2": 0.12, "exit_multiple": 25.0},
        "worst": {"g1": 0.08, "g2": 0.06, "exit_multiple": 15.0},
    },
    "XOM": {
        "normal": {"g1": 0.04, "g2": 0.04, "exit_multiple": 20.0},
        "best": {"g1": 0.06, "g2": 0.06, "exit_multiple": 30.0},
        "worst": {"g1": 0.03, "g2": 0.03, "exit_multiple": 12.0},
    },
}

WORST_EXIT_MULTIPLE_ASSET_HEAVY = 12.0
WORST_EXIT_MULTIPLE_DEFAULT = 13.0
NORMAL_EXIT_MULTIPLE = 20.0
BEST_EXIT_MULTIPLE = 25.0

# P/S-style multiples for the "revenue" basis specifically - see
# financial-sentiment-api PR (fix: revenue-basis valuation reused
# earnings-grade exit multiples) for why these can't share the eps/fcf/
# dividends multiples above (confirmed live: FLUT came back "94%
# undervalued" when routed to "revenue" with the old shared 20-25x
# multiples). Ported here so synthetic training data reflects the fix too,
# not just production.
REVENUE_WORST_EXIT_MULTIPLE = 1.0
REVENUE_NORMAL_EXIT_MULTIPLE = 3.0
REVENUE_BEST_EXIT_MULTIPLE = 6.0

GROWTH_BASIS_G2 = {"normal": 0.10, "best": 0.12, "worst": 0.04}
G1_FALLBACK = {"normal": 0.08, "best": 0.10, "worst": 0.04}


def classify_valuation_basis(eps_trailing, payout_ratio, sector, free_cash_flow):
    """See financial-sentiment-api's valuation.py for the full rationale -
    ported verbatim. Real Estate -> dividends; unprofitable/unknown ->
    revenue; high-but-plausible payout -> dividends; asset-heavy sector with
    real positive FCF -> fcf; otherwise -> eps."""
    if sector in REIT_SECTORS:
        return "dividends"
    if eps_trailing is None or eps_trailing <= 0:
        return "revenue"
    if payout_ratio is not None and DIVIDEND_PAYOUT_THRESHOLD <= payout_ratio <= DIVIDEND_PAYOUT_CEILING:
        return "dividends"
    if sector in ASSET_HEAVY_SECTORS and free_cash_flow is not None and free_cash_flow > 0:
        return "fcf"
    return "eps"


def _shares_outstanding_approx(market_cap, price):
    if not market_cap or not price:
        return None
    return market_cap / price


def cash_flow_basis_value(basis, fnd):
    """Extracts the per-share cash-flow figure for the classified basis -
    ported from valuation.py's identically-named function, adapted to this
    generator's fnd dict field names."""
    if basis == "eps":
        return fnd.get("eps_trailing")
    if basis == "dividends":
        return fnd.get("dividend_rate")

    shares = _shares_outstanding_approx(fnd.get("market_cap"), fnd.get("price"))
    if not shares:
        return None
    if basis == "revenue":
        revenue = fnd.get("total_revenue")
        return revenue / shares if revenue else None
    if basis == "fcf":
        fcf = fnd.get("free_cash_flow")
        return fcf / shares if fcf else None
    return None


def build_scenarios(ticker, fnd, basis):
    """Ported from valuation.py's identically-named function - see that
    module for the full rationale behind each piece."""
    if ticker and ticker in CURATED_SCENARIOS:
        return {
            name: {**scenario, "probability": SCENARIO_PROBABILITY}
            for name, scenario in CURATED_SCENARIOS[ticker].items()
        }

    g1_values = dict(G1_FALLBACK)
    growth_0y = fnd.get("growth_0y")
    growth_1y = fnd.get("growth_1y")
    consensus_reliable = growth_0y is not None and growth_1y is not None and (growth_0y >= 0) == (growth_1y >= 0)
    if consensus_reliable:
        g1_values["normal"] = (growth_0y + growth_1y) / 2
        growth_0y_high = fnd.get("growth_0y_high")
        growth_0y_low = fnd.get("growth_0y_low")
        if growth_0y_high is not None:
            g1_values["best"] = growth_0y_high
        if growth_0y_low is not None:
            g1_values["worst"] = growth_0y_low

    g2_values = dict(g1_values) if basis == "dividends" else dict(GROWTH_BASIS_G2)

    if basis == "revenue":
        exit_multiples = {
            "normal": REVENUE_NORMAL_EXIT_MULTIPLE,
            "best": REVENUE_BEST_EXIT_MULTIPLE,
            "worst": REVENUE_WORST_EXIT_MULTIPLE,
        }
    else:
        worst_exit_multiple = (
            WORST_EXIT_MULTIPLE_ASSET_HEAVY if fnd.get("sector") in ASSET_HEAVY_SECTORS
            else WORST_EXIT_MULTIPLE_DEFAULT
        )
        exit_multiples = {"normal": NORMAL_EXIT_MULTIPLE, "best": BEST_EXIT_MULTIPLE, "worst": worst_exit_multiple}

    return {
        name: {
            "g1": g1_values[name],
            "g2": g2_values[name],
            "exit_multiple": exit_multiples[name],
            "probability": SCENARIO_PROBABILITY,
        }
        for name in ("normal", "best", "worst")
    }


def scenario_dcf_value(cf0, g1, g2, exit_multiple, discount_rate):
    """Full-sum (interim years + terminal), used only for "dividends" -
    ported verbatim from valuation.py."""
    pv = 0.0
    cf = cf0
    for year in range(1, STAGE_1_YEARS + 1):
        cf *= 1 + g1
        pv += cf / (1 + discount_rate) ** year
    for year in range(STAGE_1_YEARS + 1, STAGE_1_YEARS + STAGE_2_YEARS + 1):
        cf *= 1 + g2
        pv += cf / (1 + discount_rate) ** year
    terminal_value = cf * exit_multiple
    pv += terminal_value / (1 + discount_rate) ** (STAGE_1_YEARS + STAGE_2_YEARS)
    return pv


def scenario_terminal_value(cf0, g1, g2, exit_multiple, discount_rate):
    """Terminal-only (no interim summation), used for "eps"/"fcf"/
    "revenue" - ported verbatim from valuation.py."""
    future_cf = cf0 * (1 + g1) ** STAGE_1_YEARS * (1 + g2) ** STAGE_2_YEARS
    terminal_value = future_cf * exit_multiple
    return terminal_value / (1 + discount_rate) ** (STAGE_1_YEARS + STAGE_2_YEARS)


def _scenario_pv(basis, cf0, g1, g2, exit_multiple, discount_rate):
    if basis == "dividends":
        return scenario_dcf_value(cf0, g1, g2, exit_multiple, discount_rate)
    return scenario_terminal_value(cf0, g1, g2, exit_multiple, discount_rate)


def scenario_present_values(cf0, basis, scenarios):
    return {
        name: _scenario_pv(basis, cf0, scenario["g1"], scenario["g2"], scenario["exit_multiple"], DISCOUNT_RATE)
        for name, scenario in scenarios.items()
    }


def intrinsic_value(cf0, basis, scenarios):
    """None (not a fetch failure) when cf0 is missing or non-positive -
    ported verbatim from valuation.py."""
    if cf0 is None or cf0 <= 0:
        return None
    pvs = scenario_present_values(cf0, basis, scenarios)
    return sum(scenario["probability"] * pvs[name] for name, scenario in scenarios.items())


def valuation_block(price, intrinsic, basis):
    """Renders the 'Valuation' prompt block - byte-identical to
    valuation.py's identically-named function, since this IS what the model
    is trained on and served against."""
    label = BASIS_LABELS[basis]
    if intrinsic is None:
        return f"Not applicable (insufficient data for the {label.lower()} valuation basis)."
    if price is None:
        return "Data unavailable."

    pct = (price - intrinsic) / intrinsic * 100
    verdict = "overvalued" if pct >= 0 else "undervalued"
    return (
        f"Intrinsic Value ({label}): ${intrinsic:.2f}\n"
        f"vs Current Price: {verdict} by ~{abs(pct):.0f}%"
    )

PUBLISHERS = [
    "Reuters", "Bloomberg", "MarketWatch", "CNBC", "Yahoo Finance",
    "Barron's", "The Motley Fool", "Seeking Alpha", "Business Insider", "AP News",
]

PRODUCTS = [
    "AI infrastructure", "cloud services", "next-gen hardware", "enterprise software",
    "flagship consumer devices", "streaming platform", "payments network",
    "electric vehicle lineup", "chip manufacturing", "logistics network",
]

REGIONS = ["Southeast Asian", "European", "Latin American", "Indian", "Middle Eastern"]
ROLES = ["Chief Marketing Officer", "VP of Engineering", "Chief Operating Officer", "Head of AI Research"]

# Market-noise headlines unrelated to the target ticker's direction - mixed
# into most examples since real feeds always contain some.
NOISE_HEADLINES = [
    "Fed leaves interest rates unchanged in split decision",
    "Oil prices dip amid oversupply concerns",
    "Treasury yields climb on inflation data",
    "Dollar strengthens against euro after jobs report",
    "Wall Street mixed as investors await earnings season",
    "Gold prices hold steady near record highs",
    "Asian markets close lower on trade tension fears",
    "Consumer confidence index ticks up slightly",
    "Housing starts fall for third consecutive month",
    "Retail sales beat expectations in broad-based gain",
    "Crude oil rebounds on OPEC+ supply cut signals",
    "Bond markets rally as recession fears ease",
]

# User questions matching the inference prompt's "User Question:" field.
# Mix of cashtag / plain-ticker / vague phrasing, since that's what real
# users actually type - see app/services/ticker.py's extract_ticker, which
# only reliably resolves a $CASHTAG or a bare capitalized ticker.
USER_QUESTION_TEMPLATES = [
    "Will ${ticker} go up or down based on recent news?",
    "Is {ticker} a buy right now?",
    "What's the outlook for {ticker} this quarter?",
    "Should I be worried about my {ticker} position?",
    "How will the latest news affect ${ticker}?",
    "Is now a good time to sell {ticker}?",
    "What's the sentiment on {ticker} today?",
    "Will {ticker} beat earnings expectations?",
    "Give me your read on {ticker}.",
    "",  # some users submit with no real question at all
]

# Index-aligned with USER_QUESTION_TEMPLATES above - each question template
# maps to exactly one question "type" used to pick the matching answer below.
QUESTION_TYPES = [
    "direction", "buy", "outlook", "worry", "impact",
    "sell", "sentiment", "earnings", "read", "none",
]

# answer text for the new `answer` output field - direct response to
# `user_query`, consistent with the resolved direction (BULLISH/BEARISH/
# NEUTRAL; MIXED_SIGNAL_SCENARIOS resolves to one of these before this
# lookup happens, so only 3 directions are needed here). One template per
# (question type, direction) - deliberately plain/formulaic since this is
# what the model should learn to produce, not literary variety.
ANSWER_TEMPLATES = {
    "direction": {
        "BULLISH": "The recent news points to upward momentum for {ticker}, so the near-term bias leans higher.",
        "BEARISH": "The recent news points to downward pressure on {ticker}, so the near-term bias leans lower.",
        "NEUTRAL": "The recent news doesn't point clearly in either direction for {ticker}, so a flat near-term move is the more likely outcome.",
    },
    "buy": {
        "BULLISH": "Yes, the current signals lean favorably enough that {ticker} looks like a reasonable buy here.",
        "BEARISH": "No, the current signals are negative enough that {ticker} doesn't look like a buy right now.",
        "NEUTRAL": "It's a close call - nothing here strongly argues for or against buying {ticker} at current levels.",
    },
    "outlook": {
        "BULLISH": "The outlook for {ticker} this quarter looks positive based on the latest developments.",
        "BEARISH": "The outlook for {ticker} this quarter looks challenged based on the latest developments.",
        "NEUTRAL": "The outlook for {ticker} this quarter looks steady, without a clear positive or negative catalyst.",
    },
    "worry": {
        "BULLISH": "No significant cause for concern - the latest news on {ticker} is constructive.",
        "BEARISH": "Some caution is warranted - the latest news on {ticker} raises real concerns.",
        "NEUTRAL": "Not particularly - nothing in the latest news materially changes the risk picture for {ticker}.",
    },
    "impact": {
        "BULLISH": "The latest news should be a net positive for {ticker}.",
        "BEARISH": "The latest news should weigh on {ticker}.",
        "NEUTRAL": "The latest news is unlikely to move {ticker} much either way.",
    },
    "sell": {
        "BULLISH": "Not really - the current signals argue for holding rather than selling {ticker}.",
        "BEARISH": "It's a reasonable moment to consider trimming {ticker}, given the negative signals.",
        "NEUTRAL": "There's no strong signal here to justify selling {ticker} now versus holding.",
    },
    "sentiment": {
        "BULLISH": "Sentiment on {ticker} is bullish today.",
        "BEARISH": "Sentiment on {ticker} is bearish today.",
        "NEUTRAL": "Sentiment on {ticker} is neutral today.",
    },
    "earnings": {
        "BULLISH": "The signals point toward {ticker} beating expectations.",
        "BEARISH": "The signals point toward {ticker} falling short of expectations.",
        "NEUTRAL": "There's no strong signal either way on whether {ticker} beats expectations.",
    },
    "read": {
        "BULLISH": "Overall, {ticker} looks bullish based on the current data and news.",
        "BEARISH": "Overall, {ticker} looks bearish based on the current data and news.",
        "NEUTRAL": "Overall, {ticker} looks balanced - no strong read either way right now.",
    },
    "none": {
        "BULLISH": "{ticker} is showing a bullish setup based on current data and news.",
        "BEARISH": "{ticker} is showing a bearish setup based on current data and news.",
        "NEUTRAL": "{ticker} looks neutral right now, without a clear directional catalyst.",
    },
}

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def random_recent_date(days_back_max=6):
    # Matches app/services/news.py's entry.get("published", "")[:16] exactly
    # (a raw RSS pubDate truncated to 16 chars, e.g. "Tue, 05 Aug 2026") -
    # training on the same date format the model sees in production.
    anchor = datetime.date(2026, 8, 5)
    d = anchor - datetime.timedelta(days=random.randint(0, days_back_max))
    return f"{WEEKDAYS[d.weekday()]}, {d.day:02d} {MONTHS[d.month - 1]} {d.year}"


STALE_HEADLINE_DAYS_BACK_MAX = 60
STALE_HEADLINE_PROB = 0.25


def format_headline(text, publisher=None, stale=False):
    date = random_recent_date(days_back_max=STALE_HEADLINE_DAYS_BACK_MAX if stale else 6)
    publisher = publisher or random.choice(PUBLISHERS)
    return f"- [{date}] {text} - {publisher}"


# ---------------------------------------------------------------------------
# Scenario templates
#
# BULLISH_SCENARIOS / BEARISH_SCENARIOS entries are 3-tuples:
# (headline_templates, reasoning_template, tier), where tier is "clear"
# (the original, blatant-signal templates) or "subtle" (real-headline-
# style, softer signal, still a decisive direction). NEUTRAL_SCENARIOS stays
# 2-tuples - there's no "subtle NEUTRAL" concept, these are meant to stay
# genuinely flat so the clear/subtle-BULLISH/BEARISH vs NEUTRAL boundary
# stays learnable. All templates use the same fill fields: {name} {ticker}
# {q} {rev} {beat} {drop} {units} {product} {region} {role} {buyback}
# {divhike} {rally} - unused fields in a given template are simply not
# referenced.
# ---------------------------------------------------------------------------

BULLISH_SCENARIOS = [
    (["{name} ({ticker}) reported Q{q} revenue of ${rev}B, beating consensus estimates by {beat}%, driven by strong {product} demand"],
     "{ticker} outperformed consensus due to strong {product} demand, with margins improving alongside the {beat}% beat.", "clear"),
    (["{name} ({ticker}) raised its full-year guidance, citing accelerating enterprise adoption of {product}"],
     "A guidance raise signals management's own confidence in sustained {product} demand - one of the strongest forward-looking bullish signals available.", "clear"),
    (["{name} ({ticker}) announced a strategic partnership to expand its {product} division into {region} markets"],
     "Geographic expansion of the {product} line opens new addressable market and increases multi-year revenue visibility for {ticker}.", "clear"),
    (["Analysts upgraded {ticker} ({name}) to 'Buy' following better-than-expected subscription growth in {product}"],
     "The upgrade reflects improving free cash flow efficiency and durable recurring revenue growth for {ticker}.", "clear"),
    (["{name} ({ticker}) announced a ${buyback}B share buyback program and raised its quarterly dividend by {divhike}%"],
     "A simultaneous buyback and dividend hike signals management confidence in sustained free cash flow generation.", "clear"),
    (["{name} ({ticker}) shares rallied {rally}% after its new {product} line sold out within hours of launch"],
     "Immediate sell-out demand for {product} is a strong, concrete signal of near-term revenue upside.", "clear"),
    (["{name} ({ticker}) received regulatory approval for its {product} expansion, clearing the path for {region} market entry"],
     "Regulatory clearance removes a key uncertainty and opens a new revenue channel via {region} market entry.", "clear"),
    (["{name} ({ticker}) announced cost-cutting measures expected to expand {product} margins by next fiscal year"],
     "Structural cost reduction targeted at {product} margins is a credible path to improved profitability.", "clear"),
    (["{name} ({ticker}) announced a 2-for-1 stock split, effective next month"],
     "A stock split often signals management's confidence in continued share price appreciation and improves retail accessibility for {ticker}.", "clear"),
    (["{name} ({ticker}) signed a multi-year contract with a major enterprise customer for its {product} offering"],
     "A large new customer commitment adds durable, contracted revenue visibility for {ticker}'s {product} business.", "clear"),
    (["Insiders at {name} ({ticker}) purchased shares on the open market following recent price weakness"],
     "Insider buying signals that those with the most information about {ticker}'s prospects see current levels as undervalued.", "clear"),
    (["{name} ({ticker}) received regulatory certification for its {product} line, clearing a key launch hurdle"],
     "Certification removes a major uncertainty and clears the path to revenue recognition for {product}.", "clear"),
    (["{name} ({ticker}) beat Q{q} estimates by {beat}% and simultaneously raised full-year guidance"],
     "Beating estimates while also raising guidance is a strong double-confirmation of accelerating {product} demand for {ticker}.", "clear"),
    (["{name} ({ticker}) refinanced its debt at a lower interest rate, reducing annual interest expense"],
     "Lower financing costs directly improve forward net margins for {ticker}.", "clear"),
    (["Analysts raised their price target on {ticker} ({name}) following strong {product} adoption trends"],
     "A raised price target reflects analysts' updated view that {product} adoption is outpacing prior expectations.", "clear"),
    (["{name} ({ticker}) completed the divestiture of its underperforming {product} unit, sharpening focus on core operations"],
     "Shedding an underperforming segment lets {ticker} redirect capital toward higher-return parts of the business.", "clear"),
    # --- subtle tier: real-headline style, softer signal, still decisive ---
    (["{name} ({ticker}) shares edged higher after a well-received product demo at an industry conference"],
     "The positive reception isn't a formal guidance change, but it signals improving market confidence in {product}'s near-term prospects.", "subtle"),
    (["{ticker} ({name}) outperformed sector peers this week following upbeat management commentary on {product} demand"],
     "Relative outperformance tied to specific positive commentary is a real, if less quantified, bullish signal for {ticker}.", "subtle"),
    (["{name} ({ticker}) expanded its {product} distribution agreement with a major retail partner, according to a regulatory filing"],
     "Incremental distribution expansion is a modest but genuine positive for {ticker}'s addressable market, even without an accompanying guidance update.", "subtle"),
    (["Options activity in {ticker} ({name}) turned notably bullish ahead of next week's earnings report"],
     "A skew toward bullish options positioning reflects market participants leaning toward a positive surprise - probabilistic, but a real signal.", "subtle"),
    (["{name} ({ticker}) shares ticked up after a well-attended investor day highlighted the {product} roadmap"],
     "Positive investor reaction to a roadmap presentation, while not a hard financial commitment, reflects improved sentiment on {ticker}'s execution.", "subtle"),
    (["A former {name} ({ticker}) executive said in an interview that {product} demand is 'stronger than the headlines suggest'"],
     "A credible insider's characterization of underlying demand, even anecdotal, tilts the outlook for {ticker} positive.", "subtle"),
    (["{ticker} ({name}) shares rose after a smaller rival's weak results highlighted {ticker}'s stronger competitive position in {product}"],
     "A competitor's stumble that specifically highlights {ticker}'s relative strength in {product} is a real, if indirect, positive signal.", "subtle"),
    (["{name} ({ticker}) saw a notable increase in institutional ownership in the latest quarterly filings"],
     "Rising institutional ownership reflects professional investors' incrementally more positive view of {ticker}, even without a specific disclosed catalyst.", "subtle"),
]

BEARISH_SCENARIOS = [
    (["{name} ({ticker}) shares dropped {drop}% after lowering its full-year guidance due to supply chain bottlenecks in {product}"],
     "The guidance cut and {product} supply constraints create direct near-term earnings headwinds for {ticker}.", "clear"),
    (["{name} ({ticker}) missed quarterly profit estimates as rising R&D expenditures and regulatory fines squeezed operating margins"],
     "Unbudgeted regulatory penalties combined with elevated operating expenditure directly erode near-term earnings for {ticker}.", "clear"),
    (["{name} ({ticker}) announced a recall affecting {units}k units of its {product} line over quality control concerns"],
     "The recall creates immediate warranty costs and brand-equity risk concentrated in the {product} line.", "clear"),
    (["Analysts downgraded {ticker} ({name}) to 'Sell', citing slowing {product} demand and margin compression"],
     "The downgrade reflects a deteriorating demand outlook for {product} and compressing unit economics.", "clear"),
    (["{name} ({ticker}) shares fell {drop}% after its CFO abruptly resigned amid an ongoing accounting review"],
     "An abrupt CFO departure during an active accounting review raises governance and financial-reporting risk.", "clear"),
    (["{name} ({ticker}) disclosed a federal investigation into its {product} business practices"],
     "An active federal investigation introduces legal and reputational risk with an uncertain, potentially material, financial outcome.", "clear"),
    (["{name} ({ticker}) lost significant {product} market share to a fast-growing competitor, per new industry data"],
     "Share loss in {product} signals weakening competitive positioning and pressure on future revenue growth.", "clear"),
    (["{name} ({ticker}) announced significant workforce reductions amid weakening demand for {product}"],
     "Reactive workforce reductions confirm weakening {product} demand rather than proactive efficiency gains.", "clear"),
    (["Insiders at {name} ({ticker}) sold significant stakes following a run-up in share price"],
     "Large insider selling can signal that those closest to {ticker}'s prospects see current valuation as stretched.", "clear"),
    (["{name} ({ticker})'s credit rating was downgraded, citing rising leverage and weakening cash flow"],
     "A credit downgrade raises {ticker}'s future borrowing costs and signals deteriorating balance sheet health.", "clear"),
    (["{name} ({ticker}) suspended its quarterly dividend to preserve cash amid weakening {product} sales"],
     "A dividend suspension is a direct signal of cash flow stress and removes a key reason income investors held {ticker}.", "clear"),
    (["{name} ({ticker}) lost a major patent infringement lawsuit related to its {product} technology"],
     "The litigation loss creates both a direct financial liability and ongoing royalty costs for {ticker}'s {product} line.", "clear"),
    (["{name} ({ticker}) disclosed a data breach affecting its {product} platform, triggering a regulatory review"],
     "A breach combined with regulatory scrutiny introduces both remediation costs and reputational risk for {ticker}.", "clear"),
    (["{name} ({ticker}) lost a major enterprise customer to a competitor, according to industry sources"],
     "Losing a significant customer relationship directly reduces {ticker}'s near-term revenue visibility.", "clear"),
    (["{name} ({ticker}) delayed the launch of its next-generation {product} line by two quarters"],
     "A launch delay pushes expected revenue further out and raises execution-risk concerns for {ticker}'s {product} roadmap.", "clear"),
    (["{name} ({ticker}) announced it will restate prior financial statements due to an accounting error"],
     "A restatement introduces uncertainty about the reliability of {ticker}'s previously reported results.", "clear"),
    # --- subtle tier: real-headline style, softer signal, still decisive ---
    (["{name} ({ticker}) shares slipped after a mixed reception to {product} updates at an investor event"],
     "A lukewarm reaction to a roadmap presentation, while not a formal guidance cut, signals softening market confidence in {product}.", "subtle"),
    (["{ticker} ({name}) underperformed sector peers this week amid soft channel checks on {product} demand"],
     "Relative underperformance tied to specific negative channel data is a real, if less quantified, bearish signal for {ticker}.", "subtle"),
    (["A closely-watched analyst trimmed estimates for {name} ({ticker}), citing early signs of slowing {product} demand"],
     "An estimate cut grounded in specific demand concerns is a credible forward-looking negative signal, even without a company-issued guidance change.", "subtle"),
    (["{name} ({ticker}) shares dipped after a well-known short-seller published a critical note on its {product} business"],
     "A public short thesis introduces a specific, researched bear case that markets are pricing in, even before any company response.", "subtle"),
    (["Options activity in {ticker} ({name}) turned notably bearish ahead of next week's earnings report"],
     "A skew toward bearish options positioning reflects market participants leaning toward a negative surprise - probabilistic, but a real signal.", "subtle"),
    (["{name} ({ticker}) shares fell after a key supplier flagged order softness that industry watchers linked to {ticker}'s {product} line"],
     "A supply-chain signal pointing to softening orders is an early, credible read on {ticker}'s near-term demand.", "subtle"),
    (["{ticker} ({name}) saw a notable increase in insider selling activity in the latest filings"],
     "A pickup in insider selling, while not conclusive on its own, reflects reduced confidence from those closest to {ticker}'s prospects.", "subtle"),
    (["A former {name} ({ticker}) executive said in an interview that {product} demand has been 'softer than the headlines suggest'"],
     "A credible insider's characterization of underlying weakness, even anecdotal, tilts the outlook for {ticker} negative.", "subtle"),
]

NEUTRAL_SCENARIOS = [
    (["{name} ({ticker}) held its annual shareholder meeting, re-electing current board members and approving standard compensation packages"],
     "Routine governance updates and board re-elections do not materially alter {ticker}'s financial outlook."),
    (["{name} ({ticker}) announced a minor realignment of internal operational segments to streamline corporate reporting"],
     "Internal restructuring without reported workforce reductions or segment sales is financially neutral for {ticker}."),
    (["{name} ({ticker}) reported Q{q} revenue of ${rev}B, in line with analyst expectations"],
     "In-line results confirm the existing outlook for {ticker} without introducing new directional information."),
    (["Analysts maintained their 'Hold' rating on {ticker} ({name}), citing balanced risk/reward at current levels"],
     "A maintained Hold rating reflects no material change in the balance of risks for {ticker}."),
    (["{name} ({ticker}) unveiled its next-generation {product} at an industry event, with pricing and availability details expected later this year"],
     "Without pricing, availability, or financial guidance attached, a product preview carries no near-term earnings implication."),
    (["{name} ({ticker}) filed its routine quarterly report with no material changes to previously issued guidance"],
     "A routine filing that reaffirms existing guidance introduces no new information."),
    (["{name} ({ticker}) named a new {role} to its leadership team, effective next quarter"],
     "A leadership appointment outside the CEO/CFO level is not typically financially material on its own."),
    (["{name} ({ticker}) made a small minority investment in a {product} startup, with terms not disclosed"],
     "An undisclosed minority stake is too small and uncertain in scope to be directionally meaningful for {ticker}."),
    (["An analyst initiated coverage on {ticker} ({name}) with a 'Neutral' rating and no strong directional view"],
     "A neutral initiation with no strong view either way introduces no new directional information for {ticker}."),
    (["{name} ({ticker}) declined to comment on market speculation regarding a potential acquisition"],
     "An unconfirmed rumor with no company statement carries no verifiable financial information for {ticker}."),
    (["An executive at {name} ({ticker}) sold shares under a pre-scheduled 10b5-1 trading plan"],
     "Pre-scheduled sales under a 10b5-1 plan are routine and don't reflect a discretionary view on {ticker}'s prospects."),
    (["{name} ({ticker}) presented at an industry conference, reiterating previously disclosed strategic priorities"],
     "Reiterating existing strategy at a conference introduces no new financial information for {ticker}."),
    (["{name} ({ticker}) settled a legal claim for an amount consistent with previously reserved funds"],
     "A settlement within already-reserved amounts has no incremental impact on {ticker}'s financial position."),
    (["{name} ({ticker}) rebranded its {product} line with a new name and visual identity"],
     "A branding update with no pricing or product changes is not typically financially material for {ticker}."),
    (["{name} ({ticker}) confirmed capital expenditure plans for {product} in line with previous guidance"],
     "Spending in line with prior guidance confirms, rather than changes, the existing outlook for {ticker}."),
    (["Analysts left their price target on {ticker} ({name}) unchanged following a routine quarterly review"],
     "An unchanged price target after a routine review reflects no material shift in analysts' view of {ticker}."),
]

# Each entry needs TWO headline templates (a positive element and a negative
# or complicating one) plus an explicit tie-break in the reasoning. This is
# the category the original dataset had zero examples of, and the one the
# inference prompt's surviving CRITICAL SENTIMENT RULE ("weigh guidance cuts
# and revenue misses higher than minor operational wins") is trying to
# hard-code in the absence of any. `resolved_direction` and `confidence_note`
# make the tie-break explicit and are what the reasoning should teach.
#
# Reasoning phrasing is deliberately varied across entries (different
# sentence order, different vocabulary for "this signal wins"/"this signal
# is minor") rather than reusing one rigid template like "X is the more
# decision-relevant signal; Y is a minor operational item" everywhere -
# confirmed live: a trained model pattern-matched that near-verbatim phrase
# strongly enough to invoke it on CLEAN, single-direction BULLISH/BEARISH
# headlines that merely had two clauses (not two conflicting ones), then
# picked the wrong direction from a tie-break that didn't apply. Varying
# the surface form is meant to make the underlying skill (weigh forward-
# looking/concrete signals over backward-looking/routine ones) harder to
# imitate as a rigid sentence shape divorced from whether a real conflict
# exists.
MIXED_SIGNAL_SCENARIOS = [
    (["{name} ({ticker}) beat Q{q} earnings estimates by {beat}%",
      "{name} ({ticker}) cut its full-year guidance, citing softening {product} demand heading into next quarter"],
     "Despite beating this quarter's estimates, the guidance cut signals deteriorating forward demand for {product} - forward guidance outweighs a backward-looking beat.",
     "BEARISH"),
    (["{name} ({ticker}) missed quarterly revenue estimates by {beat}%",
      "{name} ({ticker}) simultaneously announced a ${buyback}B share buyback program"],
     "A buyback doesn't paper over the revenue miss - underlying demand looks weaker regardless of the capital-return announcement.",
     "BEARISH"),
    (["{name} ({ticker}) reported a strong Q{q}, with revenue of ${rev}B beating estimates by {beat}%",
      "Analysts flagged {product} inventory buildup as a risk to next quarter's results"],
     "Current results are genuinely strong, but the flagged inventory risk introduces real uncertainty about next quarter - bullish, with tempered confidence.",
     "BULLISH"),
    (["{name} ({ticker}) issued a recall affecting {units}k {product} units",
      "The recall follows a quarter of record {product} sales, reported just last week"],
     "A recall's safety and legal risk outweighs the prior quarter's already-priced-in sales record.",
     "BEARISH"),
    (["{name} ({ticker}) disclosed a regulatory fine related to {product} practices",
      "{name} ({ticker}) also raised its full-year guidance, citing broad-based demand strength"],
     "A one-time fine is a sunk cost that doesn't change the forward outlook; the guidance raise, grounded in broad-based demand strength, is what should actually move the stock - bullish, tempered by the fine's reputational overhang.",
     "BULLISH"),
    (["{name} ({ticker}) reported solid Q{q} results in line with expectations",
      "Broader market headlines describe a sector-wide selloff unrelated to {ticker}'s own fundamentals"],
     "{ticker}'s own results are solid; the sector-wide selloff in the noise headlines is not company-specific and shouldn't be weighted into {ticker}'s own outlook - bullish, with confidence tempered by broader market uncertainty.",
     "BULLISH"),
    (["{name} ({ticker}) beat Q{q} revenue estimates by {beat}%",
      "{name} ({ticker}) separately announced its CEO will step down at year-end as part of a planned transition"],
     "An orderly, pre-planned leadership transition tells you little you didn't already expect; the revenue beat is the harder data point here - bullish, tempered only by the normal uncertainty a CEO change introduces.",
     "BULLISH"),
    (["Analysts upgraded {ticker} ({name}) to 'Buy' citing long-term {product} potential",
      "{name} ({ticker}) issued weak near-term guidance, citing short-term {product} softness"],
     "A long-term-oriented upgrade doesn't change what management itself just said about the next few quarters - a concrete near-term guidance cut from the company carries more weight than an analyst's multi-year thesis - bearish.",
     "BEARISH"),
    (["{name} ({ticker}) missed Q{q} earnings estimates by {beat}%",
      "{name} ({ticker}) simultaneously raised its quarterly dividend by {divhike}%"],
     "Raising the dividend doesn't undo an actual earnings miss - a payout bump is the smaller signal next to results falling short - bearish.",
     "BEARISH"),
    (["{name} ({ticker})'s new {product} line sold out within days of launch",
      "{name} ({ticker}) separately recalled a small batch of an older, legacy product line unrelated to {product}"],
     "The flagship {product} launch is the primary current growth driver; a recall isolated to an unrelated legacy line is a minor operational item by comparison - bullish, with confidence tempered by the recall.",
     "BULLISH"),
    (["{name} ({ticker}) received regulatory approval for {product} expansion into new markets",
      "{name} ({ticker})'s credit rating was downgraded the same week, citing rising leverage"],
     "A credit downgrade reflects a structural balance-sheet concern that outweighs a single market-expansion approval - bearish, with confidence tempered by the approval's longer-term upside.",
     "BEARISH"),
    (["{name} ({ticker}) reported Q{q} results in line with expectations",
      "{name} ({ticker}) raised full-year guidance, citing accelerating {product} momentum"],
     "In-line current results don't cancel out a genuine guidance raise - forward-looking guidance is what should actually be priced in here - bullish.",
     "BULLISH"),
    # --- two more orderly-CEO-transition pairings, different accompanying
    # signal each time (one more bullish-paired, one bearish) - a single
    # example of "orderly transition doesn't move the needle, the other
    # signal does" wasn't enough repetition: confirmed live across multiple
    # eval runs, the model kept defaulting to BEARISH/NEUTRAL on this exact
    # pattern regardless of what the paired signal actually said, most
    # likely because a CEO departure reads negative by default from
    # pretraining alone and one counter-example can't overcome that prior.
    # Varying which direction the OTHER signal points (not always bullish)
    # is deliberate - the lesson is "the transition itself is near-neutral,
    # weigh the real signal," not "CEO transition secretly means bullish."
    (["{name} ({ticker}) raised its full-year guidance, citing accelerating {product} demand",
      "{name} ({ticker}) separately announced its CEO will step down at year-end as part of a planned transition"],
     "A guidance raise is a concrete, forward-looking signal from management itself; an orderly, pre-planned leadership transition doesn't offset that - bullish, tempered only by the normal uncertainty a CEO change introduces.",
     "BULLISH"),
    (["{name} ({ticker}) missed Q{q} revenue estimates by {beat}%",
      "{name} ({ticker}) separately announced its CEO will step down at year-end as part of a planned transition"],
     "A revenue miss is the harder, more decision-relevant data point here; an orderly, pre-planned leadership transition doesn't make a miss any less real - bearish, tempered only by the normal uncertainty a CEO change introduces.",
     "BEARISH"),
]
# Every entry above is now a required 3-tuple (headline_templates,
# reasoning_template, direction) - used to silently default a missing
# direction to "BEARISH" for entries that omitted it. Harmless by luck (the
# one entry that relied on it did want BEARISH) but a real risk: a future
# entry wanting BULLISH that forgot the third element would've been
# silently mislabeled instead of erroring. assert here instead, so a
# missing direction fails loudly at import time.
assert all(len(s) == 3 for s in MIXED_SIGNAL_SCENARIOS), \
    "MIXED_SIGNAL_SCENARIOS entries must all be explicit 3-tuples (headline_templates, reasoning_template, direction)"

# Teaches the model to actually weigh the Valuation block, which - before
# this category existed - was rendered every example (ported scenario-DCF
# math above) but never once correlated with `direction`, in either this
# dataset or generate_real_dataset.py: render_valuation's inputs were
# rolled independently of whatever news-driven direction/reasoning the
# example resolved to, and no reasoning_template anywhere ever referenced
# it. A model can't learn to use a signal that's pure noise relative to the
# label in 100% of its training data - this is the fix.
#
# "_alone" (valuation is the only signal, no news at all) and "_reinforced"
# (valuation + a soft same-direction headline) briefly got removed after a
# v11 eval showed real-val NEUTRAL collapsing to ~10% correct, with
# reasoning matching "_alone"'s phrasing verbatim even on trivial (2%)
# valuation gaps. Root cause turned out narrower than "remove the lesson
# entirely": "_alone"'s headlines ("held a routine analyst call with no
# notable updates," "reiterated prior guidance with no other updates") are
# near-duplicates of NEUTRAL_SCENARIOS' own "routine, no material news"
# headlines - and NEUTRAL_SCENARIOS used a fully random (uncorrelated)
# valuation draw, same as every other non-VALUATION_SIGNAL category, so it
# could ALSO show a large gap purely by chance. Same surface shape (routine
# headline + some gap) trained to opposite conclusions depending on which
# category happened to roll it. Real fix (see NEUTRAL_SCENARIOS' own
# valuation-rendering below): pin NEUTRAL's gap to a small, genuinely
# insignificant range instead of letting it collide with "_alone"'s extreme
# range - so gap SIZE, not headline wording, is what actually distinguishes
# "ignore this" from "this is a real signal," which is what should have
# been distinguishing them all along. That keeps the lesson an extreme
# valuation gap with neutral-to-mild news should still lean toward the
# valuation's direction, not fall back to NEUTRAL just because there's no
# news catalyst - while a small/routine gap correctly stays NEUTRAL.
#
# Each entry: (headline_templates, reasoning_template, direction,
# valuation_verdict, gap_tier, confidence_tier). valuation_verdict/gap_tier
# feed _valuation_block_with_gap (defined below, after the ported valuation
# functions) to construct a Valuation block with a controlled, specific gap
# size instead of the normal random draw - the same reason
# MIXED_SIGNAL_SCENARIOS hand-authors its headline pairs rather than
# sampling them: this needs a deliberately clean, specific setup to teach a
# clean lesson.
VALUATION_SIGNAL_SCENARIOS = [
    # --- extreme gap, no corroborating news: valuation is the only signal ---
    (["{name} ({ticker}) held a routine analyst call with no notable updates to prior commentary"],
     "No fresh news moves the needle here, but {ticker} is trading at a steep discount to its estimated intrinsic value - a real, if imperfect, signal on its own. DCF-style estimates carry real model uncertainty, so this leans bullish without the higher confidence a concrete catalyst would justify.",
     "BULLISH", "undervalued", "extreme", "extreme_alone"),
    (["{name} ({ticker}) reiterated prior full-year guidance with no other updates this week"],
     "Nothing new in the news, but {ticker} is trading at a steep premium to its estimated intrinsic value - worth weighing even without a fresh catalyst, tempered by the real uncertainty in any DCF-style estimate.",
     "BEARISH", "overvalued", "extreme", "extreme_alone"),
    # --- moderate gap, no corroborating news: weaker evidence, lower confidence ---
    (["{name} ({ticker}) traded in a narrow range this week with no company-specific news"],
     "No headline catalyst, but {ticker}'s current price sits at a modest discount to its estimated intrinsic value - a real but comparatively soft signal, especially with no news to corroborate it, so confidence here stays low.",
     "BULLISH", "undervalued", "moderate", "moderate_alone"),
    (["{name} ({ticker}) saw light trading volume in an otherwise uneventful week"],
     "No headline catalyst, but {ticker}'s current price sits at a modest premium to its estimated intrinsic value - a real but comparatively soft signal on its own, so confidence here stays low.",
     "BEARISH", "overvalued", "moderate", "moderate_alone"),
    # --- extreme gap, reinforced by a mild/subtle same-direction headline ---
    (["{name} ({ticker}) saw a modest uptick in institutional buying interest, according to the latest filings"],
     "{ticker} already looks meaningfully undervalued against its estimated intrinsic value, and the pickup in institutional interest is a soft but same-direction confirmation - still tempered by the underlying uncertainty in any valuation estimate, but more confident than the valuation gap alone would justify.",
     "BULLISH", "undervalued", "extreme", "extreme_reinforced"),
    (["{name} ({ticker}) saw a modest uptick in insider selling activity, according to the latest filings"],
     "{ticker} already looks meaningfully overvalued against its estimated intrinsic value, and the pickup in insider selling is a soft but same-direction confirmation - still tempered by the underlying uncertainty in any valuation estimate, but more confident than the valuation gap alone would justify.",
     "BEARISH", "overvalued", "extreme", "extreme_reinforced"),
    # --- extreme gap, but a concrete near-term catalyst points the other
    # way - the news should win, same principle as MIXED_SIGNAL_SCENARIOS ---
    (["{name} ({ticker}) cut its full-year guidance, citing softening {product} demand heading into next quarter"],
     "{ticker} screens as meaningfully undervalued on an estimated-intrinsic-value basis, but a concrete, company-issued guidance cut is a more reliable near-term signal than a longer-horizon valuation estimate - the guidance cut should dominate here, not the valuation gap.",
     "BEARISH", "undervalued", "extreme", "news_wins"),
    (["{name} ({ticker}) raised its full-year guidance, citing accelerating {product} demand"],
     "{ticker} screens as meaningfully overvalued on an estimated-intrinsic-value basis, but a concrete, company-issued guidance raise is a more reliable near-term signal than a longer-horizon valuation estimate - the guidance raise should dominate here, not the valuation gap.",
     "BULLISH", "overvalued", "extreme", "news_wins"),
    # --- same principle, earnings-beat/miss phrasing (the real headline
    # style this category was missing - see comment above) ---
    (["{name} ({ticker}) missed Q{q} revenue estimates by {beat}%"],
     "{ticker} screens as meaningfully undervalued on an estimated-intrinsic-value basis, but a confirmed revenue miss is a concrete, current-quarter signal that carries more weight than a longer-horizon valuation estimate - the miss should dominate here, not the valuation gap.",
     "BEARISH", "undervalued", "extreme", "news_wins"),
    (["{name} ({ticker}) beat Q{q} revenue estimates by {beat}%"],
     "{ticker} screens as meaningfully overvalued on an estimated-intrinsic-value basis, but a confirmed revenue beat is a concrete, current-quarter signal that carries more weight than a longer-horizon valuation estimate - the beat should dominate here, not the valuation gap.",
     "BULLISH", "overvalued", "extreme", "news_wins"),
    # --- same principle, analyst rating-change phrasing ---
    (["Analysts downgraded {ticker} ({name}) to 'Sell', citing slowing {product} demand"],
     "{ticker} screens as meaningfully undervalued on an estimated-intrinsic-value basis, but a fresh analyst downgrade reflects a specific, current view of deteriorating {product} demand that a static valuation estimate can't capture - the downgrade should dominate here, not the valuation gap.",
     "BEARISH", "undervalued", "extreme", "news_wins"),
    (["Analysts upgraded {ticker} ({name}) to 'Buy', citing accelerating {product} demand"],
     "{ticker} screens as meaningfully overvalued on an estimated-intrinsic-value basis, but a fresh analyst upgrade reflects a specific, current view of improving {product} demand that a static valuation estimate can't capture - the upgrade should dominate here, not the valuation gap.",
     "BULLISH", "overvalued", "extreme", "news_wins"),
    # --- same principle, price-move-tied-to-a-demand-catalyst phrasing
    # (matches how real headlines like "stock pops after earnings" or
    # "rides winning streak to an X% gain" actually read) ---
    (["{name} ({ticker}) shares fell {drop}% after {product} demand came in well below expectations"],
     "{ticker} screens as meaningfully undervalued on an estimated-intrinsic-value basis, but a sharp move tied to a concrete demand shortfall is a more reliable near-term signal than a longer-horizon valuation estimate - the demand shortfall should dominate here, not the valuation gap.",
     "BEARISH", "undervalued", "extreme", "news_wins"),
    (["{name} ({ticker}) shares rallied {rally}% after {product} demand blew past expectations"],
     "{ticker} screens as meaningfully overvalued on an estimated-intrinsic-value basis, but a sharp move tied to a concrete demand beat is a more reliable near-term signal than a longer-horizon valuation estimate - the demand beat should dominate here, not the valuation gap.",
     "BULLISH", "overvalued", "extreme", "news_wins"),
]

VALUATION_GAP_RANGES = {"extreme": (70.0, 95.0), "moderate": (15.0, 35.0)}
# NEUTRAL_SCENARIOS' own valuation gap is pinned to this range (see
# make_example's NEUTRAL branch) instead of the fully random draw every
# other non-VALUATION_SIGNAL category gets - genuinely insignificant, so it
# can never collide with VALUATION_SIGNAL_SCENARIOS' "extreme"/"moderate"
# ranges above by chance. See VALUATION_SIGNAL_SCENARIOS' own comment for
# why this matters.
NEUTRAL_VALUATION_GAP_RANGE = (0.0, 8.0)

# BULLISH/BEARISH now nested by tier - "subtle" gets a lower range than
# "clear" (a softer signal genuinely warrants less certainty) but still well
# above NEUTRAL's range, so low confidence alone doesn't become another
# implicit way to say "I'm not sure, call it NEUTRAL."
CONFIDENCE_RANGES = {
    "BULLISH": {"clear": (0.85, 0.97), "subtle": (0.66, 0.80)},
    "BEARISH": {"clear": (0.83, 0.96), "subtle": (0.66, 0.80)},
    "NEUTRAL": {"clear": (0.60, 0.82)},
    "MIXED": {"clear": (0.55, 0.75)},
    # Deliberately lower ceilings than BULLISH/BEARISH "clear" even for the
    # "extreme" gap tier - a DCF-style estimate is documented, first-hand,
    # to carry real model risk (see VALUATION_SIGNAL_SCENARIOS' own
    # comment), so it should never earn the same confidence a concrete news
    # catalyst does. "news_wins" is the exception: confidence there reflects
    # the (real, concrete) news catalyst, not the valuation gap it overrides.
    "VALUATION_SIGNAL": {
        "extreme_alone": (0.58, 0.70),
        "moderate_alone": (0.45, 0.56),
        "extreme_reinforced": (0.68, 0.80),
        "news_wins": (0.80, 0.93),
    },
}


def held_out_template_indices(templates):
    # Evenly-spaced indices across the whole list rather than a tail slice -
    # keeps the held-out val split representative even when a list is
    # internally grouped (e.g. BULLISH/BEARISH's "clear" templates followed
    # by "subtle" ones), instead of only ever holding out whichever group
    # happens to be listed last.
    n = len(templates)
    n_holdout = max(1, int(n * VAL_HOLDOUT_TEMPLATE_INDEX_FRACTION))
    stride = n / n_holdout
    return {int(i * stride) for i in range(n_holdout)}


BULLISH_HOLDOUT_IDX = held_out_template_indices(BULLISH_SCENARIOS)
BEARISH_HOLDOUT_IDX = held_out_template_indices(BEARISH_SCENARIOS)
NEUTRAL_HOLDOUT_IDX = held_out_template_indices(NEUTRAL_SCENARIOS)
MIXED_HOLDOUT_IDX = held_out_template_indices(MIXED_SIGNAL_SCENARIOS)
VALUATION_SIGNAL_HOLDOUT_IDX = held_out_template_indices(VALUATION_SIGNAL_SCENARIOS)


def make_fields(company):
    # Generated once per example, not per headline - a multi-headline
    # scenario (e.g. a recall headline plus a "prior quarter" headline) must
    # reference the same product/quarter/numbers in both lines, not
    # independently re-rolled ones.
    ticker, name, _sector, rev_range = company
    return {
        "ticker": ticker,
        "name": name,
        "q": random.randint(1, 4),
        "rev": round(random.uniform(*rev_range), 1),
        "beat": round(random.uniform(3.2, 18.5), 1),
        "drop": round(random.uniform(2.5, 12.0), 1),
        "units": random.randint(50, 500),
        "product": random.choice(PRODUCTS),
        "region": random.choice(REGIONS),
        "role": random.choice(ROLES),
        "buyback": round(random.uniform(1.0, 20.0), 1),
        "divhike": random.randint(3, 15),
        "rally": round(random.uniform(4.0, 15.0), 1),
    }


def build_news_block(primary_headlines):
    # Every headline used to render within 6 days of the fixed anchor date -
    # real data doesn't look like that (generate_real_dataset.py pulls from
    # a multi-week lookback window, and real eval rows routinely show the
    # actual signal-bearing headline dated weeks or months back while
    # background macro noise stays fresh). Give the PRIMARY (signal)
    # headline a chance to render as older too, so the model sees that
    # pattern in training instead of only ever "everything is this week."
    # Noise headlines deliberately stay recent-only - they're generic
    # macro news, which genuinely doesn't accumulate the same way a
    # specific, dated company catalyst does.
    lines = [format_headline(h, stale=random.random() < STALE_HEADLINE_PROB) for h in primary_headlines]
    n_noise = random.randint(1, 3)
    for noise in random.sample(NOISE_HEADLINES, n_noise):
        lines.append(format_headline(noise))
    random.shuffle(lines)  # target headline isn't always first, like real feeds
    return "\n".join(lines)


def build_user_query(ticker):
    idx = random.randrange(len(USER_QUESTION_TEMPLATES))
    template = USER_QUESTION_TEMPLATES[idx]
    query = template.format(ticker=ticker) if template else ""
    return query, QUESTION_TYPES[idx]


def format_market_cap(value):
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    return f"${value / 1e9:.1f}B"


def make_fundamentals(company, fields):
    # Derives every other synthetic fundamental from a single random price
    # draw so a given example's numbers stay internally consistent (e.g.
    # price implies EPS implies intrinsic value implies over/undervalued -
    # they can't independently contradict each other the way unrelated
    # random draws could).
    ticker, _name, informal_sector, rev_range = company
    price_low, price_high = PRICE_RANGES[ticker]
    price = round(random.uniform(price_low, price_high), 2)

    pe_trailing = round(random.uniform(6.0, 28.0), 1)
    eps_trailing = round(price / pe_trailing, 2)
    if random.random() < LOSS_MAKING_PROB:
        eps_trailing = -abs(round(random.uniform(0.10, 3.0), 2))
    pe_forward = round(pe_trailing * random.uniform(0.82, 1.05), 1)

    div_yield = round(random.uniform(0.1, 3.2), 2) if random.random() < 0.6 else 0.0

    year_low = round(price * random.uniform(0.72, 0.93), 2)
    year_high = round(price * random.uniform(1.07, 1.38), 2)

    # Book value/share is no longer consumed by valuation math (the
    # scenario-DCF model doesn't use it - see build_scenarios/
    # cash_flow_basis_value above), but market_data's Graham-era P/B
    # grounding isn't part of production's market_data_block either, so
    # this is kept only as a plausible, unused-by-valuation figure - no
    # downstream consumer left to tune it against.
    pb_ratio = round(random.uniform(0.4, 4.0), 1)
    book_value_per_share = round(price / pb_ratio, 2)

    rev_low, rev_high = rev_range
    # Loosely ties market cap to the company's revenue band (bigger revenue
    # -> more shares outstanding, roughly) without needing a second hand-
    # authored per-ticker range - precision doesn't matter here, only that
    # price * shares stays a plausible, internally consistent market cap.
    shares_b = max(0.05, min(20.0, ((rev_low + rev_high) / 2) * random.uniform(0.04, 0.35)))
    market_cap = price * shares_b * 1e9

    anchor = datetime.date(2026, 8, 5)
    last_q_date = anchor - datetime.timedelta(days=random.randint(45, 100))
    next_earnings_date = anchor + datetime.timedelta(days=random.randint(35, 95))

    last_q_revenue = fields["rev"]  # reuse the same draw headline templates use,
                                     # so an example whose headline quotes revenue
                                     # can't contradict its own earnings block
    yoy_growth = round(random.uniform(-8.0, 22.0), 1)

    last_q_eps = round(max(eps_trailing, 0.05) / 4 * random.uniform(0.85, 1.15), 2) \
        if eps_trailing > 0 else round(-abs(eps_trailing) / 4 * random.uniform(0.85, 1.15), 2)

    # --- valuation-model inputs (classify_valuation_basis/build_scenarios
    # above) - none of these existed before the scenario-DCF port; each is
    # derived from figures already rolled above so nothing here can
    # contradict last_q_revenue/yoy_growth/eps_trailing.
    sector = SECTOR_MAP[informal_sector]

    # Annualized from the same quarterly draw headlines/earnings use, with
    # mild quarter-to-quarter variance - not an independent roll, so it
    # can't land on an annualized figure wildly out of step with the
    # quarter actually being discussed.
    total_revenue = last_q_revenue * 4 * random.uniform(0.92, 1.08) * 1e9

    dividend_rate = round(div_yield / 100 * price, 2) if div_yield > 0 else 0.0
    # Payout ratio only makes sense against positive earnings - matches
    # cash_flow_basis_value's own "dividends" branch, which reads
    # dividend_rate directly and never divides by eps_trailing itself, but
    # classify_valuation_basis's payout check needs a real ratio to compare
    # against DIVIDEND_PAYOUT_THRESHOLD/_CEILING.
    payout_ratio = round(dividend_rate / eps_trailing, 2) if (div_yield > 0 and eps_trailing > 0) else None

    # Mostly positive, roughly proportional to revenue (a plausible margin
    # band) - occasionally negative or None so the asset-heavy/fcf
    # classification branch's fallbacks (see classify_valuation_basis) get
    # exercised too, not just its happy path.
    fcf_roll = random.random()
    if fcf_roll < 0.85:
        free_cash_flow = total_revenue * random.uniform(0.05, 0.20)
    elif fcf_roll < 0.93:
        free_cash_flow = total_revenue * random.uniform(-0.08, -0.01)
    else:
        free_cash_flow = None

    # Consensus growth estimates (build_scenarios' g1 derivation) - based on
    # the same yoy_growth draw so a company already shown growing fast
    # doesn't also roll a consensus implying decline. ~15% of the time the
    # two years point in opposite directions on purpose, exercising
    # build_scenarios' "unreliable consensus -> G1_FALLBACK" branch the same
    # way a real rebound-then-giveback base year does in production.
    growth_0y = round(yoy_growth / 100 + random.uniform(-0.03, 0.03), 4)
    if random.random() < 0.15:
        growth_1y = round(-growth_0y * random.uniform(0.3, 1.2), 4)
    else:
        growth_1y = round(growth_0y + random.uniform(-0.04, 0.04), 4)
    spread = random.uniform(0.05, 0.15)
    growth_0y_high = round(growth_0y + spread, 4)
    growth_0y_low = round(growth_0y - spread, 4)

    return {
        "price": price,
        "pe_trailing": pe_trailing,
        "pe_forward": pe_forward,
        "eps_trailing": eps_trailing,
        "div_yield": div_yield,
        "year_low": year_low,
        "year_high": year_high,
        "book_value_per_share": book_value_per_share,
        "market_cap": market_cap,
        "last_q_date": last_q_date,
        "next_earnings_date": next_earnings_date,
        "last_q_revenue": last_q_revenue,
        "yoy_growth": yoy_growth,
        "last_q_eps": last_q_eps,
        "sector": sector,
        "total_revenue": total_revenue,
        "dividend_rate": dividend_rate,
        "payout_ratio": payout_ratio,
        "free_cash_flow": free_cash_flow,
        "growth_0y": growth_0y,
        "growth_1y": growth_1y,
        "growth_0y_high": growth_0y_high,
        "growth_0y_low": growth_0y_low,
    }


def render_market_data(fnd):
    if random.random() < DATA_UNAVAILABLE_PROB:
        return "Data unavailable."
    return (
        f"Price: ${fnd['price']:.2f} | Market Cap: {format_market_cap(fnd['market_cap'])}\n"
        f"P/E (trailing): {fnd['pe_trailing']:.1f} | P/E (forward): {fnd['pe_forward']:.1f}\n"
        f"EPS (trailing): ${fnd['eps_trailing']:.2f} | Dividend Yield: {fnd['div_yield']:.2f}%\n"
        f"52-Week Range: ${fnd['year_low']:.2f} - ${fnd['year_high']:.2f}"
    )


def render_valuation(fnd, ticker=None):
    """Classifies the valuation basis and runs the scenario-DCF math ported
    above - same pipeline as valuation.py's valuation_block_for, adapted to
    this generator's already-in-hand fnd dict instead of a live yfinance
    fetch. `ticker` (optional) lets NVDA/MSFT/PEP/NFLX/XOM draw the same
    CURATED_SCENARIOS assumptions production would use for them."""
    if random.random() < DATA_UNAVAILABLE_PROB:
        return "Data unavailable."
    basis = classify_valuation_basis(
        fnd["eps_trailing"], fnd["payout_ratio"], fnd["sector"], fnd["free_cash_flow"],
    )
    cf0 = cash_flow_basis_value(basis, fnd)
    scenarios = build_scenarios(ticker, fnd, basis)
    intrinsic = intrinsic_value(cf0, basis, scenarios)
    return valuation_block(fnd["price"], intrinsic, basis)


def valuation_block_with_gap(price, gap_pct, verdict):
    """Builds a Valuation block with a SPECIFIC, controlled over/undervalued
    percentage - used only by VALUATION_SIGNAL_SCENARIOS (see that list's
    own comment), which needs a deliberately clean, specific gap to teach a
    clean lesson, rather than whatever build_scenarios/intrinsic_value's
    full random-draw pipeline happens to produce. Still renders through the
    same valuation_block() formatter as every other row, so the shape
    (basis label, wording) is identical - only these rows' underlying
    numbers are directly constructed instead of DCF-derived. `basis` is
    randomly chosen per call purely for label variety across training
    examples; VALUATION_SIGNAL_SCENARIOS' lesson is about the gap, not
    about which basis produced it."""
    basis = random.choice(list(BASIS_LABELS))
    if verdict == "overvalued":
        intrinsic = price / (1 + gap_pct / 100)
    else:
        intrinsic = price / (1 - gap_pct / 100)
    return valuation_block(price, intrinsic, basis)


def render_earnings(fnd, direction):
    if random.random() < DATA_UNAVAILABLE_PROB:
        return "Data unavailable."
    # Coherent with the resolved direction: a bullish example's earnings
    # line shows a beat, a bearish one a miss, a neutral one in-line -
    # matching the scenario's news headlines instead of being an
    # independently-rolled, potentially contradictory number.
    surprise_pct = round(random.uniform(2.0, 12.0), 1)
    actual_eps = fnd["last_q_eps"]
    if direction == "BULLISH":
        est_eps = round(actual_eps / (1 + surprise_pct / 100), 2)
        surprise_note = f"beat est. ${est_eps:.2f}"
        yoy = abs(fnd["yoy_growth"])
    elif direction == "BEARISH":
        est_eps = round(actual_eps / (1 - surprise_pct / 100), 2)
        surprise_note = f"missed est. ${est_eps:.2f}"
        yoy = -abs(fnd["yoy_growth"])
    else:
        est_eps = actual_eps
        surprise_note = f"in line with est. ${est_eps:.2f}"
        yoy = fnd["yoy_growth"] / 3  # neutral quarters drift near flat YoY

    date_str = fnd["last_q_date"].isoformat()
    next_str = fnd["next_earnings_date"].isoformat()
    sign = "+" if yoy >= 0 else ""
    return (
        f"Last Quarter ({date_str}): Revenue ${fnd['last_q_revenue']:.1f}B "
        f"({sign}{yoy:.1f}% YoY), EPS ${actual_eps:.2f} ({surprise_note})\n"
        f"Next Earnings Date: {next_str}"
    )


def make_example(company, category):
    ticker = company[0]
    valuation_signal = None  # set below only for category == "VALUATION_SIGNAL"
    # render_earnings(fnd, direction) renders a beat/miss matching whatever
    # direction is passed - fine for every category except VALUATION_SIGNAL's
    # "_alone"/"_reinforced" tiers, whose headlines explicitly claim no fresh
    # news ("no other updates," "no notable updates"). Confirmed live: those
    # rows were getting a real, unmentioned earnings beat/miss that silently
    # contradicts the headline's own "nothing happened" claim - a second
    # channel telling the model "there IS a catalyst here" exactly where
    # NEUTRAL_VALUATION_GAP_RANGE (above) is teaching the opposite. Only
    # "_wins" describes an actual catalyst worth reflecting in earnings.
    earnings_direction = None  # None => use `direction`; set explicitly to override

    if category == "MIXED":
        idx = random.randrange(len(MIXED_SIGNAL_SCENARIOS))
        headline_templates, reasoning_template, direction = MIXED_SIGNAL_SCENARIOS[idx]
        is_holdout_template = idx in MIXED_HOLDOUT_IDX
        conf_low, conf_high = CONFIDENCE_RANGES["MIXED"]["clear"]
    elif category == "NEUTRAL":
        idx = random.randrange(len(NEUTRAL_SCENARIOS))
        headline_templates, reasoning_template = NEUTRAL_SCENARIOS[idx]
        direction = "NEUTRAL"
        is_holdout_template = idx in NEUTRAL_HOLDOUT_IDX
        conf_low, conf_high = CONFIDENCE_RANGES["NEUTRAL"]["clear"]
    elif category == "VALUATION_SIGNAL":
        idx = random.randrange(len(VALUATION_SIGNAL_SCENARIOS))
        (headline_templates, reasoning_template, direction,
         valuation_verdict, gap_tier, confidence_tier) = VALUATION_SIGNAL_SCENARIOS[idx]
        valuation_signal = (valuation_verdict, gap_tier)
        is_holdout_template = idx in VALUATION_SIGNAL_HOLDOUT_IDX
        conf_low, conf_high = CONFIDENCE_RANGES["VALUATION_SIGNAL"][confidence_tier]
        if confidence_tier != "news_wins":
            earnings_direction = "NEUTRAL"  # no real catalyst - keep earnings "in line", see comment above
    else:
        scenarios = {"BULLISH": BULLISH_SCENARIOS, "BEARISH": BEARISH_SCENARIOS}[category]
        holdout_idx = {"BULLISH": BULLISH_HOLDOUT_IDX, "BEARISH": BEARISH_HOLDOUT_IDX}[category]
        idx = random.randrange(len(scenarios))
        headline_templates, reasoning_template, tier = scenarios[idx]
        direction = category
        is_holdout_template = idx in holdout_idx
        conf_low, conf_high = CONFIDENCE_RANGES[category][tier]

    fields = make_fields(company)
    filled_headlines = [t.format(**fields) for t in headline_templates]
    reasoning = reasoning_template.format(**fields)

    news_block = build_news_block(filled_headlines)
    user_query, qtype = build_user_query(ticker)

    fnd = make_fundamentals(company, fields)
    market_data_text = render_market_data(fnd)
    if valuation_signal is not None:
        verdict, gap_tier = valuation_signal
        gap_low, gap_high = VALUATION_GAP_RANGES[gap_tier]
        gap_pct = round(random.uniform(gap_low, gap_high), 1)
        valuation_text = valuation_block_with_gap(fnd["price"], gap_pct, verdict)
    elif category == "NEUTRAL":
        # Pinned to a small, genuinely insignificant gap instead of the
        # fully random draw every other category gets - see
        # VALUATION_SIGNAL_SCENARIOS' own comment and
        # NEUTRAL_VALUATION_GAP_RANGE for why: a random draw could
        # occasionally produce a large gap by chance, teaching "routine
        # headline + large gap = NEUTRAL" in direct conflict with
        # VALUATION_SIGNAL_SCENARIOS' "_alone"/"_reinforced" tiers, which
        # teach the same input shape should lean toward the valuation's
        # direction once the gap is actually large.
        if random.random() < DATA_UNAVAILABLE_PROB:
            valuation_text = "Data unavailable."
        else:
            verdict = random.choice(["undervalued", "overvalued"])
            gap_low, gap_high = NEUTRAL_VALUATION_GAP_RANGE
            gap_pct = round(random.uniform(gap_low, gap_high), 1)
            valuation_text = valuation_block_with_gap(fnd["price"], gap_pct, verdict)
    else:
        valuation_text = render_valuation(fnd, ticker)
    earnings_text = render_earnings(fnd, earnings_direction or direction)

    answer = ANSWER_TEMPLATES[qtype][direction].format(ticker=ticker)

    confidence = round(random.uniform(conf_low, conf_high), 2)

    output_payload = {
        "impacted_stocks": [
            {
                "ticker": ticker,
                "reasoning": reasoning,
                "direction": direction,
                "confidence": confidence,
                "answer": answer,
            }
        ]
    }

    is_holdout_ticker = ticker in VAL_HOLDOUT_TICKERS
    split = "val" if (is_holdout_ticker or is_holdout_template) else "train"

    return {
        "ticker": ticker,  # duplicated from output.impacted_stocks[0].ticker so the
                           # training script can build "Target Stock: {ticker}" without
                           # parsing the output JSON string
        "user_query": user_query,
        "market_data": market_data_text,
        "valuation": valuation_text,
        "earnings": earnings_text,
        "news": news_block,
        "output": json.dumps(output_payload, indent=2),
        "_split": split,  # stripped before writing - see main()
    }


def generate(n=NUM_EXAMPLES):
    examples = []
    categories = list(SENTIMENT_WEIGHTS.keys())
    weights = list(SENTIMENT_WEIGHTS.values())
    for _ in range(n):
        category = random.choices(categories, weights=weights)[0]
        company = random.choice(ALL_COMPANIES)
        examples.append(make_example(company, category))
    return examples


def main():
    examples = generate()

    train = [{k: v for k, v in ex.items() if k != "_split"} for ex in examples if ex["_split"] == "train"]
    val = [{k: v for k, v in ex.items() if k != "_split"} for ex in examples if ex["_split"] == "val"]

    with open("dataset_train.jsonl", "w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")

    with open("dataset_val.jsonl", "w") as f:
        for row in val:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote dataset_train.jsonl: {len(train)} examples")
    print(f"Wrote dataset_val.jsonl:   {len(val)} examples (held-out tickers: {sorted(VAL_HOLDOUT_TICKERS)})")


def refresh_companies_from_yfinance(tickers):
    """
    Optional, one-off helper - NOT part of the normal generation path (see
    __main__ below), which stays dependency-free/offline/deterministic on
    purpose. Run this manually, on demand, to replace the hand-estimated
    name/sector/revenue-range values in COMPANIES above with real ones:

        python generate_synthetic_dataset.py --refresh-companies

    Requires `pip install yfinance` and network access. Prints a Python
    literal for each ticker in COMPANIES' exact tuple shape - paste the
    output back over the COMPANIES list by hand (deliberately not
    auto-written back into this file, so a bad/partial fetch can't silently
    corrupt the dataset the next time someone runs the normal path).
    """
    import yfinance as yf

    print("COMPANIES = [")
    for ticker in tickers:
        t = yf.Ticker(ticker)
        try:
            info = t.info
        except Exception as e:
            print(f'    # ("{ticker}", ...) - SKIPPED, info fetch failed: {e}')
            continue

        name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector", "unknown")

        try:
            revenue_row = t.quarterly_income_stmt.loc["Total Revenue"]
            revenues_b = [v / 1e9 for v in revenue_row.dropna().tolist()]
            rev_low, rev_high = round(min(revenues_b), 1), round(max(revenues_b), 1)
        except Exception as e:
            print(f'    # {ticker}: revenue fetch failed ({e}) - falling back to a placeholder range, fix by hand')
            rev_low, rev_high = 1.0, 10.0

        print(f'    ("{ticker}", "{name}", "{sector}", ({rev_low}, {rev_high})),')
    print("]")


if __name__ == "__main__":
    import sys

    if "--refresh-companies" in sys.argv:
        refresh_companies_from_yfinance([c[0] for c in COMPANIES])
    else:
        main()
