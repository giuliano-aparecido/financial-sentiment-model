"""
Financial sentiment training dataset generator (synthetic)

Generates synthetic (news, user_query, output) examples for fine-tuning a
model to predict a stock's directional sentiment (BUY/SELL/HOLD)
from recent news headlines and financial results. Designed to run standalone
in Google Colab - no dependencies beyond the standard library.

This is a rewrite of an earlier 1000-example generator. Changes 1-8 below
predate this repo's history; change 9 is the most recent, added after the
first trained model's eval results came back at 86.0% direction accuracy on
this dataset's own held-out val split but only 25.0% on the real-headline
supplement (generate_real_dataset.py) - a real, below-random-chance gap,
not just "real data is harder." A likely contributor: this dataset's
BUY/SELL templates all use blatant, textbook-obvious signal language
("beat consensus by X%", "lowered guidance"), so the model may have learned
"if the signal isn't textbook-clear, default to HOLD" - a habit that
costs the most on real headlines, which are almost never phrased that
plainly.

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
6. HOLD raised from 10% (2 templates) to ~28% (8 templates) to match how
   often real news is genuinely non-eventful.
7. The eval split is held out by TEMPLATE and TICKER, not a random row
   split - with this much template repetition a random split leaks near-
   duplicates into validation and produces a flattering, meaningless
   accuracy number.
8. Template count roughly doubled per category (8->16 BUY/SELL/
   HOLD, 6->12 MIXED) after a first training run showed a real train/
   eval loss gap (0.12 vs ~0.4) - with only 6-8 templates repeated across
   3 epochs, the model had a real opportunity to memorize specific
   template phrasings rather than the underlying news-pattern concept.
   More templates at the same NUM_EXAMPLES halves the average repetition
   per template without needing a bigger dataset.
9. Added a "subtle" tier of 8 more BUY and 8 more SELL templates
   alongside the existing 16 ("clear" tier, unchanged) each - phrased the
   way real headlines actually read (options positioning, analyst estimate
   trims, investor-day reception, insider filings, channel checks, a
   competitor's stumble) rather than declaring the outcome outright ("beat
   by X%", "lowered guidance"). These still resolve decisively to
   BUY/SELL - never to HOLD - with the reasoning text explicitly
   naming the softer nature of the signal, so the model learns "less
   obvious does not mean give up and hedge," the opposite of the pattern
   the eval results suggest it picked up. Subtle examples get their own,
   lower confidence range (see CONFIDENCE_RANGES) since a softer signal
   genuinely warrants less certainty than a blatant one, without collapsing
   all the way to HOLD-range confidence. held_out_template_indices was
   also changed from a tail-slice (last 20% of each list) to an
   evenly-spaced sample across the whole list, so the held-out val split
   for BUY/SELL now includes a representative mix of both tiers
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
{"impacted_stocks": [{"ticker", "reasoning", "recommendation", "confidence",
"answer"}]}. Matches the canonical prompt template shared with
generate_real_dataset.py, colab/train/gpu/train_model.py,
colab/train/tpu/train_model.py, and financial-sentiment-api's
app/services/inference.py - keep all in sync
(see CONTRIBUTING.md's 4-way sync rule).

Output: two JSONL files (one JSON object per line), train and val.
"""

import datetime
import hashlib
import json
import random
import re

import fusion_rules

random.seed(42)  # reproducible across Colab runs

NUM_EXAMPLES = 2000  # up from 1000 - more scenario categories need more rows
# Trimmed to 1600 for the training-mix rebalance below - see that
# reassignment's own comment (kept as a separate line, not folded into
# the constant above, so this file's own edit history stays legible: the
# "more categories need more rows" reasoning above is still why the
# BASELINE is 2000, this is a deliberate reduction from that baseline for
# an unrelated, later reason).
#
# generate_real_dataset.py's BUY_THRESHOLD/SELL_THRESHOLD tightened
# 2% -> 3% (see that file's own comment) shrank real's rebalanced train
# count from 1152 to 849 - real data's SHARE of the combined training set
# (train_model.py concatenates both files) would otherwise drop from
# ~45% to ~38% just from that change, working against the actual reason
# for tightening it (better real-world calibration needs MORE relative
# real-data influence, not less). Undersampling synthetic back down
# (not duplicating real rows up - see rebalance_by_direction/
# downsample_contradicts_in_place in generate_real_dataset.py for why
# duplication specifically risks re-memorization) brings the ratio back
# toward parity: ~1100 synthetic vs 849 real train rows, close to 56/44
# instead of 62/38.
NUM_EXAMPLES = 1600
                      # to keep per-scenario-per-ticker counts reasonable
# Doubled from {"META", "BA", "QRNL", "HRZN"} - confirmed live those 4
# tickers accounted for 54.8% of ALL val rows (a company's full row
# count lands in val unconditionally via VAL_HOLDOUT_TICKERS, on top of
# whatever template-holdout rows other tickers also contribute), making
# the val split's per-category accuracy largely a read on 4 specific
# companies rather than a representative sample. 2 more real + 2 more
# invented tickers, matching the original 2-real/2-invented mix.
VAL_HOLDOUT_TICKERS = {"META", "BA", "QRNL", "HRZN", "NFLX", "JPM", "ZVEX", "VLTR"}  # never seen in training
VAL_HOLDOUT_TEMPLATE_INDEX_FRACTION = 0.2                       # ~20% of each
                                                                  # category's templates
                                                                  # are validation-only

# Two-stage pipeline redesign (2026-08-19, see module docstring history
# item 12): this file no longer generates a BUY/SELL/HOLD label at all -
# every category above was really teaching one of two entangled skills at
# once (read the news correctly, AND weigh it against valuation headroom),
# which is exactly the fragile-to-learn rule fusion_rules.fuse() now
# computes deterministically instead. SENTIMENT_WEIGHTS -> REACTION_WEIGHTS:
# the five news_reaction classes Task A actually needs to classify.
# good/bad/neutral map onto the OLD BUY/SELL/HOLD scenario pools directly
# (same headline/reasoning content - it was always describing how the news
# itself reads, valuation was always a separate, independently-rolled
# field). overreaction_down/up are new. VALUATION_SIGNAL and MIXED are
# retired as standalone categories - VALUATION_SIGNAL_SCENARIOS'
# real-news tiers (news_wins/news_wins_no_headroom/_reinforced) and all of
# MIXED_SIGNAL_SCENARIOS are redistributed into BUY_SCENARIOS/
# SELL_SCENARIOS by what their headlines actually say (see the
# redistribution block right after VALUATION_SIGNAL_SCENARIOS below);
# VALUATION_SIGNAL's no-real-news tiers (_alone/moderate_alone) go to
# HOLD_SCENARIOS. Weights below deliberately oversample the two rare
# overreaction classes relative to their measured real-world rate (~5%
# combined, see calibrate_reaction_thresholds.py) - synthetic data is
# free to do this; generate_real_dataset.py's own rebalancing (never
# duplicating rows) can't.
REACTION_WEIGHTS = {
    "good": 0.22, "bad": 0.22, "neutral": 0.26,
    "overreaction_down": 0.15, "overreaction_up": 0.15,
}

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
# Raised from 0.40: live data across 26 real tickers showed payout ratios
# just above the old threshold (QCOM 0.41, LOW 0.406, AVGO 0.413, CSCO
# 0.498) belong to low-yield growth companies routed into an inappropriate
# dividend-discount model, not mature dividend payers - see valuation.py's
# own comment for the full live evidence.
DIVIDEND_PAYOUT_THRESHOLD = 0.55
DIVIDEND_PAYOUT_CEILING = 1.20
REIT_SECTORS = {"Real Estate"}

# Ported byte-identical from financial-sentiment-api's fundamentals.py (see
# that module's own comment) - rough, illustrative per-sector median
# trailing P/E, not fetched live. Real Estate deliberately omitted: REITs
# route to a dividends/P/B-based valuation instead of P/E (see REIT_SECTORS
# above), so a sector-median-P/E comparison isn't meaningful there.
SECTOR_MEDIAN_PE = {
    "Technology": 28.0,
    "Healthcare": 22.0,
    "Financial Services": 13.0,
    "Consumer Cyclical": 19.0,
    "Consumer Defensive": 21.0,
    "Communication Services": 18.0,
    "Industrials": 19.0,
    "Energy": 12.0,
    "Basic Materials": 15.0,
    "Utilities": 17.0,
}

BASIS_LABELS = {
    "revenue": "Revenue-based",
    "eps": "EPS-based",
    "fcf": "FCF-based",
    "dividends": "Dividend-based",
}

# Backstop cap on the displayed over/undervalued percentage - ported from
# valuation.py's identically-named constant, see that module's comment.
VALUATION_PCT_DISPLAY_CAP = 150.0

DISCOUNT_RATE = 0.10
SCENARIO_PROBABILITY = 1 / 3

# Same curated per-ticker DCF assumptions as valuation.py - NVDA/MSFT/PEP/
# NFLX/XOM are all in COMPANIES above, so synthetic examples for those five
# tickers get the exact same scenario assumptions production would use for
# them, instead of the generic/derived fallback every other ticker gets.
# History (2026-08-17): AAPL/NVDA/MSFT/PEP/NFLX/XOM used to bypass this
# file's DCF math via CURATED_SCENARIOS, a hand-picked, fixed set of
# g1/g2/exit_multiple assumptions calibrated once against real analyst
# targets - because at the time, the general formula was untrustworthy
# (confirmed live in generate_real_dataset.py: derived g1 blowups like
# QCOM's -150%/$44.99). Removed once the formula itself became reliable
# (Finviz EPS-next-5Y + SGR blend, ported into build_scenarios below from
# that module's own comment) - the curated numbers had also gone stale.
# All tickers now go through the same dynamic path; some of the
# constants below (G1_CAP, G2_DIVIDENDS_FLOOR, GROWTH_BASIS_G2_FLOOR)
# were originally calibrated against those curated numbers and keep that
# history in their own comments even though the source data is gone.

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
# Floor on g2 for the "dividends" basis specifically - ported from
# valuation.py's G2_DIVIDENDS_FLOOR after a confirmed live case (QCOM: g1
# ~-8% from one bad consensus year, copied uncapped into g2 for the
# dividends basis, compounding to an intrinsic value of $22.90 against a
# $165.79 price - see that module's own comment for the full rationale).
# -0.05, not less negative - PEP's own real CURATED_SCENARIOS worst-case
# g2, the most negative g2 any analyst-vetted mature-payer number on file
# actually reaches.
G2_DIVIDENDS_FLOOR = -0.05
# Same floor concept as G2_DIVIDENDS_FLOOR, extended to the eps/fcf/
# revenue basis - ported from generate_real_dataset.py after a confirmed
# live bug there (QCOM: g1 ~-8% from a bad consensus year passed straight
# through into g2 unchanged, since min() only caps from above - 10 years
# of ~-8% decline against an unremarkable 19.2x trailing P/E). 0.0, not a
# small negative like G2_DIVIDENDS_FLOOR - every EPS-basis
# CURATED_SCENARIOS ticker's analyst-vetted worst-case g2 is non-negative
# (lowest: XOM at +3%), so a 0% floor is already more conservative than
# any real vetted number on file. See that module's own comment for the
# full rationale.
GROWTH_BASIS_G2_FLOOR = 0.0
G1_FALLBACK = {"normal": 0.08, "best": 0.10, "worst": 0.04}
# Ceiling on derived g1 - ported from valuation.py's G1_CAP after a
# confirmed live DCF blowup (this exact formula, run at scale via this
# generator, showed a 90th-percentile gap of 91%, 99th of 354%, max
# 760%). Tightened 0.40 -> 0.30 after a 19-ticker analyst comparison:
# 0.40 let a derived/unverified g1 run MORE aggressive than the single
# most aggressive number a human had actually vetted at the time (0.30,
# NVDA's old curated "best" g1 - see CURATED_SCENARIOS' removal note
# above). Applies to every ticker now that curation is gone.
G1_CAP = 0.30
# Floor on DERIVED g1 - ported from valuation.py's G1_FLOOR after the same
# QCOM investigation that motivated G2_DIVIDENDS_FLOOR above (see that
# module's own comment): confirmed live this is a symmetric problem, not
# an upside-only one. Set below PEP's/XOM's own curated worst-case g1
# (+0.03, real analyst-vetted numbers), same "don't disagree with actually
# -vetted data" reasoning as G1_CAP.
G1_FLOOR = -0.10

# Rejects a consensus growth pair as "reliable" when EITHER year's
# magnitude is this extreme, even same-direction - ported from
# valuation.py's CONSENSUS_GROWTH_MAGNITUDE_CAP (GOOG's 90.4%
# same-direction-but-implausible rebound-off-a-depressed-base case). See
# that module's own comment.
CONSENSUS_GROWTH_MAGNITUDE_CAP = 0.60

# Sustainable growth rate (ROE x retention ratio) as a middle tier in g1's
# derivation - ported from valuation.py's _sustainable_growth_rate/
# SUSTAINABLE_GROWTH_*_SPREAD after being asked directly what a non-flat
# g1 default should be based on. See that module's own comment for the
# full rationale, in short: NOT P/E (circular - P/E already prices in the
# market's growth expectations, so deriving a DCF growth input from it and
# valuing the company with that input concludes "fairly valued" almost by
# construction). ROE x retention only uses the company's own profitability
# and reinvestment behavior - already-generated fnd fields (eps_trailing,
# book_value_per_share, payout_ratio), no new dependency.
SUSTAINABLE_GROWTH_BEST_SPREAD = 0.02
SUSTAINABLE_GROWTH_WORST_SPREAD = -0.04

# Ceiling on the CONSENSUS-derived best/worst offset - ported from
# generate_real_dataset.py's CONSENSUS_OFFSET_CAP after a confirmed live
# bug there (GM as of 2026-04-15: growth_0y_high sat 17.6 points above
# growth_0y/growth_1y even though both individually passed
# CONSENSUS_GROWTH_MAGNITUDE_CAP, compounded 5 years at G1_CAP with a 24x
# exit multiple into a $578.92 "best" tier that dragged the whole
# valuation to an unreviewable 75%-undervalued reading). See that
# module's own comment for the full rationale, including why an
# unbounded consensus offset perversely made having real data WORSE than
# having none (the no-data fallback spread is only 2-4 points).
CONSENSUS_OFFSET_CAP = 0.10

# Caps/floors the ROE input to the sustainable-growth-rate formula -
# ported from valuation.py's SUSTAINABLE_GROWTH_ROE_CAP/_FLOOR after a
# confirmed live 19-ticker analyst comparison: buyback-heavy companies'
# eps_trailing/book_value_per_share ratio can explode well above 100%
# (META/AMZN/TSLA/ADBE/BABA's actual overshoot mechanism), while a
# dual-class-share book-value data artifact can push it to near-zero
# (BRK-B). See that module's own comments for the full rationale -
# 0.25/0.02 ported verbatim.
SUSTAINABLE_GROWTH_ROE_CAP = 0.25
SUSTAINABLE_GROWTH_ROE_FLOOR = 0.02


def _sustainable_growth_rate(fnd):
    """ROE x (1 - payout_ratio). None (not a fetch failure) when
    eps_trailing/book_value_per_share aren't usable - caller falls back to
    G1_FALLBACK. Missing payout_ratio defaults to 0 (full reinvestment,
    correct for a non-dividend-payer), not treated as unusable."""
    eps_trailing = fnd.get("eps_trailing")
    book_value_per_share = fnd.get("book_value_per_share")
    if not eps_trailing or eps_trailing <= 0 or not book_value_per_share or book_value_per_share <= 0:
        return None
    raw_roe = eps_trailing / book_value_per_share
    if raw_roe < SUSTAINABLE_GROWTH_ROE_FLOOR:
        return None
    roe = min(raw_roe, SUSTAINABLE_GROWTH_ROE_CAP)
    payout_ratio = fnd.get("payout_ratio") or 0.0
    return roe * (1 - payout_ratio)


# See value_screen_metrics' own comment on peg_ratio - ported from
# fundamentals.py's identically-named constant.
PEG_MIN_GROWTH_FOR_COMPUTATION = 0.02


def value_screen_metrics(fnd):
    """Ported byte-identical from financial-sentiment-api's fundamentals.py
    (see that module's own comment for the full rationale) - ROE, Price/
    Sales, FCF yield, and PEG are all derived from fields already in `fnd`,
    deliberately reusing the same formulas the DCF math above already uses
    internally (e.g. ROE = eps_trailing / book_value_per_share, same as
    _sustainable_growth_rate) rather than a second, independently-rolled
    number for the same concept. Each value is None when its inputs are
    missing/unusable."""
    eps_trailing = fnd.get("eps_trailing")
    book_value_per_share = fnd.get("book_value_per_share")
    roe = None
    if eps_trailing and book_value_per_share and book_value_per_share > 0:
        roe = eps_trailing / book_value_per_share

    market_cap = fnd.get("market_cap")
    total_revenue = fnd.get("total_revenue")
    price_to_sales = None
    if market_cap and total_revenue and total_revenue > 0:
        price_to_sales = market_cap / total_revenue

    free_cash_flow = fnd.get("free_cash_flow")
    fcf_yield = None
    if free_cash_flow is not None and market_cap and market_cap > 0:
        fcf_yield = free_cash_flow / market_cap

    pe_trailing = fnd.get("pe_trailing")
    growth_0y = fnd.get("growth_0y")
    # PEG only means anything against POSITIVE expected growth - see
    # fundamentals.py's identical comment for why a negative/zero growth_0y
    # is left None rather than shown as a misleadingly "cheap" number, and
    # PEG_MIN_GROWTH_FOR_COMPUTATION for why near-zero growth is floored
    # too (confirmed live: PEG values up to 525 in this generator's own
    # output, purely from dividing by a growth rate close to 0%).
    peg_ratio = None
    if pe_trailing and growth_0y and growth_0y > PEG_MIN_GROWTH_FOR_COMPUTATION:
        peg_ratio = pe_trailing / (growth_0y * 100)

    price = fnd.get("price")
    price_to_book = None
    if price and book_value_per_share and book_value_per_share > 0:
        price_to_book = price / book_value_per_share

    sector = fnd.get("sector")
    sector_median_pe = SECTOR_MEDIAN_PE.get(sector)

    return {
        "roe": roe,
        "operating_margin": fnd.get("operating_margin"),
        "price_to_sales": price_to_sales,
        "fcf_yield": fcf_yield,
        "peg_ratio": peg_ratio,
        "price_to_book": price_to_book,
        "is_reit_sector": sector in REIT_SECTORS,
        "sector_median_pe": sector_median_pe,
    }


# Graded confidence bands, not hard pass/fail cutoffs (e.g. ROE >= 15% ->
# high confidence, <= 8% -> low, between -> medium). Deliberately only 3 tiers
# per metric (not a continuous score) since this maps to discrete
# reasoning LANGUAGE the model can actually learn to reproduce, not a
# numeric field in the output schema (the model outputs a sentiment label
# + reasoning text, not a confidence score per metric - see the earlier
# design discussion this was scoped from).
OPERATING_MARGIN_STRONG = 0.15
OPERATING_MARGIN_WEAK = 0.08
ROE_STRONG = 0.15
ROE_WEAK = 0.08
PEG_CHEAP = 1.0
PEG_RICH = 2.0


# Multiple phrasings per tier, randomly picked - NOT just style variety.
# This project already hit the exact failure mode a single fixed sentence
# per case produces: generate_real_dataset.py's history item 7 documents a
# near-identical bug (CONTRADICTS handling) where one canned sentence
# shape, repeated across hundreds of rows, got memorized by the model as a
# literal string to reproduce rather than a judgment to make - and applied
# indiscriminately to unrelated examples, since it's the same model
# weights either way. A v16 eval reproduced the same signature here: the
# model recited this exact "Operating margins and ROE both point to..."
# sentence on a REAL headline it had never been trained on, verbatim,
# rather than reasoning about that headline's actual content. Randomizing
# the surface form across several equivalent phrasings per tier means
# there's no single string to memorize - only the underlying judgment
# (which tier the metrics fall into) can actually be learned.
QUALITY_STRONG_PHRASINGS = [
    "Operating margins and ROE both point to a genuinely high-quality underlying business",
    "Strong operating margins paired with a high ROE suggest real competitive advantage here",
    "Both margins and returns on equity look like hallmarks of a well-run, capital-efficient business",
]
QUALITY_WEAK_PHRASINGS = [
    "Thin operating margins and weak ROE argue for caution regardless of the valuation gap",
    "Weak profitability metrics here are a real yellow flag, independent of what the valuation gap suggests",
    "Margins and capital efficiency both look shaky, which tempers how much confidence the valuation gap deserves",
]
QUALITY_ADEQUATE_PHRASINGS = [
    "Operating margins and ROE are unremarkable here - adequate, not standout",
    "Nothing special about the underlying profitability here - solid enough, not a standout",
    "Margins and ROE sit in an unremarkable middle ground - no particular red flag, no particular strength",
]
GROWTH_UNKNOWN_PHRASINGS = [
    "growth looks too uncertain to gauge against the price",
    "there isn't a reliable growth estimate to weigh the price against",
    "growth expectations are too unclear here to say whether the price is justified",
]
GROWTH_CHEAP_PHRASINGS = [
    "the price looks reasonable relative to expected growth (PEG under 1)",
    "relative to its growth outlook, the price doesn't look demanding (PEG under 1)",
    "growth-adjusted, this isn't an expensive price to pay (PEG under 1)",
]
GROWTH_RICH_PHRASINGS = [
    "the price looks rich relative to expected growth (PEG over 2)",
    "growth-adjusted, this price is a stretch (PEG over 2)",
    "relative to its growth outlook, this is a demanding price to pay (PEG over 2)",
]
GROWTH_FAIR_PHRASINGS = [
    "the price is roughly in line with expected growth",
    "growth-adjusted, the price looks fair, neither cheap nor expensive",
    "relative to its growth outlook, the price doesn't stand out either way",
]
# Varies the connective structure too, not just the two clauses' wording -
# same memorization concern applies to the joining phrase as much as the
# content either side of it.
COMMENTARY_CONNECTORS = [
    "{quality}, and {growth_note}.",
    "{quality}. Separately, {growth_note}.",
    "{quality} - {growth_note}.",
]


def describe_value_screen(fnd):
    """Translates operating margin/ROE/PEG (see value_screen_metrics above)
    into 1-2 sentences of graded value-checklist commentary - teaches the
    model this vocabulary explicitly (see
    VALUE_SCREEN_COMMENTARY_PROB_OTHER_CATEGORIES's own comment for why
    this is spliced into only a fraction of Task B examples) rather than
    relying purely on the model inferring quality/growth-adjusted-pricing
    language from raw numbers in market_data on its own. Returns None (not a
    placeholder sentence) when operating_margin/ROE aren't usable, so the
    caller can skip appending anything - same "don't guess" convention as
    the rest of this module's rendering.

    Each call randomly picks ONE phrasing per clause (see the
    QUALITY_*_PHRASINGS/GROWTH_*_PHRASINGS lists and their shared comment
    above) rather than a single fixed sentence per tier - deliberately, to
    avoid the exact memorizable-canned-text failure this project already
    diagnosed once in generate_real_dataset.py's CONTRADICTS handling.
    """
    screen = value_screen_metrics(fnd)
    op_margin = screen["operating_margin"]
    roe = screen["roe"]
    peg = screen["peg_ratio"]
    if op_margin is None or roe is None:
        return None

    if op_margin > OPERATING_MARGIN_STRONG and roe > ROE_STRONG:
        quality = random.choice(QUALITY_STRONG_PHRASINGS)
    elif op_margin < OPERATING_MARGIN_WEAK or roe < ROE_WEAK:
        quality = random.choice(QUALITY_WEAK_PHRASINGS)
    else:
        quality = random.choice(QUALITY_ADEQUATE_PHRASINGS)

    if peg is None:
        growth_note = random.choice(GROWTH_UNKNOWN_PHRASINGS)
    elif peg < PEG_CHEAP:
        growth_note = random.choice(GROWTH_CHEAP_PHRASINGS)
    elif peg > PEG_RICH:
        growth_note = random.choice(GROWTH_RICH_PHRASINGS)
    else:
        growth_note = random.choice(GROWTH_FAIR_PHRASINGS)

    connector = random.choice(COMMENTARY_CONNECTORS)
    return connector.format(quality=quality, growth_note=growth_note)


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


# A trailing P/E below this fraction of forward P/E flags a likely
# one-off EPS distortion that reverts by next year - ported from
# valuation.py's ONE_TIME_ITEM_PE_RATIO_THRESHOLD (CHTR: pe_trailing~4.0
# against a normal forward P/E). See that module's own comment.
ONE_TIME_ITEM_PE_RATIO_THRESHOLD = 0.5
# Absolute trailing P/E floor, checked only when pe_forward is ALSO below
# it - ported from valuation.py's PERSISTENTLY_LOW_PE_THRESHOLD (CHTR
# actually failed this way: pe_forward~3.5, ALSO abnormally low, so the
# "reverts by next year" signal above never fires - a persistent, not
# one-off, EPS distortion). See that module's own comment.
PERSISTENTLY_LOW_PE_THRESHOLD = 6.0
# Fraction above consensus EPS estimate, for the most recently reported
# quarter, that flags a likely one-time/non-operating item - ported from
# valuation.py's EARNINGS_SURPRISE_ONE_TIME_ITEM_THRESHOLD (GOOG: two
# consecutive quarters beating consensus by +94%/+213%, almost certainly
# mark-to-market gains on equity investment stakes, not organic growth -
# a distortion invisible to the two P/E-based screens above since GOOG's
# P/E looked completely normal). This generator has no real earnings-
# surprise concept (synthetic fnd has no "recent_eps_surprise" field), so
# this check is a structural no-op here, kept only to stay byte-for-byte
# in step with valuation.py per the 4-way sync rule. See that module's
# own comment for the full rationale, including why this is checked in
# render_valuation (a short-circuit BEFORE the compute/blend/fallback
# pipeline) rather than here.
EARNINGS_SURPRISE_ONE_TIME_ITEM_THRESHOLD = 0.75


def cash_flow_basis_value(basis, fnd):
    """Extracts the per-share cash-flow figure for the classified basis -
    ported from valuation.py's identically-named function, adapted to this
    generator's fnd dict field names."""
    if basis == "eps":
        eps_trailing = fnd.get("eps_trailing")
        pe_trailing = fnd.get("pe_trailing")
        pe_forward = fnd.get("pe_forward")
        price = fnd.get("price")
        if (
            eps_trailing and pe_trailing and pe_forward and price
            and pe_trailing > 0 and pe_forward > 0
            and pe_trailing < pe_forward * ONE_TIME_ITEM_PE_RATIO_THRESHOLD
        ):
            return price / pe_forward
        if (
            eps_trailing and pe_trailing and pe_trailing > 0 and pe_trailing < PERSISTENTLY_LOW_PE_THRESHOLD
            and (pe_forward is None or (pe_forward > 0 and pe_forward < PERSISTENTLY_LOW_PE_THRESHOLD))
        ):
            return None
        return eps_trailing
    if basis == "dividends":
        return fnd.get("dividend_rate")

    shares = _shares_outstanding_approx(fnd.get("market_cap"), fnd.get("price"))
    if not shares:
        return None
    # currency/financial_currency mismatch guard - ported from
    # valuation.py's identically-named check; this generator's fnd dict
    # has no such fields (synthetic data has no real cross-listing
    # currency concept), so this is a structural no-op here, kept only to
    # stay byte-for-byte in step with valuation.py per the 4-way sync rule.
    currency = fnd.get("currency")
    financial_currency = fnd.get("financial_currency")
    if currency and financial_currency and currency != financial_currency:
        return None
    if basis == "revenue":
        revenue = fnd.get("total_revenue")
        return revenue / shares if revenue else None
    if basis == "fcf":
        fcf = fnd.get("free_cash_flow")
        return fcf / shares if fcf else None
    return None


def build_scenarios(fnd, basis):
    """Ported from valuation.py's identically-named function - see that
    module for the full rationale behind each piece."""
    g1_values = dict(G1_FALLBACK)

    sustainable_g1 = _sustainable_growth_rate(fnd)
    if sustainable_g1 is not None:
        g1_values = {
            "normal": sustainable_g1,
            "best": sustainable_g1 + SUSTAINABLE_GROWTH_BEST_SPREAD,
            "worst": sustainable_g1 + SUSTAINABLE_GROWTH_WORST_SPREAD,
        }

    growth_0y = fnd.get("growth_0y")
    growth_1y = fnd.get("growth_1y")
    consensus_reliable = (
        growth_0y is not None and growth_1y is not None
        and (growth_0y >= 0) == (growth_1y >= 0)
        and abs(growth_0y) <= CONSENSUS_GROWTH_MAGNITUDE_CAP
        and abs(growth_1y) <= CONSENSUS_GROWTH_MAGNITUDE_CAP
    )
    if consensus_reliable:
        # best/worst are OFFSETS from the same 2-year blend "normal" uses,
        # not growth_0y_high/low directly - ported from valuation.py's
        # build_scenarios after a confirmed live ordering bug (see
        # G1_FLOOR's comment): the old direct-substitution version let
        # "best" end up WORSE than "normal" whenever growth_1y diverged a
        # lot from growth_0y, since best/worst never saw growth_1y at all.
        # A missing bound mirrors the OTHER bound's offset (or, if
        # NEITHER exists, the generic SUSTAINABLE_GROWTH_*_SPREAD) rather
        # than leaving a stale prior-tier value in place - ported after a
        # second confirmed live bug (see valuation.py's own comment).
        blended_normal = (growth_0y + growth_1y) / 2
        growth_0y_high = fnd.get("growth_0y_high")
        growth_0y_low = fnd.get("growth_0y_low")
        # min(..., CONSENSUS_OFFSET_CAP) - see that constant's own comment.
        raw_high_offset = min(growth_0y_high - growth_0y, CONSENSUS_OFFSET_CAP) if growth_0y_high is not None else None
        raw_low_offset = min(growth_0y - growth_0y_low, CONSENSUS_OFFSET_CAP) if growth_0y_low is not None else None
        high_offset = raw_high_offset if raw_high_offset is not None else (
            raw_low_offset if raw_low_offset is not None else SUSTAINABLE_GROWTH_BEST_SPREAD
        )
        low_offset = raw_low_offset if raw_low_offset is not None else (
            raw_high_offset if raw_high_offset is not None else -SUSTAINABLE_GROWTH_WORST_SPREAD
        )
        g1_values["normal"] = blended_normal
        g1_values["best"] = blended_normal + high_offset
        g1_values["worst"] = blended_normal - low_offset

    # see G1_CAP's/G1_FLOOR's comments
    g1_values = {name: max(min(value, G1_CAP), G1_FLOOR) for name, value in g1_values.items()}

    if basis == "dividends":
        # see G2_DIVIDENDS_FLOOR's comment
        g2_values = {name: max(value, G2_DIVIDENDS_FLOOR) for name, value in g1_values.items()}
    else:
        # g2 = clamp(this tier's own g1, GROWTH_BASIS_G2_FLOOR, flat
        # GROWTH_BASIS_G2 default) - see that constant's own comment for
        # why the floor half was added (a plain min() only caps from
        # above, letting an extreme-below-ceiling g1, most consequentially
        # negative, pass straight through unchanged instead of fading).
        g2_values = {
            name: max(GROWTH_BASIS_G2_FLOOR, min(GROWTH_BASIS_G2[name], g1_values[name]))
            for name in GROWTH_BASIS_G2
        }

    if basis == "revenue":
        exit_multiples = {
            "normal": REVENUE_NORMAL_EXIT_MULTIPLE,
            "best": REVENUE_BEST_EXIT_MULTIPLE,
            "worst": REVENUE_WORST_EXIT_MULTIPLE,
        }
    elif basis == "dividends":
        # Shares eps/fcf's flat normal/best defaults; worst still varies
        # by ASSET_HEAVY_SECTORS - ported from valuation.py after a
        # separate, higher dividends-specific exit-multiple set was tried
        # and reverted there (live QSR data showed it made the gap WORSE,
        # not better - see that module's own comment for the full story).
        worst_exit_multiple = (
            WORST_EXIT_MULTIPLE_ASSET_HEAVY if fnd.get("sector") in ASSET_HEAVY_SECTORS
            else WORST_EXIT_MULTIPLE_DEFAULT
        )
        exit_multiples = {"normal": NORMAL_EXIT_MULTIPLE, "best": BEST_EXIT_MULTIPLE, "worst": worst_exit_multiple}
    else:
        # Sector-anchored ceiling on the normal/best exit multiple - ported
        # from valuation.py's identically-structured block (min(flat
        # default, this sector's SECTOR_MEDIAN_PE), preserving Technology's
        # existing curated calibration while pulling down genuinely
        # lower-multiple sectors like Energy/Financial Services). See that
        # module's own comment for the full rationale.
        #
        # Floored at the company's OWN trailing P/E - mirrors
        # generate_real_dataset.py's identical fix (found investigating
        # XOM's -98% reading right after CURATED_SCENARIOS was dropped: a
        # flat Energy sector median capped XOM's exit multiple BELOW its
        # own real trailing P/E, i.e. assuming the market prices it more
        # cheaply in 10 years than it already does today). See that
        # module's own comment for the full story.
        sector = fnd.get("sector")
        sector_median = SECTOR_MEDIAN_PE.get(sector)
        own_pe = fnd.get("pe_trailing")
        effective_median_candidates = [v for v in (sector_median, own_pe) if v is not None]
        effective_median = max(effective_median_candidates) if effective_median_candidates else None
        normal_exit_multiple = min(NORMAL_EXIT_MULTIPLE, effective_median) if effective_median is not None else NORMAL_EXIT_MULTIPLE
        best_exit_multiple = normal_exit_multiple + (BEST_EXIT_MULTIPLE - NORMAL_EXIT_MULTIPLE)
        worst_exit_multiple = (
            WORST_EXIT_MULTIPLE_ASSET_HEAVY if sector in ASSET_HEAVY_SECTORS
            else WORST_EXIT_MULTIPLE_DEFAULT
        )
        exit_multiples = {"normal": normal_exit_multiple, "best": best_exit_multiple, "worst": worst_exit_multiple}

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


def _dividend_stream_pv(dividend_rate, g2, discount_rate):
    """PV of a full 10-year dividend stream growing at g2 - ported from
    valuation.py's identically-named function (added to the terminal-only
    result for eps/fcf companies that pay SOME dividend below the
    dividends-basis threshold - previously discarded entirely for a full
    decade). See that module's own comment."""
    pv = 0.0
    dividend = dividend_rate
    for year in range(1, STAGE_1_YEARS + STAGE_2_YEARS + 1):
        dividend *= 1 + g2
        pv += dividend / (1 + discount_rate) ** year
    return pv


def _scenario_pv(basis, cf0, g1, g2, exit_multiple, discount_rate, dividend_rate=None):
    if basis == "dividends":
        return scenario_dcf_value(cf0, g1, g2, exit_multiple, discount_rate)
    pv = scenario_terminal_value(cf0, g1, g2, exit_multiple, discount_rate)
    if dividend_rate:
        pv += _dividend_stream_pv(dividend_rate, g2, discount_rate)
    return pv


def scenario_present_values(cf0, basis, scenarios, dividend_rate=None):
    return {
        name: _scenario_pv(
            basis, cf0, scenario["g1"], scenario["g2"], scenario["exit_multiple"], DISCOUNT_RATE, dividend_rate,
        )
        for name, scenario in scenarios.items()
    }


def intrinsic_value(cf0, basis, scenarios, dividend_rate=None):
    """None (not a fetch failure) when cf0 is missing or non-positive -
    ported verbatim from valuation.py."""
    if cf0 is None or cf0 <= 0:
        return None
    pvs = scenario_present_values(cf0, basis, scenarios, dividend_rate)
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
    raw_abs_pct = abs(pct)
    # Below 1%, neither "overvalued" nor "undervalued" is a claim the
    # number actually supports - ported from valuation.py (price==
    # intrinsic used to render "overvalued by ~0%").
    if raw_abs_pct < 1.0:
        return f"Intrinsic Value ({label}): ${intrinsic:.2f}\nvs Current Price: trading near fair value"

    verdict = "overvalued" if pct >= 0 else "undervalued"
    # ">" once actually capped, not "~" - ported from valuation.py (see
    # that module's own comment on why silently understating a genuinely
    # extreme gap is worse than an honestly-flagged floor).
    if raw_abs_pct > VALUATION_PCT_DISPLAY_CAP:
        return f"Intrinsic Value ({label}): ${intrinsic:.2f}\nvs Current Price: {verdict} by >{VALUATION_PCT_DISPLAY_CAP:.0f}%"
    return (
        f"Intrinsic Value ({label}): ${intrinsic:.2f}\n"
        f"vs Current Price: {verdict} by ~{raw_abs_pct:.0f}%"
    )

PUBLISHERS = [
    "Reuters", "Bloomberg", "MarketWatch", "CNBC", "Yahoo Finance",
    "Barron's", "The Motley Fool", "Seeking Alpha", "Business Insider", "AP News",
]

PRODUCTS = [
    "AI infrastructure", "cloud services", "next-gen hardware", "enterprise software",
    "flagship consumer devices", "streaming platform", "payments network",
    "electric vehicle lineup", "chip manufacturing", "logistics network",
    "drug development pipeline", "clinical trial portfolio", "energy production capacity",
    "refining operations", "industrial equipment lineup", "manufacturing capacity",
    "aircraft manufacturing", "product lineup",
]

# Keyed off the company's informal sector (company[2], SECTOR_MAP's keys) -
# confirmed live PRODUCTS picked with no sector gating produced 217/1600
# rows (13.6%) with a nonsensical pairing ("PepsiCo... chip manufacturing
# line", "Exxon Mobil... enterprise software margins", "Boeing...
# AI infrastructure line"). Falls back to the full PRODUCTS list for any
# sector not listed here (there shouldn't be one, given COMPANIES/
# SECTOR_MAP's fixed set, but this keeps make_fields from crashing if a
# new sector is ever added without updating this table in lockstep).
PRODUCTS_BY_SECTOR = {
    "tech": ["AI infrastructure", "cloud services", "enterprise software", "next-gen hardware"],
    "software": ["enterprise software", "cloud services", "AI infrastructure"],
    "semiconductors": ["chip manufacturing", "next-gen hardware", "AI infrastructure"],
    "e-commerce": ["logistics network", "flagship consumer devices"],
    "social-media": ["AI infrastructure", "cloud services"],
    "banking": ["payments network"],
    "entertainment": ["streaming platform"],
    "streaming": ["streaming platform"],
    "fintech": ["payments network"],
    "gig-economy": ["logistics network"],
    "retail": ["flagship consumer devices", "logistics network"],
    "crypto": ["payments network"],
    "robotics": ["next-gen hardware", "AI infrastructure"],
    "biotech": ["drug development pipeline", "clinical trial portfolio"],
    "logistics": ["logistics network"],
    "energy": ["energy production capacity", "refining operations"],
    "industrials": ["industrial equipment lineup", "manufacturing capacity"],
    "aerospace": ["aircraft manufacturing", "next-gen hardware"],
    "consumer-staples": ["flagship consumer devices", "product lineup"],
    "auto": ["electric vehicle lineup"],
}

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
# `user_query`, consistent with the resolved direction (BUY/SELL/
# HOLD; MIXED_SIGNAL_SCENARIOS resolves to one of these before this
# lookup happens, so only 3 directions are needed here). One template per
# (question type, direction) - deliberately plain/formulaic since this is
# what the model should learn to produce, not literary variety.
ANSWER_TEMPLATES = {
    "direction": {
        "BUY": "The recent news points to upward momentum for {ticker}, so the near-term bias leans higher.",
        "SELL": "The recent news points to downward pressure on {ticker}, so the near-term bias leans lower.",
        "HOLD": "The recent news doesn't point clearly in either direction for {ticker}, so a flat near-term move is the more likely outcome.",
    },
    "buy": {
        "BUY": "Yes, the current signals lean favorably enough that {ticker} looks like a reasonable buy here.",
        "SELL": "No, the current signals are negative enough that {ticker} doesn't look like a buy right now.",
        "HOLD": "It's a close call - nothing here strongly argues for or against buying {ticker} at current levels.",
    },
    "outlook": {
        "BUY": "The outlook for {ticker} this quarter looks positive based on the latest developments.",
        "SELL": "The outlook for {ticker} this quarter looks challenged based on the latest developments.",
        "HOLD": "The outlook for {ticker} this quarter looks steady, without a clear positive or negative catalyst.",
    },
    "worry": {
        "BUY": "No significant cause for concern - the latest news on {ticker} is constructive.",
        "SELL": "Some caution is warranted - the latest news on {ticker} raises real concerns.",
        "HOLD": "Not particularly - nothing in the latest news materially changes the risk picture for {ticker}.",
    },
    "impact": {
        "BUY": "The latest news should be a net positive for {ticker}.",
        "SELL": "The latest news should weigh on {ticker}.",
        "HOLD": "The latest news is unlikely to move {ticker} much either way.",
    },
    "sell": {
        "BUY": "Not really - the current signals argue for holding rather than selling {ticker}.",
        "SELL": "It's a reasonable moment to consider trimming {ticker}, given the negative signals.",
        "HOLD": "There's no strong signal here to justify selling {ticker} now versus holding.",
    },
    "sentiment": {
        "BUY": "Sentiment on {ticker} is bullish today.",
        "SELL": "Sentiment on {ticker} is bearish today.",
        "HOLD": "Sentiment on {ticker} is neutral today.",
    },
    "earnings": {
        "BUY": "The signals point toward {ticker} beating expectations.",
        "SELL": "The signals point toward {ticker} falling short of expectations.",
        "HOLD": "There's no strong signal either way on whether {ticker} beats expectations.",
    },
    "read": {
        "BUY": "Overall, {ticker} looks bullish based on the current data and news.",
        "SELL": "Overall, {ticker} looks bearish based on the current data and news.",
        "HOLD": "Overall, {ticker} looks balanced - no strong read either way right now.",
    },
    "none": {
        "BUY": "{ticker} is showing a bullish setup based on current data and news.",
        "SELL": "{ticker} is showing a bearish setup based on current data and news.",
        "HOLD": "{ticker} looks neutral right now, without a clear directional catalyst.",
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
# BUY_SCENARIOS / SELL_SCENARIOS entries are 3-tuples:
# (headline_templates, reasoning_template, tier), where tier is "clear"
# (the original, blatant-signal templates) or "subtle" (real-headline-
# style, softer signal, still a decisive direction). HOLD_SCENARIOS stays
# 2-tuples - there's no "subtle HOLD" concept, these are meant to stay
# genuinely flat so the clear/subtle-BUY/SELL vs HOLD boundary
# stays learnable. All templates use the same fill fields: {name} {ticker}
# {q} {rev} {beat} {drop} {units} {product} {region} {role} {buyback}
# {divhike} {rally} - unused fields in a given template are simply not
# referenced.
# ---------------------------------------------------------------------------

BUY_SCENARIOS = [
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

SELL_SCENARIOS = [
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

# Reasoning is a LIST of 2-3 phrasing variants per entry (one picked at
# random per row, same pattern as QUALITY_*_PHRASINGS/VALUATION_CONFLICT_
# PHRASINGS above) rather than one fixed sentence - confirmed live these
# 16 templates were the actual source of the dataset's template-
# concentration problem: HOLD's 28% category weight spread across
# just 16 equally-likely templates means each one appears ~20+ times in
# train by chance alone, and with only ONE fixed sentence each, that's
# 20+ byte-identical reasoning strings apiece - the single largest
# contributor to the top-20-templates-cover-40%-of-rows finding (see
# docs/valuation-dataset-prompt-audit.md's D2). HOLD's own scenarios
# have less real narrative variety to work with than a news-driven
# category (there are only so many ways to say "nothing happened"), so
# unlike QUALITY_*_PHRASINGS this doesn't eliminate the risk, but it
# meaningfully dilutes it - the SAME underlying judgment (this event
# type is immaterial) no longer maps to one memorizable string.
HOLD_SCENARIOS = [
    (["{name} ({ticker}) held its annual shareholder meeting, re-electing current board members and approving standard compensation packages"],
     ["Routine governance updates and board re-elections do not materially alter {ticker}'s financial outlook.",
      "Standard board re-elections and routine governance matters carry no real signal for {ticker}'s fundamentals.",
      "Nothing about a routine annual meeting changes the financial picture for {ticker}."]),
    (["{name} ({ticker}) announced a minor realignment of internal operational segments to streamline corporate reporting"],
     ["Internal restructuring without reported workforce reductions or segment sales is financially neutral for {ticker}.",
      "A reporting-line realignment with no layoffs or divestitures attached doesn't move the needle on {ticker}'s fundamentals.",
      "Absent any workforce cuts or asset sales, this kind of internal reshuffling is financially inconsequential for {ticker}."]),
    (["{name} ({ticker}) reported Q{q} revenue of ${rev}B, in line with analyst expectations"],
     ["In-line results confirm the existing outlook for {ticker} without introducing new directional information.",
      "Results landing right at expectations don't shift {ticker}'s outlook in either direction.",
      "Matching consensus, rather than beating or missing it, gives no new signal on {ticker}."]),
    (["Analysts maintained their 'Hold' rating on {ticker} ({name}), citing balanced risk/reward at current levels"],
     ["A maintained Hold rating reflects no material change in the balance of risks for {ticker}.",
      "Analysts holding their rating steady signals the risk/reward on {ticker} hasn't meaningfully shifted.",
      "An unchanged Hold call is a non-event for {ticker} - the analyst's view simply hasn't moved."]),
    (["{name} ({ticker}) unveiled its next-generation {product} at an industry event, with pricing and availability details expected later this year"],
     ["Without pricing, availability, or financial guidance attached, a product preview carries no near-term earnings implication.",
      "A preview with no pricing or shipping details attached says nothing about near-term earnings for {ticker}.",
      "Until pricing and availability are announced, this kind of product reveal has no financial signal to extract."]),
    (["{name} ({ticker}) filed its routine quarterly report with no material changes to previously issued guidance"],
     ["A routine filing that reaffirms existing guidance introduces no new information.",
      "Reaffirming prior guidance in a routine filing tells the market nothing it didn't already know.",
      "This filing changes nothing - guidance is exactly where it already stood."]),
    (["{name} ({ticker}) named a new {role} to its leadership team, effective next quarter"],
     ["A leadership appointment outside the CEO/CFO level is not typically financially material on its own.",
      "A new hire below the C-suite's top two roles rarely moves the financial needle by itself.",
      "Sub-CEO/CFO leadership changes are routine and don't carry standalone financial weight for {ticker}."]),
    (["{name} ({ticker}) made a small minority investment in a {product} startup, with terms not disclosed"],
     ["An undisclosed minority stake is too small and uncertain in scope to be directionally meaningful for {ticker}.",
      "With neither size nor terms disclosed, a minority investment like this is too vague to read as a signal for {ticker}.",
      "A stake this small and this undisclosed doesn't tell us anything directional about {ticker}."]),
    (["An analyst initiated coverage on {ticker} ({name}) with a 'Neutral' rating and no strong directional view"],
     ["A neutral initiation with no strong view either way introduces no new directional information for {ticker}.",
      "An analyst starting coverage with an explicitly neutral stance isn't taking a side on {ticker}.",
      "A 'Neutral' initiation is, by design, not a directional call on {ticker}."]),
    (["{name} ({ticker}) declined to comment on market speculation regarding a potential acquisition"],
     ["An unconfirmed rumor with no company statement carries no verifiable financial information for {ticker}.",
      "Speculation the company won't confirm or deny isn't something to trade {ticker} on.",
      "Without company confirmation, this remains market chatter, not verifiable information about {ticker}."]),
    (["An executive at {name} ({ticker}) sold shares under a pre-scheduled 10b5-1 trading plan"],
     ["Pre-scheduled sales under a 10b5-1 plan are routine and don't reflect a discretionary view on {ticker}'s prospects.",
      "10b5-1 sales are set up in advance precisely so they don't signal anything about the executive's current view of {ticker}.",
      "Because these sales were scheduled ahead of time, they say nothing about how the executive feels about {ticker} today."]),
    (["{name} ({ticker}) presented at an industry conference, reiterating previously disclosed strategic priorities"],
     ["Reiterating existing strategy at a conference introduces no new financial information for {ticker}.",
      "Repeating an already-disclosed strategy at a conference doesn't add anything new about {ticker}.",
      "There's no new financial information here - just a restatement of {ticker}'s known priorities."]),
    (["{name} ({ticker}) settled a legal claim for an amount consistent with previously reserved funds"],
     ["A settlement within already-reserved amounts has no incremental impact on {ticker}'s financial position.",
      "Since the funds were already set aside, this settlement doesn't change {ticker}'s financial position at all.",
      "A settlement that matches existing reserves is a balance-sheet non-event for {ticker}."]),
    (["{name} ({ticker}) rebranded its {product} line with a new name and visual identity"],
     ["A branding update with no pricing or product changes is not typically financially material for {ticker}.",
      "A cosmetic rebrand, with pricing and the product itself unchanged, has no real financial weight for {ticker}.",
      "Visual identity changes alone don't move the financial story for {ticker}."]),
    (["{name} ({ticker}) confirmed capital expenditure plans for {product} in line with previous guidance"],
     ["Spending in line with prior guidance confirms, rather than changes, the existing outlook for {ticker}.",
      "Capex tracking exactly to prior guidance is confirmation, not new information, for {ticker}.",
      "Nothing here updates {ticker}'s outlook - spending is right where it was already expected to be."]),
    (["Analysts left their price target on {ticker} ({name}) unchanged following a routine quarterly review"],
     ["An unchanged price target after a routine review reflects no material shift in analysts' view of {ticker}.",
      "Analysts holding their target steady after a routine look tells us their view of {ticker} hasn't changed.",
      "A price target left untouched after a routine review is itself the signal: no material shift in view."]),
    # Bare price-recap headlines - reports that the stock moved, without
    # stating why. Real financial news is full of this exact shape ("Stock
    # Trades Up, Here Is Why", "X stock is up N% today - here's what we
    # see"), and confirmed live (v16 eval misclassifications on BA/XOM) the
    # model treats it as a real catalyst and follows the headline's
    # face-value tone - because every "shares rallied/fell" template
    # anywhere else in this file (VALUATION_SIGNAL_SCENARIOS' news_wins
    # tier) always pairs the move with an explicit, named reason ("after
    # its new product line sold out," "after demand came in well below
    # expectations"). The model has literally never been shown the
    # reason-withheld case, so it has nothing to fall back on except
    # over-applying the "price move = strong signal" lesson from the
    # reason-given templates. These two entries are that missing example:
    # a price move stated with NO reason is backward-looking noise, not a
    # catalyst, regardless of which direction the tone leans.
    (["{name} ({ticker}) stock traded higher today - here's what's behind the move, according to one report",
      "{name} ({ticker}) shares are up today; here's what we see in the data"],
     ["A reported price move with no stated reason gives no new information about {ticker}'s fundamentals - the move itself isn't evidence of anything.",
      "The stock already moved and the headline doesn't say why - that's a recap, not a signal to trade {ticker} on.",
      "Without a concrete, named reason behind it, a reported price move for {ticker} is backward-looking noise, not a forward-looking catalyst."]),
    (["{name} ({ticker}) stock traded lower today - here's what's behind the move, according to one report",
      "{name} ({ticker}) shares are down today; here's what we see in the data"],
     ["A reported price move with no stated reason gives no new information about {ticker}'s fundamentals - the move itself isn't evidence of anything.",
      "The stock already moved and the headline doesn't say why - that's a recap, not a signal to trade {ticker} on.",
      "Without a concrete, named reason behind it, a reported price move for {ticker} is backward-looking noise, not a forward-looking catalyst."]),
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
# strongly enough to invoke it on CLEAN, single-direction BUY/SELL
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
     "SELL"),
    (["{name} ({ticker}) missed quarterly revenue estimates by {beat}%",
      "{name} ({ticker}) simultaneously announced a ${buyback}B share buyback program"],
     "A buyback doesn't paper over the revenue miss - underlying demand looks weaker regardless of the capital-return announcement.",
     "SELL"),
    (["{name} ({ticker}) reported a strong Q{q}, with revenue of ${rev}B beating estimates by {beat}%",
      "Analysts flagged {product} inventory buildup as a risk to next quarter's results"],
     "Current results are genuinely strong, but the flagged inventory risk introduces real uncertainty about next quarter - bullish, with tempered confidence.",
     "BUY"),
    (["{name} ({ticker}) issued a recall affecting {units}k {product} units",
      "The recall follows a quarter of record {product} sales, reported just last week"],
     "A recall's safety and legal risk outweighs the prior quarter's already-priced-in sales record.",
     "SELL"),
    (["{name} ({ticker}) disclosed a regulatory fine related to {product} practices",
      "{name} ({ticker}) also raised its full-year guidance, citing broad-based demand strength"],
     "A one-time fine is a sunk cost that doesn't change the forward outlook; the guidance raise, grounded in broad-based demand strength, is what should actually move the stock - bullish, tempered by the fine's reputational overhang.",
     "BUY"),
    (["{name} ({ticker}) reported solid Q{q} results in line with expectations",
      "Broader market headlines describe a sector-wide selloff unrelated to {ticker}'s own fundamentals"],
     "{ticker}'s own results are solid; the sector-wide selloff in the noise headlines is not company-specific and shouldn't be weighted into {ticker}'s own outlook - bullish, with confidence tempered by broader market uncertainty.",
     "BUY"),
    (["{name} ({ticker}) beat Q{q} revenue estimates by {beat}%",
      "{name} ({ticker}) separately announced its CEO will step down at year-end as part of a planned transition"],
     "An orderly, pre-planned leadership transition tells you little you didn't already expect; the revenue beat is the harder data point here - bullish, tempered only by the normal uncertainty a CEO change introduces.",
     "BUY"),
    (["Analysts upgraded {ticker} ({name}) to 'Buy' citing long-term {product} potential",
      "{name} ({ticker}) issued weak near-term guidance, citing short-term {product} softness"],
     "A long-term-oriented upgrade doesn't change what management itself just said about the next few quarters - a concrete near-term guidance cut from the company carries more weight than an analyst's multi-year thesis - bearish.",
     "SELL"),
    (["{name} ({ticker}) missed Q{q} earnings estimates by {beat}%",
      "{name} ({ticker}) simultaneously raised its quarterly dividend by {divhike}%"],
     "Raising the dividend doesn't undo an actual earnings miss - a payout bump is the smaller signal next to results falling short - bearish.",
     "SELL"),
    (["{name} ({ticker})'s new {product} line sold out within days of launch",
      "{name} ({ticker}) separately recalled a small batch of an older, legacy product line unrelated to {product}"],
     "The flagship {product} launch is the primary current growth driver; a recall isolated to an unrelated legacy line is a minor operational item by comparison - bullish, with confidence tempered by the recall.",
     "BUY"),
    (["{name} ({ticker}) received regulatory approval for {product} expansion into new markets",
      "{name} ({ticker})'s credit rating was downgraded the same week, citing rising leverage"],
     "A credit downgrade reflects a structural balance-sheet concern that outweighs a single market-expansion approval - bearish, with confidence tempered by the approval's longer-term upside.",
     "SELL"),
    (["{name} ({ticker}) reported Q{q} results in line with expectations",
      "{name} ({ticker}) raised full-year guidance, citing accelerating {product} momentum"],
     "In-line current results don't cancel out a genuine guidance raise - forward-looking guidance is what should actually be priced in here - bullish.",
     "BUY"),
    # --- two more orderly-CEO-transition pairings, different accompanying
    # signal each time (one more bullish-paired, one bearish) - a single
    # example of "orderly transition doesn't move the needle, the other
    # signal does" wasn't enough repetition: confirmed live across multiple
    # eval runs, the model kept defaulting to SELL/HOLD on this exact
    # pattern regardless of what the paired signal actually said, most
    # likely because a CEO departure reads negative by default from
    # pretraining alone and one counter-example can't overcome that prior.
    # Varying which direction the OTHER signal points (not always bullish)
    # is deliberate - the lesson is "the transition itself is near-neutral,
    # weigh the real signal," not "CEO transition secretly means bullish."
    (["{name} ({ticker}) raised its full-year guidance, citing accelerating {product} demand",
      "{name} ({ticker}) separately announced its CEO will step down at year-end as part of a planned transition"],
     "A guidance raise is a concrete, forward-looking signal from management itself; an orderly, pre-planned leadership transition doesn't offset that - bullish, tempered only by the normal uncertainty a CEO change introduces.",
     "BUY"),
    (["{name} ({ticker}) missed Q{q} revenue estimates by {beat}%",
      "{name} ({ticker}) separately announced its CEO will step down at year-end as part of a planned transition"],
     "A revenue miss is the harder, more decision-relevant data point here; an orderly, pre-planned leadership transition doesn't make a miss any less real - bearish, tempered only by the normal uncertainty a CEO change introduces.",
     "SELL"),
]
# Every entry above is now a required 3-tuple (headline_templates,
# reasoning_template, direction) - used to silently default a missing
# direction to "SELL" for entries that omitted it. Harmless by luck (the
# one entry that relied on it did want SELL) but a real risk: a future
# entry wanting BUY that forgot the third element would've been
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
# v11 eval showed real-val HOLD collapsing to ~10% correct, with
# reasoning matching "_alone"'s phrasing verbatim even on trivial (2%)
# valuation gaps. Root cause turned out narrower than "remove the lesson
# entirely": "_alone"'s headlines ("held a routine analyst call with no
# notable updates," "reiterated prior guidance with no other updates") are
# near-duplicates of HOLD_SCENARIOS' own "routine, no material news"
# headlines - and HOLD_SCENARIOS used a fully random (uncorrelated)
# valuation draw, same as every other non-VALUATION_SIGNAL category, so it
# could ALSO show a large gap purely by chance. Same surface shape (routine
# headline + some gap) trained to opposite conclusions depending on which
# category happened to roll it. Real fix (see HOLD_SCENARIOS' own
# valuation-rendering below): pin HOLD's gap to a small, genuinely
# insignificant range instead of letting it collide with "_alone"'s extreme
# range - so gap SIZE, not headline wording, is what actually distinguishes
# "ignore this" from "this is a real signal," which is what should have
# been distinguishing them all along. That keeps the lesson an extreme
# valuation gap with neutral-to-mild news should still lean toward the
# valuation's direction, not fall back to HOLD just because there's no
# news catalyst - while a small/routine gap correctly stays HOLD.
#
# Each entry: (headline_templates, reasoning_template, direction,
# valuation_verdict, gap_tier, confidence_tier). This whole category is
# RETIRED as of the two-stage pipeline redesign (see REACTION_WEIGHTS' own
# comment) - valuation_verdict/gap_tier used to feed valuation_block_with_
# gap (removed) to construct a controlled, specific gap instead of the
# normal random draw, which is what taught the model to weigh valuation
# against news at all. fusion_rules.fuse() now does that job
# deterministically for every category, so this list survives only as
# SOURCE DATA: the redistribution block right after it sorts each entry's
# hand-authored headline/reasoning into BUY_SCENARIOS/SELL_SCENARIOS/
# HOLD_SCENARIOS by what the headline itself says, discarding
# valuation_verdict/gap_tier (fuse() computes the equivalent from a real
# gap now) and confidence_tier (Task B carries no confidence - see
# REACTION_WEIGHTS' own comment).
VALUATION_SIGNAL_SCENARIOS = [
    # --- extreme gap, no corroborating news: valuation is the only signal ---
    (["{name} ({ticker}) held a routine analyst call with no notable updates to prior commentary"],
     "No fresh news moves the needle here, but {ticker} is trading at a steep discount to its estimated intrinsic value - a real, if imperfect, signal on its own. DCF-style estimates carry real model uncertainty, so this leans bullish without the higher confidence a concrete catalyst would justify.",
     "BUY", "undervalued", "extreme", "extreme_alone"),
    (["{name} ({ticker}) reiterated prior full-year guidance with no other updates this week"],
     "Nothing new in the news, but {ticker} is trading at a steep premium to its estimated intrinsic value - worth weighing even without a fresh catalyst, tempered by the real uncertainty in any DCF-style estimate.",
     "SELL", "overvalued", "extreme", "extreme_alone"),
    # --- moderate gap, no corroborating news: weaker evidence, lower confidence ---
    (["{name} ({ticker}) traded in a narrow range this week with no company-specific news"],
     "No headline catalyst, but {ticker}'s current price sits at a modest discount to its estimated intrinsic value - a real but comparatively soft signal, especially with no news to corroborate it, so confidence here stays low.",
     "BUY", "undervalued", "moderate", "moderate_alone"),
    (["{name} ({ticker}) saw light trading volume in an otherwise uneventful week"],
     "No headline catalyst, but {ticker}'s current price sits at a modest premium to its estimated intrinsic value - a real but comparatively soft signal on its own, so confidence here stays low.",
     "SELL", "overvalued", "moderate", "moderate_alone"),
    # --- extreme gap, reinforced by a mild/subtle same-direction headline ---
    (["{name} ({ticker}) saw a modest uptick in institutional buying interest, according to the latest filings"],
     "{ticker} already looks meaningfully undervalued against its estimated intrinsic value, and the pickup in institutional interest is a soft but same-direction confirmation - still tempered by the underlying uncertainty in any valuation estimate, but more confident than the valuation gap alone would justify.",
     "BUY", "undervalued", "extreme", "extreme_reinforced"),
    (["{name} ({ticker}) saw a modest uptick in insider selling activity, according to the latest filings"],
     "{ticker} already looks meaningfully overvalued against its estimated intrinsic value, and the pickup in insider selling is a soft but same-direction confirmation - still tempered by the underlying uncertainty in any valuation estimate, but more confident than the valuation gap alone would justify.",
     "SELL", "overvalued", "extreme", "extreme_reinforced"),
    # --- extreme gap, but a concrete near-term catalyst points the other
    # way - the news should win, same principle as MIXED_SIGNAL_SCENARIOS ---
    (["{name} ({ticker}) cut its full-year guidance, citing softening {product} demand heading into next quarter"],
     "{ticker} screens as meaningfully undervalued on an estimated-intrinsic-value basis, but a concrete, company-issued guidance cut is a more reliable near-term signal than a longer-horizon valuation estimate - the guidance cut should dominate here, not the valuation gap.",
     "SELL", "undervalued", "extreme", "news_wins"),
    (["{name} ({ticker}) raised its full-year guidance, citing accelerating {product} demand"],
     "{ticker} screens as meaningfully overvalued on an estimated-intrinsic-value basis, but a concrete, company-issued guidance raise is a more reliable near-term signal than a longer-horizon valuation estimate - the guidance raise should dominate here, not the valuation gap.",
     "BUY", "overvalued", "extreme", "news_wins"),
    # --- a real beat/miss confirms the quarter was genuinely strong/weak,
    # but - unlike the pairs below - names no reason to expect MORE of the
    # same going forward, so it shouldn't carry the same full "news wins"
    # weight. Confirmed live (user feedback on a value-investing read of a
    # BA row): a big move shows the business had a strong/weak quarter, but
    # that alone doesn't mean there's more room to run - only a stated
    # reason for continued upside (expanding market, improving competitive
    # position) justifies overriding an already-rich valuation, and
    # symmetrically only a stated reason for continued deterioration
    # justifies overriding an already-cheap one. A bare beat/miss with no
    # such reason lands at HOLD: real evidence the business is sound
    # (or struggling), tempered by "no headroom" (or "no confirmed further
    # downside") rather than pretending that tension away. earnings_direction
    # is set explicitly below (not left to fall back to `direction`, which
    # is HOLD here) so the rendered Earnings block still honestly shows
    # the real beat/miss the headline describes. Contrast with the
    # shares-rallied/fell pair further below, which DOES name a forward
    # reason and so stays a full news_wins BUY/SELL call.
    (["{name} ({ticker}) missed Q{q} revenue estimates by {beat}%"],
     "A confirmed revenue miss is real, current-quarter evidence of business weakness for {ticker} - but the headline names no reason to expect further deterioration, and the stock already screens as meaningfully undervalued on an estimated-intrinsic-value basis. The miss is a genuine signal worth weighing; it isn't, by itself, a stronger sell case than the valuation gap already argues against.",
     "HOLD", "undervalued", "extreme", "news_wins_no_headroom"),
    (["{name} ({ticker}) beat Q{q} revenue estimates by {beat}%"],
     "A confirmed revenue beat is real, current-quarter evidence {ticker}'s business is executing well - but the headline names no reason to expect further upside, and the stock already screens as meaningfully overvalued on an estimated-intrinsic-value basis. A strong quarter confirms the business is sound; it doesn't by itself mean there's more room left to run at this price.",
     "HOLD", "overvalued", "extreme", "news_wins_no_headroom"),
    # --- same principle, analyst rating-change phrasing ---
    (["Analysts downgraded {ticker} ({name}) to 'Sell', citing slowing {product} demand"],
     "{ticker} screens as meaningfully undervalued on an estimated-intrinsic-value basis, but a fresh analyst downgrade reflects a specific, current view of deteriorating {product} demand that a static valuation estimate can't capture - the downgrade should dominate here, not the valuation gap.",
     "SELL", "undervalued", "extreme", "news_wins"),
    (["Analysts upgraded {ticker} ({name}) to 'Buy', citing accelerating {product} demand"],
     "{ticker} screens as meaningfully overvalued on an estimated-intrinsic-value basis, but a fresh analyst upgrade reflects a specific, current view of improving {product} demand that a static valuation estimate can't capture - the upgrade should dominate here, not the valuation gap.",
     "BUY", "overvalued", "extreme", "news_wins"),
    # --- same principle, price-move-tied-to-a-demand-catalyst phrasing
    # (matches how real headlines like "stock pops after earnings" or
    # "rides winning streak to an X% gain" actually read) - unlike the bare
    # beat/miss pair above, these name a FORWARD-looking reason (competitive
    # share loss, newly opened markets) for the move to continue, which is
    # what earns the full news_wins override instead of landing at HOLD
    # like the pair above ---
    (["{name} ({ticker}) shares fell {drop}% after {product} demand came in well below expectations, with management flagging continued share loss to competitors heading into next quarter"],
     "{ticker} screens as meaningfully undervalued on an estimated-intrinsic-value basis, but a sharp move tied to a concrete demand shortfall - and management's own warning of further competitive share loss - points to continued deterioration ahead, not a one-quarter blip. That forward-looking signal is a more reliable near-term read than a longer-horizon valuation estimate - the demand shortfall should dominate here, not the valuation gap.",
     "SELL", "undervalued", "extreme", "news_wins"),
    (["{name} ({ticker}) shares rallied {rally}% after {product} demand blew past expectations, with management citing accelerating adoption in newly opened markets"],
     "{ticker} screens as meaningfully overvalued on an estimated-intrinsic-value basis, but a sharp move tied to a concrete demand beat - and management's own signal of accelerating adoption in newly opened markets - points to real room for continued growth, not just a one-quarter pop. That forward-looking signal is a more reliable near-term read than a longer-horizon valuation estimate - the demand beat should dominate here, not the valuation gap.",
     "BUY", "overvalued", "extreme", "news_wins"),
]

# Redistributes VALUATION_SIGNAL_SCENARIOS' hand-authored content into the
# news_reaction pools below by what each entry's HEADLINE actually says,
# not by the old category's resolved BUY/SELL/HOLD label (which used to
# be entangled with a hand-picked valuation gap that fusion_rules.fuse()
# now computes independently - see REACTION_WEIGHTS' own comment).
# - "_alone"/"moderate_alone": the headline explicitly says nothing
#   happened ("no notable updates," "narrow range... no company-specific
#   news") - genuinely neutral regardless of what the old valuation-driven
#   direction used to be.
# - "_reinforced"/"news_wins"/"news_wins_no_headroom": all describe SOME
#   real (if sometimes soft) news content, and `direction` here was always
#   the news-implied lean (BUY==good news, SELL==bad news), never a coin
#   flip - a safe, direct proxy for news_reaction.
_VALUATION_SIGNAL_TO_GOOD = []
_VALUATION_SIGNAL_TO_BAD = []
_VALUATION_SIGNAL_TO_NEUTRAL = []
for _headline_templates, _reasoning_template, _direction, _verdict, _gap_tier, _confidence_tier in VALUATION_SIGNAL_SCENARIOS:
    if _confidence_tier in ("extreme_alone", "moderate_alone"):
        _VALUATION_SIGNAL_TO_NEUTRAL.append((_headline_templates, [_reasoning_template]))
    else:
        _tier = "subtle" if _confidence_tier == "extreme_reinforced" else "clear"
        _target = _VALUATION_SIGNAL_TO_GOOD if _direction == "BUY" else _VALUATION_SIGNAL_TO_BAD
        _target.append((_headline_templates, _reasoning_template, _tier))

# MIXED_SIGNAL_SCENARIOS' own conflict-weighing IS a "read the news
# correctly" skill (which signal should dominate a headline pair), not a
# valuation-fusion one - it maps directly onto good/bad by its own
# resolved `direction`, same principle as the VALUATION_SIGNAL
# redistribution above. "clear" tier since these are decisive,
# unambiguous-once-weighed cases, not soft/indirect signals.
_MIXED_TO_GOOD = [(h, r, "clear") for h, r, d in MIXED_SIGNAL_SCENARIOS if d == "BUY"]
_MIXED_TO_BAD = [(h, r, "clear") for h, r, d in MIXED_SIGNAL_SCENARIOS if d == "SELL"]

BUY_SCENARIOS = BUY_SCENARIOS + _VALUATION_SIGNAL_TO_GOOD + _MIXED_TO_GOOD
SELL_SCENARIOS = SELL_SCENARIOS + _VALUATION_SIGNAL_TO_BAD + _MIXED_TO_BAD
HOLD_SCENARIOS = HOLD_SCENARIOS + _VALUATION_SIGNAL_TO_NEUTRAL

# --- overreaction_down/up: NEW as of the two-stage pipeline redesign -
# modest, routine-register news (deliberately similar to HOLD_SCENARIOS'
# and the BUY/SELL "subtle" tier's real-headline style) paired with a
# fabricated price move much larger than the headline alone would justify
# (see _fabricate_move_pct below) - the actual training signal for this
# class is the (modest headline, outsized move) MISMATCH, not the
# headline's content in isolation. Reasoning explicitly names the
# overdone-ness rather than treating the move as a real catalyst.
OVERREACTION_DOWN_SCENARIOS = [
    (["{name} ({ticker}) shares fell sharply this week despite reporting Q{q} results broadly in line with estimates"],
     "A sharp decline alongside results that were merely in-line, not a genuine miss, looks disproportionate - this reads as an overreaction rather than a fundamentals-driven repricing."),
    (["{ticker} ({name}) shares dropped following a minor, routine SEC filing update with no new financial disclosures"],
     "A routine filing update carries essentially no new information, so a meaningful share-price decline around it looks like an overreaction rather than a response to real news."),
    (["{name} ({ticker}) shares slid after a single analyst trimmed their price target slightly, citing near-term caution"],
     "One analyst's modest, cautious price-target trim is a small, incremental data point - a sizable share-price drop in response looks larger than the news itself would justify."),
    (["{ticker} ({name}) shares fell amid broad market volatility, with no company-specific news reported"],
     "Without any company-specific catalyst, a notable decline in {ticker} tied to broad market volatility looks like it's tracking sentiment rather than anything specific to the business - a plausible overreaction."),
    (["{name} ({ticker}) shares dropped after a mid-tier executive departure, unrelated to the CEO or CFO roles"],
     "A departure below the C-suite's top roles is routine and rarely material - a sharp share-price move on this kind of news looks larger than the underlying event warrants."),
    (["{ticker} ({name}) shares fell after a routine, previously-scheduled regulatory filing generated some negative headlines"],
     "A routine, expected filing generating outsized negative headlines - and a matching share-price drop - looks like a reaction to the coverage itself rather than to any new substantive information."),
    (["{name} ({ticker}) shares declined following a competitor's earnings miss, despite no direct read-through disclosed"],
     "A competitor's miss doesn't automatically apply to {ticker} absent a stated read-through - a meaningful decline on secondhand, unconfirmed contagion looks like an overreaction."),
    (["{ticker} ({name}) shares slipped after a minor product recall covering a small, discontinued product line"],
     "A recall confined to a small, already-discontinued line has limited real financial exposure - a sizable share-price drop looks disproportionate to the scope of the issue."),
    (["{name} ({ticker}) shares fell after a short-seller report that cited mostly previously-disclosed information"],
     "A short report built mainly on information the company had already disclosed doesn't introduce much new risk - a sharp drop in response looks like an overreaction to the framing rather than to new facts."),
    (["{ticker} ({name}) shares dropped following a delayed but ultimately routine regulatory approval"],
     "A delay that still ends in approval is a timing footnote, not a change in outcome - a meaningful share-price decline around it looks larger than the news itself justifies."),
    (["{name} ({ticker}) shares fell after a minor guidance footnote flagged a small, one-time currency headwind"],
     "A flagged one-time currency item is a modest, non-recurring factor - a sizable drop in response looks like it's pricing in more than a single footnote actually implies."),
    (["{ticker} ({name}) shares slid after weaker-than-expected trading volume was reported with no other news"],
     "Light trading volume on its own isn't a fundamental catalyst - a notable share-price decline attributed to it looks like an overreaction to a data point that says little about the business."),
]

OVERREACTION_UP_SCENARIOS = [
    (["{name} ({ticker}) shares rallied sharply this week despite reporting Q{q} results broadly in line with estimates"],
     "A sharp rally alongside results that were merely in-line, not a genuine beat, looks disproportionate - this reads as an overreaction rather than a fundamentals-driven repricing."),
    (["{ticker} ({name}) shares jumped following a minor, routine SEC filing update with no new financial disclosures"],
     "A routine filing update carries essentially no new information, so a meaningful share-price rally around it looks like an overreaction rather than a response to real news."),
    (["{name} ({ticker}) shares rose after a single analyst raised their price target slightly, citing modest optimism"],
     "One analyst's modest price-target raise is a small, incremental data point - a sizable share-price rally in response looks larger than the news itself would justify."),
    (["{ticker} ({name}) shares rallied amid broad market optimism, with no company-specific news reported"],
     "Without any company-specific catalyst, a notable rally in {ticker} tied to broad market optimism looks like it's tracking sentiment rather than anything specific to the business - a plausible overreaction."),
    (["{name} ({ticker}) shares rose after a mid-tier executive hire, unrelated to the CEO or CFO roles"],
     "A hire below the C-suite's top roles is routine and rarely material - a sharp share-price move on this kind of news looks larger than the underlying event warrants."),
    (["{ticker} ({name}) shares rallied after a routine, previously-scheduled regulatory filing generated some positive headlines"],
     "A routine, expected filing generating outsized positive headlines - and a matching share-price rally - looks like a reaction to the coverage itself rather than to any new substantive information."),
    (["{name} ({ticker}) shares rose following a competitor's strong earnings beat, despite no direct read-through disclosed"],
     "A competitor's beat doesn't automatically apply to {ticker} absent a stated read-through - a meaningful rally on secondhand, unconfirmed optimism looks like an overreaction."),
    (["{ticker} ({name}) shares climbed after unconfirmed market chatter about a potential partnership, with no company statement"],
     "Unconfirmed speculation the company hasn't addressed is a thin basis for a real move - a sizable rally on this kind of chatter looks disproportionate until there's an actual confirmation."),
    (["{name} ({ticker}) shares rose after a bullish note that cited mostly previously-disclosed information"],
     "A bullish note built mainly on information the company had already disclosed doesn't introduce much new upside case - a sharp rally in response looks like an overreaction to the framing rather than to new facts."),
    (["{ticker} ({name}) shares jumped following an expedited but otherwise routine regulatory approval"],
     "An approval arriving a bit early is a timing footnote, not a change in outcome - a meaningful share-price rally around it looks larger than the news itself justifies."),
    (["{name} ({ticker}) shares rose after a minor guidance footnote flagged a small, one-time currency tailwind"],
     "A flagged one-time currency tailwind is a modest, non-recurring factor - a sizable rally in response looks like it's pricing in more than a single footnote actually implies."),
    (["{ticker} ({name}) shares climbed after unusually heavy trading volume was reported with no other news"],
     "Heavy trading volume on its own isn't a fundamental catalyst - a notable share-price rally attributed to it looks like an overreaction to a data point that says little about the business."),
]

# Sector-wide macro/geopolitical developments whose fundamental linkage to
# companies in that sector is direct and well-established - a REAL change
# to the business's own economics (revenue per unit sold, margin, funding
# cost, addressable market), not just market sentiment about the sector.
# Picked at generation time based on the actual company's own informal
# sector (see make_example's "good"/"bad" branch), not a static template.
# Only 7 of ~20 informal sectors are covered - covers two of the four
# held-out tickers (BA -> aerospace, HRZN -> industrials) so this pattern
# actually gets held-out eval coverage, not just training exposure.
#
# Each value: (bullish_headline, bearish_headline, linkage_phrase).
# linkage_phrase fills SECTOR_LINKAGE_REASONING's own {linkage} slot -
# names the specific economic channel, not just "this sector is affected."
SECTOR_MACRO_HEADLINES = {
    # Deliberately NOT "oil price up = bullish, oil price down = bearish" -
    # that's momentum-following, the opposite of the value-investing
    # principle this whole category exists to teach. Oil is a cyclical
    # commodity: a LOW price (~$60/bbl or below) is historically closer to
    # the trough where value investors buy energy names, not a reason to
    # be bearish - and a HIGH price is closer to a cycle peak, where
    # trailing earnings (and the P/E/valuation gap computed from them) are
    # inflated and can look deceptively cheap right before a downturn - a
    # classic value trap. So: low price + undervalued -> BUY (buying
    # the trough), high price + overvalued -> SELL (wary of a peak
    # that's about to mean-revert), matching how the direction/verdict
    # pairing already works elsewhere in this dict.
    "energy": (
        "Crude oil prices slide to multi-year lows near $55/barrel amid oversupply concerns - historically close to where energy-sector value investors start buying, not where they sell",
        "Crude oil prices surge to decade highs above $110/barrel amid supply fears - a classic late-cycle peak, when trailing earnings and P/E ratios can look deceptively cheap right before a downturn",
        "oil is a cyclical commodity, so {ticker}'s current earnings and trailing multiples reflect where in that cycle prices are right now, not the cycle's long-run average",
    ),
    "semiconductors": (
        "Global chip shortage intensifies, driving up prices and demand across the semiconductor industry",
        "New export restrictions on advanced semiconductor technology threaten industry-wide international revenue",
        "industry-wide pricing power and addressable market directly set {ticker}'s own revenue opportunity",
    ),
    "banking": (
        "The Federal Reserve signals it will hold interest rates higher for longer, boosting bank net interest margins",
        "A deepening yield curve inversion raises funding costs and default risk across the lending sector",
        "the rate environment directly sets the net interest margin {ticker}'s lending business earns",
    ),
    "consumer-staples": (
        "Consumer staples demand proves resilient as shoppers trade down during economic uncertainty",
        "Persistent input cost inflation continues to squeeze margins across the consumer staples sector",
        "sector-wide demand and input costs directly set {ticker}'s own volume and margin trajectory",
    ),
    "crypto": (
        "Regulators signal a friendlier stance toward digital assets, lifting sentiment and volumes across the crypto sector",
        "Regulators announce a sweeping crackdown on digital asset exchanges, roiling the crypto sector",
        "sector-wide regulatory treatment directly sets the addressable market {ticker} can operate in",
    ),
    "aerospace": (
        "Escalating geopolitical tensions drive a sustained increase in global defense spending",
        "The aerospace and defense industry faces sector-wide program cost overruns and delivery delays",
        "defense budgets and program execution directly set the order book {ticker} draws revenue from",
    ),
    "industrials": (
        "A new infrastructure spending package signals sustained demand for industrial materials and equipment",
        "Manufacturing activity contracts for a third straight month, signaling weakening industrial demand",
        "sector-wide demand directly sets the order volume {ticker}'s own business depends on",
    ),
}

# Rewritten for the two-stage pipeline redesign: the old version of this
# template asserted "{ticker} already looks meaningfully {verdict}
# against its estimated intrinsic value" - a claim that used to be safe
# because VALUATION_SIGNAL_SCENARIOS controlled the valuation gap directly
# (valuation_block_with_gap, since removed). Now every category draws a
# REAL, independent random DCF gap (render_valuation), so this template
# can no longer presume what that gap says - it describes only the news
# itself; _fusion_explanation (see make_example) appends the correct
# valuation-based clause afterward, whatever the random draw turns out to
# be.
SECTOR_LINKAGE_REASONING = (
    "This sector-wide development is a real reinforcement rather than generic sentiment - {linkage} - "
    "which gives {ticker} a concrete, if indirect, {tone} signal beyond company-specific news alone."
)

# Fraction of "good"/"bad" rows (company's sector permitting - see
# SECTOR_MACRO_HEADLINES) that swap in a sector-linked headline instead of
# drawing from the regular BUY_SCENARIOS/SELL_SCENARIOS pool - was
# VALUATION_SIGNAL-only before this file's two-stage redesign, now feeds
# news_reaction directly (see make_example).
SECTOR_REINFORCED_PROB = 0.40

# Fraction of Task B rows that get an appended value-screen commentary
# sentence (see describe_value_screen) - NOT 1.0, so this vocabulary is
# learned as ONE input among several the model reasons over, not a rigid
# suffix every row carries. Applied uniformly across all five news_
# reaction classes now (previously VALUATION_SIGNAL got a separate,
# higher 0.25 rate before that category was retired - see this file's
# history item 12 - and BUY/SELL/MIXED got this same 0.15). Confirmed
# live (dataset-quality audit, pre-redesign): at the old two-tier setup,
# checklist vocabulary (quality/growth-adjusted-pricing language) reached
# reasoning text on only ~8% of ALL rows even with VALUATION_SIGNAL's own
# richer 0.25, leaving Price/Book, FCF yield, Price/Sales, and the
# sector-relative P/E comparison shown in market_data but never used in
# ANY target output; a still-earlier 0.50 rate on VALUATION_SIGNAL alone
# had the model reciting this exact phrasing near-verbatim on a REAL
# headline it had never trained on. 0.15 uniformly keeps well under
# either previously-confirmed memorization threshold while giving every
# reaction class the same coverage.
VALUE_SCREEN_COMMENTARY_PROB_OTHER_CATEGORIES = 0.15


def held_out_template_indices(templates):
    # Evenly-spaced indices across the whole list rather than a tail slice -
    # keeps the held-out val split representative even when a list is
    # internally grouped (e.g. BUY/SELL's "clear" templates followed
    # by "subtle" ones), instead of only ever holding out whichever group
    # happens to be listed last.
    #
    # Mid-point offset ((i + 0.5) * stride, not i * stride): at n_holdout=1,
    # int(0 * stride) is always 0 regardless of stride - "evenly spaced"
    # degenerates into "always the first element" for any single-item pick,
    # silently zeroing that template out of train every time. Confirmed live
    # for VALUATION_SIGNAL_SCENARIOS (see held_out_template_indices_stratified
    # below, which had the same bug): its two "extreme_alone" templates -
    # the only ones teaching "no news, but the valuation gap is huge -
    # follow the valuation" - are both listed first in their direction
    # group, so this always held BOTH out and put zero of them in train.
    # The mid-point offset still lands evenly-spaced picks when
    # n_holdout > 1, just without the n_holdout=1 collapse.
    n = len(templates)
    n_holdout = max(1, int(n * VAL_HOLDOUT_TEMPLATE_INDEX_FRACTION))
    stride = n / n_holdout
    return {int((i + 0.5) * stride) for i in range(n_holdout)}


# Computed AFTER the VALUATION_SIGNAL/MIXED redistribution above, so the
# held-out fraction is drawn from each pool's FULL final content (original
# + redistributed entries), not just the original BUY_SCENARIOS/
# SELL_SCENARIOS/HOLD_SCENARIOS lists as first authored. MIXED_SIGNAL_
# SCENARIOS/VALUATION_SIGNAL_SCENARIOS no longer get their own standalone
# holdout index - they're not dispatched as separate categories anymore
# (see REACTION_WEIGHTS' own comment).
BUY_HOLDOUT_IDX = held_out_template_indices(BUY_SCENARIOS)
SELL_HOLDOUT_IDX = held_out_template_indices(SELL_SCENARIOS)
HOLD_HOLDOUT_IDX = held_out_template_indices(HOLD_SCENARIOS)
OVERREACTION_DOWN_HOLDOUT_IDX = held_out_template_indices(OVERREACTION_DOWN_SCENARIOS)
OVERREACTION_UP_HOLDOUT_IDX = held_out_template_indices(OVERREACTION_UP_SCENARIOS)


def make_fields(company):
    # Generated once per example, not per headline - a multi-headline
    # scenario (e.g. a recall headline plus a "prior quarter" headline) must
    # reference the same product/quarter/numbers in both lines, not
    # independently re-rolled ones.
    ticker, name, informal_sector, rev_range = company
    return {
        "ticker": ticker,
        "name": name,
        "q": random.randint(1, 4),
        "rev": round(random.uniform(*rev_range), 1),
        "beat": round(random.uniform(3.2, 18.5), 1),
        "drop": round(random.uniform(2.5, 12.0), 1),
        "units": random.randint(50, 500),
        "product": random.choice(PRODUCTS_BY_SECTOR.get(informal_sector, PRODUCTS)),
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


def _ticker_trait_roll(ticker, salt):
    """Stable per-ticker pseudo-random value in [0, 1), independent of the
    global `random` stream's per-row position - used for company-level
    TRAITS (does this ticker pay a dividend at all? is it a richly-valued
    growth name? what's its typical share count) that should stay
    constant across every row for the same ticker, not re-rolled per row.
    Confirmed live this was a real problem when traits WERE re-rolled per
    row: ~60% of TSLA/PLTR/COIN rows fabricated a dividend yield up to
    3.2% for companies that pay zero, and the same ticker's market cap
    varied 3-10x row to row (MSFT $931B-$9.16T) purely from an
    independent per-row multiplier draw. `salt` differentiates
    independent traits for the same ticker so they don't accidentally
    correlate (e.g. dividend-payer status vs share-count baseline)."""
    digest = hashlib.sha256(f"{ticker}:{salt}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


# Fraction of tickers treated as genuine dividend payers (a STABLE
# per-ticker trait - see _ticker_trait_roll) - roughly matches the real
# mix of large/mid-cap payers vs non-payers (growth-tech-heavy universes
# skew lower than the broader market's ~70-80% payer rate). Applies ONLY
# to INVENTED_COMPANIES (no real answer to get wrong) - see
# REAL_DIVIDEND_PAYERS/REAL_NON_DIVIDEND_PAYERS below for why COMPANIES'
# real tickers use actual knowledge instead of a hash.
DIVIDEND_PAYER_FRACTION = 0.45
# Fraction of tickers treated as richly-valued growth/momentum names -
# confirmed live the flat pe_trailing range (6.0-28.0, capped at
# Technology's own SECTOR_MEDIAN_PE) meant the model never saw an
# expensive stock, a real gap for a value-investing model whose job
# includes recognizing them. Same real-tickers-use-real-knowledge caveat
# as DIVIDEND_PAYER_FRACTION - see REAL_HIGH_MULTIPLE_TICKERS/
# REAL_LOW_MULTIPLE_TICKERS below.
HIGH_MULTIPLE_FRACTION = 0.12

# Real dividend-payer status for the actual companies in COMPANIES (not
# INVENTED_COMPANIES, which have no real answer to get wrong) - confirmed
# live the hash-based _ticker_trait_roll approach, while correctly fixing
# the original row-to-row flip-flopping bug, produced a DIFFERENT, still
# real error: a hash has no relationship to actual corporate policy, so
# AMZN (famous for NEVER paying a dividend) landed in the "payer" bucket
# by chance, and TSLA/COIN similarly, while AAPL/GOOGL/META/MSFT (real,
# if sometimes modest, payers) landed in "non-payer". NVDA pays a real
# but negligible (~0.03%) dividend - treated as a non-payer here since
# even this generator's SMALLEST yield draw (0.1%) would still overstate
# it 3x+. INTC/BA are real companies that suspended their dividends
# (2024/2020 respectively) and hadn't confirmed resumption as of this
# generator's last real-world check - treated as non-payers rather than
# assuming an unconfirmed resumption.
REAL_DIVIDEND_PAYERS = {"AAPL", "MSFT", "GOOGL", "META", "JPM", "DIS", "CRM", "SBUX", "XOM", "PEP"}
REAL_NON_DIVIDEND_PAYERS = {
    "TSLA", "NVDA", "AMZN", "AMD", "NFLX", "INTC", "BA", "PYPL", "SHOP", "UBER", "COIN", "PLTR",
}

# Real, well-known high/low trading-multiple status for the actual
# companies in COMPANIES - same "real tickers deserve real knowledge, not
# a hash" reasoning as REAL_DIVIDEND_PAYERS above. Confirmed live the
# hash-based version marked JPM (a low-multiple bank, real P/E typically
# 10-14x) as a high-multiple name while leaving TSLA/NVDA/PLTR (genuinely,
# famously high-multiple growth names) in the normal range. Deliberately
# NOT an exhaustive classification of all 22 real tickers - only the ones
# unambiguous enough in either direction to be worth hand-verifying; the
# rest keep the existing hash-based/random behavior, which is a reasonable
# approximation for names whose real multiple is more genuinely variable.
REAL_HIGH_MULTIPLE_TICKERS = {"NVDA", "TSLA", "PLTR", "SHOP"}
REAL_LOW_MULTIPLE_TICKERS = {"JPM", "XOM", "PEP", "INTC"}


def make_fundamentals(company, fields):
    # Derives every other synthetic fundamental from a single random price
    # draw so a given example's numbers stay internally consistent (e.g.
    # price implies EPS implies intrinsic value implies over/undervalued -
    # they can't independently contradict each other the way unrelated
    # random draws could).
    ticker, _name, informal_sector, rev_range = company
    price_low, price_high = PRICE_RANGES[ticker]
    price = round(random.uniform(price_low, price_high), 2)

    if ticker in REAL_HIGH_MULTIPLE_TICKERS:
        is_high_multiple = True
    elif ticker in REAL_LOW_MULTIPLE_TICKERS:
        is_high_multiple = False
    else:
        is_high_multiple = _ticker_trait_roll(ticker, "high_pe") < HIGH_MULTIPLE_FRACTION
    if is_high_multiple:
        pe_trailing = round(random.uniform(30.0, 90.0), 1)
    else:
        pe_trailing = round(random.uniform(6.0, 28.0), 1)
    eps_trailing = round(price / pe_trailing, 2)
    if random.random() < LOSS_MAKING_PROB:
        eps_trailing = -abs(round(random.uniform(0.10, 3.0), 2))
    pe_forward = round(pe_trailing * random.uniform(0.82, 1.05), 1)

    if ticker in REAL_DIVIDEND_PAYERS:
        is_dividend_payer = True
    elif ticker in REAL_NON_DIVIDEND_PAYERS:
        is_dividend_payer = False
    else:
        is_dividend_payer = _ticker_trait_roll(ticker, "dividend") < DIVIDEND_PAYER_FRACTION
    div_yield = round(random.uniform(0.1, 3.2), 2) if is_dividend_payer else 0.0

    year_low = round(price * random.uniform(0.72, 0.93), 2)
    year_high = round(price * random.uniform(1.07, 1.38), 2)

    # Book value/share isn't consumed by the DCF math itself (build_scenarios/
    # cash_flow_basis_value above don't use it), but IS consumed by
    # _sustainable_growth_rate's ROE approximation and by
    # value_screen_metrics' ROE/Price-Book below - no longer a fully unused
    # figure, despite the pb_ratio name suggesting a Graham-era P/B
    # grounding that isn't actually how it's used anymore.
    pb_ratio = round(random.uniform(0.4, 4.0), 1)
    book_value_per_share = round(price / pb_ratio, 2)

    # Loosely correlated with profitability (eps_trailing's sign, already
    # rolled above) rather than an independent draw - a company already
    # rolled as loss-making shouldn't also roll a strong operating margin,
    # which would read as an internally-contradictory example (see this
    # function's own opening comment on why every field derives from the
    # same price/eps draw chain).
    if eps_trailing < 0:
        operating_margin = round(random.uniform(-0.15, 0.05), 4)
    else:
        operating_margin = round(random.uniform(0.03, 0.35), 4)

    rev_low, rev_high = rev_range
    # Loosely ties market cap to the company's revenue band (bigger revenue
    # -> more shares outstanding, roughly) without needing a second hand-
    # authored per-ticker range - precision doesn't matter here, only that
    # price * shares stays a plausible, internally consistent market cap.
    # The multiplier's BASELINE is now a stable per-ticker trait (see
    # _ticker_trait_roll) spanning the same overall 0.04-0.35 population
    # range, with only mild +/-10% row-to-row noise on top (real day-to-
    # day share-count changes are small) - confirmed live the old fully
    # independent per-row draw let the SAME ticker's market cap vary
    # 3-10x across rows (MSFT $931B-$9.16T, NVDA capped ~1/4 of reality).
    shares_baseline = 0.04 + _ticker_trait_roll(ticker, "shares") * 0.31
    shares_b = max(0.05, min(20.0, ((rev_low + rev_high) / 2) * shares_baseline * random.uniform(0.9, 1.1)))
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
        "operating_margin": operating_margin,
    }


def render_market_data(fnd):
    if random.random() < DATA_UNAVAILABLE_PROB:
        return "Data unavailable."

    # N/A fallbacks even though synthetic fields are otherwise always
    # populated - value_screen_metrics itself can still return None here
    # (e.g. PEG when growth_0y rolled <= 0, see make_fundamentals' ~15%
    # "opposite direction" branch) - same rendering convention as
    # financial-sentiment-api's fundamentals.py/generate_real_dataset.py.
    screen = value_screen_metrics(fnd)
    operating_margin_str = f"{screen['operating_margin'] * 100:.1f}%" if screen["operating_margin"] is not None else "N/A"
    roe_str = f"{screen['roe'] * 100:.1f}%" if screen["roe"] is not None else "N/A"
    price_to_book_str = f"{screen['price_to_book']:.1f}" if screen["price_to_book"] is not None else "N/A"
    price_to_sales_str = f"{screen['price_to_sales']:.1f}" if screen["price_to_sales"] is not None else "N/A"
    fcf_yield_str = f"{screen['fcf_yield'] * 100:.1f}%" if screen["fcf_yield"] is not None else "N/A"
    peg_ratio_str = f"{screen['peg_ratio']:.1f}" if screen["peg_ratio"] is not None else "N/A"
    # Sector name now renders even when no median exists for it - ported
    # from fundamentals.py's identically-structured block (P7: the model
    # has no other reliable in-prompt signal it's looking at a REIT
    # specifically).
    if screen["sector_median_pe"] is not None:
        sector_median_pe_str = f"{screen['sector_median_pe']:.1f} ({fnd['sector']})"
    elif fnd.get("sector"):
        sector_median_pe_str = f"N/A ({fnd['sector']})"
    else:
        sector_median_pe_str = "N/A"

    return (
        f"Price: ${fnd['price']:.2f} | Market Cap: {format_market_cap(fnd['market_cap'])}\n"
        f"P/E (trailing): {fnd['pe_trailing']:.1f} | P/E (forward): {fnd['pe_forward']:.1f}\n"
        f"EPS (trailing): ${fnd['eps_trailing']:.2f} | Dividend Yield: {fnd['div_yield']:.2f}%\n"
        f"52-Week Range: ${fnd['year_low']:.2f} - ${fnd['year_high']:.2f}\n"
        f"Operating Margin: {operating_margin_str} | ROE: {roe_str} | Price/Book: {price_to_book_str}\n"
        f"Price/Sales: {price_to_sales_str} | FCF Yield: {fcf_yield_str} | PEG: {peg_ratio_str}\n"
        f"Sector Median P/E: {sector_median_pe_str}"
    )


DIVIDEND_PAYOUT_BLEND_HALF_WIDTH = 0.05
# "fcf" ranked above "dividends" - ported from valuation.py after a
# confirmed live bug: cash_flow_basis_value("dividends", ...) succeeds
# for ANY company with a nonzero dividend_rate, no payout-ratio gate at
# all, so a company whose eps got screened out fell back to a token
# dividend instead of its real free cash flow. See that module's own
# comment.
FALLBACK_BASIS_ORDER = ["fcf", "eps", "dividends", "revenue"]


def _classify_alternate_basis(sector, free_cash_flow):
    """Ported from valuation.py's identically-named function - the basis a
    ticker would get if its payout_ratio fell just outside the dividends
    band."""
    if sector in ASSET_HEAVY_SECTORS and free_cash_flow is not None and free_cash_flow > 0:
        return "fcf"
    return "eps"


def _payout_blend_fraction(payout_ratio):
    """Ported from valuation.py's identically-named function."""
    if payout_ratio is None:
        return None
    low = DIVIDEND_PAYOUT_THRESHOLD - DIVIDEND_PAYOUT_BLEND_HALF_WIDTH
    high = DIVIDEND_PAYOUT_THRESHOLD + DIVIDEND_PAYOUT_BLEND_HALF_WIDTH
    if not (low <= payout_ratio <= high):
        return None
    return (payout_ratio - low) / (high - low)


# Retired (2026-08-19, two-stage pipeline redesign): BUY/SELL/MIXED used
# to draw a REAL, independent random DCF gap and then hand-check it
# against the news-driven label - acknowledging or downgrading to HOLD
# above a threshold if the two conflicted. fusion_rules.fuse() now does
# exactly this, deterministically, for every row regardless of category -
# there's no longer a silent-contradiction risk this needed to patch, so
# VALUATION_CONFLICT_ACKNOWLEDGMENT_THRESHOLD/_PHRASINGS/_DOWNGRADE_
# THRESHOLD/_DOWNGRADE_PHRASINGS are gone. make_example's
# _fusion_explanation (near the bottom of this file) replaces all four -
# see its own comment.

_VALUATION_GAP_RE = re.compile(r"(overvalued|undervalued) by [~>](\d+)%")


def _parse_valuation_gap(valuation_text):
    """Extracts (verdict, pct) from a rendered valuation_block string, or
    None when there's no real directional gap to parse ("Data
    unavailable.", "Not applicable (...)", "trading near fair value")."""
    match = _VALUATION_GAP_RE.search(valuation_text)
    if not match:
        return None
    return match.group(1), float(match.group(2))


def render_valuation(fnd):
    """Classifies the valuation basis and runs the scenario-DCF math ported
    above - same pipeline as valuation.py's valuation_block_for, adapted to
    this generator's already-in-hand fnd dict instead of a live yfinance
    fetch."""
    if random.random() < DATA_UNAVAILABLE_PROB:
        return "Data unavailable."
    basis = classify_valuation_basis(
        fnd["eps_trailing"], fnd["payout_ratio"], fnd["sector"], fnd["free_cash_flow"],
    )

    # Short-circuits the whole compute/blend/fallback pipeline below when
    # the eps basis's trailing EPS looks one-time-item-distorted - ported
    # from valuation.py's identically-structured check (see that module's
    # own comment on EARNINGS_SURPRISE_ONE_TIME_ITEM_THRESHOLD and why
    # this renders "Not applicable" rather than falling back to another
    # basis). Structural no-op here (fnd has no real "recent_eps_surprise"
    # field), kept only to stay byte-for-byte in step per the 4-way sync
    # rule.
    if basis == "eps":
        recent_eps_surprise = fnd.get("recent_eps_surprise")
        if recent_eps_surprise is not None and recent_eps_surprise > EARNINGS_SURPRISE_ONE_TIME_ITEM_THRESHOLD:
            return valuation_block(fnd["price"], None, "eps")

    def compute(b, include_dividend_pv):
        cf0_ = cash_flow_basis_value(b, fnd)
        scenarios_ = build_scenarios(fnd, b)
        dividend_rate_ = fnd.get("dividend_rate") if include_dividend_pv else None
        intrinsic_ = intrinsic_value(cf0_, b, scenarios_, dividend_rate_)
        return cf0_, scenarios_, intrinsic_

    # Payout-threshold cliff smoothing - ported from valuation.py's
    # identically-structured block. See that module's own comment.
    blend_t = None
    if basis != "revenue" and fnd.get("sector") not in REIT_SECTORS:
        blend_t = _payout_blend_fraction(fnd.get("payout_ratio"))

    if blend_t is not None:
        alt_basis = _classify_alternate_basis(fnd.get("sector"), fnd.get("free_cash_flow"))
        div_cf0, div_scenarios, div_iv = compute("dividends", False)
        alt_cf0, alt_scenarios, alt_iv = compute(alt_basis, False)
        if div_iv is not None and alt_iv is not None:
            intrinsic = blend_t * div_iv + (1 - blend_t) * alt_iv
            basis, cf0, scenarios = (
                ("dividends", div_cf0, div_scenarios) if blend_t >= 0.5 else (alt_basis, alt_cf0, alt_scenarios)
            )
        elif div_iv is not None:
            basis, cf0, scenarios, intrinsic = "dividends", div_cf0, div_scenarios, div_iv
        else:
            basis, cf0, scenarios, intrinsic = alt_basis, alt_cf0, alt_scenarios, alt_iv
    else:
        cf0, scenarios, intrinsic = compute(basis, True)

    if intrinsic is None:
        # Basis-fallback chain - ported from valuation.py's identically-
        # structured block (a curated/REIT override with no usable cf0
        # retries other bases instead of a permanent "Not applicable").
        for fallback_basis in FALLBACK_BASIS_ORDER:
            if fallback_basis == basis:
                continue
            fb_cf0, fb_scenarios, fb_intrinsic = compute(fallback_basis, True)
            if fb_intrinsic is not None:
                basis, cf0, scenarios, intrinsic = fallback_basis, fb_cf0, fb_scenarios, fb_intrinsic
                break

    return valuation_block(fnd["price"], intrinsic, basis)


# valuation_block_with_gap (constructed a Valuation block with a
# controlled, non-DCF-derived gap - used only by the now-retired
# VALUATION_SIGNAL_SCENARIOS/HOLD's pinned-gap path) removed along with
# the VALUATION_CONFLICT_* machinery above - every category draws a real,
# independent render_valuation(fnd) gap now (see make_example).


def render_earnings(fnd, direction):
    if random.random() < DATA_UNAVAILABLE_PROB:
        return "Data unavailable."
    # Coherent with the resolved direction: a bullish example's earnings
    # line shows a beat, a bearish one a miss, a neutral one in-line -
    # matching the scenario's news headlines instead of being an
    # independently-rolled, potentially contradictory number.
    surprise_pct = round(random.uniform(2.0, 12.0), 1)
    actual_eps = fnd["last_q_eps"]
    if direction == "BUY":
        est_eps = round(actual_eps / (1 + surprise_pct / 100), 2)
        surprise_note = f"beat est. ${est_eps:.2f}"
        yoy = abs(fnd["yoy_growth"])
    elif direction == "SELL":
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


_NAIVE_LEAN = {
    "good": "BUY", "overreaction_down": "BUY",
    "bad": "SELL", "overreaction_up": "SELL",
    "neutral": None,
}

# Replace the old VALUATION_CONFLICT_*/valuation_block_with_gap machinery
# (see that code's removal note above): rather than hand-checking whether
# a randomly-drawn gap "contradicts" a category-fixed label past some
# threshold, every row now gets one short, generic clause naming
# whichever of three relationships actually holds between the news
# reaction and fuse()'s valuation bucket - "reinforcing" (valuation backs
# up the news-implied lean), "capping" (valuation caps a clear reaction
# down to a weaker call), or "driving" (news is neutral/routine and
# valuation alone is doing the work) - matching fusion_rules.FUSION_
# TABLE's own row-by-row rationale. Multiple phrasings per pool for the
# same memorization-avoidance reason as QUALITY_*_PHRASINGS/VALUATION_
# CONFLICT_PHRASINGS before it.
_REINFORCING_CLAUSES = [
    "{ticker} also screens as {bucket_word} on a DCF basis (~{pct:.0f}%), reinforcing that read.",
    "That lines up with the DCF, which reads {bucket_word} by roughly {pct:.0f}% here too.",
    "The valuation picture agrees - {bucket_word} by about {pct:.0f}% on a DCF basis, corroborating the read.",
]
_CAPPING_CLAUSES = [
    "That said, {ticker} already screens as {bucket_word} by roughly {pct:.0f}% on a DCF basis - a real signal, but not one with room left to act on, so this lands at {recommendation} rather than a stronger call.",
    "Still, the DCF reads {bucket_word} by about {pct:.0f}% here - real evidence, but at this price there's no headroom left, which caps this at {recommendation} instead of a full directional call.",
    "Weighed against a DCF that's {bucket_word} by ~{pct:.0f}%, though: the signal is real, but it doesn't by itself create room to act on - {recommendation}, not a stronger call.",
]
_DRIVING_CLAUSES = [
    "No strong catalyst either way here, but {ticker} is trading at a level that screens {bucket_word} versus its estimated intrinsic value (~{pct:.0f}%) - a real, if imperfect, signal on its own.",
    "Nothing in the news itself moves the needle, but the DCF reads {bucket_word} by about {pct:.0f}% - worth weighing even without a fresh catalyst.",
    "The news is routine, but {ticker}'s valuation - {bucket_word} by roughly {pct:.0f}% on a DCF basis - is doing the real work here.",
]
_NEAR_FAIR_CLAUSES = [
    "{ticker} is trading close to its estimated intrinsic value here, so valuation isn't a strong argument either way - that leaves {recommendation} as the call.",
    "No real valuation edge either way for {ticker} right now - it's trading near fair value on a DCF basis, so this stays at {recommendation}.",
]
_NO_DATA_CLAUSES = [
    "No valuation estimate is available here, which tempers this to {recommendation}.",
    "With no usable valuation read available, there's nothing to confirm extra headroom, so this stays at {recommendation}.",
]


def _fusion_explanation(ticker, reaction, fusion_result, gap_pct):
    """Appended to every row's base reasoning (which explains the NEWS
    itself - still accurate regardless of headroom) to make the actual
    recommendation - always fusion_rules.fuse()'s deterministic output,
    never something invented here - explicit."""
    bucket = fusion_result.valuation_bucket
    if bucket == "no_data":
        return random.choice(_NO_DATA_CLAUSES).format(recommendation=fusion_result.recommendation)
    if bucket == "near_fair":
        # Regression fix (post-migration eval): unlike every other clause
        # pool, this one used to never mention {recommendation} at all -
        # near_fair rows' reasoning only described valuation as neutral,
        # giving the model zero textual signal for fuse()'s actual call
        # (often a dampened HOLD/SELL against the news reaction's own
        # lean). Confirmed as the likely cause of near_fair's collapsed
        # eval accuracy (e.g. neutral/near_fair 0/12, good/near_fair
        # 2/14) - every other bucket's clauses always name the call.
        return random.choice(_NEAR_FAIR_CLAUSES).format(ticker=ticker, recommendation=fusion_result.recommendation)

    bucket_word = "undervalued" if bucket == "undervalued" else "overvalued"
    naive_lean = _NAIVE_LEAN[reaction]
    if naive_lean is None:
        clause_pool = _DRIVING_CLAUSES
    elif fusion_result.recommendation == naive_lean:
        clause_pool = _REINFORCING_CLAUSES
    else:
        clause_pool = _CAPPING_CLAUSES
    return random.choice(clause_pool).format(
        ticker=ticker, bucket_word=bucket_word, pct=abs(gap_pct), recommendation=fusion_result.recommendation,
    )


def price_context_block(ticker, move_pct):
    """Canonical phrasing shared with generate_real_dataset.py's
    identically-named function and financial-sentiment-api's planned
    per-headline date-specific price lookup (ported, not imported - this
    file's own long-standing "ported, not imported" convention for the
    DCF math above applies here too). Single-day phrasing since 2026-08-20
    (see generate_real_dataset.py's module docstring history item 12) -
    was "over the last 3 trading days.", now the single publish-day close-
    to-close move, matching what's actually reproducible at inference time
    for a brand-new headline."""
    return f"{ticker} moved {move_pct:+.1f}% on the day this was published."


def _fabricate_move_pct(reaction):
    """Fabricated single-day price move consistent with `reaction`'s
    class - the actual training signal for overreaction_down/up is the
    (modest headline, outsized move) MISMATCH, so those two ranges are
    deliberately much larger than good/bad's, and clearly separated from
    every other class so a given row is never ambiguous about which class
    it's demonstrating. Rescaled 2026-08-20 for the single-day move
    redefinition (generate_real_dataset.py history item 12) - single-day
    return magnitudes are structurally smaller than the old 3-day
    cumulative window's, per calibrate_reaction_thresholds.py's re-run
    (717 real samples: p90 +3.2%, p95 +4.8%, stdev 2.4%). Ranges keep the
    same ordering/separation shape as the old 3-day ranges (3.0-8.0 /
    -8.0--3.0 / -2.5-2.5 / -15.0--8.0 / 8.0-15.0), scaled down to sit
    around the new REACTION_GOOD_BAD_THRESHOLD=1%/REACTION_OVERREACTION_
    MOVE_THRESHOLD=3% boundaries instead of the old 2%/5%."""
    if reaction == "good":
        return round(random.uniform(1.2, 2.8), 1)
    if reaction == "bad":
        return round(random.uniform(-2.8, -1.2), 1)
    if reaction == "neutral":
        return round(random.uniform(-0.8, 0.8), 1)
    if reaction == "overreaction_down":
        return round(random.uniform(-9.0, -3.2), 1)
    if reaction == "overreaction_up":
        return round(random.uniform(3.2, 9.0), 1)
    raise ValueError(f"Unknown reaction: {reaction!r}")


def make_example(company, reaction):
    """Returns a list of TWO rows sharing the same headline/price-context/
    split: a Task A (task="reaction") row - the news_reaction classifier's
    target, no resolved direction anywhere in it - and a Task B (task=
    "analysis") row whose `recommendation` is ALWAYS fusion_rules.fuse
    (reaction, gap_pct)'s output, never hand-picked. See REACTION_WEIGHTS'
    own comment for the redesign this replaces."""
    ticker = company[0]

    if reaction == "neutral":
        idx = random.randrange(len(HOLD_SCENARIOS))
        headline_templates, reasoning_variants = HOLD_SCENARIOS[idx]
        reasoning_template = random.choice(reasoning_variants)
        is_holdout_template = idx in HOLD_HOLDOUT_IDX
    elif reaction == "overreaction_down":
        idx = random.randrange(len(OVERREACTION_DOWN_SCENARIOS))
        headline_templates, reasoning_template = OVERREACTION_DOWN_SCENARIOS[idx]
        is_holdout_template = idx in OVERREACTION_DOWN_HOLDOUT_IDX
    elif reaction == "overreaction_up":
        idx = random.randrange(len(OVERREACTION_UP_SCENARIOS))
        headline_templates, reasoning_template = OVERREACTION_UP_SCENARIOS[idx]
        is_holdout_template = idx in OVERREACTION_UP_HOLDOUT_IDX
    else:  # "good" / "bad"
        scenarios = {"good": BUY_SCENARIOS, "bad": SELL_SCENARIOS}[reaction]
        holdout_idx = {"good": BUY_HOLDOUT_IDX, "bad": SELL_HOLDOUT_IDX}[reaction]
        idx = random.randrange(len(scenarios))
        headline_templates, reasoning_template, _tier = scenarios[idx]
        is_holdout_template = idx in holdout_idx

        # Sector-macro reinforcement (see SECTOR_MACRO_HEADLINES's own
        # comment) - swaps in a company-sector-linked headline for extra
        # "good"/"bad" diversity. Was VALUATION_SIGNAL-only before this
        # file's two-stage redesign; feeds news_reaction directly now.
        sector_key = company[2]
        if sector_key in SECTOR_MACRO_HEADLINES and random.random() < SECTOR_REINFORCED_PROB:
            bullish_headline, bearish_headline, linkage = SECTOR_MACRO_HEADLINES[sector_key]
            headline_templates = [bullish_headline if reaction == "good" else bearish_headline]
            tone = "positive" if reaction == "good" else "negative"
            reasoning_template = SECTOR_LINKAGE_REASONING.format(ticker="{ticker}", linkage=linkage, tone=tone)
            is_holdout_template = False

    fields = make_fields(company)
    filled_headlines = [t.format(**fields) for t in headline_templates]
    reasoning = reasoning_template.format(**fields)

    move_pct = _fabricate_move_pct(reaction)
    price_context = price_context_block(ticker, move_pct)
    news_block = build_news_block(filled_headlines)
    user_query, qtype = build_user_query(ticker)

    fnd = make_fundamentals(company, fields)
    market_data_text = render_market_data(fnd)
    # Gated on market_data_text, not just `fnd` (which is always fully
    # populated) - render_market_data can independently roll "Data
    # unavailable." (DATA_UNAVAILABLE_PROB), and without this check the
    # reasoning could cite operating margin/ROE/PEG numbers the model is
    # simultaneously being shown as unavailable in market_data - a real,
    # confirmed-live contradiction (found via a full generate() run before
    # this fix: 78/2000 rows had exactly this mismatch).
    if market_data_text != "Data unavailable." and random.random() < VALUE_SCREEN_COMMENTARY_PROB_OTHER_CATEGORIES:
        commentary = describe_value_screen(fnd)
        if commentary:
            reasoning = f"{reasoning} {commentary}"

    # Every category now draws a REAL, independent random DCF gap - see
    # this file's removal note above the retired VALUATION_CONFLICT_*
    # machinery. fuse() is the only place a recommendation is decided.
    valuation_text = render_valuation(fnd)
    parsed_gap = _parse_valuation_gap(valuation_text)
    gap_pct = None
    if parsed_gap is not None:
        gap_verdict, gap_magnitude = parsed_gap
        gap_pct = gap_magnitude if gap_verdict == "overvalued" else -gap_magnitude

    fusion_result = fusion_rules.fuse(reaction, gap_pct)
    reasoning = f"{reasoning} {_fusion_explanation(ticker, reaction, fusion_result, gap_pct)}"

    # Earnings is rendered off the NEWS-implied lean (good=beat, bad=miss,
    # neutral/overreaction_*=in-line - an overreaction is BY DEFINITION
    # not matched by a proportional fundamentals change), never off
    # fuse()'s recommendation - same "Earnings must show the real
    # catalyst, not the resolved call" principle the old earnings_
    # direction/original_direction split enforced, just derived from
    # `reaction` now instead of a per-tier hand-picked override.
    earnings_lean = {"good": "BUY", "bad": "SELL"}.get(reaction, "HOLD")
    earnings_text = render_earnings(fnd, earnings_lean)

    answer = ANSWER_TEMPLATES[qtype][fusion_result.recommendation].format(ticker=ticker)

    is_holdout_ticker = ticker in VAL_HOLDOUT_TICKERS
    split = "val" if (is_holdout_ticker or is_holdout_template) else "train"

    task_a_row = {
        "task": "reaction",
        "ticker": ticker,
        "user_query": "",
        "price_context": price_context,
        "market_data": "",
        "valuation": "",
        "earnings": "",
        "news": news_block,
        "news_reaction": reaction,
        "recommendation": "",
        "valuation_bucket": "",
        "output": json.dumps({"news_reaction": reaction}),
        "_split": split,  # stripped before writing - see main()
    }
    task_b_row = {
        "task": "analysis",
        "ticker": ticker,
        "user_query": user_query,
        "price_context": price_context,
        "market_data": market_data_text,
        "valuation": valuation_text,
        "earnings": earnings_text,
        "news": news_block,
        "news_reaction": reaction,
        # Ground truth for eval's recommendation-accuracy check (fuse()'s
        # own output, not shown to the model as input anymore - see
        # docs/task-b-learned-recommendation-plan.md). valuation_bucket
        # lets eval split accuracy by (news_reaction, valuation_bucket)
        # cell without recomputing fuse().
        "recommendation": fusion_result.recommendation,
        "valuation_bucket": fusion_result.valuation_bucket,
        "output": json.dumps(
            {"recommendation": fusion_result.recommendation, "reasoning": reasoning, "answer": answer}, indent=2,
        ),
        "_split": split,
    }
    return [task_a_row, task_b_row]


def generate(n=NUM_EXAMPLES):
    examples = []
    reactions = list(REACTION_WEIGHTS.keys())
    weights = list(REACTION_WEIGHTS.values())
    for _ in range(n):
        reaction = random.choices(reactions, weights=weights)[0]
        company = random.choice(ALL_COMPANIES)
        examples.extend(make_example(company, reaction))
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
