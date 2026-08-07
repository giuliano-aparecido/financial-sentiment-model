"""
Financial sentiment training dataset generator - REAL DATA variant

Companion to generate_synthetic_dataset.py (fully synthetic, offline,
deterministic). This version pulls REAL historical news headlines from
Google News RSS and REAL subsequent price movement from yfinance for each
ticker, and derives the BULLISH/BEARISH/NEUTRAL label from what the stock
actually did in the days after the headline - not from a hand-authored,
causally-verified judgment the way the synthetic dataset's labels are
written.

This is a proxy-labeling approach, and it's a genuinely different kind of
signal than the synthetic dataset, not just a "more realistic" version of
it: any single label here is noisy - a lot of a stock's price movement in
a given window has nothing to do with the specific headline it's paired
with. The bet is that across enough examples, real language patterns that
do correlate with subsequent direction are still there for the model to
find, even though no single row is a verified causal claim the way the
synthetic dataset's rows are. Treat accuracy on this dataset's own
validation split with that in mind - it's measuring "does the model find
the same correlations in the label noise," not "is the model's causal
reasoning correct."

History, most recent first - after the first trained model's eval came
back at only 25.0% direction accuracy on real val (below the 33% random-
guess baseline for a 3-way choice, and well below the 86.0% the same model
got on synthetic val):

0. `reasoning` is now written by Gemini (see generate_grounded_reasoning),
   grounded in the actual headline, instead of the fixed template this
   file used before ("Over the N trading days following this news, TICKER
   moved X.X%, which resolves as DIRECTION..."). That template never
   referenced headline content at all, so the real portion of training
   taught the model to recall a memorized per-ticker (%, direction) pair
   instead of reading the news - confirmed live: a TPU-trained model's
   misclassified real rows showed the identical "+3.4% -> BULLISH" text
   for the same ticker across three unrelated headlines. direction and
   confidence are still derived purely from the price-move proxy below,
   unchanged - only the reasoning TEXT is regenerated to actually discuss
   the headline. The prompt explicitly tells Gemini the direction was
   already decided from price data it doesn't have access to, and to
   name the headline as a weak/indirect signal rather than invent a
   strong causal story when the connection isn't obvious - the same
   "don't fabricate a hand-verified causal read" concern the old
   template's disclaimer was protecting, just solved by writing
   headline-grounded text instead of refusing to engage with the
   headline at all.

   Follow-up, after the first full-scale (40-ticker) eval on this fix:
   real accuracy jumped to 37% (from 25%) and the class-collapse bug was
   confirmed gone, but a new, more specific pattern showed up - the
   model correctly reading a headline's plain tone in its own reasoning,
   then flipping to the opposite direction anyway via a fabricated
   "contrarian" story ("already priced in," "market sees through this")
   to force-fit a mismatched label. Root cause: the prompt's escape
   hatch only covered headlines that don't obviously SUPPORT the given
   direction (ambiguous case) - it had no guidance for headlines that
   actively CONTRADICT it (which happens constantly, since the label is
   real-price-derived and headlines don't reliably predict short-term
   moves). Added an explicit instruction for that case: acknowledge the
   mismatch honestly and use a low confidence score, instead of
   inventing a plausible-sounding but unverifiable contrarian narrative.
   Deliberately NOT fixed by feeding in what actually happened after the
   headline (real follow-up news, etc.) - that would leak information
   the model can never have at real inference time (predicting forward
   from headlines available now), training a skill that's unusable in
   production rather than a general calibration habit that is.
1. TICKERS expanded 20 -> 40. Real data volume is capped by how much
   Google News actually returns per ticker/window, so growing the
   TRAINING supplement without duplicating rows (see rebalance_by_direction
   below - duplication was deliberately ruled out earlier) requires a
   bigger raw pool to draw from, not just a higher per-ticker cap.
2. LOOKBACK_WEEKS 8 -> 18 and MAX_HEADLINES_PER_TICKER 30 -> 50, for the
   same reason - more historical windows scanned per ticker, more raw
   headlines available to become the post-rebalance train set.
3. BULLISH_THRESHOLD/BEARISH_THRESHOLD are now more visibly a label-quality
   knob, with OUTPUT_TRAIN_FILE/OUTPUT_VAL_FILE broken out as their own
   constants specifically so a stricter-threshold comparison run (e.g.
   +/-0.03 or +/-0.04 instead of the default +/-0.02) can write to
   different filenames instead of overwriting the default run. See the
   "Threshold experiment" note below the constants.
4. v4: added `market_data`, `valuation`, `earnings` fields (fetched from
   yfinance's quarterly statements/earnings_dates, AS OF the headline's own
   publish date - not today's figures) and `answer` (Gemini-written, from
   the same call as `reasoning`) - see fetch_ticker_fundamentals_history's
   docstring for exactly which pieces are genuinely as-of-date vs a
   documented current-snapshot approximation (shares outstanding, forward
   P/E, dividend yield, 52-week range). direction/confidence remain purely
   price-move-derived, unchanged. Mirrors generate_synthetic_dataset.py's
   v4 addition field-for-field so both datasets stay concatenable.

Earlier history: this script originally used yfinance's Ticker.news for
headlines, which only returns the current "latest ~10" items with no
historical/date-range support - most of what it returned was too recent to
have a forward price window yet. Switched to Google News RSS, queried for
specific past date windows using Google's undocumented `after:`/`before:`
search operators, which lets us deliberately query already-old windows so
forward price data already exists for nearly everything returned. yfinance
still supplies price history for the label - nothing about that half
changed.

`after:`/`before:` are NOT officially documented by Google and could
change or get blocked without notice - same fragility production's own
news fetch already accepts. Day-level granularity only (no time-of-day),
and any single query is capped at ~100 results, which is why this scans
multiple narrow weekly windows per ticker rather than one big range.

The train split is rebalanced to equal BULLISH/BEARISH/NEUTRAL counts by
undersampling (see rebalance_by_direction) before being written - real
market data over any specific historical window is rarely naturally
balanced. The val split is deliberately left at its natural/unbalanced
distribution - eval numbers should reflect real-world performance, not a
distribution forced to look nicer than reality. (This natural imbalance -
the val set skews BEARISH-heavy - is itself part of why a model with a
"default to NEUTRAL when unsure" habit scored so poorly on it; see
docs/training-results-analysis.md.)

Output schema, ### Input: field structure, and user_query phrasing all
match generate_synthetic_dataset.py exactly (down to reusing its
NOISE_HEADLINES, USER_QUESTION_TEMPLATES, QUESTION_TYPES, and
ANSWER_TEMPLATES verbatim), so both datasets' JSONL rows are interchangeable
and can be concatenated/mixed for training:

    data_files={"train": ["dataset_train.jsonl", "dataset_train_real.jsonl"], ...}

Requirements: `pip install yfinance httpx feedparser google-genai pandas`
(pandas is also a transitive yfinance dependency, so usually already
present), network access, and a Gemini API key (GEMINI_API_KEY) - see
generate_grounded_reasoning for where that's read from and why the model
choice is gemini-3.5-flash-lite specifically.
Unlike the synthetic generator, this is NOT reproducible/deterministic -
querying the same historical window twice can return different results as
Google's index changes, and Gemini's reasoning text varies run to run even
for the same headline (direction/confidence do not - those stay purely
proxy-derived). Both yfinance and Google News RSS are unofficial/
undocumented access - this script fails soft (skips and logs a warning)
rather than crashing on a per-ticker or per-window fetch failure; the
Gemini call fails soft too (falls back to the old template text for that
one row) rather than aborting a multi-hundred-row unattended run over a
transient API error.

Runtime note: with TICKERS x LOOKBACK_WEEKS now 40 x 18 = 720 weekly
windows, and NEWS_REQUEST_DELAY_SECONDS=1.0 between each, the news-fetch
phase alone is >=12 minutes of politeness delay before counting actual
request latency or the price-history calls on top - expect a notably
longer run than earlier, smaller configurations. On a free-tier
GEMINI_API_KEY, GEMINI_REQUEST_DELAY_SECONDS (4.5s, sized for the free
tier's 15-requests/minute cap) would add roughly another 4.5s per kept
headline on top of that, since a Gemini call happens once per row now -
with billing enabled and a paid tier's much higher per-minute limit,
GEMINI_REQUEST_DELAY_SECONDS drops to 0.1s and that cost mostly
disappears (dominated instead by actual Gemini/yfinance/RSS network
latency, not artificial pacing). Either way, this is expected, not a
hang; the per-ticker incremental writes and try/except (see
generate_and_write) mean a slow run is safe to leave unattended.

Output: two JSONL files, named by OUTPUT_TRAIN_FILE/OUTPUT_VAL_FILE below
(default: dataset_train_real.jsonl and dataset_val_real.jsonl).
"""

import datetime
import json
import os
import random
import re
import time
import urllib.parse

try:
    import feedparser
    import httpx
    import pandas as pd
    import yfinance as yf
    from google import genai
except ImportError as e:
    raise SystemExit(
        f"This script needs a package that isn't installed ({e.name}). "
        "Run: pip install yfinance httpx feedparser google-genai pandas"
    )

# get_secret() works on both Colab (Secrets, key icon in the left sidebar)
# and Kaggle (Add-ons -> Secrets) - same pattern the run/ serving scripts
# use for HF_TOKEN/NGROK_AUTH_TOKEN, reused here for GEMINI_API_KEY. Get a
# free key at https://aistudio.google.com/apikey.
#
# Checking whether `google.colab` IMPORTS is not a reliable way to detect
# Colab vs Kaggle - confirmed live: some Kaggle base images ship a
# google-colab package too, so the import succeeds there, and the
# ModuleNotFoundError this used to branch on never fires. The actual
# Colab RPC then just hangs and times out ("Secrets can only be fetched
# when running from the Colab UI") instead of falling through to
# kaggle_secrets. KAGGLE_KERNEL_RUN_TYPE is set by Kaggle's own runtime
# on every notebook, so check that directly instead of inferring the
# platform from import success.
def get_secret(name):
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    from google.colab import userdata
    return userdata.get(name)

# genai.Client() with no api_key reads GEMINI_API_KEY from the environment,
# but Colab/Kaggle secrets aren't environment variables - passed explicitly.
_gemini_client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))

# flash-lite: this is a short, repetitive, low-complexity writing task
# (headline + ticker + an already-decided direction -> 2-3 sentences) -
# the cheapest current Gemini tier fits it, not a reason to reach for a
# larger model. $0.30/$2.50 per 1M input/output tokens as of the pricing
# checked when this was written (gemini-2.5-flash-lite, the even cheaper
# tier at $0.10/$0.40, returns a 404 - "no longer available to new users" -
# confirmed against a real key) - re-verify at
# https://ai.google.dev/gemini-api/docs/pricing if it's been a while, and
# if this model name 404s the same way, that's Google retiring another
# generation, not a bug here.
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Confirmed live: the FREE TIER caps gemini-3.5-flash-lite at 15 requests
# per minute (generativelanguage.googleapis.com/generate_content_free_tier_requests) -
# that's what forced a 4.5s (60/15, plus margin) delay here originally.
# With GEMINI_API_KEY billing enabled, Tier 1 raised this to ~4000
# requests/minute (confirmed live at https://aistudio.google.com/rate-limit) -
# our whole run needs at most ~2000 calls total, so at that rate the
# per-minute cap is no longer the binding constraint. 0.1s is just a floor
# against hammering the API in a tight loop, not a real rate-limiting
# delay - the yfinance/RSS calls and actual Gemini network latency in the
# rest of the per-headline loop already pace things well under 4000/min
# on their own. If this ever moves to a free-tier key again, revert to
# 4.5 (or whatever 60/{current RPM} + margin works out to - re-check
# https://ai.google.dev/gemini-api/docs/rate-limits, don't assume 15 is
# still current).
GEMINI_REQUEST_DELAY_SECONDS = 0.1
GEMINI_MAX_RETRIES = 2

# Set by generate_grounded_reasoning once it sees a
# RequestsPerDayPerProjectPerModel quota error - see that function's
# docstring. Global, not per-ticker, since the daily cap is per API key,
# not per anything this script loops over.
_gemini_daily_quota_exhausted = False

random.seed(42)

# (ticker, company name) - name improves Google News query precision over
# a bare ticker symbol alone (e.g. "DIS" is ambiguous, "Walt Disney" isn't).
# Real tickers only - the synthetic generator's invented ones (ZVEX, QRNL,
# etc.) obviously have no real news or price history to query. All large,
# liquid, heavily-covered names so Google News RSS has enough to return per
# window.
TICKERS = [
    ("AAPL", "Apple"), ("TSLA", "Tesla"), ("NVDA", "Nvidia"), ("AMZN", "Amazon"),
    ("MSFT", "Microsoft"), ("GOOGL", "Alphabet"), ("META", "Meta Platforms"), ("AMD", "AMD"),
    ("JPM", "JPMorgan Chase"), ("DIS", "Walt Disney"), ("NFLX", "Netflix"), ("INTC", "Intel"),
    ("CRM", "Salesforce"), ("BA", "Boeing"), ("PYPL", "PayPal"), ("SHOP", "Shopify"),
    ("UBER", "Uber"), ("SBUX", "Starbucks"), ("COIN", "Coinbase"), ("PLTR", "Palantir"),
    ("GS", "Goldman Sachs"), ("V", "Visa"), ("MA", "Mastercard"), ("WMT", "Walmart"),
    ("HD", "Home Depot"), ("KO", "Coca-Cola"), ("PEP", "PepsiCo"), ("MCD", "McDonald's"),
    ("NKE", "Nike"), ("VZ", "Verizon"), ("CSCO", "Cisco"), ("ORCL", "Oracle"),
    ("IBM", "IBM"), ("QCOM", "Qualcomm"), ("ADBE", "Adobe"), ("NOW", "ServiceNow"),
    ("ABNB", "Airbnb"), ("F", "Ford"), ("GM", "General Motors"), ("XOM", "Exxon Mobil"),
]

# Smaller holdout than the synthetic generator's - real data volume per
# ticker is naturally lower than a generated quota, so holding out too
# many tickers leaves too little to actually train on.
VAL_HOLDOUT_TICKERS = {"META", "BA"}

FORWARD_WINDOW_TRADING_DAYS = 3   # how many trading days after the headline
                                   # to measure the price move over

# Threshold experiment: this is a genuine label-quality knob. The default
# (+/-2%) is loose enough that ambiguous, low-conviction moves get labeled
# with full confidence. Worth generating a second dataset variant at a
# stricter +/-3% or +/-4% and comparing eval results - fewer real examples,
# but each one a cleaner signal. To run that comparison without overwriting
# the default output:
#
#   BULLISH_THRESHOLD, BEARISH_THRESHOLD = 0.03, -0.03
#   OUTPUT_TRAIN_FILE = "dataset_train_real_strict.jsonl"
#   OUTPUT_VAL_FILE = "dataset_val_real_strict.jsonl"
#
# then point the training script's data_files at whichever variant (or
# both) you want to compare.
BULLISH_THRESHOLD = 0.02          # forward return >= +2% -> BULLISH
BEARISH_THRESHOLD = -0.02         # forward return <= -2% -> BEARISH
                                   # (between the two -> NEUTRAL)
OUTPUT_TRAIN_FILE = "dataset_train_real.jsonl"
OUTPUT_VAL_FILE = "dataset_val_real.jsonl"

# How far back, and how close to "now", to search. The gap between
# SAFETY_BUFFER_DAYS and today guarantees every queried window already has
# a complete forward price window by the time we look it up.
LOOKBACK_WEEKS = 18
SAFETY_BUFFER_DAYS = 14
MAX_HEADLINES_PER_TICKER = 50

NEWS_REQUEST_DELAY_SECONDS = 1.0   # be polite to Google's unofficial endpoint
PRICE_REQUEST_DELAY_SECONDS = 0.3  # be polite to yfinance between calls

PUBLISHER_FALLBACK = "Google News"

# Reused verbatim from generate_synthetic_dataset.py so both datasets' news
# blocks have the same shape - real feeds do mix in unrelated market
# headlines too, same as the synthetic version simulates.
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
NOISE_PUBLISHERS = ["Reuters", "Bloomberg", "MarketWatch", "CNBC"]

# Reused verbatim from generate_synthetic_dataset.py.
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

# Reused verbatim from generate_synthetic_dataset.py - index-aligned with
# USER_QUESTION_TEMPLATES above, and used as the ANSWER_TEMPLATES fallback
# below when Gemini is unavailable (quota exhausted, call failed).
QUESTION_TYPES = [
    "direction", "buy", "outlook", "worry", "impact",
    "sell", "sentiment", "earnings", "read", "none",
]

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


def build_user_query(ticker):
    idx = random.randrange(len(USER_QUESTION_TEMPLATES))
    template = USER_QUESTION_TEMPLATES[idx]
    query = template.format(ticker=ticker) if template else ""
    return query, QUESTION_TYPES[idx]


def format_market_cap(value):
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    return f"${value / 1e9:.1f}B"


def _format_date(dt):
    # Matches app/services/news.py's entry.get("published", "")[:16] shape
    # (e.g. "Tue, 05 Aug 2026"), same as the synthetic generator.
    return f"{dt.strftime('%a')}, {dt.day:02d} {dt.strftime('%b')} {dt.year}"


def weekly_windows():
    """Yields (after_date, before_date) date objects, oldest first, covering
    LOOKBACK_WEEKS weeks ending SAFETY_BUFFER_DAYS before today."""
    today = datetime.date.today()
    window_end = today - datetime.timedelta(days=SAFETY_BUFFER_DAYS)
    for i in range(LOOKBACK_WEEKS, 0, -1):
        after = window_end - datetime.timedelta(weeks=i)
        before = after + datetime.timedelta(weeks=1)
        yield after, before


def fetch_headlines_for_window(ticker, name, after_date, before_date):
    """Queries Google News RSS for `name`/`ticker` mentions published
    between after_date and before_date (inclusive/exclusive per Google's
    own semantics - not precisely documented). Returns a list of
    (title, publisher, published_at) tuples. Fails soft: returns [] and
    logs a warning on any fetch/parse error, same pattern as this
    project's production app/services/news.py.
    """
    query = f'"{name}" OR {ticker} stock after:{after_date.isoformat()} before:{before_date.isoformat()}'
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    try:
        response = httpx.get(rss_url, timeout=10.0)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except Exception as e:
        print(f"    Warning: Google News RSS fetch failed for {ticker} {after_date}..{before_date}: {e}", flush=True)
        return []

    results = []
    for entry in feed.entries:
        title = entry.get("title", "")
        if not title:
            continue
        publisher = PUBLISHER_FALLBACK
        # Google News RSS titles are conventionally "Headline - Publisher" -
        # split it out if present so it matches the synthetic dataset's
        # separate publisher field instead of duplicating it into the title.
        if " - " in title:
            title, _, publisher = title.rpartition(" - ")

        published_at = None
        if entry.get("published_parsed"):
            published_at = datetime.datetime(*entry.published_parsed[:6])

        results.append((title, publisher, published_at))

    return results


def label_from_forward_return(ticker_obj, published_at):
    """Returns (direction, pct_change, actual_window_days, skip_reason).
    skip_reason is None on success, otherwise "no_date",
    "history_fetch_failed", or "insufficient_history".

    Because callers only pass in headlines from deliberately-old query
    windows (see weekly_windows/SAFETY_BUFFER_DAYS), the full
    FORWARD_WINDOW_TRADING_DAYS should be available almost every time. The
    adaptive/floor-of-1-day behavior is kept as a safety net for edge cases
    (market holidays, sparse data), not as the primary mechanism.
    """
    if published_at is None:
        return None, None, None, "no_date"

    start_date = published_at.date()
    end_date = start_date + datetime.timedelta(days=FORWARD_WINDOW_TRADING_DAYS + 4)  # buffer for weekends/holidays
    try:
        hist = ticker_obj.history(start=start_date, end=end_date)
    except Exception as e:
        print(f"    Warning: price history fetch failed: {e}", flush=True)
        return None, None, None, "history_fetch_failed"

    if len(hist) < 2:
        return None, None, None, "insufficient_history"

    start_price = hist["Close"].iloc[0]
    end_idx = min(FORWARD_WINDOW_TRADING_DAYS, len(hist) - 1)
    actual_window_days = end_idx  # how many trading days forward this label actually reflects
    end_price = hist["Close"].iloc[end_idx]
    if not start_price:
        return None, None, None, "insufficient_history"

    pct_change = (end_price - start_price) / start_price
    if pct_change >= BULLISH_THRESHOLD:
        direction = "BULLISH"
    elif pct_change <= BEARISH_THRESHOLD:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"
    return direction, pct_change, actual_window_days, None


def confidence_from_move(direction, pct_change):
    # Bigger moves get higher confidence, on the reasoning that a move well
    # past the threshold is less likely to be pure noise than one that
    # barely cleared it. Loosely mirrors the synthetic generator's
    # per-category confidence ranges, not derived from anything rigorous.
    magnitude = min(abs(pct_change), 0.15) / 0.15  # normalize, cap at a 15% move
    if direction == "NEUTRAL":
        return round(0.55 + 0.15 * (1 - magnitude), 2)
    return round(0.65 + 0.30 * magnitude, 2)


def fetch_ticker_fundamentals_history(ticker_obj):
    """Fetches quarterly income statement, quarterly balance sheet, and the
    earnings-date history ONCE per ticker (not once per headline) - each is
    reused across all of that ticker's headlines, filtered down to
    'as-of the headline date' inside as_of_quarterly()/build_fundamentals_
    blocks() below. This is the same look-ahead-bias mitigation LOOKBACK_
    WEEKS/SAFETY_BUFFER_DAYS already apply to the price-move label: a
    training row must only ever see data that would genuinely have been
    knowable at the headline's own publish date, not a later-restated or
    since-updated figure.

    Known, documented approximation: yfinance's quarterly statements only
    go back ~4-5 quarters from TODAY (not from the headline date), and
    Ticker.info's sharesOutstanding/forward-PE/dividend-yield/52-week-range
    fields are all CURRENT snapshots with no historical equivalent exposed
    by yfinance. For headlines toward the older end of LOOKBACK_WEEKS this
    means: (a) EPS/book-value-per-share - the two inputs that actually
    drive the Graham Number valuation math - are still filtered to
    strictly-before the headline date, so that part stays real and
    look-ahead-free; (b) shares outstanding, forward P/E, dividend yield,
    and the 52-week range use today's current values as a residual
    approximation rather than the true as-of-date figures, since yfinance
    doesn't expose historical versions of those; (c) YoY revenue growth
    needs a quarter from ~a year before the as-of quarter, which the
    4-5-quarter window frequently doesn't reach - when it doesn't, the YoY
    figure is simply omitted from the earnings block rather than guessed.
    All three are deliberate, bounded approximations, not silently-ignored
    gaps - see the module docstring.

    Fails soft per-field: any individual fetch that raises leaves that
    field None rather than aborting the whole ticker.
    """
    result = {"income": None, "balance": None, "earnings_dates": None, "shares_outstanding": None}
    try:
        result["income"] = ticker_obj.quarterly_income_stmt
    except Exception as e:
        print(f"    Warning: quarterly_income_stmt fetch failed: {e}", flush=True)
    try:
        result["balance"] = ticker_obj.quarterly_balance_sheet
    except Exception as e:
        print(f"    Warning: quarterly_balance_sheet fetch failed: {e}", flush=True)
    try:
        result["earnings_dates"] = ticker_obj.earnings_dates
    except Exception as e:
        print(f"    Warning: earnings_dates fetch failed: {e}", flush=True)
    try:
        result["shares_outstanding"] = ticker_obj.info.get("sharesOutstanding")
    except Exception as e:
        print(f"    Warning: shares_outstanding/info fetch failed: {e}", flush=True)
    return result


def _row_series(df, row_names):
    """Returns the first matching row (a Series indexed by period-end
    Timestamp) from a yfinance quarterly statement DataFrame for whichever
    of row_names actually exists - the exact row label varies slightly
    across tickers/filings (e.g. some report 'Diluted EPS', others only
    'Basic EPS')."""
    if df is None or df.empty:
        return None
    for name in row_names:
        if name in df.index:
            return df.loc[name]
    return None


def _as_of_timestamp(index, as_of_date):
    """pd.Timestamp for as_of_date, localized to match `index`'s own
    tz-awareness. Confirmed live: yfinance's quarterly-statement and
    earnings_dates indices are INCONSISTENTLY tz-aware - some tickers/
    fields come back tz-naive, others localized to the exchange timezone
    (e.g. America/New_York) - and pandas raises TypeError comparing a
    naive Timestamp against a tz-aware DatetimeIndex (or vice versa)
    rather than silently coercing one to the other. Every direct
    `series.index < as_of_ts`-style comparison in this file must build its
    as_of_ts through this helper, keyed to the SPECIFIC index being
    compared against - two different fetches (e.g. quarterly financials vs
    earnings_dates) can have different tz-awareness even for the same
    ticker, so one as_of_ts computed globally isn't safe to reuse across
    both."""
    ts = pd.Timestamp(as_of_date)
    tz = getattr(index, "tz", None)
    return ts.tz_localize(tz) if tz is not None else ts


def as_of_quarterly(series, as_of_date):
    """Most recent value in a quarterly-indexed Series whose period-end is
    strictly before as_of_date, i.e. the last quarter that would already
    have been reported by then - plus that period's own end-date, so
    callers can label which quarter the figure is from. (None, None) if
    series is None or nothing qualifies."""
    if series is None:
        return None, None
    prior = series.dropna()
    as_of_ts = _as_of_timestamp(prior.index, as_of_date)
    prior = prior[prior.index < as_of_ts]
    if prior.empty:
        return None, None
    period = prior.index.max()
    return float(prior[period]), period


def as_of_price(ticker_obj, as_of_date):
    """Closing price on or just before as_of_date - the 'current market
    data' a user asking about this headline on this date would actually
    have seen. Fails soft to None."""
    try:
        start = as_of_date - datetime.timedelta(days=10)
        hist = ticker_obj.history(start=start, end=as_of_date + datetime.timedelta(days=1))
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


def build_fundamentals_blocks(ticker_obj, fundamentals_history, as_of_date):
    """Returns (market_data_block, valuation_block, earnings_block)
    strings, each independently 'Data unavailable.' if that piece couldn't
    be resolved as-of as_of_date - mirrors generate_synthetic_dataset.py's
    per-block rendering and financial-sentiment-api's real fetchers, so the
    model trains on the same range of shapes (including real gaps) it'll
    see served in production.
    """
    income = fundamentals_history["income"]
    balance = fundamentals_history["balance"]
    shares = fundamentals_history["shares_outstanding"]

    price = as_of_price(ticker_obj, as_of_date)

    eps_series = _row_series(income, ["Diluted EPS", "Basic EPS"])
    eps_trailing = None
    if eps_series is not None:
        prior_eps = eps_series.dropna()
        as_of_ts = _as_of_timestamp(prior_eps.index, as_of_date)
        prior_eps = prior_eps[prior_eps.index < as_of_ts].sort_index(ascending=False)
        if len(prior_eps) >= 4:
            eps_trailing = float(prior_eps.iloc[:4].sum())
        elif len(prior_eps) >= 1:
            # Only a single quarter available before this headline -
            # annualize it as a rough TTM stand-in (documented above).
            eps_trailing = float(prior_eps.iloc[0]) * 4

    revenue, rev_period = as_of_quarterly(_row_series(income, ["Total Revenue"]), as_of_date)
    equity, _ = as_of_quarterly(
        _row_series(balance, ["Common Stock Equity", "Stockholders Equity",
                               "Total Equity Gross Minority Interest"]),
        as_of_date,
    )

    book_value_per_share = (equity / shares) if (equity is not None and shares) else None
    market_cap = (price * shares) if (price is not None and shares) else None

    # --- market_data block ---
    if price is not None and market_cap is not None and eps_trailing:
        pe_trailing = price / eps_trailing if eps_trailing > 0 else None
        div_yield, year_low, year_high, forward_pe = None, None, None, None
        try:
            info = ticker_obj.info
            div_yield = info.get("dividendYield")
            year_low = info.get("fiftyTwoWeekLow")
            year_high = info.get("fiftyTwoWeekHigh")
            forward_pe = info.get("forwardPE")
        except Exception:
            pass
        pe_trailing_str = f"{pe_trailing:.1f}" if pe_trailing else "N/A"
        pe_forward_str = f"{forward_pe:.1f}" if forward_pe else "N/A"
        div_yield_str = f"{div_yield:.2f}%" if div_yield else "0.00%"
        range_str = (f"${year_low:.2f} - ${year_high:.2f}"
                     if (year_low and year_high) else "N/A")
        market_data_block = (
            f"Price: ${price:.2f} | Market Cap: {format_market_cap(market_cap)}\n"
            f"P/E (trailing): {pe_trailing_str} | P/E (forward): {pe_forward_str}\n"
            f"EPS (trailing): ${eps_trailing:.2f} | Dividend Yield: {div_yield_str}\n"
            f"52-Week Range: {range_str}"
        )
    else:
        market_data_block = "Data unavailable."

    # --- valuation block ---
    if eps_trailing and book_value_per_share and price is not None:
        if eps_trailing <= 0 or book_value_per_share <= 0:
            valuation_block = "Not applicable (negative or missing EPS/book value)."
        else:
            graham = (22.5 * eps_trailing * book_value_per_share) ** 0.5
            pct = (price - graham) / graham * 100
            verdict = "overvalued" if pct >= 0 else "undervalued"
            valuation_block = (
                f"Intrinsic Value (Graham Number): ${graham:.2f}\n"
                f"vs Current Price: {verdict} by ~{abs(pct):.0f}%"
            )
    else:
        valuation_block = "Data unavailable."

    # --- earnings block ---
    earnings_dates = fundamentals_history["earnings_dates"]
    earnings_block = "Data unavailable."
    if revenue is not None:
        # YoY needs a same-quarter-prior-year revenue figure, which the
        # ~4-5-quarter yfinance window frequently doesn't reach - omit the
        # parenthetical when it's not available rather than guess it.
        rev_series = _row_series(income, ["Total Revenue"])
        yoy_str = ""
        if rev_series is not None:
            prior_year_period = rev_period - pd.DateOffset(months=12)
            candidates = rev_series.dropna()
            nearest = candidates[(candidates.index - prior_year_period).map(abs) < pd.Timedelta(days=20)]
            if not nearest.empty:
                prior_rev = float(nearest.iloc[0])
                if prior_rev:
                    yoy = (revenue - prior_rev) / prior_rev * 100
                    yoy_str = f" ({'+' if yoy >= 0 else ''}{yoy:.1f}% YoY)"

        eps_note = ""
        if earnings_dates is not None and not earnings_dates.empty:
            reported = earnings_dates.dropna(subset=["Reported EPS"]) \
                if "Reported EPS" in earnings_dates.columns else earnings_dates.iloc[0:0]
            reported_as_of_ts = _as_of_timestamp(reported.index, as_of_date)
            prior_reports = reported[reported.index < reported_as_of_ts]
            if not prior_reports.empty:
                row = prior_reports.sort_index(ascending=False).iloc[0]
                actual = row.get("Reported EPS")
                est = row.get("EPS Estimate")
                if actual is not None and est is not None and est:
                    if actual > est:
                        eps_note = f", EPS ${actual:.2f} (beat est. ${est:.2f})"
                    elif actual < est:
                        eps_note = f", EPS ${actual:.2f} (missed est. ${est:.2f})"
                    else:
                        eps_note = f", EPS ${actual:.2f} (in line with est. ${est:.2f})"

        next_line = ""
        if earnings_dates is not None and not earnings_dates.empty:
            future_as_of_ts = _as_of_timestamp(earnings_dates.index, as_of_date)
            future = earnings_dates[earnings_dates.index > future_as_of_ts]
            if not future.empty:
                next_date = future.sort_index().index.min()
                next_line = f"\nNext Earnings Date: {next_date.date().isoformat()}"

        earnings_block = (
            f"Last Quarter ({rev_period.date().isoformat()}): "
            f"Revenue {format_market_cap(revenue)}{yoy_str}{eps_note}"
            f"{next_line}"
        )

    return market_data_block, valuation_block, earnings_block


def build_news_block(primary_headline_line):
    lines = [primary_headline_line]
    n_noise = random.randint(1, 3)
    for noise in random.sample(NOISE_HEADLINES, n_noise):
        date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=random.randint(0, 6))
        lines.append(f"- [{_format_date(date)}] {noise} - {random.choice(NOISE_PUBLISHERS)}")
    random.shuffle(lines)  # target headline isn't always first, like real feeds
    return "\n".join(lines)


def _template_reasoning(ticker, direction, pct_change, actual_window_days):
    # The original mechanism, kept only as generate_grounded_reasoning's
    # fallback for when the Gemini call itself fails - see that function's
    # docstring for why this text alone was the root cause of the model
    # learning to recall a memorized per-ticker answer instead of reading
    # the headline. Losing headline-grounding on an occasional row (a
    # transient API hiccup) is an acceptable degradation; losing it on
    # every row (the old default) is what broke real-data generalization.
    day_word = "trading day" if actual_window_days == 1 else "trading days"
    return (
        f"Over the {actual_window_days} {day_word} following this news, "
        f"{ticker} moved {pct_change * 100:+.1f}%, which resolves as {direction}. "
        f"This label reflects the market's actual subsequent move, not a "
        f"hand-verified causal read of the headline itself."
    )


GEMINI_REASONING_PROMPT = """You are labeling training data for a financial-news analyst model.

You are given a stock ticker, its current market data, a valuation estimate, its most recent earnings, a real news headline about it, a user's question, and a directional label (BULLISH, BEARISH, or NEUTRAL). That label was already determined from the stock's ACTUAL subsequent price move over the next few trading days - not from reading anything below. You do not have access to that price-move data, and you must not reference it, invent a percentage move, or write anything implying you know what the stock did afterward.

Write TWO things, each as its own labeled line (see OUTPUT FORMAT):

1. REASONING (2-3 sentences): Reads the headline and explains why it's plausibly consistent with a {direction} outlook - the way a financial analyst would talk through the available evidence, not the outcome. Weave in the market data, valuation, or earnings below ONLY where they genuinely reinforce or complicate the headline's own signal - don't force a mention if a block is irrelevant to this specific headline or says "Data unavailable."/"Not applicable", and never invent facts or numbers that aren't in what you were given.
   - If the headline's content does not obviously support {direction} (this happens often - many price moves in a short window are unrelated to the nearest headline), say so plainly - call it a weak or indirect signal rather than forcing a confident causal claim that isn't there.
   - If the headline's content clearly points the OPPOSITE way from {direction} (e.g. a headline reporting good news paired with a BEARISH label, or bad news paired with BULLISH - this happens often, since the label reflects the actual subsequent move and headlines don't always predict it), do NOT invent a contrarian story to force a fit - phrases like "already priced in," "overbought/oversold," or "the market sees through this" sound analytical but aren't something you can actually know from a single headline, and a model trained on that kind of reasoning learns to talk itself out of headlines it read correctly. Instead say plainly that the headline's own content points the other way, and that the labeled outcome likely reflects other developments not visible here.
2. ANSWER (1-2 sentences): A direct, plain answer to the user's question below, consistent with {direction} and, where relevant, the data above. If the question is empty, give a general one-line read on {ticker} instead.

Ticker: {ticker}
Current Market Data:
{market_data}

Valuation:
{valuation}

Recent Earnings:
{earnings}

Headline: {headline}
User Question: {user_query}
Direction: {direction}

OUTPUT FORMAT - exactly two lines, nothing else, no preamble or quotes:
REASONING: <text>
ANSWER: <text>"""


def _retry_delay_seconds(error_text, default=10.0):
    # google-genai's 429 error message embeds Google's own suggested wait
    # as a JSON-ish string, e.g. "'retryDelay': '46s'" - pull that out and
    # honor it instead of guessing a fixed backoff. Falls back to a fixed
    # default if the message shape ever changes (string-matched, not
    # parsed as real JSON, since this is on the exception's str(), not a
    # structured field the SDK is documented to expose).
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", error_text)
    return float(match.group(1)) if match else default


def _parse_gemini_output(text):
    """Splits Gemini's 'REASONING: ...\\nANSWER: ...' response into (reasoning,
    answer). Raises ValueError if the REASONING section is missing/empty -
    callers catch that as a normal Gemini-call failure and fall back to the
    template, same as any other malformed/empty response. A missing ANSWER
    section alone is NOT fatal - the caller fills it from ANSWER_TEMPLATES,
    since losing just the answer half shouldn't discard a good REASONING."""
    reasoning_match = re.search(r"REASONING:\s*(.*?)(?:\n\s*ANSWER:|$)", text, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r"ANSWER:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    if not reasoning:
        raise ValueError("no REASONING section in Gemini output")
    return reasoning, answer


def generate_grounded_reasoning(ticker, title, direction, pct_change, actual_window_days,
                                 market_data, valuation, earnings, user_query, qtype):
    """Replaces the old fixed template (ticker + price move + direction,
    never the headline itself) with headline-grounded reasoning from
    Gemini, now also weaving in market_data/valuation/earnings and writing
    the `answer` field for user_query. That template was confirmed live as
    the root cause of a trained model reproducing an identical memorized
    answer for a given ticker across unrelated headlines - see the module
    docstring's item 0.

    Returns (reasoning, answer) - both fall back independently: a Gemini
    failure/empty response/quota exhaustion falls back to
    (_template_reasoning(...), ANSWER_TEMPLATES[qtype][direction]); a
    response with REASONING but no parseable ANSWER line keeps Gemini's
    reasoning and only falls back the answer half.

    Confirmed live: the free tier's 15-requests/minute cap gets hit almost
    immediately with no pacing, and every call after that silently fell
    back to the template - defeating the whole point of this function
    without ever raising an error you'd notice. A 429/RESOURCE_EXHAUSTED
    is retried (honoring Google's suggested retryDelay) up to
    GEMINI_MAX_RETRIES times before giving up; every other failure (network
    error, empty response, safety block, etc.) falls back to the template
    immediately, same as before, so one non-recoverable bad call still
    can't abort an unattended multi-hundred-row run. Paces itself to
    GEMINI_REQUEST_DELAY_SECONDS between calls either way, to avoid
    re-triggering the same limit on the next row.

    Also confirmed live, and more serious: the free tier separately caps
    total requests at 500/DAY (RequestsPerDayPerProjectPerModel), distinct
    from the 15/minute cap above - a 40-ticker run needs up to ~2000 Gemini
    calls (kept headlines only), so hitting this is expected, not a fluke.
    Unlike the per-minute cap, no amount of waiting fixes this within the
    same day - the very first version of this retry loop didn't
    distinguish the two, so once the daily cap hit it wasted a full
    per-minute-style retry (tens of seconds) on every single remaining
    headline for the rest of the run before falling back. Detected
    separately here: once seen, every later call in this process skips
    straight to the template with no retry and no per-call pacing delay -
    there's nothing to wait out until the quota resets (~24h from first
    use)."""
    global _gemini_daily_quota_exhausted
    reasoning = _template_reasoning(ticker, direction, pct_change, actual_window_days)
    answer = ANSWER_TEMPLATES[qtype][direction].format(ticker=ticker)
    if _gemini_daily_quota_exhausted:
        return reasoning, answer

    prompt = GEMINI_REASONING_PROMPT.format(
        ticker=ticker, headline=title, direction=direction,
        market_data=market_data, valuation=valuation, earnings=earnings,
        user_query=user_query or "(none)",
    )
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            response = _gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = (response.text or "").strip()
            if not text:
                raise ValueError("empty response")
            parsed_reasoning, parsed_answer = _parse_gemini_output(text)
            reasoning = parsed_reasoning
            if parsed_answer:
                answer = parsed_answer
            break
        except Exception as e:
            error_text = str(e)
            if "RequestsPerDayPerProjectPerModel" in error_text:
                _gemini_daily_quota_exhausted = True
                print(f"    Gemini free-tier DAILY quota exhausted - the rest of this run will use the "
                      f"template fallback (nothing to wait out until the quota resets, ~24h from first "
                      f"use). Options: re-run tomorrow, lower MAX_HEADLINES_PER_TICKER so total calls "
                      f"stay under 500, or use a billed key.", flush=True)
                break
            is_rate_limited = "RESOURCE_EXHAUSTED" in error_text or "429" in error_text
            if is_rate_limited and attempt < GEMINI_MAX_RETRIES:
                wait = _retry_delay_seconds(error_text)
                print(f"    Gemini rate limit hit for {ticker!r} - waiting {wait:.0f}s before retry {attempt + 1}/{GEMINI_MAX_RETRIES}...", flush=True)
                time.sleep(wait)
                continue
            print(f"    Warning: Gemini reasoning call failed for {ticker!r} ({e!r}) - using template fallback.", flush=True)
            break
    if not _gemini_daily_quota_exhausted:
        time.sleep(GEMINI_REQUEST_DELAY_SECONDS)
    return reasoning, answer


def make_real_example(ticker, ticker_obj, fundamentals_history, title, publisher, published_at):
    """Returns (example_or_None, skip_reason). skip_reason is None on
    success, otherwise whatever label_from_forward_return reported."""
    direction, pct_change, actual_window_days, skip_reason = label_from_forward_return(ticker_obj, published_at)
    if direction is None:
        return None, skip_reason

    date_str = _format_date(published_at) if published_at else "recent"
    headline_line = f"- [{date_str}] {title} - {publisher}"

    confidence = confidence_from_move(direction, pct_change)
    user_query, qtype = build_user_query(ticker)

    as_of_date = published_at.date() if published_at else datetime.date.today()
    market_data, valuation, earnings = build_fundamentals_blocks(ticker_obj, fundamentals_history, as_of_date)

    reasoning, answer = generate_grounded_reasoning(
        ticker, title, direction, pct_change, actual_window_days,
        market_data, valuation, earnings, user_query, qtype,
    )

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

    example = {
        "ticker": ticker,
        "user_query": user_query,
        "market_data": market_data,
        "valuation": valuation,
        "earnings": earnings,
        "news": build_news_block(headline_line),
        "output": json.dumps(output_payload, indent=2),
    }
    return example, None


def append_examples(filepath, examples):
    with open(filepath, "a") as f:
        for row in examples:
            f.write(json.dumps(row) + "\n")


def _count_jsonl_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def already_completed_tickers():
    """Tickers with at least one row already written to either output
    file. A ticker only ever gets written after its whole per-ticker loop
    finishes (see generate_and_write) - so a ticker that was mid-progress
    when a run was interrupted has ZERO rows here and will correctly be
    reprocessed from scratch, while a ticker that fully finished won't be.
    Used to resume after a manual restart (Kaggle/Colab session died, or
    you interrupted deliberately - e.g. after a code fix mid-run) without
    reprocessing tickers already done, which would otherwise re-spend
    real Gemini quota and RSS/yfinance requests for no reason. Empty set
    on a genuinely fresh run, since the files don't exist yet."""
    done = set()
    for path in (OUTPUT_TRAIN_FILE, OUTPUT_VAL_FILE):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["ticker"])
    return done


def generate_and_write():
    """Writes each ticker's results to disk as soon as that ticker finishes,
    instead of accumulating everything in memory and writing once at the
    end. This is meant to survive an unattended Colab run: if something
    interrupts the process partway through (a Colab disconnect, an
    unexpected error on one ticker), whatever tickers already completed
    are safely on disk rather than lost entirely. Each ticker also runs
    inside its own try/except so one unexpected failure can't take down
    the other 39.

    Resumable: if the output files already have rows in them (a prior run
    was interrupted and this cell is being re-run), those tickers are
    skipped instead of the files being wiped and everything redone from
    scratch - confirmed live as a real problem (a 7+ hour run, ~15 tickers
    in, needed a restart for a code fix; wiping the files would have
    thrown all of that away, including real, already-spent Gemini quota).
    To force a genuinely fresh run, delete dataset_train_real.jsonl and
    dataset_val_real.jsonl yourself first."""
    resume_skip = already_completed_tickers()
    if resume_skip:
        print(f"Resuming - {len(resume_skip)} ticker(s) already in the output files, skipping: {sorted(resume_skip)}", flush=True)
    else:
        open(OUTPUT_TRAIN_FILE, "w").close()
        open(OUTPUT_VAL_FILE, "w").close()

    skip_reason_totals = {}
    # Seeded from what's already on disk (0 on a fresh run) so the final
    # "Collected N examples" summary reflects the true total, not just
    # this process's own additions on top of a resumed run.
    total_train = _count_jsonl_lines(OUTPUT_TRAIN_FILE)
    total_val = _count_jsonl_lines(OUTPUT_VAL_FILE)

    for ticker, name in TICKERS:
        if ticker in resume_skip:
            continue
        print(f"Fetching {ticker} ({name})...", flush=True)
        try:
            ticker_examples = []
            ticker_obj = yf.Ticker(ticker)
            # Fetched ONCE per ticker and reused across all of its
            # headlines below - avoids N x the quarterly-statement/
            # earnings-dates calls per ticker (see fetch_ticker_
            # fundamentals_history's docstring for why this is safe: each
            # headline still gets its own as-of-date filtering downstream).
            fundamentals_history = fetch_ticker_fundamentals_history(ticker_obj)
            is_val = ticker in VAL_HOLDOUT_TICKERS

            seen_titles = set()
            kept = 0
            ticker_skips = {}

            for window_i, (after_date, before_date) in enumerate(weekly_windows(), 1):
                if kept >= MAX_HEADLINES_PER_TICKER:
                    break

                headlines = fetch_headlines_for_window(ticker, name, after_date, before_date)
                time.sleep(NEWS_REQUEST_DELAY_SECONDS)

                # Per-window/per-headline output - without this, a ticker
                # can go silent for minutes at a time (each headline now
                # costs a real Gemini call: GEMINI_REQUEST_DELAY_SECONDS at
                # minimum, up to tens of seconds more on a rate-limit
                # retry) with nothing printed to distinguish "still
                # working" from "hung". Confirmed live: an interrupted run
                # that looked stuck for over a minute turned out to be mid-
                # loop, already well past the fetch, just silently working
                # through headlines one at a time.
                print(f"    window {window_i}/{LOOKBACK_WEEKS} ({after_date}..{before_date}): {len(headlines)} headlines", flush=True)

                for title, publisher, published_at in headlines:
                    if kept >= MAX_HEADLINES_PER_TICKER:
                        break
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)

                    example, skip_reason = make_real_example(
                        ticker, ticker_obj, fundamentals_history, title, publisher, published_at)
                    time.sleep(PRICE_REQUEST_DELAY_SECONDS)

                    if example:
                        ticker_examples.append(example)
                        kept += 1
                        print(f"      [{kept}/{MAX_HEADLINES_PER_TICKER}] kept: {title[:70]!r}", flush=True)
                    else:
                        ticker_skips[skip_reason] = ticker_skips.get(skip_reason, 0) + 1
                        skip_reason_totals[skip_reason] = skip_reason_totals.get(skip_reason, 0) + 1
                        print(f"      skipped ({skip_reason}): {title[:70]!r}", flush=True)

            skip_summary = ", ".join(f"{reason}={count}" for reason, count in ticker_skips.items())
            print(f"  {kept} labeled examples, {len(seen_titles)} unique headlines seen" + (f" (skipped: {skip_summary})" if skip_summary else ""), flush=True)

            target_file = OUTPUT_VAL_FILE if is_val else OUTPUT_TRAIN_FILE
            append_examples(target_file, ticker_examples)
            if is_val:
                total_val += len(ticker_examples)
            else:
                total_train += len(ticker_examples)

        except Exception as e:
            # Whatever prior tickers already wrote stays on disk; this
            # ticker is skipped entirely and the run moves on.
            print(f"  Warning: {ticker} failed unexpectedly, skipping it: {e!r}", flush=True)
            continue

    if skip_reason_totals:
        print()
        print("Skip reasons across all tickers:", skip_reason_totals)

    print(f"Collected {total_train} raw train examples, {total_val} val examples (before rebalancing)", flush=True)


def direction_of(example):
    return json.loads(example["output"])["impacted_stocks"][0]["direction"]


def rebalance_by_direction(examples):
    """Undersamples down to the minority class's count, so BULLISH/BEARISH/
    NEUTRAL are equally represented. Undersampling (not duplicating the
    minority classes up) is deliberate - this project has already run into
    a real overfitting problem from repeated content once (see the
    synthetic generator's template-count history), and duplicating real
    rows to pad a minority class would risk the same thing here. The cost
    is fewer total rows; that's an accepted tradeoff for a dataset this is
    only ever meant to be a supplement to, not the primary training set."""
    by_direction = {}
    for ex in examples:
        by_direction.setdefault(direction_of(ex), []).append(ex)

    if not by_direction:
        return examples

    minority_count = min(len(rows) for rows in by_direction.values())
    rebalanced = []
    for rows in by_direction.values():
        rebalanced.extend(random.sample(rows, minority_count))
    random.shuffle(rebalanced)

    before = {d: len(rows) for d, rows in by_direction.items()}
    print(f"Rebalanced train set by direction: {before} -> {minority_count} each ({minority_count * len(by_direction)} total)", flush=True)
    return rebalanced


def rebalance_file_in_place(filepath):
    with open(filepath) as f:
        rows = [json.loads(line) for line in f]
    rebalanced = rebalance_by_direction(rows)
    with open(filepath, "w") as f:
        for row in rebalanced:
            f.write(json.dumps(row) + "\n")
    return len(rebalanced)


def main():
    generate_and_write()

    # Rebalance train only - an artificially-balanced val set would give a
    # less honest read of real-world performance than val's actual (skewed)
    # distribution, which is what the model will actually be judged against.
    # This reads back whatever actually made it to disk rather than an
    # in-memory list, so it still produces a correctly-balanced file even if
    # generate_and_write() above was interrupted partway through.
    train_count = rebalance_file_in_place(OUTPUT_TRAIN_FILE)

    with open(OUTPUT_VAL_FILE) as f:
        val_count = sum(1 for _ in f)

    print(f"Wrote {OUTPUT_TRAIN_FILE}: {train_count} examples (rebalanced)", flush=True)
    print(f"Wrote {OUTPUT_VAL_FILE}:   {val_count} examples (held-out tickers: {sorted(VAL_HOLDOUT_TICKERS)}, natural/unbalanced distribution)", flush=True)


if __name__ == "__main__":
    main()
