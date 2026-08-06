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
NOISE_HEADLINES and USER_QUESTION_TEMPLATES verbatim), so both datasets'
JSONL rows are interchangeable and can be concatenated/mixed for training:

    data_files={"train": ["dataset_train.jsonl", "dataset_train_real.jsonl"], ...}

Requirements: `pip install yfinance httpx feedparser google-genai`, network
access, and a Gemini API key (GEMINI_API_KEY) - see generate_grounded_
reasoning for where that's read from and why the model choice is
gemini-3.5-flash-lite specifically.
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
longer run than earlier, smaller configurations. GEMINI_REQUEST_DELAY_SECONDS
(4.5s, sized for the free tier's 15-requests/minute cap on
gemini-3.5-flash-lite - see the comment above that constant) adds roughly
another 4.5s per kept headline on top of that, since a Gemini call happens
once per row now. This is expected, not a hang; the per-ticker incremental
writes and try/except (see generate_and_write) mean a slow run is safe to
leave unattended.

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
    import yfinance as yf
    from google import genai
except ImportError as e:
    raise SystemExit(
        f"This script needs a package that isn't installed ({e.name}). "
        "Run: pip install yfinance httpx feedparser google-genai"
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
# per minute (generativelanguage.googleapis.com/generate_content_free_tier_requests).
# With zero pacing between calls, a run blows through that almost
# immediately and every call after the first ~15 falls back to the
# template - silently defeating the whole point of this feature (100% of
# rows end up ungrounded again, just without an obvious error). 60/15 = 4s
# minimum between calls; this adds margin. If GEMINI_API_KEY has billing
# enabled, this limit is much higher and the delay can be lowered - check
# https://ai.google.dev/gemini-api/docs/rate-limits for the current tier.
GEMINI_REQUEST_DELAY_SECONDS = 4.5
GEMINI_MAX_RETRIES = 2

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


def build_user_query(ticker):
    template = random.choice(USER_QUESTION_TEMPLATES)
    return template.format(ticker=ticker) if template else ""


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


GEMINI_REASONING_PROMPT = """You are labeling training data for a financial-news sentiment model.

You are given a stock ticker, a real news headline about it, and a directional label (BULLISH, BEARISH, or NEUTRAL). That label was already determined from the stock's ACTUAL subsequent price move over the next few trading days - not from reading the headline. You do not have access to that price data, and you must not reference it, invent a percentage move, or write anything implying you know what the stock did afterward.

Write 2-3 sentences of reasoning that:
- Reads the headline itself and explains why this kind of news is plausibly consistent with a {direction} outlook - the way a financial analyst would talk through the headline, not the outcome.
- Does not invent facts, numbers, or details that are not in the headline.
- If the headline's content does not obviously support {direction} (this happens often - many price moves in a short window are unrelated to the nearest headline), say so plainly - call it a weak or indirect signal rather than forcing a confident causal claim that isn't there.

Ticker: {ticker}
Headline: {headline}
Direction: {direction}

Output ONLY the reasoning text - no preamble, no headers, no quotes around it."""


def _retry_delay_seconds(error_text, default=10.0):
    # google-genai's 429 error message embeds Google's own suggested wait
    # as a JSON-ish string, e.g. "'retryDelay': '46s'" - pull that out and
    # honor it instead of guessing a fixed backoff. Falls back to a fixed
    # default if the message shape ever changes (string-matched, not
    # parsed as real JSON, since this is on the exception's str(), not a
    # structured field the SDK is documented to expose).
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", error_text)
    return float(match.group(1)) if match else default


def generate_grounded_reasoning(ticker, title, direction, pct_change, actual_window_days):
    """Replaces the old fixed template (ticker + price move + direction,
    never the headline itself) with headline-grounded reasoning from
    Gemini. That template was confirmed live as the root cause of a
    trained model reproducing an identical memorized answer for a given
    ticker across unrelated headlines - see the module docstring's item 0.

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
    re-triggering the same limit on the next row."""
    prompt = GEMINI_REASONING_PROMPT.format(ticker=ticker, headline=title, direction=direction)
    reasoning = _template_reasoning(ticker, direction, pct_change, actual_window_days)
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            response = _gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = (response.text or "").strip()
            if not text:
                raise ValueError("empty response")
            reasoning = text
            break
        except Exception as e:
            error_text = str(e)
            is_rate_limited = "RESOURCE_EXHAUSTED" in error_text or "429" in error_text
            if is_rate_limited and attempt < GEMINI_MAX_RETRIES:
                wait = _retry_delay_seconds(error_text)
                print(f"    Gemini rate limit hit for {ticker!r} - waiting {wait:.0f}s before retry {attempt + 1}/{GEMINI_MAX_RETRIES}...", flush=True)
                time.sleep(wait)
                continue
            print(f"    Warning: Gemini reasoning call failed for {ticker!r} ({e!r}) - using template fallback.", flush=True)
            break
    time.sleep(GEMINI_REQUEST_DELAY_SECONDS)
    return reasoning


def make_real_example(ticker, ticker_obj, title, publisher, published_at):
    """Returns (example_or_None, skip_reason). skip_reason is None on
    success, otherwise whatever label_from_forward_return reported."""
    direction, pct_change, actual_window_days, skip_reason = label_from_forward_return(ticker_obj, published_at)
    if direction is None:
        return None, skip_reason

    date_str = _format_date(published_at) if published_at else "recent"
    headline_line = f"- [{date_str}] {title} - {publisher}"

    confidence = confidence_from_move(direction, pct_change)
    reasoning = generate_grounded_reasoning(ticker, title, direction, pct_change, actual_window_days)

    output_payload = {
        "impacted_stocks": [
            {
                "ticker": ticker,
                "reasoning": reasoning,
                "direction": direction,
                "confidence": confidence,
            }
        ]
    }

    example = {
        "ticker": ticker,
        "user_query": build_user_query(ticker),
        "news": build_news_block(headline_line),
        "output": json.dumps(output_payload, indent=2),
    }
    return example, None


def append_examples(filepath, examples):
    with open(filepath, "a") as f:
        for row in examples:
            f.write(json.dumps(row) + "\n")


def generate_and_write():
    """Writes each ticker's results to disk as soon as that ticker finishes,
    instead of accumulating everything in memory and writing once at the
    end. This is meant to survive an unattended Colab run: if something
    interrupts the process partway through (a Colab disconnect, an
    unexpected error on one ticker), whatever tickers already completed
    are safely on disk rather than lost entirely. Each ticker also runs
    inside its own try/except so one unexpected failure can't take down
    the other 39."""
    open(OUTPUT_TRAIN_FILE, "w").close()
    open(OUTPUT_VAL_FILE, "w").close()

    skip_reason_totals = {}
    total_train = 0
    total_val = 0

    for ticker, name in TICKERS:
        print(f"Fetching {ticker} ({name})...", flush=True)
        try:
            ticker_examples = []
            ticker_obj = yf.Ticker(ticker)
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

                    example, skip_reason = make_real_example(ticker, ticker_obj, title, publisher, published_at)
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
