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

Output schema is UNCHANGED from the original -
{"impacted_stocks": [{"ticker", "reasoning", "direction", "confidence"}]} -
because app/services/inference.py in financial-sentiment-api parses
analysis_json["impacted_stocks"][0] directly. Changing this schema requires
a matching change there; out of scope here.

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

SENTIMENT_WEIGHTS = {"BULLISH": 0.34, "BEARISH": 0.34, "NEUTRAL": 0.22, "MIXED": 0.10}

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

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def random_recent_date(days_back_max=6):
    # Matches app/services/news.py's entry.get("published", "")[:16] exactly
    # (a raw RSS pubDate truncated to 16 chars, e.g. "Tue, 05 Aug 2026") -
    # training on the same date format the model sees in production.
    anchor = datetime.date(2026, 8, 5)
    d = anchor - datetime.timedelta(days=random.randint(0, days_back_max))
    return f"{WEEKDAYS[d.weekday()]}, {d.day:02d} {MONTHS[d.month - 1]} {d.year}"


def format_headline(text, publisher=None):
    date = random_recent_date()
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
MIXED_SIGNAL_SCENARIOS = [
    (["{name} ({ticker}) beat Q{q} earnings estimates by {beat}%",
      "{name} ({ticker}) cut its full-year guidance, citing softening {product} demand heading into next quarter"],
     "Despite beating this quarter's estimates, the guidance cut signals deteriorating forward demand for {product} - forward guidance outweighs a backward-looking beat.",
     "BEARISH"),
    (["{name} ({ticker}) missed quarterly revenue estimates by {beat}%",
      "{name} ({ticker}) simultaneously announced a ${buyback}B share buyback program"],
     "The revenue miss is the more decision-relevant signal; a buyback doesn't offset weakening underlying demand.",
     "BEARISH"),
    (["{name} ({ticker}) reported a strong Q{q}, with revenue of ${rev}B beating estimates by {beat}%",
      "Analysts flagged {product} inventory buildup as a risk to next quarter's results"],
     "Current results are genuinely strong, but the flagged inventory risk introduces real uncertainty about next quarter - bullish, with tempered confidence.",
     "BULLISH"),
    (["{name} ({ticker}) issued a recall affecting {units}k {product} units",
      "The recall follows a quarter of record {product} sales, reported just last week"],
     "A recall's safety and legal risk outweighs the prior quarter's already-priced-in sales record."),
    (["{name} ({ticker}) disclosed a regulatory fine related to {product} practices",
      "{name} ({ticker}) also raised its full-year guidance, citing broad-based demand strength"],
     "The guidance raise is a stronger forward signal than a one-time fine, which is a minor operational item rather than a guidance or revenue signal - bullish, with reduced confidence given the fine.",
     "BULLISH"),
    (["{name} ({ticker}) reported solid Q{q} results in line with expectations",
      "Broader market headlines describe a sector-wide selloff unrelated to {ticker}'s own fundamentals"],
     "{ticker}'s own results are solid; the sector-wide selloff in the noise headlines is not company-specific and shouldn't be weighted into {ticker}'s own outlook - bullish, with confidence tempered by broader market uncertainty.",
     "BULLISH"),
    (["{name} ({ticker}) beat Q{q} revenue estimates by {beat}%",
      "{name} ({ticker}) separately announced its CEO will step down at year-end as part of a planned transition"],
     "A beat is the more decision-relevant signal; a planned, orderly CEO transition is a minor operational item, not a guidance or revenue signal - bullish, with confidence tempered by leadership transition uncertainty.",
     "BULLISH"),
    (["Analysts upgraded {ticker} ({name}) to 'Buy' citing long-term {product} potential",
      "{name} ({ticker}) issued weak near-term guidance, citing short-term {product} softness"],
     "The near-term guidance is the more decision-relevant, dated signal; a long-term-oriented upgrade doesn't offset a concrete near-term guidance cut - bearish.",
     "BEARISH"),
    (["{name} ({ticker}) missed Q{q} earnings estimates by {beat}%",
      "{name} ({ticker}) simultaneously raised its quarterly dividend by {divhike}%"],
     "The earnings miss is the more decision-relevant signal; a dividend increase is a minor operational item relative to a revenue/earnings miss - bearish.",
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
     "The guidance raise is the more decision-relevant, forward-looking signal; in-line current results don't offset positive forward guidance - bullish.",
     "BULLISH"),
]
# Backfill a uniform 4-tuple shape (some entries above omit the explicit
# resolved_direction when it's unambiguous from the reasoning text itself).
MIXED_SIGNAL_SCENARIOS = [
    s if len(s) == 3 else (s[0], s[1], "BEARISH") for s in MIXED_SIGNAL_SCENARIOS
]

# BULLISH/BEARISH now nested by tier - "subtle" gets a lower range than
# "clear" (a softer signal genuinely warrants less certainty) but still well
# above NEUTRAL's range, so low confidence alone doesn't become another
# implicit way to say "I'm not sure, call it NEUTRAL."
CONFIDENCE_RANGES = {
    "BULLISH": {"clear": (0.85, 0.97), "subtle": (0.66, 0.80)},
    "BEARISH": {"clear": (0.83, 0.96), "subtle": (0.66, 0.80)},
    "NEUTRAL": {"clear": (0.60, 0.82)},
    "MIXED": {"clear": (0.55, 0.75)},
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
    lines = [format_headline(h) for h in primary_headlines]
    n_noise = random.randint(1, 3)
    for noise in random.sample(NOISE_HEADLINES, n_noise):
        lines.append(format_headline(noise))
    random.shuffle(lines)  # target headline isn't always first, like real feeds
    return "\n".join(lines)


def build_user_query(ticker):
    template = random.choice(USER_QUESTION_TEMPLATES)
    return template.format(ticker=ticker) if template else ""


def make_example(company, category):
    ticker = company[0]

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
    user_query = build_user_query(ticker)

    confidence = round(random.uniform(conf_low, conf_high), 2)

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

    is_holdout_ticker = ticker in VAL_HOLDOUT_TICKERS
    split = "val" if (is_holdout_ticker or is_holdout_template) else "train"

    return {
        "ticker": ticker,  # duplicated from output.impacted_stocks[0].ticker so the
                           # training script can build "Target Stock: {ticker}" without
                           # parsing the output JSON string
        "user_query": user_query,
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
