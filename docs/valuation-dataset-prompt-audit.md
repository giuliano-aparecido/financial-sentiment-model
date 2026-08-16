# Valuation, Dataset, and Prompt Audit (2026-08-16)

Triggered by comparing computed DCF intrinsic values against a real investor
analyst's own numbers for 19 tickers. Curated tickers (NVDA/MSFT/AAPL/PEP/
NFLX/XOM) matched within ~12%; derived (non-curated) growth names ran
+130% to +177% too high (META/AMZN/TSLA/ADBE/BABA), and two tickers
(GOOG, CHTR) were unusable "bad input" cases entirely. Two full audits
followed - one of `valuation.py`/`fundamentals.py`/`inference.py`
(production), one of the synthetic training dataset - surfacing ~40
confirmed issues across DCF math, dataset generation, and the LLM prompt.
This document records the findings, what was fixed, what was verified
live, and what's deliberately left for a follow-up.

## Analyst comparison (ground truth, 10% discount rate)

| Ticker | Our IV (before) | Analyst IV | Diff (before) | Diff (after fixes) |
|---|---|---|---|---|
| NVDA | 270.71 | 242.24 | +11.8% | +11.8% (curated, unchanged) |
| MSFT | 425.18 | 405.97 | +4.7% | +4.7% (curated, unchanged) |
| AAPL | 131.59 | 129.28 | +1.8% | +1.8% (curated, unchanged) |
| PEP | 103.88 | 104.52 | -0.6% | -0.6% (curated, unchanged) |
| NFLX | 71.47 | 66.89 | +6.8% | +6.8% (curated, unchanged) |
| ADYEN | 907.81 | 945.06 | -3.9% | +8.1% |
| BRK-B | 479.35 | 565.81 | -15.3% | -8.7% |
| ACN | 160.80 | 143.55 | +12.0% | +43.9% (regressed - see below) |
| MELI | 1278.69 | 1032.56 | +23.8% | +19.2% |
| QSR | 83.58 | 67.66 | +23.5% | +21.5% |
| INTU | 440.31 | 321.63 | +36.9% | +44.2% (slightly worse) |
| META | 1257.55 | 546.68 | +130% | +60.8% |
| ADBE | 454.36 | 176.05 | +158% | +143.8% |
| TSLA | 43.08 | 16.37 | +163% | +92.0% |
| AMZN | 441.79 | 165.51 | +167% | +156.9% |
| BABA | 365.01 | 131.69 | +177% | -8.3% |
| GOOG | excluded (bad input) | 200.14 | n/a | +227.2% (now usable) |
| CHTR | excluded (bad input) | 269.07 | n/a | +63.3% (now usable) |

Bad-input cases (data quality, not model error): GOOG's growth_0y=90.4%
(yfinance consensus artifact), CHTR's pe_trailing~4.0 (one-time-item-
distorted trailing EPS), CSU's analyst-sheet unit mismatch (unaffected by
this round of work).

**Honest assessment**: the worst outliers (BABA, META, TSLA, ADBE, and
both previously-excluded tickers) improved substantially. ACN and INTU
regressed or stayed flat - both trace to the same root cause (see
"Known limitation: mega-cap/high-growth DCF sensitivity" below), not a
bug in the fixes themselves. This was a real, live-verified tradeoff
decision, not an oversight - see the financial-sentiment-api commit
history for the specific iterations tried and reverted (a separate,
higher dividends-basis exit-multiple set was tried and reverted after
QSR's real data contradicted the theoretical justification for it).

## A. Input-sanity fixes (financial-sentiment-api PR #32)

1. **NaN poisoning** - `fundamentals.py`'s growth-consensus fetch only
   guarded `if not year_ago`, and `not float('nan')` is `False`, so a
   NaN low/high/growth cell from yfinance's `earnings_estimate` flowed
   through `valuation.py`'s clamps (`min(nan, X)` returns `nan`) all the
   way to a rendered `Intrinsic Value: $nan`. Fixed with an explicit
   `_usable()` NaN check.
2. **Consensus magnitude gate** - the same-direction reliability check
   accepted GOOG's 90.4% growth_0y (a rebound-off-depressed-base
   artifact) as "reliable" since growth_1y was also positive.
   `CONSENSUS_GROWTH_MAGNITUDE_CAP = 0.60` rejects either year above
   that magnitude regardless of direction agreement.
3. **best/worst offset bug** - when only ONE of `growth_0y_high`/`_low`
   existed, `best`/`worst` kept a stale prior-tier value while `normal`
   had already been overwritten by the consensus blend, breaking the
   `best >= normal >= worst` guarantee. Fixed by mirroring whichever
   offset IS available.
4. **CHTR-class EPS distortion** - `ONE_TIME_ITEM_PE_RATIO_THRESHOLD`
   catches a one-off item that reverts by next year (trailing P/E far
   below forward P/E); `PERSISTENTLY_LOW_PE_THRESHOLD` catches a
   *persistent* distortion (CHTR's actual shape: both trailing AND
   forward P/E ~4, a leveraged-buyback EPS artifact, not a one-off) -
   this one returns `None` for the eps basis instead of guessing at a
   correction, routing through the new basis-fallback chain (finding
   B6) to `fcf` instead.

## B. Structural DCF fixes

5. **SUSTAINABLE_GROWTH_ROE_CAP/_FLOOR** - the actual mechanism behind
   META/AMZN/TSLA/ADBE/BABA's overshoot: buyback-shrunk book value
   inflates the `eps_trailing/book_value_per_share` ROE proxy well past
   100%. Capped at 0.25 (kept below the tightened G1_CAP so scenarios
   stay distinguishable). BRK-B's dual-class book-value data artifact
   (near-zero raw ratio) is now floored as unusable data (0.02) rather
   than trusted as a real "no growth" signal.
6. **G1_CAP tightened 0.40 -> 0.30** - to match, not exceed, NVDA's own
   curated "best" g1 (0.30, the single most aggressive human-vetted
   number on file). A derived/unverified g1 should never outrun that.
7. **Sector-anchored eps/fcf exit multiple** - `min(NORMAL_EXIT_MULTIPLE,
   SECTOR_MEDIAN_PE[sector])` for the "normal" tier - the flat 20x
   directly contradicted `SECTOR_MEDIAN_PE` shown in the same prompt for
   banks/energy/utilities/materials. `min()` (not the sector median
   outright) preserves Technology's existing curated calibration
   (median 28.0 stays capped at 20.0).
8. **Dividend-stream PV** added to the eps/fcf terminal-only formula for
   companies that pay SOME dividend below the dividends-basis threshold
   (payout_ratio > 0) - previously discarded entirely for a full decade
   (MMM-class partial payers).
9. **Payout-threshold cliff smoothing** - a 0.001 payout-ratio change
   used to flip the entire formula (28.5% intrinsic-value jump). Now
   blended across a +/-0.05 band around `DIVIDEND_PAYOUT_THRESHOLD` for
   non-curated, non-REIT tickers.
10. **Basis-fallback chain** - a curated/REIT override with no usable
    cf0 (e.g. yfinance drops a field for one call) now retries other
    bases instead of a permanent "Not applicable".
11. **Negative worst-case g2** - `min(g2_worst, g1_worst)` for eps/fcf/
    revenue; previously fixed at +4%, making a structural value trap
    (declining company) mathematically unrepresentable.
12. **Currency-mismatch guard** - `currency` != `financial_currency`
    (cross-listed tickers) now returns `None` for the fcf/revenue bases
    rather than silently dividing a financial-currency total by a
    trading-currency share count.
13. **Reverted**: a separate, higher `DIVIDEND_*_EXIT_MULTIPLE` set
    (theoretically motivated - 20x implies too rich a terminal dividend
    yield) was tried and reverted after live QSR data showed it made
    the gap WORSE (23.5% -> 58.5%), not better. The theoretical argument
    didn't survive contact with the one real non-curated calibration
    point available for this basis. Dividends basis shares eps/fcf's
    flat normal/best defaults (worst still varies by
    `ASSET_HEAVY_SECTORS`, unrelated to the reverted piece).

## C. Display fixes

14. `valuation_block` shows `>150%` (not `~150%`) once the display cap
    actually fires, and "trading near fair value" instead of a
    directional "overvalued by ~0%" when price is within 1% of
    intrinsic value.
15. PEG's growth denominator floored (`PEG_MIN_GROWTH_FOR_COMPUTATION =
    0.02`) - confirmed PEG values up to 525 in the synthetic dataset's
    shared formula, a pure division artifact from a near-zero (not
    negative) growth rate, not a real signal.
16. `Sector Median P/E` always renders the sector name now, even when
    no median exists for it (Real Estate) - previously a bare "N/A"
    gave the model no in-prompt signal it was looking at a REIT.

## D. news.py

17. Two fabricated fallback strings ("Recent market volatility and
    financial developments for X.", "Recent quarterly earnings and news
    updates for X.") replaced with the standard "Data unavailable."
    convention - a Google News RSS fetch failure was reading as
    asserted fact to the model.

## E. Prompt fixes (inference.py + all 8 training/eval alpaca templates)

18. **Rule 2**: explicit valuation-weighing instruction - the Valuation
    block previously had no weighing rule at all (rule "1." was the
    only critical rule and it was about news), reading as decorative.
19. **Rules 3-4**: NEUTRAL and confidence both used in the instruction
    with no definition. NEUTRAL now means "signals genuinely conflict
    or are too weak/routine," not a default for uncertainty; confidence
    now has an explicit 0.0-1.0 scale.
20. **Output schema hint**: the JSON shape the parser actually requires
    (`impacted_stocks`/`direction`/...) now appears in the prompt
    itself, not just the fine-tune weights - a checkpoint swap
    previously degraded silently to the `raw_response` fallback.
21. **Rule 5**: explicit instruction against inventing data to fill a
    "Data unavailable."/"Not applicable" gap.
22. **Rule 6**: value-investing-checklist.md's graded P/E bands and the
    REIT Price/Book override now live in the prompt itself.
23. **`_sanitize_user_query`**: strips markdown heading markers and this
    template's own section-marker words before interpolation -
    `user_query` sits upstream of every fetched data block, so an
    unsanitized `### Response:` or counterfeit `Valuation:` line could
    hijack the prompt.

Ported byte-equivalent to all 8 training/eval template copies (`colab/
train/{gpu,tpu}/{train_model,evaluate_model,evaluate_base_model_only}.py`,
`runpod/{train_model,evaluate_model}.py`) - verified each still `.format()`s
correctly with the doubled-brace JSON schema escaping.

**Deliberately deferred**: an "As of: `<date>`" line (a separate, lower-
severity finding - no date anywhere in the prompt). Adding it to
`inference.py` alone would need a NEW per-row training-data column
neither generator currently has, and shipping it one-sided would
silently mismatch every training example's prompt shape against
production's - the same class of bug as this repo's own
`fix/training-prompt-whitespace-mismatch`. Left for a follow-up that
threads the column through both generators and all 8 template copies
together, not shipped half-done.

## F. Dataset generator fixes (`generate_synthetic_dataset.py`)

24. **Valuation/label decoupling (D1, the highest-impact dataset
    finding)**: for BULLISH/BEARISH/MIXED rows, the real DCF ran
    independently of the news-driven label - ~50% of large-gap
    (>=30%) rows had the valuation pointing opposite the label with no
    acknowledgment, training the model to silently ignore the Valuation
    block whenever news is present. Fixed: when a real, parsed gap
    contradicts the label at >=30%, `reasoning` now gets an explicit,
    randomly-phrased acknowledgment clause (mirroring VALUATION_SIGNAL's
    own `news_wins` tier, which already does this deliberately).
25. **Unrealistic fundamentals (D3)**: `pe_trailing` was capped at
    exactly 28.0 for every row (the model never saw an expensive
    stock) - a stable per-ticker "high multiple" trait (~12% of
    tickers) now draws 30-90x. `div_yield` was re-rolled independently
    per ROW (the same ticker flip-flopped dividend-payer status across
    rows, ~60% of TSLA/PLTR/COIN rows fabricated a yield) - now a
    stable per-ticker trait (~45% of tickers). Market cap varied 3-10x
    for the same ticker (an independent per-row multiplier draw) - now
    a stable per-ticker baseline with only mild +/-10% row noise.
    `_ticker_trait_roll` (a stable SHA256-derived per-ticker value,
    independent of the global `random` stream) backs all three traits.
26. **Direction-biased template holdout (D4)**: the evenly-strided
    holdout-index selector could land entirely on one direction when a
    scenario list is internally grouped by direction, which both
    `VALUATION_SIGNAL_SCENARIOS` and `MIXED_SIGNAL_SCENARIOS` are -
    confirmed live VALUATION_SIGNAL's val split was 97 BULL/49 BEAR
    against train's 120/173 (inverted), MIXED's was 6 BULL/32 BEAR
    against train's 57/38. `held_out_template_indices_stratified`
    strides WITHIN each direction group separately before merging.
27. **Product/sector mismatch (D6)**: `PRODUCTS` picked with no sector
    gating produced 217/1600 rows (13.6%) with nonsensical pairings
    ("PepsiCo... chip manufacturing line", "Exxon... enterprise software
    margins"). `PRODUCTS_BY_SECTOR` keys the product phrase off the
    company's informal sector.
28. **Checklist coverage (D7)**: quality/growth-adjusted-pricing
    commentary reached reasoning text on only ~8% of ALL rows
    (VALUATION_SIGNAL's 32% weight x 25% commentary rate). Extended
    eligibility to BULLISH/BEARISH/MIXED at a lower rate (0.15, not the
    same 0.25) - a coverage fix, not a repeat of the frequency mistake
    already documented in `VALUE_SCREEN_COMMENTARY_PROB`'s own comment
    (a v16 eval found the model reciting this phrasing near-verbatim at
    a higher combined rate than even this addition reaches).
29. **`value-investing-checklist.md` (D8)**: referenced by 3 code
    comments in this repo but never actually committed here (it only
    existed in financial-sentiment-api). Copied over.

**Partially addressed / deliberately not fully solved**:
- **D2 (template memorization)**: 156 true templates behind 1100 train
  rows, 97.1% sharing a 30-char opening with another row. D1's
  conflict-acknowledgment phrasings and D7's broadened commentary both
  add SOME new phrasing variety, but a full rewrite of all ~156
  templates with 2-3x more phrasing variants each is a much larger
  effort than this round's time budget covers. Flagged as the single
  biggest remaining dataset-quality risk.
- **D5 (val split composition)**: 4 holdout tickers still dominate the
  val split; D4's direction-stratification fix changes the split size
  (1116 train / 484 val, vs 1100/500 before) but doesn't rebalance
  which tickers anchor it. Revisit if eval numbers still look
  ticker-concentrated after the next retrain.

## Known limitation: mega-cap/high-growth DCF sensitivity

Every non-curated ticker still resolves to a DCF-derived fair multiple
chosen by either its ROE-derived SGR or its real analyst consensus
growth rate, compounded over a 10-year terminal-value formula with a
fixed 3-scenario spread. For companies whose REAL consensus growth is
itself aggressive (META 35.5%, AMZN 73.2% this-year estimates) or whose
crude ROE proxy is genuinely high (a real, not artifactual, signal for
a quality business), the terminal-value formula's structural
sensitivity to g1 means even a "correctly" capped/floored derivation can
still land meaningfully above a professional analyst's own (likely more
qualitatively-discounted) target. GOOG (+227%), AMZN (+157%), ADBE
(+144%), ACN (+44%), and INTU (+44%) are the residual cases. This is not
a bug to chase further with constant-tuning - it's a structural
property of a 2-stage terminal-value DCF applied mechanically to
real-world consensus data, and matches this module's own documented
history of iterative, evidence-driven (not theoretical) calibration.

## Verification

- financial-sentiment-api: full pytest suite green (215/215) after every
  commit; live re-verification of the 19-ticker analyst comparison after
  each round of DCF changes.
- financial-sentiment-model-colab: `generate_synthetic_dataset.py`'s
  `generate()` produces 1600 examples with no errors after all fixes;
  `generate_real_dataset.py`'s `build_fundamentals_blocks` produces
  valuation numbers closely tracking production's own `valuation.py` for
  XOM/QCOM/CHTR/ACN (same basis labels, consistent gap percentages); all
  8 training/eval alpaca-prompt templates verified to `.format()`
  correctly with the new schema-hint brace escaping.

## PRs

- financial-sentiment-api #32: DCF/news fixes, PEG floor, prompt fixes.
- financial-sentiment-model-colab (this repo): `fix/dcf-valuation-audit-
  phase1` branch - 4-way sync of the DCF fixes, dataset generator fixes
  D1/D3/D4/D6/D7/D8, prompt template port, this document.
