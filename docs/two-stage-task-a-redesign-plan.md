# Plan: Task A redesign — single-day move + relevance-filtered headline selection

## Context

The two-stage pipeline (Task A classifies news reaction → `fusion.py` deterministically picks BUY/SELL/HOLD from reaction × valuation gap → Task B explains the given recommendation) is **fully built** but sitting on two unmerged, budget-gated PRs, by original design:
- `financial-sentiment-api` PR #39 (`feat/two-stage-fusion`)
- `financial-sentiment-model` PR #48 (`feat/two-stage-reaction-pipeline`)

Both are intentionally left open pending GATE A (Gemini Task B prose regen) and GATE B (bundled GPU retrain) — this is expected, not neglect (see the original plan's "Deployment coupling" note).

This plan refines **Task A specifically**, prompted by two live observations this session:

1. **Window dilution**: the current design labels/describes the "immediate move" as a **3-trading-day forward window** from the headline's publish date. User's live counter-example: SIGN.SW had CEO-change news, dropped >10% the SAME day, then recovered +4% over the next two days. A 3-day cumulative window nets these together (~-6%), understating the obvious same-day overreaction the user actually observed. **Fix: redefine the "immediate move" as the single publish-day close-to-close move.** (The forward-looking retracement check, ~day 21, stays as-is — it's training-label-only and inherently retrospective, never shown to the model, so it's unaffected by this "no future data at inference" concern.)

2. **Irrelevant headlines still reaching the LLM**: existing filters (`LOW_QUALITY_PUBLISHERS`, `_is_low_content_headline`) catch bad *sources* and bad *shapes* (price-recap wrappers, listicles, fund-flow spam) but not well-sourced, well-shaped headlines that are simply **not about this company** (generic macro news, sector roundups). Need a relevance pre-filter: headline must reference the company (name/ticker) **or** its sector/industry (keyword match) to even be considered. Headlines matching neither are dropped before pricing or classification — never shown to Task A. If nothing relevant survives, no forced reaction: falls through to "neutral" (no new class needed).

**Key finding from this session's investigation** (verified via `git show <branch>:<path>`, not assumed): the model repo's real-dataset generator (`generate_real_dataset.py` on `feat/two-stage-reaction-pipeline`) is **already single-headline-per-row** — `process_ticker` iterates one real headline at a time; there's no "batch of headlines classified together" to fix on the training side. What's NOT yet single-headline is **production inference**: `financial-sentiment-api`'s `news.py` (on `feat/two-stage-fusion`) fetches up to 4 filtered headlines and joins them into ONE block per request, with one shared *trailing-as-of-now* price move — this needs rework to match training's actual shape.

**Headline-selection rule when multiple relevant candidates exist** (user-confirmed): classify the **most recently published** qualifying headline. Simple, matches "what's the latest news reaction," and doesn't require pricing/classifying every candidate before choosing.

## What's changing, concretely

### 1. Model repo (`financial-sentiment-model`, branch `feat/two-stage-reaction-pipeline`)

**`generate_real_dataset.py`**:
- `measure_reaction_windows` (currently ~line 796): change the "immediate move" from *close at publish+3 days vs. publish-day close* to **publish-day close vs. previous-day close** (single trading day). Leave `move_21d` (forward from publish day) untouched — still training-label-only, used only for the retracement check.
- `classify_reaction` (~line 869): same overreaction-first-else-good/bad/neutral logic, now driven by the single-day figure instead of the 3-day one.
- `price_context_block` (~line 910): update phrasing from `"{ticker} moved {move:+.1f}% over the last 3 trading days."` to a single-day phrasing (e.g. `"{ticker} moved {move:+.1f}% on the day this was published."`) — must land byte-identical with the new production-side helper (see API section below), per this repo's existing cross-repo sync convention.
- **New relevance pre-filter** (e.g. `_is_relevant_headline(ticker, sector, title) -> bool`): passes if the headline mentions the company name/ticker **or** matches a sector/industry keyword (see `SECTOR_KEYWORDS` starter list below). Wire into `process_ticker`'s per-headline loop alongside the existing publisher/shape filters, same skip-reason-tracking pattern already used for `low_quality_publisher`/`low_content_headline`.
- `calibrate_reaction_thresholds.py` (already exists, free/yfinance-only): update to sweep against the new single-day move definition instead of the 3-day one (single-day return volatility is smaller in magnitude than 3-day cumulative, so `REACTION_GOOD_BAD_THRESHOLD` (currently 0.02), `REACTION_OVERREACTION_MOVE_THRESHOLD` (currently 0.05), and `REACTION_RETRACEMENT_FRACTION` (currently 0.5) all need re-derivation). Re-run against the same acceptance bands as before (overreaction 5–12% of labelable rows, neutral 25–45%); record new constants in `generate_real_dataset.py`.
- `diagnose_return_distribution.py`: already dead on this branch (imports a function retired in an earlier phase) — delete as unrelated cleanup, low priority.
- Full real-dataset regeneration required after the above (free — yfinance + Google News RSS only, no Gemini needed for Task A rows, same as before).

**`generate_synthetic_dataset.py`**:
- `_fabricate_move_pct` (~line 2404): rescale fabricated per-class move ranges to single-day magnitudes (currently good/bad 3–8%, overreaction 8–15%, tuned for the 3-day window) — new ranges come out of the recalibration study above.
- Minor/non-blocking: reconcile `build_news_block` taking a list of headlines (synthetic) vs. the real generator's single-headline shape — cosmetic, not required for this change.

**Starter `SECTOR_KEYWORDS`** (first pass — needs live refinement against real headlines per sector, same iterative process already used for `LOW_QUALITY_PUBLISHERS`/`_is_low_content_headline`, which took 3–4 live rounds to get right):

| Sector (Yahoo `.info` sector string) | Keywords |
|---|---|
| Technology | ai, artificial intelligence, chip, semiconductor, software, cloud, cybersecurity, data center |
| Energy | oil, gas, crude, opec, drilling, refinery, pipeline |
| Healthcare | drug, fda, clinical trial, biotech, pharma, vaccine |
| Industrials | aerospace, defense, manufacturing, supply chain, factory |
| Financial Services | rate hike, federal reserve, banking, interest rates, credit |
| Consumer Cyclical / Defensive | retail sales, consumer spending, e-commerce |
| Real Estate, Utilities, Basic Materials, Communication Services | TBD — derive from real headlines seen in the ticker universe during implementation |

Known gap: "space" (the user's own example) doesn't map cleanly to one Yahoo sector (space/satellite companies often sit under Industrials or Communication Services) — may need a small ticker-level keyword override list rather than a pure sector mapping. Address case-by-case once real examples are seen.

### 2. API repo (`financial-sentiment-api`, branch `feat/two-stage-fusion`)

- **`app/services/news.py`**: rework `fetch_live_news_rag` to return a list of structured, already-quality-filtered candidates (`title`, `publisher`, `published`) instead of one joined string. Apply the same new relevance pre-filter (ported byte-identical from the model repo, per sync convention). Select the **most recently published** surviving candidate as the one headline used for this request.
- **`app/services/fundamentals.py`**: new function computing a **specific day's** close-to-close move (e.g. `price_move_on_date(ticker, date) -> (text, move_pct|None)`), replacing `recent_price_move`'s "trailing as of now" semantics for Task A's `price_context`. Check whether anything else still needs `recent_price_move` before removing it.
- **`app/services/inference.py`**: prompt *shapes* (`_build_reaction_prompt`/`_build_analysis_prompt`) stay the same — only what's substituted into `price_context`/`live_context` changes (now the one selected headline + its own day's move, not a 4-headline block + a trailing figure). No fan-out/multi-classify needed in `analyze_two_stage`, since selection happens in `news.py` before Task A is ever called.
- **`app/routers/analyze.py`**: wire the new structured fetch + date-specific price move into the existing `asyncio.gather`.
- **Task B** keeps receiving the same single headline that drove the classification (for traceability — reasoning should reference the actual news reacted to), not a separately-fetched multi-headline block.
- Tests to update/add: `tests/test_news.py` (relevance filter, structured candidates, most-recent-wins selection), `tests/test_fundamentals.py` (new date-specific price-move function), `tests/test_inference.py` (updated price_context/live_context shape), `tests/test_analyze_router.py` (updated wiring).

## Order of work (when resumed — NOT decided today)

1. Recalibration study (free): update `calibrate_reaction_thresholds.py` for single-day move, re-run, pick new constants.
2. Model repo: single-day move + relevance pre-filter in `generate_real_dataset.py`, regenerate real Task A dataset (free), update `generate_synthetic_dataset.py`'s fabricated ranges.
3. API repo: structured multi-candidate news fetch + ported relevance filter + most-recent selection + date-specific price move; update `analyze_two_stage` wiring and tests.
4. Confirm GATE A / GATE B still apply as originally scoped (this redesign changes Task A's input/label shape, not the Gemini/GPU cost estimates) before spending budget on either.
5. Ship, per the original plan's Phase 3 (push checkpoint, repoint via `/api/update-inference-url`, E2E smoke test).

## Verification (once resumed)

- Recalibration script output reviewed against the same acceptance bands as before.
- Spot-check regenerated real dataset rows: (a) `price_context` correctly reflects the single-day move, (b) relevance filter keeps genuinely sector-relevant headlines (test against a real oil/tech/space example) and drops off-topic noise, (c) the SIGN.SW-style same-day-crash-then-recovery case classifies as `overreaction_down` under the new single-day definition — manual spot check against the real historical example that prompted this redesign.
- New API tests pass; live smoke-test against a real ticker query, confirm the single selected headline + its date-specific move appear correctly in both Task A and Task B prompts (via logging, same method used earlier this session).
