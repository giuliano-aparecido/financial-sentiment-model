# Plan: Task A on each of the top 3 headlines, Task B sees all 3 with reactions

Status: proposal, not started (2026-09-24).

## Goal

- **Today:**
  - `news.py` picks 1 headline.
  - Task A classifies the market's reaction to it (`news_reaction`).
  - Task B gets that one reaction plus the headline and decides BUY/SELL/HOLD.
- **Target:**
  - `news.py` picks the **top 3** screened headlines.
  - **Task A runs once per headline**, each with its own single headline and its own publish-day price move, exactly the same input shape as today.
  - **Task B** gets all 3 headlines, each with its reaction, and decides from them together.

## Current state (verified in code)

**API** (`financial-sentiment-api`):
- `fetch_live_news_rag` returns `(one line, published_date)`. `_classified_pick` takes the best headline by importance within 7 days, else the newest.
- `price_move_on_date(ticker, published_date)` returns the price context for that day.
- `analyze_two_stage`:
  - calls `classify_news(ticker, price_context, live_context)` once;
  - uses `"neutral"` if Task A can't be parsed;
  - calls `generate_analysis(..., news_reaction, context)`.
- The Task B prompt has `News Reaction: {x}` and `Recent News & Results:\n{live_context}`.
- The response dict exposes a single `news_reaction` string and `live_news_retrieved`.

**Training** (`financial-sentiment-model`):
- **Task A rows:** one headline plus 1–3 synthetic noise lines. The label comes from that headline's day-0 move.
- **Task B rows:**
  - `recommendation = fusion_rules.fuse(reaction, gap_pct)`, which takes **one** reaction;
  - reasoning and answer are written by Gemini (`generate_grounded_reasoning`) from **one** headline and reaction;
  - real Task B rows only exist when `ENABLE_TASK_B_GENERATION=True`, at one Gemini call per row.
- **Synthetic:** `make_example` returns one Task A row and one Task B row that share one reaction.
- **Prompt copies:** the Task B template is copied in `inference.py` and in 6 train/eval scripts. CONTRIBUTING's sync rule applies to all of them: `colab/train/{gpu,tpu}/{train_model,evaluate_model,evaluate_base_model_only}.py` and `runpod/{train_model,evaluate_model}.py`.

## What changes and what doesn't

| Part | Change | Retrain needed |
|---|---|---|
| Task A prompt and data | **None.** Each call still gets 1 headline and that headline's own price move | No |
| `news.py` selection | Return the top 3 ranked, not the top 1 | — |
| Task A calls | 3 per request (in parallel), each with its own `price_move_on_date` | — |
| Task B prompt | New "news and reactions" section replacing the single `News Reaction:` line | **Yes** |
| Task B training data | Rows with 1–3 (headline, reaction) items | **Yes** |

A single LoRA adapter serves both tasks, so this is **one retrain of the whole adapter**. Task A's data doesn't change, so its eval should hold, and that gets checked.

## Design

### Task B input format

Replace:

```
News Reaction: good
...
Recent News & Results:
- [Tue, 23 Sep 2026] Novo Nordisk Stock Slides After ... - TIKR
```

with (the other sections unchanged):

```
Recent News & Market Reactions:
- [Tue, 23 Sep 2026] Novo Nordisk Stock Slides After ... - TIKR | reaction: bad
- [Tue, 23 Sep 2026] Novo Nordisk open to direct New York listing, CEO says - Proactive | reaction: neutral
- [Mon, 04 Aug 2026] Novo Nordisk lifts outlook; second-quarter sales rise 7% - Stock Titan | reaction: good
```

- **Order:** most important first, the same ranking `news.py` uses.
- **Count:** 1–3 lines. There are fewer when fewer headlines pass the screen or a price move can't be measured. Training shows the same spread.
- **Noise:** no noise lines. The screen already removed noise, so training should match.

### Task B training label: `recommendation` from several reactions

`fuse()` takes one reaction. **Decision needed.** Options:

1. **Anchor reaction (recommended).** Label with `fuse(reaction of the #1 headline, gap)`.
   - This is exactly today's label, so the table and its thresholds stay put.
   - Headlines #2 and #3 add context and reasoning, not a different verdict.
   - Low risk for a single retrain cycle.
   - Downside: the model may learn to rely only on line 1.
2. **Aggregated reaction.** Add a `combine_reactions([(reaction, importance)])` to `fusion_rules.py`, for example:
   - a vote weighted by importance over good/bad/neutral;
   - overreaction classes kept only when they're the anchor's reaction.
   - This makes the other two matter, but it's a new hand-written rule with no evidence behind it yet.
3. **Let Gemini decide the label.** Rejected: it's the move away from deterministic labels that `docs/task-b-learned-recommendation-plan.md` argued against.

### Inference (`financial-sentiment-api`)

- **`news.py`:**
  - `_classified_pick` becomes `_classified_ranking(candidates, ..., limit)`, returning the top `limit` relevant headlines.
  - Ranking: headlines with importance > 1 from the last 7 days by (importance, date) come first, then the rest by the current fallback order.
  - `fetch_live_news_rag` returns `list[(line, published_date)]`.
  - The regex fallback still returns 1.
- **`analyze.py`:**
  - For each item, `price_move_on_date` then `classify_news`, all running in parallel (`asyncio.gather`).
  - An item with no price move ("Data unavailable.") still goes to Task A, same as today.
  - A Task A parse failure falls back to `"neutral"` for that item only.
- **`inference.py`:**
  - The Task B prompt builder takes `list[(line, reaction)]`.
  - `analyze_two_stage` takes the items instead of one `live_context` string.
- **Response compatibility:**
  - Keep `news_reaction` as the #1 headline's reaction and `live_news_retrieved` as the joined lines, so current frontend consumers keep working.
  - Add `news_items: [{headline, published_date, price_context, news_reaction}]`.
  - Check financial-sentiment-web (or whatever else reads `/api/analyze`) before changing any existing field.
- **Config:** `TASK_B_NEWS_ITEMS` in `app/config.py`, default **1**.
  - At 1, the Task B prompt must be **byte-identical to today's**, and a test pins that.
  - The API PR can then merge and deploy before the retrained model exists. Switching is only an env-var change.
- **Latency:** 3 Task A generations per request. Task A output is short (one JSON field), so in parallel this adds little wall time.
  - Check whether the Modal serve config accepts concurrent inputs. If it doesn't, the 3 calls queue one after another.
  - GPU time per request for Task A roughly triples.

### Training data (`financial-sentiment-model`)

- **`generate_real_dataset.py`:**
  - **Task A rows:** unchanged, one per labeled headline.
  - **Task B rows (when `ENABLE_TASK_B_GENERATION`):** one per weekly window, built from the window's top 3 successfully labeled headlines (the screen's importance order, each with the reaction already computed for its Task A row).
    - Label per the decision above.
    - Market data, valuation and earnings as of the #1 headline's date.
    - `generate_grounded_reasoning` / `GEMINI_REASONING_PROMPT` gets all 3 headlines and reactions.
  - **Cost:** Gemini reasoning calls drop from one per kept headline to one per window, about 720 at most instead of up to ~2000. That's cheaper than today.
- **`generate_synthetic_dataset.py`:**
  - `make_example` produces 1–3 (headline, reaction) items with one Task A row each, plus one Task B row over all of them.
  - Keep the reaction distribution per item as today. Draw the item count from a spread such as 1: 30%, 2: 30%, 3: 40%, so the model sees every count.
- **Prompt templates:** update the Task B template in all 6 copied scripts, the same as `inference.py`, per CONTRIBUTING's sync rule.
- **Eval:** Task B recommendation accuracy against the label, split by the number of items and by whether the items agree or conflict. Task A eval is unchanged and serves as the regression check.

## Rollout order

1. **API PR:** the top-3 ranking, parallel Task A calls, the new Task B builder and the `news_items` response, all behind `TASK_B_NEWS_ITEMS=1`. Tests prove the default is byte-identical.
2. **Model repo PR:** the Task B format in both generators, the label rule, the reasoning prompt and the 6 template copies. Tests cover row shape, label and the template sync.
3. **Dry run** of the real generator on ~5 tickers with `ENABLE_TASK_B_GENERATION=True`. Inspect the Task B rows and label balance.
4. **Full regeneration** of both datasets. Use the billed Gemini key: about 720 screen calls plus about 720 reasoning calls.
5. **One retrain** (a scarce Colab cycle), bundled with other pending dataset fixes.
6. **Eval:**
   - Task A must hold.
   - Task B is judged on recommendation accuracy, overall and on rows with conflicting items.
7. **If eval passes:** deploy the model, then set `TASK_B_NEWS_ITEMS=3` on Render. If it regresses, keep the old model at 1; production is unaffected.

## Open questions

1. **Label rule:** anchor reaction (option 1, recommended) or an aggregated rule (option 2)?
2. **Show the price move?** Should Task B also see each headline's price move (`moved -4.2% that day`) next to its reaction? Today Task B never sees `price_context`. Recommended: no, to keep the change small.
3. **Response field:** keep `news_reaction` as the #1 headline's reaction for compatibility, or replace it with the list and update the frontend at the same time?
4. **Old headlines:** should any of the 3 be allowed to be older than 7 days when fewer than 3 recent ones exist? Recommended: yes, keeping the current fallback order. Otherwise quiet weeks give 1 item.
