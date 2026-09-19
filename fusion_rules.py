"""Deterministic BUY/SELL/HOLD fusion - turns the LLM's Task A news_reaction
classification and the numeric DCF valuation gap into the actual trading
recommendation and confidence, with no LLM involved. This is the mechanism
that used to be an implicit rule the model had to learn from training
examples (the old single-call prompt's CRITICAL SENTIMENT RULES 2-3, and
generate_real_dataset.py's valuation_alignment/headroom-downgrade gate and
generate_synthetic_dataset.py's VALUATION_CONFLICT_DOWNGRADE_THRESHOLD,
both now retired); now it's a fixed, testable table instead of something
trained into the model.

This repo uses it to compute the `recommendation` given to Task B training
rows (both real and synthetic) as the label the model is taught to
reproduce (see docs/task-b-learned-recommendation-plan.md) - Task B's LLM
call now decides the recommendation itself at inference, so this file has
no synced production counterpart anymore: financial-sentiment-api's
former app/services/fusion.py was deleted once nothing there called it.
This file stays here unchanged, purely a training-data concern.
"""

from typing import NamedTuple, Optional

NEWS_REACTIONS = ("good", "bad", "neutral", "overreaction_down", "overreaction_up")

# +/- this gap % -> over/undervalued; between the two -> near_fair. Half of
# VALUATION_CONFLICT_DOWNGRADE_THRESHOLD (40%, generate_synthetic_
# dataset.py) / the "extreme" gap tier's own floor (30%) - the DCF estimate
# carries real noise, so a bucket boundary needs to be a gap large enough
# to be signal rather than estimate error, without being so large that only
# the most extreme rows ever leave "near_fair".
VALUATION_GAP_BUCKET_THRESHOLD_PCT = 15.0


class FusionResult(NamedTuple):
    recommendation: str  # "BUY" | "SELL" | "HOLD"
    confidence: float
    valuation_bucket: str  # "undervalued" | "near_fair" | "overvalued" | "no_data"


# (recommendation, base_confidence) per (news_reaction, valuation_bucket).
# Row-by-row rationale (encodes the old CRITICAL SENTIMENT RULES headroom
# doctrine deterministically):
#   good: catalyst + headroom = BUY; catalyst without headroom = HOLD, never
#     BUY. Good news on an already-overvalued stock partially justifies the
#     premium (HOLD), not a SELL signal.
#   bad: bad news with no margin-of-safety cushion = SELL; bad news on an
#     undervalued stock is a genuine conflict = HOLD. bad+no_data -> SELL is
#     a deliberate loss-aversion asymmetry vs good+no_data -> HOLD: without
#     valuation data, protecting capital on real bad news beats chasing
#     good news.
#   neutral: valuation alone drives the call (undervalued -> BUY, overvalued
#     -> SELL, else HOLD) - this is what VALUATION_SIGNAL_SCENARIOS used to
#     teach as a special case; here it's just the neutral row of one table.
#   overreaction_down: the thesis class - an overdone crash in an already-
#     cheap stock is the strongest BUY signal; overdone but still expensive
#     is HOLD (no headroom to act on the reversion).
#   overreaction_up: mirror of overreaction_down - overdone spike in an
#     expensive stock is the strongest SELL; overdone but still cheap is
#     HOLD (don't chase, but don't sell a cheap stock either - a genuine
#     conflict).
FUSION_TABLE = {
    ("good", "undervalued"): ("BUY", 0.80),
    ("good", "near_fair"): ("HOLD", 0.60),
    ("good", "overvalued"): ("HOLD", 0.65),
    ("good", "no_data"): ("HOLD", 0.55),

    ("bad", "undervalued"): ("HOLD", 0.60),
    ("bad", "near_fair"): ("SELL", 0.70),
    ("bad", "overvalued"): ("SELL", 0.80),
    ("bad", "no_data"): ("SELL", 0.55),

    ("neutral", "undervalued"): ("BUY", 0.65),
    ("neutral", "near_fair"): ("HOLD", 0.70),
    ("neutral", "overvalued"): ("SELL", 0.65),
    ("neutral", "no_data"): ("HOLD", 0.60),

    ("overreaction_down", "undervalued"): ("BUY", 0.85),
    ("overreaction_down", "near_fair"): ("BUY", 0.70),
    ("overreaction_down", "overvalued"): ("HOLD", 0.60),
    ("overreaction_down", "no_data"): ("HOLD", 0.55),

    ("overreaction_up", "undervalued"): ("HOLD", 0.60),
    ("overreaction_up", "near_fair"): ("SELL", 0.70),
    ("overreaction_up", "overvalued"): ("SELL", 0.85),
    ("overreaction_up", "no_data"): ("HOLD", 0.55),
}


def valuation_bucket(gap_pct: Optional[float]) -> str:
    """gap_pct = (price - intrinsic) / intrinsic * 100, positive = overvalued.
    Expects the raw, uncapped/unrounded gap - the 150% display cap applied
    when rendering the Valuation block's text is a rendering concern only,
    not part of this decision."""
    if gap_pct is None:
        return "no_data"
    if gap_pct <= -VALUATION_GAP_BUCKET_THRESHOLD_PCT:
        return "undervalued"
    if gap_pct >= VALUATION_GAP_BUCKET_THRESHOLD_PCT:
        return "overvalued"
    return "near_fair"


def fuse(news_reaction: str, gap_pct: Optional[float]) -> FusionResult:
    """The deterministic recommendation used ONLY as the label Task B's
    LLM call is trained to reproduce (see docs/task-b-learned-
    recommendation-plan.md) - Task B decides the recommendation itself at
    inference now, this is never called there anymore. Raises ValueError
    for an unrecognized news_reaction; callers
    must normalize a bad/unparseable LLM classification (e.g. to "neutral")
    before calling this, so a real bug here is never silently masked by an
    accidental fallback recommendation.

    confidence scales with how decisive the valuation leg is: a larger gap
    (up to a 50-point span past the bucket boundary) adds up to 0.10 on top
    of the table's base_confidence, capped at 0.95. no_data gap keeps the
    base confidence unadjusted.
    """
    if news_reaction not in NEWS_REACTIONS:
        raise ValueError(f"Unknown news_reaction: {news_reaction!r}")

    bucket = valuation_bucket(gap_pct)
    recommendation, base_confidence = FUSION_TABLE[(news_reaction, bucket)]

    confidence = base_confidence
    if bucket != "no_data":
        magnitude = min(abs(gap_pct) / 50.0, 1.0)
        confidence = min(0.95, base_confidence + 0.10 * magnitude)

    return FusionResult(recommendation=recommendation, confidence=round(confidence, 2), valuation_bucket=bucket)
