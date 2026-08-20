"""Tests for the Task B direction-consistency heuristic (_has_opposite_
action_language / _OPPOSITE_ACTION_WORDS), duplicated byte-identically
across 5 eval scripts (colab/train/{gpu,tpu}/evaluate_model.py, their
evaluate_base_model_only.py siblings, runpod/evaluate_model.py - see
CONTRIBUTING.md's sync convention). None of those scripts are directly
importable: they open with Colab `!pip install` magic lines and depend on
unsloth/torch/a live loaded model at import time. So each test here reads
the source file as text and execs just the self-contained regex/function
block (no external deps beyond the stdlib `re` module already used by that
block) into an isolated namespace - this exercises the REAL code that ships
in each file, not a hand-copied re-implementation that could silently drift
from it.

Parametrized across all 5 files so a future edit to only one of them (a
sync break) fails these tests, not just a manual eyeball diff.

Run from the repo root:
    python -m pytest tests/
"""

import re

import pytest

EVAL_SCRIPT_PATHS = [
    "colab/train/gpu/evaluate_model.py",
    "colab/train/gpu/evaluate_base_model_only.py",
    "colab/train/tpu/evaluate_model.py",
    "colab/train/tpu/evaluate_base_model_only.py",
    "runpod/evaluate_model.py",
]


def _load_detector(path):
    """Execs the self-contained _BENIGN_COMPOUND_SUFFIX_RE.._has_opposite_
    action_language block out of `path` in isolation and returns
    (_OPPOSITE_ACTION_WORDS, _has_opposite_action_language)."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("_BENIGN_COMPOUND_SUFFIX_RE"))
    end = next(i for i, l in enumerate(lines) if i > start and l.rstrip() == "    return False")
    block = "".join(lines[start:end + 1])
    ns = {"re": re}
    exec(block, ns)
    return ns["_OPPOSITE_ACTION_WORDS"], ns["_has_opposite_action_language"]


@pytest.fixture(params=EVAL_SCRIPT_PATHS, ids=lambda p: p)
def detector(request):
    opposite_words, has_opposite = _load_detector(request.param)

    def contradicts(recommendation, reasoning, answer):
        opposite_re = opposite_words.get(recommendation)
        return bool(opposite_re and (has_opposite(answer, opposite_re) or has_opposite(reasoning, opposite_re)))

    return contradicts


# --- False positives from the real 2026-08-20 eval run (representative
# subset of the 19 - one example per bug class fixed, not all 19) ---


def test_sell_out_hyphen_compound_not_flagged(detector):
    assert not detector(
        "BUY",
        "Immediate sell-out demand for next-gen hardware is a strong, concrete signal of near-term revenue upside.",
        "Yes, the current signals lean favorably enough that ZVEX looks like a reasonable buy here.",
    )


def test_selling_out_space_variant_not_flagged(detector):
    assert not detector(
        "BUY",
        "The headline about a new streaming platform selling out within days of launch is a strong, positive signal.",
        "NFLX is currently trading at an attractive discount relative to its intrinsic value.",
    )


def test_short_term_compound_not_flagged(detector):
    assert not detector(
        "BUY",
        "The headline offers a deep dive into JPMorgan Chase's fundamentals rather than reporting a specific short-term catalyst.",
        "The recent market correction and broader sector consolidation offer a potential entry point for JPMorgan Chase.",
    )


def test_short_report_not_flagged_bare_short_no_longer_a_trigger(detector):
    assert not detector(
        "BUY",
        "A short report built mainly on information the company had already disclosed doesn't introduce much new risk.",
        "Sentiment on JPM is bullish today.",
    )


def test_negated_doesnt_look_like_a_buy_not_flagged(detector):
    assert not detector(
        "SELL",
        "VLTR screens as overvalued on a DCF basis, reinforcing that read.",
        "No, the current signals are negative enough that VLTR doesn't look like a buy right now.",
    )


def test_hedge_word_after_trigger_not_flagged(detector):
    # "premature" lands AFTER "selling" - the check must look both ways,
    # not just backward from the match.
    assert not detector(
        "BUY",
        "Furthermore, the strong operating margin of 34.8% provides fundamental support.",
        "Given the significant upside potential and strong operating performance, selling now may be premature.",
    )


def test_model_quoting_headline_then_correctly_overriding_not_flagged(detector):
    # The model transparently acknowledges the headline argues the
    # opposite before correctly landing on the given recommendation via
    # valuation - sophisticated, correct reasoning, not a contradiction.
    assert not detector(
        "SELL",
        "The headline explicitly frames a potential buying opportunity at key support, which runs directly counter to a SELL outlook. However, the valuation estimate indicates the stock is overvalued by roughly 33%, providing fundamental support for a downward correction.",
        "Netflix faces downward pressure as elevated valuation multiples leave little room for further upside.",
    )


def test_upgrade_to_buy_rating_quoted_then_overridden_not_flagged(detector):
    assert not detector(
        "SELL",
        "The headline reports an upgrade to a Buy rating from Goldman Sachs, which runs directly counter to a SELL outlook, making it a weak or indirect signal for this particular move. However, the valuation estimate indicates Netflix is overvalued by roughly 48%.",
        "The recent upgrade from Goldman Sachs points in the opposite direction of a sell recommendation, but the stock's rich valuation leaves it vulnerable to a correction.",
    )


# --- Genuine contradictions MUST still be caught (the whole point of the
# detector) - these are constructed, not from a real eval run ---


def test_genuine_sell_advice_under_buy_recommendation_is_flagged(detector):
    assert detector(
        "BUY",
        "The news is bad and the stock looks overvalued.",
        "You should sell this stock immediately given the bad news.",
    )


def test_genuine_buy_advice_under_sell_recommendation_is_flagged(detector):
    assert detector(
        "SELL",
        "The catalyst is strong and the stock looks cheap.",
        "This is a great buying opportunity, you should buy now.",
    )


def test_genuine_short_recommendation_under_buy_is_flagged(detector):
    assert detector(
        "BUY",
        "Given the risk, consider shorting the stock here.",
        "This looks risky.",
    )


def test_genuine_accumulate_advice_under_sell_is_flagged(detector):
    assert detector(
        "SELL",
        "The valuation and momentum both look attractive.",
        "I'd recommend accumulating shares at this price.",
    )


def test_genuine_exit_and_take_profits_advice_under_buy_is_flagged(detector):
    assert detector(
        "BUY",
        "The setup looks weak.",
        "Now would be a good time to exit your position and take profits.",
    )


def test_hold_recommendation_never_flagged():
    # HOLD has no single "opposite" action - _OPPOSITE_ACTION_WORDS["HOLD"]
    # is None by design, so the caller's `bool(opposite_re and ...)` short-
    # circuits to False regardless of content. Only needs checking once,
    # not parametrized across all 5 files - the dict entry itself, not
    # regex behavior, is what's being verified.
    opposite_words, _has_opposite = _load_detector(EVAL_SCRIPT_PATHS[0])
    assert opposite_words["HOLD"] is None
