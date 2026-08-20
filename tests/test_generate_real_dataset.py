"""First test coverage for generate_real_dataset.py's headline filters -
this repo had none before (it's normally run as a Colab/Kaggle script, not
a tested local package). Scoped to the two pure, regex/logic-only filter
functions (_is_low_content_headline, _is_relevant_headline) - everything
else in this module does live yfinance/Google-News/Gemini I/O at import or
call time and isn't practically unit-testable without a much bigger
mocking investment than these filters need.

Run from the repo root:
    python -m pytest tests/
"""

from generate_real_dataset import _is_low_content_headline, _is_relevant_headline


# --- _is_low_content_headline ---


def test_catches_trades_up_here_is_why():
    assert _is_low_content_headline("Intel (INTC) Stock Trades Up, Here Is Why")


def test_catches_why_stock_dropped_variant():
    assert _is_low_content_headline("Why Tesla Stock Dropped on Tuesday")
    assert _is_low_content_headline("Why is Amazon stock rallying today?")
    assert _is_low_content_headline("Why Adobe (ADBE) Stock Is Falling Today")


def test_catches_listicle_bait():
    assert _is_low_content_headline("Should You Buy Microsoft Stock Before April 29?")
    assert _is_low_content_headline("Is Oracle Stock a Buy at $245?")


def test_catches_fund_flow_spam():
    assert _is_low_content_headline("46,643 Shares in Oracle Corporation $ORCL Purchased by Trust Co. of Vermont")


def test_leaves_real_news_alone():
    assert not _is_low_content_headline("Oracle cuts guidance, citing softening cloud demand")
    assert not _is_low_content_headline("Tesla signs major Arizona power deal")
    assert not _is_low_content_headline("Exxon Mobil raises full-year production guidance")
    assert not _is_low_content_headline("JPMorgan Chase reports record quarterly trading revenue")


# --- 2026-08-20 additions (found via the fine-tune's eval misclassifications) ---


def test_catches_stock_is_up_today_not_just_shares_are_up_today():
    # The pre-existing "shares are/is up/down/higher/lower today" branch
    # only recognized "shares" as the subject - "stock" slipped through,
    # and this exact headline became a real-dataset row's (content-free)
    # signal headline, contributing to a Task A eval misclassification.
    assert _is_low_content_headline("Why Netflix (NFLX) Stock Is Up Today")
    assert _is_low_content_headline("NVDA Stock Is Down Today")
    assert _is_low_content_headline("AAPL stock is higher today")


def test_catches_is_x_still_attractive_after_bait_variant():
    # Generalizes the existing "^Is .+ a Good Stock" bait pattern to the
    # "Is X Still [adjective] After [[[a move]]]?" phrasing - same
    # rhetorical-question shape, same lack of any actual reason given.
    assert _is_low_content_headline("Is Exxon Mobil (XOM) Still Attractive After A 49% One Year Share Price Surge?")
    assert _is_low_content_headline("Is Tesla Still a Buy After Its Recent Rally?")


def test_does_not_catch_is_questions_about_non_stock_topics():
    assert not _is_low_content_headline("Federal Reserve Is Still Fighting Inflation After Latest CPI Data")


def test_does_not_catch_genuine_is_x_still_after_analysis_headline():
    # Regression: an early version of the "^Is ... Still ... After ..."
    # bait pattern matched ANY such shape regardless of what the "After"
    # clause was actually about - this genuine macro-analysis headline
    # (found via this file's own adversarial check while building the
    # fix, not a live report) false-positived on that version. The final
    # pattern requires the After-clause to itself contain a price-move
    # signal (%, rally/surge/drop/etc.) - this headline has none.
    assert not _is_low_content_headline(
        "Is the Federal Reserve Still Fighting Inflation After the Latest CPI Report Showed a Surprise Uptick"
    )
    assert not _is_low_content_headline(
        "Is Apple Still Investing Heavily in AI After Its Latest Earnings Call Detailed Capex Plans"
    )


def test_does_not_catch_after_percent_move_declarative_variant_known_gap():
    # KNOWN GAP, deliberately unfixed (see generate_real_dataset.py's
    # module docstring, 2026-08-20 history note) - the mid-sentence
    # declarative form of the same bait ("X Stock After N% Gain Is The
    # Price Still Reasonable") isn't anchored the same way as the "^Is"
    # question form, and a safe narrow pattern for it wasn't obvious
    # without more real examples. This test documents the gap so a
    # future fix attempt has a concrete target, not so it stays broken
    # forever - remove/flip this assertion once that pattern is added.
    assert not _is_low_content_headline(
        "JPMorgan Chase (JPM) Stock After 25% Yearly Gain Is The Price Still Reasonable"
    )


# --- _is_relevant_headline ---


def test_relevant_matches_company_name():
    assert _is_relevant_headline("ORCL", "Oracle", None, "Oracle cuts cloud guidance")


def test_relevant_matches_ticker_as_standalone_token():
    assert _is_relevant_headline("NVDA", "Nvidia", None, "NVDA jumps on datacenter demand")


def test_relevant_does_not_match_ticker_as_lowercase_substring():
    # "MA" (Mastercard) must not match inside ordinary words like "market"
    assert not _is_relevant_headline("MA", "Mastercard", "Financial Services", "market rally continues")


def test_relevant_matches_sector_keyword_without_naming_company():
    assert _is_relevant_headline("XOM", "Exxon Mobil", "Energy", "OPEC agrees to cut oil output")


def test_relevant_sector_keyword_uses_word_boundaries():
    # "oil" must not match inside "turmoil", "gas" must not match inside "Vegas"
    assert not _is_relevant_headline("XOM", "Exxon Mobil", "Energy", "Markets face turmoil in Las Vegas")


def test_relevant_rejects_unrelated_macro_news():
    assert not _is_relevant_headline("ORCL", "Oracle", "Technology", "Fed leaves interest rates unchanged")


def test_relevant_unlisted_sector_falls_back_to_name_ticker_only():
    assert not _is_relevant_headline("O", "Realty Income", "Real Estate", "Mortgage rates tick higher")
