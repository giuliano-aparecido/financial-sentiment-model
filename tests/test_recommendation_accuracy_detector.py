"""Tests for the Task B recommendation-accuracy regex (RECOMMENDATION_RE),
duplicated byte-identically across 5 eval scripts (colab/train/{gpu,tpu}/
evaluate_model.py, their evaluate_base_model_only.py siblings, runpod/
evaluate_model.py - see CONTRIBUTING.md's sync convention). Replaces
test_direction_consistency_detector.py: that file tested the old
_has_opposite_action_language/_OPPOSITE_ACTION_WORDS heuristic, which
existed only because Task B was handed a recommendation and merely had to
avoid contradicting it. Task B now decides its own recommendation (see
docs/task-b-learned-recommendation-plan.md), so eval instead parses that
recommendation straight out of the model's JSON and compares it to
fuse()'s ground truth - RECOMMENDATION_RE is the parsing half of that.

None of those scripts are directly importable: they open with Colab `!pip
install` magic lines and depend on unsloth/torch/a live loaded model at
import time. So each test here reads the source file as text and extracts
just the RECOMMENDATION_RE line (no external deps beyond the stdlib `re`
module already used there) - this exercises the REAL regex that ships in
each file, not a hand-copied re-implementation that could silently drift
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


def _load_recommendation_re(path):
    """Extracts the `RECOMMENDATION_RE = re.compile(...)` line out of
    `path` and evals just that expression - the same isolation approach
    test_direction_consistency_detector.py used for its (larger) block."""
    with open(path, encoding="utf-8") as f:
        line = next(l for l in f if l.startswith("RECOMMENDATION_RE"))
    _, _, expr = line.partition("=")
    return eval(expr.strip(), {"re": re})


@pytest.fixture(params=EVAL_SCRIPT_PATHS, ids=lambda p: p)
def recommendation_re(request):
    return _load_recommendation_re(request.param)


@pytest.mark.parametrize("action", ["BUY", "SELL", "HOLD"])
def test_extracts_recommendation_from_model_json(recommendation_re, action):
    text = f'{{"recommendation": "{action}", "reasoning": "...", "answer": "..."}}'
    m = recommendation_re.search(text)
    assert m is not None
    assert m.group(1) == action


def test_no_match_on_unparseable_output(recommendation_re):
    assert recommendation_re.search("not json at all, no braces here") is None


def test_no_match_on_invalid_recommendation_class(recommendation_re):
    # Pre-migration model checkpoints (or a badly hallucinating one) might
    # emit an old-vocabulary or made-up class - must not match, so eval
    # correctly counts it as UNPARSEABLE rather than a wrong guess.
    assert recommendation_re.search('{"recommendation": "STRONG_BUY", "reasoning": "..."}') is None


def test_matches_regardless_of_surrounding_whitespace(recommendation_re):
    text = '{\n  "recommendation" :  "SELL" ,\n  "reasoning": "..."\n}'
    m = recommendation_re.search(text)
    assert m is not None
    assert m.group(1) == "SELL"
