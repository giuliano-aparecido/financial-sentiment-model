"""
RunPod (or any plain GPU box) copy of ../colab/train/gpu/evaluate_model.py.
Same evaluation logic, byte-identical task_a_prompt/task_b_prompt (see
CONTRIBUTING.md's sync rule - covers this file too).

Two differences from the Colab version:
1. get_secret() reads plain environment variables directly, no Colab/
   Kaggle secrets-vault branching - see runpod/train_model.py's own
   get_secret() for the same reasoning.
2. No "reuse model/tokenizer already in memory" check - that only makes
   sense in a notebook session where a training cell might have already
   run earlier in the SAME kernel. A standalone script run
   (`python evaluate_model.py`) is always a fresh process, so this always
   reloads the already-trained, already-pushed model fresh from Hugging
   Face - simpler than carrying a branch that can never actually take the
   other path here.

What it measures - the metric that actually matters, which token loss
doesn't (see docs/training-results-analysis.md):
  - recommendation accuracy (BUY/SELL/HOLD correct or not), overall
    and split by val source (synthetic vs real) and by class (confusion
    matrix per source, to spot "always answers HOLD"-style bias and
    whether it's source-specific)
  - JSON validity rate of the model's raw output
  - the same numbers for the BASE model (LoRA adapter temporarily
    disabled) as the missing baseline - the base-vs-tuned delta is the
    true value of the whole training pipeline. Set SKIP_BASE_MODEL_EVAL=1
    to skip this pass once you've already established that baseline once
    (see its own comment near Pass 2 below) - roughly halves runtime and,
    on RunPod, the GPU-billing cost of a routine eval run.

Runtime expectation: ~160 rows x 2 passes, one generation each - roughly
15-30 minutes on a T4-class GPU. Progress prints every 20 rows. The TUNED
pass runs FIRST so that if the run dies partway, the number you care most
about is already printed.

Usage (same pod/session you trained in, or a fresh one - either way, this
reloads the pushed model from Hugging Face rather than needing anything
left over in memory):

    export HF_TOKEN=...
    export HF_USER=...
    # export SKIP_BASE_MODEL_EVAL=1   # optional, see above
    # upload dataset_val.jsonl and dataset_val_real.jsonl into this same
    # working directory (train files aren't needed for eval)
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes
    python evaluate_model.py
"""

import json
import os
import random
import re

import torch
from unsloth import FastLanguageModel

# Rows to sample per val source. Real val's actual row count depends on
# how many headlines generate_real_dataset.py's non-deterministic fetch
# turned up for VAL_HOLDOUT_TICKERS this run (widened to 6 tickers - was
# 2 - specifically so this sample draws from more than one or two
# companies' idiosyncratic news cycle; see that constant's own comment).
# None = evaluate every row (slower: 657 synthetic rows).
EVAL_SAMPLE_PER_SOURCE = 100

# How many full generations to keep and print per (source, expected,
# predicted) wrong-answer combination - the confusion matrix says WHAT
# went wrong, this shows WHAT THE MODEL ACTUALLY WROTE for a handful of
# those rows (ticker, headlines, full raw output), for cases where the
# aggregate numbers alone don't explain a pattern (e.g. a class collapse
# that isn't a straightforward data-imbalance artifact).
SAMPLE_MISCLASSIFICATIONS_PER_PAIR = 3


def get_secret(name):
    """Plain environment variable - see this file's own docstring for
    where to set these before running. No Colab/Kaggle fallback needed
    here (unlike ../colab/train/gpu/evaluate_model.py's version), since
    this file is specifically for environments that have neither."""
    return os.environ.get(name)


print("Reloading the already-trained, already-pushed model from Hugging Face...")

# MODEL_CHOICE_DEFAULT is the git-committed baseline. Set an OPTIONAL
# MODEL_CHOICE environment variable to reload a different base-model
# family ad-hoc, without editing this file - must match whatever
# MODEL_CHOICE the target HF_REPO was actually trained/pushed under.
MODEL_CHOICE_DEFAULT = "llama-3.1-8b"
MODEL_CHOICE = get_secret("MODEL_CHOICE") or MODEL_CHOICE_DEFAULT

MAX_SEQ_LENGTH = 2048
HF_USER = get_secret("HF_USER")

# MODEL_VERSION_DEFAULT is the git-committed baseline (bumped by
# bump_model_version.py). Set an OPTIONAL MODEL_VERSION environment
# variable to reload a different push ad-hoc, without editing this file.
MODEL_VERSION_DEFAULT = "v1"
MODEL_VERSION = get_secret("MODEL_VERSION") or MODEL_VERSION_DEFAULT

HF_REPO = f"{HF_USER}/{MODEL_CHOICE}-financial-reasoner-{MODEL_VERSION}"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=HF_REPO,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Must match ../colab/train/gpu/train_model.py's task_a_prompt/
# task_b_prompt exactly (see CONTRIBUTING.md's sync rule) - these are
# separate copies because a reload means the training run's own copies
# never ran in this process.
task_a_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

Classify how the market has reacted to the following news for this stock, given its recent price move, and output JSON containing your classification, in exactly this shape:
{{"news_reaction": "good|bad|neutral|overreaction_down|overreaction_up"}}

news_reaction definitions:
- good: the news is genuinely positive for the stock.
- bad: the news is genuinely negative for the stock.
- neutral: the news is routine/ambiguous, not a real catalyst either way.
- overreaction_down: the price fell more than this news alone would justify - a plausible overreaction to the downside.
- overreaction_up: the price rose more than this news alone would justify - a plausible overreaction to the upside.

### Input:

Target Stock: {}
Recent Price Move: {}

Recent News & Results:
{}

### Response:

{}"""

task_b_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

You are given a recommended action for this stock, already determined from valuation and news analysis - your job is to explain it, not decide it. Output JSON containing detailed reasoning and a direct answer to the user's question, in exactly this shape:
{{"reasoning": "...", "answer": "..."}}

Your reasoning and answer must be consistent with the Recommended Action below and must never advise the opposite. Treat News Reaction and Recommended Action as given facts, not conclusions to re-derive.

### Input:

Target Stock: {}
User Question: {}
News Reaction: {}
Recommended Action: {}

Current Market Data:
{}

Valuation:
{}

Recent Earnings:
{}

Recent News & Results:
{}

### Response:

{}"""

FastLanguageModel.for_inference(model)  # unsloth's fast-generation mode

VAL_FILES = {"synthetic": "dataset_val.jsonl", "real": "dataset_val_real.jsonl"}
# Optional - only present if convert_existing_to_taskb.py or a GATE-A real
# regen produced it (see train_model.py's own comment on this pair).
if os.path.exists("dataset_val_real_taskb.jsonl"):
    VAL_FILES["real_taskb"] = "dataset_val_real_taskb.jsonl"

REACTION_RE = re.compile(r'"news_reaction"\s*:\s*"(good|bad|neutral|overreaction_down|overreaction_up)"')
# Not used for accuracy scoring, just to surface the model's generated
# text in the misclassification/flagged-sample dumps below.
ANSWER_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"')
REASONING_RE = re.compile(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"')

# Crude but effective direction-consistency check for Task B: given a
# GIVEN recommendation (fed to the model as input, never predicted by
# it), flag generated prose that explicitly advises the OPPOSITE action -
# the one failure mode task_b_prompt's own instruction explicitly
# forbids. Not a full correctness check (an evasive or vacuous answer
# would pass this too), just the sharpest, cheapest signal available
# without a second model-as-judge call.
#
# Revised 2026-08-20 (v1 fine-tune eval): the original version was a
# bare \b(word)\b search with no negation awareness and no allowance
# for common hyphenated finance jargon - confirmed live, EVERY ONE of a
# batch of 19 rows a real eval run flagged turned out to be a false
# positive on manual read-through, from two bug classes:
#   1. \b treats a hyphen (and a plain space) as a word boundary, so
#      "sell-out"/"sell out"/"sell-off"/"short-term"/"buy-in" (all
#      completely benign, common finance vocabulary) matched the bare
#      trigger word inside them - fixed by requiring "short"/"shorting"
#      appear with an actual object (short(?:ing)? the stock/it/shares/
#      a position) instead of as a bare word at all (this alone removes
#      "short-term"/"short report"/"in short" as false-trigger sources),
#      and by a negative lookahead (_BENIGN_COMPOUND_SUFFIX_RE) excluding
#      the sell/buy compounds.
#   2. No negation/hedge handling - "doesn't look like a buy", "selling
#      now may be premature", and the model's own "...which runs counter
#      to a SELL outlook. However, the valuation..." pivot phrasing (the
#      model correctly acknowledging a conflicting individual signal
#      before reconciling toward the given recommendation - sophisticated,
#      CORRECT reasoning, not a contradiction) were all flagged despite
#      not actually advising the opposite action. _NEGATION_CUE_RE,
#      checked in a window around each match (not just before it - some
#      of these cues land after, e.g. "selling now may be premature"),
#      addresses this.
# Verified empirically against all 19 original false positives (now 0/19
# flagged) AND five constructed genuine-contradiction cases (still 5/5
# caught) before landing this - see the PR description/commit for the
# actual test script, not committed here since it's a one-off validation
# aid, not part of the eval pipeline itself.
_BENIGN_COMPOUND_SUFFIX_RE = r"(?![\s-](?:out|off|side|in)\b)"
_NEGATION_CUE_RE = re.compile(
    r"n['’]t\b|\b(?:not|no|never|rather than|instead of|more attractive than|"
    r"premature|avoid|against|however|but|despite|nevertheless|nonetheless|"
    r"missed opportunity|counter to|weak (?:or indirect )?signal|indirect signal)\b",
    re.IGNORECASE,
)
_OPPOSITE_ACTION_WORDS = {
    "BUY": re.compile(
        rf"\b(sell|selling|exit|take profits?)\b{_BENIGN_COMPOUND_SUFFIX_RE}"
        r"|\bshort(?:ing)?\b\s+(?:the stock|it|shares|a position)\b"
        r"|\bgo short\b|\binitiate a short\b|\bshort position\b",
        re.IGNORECASE,
    ),
    "SELL": re.compile(
        rf"\b(buy|buying|accumulate|accumulating|add(?:ing)? (?:shares|to (?:a |the )?position)|initiate a (?:buy|long) position)\b{_BENIGN_COMPOUND_SUFFIX_RE}",
        re.IGNORECASE,
    ),
    "HOLD": None,  # HOLD has no single "opposite" action to flag against
}


def _has_opposite_action_language(text: str, opposite_re, window: int = 100) -> bool:
    """True if `text` contains real opposite-action language per
    `opposite_re` - a match with no negation/hedge cue (n't, not,
    however, counter to, etc. - see _NEGATION_CUE_RE) within `window`
    characters on either side. See _OPPOSITE_ACTION_WORDS' own comment
    for the false-positive classes this fixes and why the window checks
    BOTH sides, not just the text before a match."""
    for match in opposite_re.finditer(text):
        span = text[max(0, match.start() - window):match.end() + window]
        if _NEGATION_CUE_RE.search(span):
            continue
        return True
    return False

random.seed(42)
task_a_rows, task_b_rows = [], []
for source, path in VAL_FILES.items():
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    for row in rows:
        row["_source"] = source
    a_rows = [r for r in rows if r["task"] == "reaction"]
    b_rows = [r for r in rows if r["task"] == "analysis"]
    if EVAL_SAMPLE_PER_SOURCE is not None:
        if len(a_rows) > EVAL_SAMPLE_PER_SOURCE:
            a_rows = random.sample(a_rows, EVAL_SAMPLE_PER_SOURCE)
        if len(b_rows) > EVAL_SAMPLE_PER_SOURCE:
            b_rows = random.sample(b_rows, EVAL_SAMPLE_PER_SOURCE)
    task_a_rows.extend(a_rows)
    task_b_rows.extend(b_rows)

print(f"Task A eval: {len(task_a_rows)} rows across {list(VAL_FILES)}. "
      f"Task B eval: {len(task_b_rows)} rows across {list(VAL_FILES)}.")


def is_valid_json(text):
    # The output should be a JSON object; grab the outermost braces.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return False
    try:
        json.loads(text[start:end + 1])
        return True
    except json.JSONDecodeError:
        return False


def generate(prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy = deterministic, comparable across passes
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def run_task_a_eval(label):
    per_source = {s: {"total": 0, "correct": 0, "json_ok": 0} for s in VAL_FILES}
    confusion = {s: {} for s in VAL_FILES}  # source -> (expected, predicted-or-None) -> count
    samples = {s: {} for s in VAL_FILES}  # source -> (expected, predicted-or-None) -> [(row, raw_text), ...]

    for i, row in enumerate(task_a_rows, 1):
        prompt = task_a_prompt.format(row["ticker"], row["price_context"], row["news"], "")
        text = generate(prompt, max_new_tokens=48)
        exp = row["news_reaction"]
        m = REACTION_RE.search(text)
        pred = m.group(1) if m else None
        source = row["_source"]

        stats = per_source[source]
        stats["total"] += 1
        is_correct = (pred == exp)
        stats["correct"] += is_correct
        stats["json_ok"] += is_valid_json(text)
        confusion[source][(exp, pred)] = confusion[source].get((exp, pred), 0) + 1

        if not is_correct:
            bucket = samples[source].setdefault((exp, pred), [])
            if len(bucket) < SAMPLE_MISCLASSIFICATIONS_PER_PAIR:
                bucket.append((row, text))

        if i % 20 == 0:
            done_correct = sum(s["correct"] for s in per_source.values())
            print(f"  [{label}/TaskA] {i}/{len(task_a_rows)} rows, running accuracy {done_correct / i:.0%}")

    print(f"\n===== {label} - Task A (news_reaction classification) =====")
    total = sum(s["total"] for s in per_source.values())
    correct = sum(s["correct"] for s in per_source.values())
    json_ok = sum(s["json_ok"] for s in per_source.values())
    if total:
        print(f"Overall accuracy: {correct}/{total} = {correct / total:.1%} (random baseline for 5 classes: 20%)")
        print(f"Valid-JSON rate:  {json_ok}/{total} = {json_ok / total:.1%}")
    for source, s in per_source.items():
        if s["total"]:
            print(f"  {source:<12} accuracy: {s['correct']}/{s['total']} = {s['correct'] / s['total']:.1%}")

    class_totals, class_correct = {}, {}
    for source_confusion in confusion.values():
        for (exp, pred), count in source_confusion.items():
            class_totals[exp] = class_totals.get(exp, 0) + count
            if pred == exp:
                class_correct[exp] = class_correct.get(exp, 0) + count
    print("Per-class recall:")
    for cls in ("good", "bad", "neutral", "overreaction_down", "overreaction_up"):
        t = class_totals.get(cls, 0)
        c = class_correct.get(cls, 0)
        if t:
            print(f"  {cls:<18} {c}/{t} = {c / t:.1%}")

    for source, source_confusion in confusion.items():
        if not source_confusion:
            continue
        print(f"Confusion for {source} (expected -> predicted: count):")
        # sorted() on the raw (exp, pred) tuple crashes with TypeError the
        # first time any row is genuinely unparseable (pred=None) - Python
        # can't order None against a str. key= treats None as "" so it
        # sorts first (before any real class) instead of crashing.
        for (exp, pred), count in sorted(source_confusion.items(), key=lambda item: (item[0][0], item[0][1] or "")):
            print(f"  {exp:<18} -> {pred or 'UNPARSEABLE':<18} {count}")
    print()

    print(f"----- Sample Task A misclassifications for {label} (up to {SAMPLE_MISCLASSIFICATIONS_PER_PAIR} per source/expected/predicted) -----")
    for source, source_samples in samples.items():
        for (exp, pred), rows in sorted(source_samples.items(), key=lambda item: (item[0][0], item[0][1] or "")):
            for row, text in rows:
                print(f"\n[{source}] expected {exp} -> predicted {pred or 'UNPARSEABLE'} | ticker={row['ticker']}")
                print(f"Price context: {row['price_context']}")
                print(f"News:\n{row['news']}")
                print(f"Model output:\n{text}")
    print()
    return per_source


def run_task_b_eval(label):
    per_source = {s: {"total": 0, "json_ok": 0, "nonempty": 0, "consistent": 0} for s in VAL_FILES}
    flagged_samples = []

    for i, row in enumerate(task_b_rows, 1):
        prompt = task_b_prompt.format(
            row["ticker"], row["user_query"], row["news_reaction"], row["recommendation"],
            row["market_data"], row["valuation"], row["earnings"], row["news"], "",
        )
        text = generate(prompt, max_new_tokens=512)
        source = row["_source"]
        stats = per_source[source]
        stats["total"] += 1
        stats["json_ok"] += is_valid_json(text)

        answer_m = ANSWER_RE.search(text)
        reasoning_m = REASONING_RE.search(text)
        answer = answer_m.group(1) if answer_m else ""
        reasoning = reasoning_m.group(1) if reasoning_m else ""
        if answer.strip() and reasoning.strip():
            stats["nonempty"] += 1

        opposite_re = _OPPOSITE_ACTION_WORDS.get(row["recommendation"])
        contradicts = bool(opposite_re and (_has_opposite_action_language(answer, opposite_re) or _has_opposite_action_language(reasoning, opposite_re)))
        if not contradicts:
            stats["consistent"] += 1
        elif len(flagged_samples) < 20:
            flagged_samples.append((row, text))

        if i % 20 == 0:
            print(f"  [{label}/TaskB] {i}/{len(task_b_rows)} rows")

    print(f"\n===== {label} - Task B (reasoning/answer generation) =====")
    total = sum(s["total"] for s in per_source.values())
    json_ok = sum(s["json_ok"] for s in per_source.values())
    nonempty = sum(s["nonempty"] for s in per_source.values())
    consistent = sum(s["consistent"] for s in per_source.values())
    if total:
        print(f"Valid-JSON rate:                  {json_ok}/{total} = {json_ok / total:.1%}")
        print(f"Non-empty reasoning+answer rate:  {nonempty}/{total} = {nonempty / total:.1%}")
        print(f"Direction-consistency rate:       {consistent}/{total} = {consistent / total:.1%} (no explicit opposite-action language)")
    for source, s in per_source.items():
        if s["total"]:
            print(f"  {source:<12} json_ok={s['json_ok']}/{s['total']} consistent={s['consistent']}/{s['total']}")

    if flagged_samples:
        print(f"\n----- Flagged possible direction-contradictions for {label} (up to 20) -----")
        for row, text in flagged_samples:
            print(f"\nGiven recommendation={row['recommendation']} news_reaction={row['news_reaction']} | ticker={row['ticker']}")
            print(f"Model output:\n{text}")
    print()
    return per_source


# Pass 1: the fine-tuned model (adapter active) - the numbers that matter.
tuned_a = run_task_a_eval("FINE-TUNED model")
tuned_b = run_task_b_eval("FINE-TUNED model")

# Pass 2 (base model) roughly doubles total eval time (320 generations
# instead of 160) - worth it the FIRST time you eval a given prompt/schema
# shape, since without it there's no baseline to tell "65% accuracy" apart
# from "would have scored 65% doing nothing" (see
# docs/training-results-analysis.md's "No baseline" section - this pass
# exists specifically to fix that gap for recommendation accuracy, not just
# loss). On RunPod that extra time is real, direct GPU-billing cost
# (unlike free Colab), so once you've established the baseline once, set
# SKIP_BASE_MODEL_EVAL=1 (or "true"/"yes") to skip straight to just the
# tuned-model number on later quick-iteration runs.
SKIP_BASE_MODEL_EVAL = (get_secret("SKIP_BASE_MODEL_EVAL") or "").strip().lower() in ("1", "true", "yes")

if SKIP_BASE_MODEL_EVAL:
    print("SKIP_BASE_MODEL_EVAL set - skipping the base-model comparison pass.")
else:
    # Two full generation passes (320 generations total) back-to-back
    # without clearing cache in between is a real way to run a GPU out of
    # memory partway through pass 2 - this happened live during
    # development on a free Colab T4. Free whatever's reclaimable before
    # starting the second pass.
    torch.cuda.empty_cache()

    # Base repos for the fallback path below - mirrors
    # ../colab/train/gpu/train_model.py's MODEL_REGISTRY "repo" field per
    # MODEL_CHOICE (see CONTRIBUTING.md's sync rule; keep in sync with
    # that file, not just task_a_prompt/task_b_prompt).
    BASE_MODEL_REPO_BY_CHOICE = {
        "llama-3.2-3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "llama-3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "apertus-8b": "swiss-ai/Apertus-8B-Instruct-2509",
        "apertus-0.5b": "swiss-ai/Apertus-v1.1-0.5B-Instruct",
        "qwen-2.5-7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "mistral-7b": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    }

    # Pass 2: the base model. Prefers temporarily disabling the LoRA
    # adapter on the already-loaded model (no second download, no extra
    # memory) - works when `model` is a plain peft-wrapped object with
    # disable_adapter() available. Confirmed live (on the Colab version of
    # this script): this can fail with "'LlamaForCausalLM' object has no
    # attribute 'disable_adapter'" - some unsloth code path can
    # return/transform the model into something that no longer exposes
    # peft's disable_adapter() context manager. Falls back to loading a
    # genuinely separate, adapter-free base model instance when
    # disable_adapter() isn't available - costs a second download/load,
    # but doesn't depend on guessing which unsloth-internal transformation
    # caused the first approach to fail.
    try:
        with model.disable_adapter():
            base_a = run_task_a_eval("BASE model (adapter disabled)")
            base_b = run_task_b_eval("BASE model (adapter disabled)")
    except Exception as e:
        print(f"model.disable_adapter() unavailable/failed ({e!r}) - loading a "
              f"separate, genuinely adapter-free base model instance instead...")
        try:
            _base_repo = BASE_MODEL_REPO_BY_CHOICE[MODEL_CHOICE]

            base_model, base_tokenizer = FastLanguageModel.from_pretrained(
                model_name=_base_repo,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(base_model)

            # generate()/run_task_{a,b}_eval() close over the module-level
            # `model`/`tokenizer` names, looked up at call time - swap them
            # to the base model for this pass, then restore, so nothing
            # below silently ends up with the base model instead of the
            # fine-tuned one.
            _tuned_model, _tuned_tokenizer = model, tokenizer
            model, tokenizer = base_model, base_tokenizer
            try:
                base_a = run_task_a_eval("BASE model (separately loaded, no adapter)")
                base_b = run_task_b_eval("BASE model (separately loaded, no adapter)")
            finally:
                model, tokenizer = _tuned_model, _tuned_tokenizer
        except Exception as e2:
            print(f"Base-model fallback load also failed ({e2!r}) - tuned results "
                  "above still stand. Fallback: rerun this script, or share this error.")
