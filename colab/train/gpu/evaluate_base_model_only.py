!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes

# Paste as ONE Colab cell. Self-contained, including these installs (match
# gpu/train_model.py's exactly, and are safe to re-run if the training
# cell already ran this session - pip no-ops on an already-satisfied
# requirement): works whether the previous session is still alive (reuses
# model/tokenizer already in memory) or crashed (reloads the finished
# model fresh from Hugging Face). Runs ONLY the base-model (adapter-
# disabled) pass - use this when you already have the fine-tuned numbers
# from evaluate_model.py and just need the untrained baseline for
# comparison, without redoing the tuned pass.

import json
import os
import random
import re

import torch
from unsloth import FastLanguageModel


# Checking whether `google.colab` IMPORTS is not a reliable way to detect
# Colab vs Kaggle - some Kaggle base images ship a google-colab package
# too (Kaggle also offers T4/P100 GPUs, so this path can run there too),
# so the import succeeds and userdata.get() just hangs and times out
# instead of raising. KAGGLE_KERNEL_RUN_TYPE is set by Kaggle's own
# runtime on every notebook - check that directly instead.
def get_secret(name):
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    try:
        from google.colab import userdata
        return userdata.get(name)
    except ImportError:
        # Not Colab or Kaggle - e.g. RunPod, or any plain GPU box. Neither
        # has a secrets-vault API to call, so fall back to a real
        # environment variable (set via RunPod's pod env-var config, a
        # .env file, or `export NAME=value` before running this script).
        return os.environ.get(name)


try:
    model, tokenizer
    print("Reusing model already in memory - skipping reload.")
except NameError:
    print("Model not in memory (new or crashed session) - reloading the "
          "already-trained, already-pushed model from Hugging Face...")

    # MODEL_CHOICE_DEFAULT is the git-committed baseline. Add an OPTIONAL
    # "MODEL_CHOICE" Colab/Kaggle Secret to reload a different base-model
    # family ad-hoc, without editing this file - must match whatever
    # MODEL_CHOICE the target HF_REPO was actually trained/pushed under.
    MODEL_CHOICE_DEFAULT = "llama-3.1-8b"
    try:
        MODEL_CHOICE = get_secret("MODEL_CHOICE") or MODEL_CHOICE_DEFAULT
    except Exception:
        MODEL_CHOICE = MODEL_CHOICE_DEFAULT

    MAX_SEQ_LENGTH = 2048
    HF_USER = get_secret("HF_USER")

    # MODEL_VERSION_DEFAULT is the git-committed baseline (bumped by
    # bump_model_version.py). Add an OPTIONAL "MODEL_VERSION" Colab/Kaggle
    # Secret to reload a different push ad-hoc, without editing this file.
    MODEL_VERSION_DEFAULT = "v1"
    try:
        MODEL_VERSION = get_secret("MODEL_VERSION") or MODEL_VERSION_DEFAULT
    except Exception:
        MODEL_VERSION = MODEL_VERSION_DEFAULT

    HF_REPO = f"{HF_USER}/{MODEL_CHOICE}-financial-reasoner-{MODEL_VERSION}"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=HF_REPO,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

FastLanguageModel.for_inference(model)

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

EVAL_SAMPLE_PER_SOURCE = 100
VAL_FILES = {"synthetic": "dataset_val.jsonl", "real": "dataset_val_real.jsonl"}
# Optional - only present if convert_existing_to_taskb.py or a GATE-A real
# regen produced it (see train_model.py's own comment on this pair).
if os.path.exists("dataset_val_real_taskb.jsonl"):
    VAL_FILES["real_taskb"] = "dataset_val_real_taskb.jsonl"

# How many full generations to keep and print per (source, expected,
# predicted) wrong-answer combination - see evaluate_model.py's comment
# above the same constant for why.
SAMPLE_MISCLASSIFICATIONS_PER_PAIR = 3

random.seed(42)
task_a_rows, task_b_rows = [], []
for source, path in VAL_FILES.items():
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    for row in rows:
        row["_source"] = source
    a_rows = [r for r in rows if r["task"] == "reaction"]
    b_rows = [r for r in rows if r["task"] == "analysis"]
    if len(a_rows) > EVAL_SAMPLE_PER_SOURCE:
        a_rows = random.sample(a_rows, EVAL_SAMPLE_PER_SOURCE)
    if len(b_rows) > EVAL_SAMPLE_PER_SOURCE:
        b_rows = random.sample(b_rows, EVAL_SAMPLE_PER_SOURCE)
    task_a_rows.extend(a_rows)
    task_b_rows.extend(b_rows)

print(f"Evaluating base model - Task A: {len(task_a_rows)} rows, Task B: {len(task_b_rows)} rows "
      f"across {list(VAL_FILES)}")


def is_valid_json(text):
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
            do_sample=False,
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


torch.cuda.empty_cache()

# Base repos for the fallback path below - mirrors gpu/train_model.py's
# MODEL_REGISTRY "repo" field per MODEL_CHOICE (see CONTRIBUTING.md's sync
# rule; keep in sync with that file, not just task_a_prompt/task_b_prompt).
BASE_MODEL_REPO_BY_CHOICE = {
    "llama-3.2-3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "llama-3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "apertus-8b": "swiss-ai/Apertus-8B-Instruct-2509",
    "apertus-0.5b": "swiss-ai/Apertus-v1.1-0.5B-Instruct",
    "qwen-2.5-7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "mistral-7b": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
}

# Prefers temporarily disabling the LoRA adapter on the already-loaded
# model (no second download, no extra memory). Confirmed live: this can
# fail with "'LlamaForCausalLM' object has no attribute 'disable_adapter'"
# - some unsloth code path can return/transform the model into something
# that no longer exposes peft's disable_adapter() context manager. Since
# this script's ENTIRE purpose is exactly the "reload fresh, then run
# base-only" scenario most likely to trigger that, an unhandled crash here
# would defeat the script - falls back to loading a genuinely separate,
# adapter-free base model instance instead, which doesn't depend on
# guessing which unsloth-internal transformation caused the first
# approach to fail.
try:
    with model.disable_adapter():
        base_a = run_task_a_eval("BASE model (adapter disabled)")
        base_b = run_task_b_eval("BASE model (adapter disabled)")
except Exception as e:
    print(f"model.disable_adapter() unavailable/failed ({e!r}) - loading a "
          f"separate, genuinely adapter-free base model instance instead...")
    _model_choice = globals().get("MODEL_CHOICE", "llama-3.1-8b")
    _max_seq_length = globals().get("MAX_SEQ_LENGTH", 2048)
    _base_repo = BASE_MODEL_REPO_BY_CHOICE[_model_choice]

    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=_base_repo,
        max_seq_length=_max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base_model)

    # generate()/run_task_{a,b}_eval() close over the module-level
    # `model`/`tokenizer` names, looked up at call time - swap them to the
    # base model for this pass.
    model, tokenizer = base_model, base_tokenizer
    base_a = run_task_a_eval("BASE model (separately loaded, no adapter)")
    base_b = run_task_b_eval("BASE model (separately loaded, no adapter)")
