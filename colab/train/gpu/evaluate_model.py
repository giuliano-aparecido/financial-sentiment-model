!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes

# Direction-accuracy evaluation - paste as ONE Colab cell. Self-contained:
# the two pip installs above match gpu/train_model.py's exactly - PR #16
# made this script reload the model from HF instead of requiring the
# training cell's variables in memory, but stopped short of installing its
# own dependencies too, so a genuinely fresh session (no training cell run
# at all this session) still hit ModuleNotFoundError on `unsloth` even
# after that fix. These installs are safe to re-run if the training cell
# already ran in this session too - pip just no-ops on an already-
# satisfied requirement. Works whether the previous session is still alive
# (reuses `model`,
# `tokenizer`, `alpaca_prompt` already in memory - the normal case,
# right after the training cell) or crashed/expired (reloads the
# finished, already-pushed model fresh from Hugging Face - e.g. re-running
# this cell after a prior run of THIS SAME script crashed partway through,
# such as the confusion-matrix sort TypeError this eval script used to hit
# on any row with an unparseable model output; that crash happened after
# training had already finished and pushed, so there was nothing left to
# retrain - only this cell needed re-running). Either way, needs the
# dataset_val*.jsonl files present on disk (regenerate
# generate_synthetic_dataset.py/generate_real_dataset.py first if this is
# a fresh session that doesn't have them).
#
# What it measures - the metric that actually matters, which token loss
# doesn't (see docs/training-results-analysis.md):
#   - direction accuracy (BULLISH/BEARISH/NEUTRAL correct or not), overall
#     and split by val source (synthetic vs real) and by class (confusion
#     matrix per source, to spot "always answers NEUTRAL"-style bias and
#     whether it's source-specific)
#   - JSON validity rate of the model's raw output
#   - the same numbers for the BASE model (LoRA adapter temporarily
#     disabled) as the missing baseline - the base-vs-tuned delta is the
#     true value of the whole training pipeline.
#
# Runtime expectation: ~160 rows x 2 passes, one generation each - roughly
# 15-30 minutes on a T4. Progress prints every 20 rows. The TUNED pass runs
# FIRST so that if the session dies partway, the number you care most about
# is already printed.

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


# Checking whether `google.colab` IMPORTS is not a reliable way to detect
# Colab vs Kaggle - some Kaggle base images ship a google-colab package
# too, so the import succeeds there and userdata.get() just hangs and
# times out instead of raising. KAGGLE_KERNEL_RUN_TYPE is set by Kaggle's
# own runtime on every notebook - check that directly instead.
def get_secret(name):
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    from google.colab import userdata
    return userdata.get(name)


try:
    model, tokenizer, alpaca_prompt
    print("Reusing model already in memory - skipping reload.")
except NameError:
    print("Model/tokenizer not in memory (new or crashed session) - "
          "reloading the already-trained, already-pushed model from "
          "Hugging Face...")

    # MODEL_CHOICE_DEFAULT is the git-committed baseline. Add an OPTIONAL
    # "MODEL_CHOICE" Colab/Kaggle Secret to reload a different base-model
    # family ad-hoc, without editing this file - must match whatever
    # MODEL_CHOICE the target HF_REPO was actually trained/pushed under.
    MODEL_CHOICE_DEFAULT = "llama-3.2-3b"
    try:
        MODEL_CHOICE = get_secret("MODEL_CHOICE") or MODEL_CHOICE_DEFAULT
    except Exception:
        MODEL_CHOICE = MODEL_CHOICE_DEFAULT

    MAX_SEQ_LENGTH = 2048
    HF_USER = get_secret("HF_USER")

    # MODEL_VERSION_DEFAULT is the git-committed baseline (bumped by
    # bump_model_version.py). Add an OPTIONAL "MODEL_VERSION" Colab/Kaggle
    # Secret to reload a different push ad-hoc, without editing this file.
    MODEL_VERSION_DEFAULT = "v12"
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

    # Must match gpu/train_model.py's alpaca_prompt exactly (see
    # CONTRIBUTING.md's 4-way sync rule) - this is a separate copy because
    # a reload means the training cell's own copy never ran in this
    # session.
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

Analyze the following financial data and news and output JSON containing the impacted stock ticker, detailed reasoning, directional sentiment (BULLISH/BEARISH/NEUTRAL), confidence score, and a direct answer to the user's question.

CRITICAL SENTIMENT RULES:

1. Weigh guidance cuts and revenue misses higher than minor operational wins.

### Input:

Target Stock: {}
User Question: {}

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
DIRECTION_RE = re.compile(r'"direction"\s*:\s*"(BULLISH|BEARISH|NEUTRAL)"')
# Not used for accuracy scoring (direction is), just to surface the model's
# generated answer text in the misclassification dump below so answer
# quality can be eyeballed alongside the direction miss.
ANSWER_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"')

random.seed(42)
eval_rows = []
for source, path in VAL_FILES.items():
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    for row in rows:
        row["_source"] = source
    if EVAL_SAMPLE_PER_SOURCE is not None and len(rows) > EVAL_SAMPLE_PER_SOURCE:
        rows = random.sample(rows, EVAL_SAMPLE_PER_SOURCE)
    eval_rows.extend(rows)

print(f"Evaluating on {len(eval_rows)} val rows "
      f"({sum(1 for r in eval_rows if r['_source'] == 'synthetic')} synthetic, "
      f"{sum(1 for r in eval_rows if r['_source'] == 'real')} real)")


def expected_direction(row):
    return json.loads(row["output"])["impacted_stocks"][0]["direction"]


def generate_response(row):
    # Same template as training, with the response slot left empty - the
    # prompt ends at "### Response:\n\n" and the model continues from there.
    prompt = alpaca_prompt.format(
        row["ticker"], row["user_query"], row["market_data"],
        row["valuation"], row["earnings"], row["news"], "",
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,  # v4's `answer` field adds length beyond the
                                  # old 300-token cap, which was already
                                  # keyed to a 4-field JSON output.
            do_sample=False,  # greedy = deterministic, comparable across passes
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


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


def run_eval(label):
    per_source = {s: {"total": 0, "correct": 0, "json_ok": 0} for s in VAL_FILES}
    confusion = {s: {} for s in VAL_FILES}  # source -> (expected, predicted-or-None) -> count
    samples = {s: {} for s in VAL_FILES}  # source -> (expected, predicted-or-None) -> [(row, raw_text), ...]

    for i, row in enumerate(eval_rows, 1):
        text = generate_response(row)
        exp = expected_direction(row)
        m = DIRECTION_RE.search(text)
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
            print(f"  [{label}] {i}/{len(eval_rows)} rows, running accuracy {done_correct / i:.0%}")

    print(f"\n===== {label} =====")
    total = sum(s["total"] for s in per_source.values())
    correct = sum(s["correct"] for s in per_source.values())
    json_ok = sum(s["json_ok"] for s in per_source.values())
    print(f"Overall direction accuracy: {correct}/{total} = {correct / total:.1%}")
    print(f"Valid-JSON rate:            {json_ok}/{total} = {json_ok / total:.1%}")
    for source, s in per_source.items():
        if s["total"]:
            print(f"  {source:<10} accuracy: {s['correct']}/{s['total']} = {s['correct'] / s['total']:.1%}")
    for source, source_confusion in confusion.items():
        if not source_confusion:
            continue
        print(f"Confusion for {source} (expected -> predicted: count):")
        # sorted() on the raw (exp, pred) tuple crashes with TypeError the
        # first time any row is genuinely unparseable (pred=None) - Python
        # can't order None against a str. key= treats None as "" so it
        # sorts first (before any real direction) instead of crashing.
        for (exp, pred), count in sorted(source_confusion.items(), key=lambda item: (item[0][0], item[0][1] or "")):
            print(f"  {exp:<8} -> {pred or 'UNPARSEABLE':<12} {count}")
    print()

    print(f"----- Sample misclassifications for {label} (up to {SAMPLE_MISCLASSIFICATIONS_PER_PAIR} per source/expected/predicted) -----")
    for source, source_samples in samples.items():
        # Same None-vs-str sort gotcha as the confusion matrix above.
        for (exp, pred), rows in sorted(source_samples.items(), key=lambda item: (item[0][0], item[0][1] or "")):
            for row, text in rows:
                m = ANSWER_RE.search(text)
                answer = m.group(1) if m else "(no answer field found)"
                print(f"\n[{source}] expected {exp} -> predicted {pred or 'UNPARSEABLE'} | ticker={row['ticker']}")
                print(f"User Question: {row['user_query']}")
                print(f"News:\n{row['news']}")
                print(f"Model output:\n{text}")
                print(f"Answer: {answer}")
    print()
    return per_source


# Pass 1: the fine-tuned model (adapter active) - the number that matters.
tuned = run_eval("FINE-TUNED model")

# Two full generation passes (320 generations total) back-to-back on a free
# T4 without clearing cache in between is a real way to run a session out of
# GPU memory partway through pass 2 - this happened live during development.
# Free whatever's reclaimable before starting the second pass.
torch.cuda.empty_cache()

# Base repos for the fallback path below - mirrors gpu/train_model.py's
# MODEL_REGISTRY "repo" field per MODEL_CHOICE (see CONTRIBUTING.md's sync
# rule; keep in sync with that file, not just the alpaca_prompt).
BASE_MODEL_REPO_BY_CHOICE = {
    "llama-3.2-3b": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "apertus-8b": "swiss-ai/Apertus-8B-Instruct-2509",
    "apertus-0.5b": "swiss-ai/Apertus-v1.1-0.5B-Instruct",
    "qwen-2.5-7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "mistral-7b": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
}

# Pass 2: the base model. Prefers temporarily disabling the LoRA adapter on
# the already-loaded model (no second download, no extra memory) - works
# when `model` is a plain peft-wrapped object with disable_adapter()
# available. Confirmed live: this can fail with
# "'LlamaForCausalLM' object has no attribute 'disable_adapter'" - some
# unsloth code path (FastLanguageModel.for_inference(), or how it loads an
# adapter-only repo) can return/transform the model into something that no
# longer exposes peft's disable_adapter() context manager, and this
# apparently happened on every run this session (no base-model pass had
# completed before this fix - earlier attempts were masked by other
# crashes, like the confusion-matrix sort TypeError, that happened first
# and prevented execution from ever reaching this point). Falls back to
# loading a genuinely separate, adapter-free base model instance when
# disable_adapter() isn't available - costs a second download/load, but
# doesn't depend on guessing which unsloth-internal transformation caused
# the first approach to fail.
try:
    with model.disable_adapter():
        base = run_eval("BASE model (adapter disabled)")
except Exception as e:
    print(f"model.disable_adapter() unavailable/failed ({e!r}) - loading a "
          f"separate, genuinely adapter-free base model instance instead...")
    try:
        _model_choice = globals().get("MODEL_CHOICE", "llama-3.2-3b")
        _max_seq_length = globals().get("MAX_SEQ_LENGTH", 2048)
        _base_repo = BASE_MODEL_REPO_BY_CHOICE[_model_choice]

        base_model, base_tokenizer = FastLanguageModel.from_pretrained(
            model_name=_base_repo,
            max_seq_length=_max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(base_model)

        # run_eval()/generate_response() close over the module-level
        # `model`/`tokenizer` names, looked up at call time - swap them to
        # the base model for this pass, then restore, so nothing below (or
        # a later cell reusing `model`) silently ends up with the base
        # model instead of the fine-tuned one.
        _tuned_model, _tuned_tokenizer = model, tokenizer
        model, tokenizer = base_model, base_tokenizer
        try:
            base = run_eval("BASE model (separately loaded, no adapter)")
        finally:
            model, tokenizer = _tuned_model, _tuned_tokenizer
    except Exception as e2:
        print(f"Base-model fallback load also failed ({e2!r}) - tuned results "
              "above still stand. Fallback: reload the base model fresh in a "
              "new cell and rerun run_eval manually, or share this error.")
