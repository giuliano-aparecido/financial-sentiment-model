!pip install -q -U transformers peft accelerate

# Paste as ONE Colab cell. Self-contained, including the install above
# (matches tpu/train_model.py's, minus trl/datasets which this script
# doesn't need; torch_xla intentionally excluded - see tpu/evaluate_
# model.py's comment on the same install for why). Safe to re-run if the
# training cell already ran this session. Works whether the previous
# session is still alive (reuses model/tokenizer already in memory) or
# crashed (reloads the finished model fresh from Hugging Face). Runs ONLY
# the base-model (adapter-disabled) pass - use this when you already have
# the fine-tuned numbers from evaluate_model.py and just need the
# untrained baseline for comparison, without redoing the tuned pass.

import json
import os
import random
import re

import torch


# Checking whether `google.colab` IMPORTS is not a reliable way to detect
# Colab vs Kaggle - some Kaggle base images ship a google-colab package
# too, so the import succeeds there and userdata.get() just hangs and
# times out instead of raising. KAGGLE_KERNEL_RUN_TYPE is set by Kaggle's
# own runtime on every notebook - check that directly instead.
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
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    import torch_xla.core.xla_model as xm

    # MODEL_CHOICE_DEFAULT is the git-committed baseline. Add an OPTIONAL
    # "MODEL_CHOICE" Colab/Kaggle Secret to reload a different base-model
    # family ad-hoc, without editing this file - must match whatever
    # MODEL_CHOICE the target HF_REPO was actually trained/pushed under.
    MODEL_CHOICE_DEFAULT = "llama-3.2-3b"
    try:
        MODEL_CHOICE = get_secret("MODEL_CHOICE") or MODEL_CHOICE_DEFAULT
    except Exception:
        MODEL_CHOICE = MODEL_CHOICE_DEFAULT

    HF_USER = get_secret("HF_USER")

    # MODEL_VERSION_DEFAULT is the git-committed baseline (bumped by
    # bump_model_version.py). Add an OPTIONAL "MODEL_VERSION" Colab/Kaggle
    # Secret to reload a different push ad-hoc, without editing this file.
    MODEL_VERSION_DEFAULT = "v1"
    try:
        MODEL_VERSION = get_secret("MODEL_VERSION") or MODEL_VERSION_DEFAULT
    except Exception:
        MODEL_VERSION = MODEL_VERSION_DEFAULT

    # Matches the "-tpu" suffix train_model.py pushes to, so this reloads
    # the TPU-trained adapter rather than the GPU-trained one at the plain
    # (no "-tpu") name.
    HF_REPO = f"{HF_USER}/{MODEL_CHOICE}-financial-reasoner-{MODEL_VERSION}-tpu"

    # The pushed repo is adapter-only (see train_model.py's push-to-hub
    # comment) - AutoPeftModelForCausalLM is peft's loader built
    # specifically for that: it reads adapter_config.json, resolves the
    # base model automatically, and wraps it with the adapter. This is the
    # vanilla-peft equivalent of the GPU script's
    # FastLanguageModel.from_pretrained(..., load_in_4bit=True) reload -
    # bf16 here since bitsandbytes has no TPU backend.
    model = AutoPeftModelForCausalLM.from_pretrained(
        HF_REPO,
        torch_dtype=torch.bfloat16,
    ).to(xm.xla_device())
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)

# unsloth's FastLanguageModel.for_inference(model) has no TPU equivalent
# (unsloth doesn't support TPU) - plain eval mode is all that's needed.
model.eval()

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
_OPPOSITE_ACTION_WORDS = {
    "BUY": re.compile(r"\b(sell|selling|short|shorting|exit|take profits?)\b", re.IGNORECASE),
    "SELL": re.compile(r"\b(buy|buying|accumulate|accumulating|add(?:ing)? (?:shares|to (?:a |the )?position)|initiate a (?:buy|long) position)\b", re.IGNORECASE),
    "HOLD": None,  # HOLD has no single "opposite" action to flag against
}

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
        contradicts = bool(opposite_re and (opposite_re.search(answer) or opposite_re.search(reasoning)))
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


# No CUDA cache to clear on TPU (see evaluate_model.py's equivalent
# comment) - XLA's allocator has no manual-release call.
#
# Unlike the GPU scripts' unsloth-loaded model (confirmed live to
# sometimes lose its disable_adapter() method after certain unsloth code
# paths - see gpu/evaluate_base_model_only.py's comment on the same
# call), this script's model comes from vanilla peft's
# AutoPeftModelForCausalLM, which reliably supports disable_adapter() as
# documented, standard behavior - no separate-model-load fallback needed
# here. Still guarded, matching evaluate_model.py's TPU pass 2, so an
# unexpected failure has a clear message instead of a bare traceback.
try:
    with model.disable_adapter():
        base_a = run_task_a_eval("BASE model (adapter disabled)")
        base_b = run_task_b_eval("BASE model (adapter disabled)")
except Exception as e:
    print(f"Base-model pass failed ({e!r}). Fallback: reload the base "
          "model fresh in a new cell and rerun run_task_a_eval/"
          "run_task_b_eval, or share this error.")
