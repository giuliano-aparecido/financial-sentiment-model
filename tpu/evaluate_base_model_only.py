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
    from google.colab import userdata
    return userdata.get(name)


try:
    model, tokenizer
    print("Reusing model already in memory - skipping reload.")
except NameError:
    print("Model not in memory (new or crashed session) - reloading the "
          "already-trained, already-pushed model from Hugging Face...")
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    import torch_xla.core.xla_model as xm

    MODEL_CHOICE = "llama-3.2-3b"
    HF_USER = get_secret("HF_USER")

    # MODEL_VERSION_DEFAULT is the git-committed baseline (bumped by
    # bump_model_version.py). Add an OPTIONAL "MODEL_VERSION" Colab/Kaggle
    # Secret to reload a different push ad-hoc, without editing this file.
    MODEL_VERSION_DEFAULT = "v7"
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

DIRECTION_RE = re.compile(r'"direction"\s*:\s*"(BULLISH|BEARISH|NEUTRAL)"')
# Not used for accuracy scoring (direction is), just to surface the model's
# generated answer text in the misclassification dump below so answer
# quality can be eyeballed alongside the direction miss.
ANSWER_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"')
EVAL_SAMPLE_PER_SOURCE = 100
VAL_FILES = {"synthetic": "dataset_val.jsonl", "real": "dataset_val_real.jsonl"}

# How many full generations to keep and print per (source, expected,
# predicted) wrong-answer combination - see evaluate_model.py's comment
# above the same constant for why.
SAMPLE_MISCLASSIFICATIONS_PER_PAIR = 3

random.seed(42)
eval_rows = []
for source, path in VAL_FILES.items():
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    for row in rows:
        row["_source"] = source
    if len(rows) > EVAL_SAMPLE_PER_SOURCE:
        rows = random.sample(rows, EVAL_SAMPLE_PER_SOURCE)
    eval_rows.extend(rows)

print(f"Evaluating base model on {len(eval_rows)} val rows "
      f"({sum(1 for r in eval_rows if r['_source'] == 'synthetic')} synthetic, "
      f"{sum(1 for r in eval_rows if r['_source'] == 'real')} real)")


def expected_direction(row):
    return json.loads(row["output"])["impacted_stocks"][0]["direction"]


def generate_response(row):
    prompt = alpaca_prompt.format(
        row["ticker"], row["user_query"], row["market_data"],
        row["valuation"], row["earnings"], row["news"], "",
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def is_valid_json(text):
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


# No CUDA cache to clear on TPU (see evaluate_model.py's equivalent
# comment) - XLA's allocator has no manual-release call.
with model.disable_adapter():
    base = run_eval("BASE model (adapter disabled)")
