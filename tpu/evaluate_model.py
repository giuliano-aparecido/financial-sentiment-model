# Direction-accuracy evaluation - paste as ONE Colab cell and run it in the
# SAME session, right AFTER the training cell (it reuses the `model`,
# `tokenizer`, and `alpaca_prompt` variables that cell leaves in memory, and
# the dataset_val*.jsonl files already on this Colab's disk).
#
# What it measures - the metric that actually matters, which token loss
# doesn't (see ../docs/training-results-analysis.md):
#   - direction accuracy (BULLISH/BEARISH/NEUTRAL correct or not), overall
#     and split by val source (synthetic vs real) and by class (confusion
#     matrix per source, to spot "always answers NEUTRAL"-style bias and
#     whether it's source-specific)
#   - JSON validity rate of the model's raw output
#   - the same numbers for the BASE model (LoRA adapter temporarily
#     disabled) as the missing baseline - the base-vs-tuned delta is the
#     true value of the whole training pipeline.
#
# Runtime expectation: unverified on TPU v5e-1 (the GPU version takes
# roughly 15-30 minutes on a T4 for ~160 rows x 2 passes) - progress prints
# every 20 rows regardless. The TUNED pass runs FIRST so that if the
# session dies partway, the number you care most about is already printed.

import json
import random
import re

import torch

# Rows to sample per val source (real val only has 60 rows, so it always
# runs in full). None = evaluate every row (slower: 657 synthetic rows).
EVAL_SAMPLE_PER_SOURCE = 100

try:
    model, tokenizer, alpaca_prompt
except NameError:
    raise SystemExit(
        "Run this cell in the same session as the training cell - it needs "
        "the model/tokenizer/alpaca_prompt still in memory. If the runtime "
        "restarted, re-run the training cell first (or load the pushed "
        "model from HF)."
    )

# unsloth's FastLanguageModel.for_inference(model) has no TPU equivalent
# (unsloth doesn't support TPU) - plain eval mode is all that's needed
# here, unsloth's call is a generation-speed optimization, not a
# correctness requirement.
model.eval()

VAL_FILES = {"synthetic": "dataset_val.jsonl", "real": "dataset_val_real.jsonl"}
DIRECTION_RE = re.compile(r'"direction"\s*:\s*"(BULLISH|BEARISH|NEUTRAL)"')

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
    prompt = alpaca_prompt.format(row["ticker"], row["user_query"], row["news"], "")
    # model.device is backend-agnostic - once the model has been placed on
    # the XLA device during training, this just works.
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
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

    for i, row in enumerate(eval_rows, 1):
        text = generate_response(row)
        exp = expected_direction(row)
        m = DIRECTION_RE.search(text)
        pred = m.group(1) if m else None
        source = row["_source"]

        stats = per_source[source]
        stats["total"] += 1
        stats["correct"] += (pred == exp)
        stats["json_ok"] += is_valid_json(text)
        confusion[source][(exp, pred)] = confusion[source].get((exp, pred), 0) + 1

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
        for (exp, pred), count in sorted(source_confusion.items()):
            print(f"  {exp:<8} -> {pred or 'UNPARSEABLE':<12} {count}")
    print()
    return per_source


# Pass 1: the fine-tuned model (adapter active) - the number that matters.
tuned = run_eval("FINE-TUNED model")

# The GPU script clears the CUDA cache here between passes to avoid an
# OOM partway through pass 2 (this happened live during development on a
# free T4). XLA's memory allocator has no direct manual-release equivalent
# to CUDA's caching allocator, so there's nothing to call here - if pass 2
# runs short on TPU memory, that's a real capacity issue to address (e.g.
# smaller EVAL_SAMPLE_PER_SOURCE), not a cache to clear.

# Pass 2: the base model, by temporarily disabling the LoRA adapter on the
# same loaded model - no second download, no extra memory. This is the
# baseline that tells us whether training added value at all. Guarded so a
# surprise here can't erase the tuned results already printed above.
try:
    with model.disable_adapter():
        base = run_eval("BASE model (adapter disabled)")
except Exception as e:
    print(f"Base-model pass failed ({e!r}) - tuned results above still stand. "
          "Fallback: reload the base model fresh in a new cell and rerun "
          "run_eval, or share this error.")
