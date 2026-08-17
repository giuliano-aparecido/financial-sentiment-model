"""
RunPod (or any plain GPU box - a bare EC2/GCS instance, etc.) copy of
../colab/train/gpu/train_model.py. Same training logic, byte-identical
alpaca_prompt (see CONTRIBUTING.md's 4-way sync rule - now covers this
file too), and same MODEL_VERSION_DEFAULT (bumped by bump_model_version.py,
which scans the whole repo tree, so this file is already covered without
needing its own logic added there).

The one real difference: get_secret() below reads plain environment
variables directly, with no Colab/Kaggle secrets-vault branching - RunPod
has no such API, so that complexity doesn't apply here. See the Colab
version of this file if you're running there instead.

Usage (after deploying a GPU pod and connecting via SSH or a web
terminal - no Jupyter needed, this runs as a plain script):

    export HF_TOKEN=...      # a Hugging Face WRITE-access token
    export HF_USER=...       # your Hugging Face username
    # optional overrides, same as the Colab Secrets they mirror:
    # export MODEL_CHOICE=apertus-8b
    # export MODEL_VERSION=v6

    # Upload all four dataset files into this same working directory:
    #   dataset_train.jsonl / dataset_val.jsonl       (generate_synthetic_dataset.py)
    #   dataset_train_real.jsonl / dataset_val_real.jsonl  (generate_real_dataset.py)
    # then:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes
    python train_model.py

At the end, the trained adapter is pushed straight to Hugging Face via
HF_TOKEN/HF_USER - same output artifact as a Colab run, nothing
RunPod-specific about that step.
"""

import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


def get_secret(name):
    """Plain environment variable - see this file's own docstring for
    where to set these before running. No Colab/Kaggle fallback needed
    here (unlike ../colab/train/gpu/train_model.py's version), since this
    file is specifically for environments that have neither."""
    return os.environ.get(name)


# MODEL_CHOICE_DEFAULT is the git-committed baseline - which entry of
# MODEL_REGISTRY below to train. Set an OPTIONAL MODEL_CHOICE environment
# variable to switch base models ad-hoc for this run only, without editing
# this file - e.g. "apertus-8b" to try a different family. Falls back to
# the default below if unset. An override naming a key that doesn't exist
# in MODEL_REGISTRY still fails loudly at the dict lookup below - not
# worth adding extra validation for a typo in an advanced, opt-in
# override.
MODEL_CHOICE_DEFAULT = "llama-3.1-8b"
MODEL_CHOICE = get_secret("MODEL_CHOICE") or MODEL_CHOICE_DEFAULT

MODEL_REGISTRY = {
    "llama-3.2-3b": {
        "repo": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 2048,
    },
    # Default as of v1 - see colab/train/gpu/train_model.py's identically-
    # named entry for why 8B is safe here (bnb-4bit) but blocked on the TPU
    # path.
    "llama-3.1-8b": {
        "repo": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 4096,
    },
    "apertus-8b": {
        "repo": "swiss-ai/Apertus-8B-Instruct-2509",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 4096,
    },
    # Added to try a size step up from the 3B default (see TODO.md in
    # D:/projects) without changing infra - Unsloth's 4-bit QLoRA
    # approach is specifically built to make 7-8B models trainable on
    # the same GPU tier as 3B, unlike a genuine 70B jump. Note this is
    # Llama 3.1's 8B, not a "7B" - current Llama generations don't have
    # that size. MODEL_CHOICE_DEFAULT below is deliberately NOT changed
    # to this - still 3B until a head-to-head eval justifies it.
    "llama-3.1-8b": {
        "repo": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 2048,
    },
    "apertus-0.5b": {
        "repo": "swiss-ai/Apertus-v1.1-0.5B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 2048,
    },
    "qwen-2.5-7b": {
        "repo": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 2048,
    },
    "mistral-7b": {
        "repo": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 2048,
    },
}

selected_config = MODEL_REGISTRY[MODEL_CHOICE]
MODEL_NAME = selected_config["repo"]
MAX_SEQ_LENGTH = selected_config["max_seq_length"]
TARGET_MODULES = selected_config["target_modules"]

print(f"Loading Model Family: {MODEL_CHOICE} -> {MODEL_NAME}")

# 3. Load 4-bit Quantized Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# 4. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=TARGET_MODULES,
    lora_alpha=16,
    lora_dropout=0.05,  # The train/eval loss gap from an early run (0.12
                        # vs ~0.4) is a real generalization gap, likely
                        # template memorization given the dataset's
                        # limited scenario count relative to multiple
                        # epochs of repetition. Dropout is a standard,
                        # low-cost regularizer against exactly that.
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 5. Load and format dataset (upload all four files listed in this file's
# own docstring into the working directory first)
#
# Mixes the synthetic dataset with the real, proxy-labeled one -
# load_dataset accepts a list of files per split and concatenates them, so
# this is the whole mechanism. Both generators produce the identical
# {ticker, user_query, market_data, valuation, earnings, news, output}
# schema (v4) on purpose, specifically so this merge needs no
# reconciliation. The real dataset is already rebalanced by recommendation on
# the train side and left at its natural distribution on the val side (see
# that generator's docstring) - nothing further to do here.
dataset_dict = load_dataset(
    "json",
    data_files={
        "train": ["dataset_train.jsonl", "dataset_train_real.jsonl"],
        "validation": ["dataset_val.jsonl", "dataset_val_real.jsonl"],
    },
)

# alpaca_prompt's ### Input: section is built with the exact same structure
# as the live inference prompt in financial-sentiment-api's
# app/services/inference.py (Target Stock / User Question / Current Market
# Data / Valuation / Recent Earnings / Recent News & Results) - the model
# needs to learn to read all of these, since that's what it's actually
# given in production. Keep this in sync any time inference.py's prompt
# changes, and in sync with ../colab/train/{gpu,tpu}/train_model.py's
# copies of this same string and the ../colab/train/{gpu,tpu}/evaluate_*.py
# scripts' copies (see CONTRIBUTING.md's 4-way sync rule); the "CRITICAL
# SENTIMENT RULES" block matches inference.py's current state.
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:

Analyze the following financial data and news and output JSON containing the impacted stock ticker, detailed reasoning, a recommended action (BUY/SELL/HOLD), confidence score, and a direct answer to the user's question, in exactly this shape:
{{"impacted_stocks": [{{"ticker": "...", "reasoning": "...", "recommendation": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "answer": "..."}}]}}

CRITICAL SENTIMENT RULES:

1. Weigh guidance cuts and revenue misses higher than minor operational wins.
2. The Valuation block is a real, structural signal, not decoration - a large over/undervaluation gap should meaningfully shape your recommendation and confidence, not just recent news. Only let concrete, current news override it when the news describes a specific catalyst (an actual event, not a generic "market volatility" statement) the valuation estimate couldn't have priced in. Even then, a strong catalyst alone doesn't earn BUY (or SELL): BUY requires the stock isn't already priced beyond what the news justifies - a real catalyst with no valuation headroom, and no stated reason for further upside, is HOLD, not BUY. The same applies symmetrically to SELL and further downside.
3. HOLD means either the available signals genuinely conflict or are too weak/routine to support a BUY/SELL call, or a real catalyst exists but the stock has no valuation headroom left to act on it - not a default for "I'm not sure." Use it when Valuation, Market Data, Earnings, and News don't converge on one recommendation, when nothing in the input is materially new, or when a strong catalyst is real but the price already exceeds what it justifies.
4. confidence is a 0.0-1.0 score for how strongly the evidence supports your recommendation, not how certain you are a clear case exists at all - a HOLD call can still carry moderate confidence when "no room to act" is itself well-supported.
5. "Data unavailable." or "Not applicable (...)" in any block means exactly that - treat it as missing information, never invent numbers or events to fill the gap.
6. P/E under 20 (sector-adjusted via Sector Median P/E) suggests undervaluation; 20-30 is roughly neutral; over 30 suggests a richer valuation that needs a real growth story to justify. For a Real Estate-sector company specifically, Price/Book below 1.0 is the more meaningful signal - GAAP depreciation makes P/E unreliable for that sector.

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


def format_prompts(examples):
    texts = []
    fields = zip(
        examples["ticker"], examples["user_query"], examples["market_data"],
        examples["valuation"], examples["earnings"], examples["news"], examples["output"],
    )
    for ticker, user_query, market_data, valuation, earnings, news, output in fields:
        text = alpaca_prompt.format(ticker, user_query, market_data, valuation, earnings, news, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}


dataset_dict = dataset_dict.map(format_prompts, batched=True)

train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["validation"]

# 6. Set up Trainer using SFTConfig
from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Without this, nothing measures whether
                                 # the model generalized beyond the
                                 # training examples. The generator holds
                                 # out specific tickers/templates for
                                 # exactly this purpose.
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # num_train_epochs is an upper-bound safety ceiling, not the actual
        # target - early stopping below decides the real stopping point,
        # based on eval_loss rather than a guessed epoch/step count. See
        # docs/training-results-analysis.md for why recommendation accuracy
        # (evaluate_model.py), not this loss curve, is what should actually
        # decide whether a given run is good.
        num_train_epochs=5,
        eval_strategy="steps",
        eval_steps=50,
        # weight_decay is a standard, low-cost regularizer alongside
        # lora_dropout, for the same train/eval loss gap reason noted above.
        weight_decay=0.01,
        # save_strategy/save_steps must match eval_strategy/eval_steps for
        # load_best_model_at_end to work - the Trainer needs a checkpoint
        # saved at the exact step an eval was run in order to reload it.
        # save_total_limit keeps disk usage bounded given how often this
        # saves.
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        # Stop training at the point of best eval_loss instead of blindly
        # running num_train_epochs - directly addresses overfitting rather
        # than just guessing a smaller epoch count.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        output_dir="outputs",
    ),
    # Stops training once eval_loss hasn't improved for 3 consecutive evals
    # (3 x eval_steps=50 = 150 steps of no improvement), rather than
    # training all the way to num_train_epochs regardless.
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Completion-only loss masking. Without this, SFTTrainer computes loss over
# the ENTIRE text field - the "Below is an instruction that describes a
# task..." preamble and the ### Instruction:/### Input: boilerplate
# included, not just the ### Response: completion. That boilerplate is
# nearly identical across every example, so the model can drive loss down
# substantially just by memorizing it, which inflates the reported loss
# numbers without reflecting how well it's actually learning to classify
# sentiment. This masks the loss to only the ### Response: continuation,
# matching the alpaca_prompt's own instruction/response markers.
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    # IMPORTANT: these markers must match the LITERAL text exactly,
    # including incidental whitespace - alpaca_prompt has a blank line
    # after each header, so the actual text is "### Instruction:\n\n" /
    # "### Response:\n\n" (double newline), not "### Instruction:\n" /
    # "### Response:\n" (single newline). This is not cosmetic to a BPE
    # tokenizer: ":\n\n" tokenizes as one atomic token distinct from
    # ":\n", so a single-newline marker will never be found in the
    # tokenized data at all, silently masking every sample's loss to -100
    # (train_on_responses_only raises a clear error when this happens - if
    # you see "masked every label to -100... marker was not found," this
    # mismatch is almost certainly why). Verify any change here by loading
    # the target model's tokenizer, tokenizing the marker in isolation and
    # embedded in a real formatted example, and confirming the token
    # sequence actually appears - don't assume a string that "looks like a
    # substring" tokenizes as one.
    instruction_part="### Instruction:\n\n",
    response_part="### Response:\n\n",
)


# 7. Train & Push Adapter to Hugging Face
trainer.train()

# get_secret() is defined at the top of this file (it's needed there too,
# for the MODEL_CHOICE override) - reused here for HF_TOKEN/HF_USER/
# MODEL_VERSION rather than redefined.
HF_TOKEN = get_secret("HF_TOKEN")
HF_USER = get_secret("HF_USER")

# MODEL_VERSION_DEFAULT is the git-committed baseline (bumped by
# bump_model_version.py, same as the Colab copy) - what every run uses
# unless overridden. Set an OPTIONAL MODEL_VERSION environment variable to
# try a different push ad-hoc for this run only, without editing this file
# - e.g. "v6" to compare against an older push. Falls back to the default
# below if unset.
MODEL_VERSION_DEFAULT = "v1"
MODEL_VERSION = get_secret("MODEL_VERSION") or MODEL_VERSION_DEFAULT

HF_REPO = f"{HF_USER}/{MODEL_CHOICE}-financial-reasoner-{MODEL_VERSION}"

model.push_to_hub_merged(HF_REPO, tokenizer, save_method="lora", token=HF_TOKEN)
