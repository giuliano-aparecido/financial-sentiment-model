!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes

import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MODEL_CHOICE = "llama-3.2-3b"

MODEL_REGISTRY = {

    "llama-3.2-3b": {

        "repo": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

        "max_seq_length": 2048,

    },

    "apertus-8b": {

        "repo": "swiss-ai/Apertus-8B-Instruct-2509",

        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

        "max_seq_length": 4096,

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

    }

}

selected_config = MODEL_REGISTRY[MODEL_CHOICE]

MODEL_NAME = selected_config["repo"]

MAX_SEQ_LENGTH = selected_config["max_seq_length"]

TARGET_MODULES = selected_config["target_modules"]

print(f"Loading Model Family: {MODEL_CHOICE} -> {MODEL_NAME}")

# 3. Load 4-bit Quantized Model

model, tokenizer = FastLanguageModel.from_pretrained(

    model_name = MODEL_NAME,

    max_seq_length = MAX_SEQ_LENGTH,

    dtype = None,

    load_in_4bit = True,

)

# 4. Add LoRA Adapters

model = FastLanguageModel.get_peft_model(

    model,

    r = 16,

    target_modules = TARGET_MODULES,

    lora_alpha = 16,

    lora_dropout = 0.05,  # The train/eval loss gap from an early run
                          # (0.12 vs ~0.4) is a real generalization gap,
                          # likely template memorization given the
                          # dataset's limited scenario count relative to
                          # multiple epochs of repetition. Dropout is a
                          # standard, low-cost regularizer against exactly
                          # that.

    bias = "none",

    use_gradient_checkpointing = "unsloth",

)

# 5. Load and format dataset (upload all four files below to Colab -
# dataset_train.jsonl/dataset_val.jsonl from generate_synthetic_dataset.py,
# dataset_train_real.jsonl/dataset_val_real.jsonl from
# generate_real_dataset.py)
#
# Mixes the synthetic dataset with the real, proxy-labeled one -
# load_dataset accepts a list of files per split and concatenates them, so
# this is the whole mechanism. Both generators produce the identical
# {ticker, user_query, market_data, valuation, earnings, news, output}
# schema (v4) on purpose, specifically so this merge needs no
# reconciliation. The real dataset is already rebalanced by direction on
# the train side and left at its natural distribution on the val side (see
# that generator's docstring) - nothing further to do here.
# load_dataset("json", ...) handles JSON Lines natively.
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
# changes, and in sync with ../tpu/train_model.py's copy of this same
# string and the ../{gpu,tpu}/evaluate_*.py scripts' copies (see
# CONTRIBUTING.md's 4-way sync rule); the "CRITICAL SENTIMENT RULES" block
# matches inference.py's current state (a rule covering "beat but cut
# guidance"-style cases is kept here pending mixed-signal examples proving
# out in eval before also dropping it from inference.py).
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

def format_prompts(examples):

    texts = []

    fields = zip(
        examples["ticker"], examples["user_query"], examples["market_data"],
        examples["valuation"], examples["earnings"], examples["news"], examples["output"],
    )
    for ticker, user_query, market_data, valuation, earnings, news, output in fields:

        text = alpaca_prompt.format(ticker, user_query, market_data, valuation, earnings, news, output) + tokenizer.eos_token

        texts.append(text)

    return { "text" : texts }

dataset_dict = dataset_dict.map(format_prompts, batched = True)

train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["validation"]

# 6. Set up Trainer using SFTConfig
from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,  # Without this, nothing measures whether
                                   # the model generalized beyond the
                                   # training examples. The generator holds
                                   # out specific tickers/templates for
                                   # exactly this purpose.
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs is an upper-bound safety ceiling, not the actual
        # target - early stopping below decides the real stopping point,
        # based on eval_loss rather than a guessed epoch/step count. See
        # docs/training-results-analysis.md for why direction accuracy
        # (evaluate_model.py), not this loss curve, is what should actually
        # decide whether a given run is good.
        num_train_epochs = 5,
        eval_strategy = "steps",
        eval_steps = 50,
        # weight_decay is a standard, low-cost regularizer alongside
        # lora_dropout, for the same train/eval loss gap reason noted above.
        weight_decay = 0.01,
        # save_strategy/save_steps must match eval_strategy/eval_steps for
        # load_best_model_at_end to work - the Trainer needs a checkpoint
        # saved at the exact step an eval was run in order to reload it.
        # save_total_limit keeps disk usage bounded given how often this
        # saves.
        save_strategy = "steps",
        save_steps = 50,
        save_total_limit = 3,
        # Stop training at the point of best eval_loss instead of blindly
        # running num_train_epochs - directly addresses overfitting rather
        # than just guessing a smaller epoch count.
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "outputs",
    ),
    # Stops training once eval_loss hasn't improved for 3 consecutive evals
    # (3 x eval_steps=50 = 150 steps of no improvement), rather than
    # training all the way to num_train_epochs regardless.
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)],
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
    instruction_part = "### Instruction:\n\n",
    response_part = "### Response:\n\n",
)


# 7. Train & Push Adapter to Hugging Face

trainer.train()

# Pull credentials from Colab/Kaggle's own Secrets manager rather than
# hardcoding them - anything written into a saved/shared .py file is one
# accidental commit or shared-notebook-link away from being a real leaked
# credential. Add secrets named HF_TOKEN (a write-access token) and
# HF_USER (your Hugging Face username), and grant this notebook access
# when prompted - do this BEFORE an unattended run; the permission grant
# is interactive and will block otherwise. See the README's "Required
# Colab Secrets" section.
#
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
    from google.colab import userdata
    return userdata.get(name)

HF_TOKEN = get_secret("HF_TOKEN")

HF_USER = get_secret("HF_USER")

HF_REPO = f"{HF_USER}/{MODEL_CHOICE}-financial-reasoner-v7"

model.push_to_hub_merged(HF_REPO, tokenizer, save_method = "lora", token = HF_TOKEN)
