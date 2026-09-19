!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes

import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Pull config from Colab/Kaggle's own Secrets manager rather than
# hardcoding it - anything written into a saved/shared .py file is one
# accidental commit or shared-notebook-link away from being a real leaked
# credential (for HF_TOKEN/HF_USER below) or just annoying to edit and
# re-paste (for MODEL_CHOICE/MODEL_VERSION). Add secrets named HF_TOKEN (a
# write-access token) and HF_USER (your Hugging Face username), and grant
# this notebook access when prompted - do this BEFORE an unattended run;
# the permission grant is interactive and will block otherwise. See the
# README's "Required Colab Secrets" section.
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
    try:
        from google.colab import userdata
        return userdata.get(name)
    except ImportError:
        # Not Colab or Kaggle - e.g. RunPod, or any plain GPU box. Neither
        # has a secrets-vault API to call, so fall back to a real
        # environment variable (set via RunPod's pod env-var config, a
        # .env file, or `export NAME=value` before running this script).
        return os.environ.get(name)

# MODEL_CHOICE_DEFAULT is the git-committed baseline - which entry of
# MODEL_REGISTRY below to train. Add an OPTIONAL "MODEL_CHOICE" Colab/
# Kaggle Secret (same mechanism as HF_USER/HF_TOKEN, see README's
# "Required Colab Secrets") to switch base models ad-hoc in this session
# only, without editing this file - e.g. set it to "apertus-8b" to try a
# different family. Falls back to the default below if the secret was
# never created (not just ungranted) - a broad except is deliberate here
# since Colab/Kaggle raise different exception types for "no such
# secret", and this one specific secret is optional by design, so any
# failure to read it should silently fall back, never block or crash. An
# override naming a key that doesn't exist in MODEL_REGISTRY still fails
# loudly at the dict lookup below - not worth adding extra validation for
# a typo in an advanced, opt-in override.
MODEL_CHOICE_DEFAULT = "llama-3.1-8b"
try:
    MODEL_CHOICE = get_secret("MODEL_CHOICE") or MODEL_CHOICE_DEFAULT
except Exception:
    MODEL_CHOICE = MODEL_CHOICE_DEFAULT

MODEL_REGISTRY = {

    "llama-3.2-3b": {

        "repo": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

        "max_seq_length": 2048,

    },

    # Default as of v1 - a real 8B-class instruct model (vs. llama-3.2-3b's
    # 3B) fits comfortably in this path's bnb-4bit quantization on a free
    # T4/A100, unlike the TPU path (see colab/train/tpu/train_model.py's
    # identically-named entry, which stays blocked there - bf16-only, no
    # quantization, makes 8B a tight/unsafe fit on a v5e-1's 16GB HBM).
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
# {task, ticker, user_query, price_context, market_data, valuation,
# earnings, news, news_reaction, recommendation, valuation_bucket, output}
# superset schema on purpose (task="reaction" rows leave market_data/
# valuation/earnings/user_query/recommendation/valuation_bucket as "";
# task="analysis" rows leave nothing empty), specifically so this merge
# needs no reconciliation and format_prompts below can branch on "task"
# alone. recommendation/valuation_bucket are eval-only ground truth now
# (see docs/task-b-learned-recommendation-plan.md) - format_prompts never
# reads them, since the model must decide the recommendation itself and
# it's already embedded in "output"'s JSON for the "analysis" case. The
# real dataset's
# Task A rows are rebalanced by news_reaction on the train side and left
# at their natural distribution on the val side (see that generator's
# docstring) - nothing further to do here.
#
# dataset_{train,val}_real_taskb.jsonl (from convert_existing_to_taskb.py,
# or a fresh GATE-A-approved real-dataset regen with ENABLE_TASK_B_
# GENERATION=True) supply the real dataset's Task B half - dataset_
# {train,val}_real.jsonl only ever carries task="reaction" rows going
# forward (see generate_real_dataset.py's ENABLE_TASK_B_GENERATION, off
# by default). Optional: upload them if you have them, otherwise real
# Task B coverage comes from the synthetic dataset alone - checked for
# existence rather than hard-required, since a notebook run shouldn't
# fail just because this optional pair wasn't uploaded this time.
# load_dataset("json", ...) handles JSON Lines natively.
train_files = ["dataset_train.jsonl", "dataset_train_real.jsonl"]
val_files = ["dataset_val.jsonl", "dataset_val_real.jsonl"]
if os.path.exists("dataset_train_real_taskb.jsonl"):
    train_files.append("dataset_train_real_taskb.jsonl")
if os.path.exists("dataset_val_real_taskb.jsonl"):
    val_files.append("dataset_val_real_taskb.jsonl")

dataset_dict = load_dataset(
    "json",
    data_files={
        "train": train_files,
        "validation": val_files,
    },
)

# Two-stage pipeline (2026-08-19): a single call is no longer asked to
# both read the news AND weigh it against valuation headroom to produce a
# BUY/SELL/HOLD label - that entangled rule is exactly what was fragile
# to learn (see financial-sentiment-model's CONTRIBUTING.md/plan file for
# the full redesign). Every training row now carries a "task" field:
# task="reaction" rows train task_a_prompt (classify news_reaction from
# the news + a recent price move, no direction anywhere in it);
# task="analysis" rows train task_b_prompt (given an ALREADY-DECIDED
# news_reaction, decide the recommendation itself AND write reasoning/
# answer consistent with it - see docs/task-b-learned-recommendation-
# plan.md; recommendation is no longer handed to it as an input). Both
# templates' ### Input: sections mirror financial-sentiment-api's
# app/services/inference.py exactly (Task A: Target Stock/Recent Price
# Move/Recent News & Results; Task B: adds News Reaction alongside the
# original Target Stock/User Question/Current Market Data/Valuation/
# Recent Earnings/Recent News & Results).
# Keep BOTH templates in sync any time inference.py's prompts change, and
# in sync with ../tpu/train_model.py's copies of these same two strings
# and the ../{gpu,tpu}/evaluate_*.py and runpod/*.py scripts' copies (see
# CONTRIBUTING.md's sync rule).
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

Decide the recommended action for this stock (BUY, SELL, or HOLD) yourself from the news reaction, valuation, earnings, and market data below, then explain your reasoning and answer the user's question. Output JSON in exactly this shape:
{{"recommendation": "BUY|SELL|HOLD", "reasoning": "...", "answer": "..."}}

### Input:

Target Stock: {}
User Question: {}
News Reaction: {}

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
        examples["task"], examples["ticker"], examples["user_query"], examples["price_context"],
        examples["market_data"], examples["valuation"], examples["earnings"], examples["news"],
        examples["news_reaction"], examples["output"],
    )
    for task, ticker, user_query, price_context, market_data, valuation, earnings, news, news_reaction, output in fields:

        if task == "reaction":
            text = task_a_prompt.format(ticker, price_context, news, output) + tokenizer.eos_token
        elif task == "analysis":
            # recommendation is NOT an input here (see task_b_prompt's own
            # comment) - it's inside `output`'s JSON, which the model must
            # produce itself.
            text = task_b_prompt.format(
                ticker, user_query, news_reaction, market_data, valuation, earnings, news, output,
            ) + tokenizer.eos_token
        else:
            raise ValueError(f"Unknown task: {task!r}")

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
        # docs/training-results-analysis.md for why recommendation accuracy
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
# matching task_a_prompt/task_b_prompt's own instruction/response markers.
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    # IMPORTANT: these markers must match the LITERAL text exactly,
    # including incidental whitespace - both prompt templates have a blank line
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

# get_secret() is defined at the top of this file (it's needed there too,
# for the MODEL_CHOICE override) - reused here for HF_TOKEN/HF_USER/
# MODEL_VERSION rather than redefined.
HF_TOKEN = get_secret("HF_TOKEN")

HF_USER = get_secret("HF_USER")

# MODEL_VERSION_DEFAULT is the git-committed baseline (bumped by
# bump_model_version.py, same as before) - what every session uses unless
# overridden. Add an OPTIONAL "MODEL_VERSION" Colab/Kaggle Secret (same
# mechanism as HF_USER/HF_TOKEN above, see README's "Required Colab
# Secrets") to try a different push ad-hoc in this session only, without
# editing this file at all - e.g. set it to "v6" to compare against an
# older push. Falls back to the default below if the secret was never
# created (not just ungranted) - a broad except is deliberate here since
# Colab/Kaggle raise different exception types for "no such secret", and
# this one specific secret is optional by design, so any failure to read
# it should silently fall back, never block or crash.
MODEL_VERSION_DEFAULT = "v1"
try:
    MODEL_VERSION = get_secret("MODEL_VERSION") or MODEL_VERSION_DEFAULT
except Exception:
    MODEL_VERSION = MODEL_VERSION_DEFAULT

HF_REPO = f"{HF_USER}/{MODEL_CHOICE}-financial-reasoner-{MODEL_VERSION}"

model.push_to_hub_merged(HF_REPO, tokenizer, save_method = "lora", token = HF_TOKEN)
