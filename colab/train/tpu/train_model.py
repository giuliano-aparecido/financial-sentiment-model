!pip install -q -U transformers peft trl accelerate datasets

# torch_xla is expected to already be present and version-matched in a
# Colab TPU v5e-1 runtime - pip-installing it separately here risks
# pairing it with a mismatched torch build. If this import fails, the
# runtime isn't actually set to TPU (Runtime > Change runtime type),
# not a missing-package problem to pip install around.
import os
import torch
import torch_xla.core.xla_model as xm

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
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
# too, so the import succeeds there and userdata.get() just hangs and
# times out instead of raising. KAGGLE_KERNEL_RUN_TYPE is set by Kaggle's
# own runtime on every notebook - check that directly instead.
def get_secret(name):
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    from google.colab import userdata
    return userdata.get(name)

# MODEL_CHOICE_DEFAULT is the git-committed baseline - which entry of
# MODEL_REGISTRY below to train. Add an OPTIONAL "MODEL_CHOICE" Colab/
# Kaggle Secret (same mechanism as HF_USER/HF_TOKEN, see README's
# "Required Colab Secrets") to switch base models ad-hoc in this session
# only, without editing this file. Falls back to the default below if the
# secret was never created (not just ungranted) - a broad except is
# deliberate here since Colab/Kaggle raise different exception types for
# "no such secret", and this one specific secret is optional by design,
# so any failure to read it should silently fall back, never block or
# crash. An override naming a key that doesn't exist in MODEL_REGISTRY
# still fails loudly at the dict lookup below - not worth adding extra
# validation for a typo in an advanced, opt-in override.
MODEL_CHOICE_DEFAULT = "llama-3.2-3b"
try:
    MODEL_CHOICE = get_secret("MODEL_CHOICE") or MODEL_CHOICE_DEFAULT
except Exception:
    MODEL_CHOICE = MODEL_CHOICE_DEFAULT

MODEL_REGISTRY = {

    "llama-3.2-3b": {

        # Swapped from unsloth/Llama-3.2-3B-Instruct-bnb-4bit (the GPU
        # script's repo) to the non-quantized bf16 mirror - bitsandbytes
        # (and its pre-quantized checkpoints) has no TPU backend, so
        # training here runs in plain bf16 instead of 4-bit.
        "repo": "unsloth/Llama-3.2-3B-Instruct",

        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

        "max_seq_length": 2048,

    },

    "apertus-0.5b": {

        "repo": "swiss-ai/Apertus-v1.1-0.5B-Instruct",

        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

        "max_seq_length": 2048,

    },

    # Not enabled on the TPU path: bf16 with no quantization available
    # (bitsandbytes is CUDA-only) makes 7B/8B a tight-to-unsafe fit on a
    # single v5e-1's 16GB HBM, and qwen/mistral additionally have no
    # confirmed non-quantized mirror the way llama-3.2-3b does. Kept as
    # explicit entries with repo=None so picking one fails with a clear
    # message below instead of a confusing bitsandbytes import error deep
    # inside from_pretrained.
    "apertus-8b": {
        "repo": None,
        "blocked_reason": "bf16 8B params is a tight/unsafe fit on a single v5e-1's 16GB HBM alongside LoRA optimizer state and activations - not validated here.",
    },

    "qwen-2.5-7b": {
        "repo": None,
        "blocked_reason": "unsloth only publishes this as -bnb-4bit (bitsandbytes-only, no TPU support) - no non-quantized mirror confirmed.",
    },

    "mistral-7b": {
        "repo": None,
        "blocked_reason": "unsloth only publishes this as -bnb-4bit (bitsandbytes-only, no TPU support) - no non-quantized mirror confirmed.",
    },

}

selected_config = MODEL_REGISTRY[MODEL_CHOICE]

if selected_config["repo"] is None:
    raise ValueError(f"{MODEL_CHOICE!r} isn't supported on the TPU path yet: {selected_config['blocked_reason']}")

MODEL_NAME = selected_config["repo"]

MAX_SEQ_LENGTH = selected_config["max_seq_length"]

TARGET_MODULES = selected_config["target_modules"]

print(f"Loading Model Family: {MODEL_CHOICE} -> {MODEL_NAME}")

# 3. Load model in bf16 onto the TPU device (no quantization - bitsandbytes
# has no TPU backend, and TPU v5e supports bf16 natively)

device = xm.xla_device()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
).to(device)

# 4. Add LoRA Adapters

lora_config = LoraConfig(

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

    task_type = "CAUSAL_LM",

)

model = get_peft_model(model, lora_config)

# Required for gradients to reach the LoRA layers when the base model is
# frozen and gradient checkpointing is on. unsloth's get_peft_model does
# this internally; vanilla peft does not.
model.enable_input_require_grads()

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
# changes, and in sync with ../gpu/train_model.py's copy of this same
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

# Completion-only loss masking - the vanilla-peft/trl equivalent of
# unsloth's train_on_responses_only (unsloth-only, not available here).
# IMPORTANT: these markers must match the LITERAL text exactly, including
# incidental whitespace - alpaca_prompt has a blank line after each
# header, so the actual text is "### Instruction:\n\n" / "###
# Response:\n\n" (double newline), not "### Instruction:\n" / "###
# Response:\n" (single newline). This is not cosmetic to a BPE tokenizer:
# ":\n\n" tokenizes as one atomic token distinct from ":\n", so a
# single-newline marker will never be found in the tokenized data at all,
# silently masking every sample's loss to -100. Verify any change here by
# loading the target model's tokenizer, tokenizing the marker in isolation
# and embedded in a real formatted example, and confirming the token
# sequence actually appears - don't assume a string that "looks like a
# substring" tokenizes as one.
collator = DataCollatorForCompletionOnlyLM(
    instruction_template = "### Instruction:\n\n",
    response_template = "### Response:\n\n",
    tokenizer = tokenizer,
)

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
    data_collator = collator,
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
        # TPU v5e supports bf16 natively; there's no CUDA device here to
        # query, so (unlike the GPU script) this isn't conditional.
        fp16 = False,
        bf16 = True,
        # adamw_8bit is a bitsandbytes optimizer (CUDA-only) - plain
        # adamw_torch instead.
        optim = "adamw_torch",
        gradient_checkpointing = True,
        logging_steps = 1,
        output_dir = "outputs",
    ),
    # Stops training once eval_loss hasn't improved for 3 consecutive evals
    # (3 x eval_steps=50 = 150 steps of no improvement), rather than
    # training all the way to num_train_epochs regardless.
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)],
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
# editing this file at all. Falls back to the default below if the secret
# was never created (not just ungranted) - a broad except is deliberate
# here since Colab/Kaggle raise different exception types for "no such
# secret", and this one specific secret is optional by design, so any
# failure to read it should silently fall back, never block or crash.
MODEL_VERSION_DEFAULT = "v10"
try:
    MODEL_VERSION = get_secret("MODEL_VERSION") or MODEL_VERSION_DEFAULT
except Exception:
    MODEL_VERSION = MODEL_VERSION_DEFAULT

# "-tpu" suffix keeps this from silently overwriting the already-pushed
# GPU-trained adapter at the plain (no "-tpu") name.
HF_REPO = f"{HF_USER}/{MODEL_CHOICE}-financial-reasoner-{MODEL_VERSION}-tpu"

# model.push_to_hub_merged(..., save_method="lora") in the GPU script
# pushes the adapter only, not a merged model, despite the method name -
# confirmed via unsloth's docs/issues. A PeftModel's own push_to_hub does
# the same adapter-only push, which is what's actually served in
# production (financial-sentiment-api points its inference URL directly
# at this repo).
model.push_to_hub(HF_REPO, token = HF_TOKEN)
tokenizer.push_to_hub(HF_REPO, token = HF_TOKEN)
