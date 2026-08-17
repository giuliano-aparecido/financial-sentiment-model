# LLM training, explained from zero

A from-zero walkthrough of what `train_model.py` actually does, written for
anyone reading this without an ML background. No assumed knowledge of
machine learning or the underlying math.

## Part 1 — What is an LLM, actually?

A Large Language Model (LLM) like Llama 3.2 is, underneath everything, a
giant math function: you feed it some text, and it predicts the next most
likely chunk of text, over and over, one piece at a time.

A few core concepts that show up throughout this script:

- **Tokens** — the model doesn't read letters or whole words. It reads
  "tokens," which are word-pieces. `"Response"` might be one token, `"ing"`
  might be another. A **tokenizer** is the translator that converts text
  into a list of numbers (token IDs) the model can do math on, and converts
  the model's numeric output back into text. This matters more than it
  sounds like it should — see the `train_on_responses_only` section below
  for a real bug that came from underestimating it.
- **Parameters (a.k.a. weights)** — think of the model as an enormous
  dial-board with billions of dials, each set to some number.
  "Llama-3.2-3B" means 3 billion dials. Training = nudging those dials. A
  trained model isn't code with if/else logic — it's just the settings of
  billions of dials, learned from data.
- **Pretraining vs. fine-tuning** — Meta already spent millions of dollars
  and huge amounts of text turning random dials into a model that
  understands English, facts, reasoning, JSON, etc. That's **pretraining**,
  and this project doesn't do that. This project does **fine-tuning**:
  taking that already-competent model and nudging its dials a little
  further, using a much smaller, specific dataset, so it gets good at one
  narrow task — reading financial news and calling BULLISH/BEARISH/NEUTRAL
  in a specific JSON format.

**Analogy for the whole script:** this isn't raising a child from birth.
It's hiring a widely-read, generally-competent adult (pretrained Llama) and
putting them through a short, specialized apprenticeship (fine-tuning) so
they pick up a specific set of judgment calls and a specific report format.

## Part 2 — Setup: installing tools and picking the model

```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
```

These are borrowed toolkits, not project-specific code:

- **unsloth** — makes fine-tuning fast and memory-cheap (this is why a
  3-billion-parameter model can train on Colab's free GPU instead of
  needing a data center).
- **trl** (Transformer Reinforcement Learning) — provides `SFTTrainer`, the
  actual training-loop machinery.
- **peft**, **accelerate**, **bitsandbytes** — supporting libraries
  `unsloth`/`trl` lean on internally.

```python
MODEL_CHOICE = "llama-3.2-3b"
MODEL_REGISTRY = { "llama-3.2-3b": {...}, "apertus-8b": {...}, ... }
```

`MODEL_REGISTRY` is just a lookup table — "if you pick this name, use this
Hugging Face repo, these settings." It lets you swap between several base
models by changing one line, instead of rewriting the whole script per
model.

## Part 3 — Loading the pretrained model

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,      # "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    max_seq_length = MAX_SEQ_LENGTH,  # 2048
    dtype = None,
    load_in_4bit = True,
)
```

This downloads the actual pretrained dial-board from Hugging Face (a
hosting site for models, like GitHub but for trained models/datasets).

- `model_name` — which pretrained model to fetch.
- `max_seq_length = 2048` — the model can only "look at" a fixed-size
  window of tokens at once (its short-term memory span). 2048 tokens is
  roughly 1500 words. Anything longer than that in one training example
  gets truncated — this is why news blocks and prompts can't be unlimited
  length.
- `load_in_4bit = True` — **quantization**. Normally each of those billions
  of dials is stored as a precise number needing 16 or 32 bits of memory.
  4-bit quantization compresses each dial down to a much coarser number
  using only 4 bits. **Analogy:** instead of storing someone's exact height
  as `172.384cm`, you round it to the nearest 10cm bucket. You lose a
  little precision, but the memory footprint shrinks by 4-8x — which is the
  only reason a free Colab GPU can hold a 3-billion-parameter model at all.
- `tokenizer` — the translator described above, paired specifically with
  this model (different models chop words up differently).

## Part 4 — LoRA: training a sticky note, not rewriting the textbook

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = TARGET_MODULES,
    lora_alpha = 16,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)
```

This is the single most important concept to understand: this is **not**
retraining all 3 billion dials. That would need enormous GPU memory and
enormous data to avoid destroying the model's general knowledge. Instead it
uses **LoRA** (Low-Rank Adaptation).

**Analogy:** imagine the pretrained model is a massive, expensively-printed
textbook. Editing the textbook itself (updating every page) is expensive
and risks messing up chapters you didn't mean to touch. LoRA instead sticks
a small removable notepad of sticky notes on top of a few specific pages —
you only write on the sticky notes, the textbook underneath stays
untouched, and at the end you can peel the notes off or merge them in.
Training becomes cheap because you're only ever updating the sticky notes
(a tiny fraction of the total dials — often under 1%).

- `r = 16` — "rank." Controls how large/expressive those sticky notes are.
  Higher `r` means more room to write more nuanced corrections, but more to
  train and more risk of the notes just memorizing instead of generalizing.
  16 is a common, modest default.
- `target_modules` — which pages of the textbook get sticky notes attached.
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj` are specific internal components of each of the model's
  layers (the "attention" and "feed-forward" math blocks) — this says
  "attach notes to all the major decision-making components," the
  standard, thorough choice.
- `lora_alpha = 16` — a scaling knob for how strongly the sticky notes'
  corrections get blended back into the original text when the model
  reads. Works together with `r`; the ratio between them matters more than
  either number alone.
- `lora_dropout = 0.05` — a regularizer, i.e. an anti-memorization device.
  During training, 5% of the sticky-note connections get randomly,
  temporarily blanked out on each pass. **Analogy:** studying with random
  pages of your notes covered up each time forces you to actually
  understand the material robustly instead of memorizing "note #7 says X"
  verbatim.
- `bias = "none"` — a minor technical LoRA setting (whether to also train
  certain small offset numbers); "none" is the standard efficient default.
- `use_gradient_checkpointing = "unsloth"` — a memory-saving trick during
  training (trades a bit of speed for a lot less GPU memory used) —
  Unsloth's own optimized version of it.

## Part 5 — Loading and formatting the dataset

```python
dataset_dict = load_dataset(
    "json",
    data_files={
        "train": ["dataset_train.jsonl", "dataset_train_real.jsonl"],
        "validation": ["dataset_val.jsonl", "dataset_val_real.jsonl"],
    },
)
```

The dataset is a big pile of example flashcards:
`{ticker, user_query, news, output}`. This loads and merges two sources
(the synthetic generator and the real-headline generator) into one
combined set.

**Train vs. validation split — a core ML concept:**

- **Train set** — the flashcards the model is actually allowed to study
  from and adjust its dials against.
- **Validation ("val") set** — flashcards deliberately withheld from
  studying, used only to check afterward: did the model actually learn the
  general skill, or did it just memorize the specific training flashcards?
  This is exactly like a teacher keeping a few practice problems secret to
  use as the real exam, rather than testing on problems already seen with
  the answers. The generators deliberately hold out specific tickers and
  templates so val genuinely represents "unseen" cases.

```python
alpaca_prompt = """Below is an instruction...
### Instruction:
...
### Input:
Target Stock: {}
User Question: {}
Recent News & Results:
{}
### Response:
{}"""
```

This is the **prompt template** — the fixed "worksheet format" every
flashcard gets stuffed into. It's called "Alpaca-style" (a well-known
instruction-tuning format). The `{}` placeholders get filled in per
example.

```python
def format_prompts(examples):
    texts = []
    for ticker, user_query, news, output in zip(...):
        text = alpaca_prompt.format(ticker, user_query, news, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts }

dataset_dict = dataset_dict.map(format_prompts, batched = True)
```

This runs that template-fill over every row of the dataset, producing one
big text blob per example: instructions + input + the correct answer, all
glued together. `tokenizer.eos_token` ("end of sequence") is a special
marker token appended at the end — it tells the model "this response is
complete, stop here," the same way a period ends a sentence.

## Part 6 — Setting up the actual training run

```python
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False,
    args = SFTConfig(...),
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)],
)
```

`SFTTrainer` ("Supervised Fine-Tuning Trainer") is the machine that
actually runs the study sessions. Unpacking every setting inside
`SFTConfig`:

**`per_device_train_batch_size = 2`, `gradient_accumulation_steps = 4`** —
batch size = how many flashcards the model looks at together before
updating its dials once. **Analogy:** instead of re-grading an answer after
every single flashcard (slow, jittery), review a small stack of 2 at a
time, then adjust. `gradient_accumulation_steps = 4` means: do that 4 times
(2×4 = 8 flashcards' worth) and pool the corrections together before
actually updating the dials — this simulates a bigger batch size than the
GPU's memory could hold directly, at the cost of a few extra passes.

**`warmup_steps = 5`, `learning_rate = 2e-4`** — learning rate = how big a
step to take when correcting a mistake. **Analogy:** adjusting a shower's
temperature knob — a high learning rate is "big confident twists" (fast
progress, but risk of overshooting); a low learning rate is "tiny careful
twists" (safe, but slow). `2e-4` (0.0002) is a small, standard, safe value
for LoRA fine-tuning. `warmup_steps = 5` means: start with even tinier
twists for the first 5 steps and ramp up to full strength, avoiding a
jolting, unstable start.

**`num_train_epochs = 5`** — an epoch = one full pass through the entire
training set. 5 here is a ceiling, not a target — "train for at most 5 full
read-throughs of the flashcard deck." Early stopping (below) decides when
to actually stop.

**`eval_strategy = "steps"`, `eval_steps = 50`** — every 50 training steps,
pause and quiz the model on the validation set (the flashcards it never
studies from) and record how it does. This is the "practice exam
checkpoint."

**`weight_decay = 0.01`** — another anti-memorization regularizer,
alongside `lora_dropout`. It gently nudges all the trainable dials toward
smaller values by default, discouraging the model from developing any
single wildly-overconfident dial setting that only makes sense for one
memorized flashcard.

**`save_strategy = "steps"`, `save_steps = 50`, `save_total_limit = 3`,
`load_best_model_at_end = True`, `metric_for_best_model = "eval_loss"`,
`greater_is_better = False`** — loss = a single number measuring how wrong
the model's predictions were (lower is better). Every 50 steps (matching
the eval checkpoints above), a snapshot ("checkpoint") of the model's
current dial settings gets saved to disk, keeping only the 3 most recent.
`load_best_model_at_end = True` means: at the very end, instead of keeping
whatever the last checkpoint happened to be, go back and reload whichever
checkpoint had the lowest eval_loss — the snapshot that actually did best
on the "unseen practice exam," not just whichever ran last. This is a
direct defense against overfitting: performance on unseen data often gets
worse the longer training continues, even while it looks better and better
on the training data being memorized.

**`fp16`/`bf16`** — which numeric precision format the GPU uses for the
math during training. Auto-detects whichever one Colab's assigned GPU
supports better; not something that needs manual tuning.

**`logging_steps = 1`, `optim = "adamw_8bit"`, `output_dir = "outputs"`** —
print progress after every step (to watch loss live); `adamw_8bit` is the
specific algorithm used to decide how to adjust the dials given an error
signal (Adam is the field's default optimizer, `8bit` a memory-compressed
version of it); `output_dir` is where checkpoints get saved.

**`callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]`** —
if eval_loss hasn't improved for 3 consecutive checkpoints (150 steps) in a
row, stop training entirely, even short of the epoch ceiling. **Analogy:**
a student who keeps re-reading the same study guide — after a while, more
re-reading just means memorizing filler details rather than genuinely
improving, and can start actively hurting exam performance. Early stopping
says "scores stopped improving three checks in a row — stop here."

## Part 7 — Completion-only loss masking (and its real tokenization gotcha)

```python
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "### Instruction:\n\n",
    response_part = "### Response:\n\n",
)
```

By default, the trainer measures "how wrong was the model" across the
entire text blob — including the boilerplate instructions and input
section, which is nearly identical across every example. **Analogy:**
grading an exam by including how accurately a student copied the exam
question onto their answer sheet — they'll score well at that part just
from habit, inflating the grade without proving they understood anything.
`train_on_responses_only` tells the trainer: only grade (and only learn
from) the `### Response:` portion — ignore the boilerplate.

This is a real, silent-failure-prone gotcha worth internalizing: this
function works by searching for the *exact token sequence* of
`instruction_part`/`response_part` inside the tokenized data. Because
`alpaca_prompt` has a blank line after each header, the actual text
contains a double newline, which the tokenizer treats as one distinct
"unit" different from a single newline — a marker string ending in a
single `\n` will never be found inside text that actually contains `\n\n`
at that position, even though it looks like a substring to a human eye.
When that happens, every sample gets fully masked and there's nothing left
to learn from. Any time this marker changes, verify it by tokenizing both
the marker in isolation and a full formatted example, and confirming the
marker's token sequence actually appears inside the example's — don't
assume a string that "looks like a substring" tokenizes as one.

## Part 8 — Actually training

```python
trainer.train()
```

This one line kicks off the whole study loop: read a batch, make
predictions, measure loss on just the response portion, nudge the LoRA
sticky-note dials a small step, repeat — pausing every 50 steps to check
the hidden practice exam, until either the epoch ceiling is hit or early
stopping fires.

## Part 9 — Publishing the finished model

```python
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
HF_USER = userdata.get("HF_USER")
MODEL_VERSION_DEFAULT = "v1"  # bumped by bump_model_version.py
MODEL_VERSION = userdata.get("MODEL_VERSION") or MODEL_VERSION_DEFAULT  # optional Secret override, see README
HF_REPO = f"{HF_USER}/{MODEL_CHOICE}-financial-reasoner-{MODEL_VERSION}"

model.push_to_hub_merged(HF_REPO, tokenizer, save_method = "lora", token = HF_TOKEN)
```

(The real script's version in `colab/train/gpu/train_model.py` wraps that `MODEL_VERSION` line
in a try/except, since `userdata.get(...)` raises if the "MODEL_VERSION" Secret was
never created at all - simplified here for readability.)

`userdata.get(...)` pulls Hugging Face upload credentials from Colab's
secure Secrets vault — never hardcoded in the script. `push_to_hub_merged`
takes the original textbook (base model) plus the trained sticky notes
(LoRA weights) and merges them into one combined, standalone model, then
uploads it to the target Hugging Face repo — the final, usable artifact
that `financial-sentiment-api`'s `inference.py` calls at runtime.

## LLM reasoning
1. Base model already knows language and finance vocabulary. It starts from a pretrained instruct model (Llama-3.1-8B-Instruct by default on the GPU/RunPod paths as of v1, Llama-3.2-3B-Instruct on the TPU path; Qwen/Mistral/Apertus are swappable via config).
2. Fine-tuning teaches the specific skill. LoRA adapters are trained via supervised fine-tuning on labeled examples: {ticker, question, market_data, valuation, earnings, news} → {reasoning, recommendation, confidence, answer}.
3. Loss is masked to only the response. During training, the model is only scored on generating the ### Response: JSON, not on reproducing the input boilerplate — so it specifically learns "given this input shape, produce this output shape."
4. The dataset explicitly teaches signal-weighting. A big chunk of the training scenarios are mixed-signal on purpose — e.g. "beat earnings but cut guidance," "missed revenue but announced a buyback" — each hand-labeled with which signal should win and why, reinforcing the pattern "forward-looking/concrete signals beat backward-looking/routine ones," not just "positive word → BUY."
5. Confidence is tiered in training too. Clear-cut examples get labeled with high confidence; subtle or conflicting ones get labeled lower — so the model learns to calibrate confidence, not just pick a direction.
6. Phrasing is deliberately varied across similar examples — an earlier version showed the model pattern-matching a near-verbatim reasoning sentence and misapplying it to unrelated cases — so training examples now vary sentence structure to force it to generalize the underlying reasoning, not memorize a phrase.
7. At inference, no retrieval happens. A brand-new ticker/news combination the model has never seen gets reasoning generated by generalizing learned patterns onto the actual input tokens via attention — not by matching to a stored example.
8. One rule is also hardcoded directly in the prompt template itself ("Weigh guidance cuts and revenue misses higher than minor operational wins") as a redundant nudge on top of what training already teaches.