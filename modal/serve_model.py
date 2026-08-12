"""
Serves the fine-tuned financial-reasoner model on Modal, as an alternative
to ../colab/run/run_model.py's Colab+ngrok tunnel. Same model-loading call
(FastLanguageModel.from_pretrained, 4-bit, HF_USER/MODEL_CHOICE/
MODEL_VERSION resolution) and the exact same /generate request/response
shape ({"inputs": ..., "parameters": {...}} in, [{"generated_text": ...}]
out) - financial-sentiment-api's app/services/inference.py needs no code
change, only repointing HF_INFERENCE_URL (its existing runtime-mutable
/api/update-inference-url endpoint already exists for exactly this kind of
swap) and matching HF_API_TOKEN to ENDPOINT_AUTH_TOKEN below.

One real behavior difference from run_model.py: this actually reads
parameters.max_new_tokens from the request instead of ignoring it and
hardcoding 350. inference.py has sent up to 512 since the v4 answer-field
change, which the Colab server has been silently discarding - this fixes
that rather than reproducing it. temperature/return_full_text are still
unused (matching today's actual generation call, which is greedy decoding,
not sampling).

Setup (one-time, under your own Modal account - `modal deploy` cannot run
without it):
    pip install modal fastapi
    modal setup
    modal secret create financial-sentiment-model-secrets \
        HF_USER=<your-hf-username> \
        ENDPOINT_AUTH_TOKEN=<pick-a-random-secret-string> \
        HF_TOKEN=<only-if-the-model-repo-is-private>

Deploy:
    modal deploy modal/serve_model.py

Cost shape - the whole point of this file over an always-on host:
  - min_containers is deliberately NOT set (defaults to 0): the GPU
    container, and its billing, fully disappears between requests. Every
    request pays a cold start instead.
  - max_containers=1: hard cap so no burst of concurrent requests (or
    abuse against a leaked URL) can multiply cost - this is a single-user
    research tool, there's no legitimate case for concurrency here.
  - scaledown_window=60: a container survives 1 minute after its last
    request before scaling back to zero, so a quick follow-up query
    reuses the warm container instead of paying another cold start - but
    still bounded, since idle time in that window is billed at the same
    GPU rate as active compute (confirmed against Modal's own autoscaling
    docs - the pricing page's "never pay for idle" language means fully
    scaled-to-zero, not "warm but unused"). At this project's real usage
    (a handful of one-off queries a day, rarely two within the same
    couple minutes), the window mostly goes unused anyway - 60s instead
    of the original 120s just halves what gets spent on follow-ups that
    don't happen, without giving up the benefit for the ones that do.
  - timeout=300 on the GPU class: caps worst-case cost from a single
    request that somehow hangs, while still leaving room for a real cold
    start. Confirmed live this needs to be generous, not tight - a cold
    request (container boot + unsloth/torch import + weight load) alone
    measured ~120s for a trivial 20-token generation, so the original
    timeout=120 here was cutting off a real 512-token request (financial-
    sentiment-api's actual max_new_tokens) before it could ever finish,
    silently, since Modal kills the call rather than raising an app-level
    error. See inference.py's httpx timeout and financial-sentiment-web's
    proxy maxDuration/AbortSignal - both had to be raised to match, since
    a shorter timeout anywhere upstream just moves where the same request
    dies.
"""

import os

import modal
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

app = modal.App("financial-sentiment-reasoner")

MODEL_CHOICE_DEFAULT = "llama-3.2-3b"
MODEL_VERSION_DEFAULT = "v10"

# unsloth is the one dependency most likely to need a version pin on your
# first `modal deploy` - it's picky about matching torch/CUDA versions, and
# that pairing can only really be confirmed by actually building the image
# (not something verifiable without a Modal account). If the build fails on
# unsloth, pin it and torch together per https://docs.unsloth.ai/get-started/installing-+-updating.
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "unsloth",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "fastapi[standard]",
)

# Persists downloaded model weights across cold starts. Without this, every
# scale-to-zero -> scale-to-one cycle re-downloads the full model from
# Hugging Face, which dwarfs the actual load-from-disk time.
model_cache = modal.Volume.from_name("financial-sentiment-model-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/cache"

secrets = [modal.Secret.from_name("financial-sentiment-model-secrets")]


@app.cls(
    image=image,
    gpu="T4",
    volumes={MODEL_CACHE_DIR: model_cache},
    secrets=secrets,
    scaledown_window=60,
    max_containers=1,
    timeout=300,
)
class Model:
    @modal.enter()
    def load(self) -> None:
        os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)
        from unsloth import FastLanguageModel

        hf_user = os.environ["HF_USER"]
        model_choice = os.environ.get("MODEL_CHOICE") or MODEL_CHOICE_DEFAULT
        model_version = os.environ.get("MODEL_VERSION") or MODEL_VERSION_DEFAULT
        model_name = f"{hf_user}/{model_choice}-financial-reasoner-{model_version}"

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            token=os.environ.get("HF_TOKEN"),
        )
        FastLanguageModel.for_inference(self.model)

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class Parameters(BaseModel):
    max_new_tokens: int = 350
    temperature: float | None = None
    return_full_text: bool | None = None


class GenerateRequest(BaseModel):
    inputs: str
    parameters: Parameters = Parameters()


auth_scheme = HTTPBearer()


@app.function(image=image, secrets=secrets)
@modal.fastapi_endpoint(method="POST")
async def generate(
    req: GenerateRequest,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
) -> list[dict]:
    if token.credentials != os.environ["ENDPOINT_AUTH_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    generated_text = Model().generate.remote(req.inputs, req.parameters.max_new_tokens)
    return [{"generated_text": generated_text}]
