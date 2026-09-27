"""
Serves one fine-tuned financial-reasoner model on Modal as its own
scale-to-zero app, as an alternative to ../notebooks/run/run_model.py's
Colab+ngrok tunnel. Same model-loading call (FastLanguageModel.from_pretrained,
4-bit, HF_USER/MODEL_CHOICE/MODEL_VERSION resolution) and the exact same
/generate request/response shape ({"inputs": ..., "parameters": {...}} in,
[{"generated_text": ...}] out) - financial-sentiment-api registers the
resulting URL as <NAME>_INFERENCE_URL (or via /api/update-inference-url) and
matches HF_TOKEN to ENDPOINT_AUTH_TOKEN below.

One deploy per model, parameterized by env vars read at `modal deploy` time:

    MODEL_CHOICE   required - the HF repo infix, e.g. llama-3.1-8b, apertus-8b
                   (repo = <HF_USER>/<MODEL_CHOICE>-financial-reasoner-<MODEL_VERSION>)
    MODEL_VERSION  default v1
    MODEL          short name used by financial-sentiment-api's ?model= and in
                   the app/volume names; default: MODEL_CHOICE up to its first
                   "-" (llama-3.1-8b -> llama, apertus-8b -> apertus)
    GPU            default T4. Any model that is bf16-trained and overflows to
                   NaN in fp16 needs a bf16-capable GPU (L4 or better): the T4
                   has no bf16, and Unsloth silently downgrades to fp16 there -
                   confirmed live with apertus-8b (all-NaN logits, every token
                   id 0, empty output). llama-3.1-8b tolerates fp16 on a T4.

    MODEL_CHOICE=apertus-8b GPU=L4 modal deploy modal/serve.py
    MODEL_CHOICE=llama-3.1-8b        modal deploy modal/serve.py

Each model gets app financial-sentiment-reasoner-<MODEL> and volume
financial-sentiment-model-cache-<MODEL>, so one model's traffic, cold starts
and weight cache never affect another's.

Setup (one-time, under your own Modal account):
    pip install modal fastapi
    modal setup
    modal secret create financial-sentiment-model-secrets \
        HF_USER=<your-hf-username> \
        ENDPOINT_AUTH_TOKEN=<pick-a-random-secret-string> \
        HF_TOKEN=<only-if-the-model-repo-is-private>

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
    couple minutes), the window mostly goes unused anyway.
  - timeout=300 on the GPU class: caps worst-case cost from a single
    request that somehow hangs, while still leaving room for a real cold
    start. Confirmed live this needs to be generous, not tight - a cold
    request (container boot + unsloth/torch import + weight load) alone
    measured ~120s for a trivial 20-token generation, so a shorter
    timeout cuts off a real 512-token request (financial-sentiment-api's
    actual max_new_tokens) before it can finish, silently, since Modal
    kills the call rather than raising an app-level error. See
    inference.py's httpx timeout and financial-sentiment-web's proxy
    maxDuration/AbortSignal - both had to be raised to match, since a
    shorter timeout anywhere upstream just moves where the same request
    dies.
"""

import os
import secrets

import modal
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

MODEL_CHOICE = os.environ.get("MODEL_CHOICE")
if not MODEL_CHOICE:
    raise SystemExit("MODEL_CHOICE is required, e.g.: MODEL_CHOICE=apertus-8b GPU=L4 modal deploy modal/serve.py")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
MODEL = os.environ.get("MODEL") or MODEL_CHOICE.split("-", 1)[0]
GPU = os.environ.get("GPU", "T4")
DEPLOY_PARAMS = {"MODEL_CHOICE": MODEL_CHOICE, "MODEL_VERSION": MODEL_VERSION, "MODEL": MODEL, "GPU": GPU}

app = modal.App(f"financial-sentiment-reasoner-{MODEL}")

# unsloth is the one dependency most likely to need a version pin on your
# first `modal deploy` - it's picky about matching torch/CUDA versions, and
# that pairing can only really be confirmed by actually building the image
# (not something verifiable without a Modal account). If the build fails on
# unsloth, pin it and torch together per https://docs.unsloth.ai/get-started/installing-+-updating.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "unsloth",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "fastapi[standard]",
    )
    # Baked in so the container-side import of this module sees the same
    # deploy-time parameters the local `modal deploy` did.
    .env(DEPLOY_PARAMS)
)

# Persists downloaded model weights across cold starts. Without this, every
# scale-to-zero -> scale-to-one cycle re-downloads the full model from
# Hugging Face, which dwarfs the actual load-from-disk time.
model_cache = modal.Volume.from_name(f"financial-sentiment-model-cache-{MODEL}", create_if_missing=True)
MODEL_CACHE_DIR = "/cache"

modal_secrets = [modal.Secret.from_name("financial-sentiment-model-secrets")]


@app.cls(
    image=image,
    gpu=GPU,
    volumes={MODEL_CACHE_DIR: model_cache},
    secrets=modal_secrets,
    scaledown_window=60,
    max_containers=1,
    timeout=300,
)
class Model:
    @modal.enter()
    def load(self) -> None:
        os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)
        from unsloth import FastLanguageModel

        model_name = f"{os.environ['HF_USER']}/{os.environ['MODEL_CHOICE']}-financial-reasoner-{os.environ['MODEL_VERSION']}"

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            token=os.environ.get("HF_TOKEN"),
        )
        FastLanguageModel.for_inference(self.model)
        # Some checkpoints ship use_cache=false in config.json with no
        # generation_config.json to override it (swiss-ai/Apertus-8B does,
        # and every fine-tune inherits it), which leaves the KV cache off
        # and blows the 300s timeout on a 512-token call. Passing
        # use_cache=True to generate() below did not fix it under Unsloth's
        # generic path (the hang reproduced with it present), so pin it on
        # both configs; a no-op for checkpoints that already have it on.
        self.model.config.use_cache = True
        self.model.generation_config.use_cache = True

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


@app.function(image=image, secrets=modal_secrets)
@modal.fastapi_endpoint(method="POST")
async def generate(
    req: GenerateRequest,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
) -> list[dict]:
    if not secrets.compare_digest(token.credentials, os.environ["ENDPOINT_AUTH_TOKEN"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    generated_text = await Model().generate.remote.aio(req.inputs, req.parameters.max_new_tokens)
    return [{"generated_text": generated_text}]
