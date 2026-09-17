"""
Serves the apertus-8b-financial-reasoner model on Modal, as a separate
endpoint from serve_model.py's llama deployment. Same model-loading call,
request/response shape, and autoscaling config as serve_model.py - see that
file's own docstring for the full rationale (cold start budget, cost shape,
timeout sizing) - except the GPU: L4 rather than T4, see the comment on the
class decorator, and price the cost shape accordingly. This file exists to give apertus its own
Modal app/URL so financial-sentiment-api's ?model=apertus routing (see
APERTUS_INFERENCE_URL) can hit it independently of the llama endpoint,
without either one's traffic/cold-starts affecting the other.

Setup (one-time, under your own Modal account):
    modal secret create financial-sentiment-model-secrets \
        HF_USER=<your-hf-username> \
        ENDPOINT_AUTH_TOKEN=<pick-a-random-secret-string> \
        HF_TOKEN=<only-if-the-model-repo-is-private>
    (reuses the same secret as serve_model.py if you already created it)

Deploy:
    modal deploy modal/serve_apertus.py
"""

import os
import secrets

import modal
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

app = modal.App("financial-sentiment-reasoner-apertus")

MODEL_CHOICE_DEFAULT = "apertus-8b"
MODEL_VERSION_DEFAULT = "v1"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "unsloth",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "fastapi[standard]",
)

# Separate volume from llama's, so cache size/contents for one model never
# affects the other's cold-start behavior.
model_cache = modal.Volume.from_name("financial-sentiment-model-cache-apertus", create_if_missing=True)
MODEL_CACHE_DIR = "/cache"

modal_secrets = [modal.Secret.from_name("financial-sentiment-model-secrets")]


# L4, not the T4 serve_model.py uses: Apertus is bf16-trained and overflows
# to NaN in fp16 (confirmed live on a T4 - all-NaN logits, every generated
# token id 0 / <unk>, empty output), and the T4 has no bf16 support, so
# Unsloth silently downgrades to fp16 there. Any bf16-capable GPU works.
@app.cls(
    image=image,
    gpu="L4",
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
        # Both the upstream swiss-ai/Apertus-8B config.json and this merged
        # checkpoint ship use_cache=false, and the merged push has no
        # generation_config.json to override it, so the KV cache is off by
        # default and a 512-token call blows the 300s timeout. This is
        # inherited from upstream, not a training artifact - a retrain won't
        # remove it. Passing use_cache=True to generate() below did not fix
        # it under Unsloth's generic path for this architecture (the hang
        # reproduced with it present), so pin it on both configs.
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
