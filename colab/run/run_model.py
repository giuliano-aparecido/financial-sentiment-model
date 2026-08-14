# Cell 2: Load Model + Start FastAPI + ngrok

import nest_asyncio
import uvicorn
from pyngrok import ngrok
from fastapi import FastAPI
from pydantic import BaseModel
from unsloth import FastLanguageModel
import asyncio
import os
import threading    # ← ADD THIS LINE
import time         # ← ADD THIS LINE

# get_secret() works on both Colab (Secrets, key icon in the left sidebar)
# and Kaggle (Add-ons -> Secrets) - same secret name, different underlying
# API. update_backend.py's cell later in this same session reuses this
# function too, the same way it already reuses `endpoint` below.
#
# Checking whether `google.colab` IMPORTS is not a reliable way to detect
# Colab vs Kaggle - confirmed live: some Kaggle base images ship a
# google-colab package too, so the import succeeds there, and the
# ModuleNotFoundError this used to branch on never fires. The actual
# Colab RPC then just hangs and times out ("Secrets can only be fetched
# when running from the Colab UI") instead of falling through to
# kaggle_secrets. KAGGLE_KERNEL_RUN_TYPE is set by Kaggle's own runtime
# on every notebook, so check that directly instead of inferring the
# platform from import success.
def get_secret(name):
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    from google.colab import userdata
    return userdata.get(name)

# 1. Load fine-tuned weights directly from Hugging Face
HF_USER = get_secret("HF_USER")

# MODEL_CHOICE_DEFAULT is the git-committed baseline. Add an OPTIONAL
# "MODEL_CHOICE" Colab/Kaggle Secret to serve a different base-model
# family ad-hoc, without editing this file - must match whatever
# MODEL_CHOICE the target repo was actually trained/pushed under (e.g.
# "apertus-0.5b" instead of the default "llama-3.2-3b").
MODEL_CHOICE_DEFAULT = "llama-3.2-3b"
try:
    MODEL_CHOICE = get_secret("MODEL_CHOICE") or MODEL_CHOICE_DEFAULT
except Exception:
    MODEL_CHOICE = MODEL_CHOICE_DEFAULT

# MODEL_VERSION_DEFAULT is the git-committed baseline (bumped by
# bump_model_version.py, at the repo root). Add an OPTIONAL "MODEL_VERSION"
# Colab/Kaggle Secret to serve a different push ad-hoc, without editing
# this file - useful for a quick rollback if a newly-trained version turns
# out worse than the one it replaced.
MODEL_VERSION_DEFAULT = "v16"
try:
    MODEL_VERSION = get_secret("MODEL_VERSION") or MODEL_VERSION_DEFAULT
except Exception:
    MODEL_VERSION = MODEL_VERSION_DEFAULT

MODEL_NAME = f"{HF_USER}/{MODEL_CHOICE}-financial-reasoner-{MODEL_VERSION}"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# 2. Setup FastAPI App
app = FastAPI()

class InferenceRequest(BaseModel):
    inputs: str

@app.post("/generate")
def generate(req: InferenceRequest):
    # 1. Tokenize input
    inputs = tokenizer([req.inputs], return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    # 2. Generate new tokens
    outputs = model.generate(**inputs, max_new_tokens=350, use_cache=True)

    # 3. Slice out the prompt tokens and decode ONLY newly generated tokens
    new_tokens = outputs[0][input_len:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return [{"generated_text": generated_text.strip()}]

# 3. Authenticate & Connect Ngrok
# Get your free token at: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = get_secret("NGROK_AUTH_TOKEN")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(8000).public_url
endpoint = public_url + "/generate"

print(f"\n🚀 SUCCESS! YOUR ENDPOINT IS LIVE AT:\n{endpoint}\n")

# 2. Configure Uvicorn Server
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)

# 3. Use top-level await to attach to Colab's running event loop
def run_server():
    asyncio.run(server.serve())

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

print("✅ FastAPI server started in background")
time.sleep(2)