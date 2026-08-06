# Cell 2: Load Model + Start FastAPI + ngrok

import nest_asyncio
import uvicorn
from pyngrok import ngrok
from fastapi import FastAPI
from pydantic import BaseModel
from unsloth import FastLanguageModel
import asyncio
import threading    # ← ADD THIS LINE
import time         # ← ADD THIS LINE

# get_secret() works on both Colab (Secrets, key icon in the left sidebar)
# and Kaggle (Add-ons -> Secrets) - same secret name, different underlying
# API. update_backend.py's cell later in this same session reuses this
# function too, the same way it already reuses `endpoint` below.
def get_secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except ModuleNotFoundError:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)

# 1. Load fine-tuned weights directly from Hugging Face
HF_USER = get_secret("HF_USER")
MODEL_NAME = f"{HF_USER}/llama-3.2-3b-financial-reasoner-v2"  # or f"{HF_USER}/apertus-0.5b-financial-reasoner"

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