"""One-off smoke test for serve/modal/serve.py's deployed endpoint - a
real Task A (news_reaction classification) prompt, not a toy query.
Reads the auth token from MODAL_ENDPOINT_TOKEN so it never appears in the
command itself. Not part of the eval/training pipeline, just a manual
post-deploy check.

Usage (PowerShell):
    $env:MODAL_ENDPOINT_TOKEN = "..."; python serve/modal/test_endpoint.py

Usage (bash):
    MODAL_ENDPOINT_TOKEN=... python serve/modal/test_endpoint.py
"""

import os
import sys
import time

import httpx

ENDPOINT_URL = "https://giulianoaparecido--financial-sentiment-reasoner-generate.modal.run"

TASK_A_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

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

Target Stock: AAPL
Recent Price Move: AAPL moved -6.2% on the day this was published.

Recent News & Results:
- [Thu, 20 Aug 2026] Apple recalls MacBook Pro batteries over fire risk - Reuters

### Response:

"""


def main():
    token = os.environ.get("MODAL_ENDPOINT_TOKEN")
    if not token:
        print("Set MODAL_ENDPOINT_TOKEN first (see this file's docstring).", file=sys.stderr)
        sys.exit(1)

    payload = {"inputs": TASK_A_PROMPT, "parameters": {"max_new_tokens": 48}}
    headers = {"Authorization": f"Bearer {token}"}

    print(f"POSTing to {ENDPOINT_URL} ...")
    print("First request after a fresh deploy is a cold start - can take 2+ minutes. Waiting...")
    start = time.monotonic()
    response = httpx.post(ENDPOINT_URL, json=payload, headers=headers, timeout=300.0)
    elapsed = time.monotonic() - start

    print(f"\nStatus: {response.status_code}  (took {elapsed:.1f}s)")
    print(f"Body: {response.text}")


if __name__ == "__main__":
    main()
