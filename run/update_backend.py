# Cell 3: Auto-update Render with new ngrok URL

import requests
import json
from datetime import datetime

# Configuration
RENDER_WEBHOOK_URL = "https://financial-sentiment-api-qs45.onrender.com/api/update-inference-url"
# get_secret() is defined in run_model.py's cell, still in scope from the
# same session (same pattern as reusing `endpoint` below) - works on both
# Colab and Kaggle. Must match the API_KEY set in Render's own env vars
# for this to authenticate.
RENDER_SECRET_KEY = get_secret("RENDER_SECRET_KEY")

def send_url_to_render(ngrok_url):
    """Send the new ngrok URL to your Render app"""
    payload = {
        'url': ngrok_url
    }

    try:
        response = requests.post(
            RENDER_WEBHOOK_URL,
            json=payload,
            headers={'X-API-Key': RENDER_SECRET_KEY},
            timeout=5,
        )
        if response.status_code == 200:
            print(f"✅ Successfully sent URL to Render: {ngrok_url}")
            return True
        else:
            print(f"❌ Failed to send URL to Render: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ Error sending URL to Render: {str(e)}")
        return False

# Send the URL immediately after ngrok connects
print(f"Sending ngrok URL to Render...")
send_url_to_render(endpoint)