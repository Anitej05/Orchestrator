"""
Test Ollama Cloud Vision API - Simple direct test
"""
import requests
import base64
import os
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')

print("="*60)
print("TESTING OLLAMA CLOUD VISION")
print("="*60)

if not OLLAMA_API_KEY:
    print("ERROR: OLLAMA_API_KEY not set")
    exit(1)

print(f"API Key: {OLLAMA_API_KEY[:10]}...")

# Create a test image
img = Image.new('RGB', (200, 200), color='red')
buffer = io.BytesIO()
img.save(buffer, format='PNG')
image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
print(f"Image size: {len(image_b64)} chars")

# Test different message formats

# Format 1: images in message (native Ollama)
print("\n[1] Format: images array in message...")
try:
    response = requests.post(
        "https://ollama.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OLLAMA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "qwen3-vl:235b-cloud",
            "messages": [
                {
                    "role": "user",
                    "content": "Describe the color of this image in one word.",
                    "images": [image_b64]
                }
            ]
        },
        timeout=60
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"Response: {content[:300]}")
    else:
        print(f"Error: {response.text[:300]}")
except Exception as e:
    print(f"Exception: {e}")

# Format 2: OpenAI Vision style (multipart content)
print("\n[2] Format: OpenAI vision style (content array)...")
try:
    response = requests.post(
        "https://ollama.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OLLAMA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "qwen3-vl:235b-cloud",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the color of this image in one word."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ]
        },
        timeout=60
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"Response: {content[:300]}")
    else:
        print(f"Error: {response.text[:300]}")
except Exception as e:
    print(f"Exception: {e}")

# Format 3: Native Ollama API (not OpenAI compatible)
print("\n[3] Format: Native Ollama /api/chat endpoint...")
try:
    response = requests.post(
        "https://ollama.com/api/chat",
        headers={
            "Authorization": f"Bearer {OLLAMA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "qwen3-vl:235b-cloud",
            "messages": [
                {
                    "role": "user",
                    "content": "Describe the color of this image in one word.",
                    "images": [image_b64]
                }
            ],
            "stream": False
        },
        timeout=60
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        content = data.get("message", {}).get("content", "") or data.get("response", "")
        print(f"Response: {content[:300]}")
    else:
        print(f"Error: {response.text[:300]}")
except Exception as e:
    print(f"Exception: {e}")

print("\n" + "="*60)
