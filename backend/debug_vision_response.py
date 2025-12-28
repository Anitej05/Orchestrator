import asyncio
import os
import base64
import time
import io
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Try importing PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: PIL not found, using dummy large string")

load_dotenv()

# Configuration
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
MODEL = "qwen3-vl:235b-cloud" 
BASE_URL = "https://ollama.com/v1"

# Generate Image
if PIL_AVAILABLE:
    # 800x600 image with random noise/lines to prevent compression
    import random
    from PIL import ImageDraw
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    # Draw 1000 random lines to add entropy
    for _ in range(1000):
        x1 = random.randint(0, 800)
        y1 = random.randint(0, 600)
        x2 = random.randint(0, 800)
        y2 = random.randint(0, 600)
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        draw.line([(x1,y1), (x2,y2)], fill=color, width=2)
        
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    DUMMY_IMAGE_B64 = base64.b64encode(buf.getvalue()).decode()
else:
    # Fallback if no PIL (unlikely given previous logs)
    # Just repeating a small valid png header + junk might not work for vision model
    # So we hope PIL is there.
    DUMMY_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg=="

print(f"Generated dummy image size: {len(DUMMY_IMAGE_B64)} chars")

# STRICT SYSTEM MESSAGE - matches vision.py exactly (as of last edit)
SYSTEM_PROMPT = """You are a browser automation agent. You MUST respond with ONLY valid JSON.

CRITICAL RULES:
1. Your response MUST be a valid JSON object starting with { and ending with }
2. NO text before or after JSON
3. NO markdown code blocks
4. NO explanation outside JSON
5. DO NOT LOOP: If the previous action was also a vision analysis of the SAME page state, you MUST take a constructive action (click, type, done, or switch to text mode). Do not just describe the page again.
6. EFFICIENCY: Switch to "next_mode": "text" as soon as you have understood the visual layout. Text mode is faster and cheaper.

REQUIRED JSON FORMAT:
{
  "reasoning": "Brief explanation of what you see and your decision effectively.",
  "actions": [{"name": "action_name", "params": {...}}],
  "confidence": 0.9,
  "next_mode": "text"
}

Valid action names: click, type, scroll, navigate, done, wait, hover, press, go_back, save_info, skip_subtask
For clicks: use {"mark": N} where N is the number you see on the screenshot like [1], [2], [3]"""

# User prompt part
USER_PART = "Analyze this screenshot and decide the next action."

async def test_vision():
    print(f"\n\n🔍 Testing Large Payload on {MODEL}...")
    
    client = AsyncOpenAI(
        api_key=OLLAMA_API_KEY or "ollama",
        base_url=BASE_URL,
        timeout=120.0 # generous timeout
    )
    
    # MERGE SYSTEM PROMPT INTO USER MESSAGE
    full_prompt_text = SYSTEM_PROMPT + "\n\n" + USER_PART
    
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": full_prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{DUMMY_IMAGE_B64}"}}
            ]
        }
    ]
    
    try:
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 1200,
            "temperature": 0.1
        }
        
        print(f"📤 Sending request (Prompt len: {len(full_prompt_text)})...")
        start = time.time()
        response = await client.chat.completions.create(**kwargs)
        duration = time.time() - start
        
        print(f"✅ Response Received in {duration:.2f}s!")
        print("-" * 50)
        if response and response.choices:
            content = response.choices[0].message.content
            print(f"RAW CONTENT:\n{content}")
        else:
            print("RAW CONTENT: [None/Empty]")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_vision())
