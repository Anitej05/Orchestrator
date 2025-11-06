"""
Test direct Cerebras API call to see what works and what doesn't.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def test_cerebras_direct():
    """Test direct Cerebras API call"""
    
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        print("❌ ERROR: CEREBRAS_API_KEY is not set")
        return
    
    print("🧪 Testing direct Cerebras API call...")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1"
    )
    
    # Test 1: Simple chat completion
    print("\n📤 Test 1: Simple chat completion")
    try:
        response = client.chat.completions.create(
            model="qwen-3-235b-a22b-instruct-2507",
            messages=[{"role": "user", "content": "Say hello"}],
            temperature=0.2,
        )
        print(f"✅ Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: With system message
    print("\n📤 Test 2: With system message")
    try:
        response = client.chat.completions.create(
            model="qwen-3-235b-a22b-instruct-2507",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"}
            ],
            temperature=0.2,
        )
        print(f"✅ Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: With function calling (this might be the issue)
    print("\n📤 Test 3: With function/tool calling")
    try:
        response = client.chat.completions.create(
            model="qwen-3-235b-a22b-instruct-2507",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }],
            temperature=0.2,
        )
        print(f"✅ Response: {response.choices[0].message}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   This is likely the issue - Cerebras may not support function calling")
    
    # Test 4: Check model list
    print("\n📤 Test 4: List available models")
    try:
        models = client.models.list()
        print("✅ Available models:")
        for model in models:
            print(f"   - {model.id}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_cerebras_direct()
