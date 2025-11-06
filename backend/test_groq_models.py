"""
Test different Groq models to see which ones work with browser-use.
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def test_groq_models():
    """List available Groq models and test structured output support"""
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY is not set")
        return
    
    print("🧪 Testing Groq models...")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    
    # List available models
    print("\n📋 Available Groq models:")
    try:
        models = client.models.list()
        for model in models:
            print(f"   - {model.id}")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return
    
    # Test structured output with different models
    test_models = [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
    ]
    
    print("\n🧪 Testing structured output support:")
    
    for model_name in test_models:
        print(f"\n📤 Testing {model_name}...")
        try:
            # Test with a simple schema that includes minimum
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Return a number between 1 and 10"}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "number_response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "number": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10
                                }
                            },
                            "required": ["number"]
                        }
                    }
                },
                temperature=0.2,
            )
            print(f"   ✅ Success: {response.choices[0].message.content}")
        except Exception as e:
            error_msg = str(e)
            if "minimum is not supported" in error_msg:
                print(f"   ❌ Does not support 'minimum' in schema")
            elif "not found" in error_msg or "does not exist" in error_msg:
                print(f"   ⚠️  Model not available")
            else:
                print(f"   ❌ Error: {error_msg[:100]}")

if __name__ == "__main__":
    test_groq_models()
