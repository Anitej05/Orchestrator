"""
Test script to verify Cerebras API connectivity and model availability.
"""
import asyncio
import os
from dotenv import load_dotenv
from browser_use import ChatOpenAI

load_dotenv()

async def test_cerebras_text_model():
    """Test the Cerebras text-only model with retry logic"""
    
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        print("❌ ERROR: CEREBRAS_API_KEY is not set in .env file")
        return False
    
    print("🧪 Testing Cerebras text-only model...")
    print(f"   Model: qwen-3-235b-a22b-instruct-2507")
    print(f"   API Key: {api_key[:10]}...")
    
    from openai import OpenAI
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1"
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Test with a simple prompt
            test_prompt = "Say 'Hello from Cerebras!' and nothing else."
            print(f"\n📤 Attempt {attempt + 1}/{max_retries}: Sending test prompt")
            
            response = client.chat.completions.create(
                model="qwen-3-235b-a22b-instruct-2507",
                messages=[{"role": "user", "content": test_prompt}],
                temperature=0.2,
                max_tokens=50,
            )
            
            print(f"✅ Response received: {response.choices[0].message.content}")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate" in error_msg or "too_many_requests" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⏳ Rate limit hit. Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Rate limit persists after {max_retries} attempts")
                    return False
            else:
                print(f"❌ Error connecting to Cerebras: {e}")
                print(f"   Error type: {type(e).__name__}")
                return False
    
    return False

async def test_vision_decision():
    """Test the vision decision logic"""
    
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        print("❌ ERROR: CEREBRAS_API_KEY is not set")
        return
    
    print("\n🧪 Testing vision decision logic...")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1"
        )
        
        # Test cases
        test_cases = [
            ("Take a screenshot of google.com", True),
            ("Navigate to github.com and click the login button", False),
            ("Describe what you see on the homepage", True),
            ("Fill out the form with my name and email", False),
        ]
        
        for task, expected_vision in test_cases:
            decision_prompt = f"""Analyze this browser automation task and determine if visual analysis (screenshots, image recognition) is required.

Task: {task}

Consider vision is needed if the task involves:
- Taking screenshots
- Analyzing visual content or layout
- Describing what's visible on a page
- Checking visual elements, colors, or design
- Comparing visual appearances
- Any task that requires "seeing" the page

Respond with ONLY "YES" if vision is needed, or "NO" if text-based interaction is sufficient."""

            response = client.chat.completions.create(
                model="qwen-3-235b-a22b-instruct-2507",
                messages=[{"role": "user", "content": decision_prompt}],
                temperature=0.2,
            )
            
            decision = response.choices[0].message.content.strip().upper()
            needs_vision = "YES" in decision
            
            status = "✅" if needs_vision == expected_vision else "⚠️"
            print(f"{status} Task: '{task[:50]}...'")
            print(f"   Decision: {decision} (Vision: {needs_vision})")
        
    except Exception as e:
        print(f"❌ Error testing vision decision: {e}")

async def main():
    print("=" * 70)
    print("Cerebras API Connection Test")
    print("=" * 70)
    
    # Test 1: Basic connectivity
    success = await test_cerebras_text_model()
    
    if success:
        # Test 2: Vision decision logic
        await test_vision_decision()
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Connection test failed. Please check your API key and network.")

if __name__ == "__main__":
    asyncio.run(main())
