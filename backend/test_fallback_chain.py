"""
Test the fallback chain: Cerebras → Groq for vision decision logic.
"""
import asyncio
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

async def test_cerebras():
    """Test Cerebras API"""
    print("\n" + "="*70)
    print("TEST 1: Cerebras API (Primary)")
    print("="*70)
    
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        print("❌ CEREBRAS_API_KEY not set")
        return False
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1"
        )
        
        response = client.chat.completions.create(
            model="qwen-3-235b-a22b-instruct-2507",
            messages=[{"role": "user", "content": "Say 'Cerebras works!' and nothing else."}],
            temperature=0.2,
            max_tokens=20,
        )
        
        result = response.choices[0].message.content
        print(f"✅ Cerebras response: {result}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate" in error_msg:
            print(f"⚠️  Cerebras rate limited (expected): {e}")
        else:
            print(f"❌ Cerebras error: {e}")
        return False

async def test_groq():
    """Test Groq API"""
    print("\n" + "="*70)
    print("TEST 2: Groq API (Fallback)")
    print("="*70)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not set")
        return False
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say 'Groq works!' and nothing else."}],
            temperature=0.2,
            max_tokens=20,
        )
        
        result = response.choices[0].message.content
        print(f"✅ Groq response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Groq error: {e}")
        return False

async def test_vision_decision_with_fallback():
    """Test vision decision with fallback chain"""
    print("\n" + "="*70)
    print("TEST 3: Vision Decision with Fallback Chain")
    print("="*70)
    
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not cerebras_key and not groq_key:
        print("❌ Neither CEREBRAS_API_KEY nor GROQ_API_KEY is set")
        return False
    
    decision_prompt = """Analyze this browser automation task and determine if visual analysis (screenshots, image recognition) is required.

Task: Take a screenshot of google.com

Consider vision is needed if the task involves:
- Taking screenshots
- Analyzing visual content or layout
- Describing what's visible on a page
- Checking visual elements, colors, or design
- Comparing visual appearances
- Any task that requires "seeing" the page

Respond with ONLY "YES" if vision is needed, or "NO" if text-based interaction is sufficient."""

    # Try Cerebras first
    if cerebras_key:
        try:
            print("\n🔄 Trying Cerebras...")
            client = OpenAI(
                api_key=cerebras_key,
                base_url="https://api.cerebras.ai/v1"
            )
            
            response = client.chat.completions.create(
                model="qwen-3-235b-a22b-instruct-2507",
                messages=[{"role": "user", "content": decision_prompt}],
                temperature=0.2,
                max_tokens=10,
            )
            
            decision = response.choices[0].message.content.strip().upper()
            print(f"✅ Cerebras decision: {decision}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate" in error_msg:
                print(f"⚠️  Cerebras rate limited, falling back to Groq...")
            else:
                print(f"⚠️  Cerebras failed: {e}, falling back to Groq...")
    
    # Fallback to Groq
    if groq_key:
        try:
            print("\n🔄 Trying Groq...")
            client = OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1"
            )
            
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": decision_prompt}],
                temperature=0.2,
                max_tokens=10,
            )
            
            decision = response.choices[0].message.content.strip().upper()
            print(f"✅ Groq decision: {decision}")
            return True
            
        except Exception as e:
            print(f"❌ Groq also failed: {e}")
            return False
    
    return False

async def main():
    print("="*70)
    print("Fallback Chain Test: Cerebras → Groq")
    print("="*70)
    
    results = []
    
    # Test 1: Cerebras
    result1 = await test_cerebras()
    results.append(("Cerebras API", result1))
    
    # Test 2: Groq
    result2 = await test_groq()
    results.append(("Groq API", result2))
    
    # Test 3: Vision decision with fallback
    result3 = await test_vision_decision_with_fallback()
    results.append(("Vision Decision Fallback", result3))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    # Check if at least one provider works
    at_least_one_works = results[0][1] or results[1][1]
    
    if at_least_one_works and results[2][1]:
        print("\n🎉 Fallback chain is working! At least one provider is available.")
    elif not at_least_one_works:
        print("\n⚠️  Both providers failed. Check your API keys and rate limits.")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")

if __name__ == "__main__":
    asyncio.run(main())
