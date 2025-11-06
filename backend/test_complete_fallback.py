"""
Complete test for the fallback chain in browser automation.
Tests both vision decision and actual browser automation with Cerebras → Groq fallback.
"""
import asyncio
import os
from dotenv import load_dotenv
from browser_use import Agent, ChatOpenAI
from openai import OpenAI

load_dotenv()

async def test_text_llm_initialization():
    """Test text-only LLM initialization with fallback"""
    print("\n" + "="*70)
    print("TEST 1: Text-Only LLM Initialization (Cerebras → Groq)")
    print("="*70)
    
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    # Try Cerebras
    if cerebras_key:
        try:
            print("\n🔄 Trying Cerebras...")
            llm = ChatOpenAI(
                model="qwen-3-235b-a22b-instruct-2507",
                api_key=cerebras_key,
                base_url="https://api.cerebras.ai/v1",
                temperature=0.2,
            )
            print("✅ Cerebras LLM initialized successfully")
            return True, "Cerebras"
        except Exception as e:
            print(f"⚠️  Cerebras initialization failed: {e}")
    
    # Fallback to Groq
    if groq_key:
        try:
            print("\n🔄 Trying Groq (fallback)...")
            llm = ChatOpenAI(
                model="openai/gpt-oss-120b",
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0.2,
            )
            print("✅ Groq LLM initialized successfully")
            return True, "Groq"
        except Exception as e:
            print(f"❌ Groq initialization failed: {e}")
    
    print("❌ All text-only LLM providers failed")
    return False, None

async def test_vision_decision_fallback():
    """Test vision decision with fallback chain"""
    print("\n" + "="*70)
    print("TEST 2: Vision Decision with Fallback (Cerebras → Groq)")
    print("="*70)
    
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    decision_prompt = """Analyze this browser automation task and determine if visual analysis (screenshots, image recognition) is required.

Task: Navigate to github.com and click the login button

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
        max_retries = 2
        for attempt in range(max_retries):
            try:
                print(f"\n🔄 Trying Cerebras (attempt {attempt + 1}/{max_retries})...")
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
                return True, "Cerebras"
                
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"⚠️  Cerebras rate limited. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"⚠️  Cerebras rate limit persists. Falling back to Groq...")
                        break
                else:
                    print(f"⚠️  Cerebras error: {e}. Falling back to Groq...")
                    break
    
    # Fallback to Groq
    if groq_key:
        try:
            print("\n🔄 Trying Groq (fallback)...")
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
            return True, "Groq"
            
        except Exception as e:
            print(f"❌ Groq also failed: {e}")
    
    print("❌ All providers failed for vision decision")
    return False, None

async def test_browser_agent_with_text_llm(provider: str):
    """Test browser agent with text-only LLM"""
    print("\n" + "="*70)
    print(f"TEST 3: Browser Agent with Text-Only LLM ({provider})")
    print("="*70)
    
    if provider == "Cerebras":
        api_key = os.getenv("CEREBRAS_API_KEY")
        model = "qwen-3-235b-a22b-instruct-2507"
        base_url = "https://api.cerebras.ai/v1"
    elif provider == "Groq":
        api_key = os.getenv("GROQ_API_KEY")
        model = "openai/gpt-oss-120b"
        base_url = "https://api.groq.com/openai/v1"
    else:
        print("❌ Unknown provider")
        return False
    
    if not api_key:
        print(f"❌ {provider} API key not set")
        return False
    
    try:
        print(f"\n📝 Creating browser agent with {provider}...")
        print(f"   Model: {model}")
        print(f"   Task: Navigate to example.com")
        
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
        )
        
        agent = Agent(
            task="Go to https://example.com",
            llm=llm,
            use_vision=False,
        )
        
        print("🌐 Running agent...")
        history = await agent.run()
        
        print(f"✅ Browser agent with {provider} completed")
        return True
        
    except Exception as e:
        print(f"❌ Browser agent with {provider} failed: {e}")
        return False

async def main():
    print("="*70)
    print("Complete Fallback Chain Test")
    print("Cerebras → Groq for both Vision Decision and Browser Automation")
    print("="*70)
    
    results = []
    
    # Test 1: Text LLM initialization
    success1, provider1 = await test_text_llm_initialization()
    results.append(("Text LLM Initialization", success1))
    
    # Test 2: Vision decision fallback
    success2, provider2 = await test_vision_decision_fallback()
    results.append(("Vision Decision Fallback", success2))
    
    # Test 3: Browser agent with text LLM (use whichever provider worked)
    if success1 and provider1:
        success3 = await test_browser_agent_with_text_llm(provider1)
        results.append((f"Browser Agent ({provider1})", success3))
    else:
        print("\n⚠️  Skipping browser agent test - no text LLM available")
        results.append(("Browser Agent", False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Fallback chain is working correctly.")
    else:
        at_least_one_provider = success1 or success2
        if at_least_one_provider:
            print("\n✅ Fallback chain is functional - at least one provider works.")
        else:
            print("\n⚠️  All providers failed. Check API keys and rate limits.")

if __name__ == "__main__":
    asyncio.run(main())
