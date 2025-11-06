"""
Comprehensive test for the browser automation agent with hybrid vision strategy.
"""
import asyncio
import os
from dotenv import load_dotenv
from browser_use import Agent, ChatOpenAI
from openai import OpenAI

load_dotenv()

async def test_vision_decision():
    """Test that vision decision logic works correctly"""
    print("\n" + "="*70)
    print("TEST 1: Vision Decision Logic")
    print("="*70)
    
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    if not cerebras_key:
        print("❌ CEREBRAS_API_KEY not set")
        return False
    
    client = OpenAI(
        api_key=cerebras_key,
        base_url="https://api.cerebras.ai/v1"
    )
    
    test_cases = [
        ("Take a screenshot of google.com", True, "Should need vision for screenshots"),
        ("Navigate to github.com and click login", False, "Should NOT need vision for simple navigation"),
        ("Describe the visual layout of the page", True, "Should need vision for visual description"),
        ("Fill in the form with name John", False, "Should NOT need vision for form filling"),
    ]
    
    all_passed = True
    for task, expected_vision, reason in test_cases:
        try:
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
                max_tokens=10,
            )
            
            decision = response.choices[0].message.content.strip().upper()
            needs_vision = "YES" in decision
            
            passed = needs_vision == expected_vision
            status = "✅ PASS" if passed else "❌ FAIL"
            all_passed = all_passed and passed
            
            print(f"\n{status}")
            print(f"  Task: {task}")
            print(f"  Expected: {'Vision' if expected_vision else 'Text-only'}")
            print(f"  Got: {'Vision' if needs_vision else 'Text-only'}")
            print(f"  Reason: {reason}")
            
        except Exception as e:
            print(f"❌ FAIL - Error: {e}")
            all_passed = False
    
    return all_passed

async def test_text_only_agent():
    """Test browser agent with text-only model"""
    print("\n" + "="*70)
    print("TEST 2: Text-Only Browser Agent")
    print("="*70)
    
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    if not cerebras_key:
        print("❌ CEREBRAS_API_KEY not set")
        return False
    
    try:
        print("\n📝 Creating text-only agent...")
        print("   Model: qwen-3-235b-a22b-instruct-2507")
        print("   Task: Navigate to example.com and click a link")
        
        llm = ChatOpenAI(
            model="qwen-3-235b-a22b-instruct-2507",
            api_key=cerebras_key,
            base_url="https://api.cerebras.ai/v1",
            temperature=0.2,
        )
        
        agent = Agent(
            task="Go to https://example.com",
            llm=llm,
            use_vision=False,
        )
        
        print("🌐 Running agent...")
        history = await agent.run()
        
        print("✅ Text-only agent completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Text-only agent failed: {e}")
        return False

async def test_vision_agent():
    """Test browser agent with vision model"""
    print("\n" + "="*70)
    print("TEST 3: Vision-Enabled Browser Agent")
    print("="*70)
    
    ollama_key = os.getenv("OLLAMA_API_KEY")
    if not ollama_key:
        print("❌ OLLAMA_API_KEY not set")
        return False
    
    try:
        print("\n📝 Creating vision-enabled agent...")
        print("   Model: qwen3-vl:235b-cloud")
        print("   Task: Take a screenshot and describe it")
        
        llm = ChatOpenAI(
            model="qwen3-vl:235b-cloud",
            api_key=ollama_key,
            base_url="https://ollama.com/v1",
            temperature=0.2,
        )
        
        agent = Agent(
            task="Go to https://example.com and describe what you see",
            llm=llm,
            use_vision=True,
        )
        
        print("🌐 Running agent...")
        history = await agent.run()
        
        print("✅ Vision agent completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Vision agent failed: {e}")
        return False

async def main():
    print("="*70)
    print("Browser Automation Agent - Comprehensive Test Suite")
    print("="*70)
    
    results = []
    
    # Test 1: Vision decision logic
    result1 = await test_vision_decision()
    results.append(("Vision Decision Logic", result1))
    
    # Test 2: Text-only agent
    result2 = await test_text_only_agent()
    results.append(("Text-Only Agent", result2))
    
    # Test 3: Vision agent
    result3 = await test_vision_agent()
    results.append(("Vision Agent", result3))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(main())
