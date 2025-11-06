"""
Test browser-use with Groq model to see if it works better.
"""
import asyncio
import os
from dotenv import load_dotenv
from browser_use import Agent, ChatOpenAI

load_dotenv()

async def test_browser_use_with_groq():
    """Test browser-use Agent with Groq model"""
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY is not set")
        return
    
    print("🧪 Testing browser-use with Groq model...")
    print(f"   Model: openai/gpt-oss-120b")
    print(f"   API Key: {api_key[:10]}...")
    
    try:
        # Initialize ChatOpenAI with Groq
        llm = ChatOpenAI(
            model="openai/gpt-oss-120b",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.2,
        )
        
        print("✅ ChatOpenAI initialized successfully")
        
        # Create a simple browser agent
        agent = Agent(
            task="Go to google.com and tell me what you see",
            llm=llm,
            use_vision=False,
        )
        
        print("✅ Agent created successfully")
        print("🚀 Running agent...")
        
        # Run the agent
        history = await agent.run()
        
        print("✅ Agent completed successfully!")
        
        # Try to get final result
        if hasattr(history, 'final_result'):
            if callable(history.final_result):
                result = history.final_result()
            else:
                result = history.final_result
            print(f"📝 Final result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_browser_use_with_groq())
