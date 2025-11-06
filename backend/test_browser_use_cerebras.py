"""
Test browser-use with Cerebras model to identify the exact error.
"""
import asyncio
import os
from dotenv import load_dotenv
from browser_use import Agent, ChatOpenAI

load_dotenv()

async def test_browser_use_with_cerebras():
    """Test browser-use Agent with Cerebras model"""
    
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        print("❌ ERROR: CEREBRAS_API_KEY is not set")
        return
    
    print("🧪 Testing browser-use with Cerebras model...")
    print(f"   Model: qwen-3-235b-a22b-instruct-2507")
    print(f"   API Key: {api_key[:10]}...")
    
    try:
        # Initialize ChatOpenAI with Cerebras
        llm = ChatOpenAI(
            model="qwen-3-235b-a22b-instruct-2507",
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1",
            temperature=0.2,
        )
        
        print("✅ ChatOpenAI initialized successfully")
        
        # Create a simple browser agent
        agent = Agent(
            task="Go to google.com",
            llm=llm,
            use_vision=False,
        )
        
        print("✅ Agent created successfully")
        print("🚀 Running agent...")
        
        # Run the agent
        history = await agent.run()
        
        print("✅ Agent completed successfully!")
        print(f"   History: {history}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_browser_use_with_cerebras())
