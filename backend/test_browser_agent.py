"""
Test script for the browser automation agent.
This script tests the agent locally before registering it with the orchestrator.
"""
import asyncio
import os
from dotenv import load_dotenv
from browser_use import Agent, ChatOpenAI

load_dotenv()

async def test_browser_agent():
    """Test the browser automation agent with a simple task"""
    
    # Check if API key is set
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        print("❌ ERROR: OLLAMA_API_KEY is not set in .env file")
        print("Please add your Ollama API key to the .env file")
        return
    
    print("🚀 Starting browser automation test...")
    print("📝 Task: Navigate to GitHub and extract repository information")
    
    try:
        # Initialize the vision LLM
        llm = ChatOpenAI(
            model="qwen3-vl:235b-cloud",
            api_key=api_key,
            base_url="https://ollama.com/v1",
        )
        
        # Create the agent
        agent = Agent(
            task="Navigate to https://github.com/browser-use/browser-use, take a screenshot, and describe what you see including the number of stars",
            llm=llm,
            use_vision=True,
        )
        
        # Run the agent
        print("🌐 Browser window will open shortly...")
        print("👀 Watch the browser automation in action!")
        history = await agent.run()
        
        print("\n✅ Browser automation completed!")
        print(f"📊 History: {history}")
        
    except Exception as e:
        print(f"\n❌ Error during browser automation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure OLLAMA_API_KEY is set correctly in .env")
        print("2. Ensure browser-use is installed: pip install browser-use")
        print("3. Check your internet connection")

if __name__ == "__main__":
    print("=" * 60)
    print("Browser Automation Agent Test")
    print("=" * 60)
    asyncio.run(test_browser_agent())
