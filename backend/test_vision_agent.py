"""Test script for Browser Agent - Vision Capabilities"""
import asyncio
import logging
from agents.browser_agent.agent import BrowserAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

async def main():
    # We use a task that explicitly mentions "visual" or "icon" to encourage vision usage.
    # Wikipedia's search icon is a good standard test.
    # Or we can ask it to describe an image.
    
    task = 'Navigate to https://www.wikipedia.org/. Use your vision to find and click the search icon (magnifying glass) button explicitly. Do not use text search. Then verify you are on the search page.'
    
    # Alternative: Amazon Cart icon which doesn't always have text
    # task = 'Navigate to https://www.amazon.in/. Use your vision to find and click the shopping cart icon at the top right. Then extract the text "Your Amazon Cart is empty" using save_info.'

    agent = BrowserAgent(
        task=task,
        headless=False
    )
    result = await agent.run()
    print(f"\n\n{'='*60}")
    print(f"FINAL RESULT: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Summary: {result.task_summary}")
    print(f"Actions taken: {len(result.actions_taken)}")
    print(f"{'='*60}")
    return result

if __name__ == "__main__":
    asyncio.run(main())
