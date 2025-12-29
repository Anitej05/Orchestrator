
import asyncio
import logging
import sys

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from agents.browser_agent.agent import BrowserAgent

async def main():
    # Wikipedia Picture of the Day Task - FULL VERSION
    # Tests multi-step goal handling: describe AND click
    task = """
    1. Navigate to 'https://en.wikipedia.org/wiki/Main_Page'.
    2. Find the 'Picture of the day' or 'Today's featured picture' section.
    3. Use your VISION to describe exactly what is shown in the picture.
    4. Click on the picture to open the file page.
    """
    
    print("\n\n" + "="*60)
    print("🚀 STARTING COMPLEX VISION TEST: Wikipedia Picture of the Day")
    print("="*60 + "\n")
    
    agent = BrowserAgent(task=task, headless=False) # Run headed to see what's happening
    
    try:
        result = await agent.run()
        print("\n\n" + "="*60)
        print("✅ TEST COMPLETED")
        print(f"Final Result: {result}")
        print("="*60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup if needed
        pass

if __name__ == "__main__":
    asyncio.run(main())
