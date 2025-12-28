import asyncio
import logging
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.browser_agent.agent import BrowserAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def main():
    print("\n" + "="*60)
    print("🚀 STARTING GOOGLE DOODLE TEST")
    print("="*60 + "\n")
    
    # Task: Check for Doodle and Describe it
    task_desc = """
    1. Navigate to 'https://www.google.com'.
    2. Look at the main logo image above the search bar.
    3. Determine if it is a special "Google Doodle" or the standard "Google" text logo.
    4. If it is a Doodle, describe what you see in the image (colors, characters, theme).
    5. If it is standard, just say "Standard Google Logo".
    6. Save your finding using save_info("doodle_description", "YOUR DESCRIPTION").
    """
    
    agent = BrowserAgent(task=task_desc)
    
    try:
        result = await agent.run()
        
        print("\n" + "="*60)
        print("✅ TEST COMPLETED")
        print(f"Final Result: success={result.success}")
        print(f"Extracted Data: {result.extracted_data}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
