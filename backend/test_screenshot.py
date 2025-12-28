import asyncio
import logging
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.browser_agent.agent import BrowserAgent
from agents.browser_agent.config import CONFIG

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
    print("📸 STARTING SCREENSHOT CAPTURE TEST")
    print("="*60 + "\n")
    
    # Task: Take a screenshot
    task_desc = """
    1. Navigate to 'https://example.com'.
    2. Wait for the page to load.
    3. IMPORTANT: Use the `save_screenshot` action to save a screenshot as "example_test.jpg".
    4. Verify the action success.
    """
    
    agent = BrowserAgent(task=task_desc, headless=False)
    
    try:
        result = await agent.run()
        
        print("\n" + "="*60)
        print("✅ TASK COMPLETED")
        print(f"Final Result: success={result.success}")
        
        print("\n📝 Actions Taken:")
        for action in result.actions_taken:
            print(f"  - {action}")
            
        print("="*60)
        
        # Verify Screenshot File
        screenshot_dir = CONFIG.SCREENSHOTS_DIR
        print(f"\n📂 Checking Screenshots in: {screenshot_dir}")
        
        found = list(screenshot_dir.glob("*example_test*.jpg"))
        if found:
            print(f"✅ Found {len(found)} screenshot(s):")
            for f in found:
                print(f"  - {f.name} (Size: {f.stat().st_size} bytes)")
        else:
            print("⚠️ No screenshot found with expected name.")
            print("All files:", list(screenshot_dir.glob("*")))
            
    except Exception as e:
        logger.error(f"Task failed with error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
