import asyncio
import logging
import sys
import os

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
    print("🚀 STARTING SYLLABUS RETRIEVAL TASK")
    print("="*60 + "\n")
    
    # Clean verification directory
    download_dir = str(CONFIG.DOWNLOADS_DIR)
    if os.path.exists(download_dir):
        for f in os.listdir(download_dir):
            try:
                os.remove(os.path.join(download_dir, f))
            except Exception: pass
        print(f"🧹 Cleaned download directory: {download_dir}")
    
    # Task: Navigate and find specific syllabus
    task_desc = """
    1. Navigate to 'http://www.vnrvjiet.ac.in/'.
    2. Locate the 'Syllabus' section. Hint: It might be under 'Academics' or 'Examination' or a direct link.
    3. Look for 'R22 B.Tech' syllabus.
    4. Specifically find the syllabus for 'III Year' (Third Year) - 'AIML' (Artificial Intelligence and Machine Learning).
    5. CRITICAL: You must DOWNLOAD the PDF file. Click the link to download it.
    6. Waiting for the download to complete is important.
    """
    
    agent = BrowserAgent(task=task_desc, headless=False)
    
    try:
        result = await agent.run()
        
        print("\n" + "="*60)
        print("✅ TASK COMPLETED")
        print(f"Final Result: success={result.success}")
        print(f"Actions Taken: {len(result.actions_taken)}")
        print("="*60)
        
        # Verify Downloads
        download_dir = str(CONFIG.DOWNLOADS_DIR)
        print(f"\n📂 Verifying Downloads in: {download_dir}")
        if os.path.exists(download_dir):
            files = os.listdir(download_dir)
            if files:
                print(f"✅ Found {len(files)} files:")
                for f in files:
                    print(f"  - {f}")
            else:
                print("⚠️ Directory is empty.")
        else:
            print("❌ Download directory not found.")
            
    except Exception as e:
        logger.error(f"Task failed with error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
