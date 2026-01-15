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
    print("🚀 STARTING UPLOAD & CONVERSION TEST")
    print("="*60 + "\n")
    
    # Locate the source file
    source_file = CONFIG.DOWNLOADS_DIR / "AIML_R22.pdf"
    if not source_file.exists():
        print(f"❌ Source file not found: {source_file}")
        print("Please run the syllabus download test first.")
        return
        
    print(f"📄 Source File: {source_file}")
    
    # Task: Convert PDF to Word
    # using a reliable free tool
    task_desc = f"""
    1. Navigate to 'https://www.ilovepdf.com/pdf_to_word'.
    2. Click the 'Select PDF file' button to upload a file.
    3. IMPORTANT: You must use the `upload_file` action.
    4. Upload the file at this EXACT path: '{source_file.absolute()}'.
    5. After upload, click the 'Convert to WORD' button.
    6. Wait for the conversion to finish (this might take 10-20 seconds).
    7. CRITICAL: Use the `download_file` action to click the big 'Download WORD' button.
    8. You must verify the file is saved.
    """
    
    agent = BrowserAgent(task=task_desc, headless=False)
    
    try:
        result = await agent.run()
        
        print("\n" + "="*60)
        print("✅ TASK COMPLETED")
        print(f"Final Result: success={result.success}")
        print("="*60)
        
        # Verify New Download
        download_dir = CONFIG.DOWNLOADS_DIR
        print(f"\n📂 Checking Downloads in: {download_dir}")
        docx_files = list(download_dir.glob("*.docx"))
        if docx_files:
            print(f"✅ Found {len(docx_files)} Word files:")
            for f in docx_files:
                print(f"  - {f.name} (Size: {f.stat().st_size} bytes)")
        else:
            print("⚠️ No .docx files found.")
            
    except Exception as e:
        logger.error(f"Task failed with error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
