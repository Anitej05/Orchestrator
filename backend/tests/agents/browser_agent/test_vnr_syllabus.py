"""
Browser Agent Test - VNR VJIET Syllabus Navigation
Open VNR VJIET website, navigate to syllabus page, and open 3rd year AI&ML R22 syllabus
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging with timestamp in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/vnr_syllabus_test_{timestamp}.log"

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True  # Force reconfigure
)

# Set levels for specific loggers
logging.getLogger('browser_agent').setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('playwright').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

print(f"\n📝 Logs are being saved to: {log_file}\n")
logger.info(f"=== VNR VJIET Syllabus Test Started ===")
logger.info(f"Log file: {log_file}")

# NOW import browser agent (after logging is configured)
from backend.agents.browser_agent.agent import BrowserAgent

async def main():
    task = """
    1. Navigate to VNR VJIET's official website (vnrvjiet.ac.in)
    2. Find and navigate to the Syllabus page/section
    3. Look for the Third Year (3rd year) AI&ML (Artificial Intelligence and Machine Learning) R22 regulation syllabus
    4. Open/Download the syllabus PDF file for 3rd year AI&ML R22
    5. Save the details of the syllabus file found using save_info
    """
    
    print(f"\n{'='*60}")
    print("🎓 Starting Browser Agent - VNR VJIET Syllabus Search")
    print(f"{'='*60}")
    print(f"Task: {task.strip()}")
    print(f"{'='*60}\n")
    
    logger.info(f"Task: {task.strip()}")
    
    agent = BrowserAgent(
        task=task,
        headless=False  # Show browser for debugging
    )
    
    result = await agent.run()
    
    print(f"\n{'='*60}")
    print("📊 RESULT SUMMARY")
    print(f"{'='*60}")
    print(f"Success: {result.success}")
    print(f"Summary: {result.task_summary}")
    
    logger.info(f"=== RESULT ===")
    logger.info(f"Success: {result.success}")
    logger.info(f"Summary: {result.task_summary}")
    
    if result.extracted_data:
        print(f"\n📦 Extracted Data:")
        import json
        data_str = json.dumps(result.extracted_data, indent=2, default=str)
        print(data_str)
        logger.info(f"Extracted Data: {data_str}")
    
    if result.error:
        print(f"\n❌ Error: {result.error}")
        logger.error(f"Error: {result.error}")
    
    print(f"\n⏱️ Execution Time: {result.metrics.get('total_time', 0):.1f}s")
    print(f"{'='*60}\n")
    
    logger.info(f"Execution Time: {result.metrics.get('total_time', 0):.1f}s")
    logger.info(f"=== VNR VJIET Syllabus Test Completed ===")
    
    # Flush all handlers
    for handler in logging.root.handlers:
        handler.flush()
    
    print(f"\n📝 Full logs saved to: {log_file}")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result.success else 1)
