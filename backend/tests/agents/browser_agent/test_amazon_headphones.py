"""
Browser Agent Test - Amazon Wireless Headphones Search
Find the cheapest wireless headphones on Amazon
"""

import asyncio
import logging
import sys
import os

# Add backend directory to path (4 levels up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agents.browser_agent.agent import BrowserAgent

# Configure logging
log_file = "browser_headphones_test.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.getLogger('browser_agent').setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

print(f"\n📝 Detailed logs are being saved to: {log_file}\n")

async def main():
    task = """
    Go to Amazon.com, search for "wireless headphones".
    Apply the "Low to High" price sort filter to find the cheapest options.
    Identify and extract the details of the cheapest wireless headphones available.
    Record the product name, price, ratings, and any other key details using save_info.
    """
    
    print(f"\n{'='*60}")
    print("🎧 Starting Browser Agent - Wireless Headphones Search")
    print(f"{'='*60}")
    print(f"Task: {task.strip()}")
    print(f"{'='*60}\n")
    
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
    
    if result.extracted_data:
        print(f"\n📦 Extracted Data:")
        import json
        print(json.dumps(result.extracted_data, indent=2, default=str))
    
    if result.error:
        print(f"\n❌ Error: {result.error}")
    
    print(f"\n⏱️ Execution Time: {result.metrics.get('total_time', 0):.1f}s")
    print(f"{'='*60}\n")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(main())
