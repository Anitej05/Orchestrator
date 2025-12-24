"""
Browser Agent Test - Amazon S25 Ultra Search

Run this script to test the browser agent with the Amazon search task.
"""

import asyncio
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from browser_agent.agent import BrowserAgent

# Configure logging - both console AND file
log_file = "browser_agent_test.log"
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum detail
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),  # File handler
        logging.StreamHandler()  # Console handler
    ]
)

# Set specific loggers to appropriate levels
logging.getLogger('browser_agent').setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.WARNING)  # Reduce noise from HTTP client
logging.getLogger('httpcore').setLevel(logging.WARNING)

print(f"\n📝 Detailed logs are being saved to: {log_file}\n")

async def main():
    task = """
    Go to Amazon India (amazon.in), search for "Samsung Galaxy S25 Ultra".
    Apply the "Low to High" price sort filter.
    Find and click on the cheapest S25 Ultra phone available.
    Extract all product details including: name, price, specifications, ratings, and availability.
    Use save_info to record the key details found.
    """
    
    print(f"\n{'='*60}")
    print("🚀 Starting Browser Agent Test")
    print(f"{'='*60}")
    print(f"Task: {task.strip()}")
    print(f"{'='*60}\n")
    
    agent = BrowserAgent(
        task=task,
        max_steps=25,  # Allow more steps for this complex task
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
