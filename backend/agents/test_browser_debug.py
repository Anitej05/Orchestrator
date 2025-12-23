"""
Browser Agent DEEP DEBUG Test

Captures EVERY LLM call, response, action, and decision to file for analysis.
"""

import asyncio
import logging
import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from browser_agent.agent import BrowserAgent

# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"browser_debug_{timestamp}.log"

# Custom formatter for detailed output
class DetailedFormatter(logging.Formatter):
    def format(self, record):
        # Add separator for important logs
        msg = super().format(record)
        if 'LLM Request' in msg or 'RESPONSE' in msg or 'Calling' in msg:
            return f"\n{'='*80}\n{msg}\n{'='*80}"
        return msg

# Setup comprehensive logging
file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(DetailedFormatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

# Root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Set ALL browser agent loggers to DEBUG
for module in ['browser_agent', 'browser_agent.agent', 'browser_agent.llm', 
               'browser_agent.dom', 'browser_agent.actions', 'browser_agent.vision',
               'browser_agent.browser', 'browser_agent.planner', 'browser_agent.state']:
    logging.getLogger(module).setLevel(logging.DEBUG)

# Reduce noise from HTTP libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

print(f"\n{'#'*60}")
print(f"# BROWSER AGENT DEBUG TEST")
print(f"# Log file: {LOG_FILE}")
print(f"{'#'*60}\n")

async def main():
    task = """Go to amazon.in, search "Samsung Galaxy S25 Ultra", sort by "Low to High" price, 
    click the cheapest S25 Ultra result, extract product name and price using save_info action."""
    
    print(f"Task: {task}\n")
    logging.info(f"TASK: {task}")
    
    agent = BrowserAgent(
        task=task,
        max_steps=15,  # Reduced - should be enough
        headless=False
    )
    
    try:
        result = await agent.run()
        
        print(f"\n{'='*60}")
        print(f"SUCCESS: {result.success}")
        print(f"SUMMARY: {result.task_summary}")
        if result.extracted_data:
            print(f"DATA: {json.dumps(result.extracted_data, indent=2, default=str)}")
        if result.error:
            print(f"ERROR: {result.error}")
        print(f"{'='*60}\n")
        
        logging.info(f"FINAL RESULT: {result.model_dump_json(indent=2)}")
        
    except Exception as e:
        logging.exception(f"CRITICAL EXCEPTION: {e}")
        print(f"\n❌ EXCEPTION: {e}")
    
    print(f"\n📝 Full logs saved to: {LOG_FILE}")
    return result

if __name__ == "__main__":
    asyncio.run(main())
