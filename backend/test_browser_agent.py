"""Test script for Browser Agent - Amazon S25 Ultra Search"""
import asyncio
import logging
from agents.browser_agent.agent import BrowserAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

async def main():
    agent = BrowserAgent(
        task='Go to amazon.in, search "Samsung Galaxy S25 Ultra", sort by "Low to High" price, click the cheapest S25 Ultra result, extract product name and price using save_info action.',
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
