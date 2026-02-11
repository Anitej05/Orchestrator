
import asyncio
import logging
import sys

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from backend.agents.browser_agent.agent import BrowserAgent

async def main():
    # Amazon Sports Shoes Task
    # Search, filter by color preference, find the cheapest option
    task = """
    1. Navigate to 'https://www.amazon.in'.
    2. Search for 'sports shoes'.
    3. Look for RED colored shoes (I prefer red shoes).
    4. Find the CHEAPEST red sports shoe available.
    5. Save the product name, price, and any other relevant details of the cheapest red shoe.
    """
    
    print("\n\n" + "="*60)
    print("🚀 STARTING AMAZON TEST: Find Cheapest Red Sports Shoes")
    print("="*60 + "\n")
    
    agent = BrowserAgent(task=task, headless=False)  # Run headed to see what's happening
    
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
