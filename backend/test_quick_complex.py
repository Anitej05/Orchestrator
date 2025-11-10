"""
Quick Complex Task Test - Simplified version for rapid testing

This is a lighter version that tests core capabilities without the full complexity.
Good for quick validation before running the full complex task.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.browser_automation_agent import BrowserAgent


async def quick_test():
    """Run a quick but still complex task"""
    
    # Simpler but still challenging task
    task = """Go to Best Buy and search for 'laptops'. 
Find 2 laptops with at least 4-star ratings. 
For each, extract: model name, price, and rating."""
    
    print("🚀 Quick Complex Task Test")
    print(f"📋 Task: {task}")
    print()
    
    try:
        async with BrowserAgent(
            task=task,
            max_steps=15,
            headless=False,
            enable_streaming=True
        ) as agent:
            
            print(f"✅ Agent initialized (ID: {agent.task_id})")
            result = await agent.run()
            
            print()
            print("=" * 60)
            print(f"✅ Success: {result.get('success')}")
            print(f"📝 Steps: {len(result.get('actions_taken', []))}")
            print(f"📋 Summary: {result.get('task_summary', 'N/A')}")
            
            # Show extracted data
            extracted = result.get('extracted_data', {})
            if extracted:
                items = extracted.get('structured_items', [])
                if items:
                    print(f"\n📦 Extracted {len(items)} products:")
                    for i, item in enumerate(items, 1):
                        print(f"   {i}. {item.get('title', 'N/A')[:50]}")
                        print(f"      Price: {item.get('price', 'N/A')}")
                        print(f"      Rating: {item.get('rating', 'N/A')}")
            
            print("=" * 60)
            
            return result
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    result = asyncio.run(quick_test())
    sys.exit(0 if result.get('success') else 1)
