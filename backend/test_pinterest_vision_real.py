"""
Real Pinterest vision test - requires analyzing images and making decisions
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.browser_automation_agent import BrowserAgent

async def test_pinterest_vision():
    """Test Pinterest with actual vision requirement - pick best vintage cars"""
    
    task = "Go to pinterest.com, search for 'vintage cars', look at the first 3 images, and pick the best one based on visual appeal and classic design"
    
    print("=" * 80)
    print("TEST: Pinterest Vision - Analyze Images and Make Decision")
    print("=" * 80)
    print(f"Task: {task}")
    print()
    
    agent = BrowserAgent(
        task=task,
        thread_id="test_vision_real",
        max_steps=15,
        headless=False
    )
    
    try:
        async with agent:
            result = await agent.run()
        
        print("\n" + "=" * 80)
        print("RESULT:")
        print("=" * 80)
        print(f"Success: {result.get('success', False)}")
        print(f"Steps: {result.get('steps_taken', 0)}")
        print(f"Time: {result.get('execution_time', 0):.1f}s")
        print(f"Summary: {result.get('result_summary', 'N/A')}")
        
        # Check if vision was used
        actions = result.get('actions', [])
        vision_used = any('vision' in str(action).lower() for action in actions)
        print(f"\nVision used: {vision_used} (should be True)")
        
        if result.get('success') and vision_used:
            print("\n✅ TEST PASSED: Task completed with vision")
            return True
        elif not vision_used:
            print("\n❌ TEST FAILED: Vision was not used when it should be")
            return False
        else:
            print("\n❌ TEST FAILED: Task did not complete")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_pinterest_vision())
    sys.exit(0 if success else 1)
