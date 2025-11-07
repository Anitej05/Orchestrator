"""
Simple Pinterest test - should work WITHOUT vision for basic navigation
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.browser_automation_agent import BrowserAgent

async def test_pinterest_simple():
    """Test basic Pinterest search - should NOT use vision"""
    
    task = "Go to pinterest.com and search for 'vintage cars'"
    
    print("=" * 80)
    print("TEST: Pinterest Simple Navigation (NO VISION)")
    print("=" * 80)
    print(f"Task: {task}")
    print()
    
    agent = BrowserAgent(
        task=task,
        thread_id="test_simple",
        max_steps=10
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
        print(f"Subtasks: {result.get('subtasks_completed', 0)}/{result.get('total_subtasks', 0)}")
        
        # Check if vision was used (it shouldn't be)
        vision_used = any('vision' in str(action).lower() for action in result.get('actions', []))
        print(f"\nVision used: {vision_used} (should be False)")
        
        if result.get('success') and not vision_used:
            print("\n✅ TEST PASSED: Task completed without vision")
            return True
        elif vision_used:
            print("\n❌ TEST FAILED: Vision was used when it shouldn't be")
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
    success = asyncio.run(test_pinterest_simple())
    sys.exit(0 if success else 1)
