"""
Complete Pinterest test - Search, analyze images with vision, save them, and describe them
"""
import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.browser_automation_agent import BrowserAgent

async def test_pinterest_complete():
    """
    Complete Pinterest workflow:
    1. Search for vintage cars
    2. Use vision to look at and analyze 3 images
    3. Save those images
    4. Describe each image in detail
    """
    
    task = """Go to pinterest.com and search for 'vintage cars'. 
    Look at 3 different vintage car images using vision. 
    For each image, describe what you see (car model, color, style, features).
    Save all 3 images to the downloads folder."""
    
    print("=" * 80)
    print("TEST: Complete Pinterest Vision Workflow")
    print("=" * 80)
    print(f"Task: {task}")
    print()
    
    agent = BrowserAgent(
        task=task,
        thread_id="test_pinterest_complete",
        max_steps=20,
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
        print(f"Subtasks: {result.get('subtasks_completed', 0)}/{result.get('total_subtasks', 0)}")
        print(f"\nSummary:\n{result.get('result_summary', 'N/A')}")
        
        # Check if vision was used
        actions_str = str(result.get('actions', []))
        vision_used = 'vision' in actions_str.lower() or 'analyze' in actions_str.lower()
        
        # Check if images were saved
        downloads_dir = Path("downloads")
        image_files = list(downloads_dir.glob("*.png")) + list(downloads_dir.glob("*.jpg")) + list(downloads_dir.glob("*.jpeg"))
        images_saved = len(image_files) >= 3
        
        print(f"\nVision used: {vision_used} (should be True)")
        print(f"Images saved: {len(image_files)} (should be >= 3)")
        
        if images_saved:
            print("\nSaved images:")
            for img in image_files[-3:]:
                print(f"  - {img.name}")
        
        # Check if descriptions were provided
        summary = result.get('result_summary', '')
        has_descriptions = len(summary) > 100 and any(word in summary.lower() for word in ['car', 'vintage', 'color', 'style'])
        
        print(f"Has descriptions: {has_descriptions} (should be True)")
        
        if result.get('success') and vision_used and images_saved and has_descriptions:
            print("\n✅ TEST PASSED: Complete workflow successful")
            return True
        else:
            print("\n❌ TEST FAILED:")
            if not vision_used:
                print("  - Vision was not used")
            if not images_saved:
                print("  - Images were not saved")
            if not has_descriptions:
                print("  - Descriptions not provided")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_pinterest_complete())
    sys.exit(0 if success else 1)
