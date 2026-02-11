"""Quick test for vision module"""
import asyncio
import sys
import base64
sys.path.insert(0, 'd:/Internship/Orbimesh/backend')

from backend.agents.browser_agent.vision import VisionClient

async def test_vision():
    v = VisionClient()
    print(f"Vision available: {v.available}")
    print(f"Primary: {v.model}")
    print(f"Fallback: {v.model_nvidia}")
    
    # Create a simple test with minimal image (1x1 red pixel)
    # This just tests the API connection, not real vision
    test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    
    page_content = {
        'url': 'https://example.com',
        'title': 'Test Page',
        'elements': [{'role': 'link', 'name': 'Test Link', 'x': 100, 'y': 100}],
        'scroll_position': 0,
        'max_scroll': 1000,
        'scroll_percent': 0
    }
    
    print("\n🎨 Testing vision planning...")
    try:
        result = await v.plan_action_with_vision(
            task="Click the test link",
            screenshot_base64=test_image,
            page_content=page_content,
            history=[],
            step=1
        )
        if result:
            print(f"✅ Vision returned: {result.reasoning[:100]}...")
            print(f"   Action: {result.actions[0].name}")
        else:
            print("⚠️ Vision returned None (check logs for fallback attempts)")
    except Exception as e:
        print(f"❌ Vision error: {e}")

if __name__ == "__main__":
    asyncio.run(test_vision())
