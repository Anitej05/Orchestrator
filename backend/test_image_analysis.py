"""
Test vision-based image analysis on a simpler site
"""

import requests
import time

def test_image_analysis():
    """Test vision analyzing images"""
    print("\n" + "="*80)
    print("IMAGE ANALYSIS TEST")
    print("Testing vision model's ability to see and describe images")
    print("="*80)
    
    url = "http://localhost:8000/api/chat"
    payload = {
        "prompt": "Go to https://www.wikipedia.org and look at the main page. Describe what you see visually - the logo, images, layout, and colors.",
        "thread_id": f"image_test_{int(time.time())}",
        "planning_mode": False
    }
    
    print(f"\n📤 Testing image analysis...")
    print(f"Task: {payload['prompt']}")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=180)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            final_response = result.get('final_response', '') or ''
            
            print(f"\n✅ Completed in {elapsed:.1f}s")
            print(f"\n📝 Response:")
            print(f"{'-'*80}")
            print(final_response if final_response else "No response")
            print(f"{'-'*80}")
            
            # Check if vision was used
            has_canvas = result.get('has_canvas', False)
            has_visual_desc = any(word in final_response.lower() for word in ['logo', 'image', 'color', 'visual', 'see'])
            
            print(f"\n✅ Results:")
            print(f"  {'✅' if has_canvas else '❌'} Canvas with screenshots")
            print(f"  {'✅' if has_visual_desc else '❌'} Visual descriptions")
            print(f"  {'✅' if len(final_response) > 100 else '❌'} Detailed response")
            
            print(f"\n💡 Check logs for:")
            print(f"   - '🎨 Vision needed (task requires image analysis)'")
            print(f"   - '✅ Ollama vision action: analyze_images'")
            
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    success = test_image_analysis()
    exit(0 if success else 1)
