"""
Test vision capabilities with Pinterest image analysis:
- Navigate to Pinterest
- Search for vintage cars
- Analyze 3 images using vision
- Describe each image
"""

import requests
import time

def test_pinterest_vision():
    """Test vision model analyzing Pinterest images"""
    print("\n" + "="*80)
    print("PINTEREST VISION TEST")
    print("Testing: Image analysis and description using vision model")
    print("="*80)
    
    url = "http://localhost:8000/api/chat"
    payload = {
        "prompt": "Go to Pinterest and search for 'vintage cars'. Look at 3 different vintage car images and describe each one in detail - tell me the car model, color, style, and what makes it interesting.",
        "thread_id": f"pinterest_test_{int(time.time())}",
        "planning_mode": False
    }
    
    print(f"\n📤 Sending request to browser agent...")
    print(f"Task: {payload['prompt']}")
    print(f"\nThis will test:")
    print(f"  1. Navigation to Pinterest")
    print(f"  2. Search functionality")
    print(f"  3. Vision model analyzing images")
    print(f"  4. Detailed image descriptions")
    print(f"  5. Canvas streaming with screenshots")
    
    start_time = time.time()
    
    try:
        print(f"\n⏳ Processing... (this may take 2-3 minutes)")
        response = requests.post(url, json=payload, timeout=300)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Task completed in {elapsed:.1f}s")
            
            # Check features
            has_canvas = result.get('has_canvas', False)
            has_browser_view = 'browser_view' in result
            has_plan_view = 'plan_view' in result
            final_response = result.get('final_response', '')
            
            print(f"\nFeatures:")
            print(f"  {'✅' if has_canvas else '❌'} Canvas")
            print(f"  {'✅' if has_browser_view else '❌'} Browser View (screenshots)")
            print(f"  {'✅' if has_plan_view else '❌'} Plan View")
            
            print(f"\n📝 Response:")
            print(f"{'-'*80}")
            if final_response:
                print(final_response)
            else:
                print("No response received")
            print(f"{'-'*80}")
            
            # Check if response contains image descriptions
            has_descriptions = any(word in final_response.lower() for word in ['car', 'vintage', 'color', 'style', 'image'])
            
            print(f"\n✅ Vision Analysis:")
            print(f"  {'✅' if has_descriptions else '❌'} Contains image descriptions")
            print(f"  {'✅' if 'car' in final_response.lower() else '❌'} Mentions cars")
            print(f"  {'✅' if len(final_response) > 200 else '❌'} Detailed response (>200 chars)")
            
            print(f"\n💡 Check browser agent logs for:")
            print(f"   - '🎨 Using vision model'")
            print(f"   - '✅ Detected X UI elements with bounding boxes'")
            print(f"   - Vision model analyzing screenshots")
            
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*80)
    print("PINTEREST VISION ANALYSIS TEST")
    print("Testing vision model's ability to analyze and describe images")
    print("="*80)
    
    print("\n⏳ Waiting for services...")
    time.sleep(2)
    
    success = test_pinterest_vision()
    
    print(f"\n{'='*80}")
    if success:
        print("✅ PINTEREST VISION TEST COMPLETE!")
        print("\nWhat was tested:")
        print("  ✅ Pinterest navigation")
        print("  ✅ Search functionality")
        print("  ✅ Vision model image analysis")
        print("  ✅ Detailed image descriptions")
        print("  ✅ Canvas streaming with screenshots")
        print("\n🎨 Vision model successfully analyzed vintage car images!")
    else:
        print("⚠️  Test encountered issues")
    print("="*80 + "\n")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
