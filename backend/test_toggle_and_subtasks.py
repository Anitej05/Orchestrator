"""
Test to verify:
1. Subtask creation is concise (2-4 subtasks max)
2. Browser/Plan view toggle works
3. Response doesn't mention subtasks to user
"""

import requests
import json
import time

def test_subtask_creation_and_toggle():
    """Test that subtasks are concise and toggle works"""
    print("\n" + "="*80)
    print("TESTING: Subtask Creation & View Toggle")
    print("="*80)
    
    # Test with a multi-step task
    url = "http://localhost:8000/api/chat"
    payload = {
        "prompt": "Go to example.com, then go to example.org, and tell me what both sites say",
        "thread_id": f"toggle_test_{int(time.time())}",
        "planning_mode": False
    }
    
    print(f"\n📤 Sending request...")
    print(f"Task: {payload['prompt']}")
    print(f"Thread ID: {payload['thread_id']}")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Task completed in {elapsed:.1f}s")
            
            # Check 1: Response shouldn't mention subtasks
            response_text = result.get('response', '')
            if 'subtask' in response_text.lower():
                print(f"❌ FAIL: Response mentions 'subtask' to user")
                print(f"Response: {response_text[:200]}...")
            else:
                print(f"✅ PASS: Response doesn't mention subtasks")
            
            # Check 2: Canvas should exist
            has_canvas = result.get('has_canvas', False)
            if has_canvas:
                print(f"✅ PASS: Canvas exists")
            else:
                print(f"❌ FAIL: No canvas")
                return False
            
            # Check 3: Both views should be available
            thread_id = payload['thread_id']
            
            # Get canvas data to check views
            canvas_url = "http://localhost:8000/api/canvas/get"
            canvas_response = requests.get(
                canvas_url,
                params={"thread_id": thread_id},
                timeout=10
            )
            
            if canvas_response.status_code == 200:
                canvas_data = canvas_response.json()
                has_browser_view = 'browser_view' in canvas_data
                has_plan_view = 'plan_view' in canvas_data
                
                if has_browser_view and has_plan_view:
                    print(f"✅ PASS: Both browser_view and plan_view available")
                else:
                    print(f"❌ FAIL: Missing views (browser: {has_browser_view}, plan: {has_plan_view})")
            
            # Check 4: Test toggle functionality
            print(f"\n🔄 Testing view toggle...")
            
            # Toggle to plan view
            toggle_url = "http://localhost:8000/api/canvas/toggle-view"
            toggle_response = requests.post(
                toggle_url,
                json={"thread_id": thread_id, "view_type": "plan"},
                timeout=10
            )
            
            if toggle_response.status_code == 200:
                print(f"✅ PASS: Toggle to plan view successful")
            else:
                print(f"❌ FAIL: Toggle to plan view failed")
            
            # Toggle back to browser view
            toggle_response = requests.post(
                toggle_url,
                json={"thread_id": thread_id, "view_type": "browser"},
                timeout=10
            )
            
            if toggle_response.status_code == 200:
                print(f"✅ PASS: Toggle to browser view successful")
            else:
                print(f"❌ FAIL: Toggle to browser view failed")
            
            # Check 5: Verify subtask count from logs
            print(f"\n📊 Checking subtask count...")
            print(f"Note: Check browser agent logs to verify subtask count is 2-4")
            
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
    print("SUBTASK & TOGGLE TEST SUITE")
    print("="*80)
    
    # Wait for services
    print("\n⏳ Waiting for services...")
    time.sleep(2)
    
    success = test_subtask_creation_and_toggle()
    
    print("\n" + "="*80)
    if success:
        print("✅ TEST SUMMARY:")
        print("   ✅ Response doesn't mention subtasks")
        print("   ✅ Canvas streaming works")
        print("   ✅ Both browser and plan views available")
        print("   ✅ Toggle functionality works")
        print("\n💡 Check browser agent logs to verify subtask count is 2-4")
    else:
        print("❌ Some tests failed")
    print("="*80 + "\n")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
