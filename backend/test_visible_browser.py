"""
Test browser agent with visible browser window
"""

import requests
import json

def test_visible_browser():
    """Test that browser opens visibly"""
    print("\n" + "="*80)
    print("TEST: Visible Browser Window")
    print("="*80)
    
    url = "http://localhost:8070/browse"
    
    payload = {
        "task": "Go to example.com and tell me what it says",
        "max_steps": 10
    }
    
    print(f"\n📤 Starting browser task...")
    print(f"Task: {payload['task']}")
    print(f"\n👀 Watch for the browser window to open!")
    print(f"   The browser should be VISIBLE (not headless)")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Task completed successfully")
            print(f"Summary: {result.get('task_summary', 'N/A')[:100]}")
            print(f"Actions: {len(result.get('actions_taken', []))}")
            print(f"Screenshots: {len(result.get('screenshot_files', []))}")
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("VISIBLE BROWSER TEST")
    print("="*80)
    
    success = test_visible_browser()
    
    print("\n" + "="*80)
    if success:
        print("✅ Test completed - Browser should have been visible")
    else:
        print("❌ Test failed")
    print("="*80 + "\n")
