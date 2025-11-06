"""
Test that orchestrator properly handles browser agent responses without pausing
"""

import requests
import json
import time

def test_browser_agent_no_pause():
    """Test that browser agent completes without unnecessary pauses"""
    print("\n" + "="*80)
    print("TEST: Browser Agent - No Unnecessary Pauses")
    print("="*80)
    
    url = "http://localhost:8000/api/chat"
    
    # Simple task that should complete without asking for confirmation
    payload = {
        "prompt": "Use the browser to go to example.com and tell me what it says",
        "thread_id": f"test_no_pause_{int(time.time())}",
        "planning_mode": False
    }
    
    print(f"\n📤 Sending request...")
    print(f"Prompt: {payload['prompt']}")
    print(f"Thread ID: {payload['thread_id']}")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if it paused for user input
            paused = result.get('pending_user_input', False)
            question = result.get('question_for_user')
            
            print(f"\n📊 Results:")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Status: {response.status_code}")
            print(f"   Paused for input: {paused}")
            
            if paused:
                print(f"   ❌ FAIL: Orchestrator paused unnecessarily")
                print(f"   Question asked: {question}")
                return False
            else:
                print(f"   ✅ PASS: Completed without pausing")
                
                # Check if we got a response
                if 'response' in result or 'final_response' in result:
                    response_text = result.get('response') or result.get('final_response', '')
                    print(f"   Response preview: {response_text[:100]}...")
                
                return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*80)
    print("ORCHESTRATOR FIX VERIFICATION")
    print("="*80)
    
    # Wait for services to be ready
    print("\n⏳ Waiting for services to start...")
    time.sleep(3)
    
    # Run test
    success = test_browser_agent_no_pause()
    
    print("\n" + "="*80)
    if success:
        print("✅ FIX VERIFIED: Orchestrator no longer pauses unnecessarily")
    else:
        print("❌ FIX FAILED: Orchestrator still pausing")
    print("="*80 + "\n")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
