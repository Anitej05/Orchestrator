import requests
import json
import time

def test_browser_agent(task, thread_id="test-001"):
    url = "http://localhost:8070/browse"
    payload = {
        "task": task,
        "thread_id": thread_id
    }
    
    print(f"\n{'='*60}")
    print(f"Testing: {task}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nResult: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

if __name__ == "__main__":
    # Test 1: Simple Google search
    test_browser_agent("Go to google.com and search for 'artificial intelligence'")
    
    time.sleep(2)
    
    # Test 2: Navigate and extract
    test_browser_agent("Go to example.com and tell me what the page says")
