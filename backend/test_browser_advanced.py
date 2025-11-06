import requests
import json
import time

def test_browser_agent(task, thread_id="test-advanced"):
    url = "http://localhost:8070/browse"
    payload = {
        "task": task,
        "thread_id": thread_id
    }
    
    print(f"\n{'='*80}")
    print(f"Testing: {task}")
    print(f"{'='*80}")
    
    try:
        response = requests.post(url, json=payload, timeout=180)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nSuccess: {result['success']}")
            print(f"Summary: {result['task_summary']}")
            print(f"Actions taken: {len(result['actions_taken'])}")
            print(f"Screenshots: {len(result['screenshot_files']) if result['screenshot_files'] else 0}")
            return result
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

if __name__ == "__main__":
    # Test 1: Simple navigation and extraction
    test_browser_agent("Go to example.org and tell me what it says")
    time.sleep(2)
    
    # Test 2: Wikipedia search
    test_browser_agent("Go to wikipedia.org and search for Python programming language")
    time.sleep(2)
    
    # Test 3: Multiple steps
    test_browser_agent("Visit github.com and find the trending repositories")
    time.sleep(2)
    
    # Test 4: Data extraction
    test_browser_agent("Go to httpbin.org/html and extract the main heading")
