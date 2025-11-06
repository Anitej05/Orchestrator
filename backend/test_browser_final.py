import requests
import json
import time

def test_task(task, expected_success=True):
    url = "http://localhost:8070/browse"
    payload = {"task": task, "thread_id": f"test-{hash(task)}"}
    
    print(f"\n{'='*80}")
    print(f"Task: {task}")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=180)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            success = result['success']
            actions = len(result['actions_taken'])
            
            status = "✅ PASS" if success == expected_success else "❌ FAIL"
            print(f"{status} | Time: {elapsed:.1f}s | Actions: {actions}")
            print(f"Summary: {result['task_summary'][:100]}...")
            
            return success == expected_success
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception: {str(e)[:100]}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("BROWSER AGENT COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Go to example.com and tell me what it says", True),
        ("Visit example.org and extract the main text", True),
        ("Navigate to httpbin.org/html and get the page title", True),
        ("Go to example.com, then example.org, and compare them", True),
    ]
    
    passed = 0
    total = len(tests)
    
    for task, expected in tests:
        if test_task(task, expected):
            passed += 1
        time.sleep(2)
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {passed}/{total} tests passed ({100*passed//total}%)")
    print(f"{'='*80}\n")
