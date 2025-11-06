import requests
import json
import time

def test_task(task, description):
    url = "http://localhost:8070/browse"
    payload = {"task": task, "thread_id": f"test-{hash(task)}", "max_steps": 15}
    
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"Task: {task}")
    print(f"{'='*80}")
    
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=180)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['success']} | Time: {elapsed:.1f}s | Actions: {len(result['actions_taken'])}")
            print(f"📝 {result['task_summary'][:150]}")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception: {str(e)[:100]}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE DOM AWARENESS TEST SUITE")
    print("="*80)
    
    tests = [
        ("Go to google.com and search for 'artificial intelligence'", 
         "Google Search - Complex Input Detection"),
        
        ("Visit github.com/trending and extract the first trending repository name",
         "GitHub - Dynamic Content Extraction"),
        
        ("Go to wikipedia.org, search for 'Python', and tell me the first paragraph",
         "Wikipedia - Multi-step with Content Extraction"),
    ]
    
    passed = 0
    for task, desc in tests:
        if test_task(task, desc):
            passed += 1
        time.sleep(3)
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print(f"{'='*80}\n")
