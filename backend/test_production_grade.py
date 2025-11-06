import requests
import json
import time

def test_task(task, description, expected_success=True):
    url = "http://localhost:8070/browse"
    payload = {"task": task, "thread_id": f"test-{hash(task)}", "max_steps": 15}
    
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=180)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            success = result['success']
            actions = len(result['actions_taken'])
            
            # Check for failures in actions
            failed_actions = [a for a in result['actions_taken'] if a.get('status') == 'failed']
            
            status = "✅ PASS" if success == expected_success else "❌ FAIL"
            print(f"{status} | Time: {elapsed:.1f}s | Actions: {actions} | Failed: {len(failed_actions)}")
            print(f"Summary: {result['task_summary'][:120]}")
            
            if failed_actions:
                print(f"⚠️  Failed actions: {len(failed_actions)}")
                for fa in failed_actions[:3]:
                    print(f"   - {fa.get('action')}: {fa.get('error', 'Unknown')[:60]}")
            
            return success == expected_success and len(failed_actions) == 0
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception: {str(e)[:100]}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PRODUCTION-GRADE BROWSER AGENT TEST SUITE")
    print("="*80)
    
    tests = [
        # Basic navigation
        ("Go to example.com and tell me what it says", 
         "Basic Navigation & Extraction"),
        
        # Search functionality
        ("Go to google.com and search for 'machine learning'", 
         "Google Search - Complex DOM"),
        
        ("Go to wikipedia.org and search for 'Artificial Intelligence'",
         "Wikipedia Search - Auto-submit"),
        
        # Multi-step tasks
        ("Visit github.com/trending and tell me the top 3 trending repositories",
         "GitHub Trending - Dynamic Content"),
        
        # Data extraction
        ("Go to httpbin.org/html and extract all the headings",
         "Data Extraction - Structured Content"),
        
        # Complex navigation
        ("Go to example.com, then navigate to example.org, and compare their content",
         "Multi-site Navigation & Comparison"),
    ]
    
    passed = 0
    total = len(tests)
    
    for task, desc in tests:
        if test_task(task, desc):
            passed += 1
        time.sleep(3)  # Cooldown between tests
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS: {passed}/{total} tests passed ({100*passed//total}%)")
    print(f"{'='*80}\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Production Ready!")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed - Minor issues remain")
    else:
        print("❌ Multiple failures - Needs more work")
