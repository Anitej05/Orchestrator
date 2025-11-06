import requests
import json

def test_complex_task(task):
    url = "http://localhost:8070/browse"
    payload = {
        "task": task,
        "thread_id": "test-complex",
        "max_steps": 15
    }
    
    print(f"\n{'='*80}")
    print(f"COMPLEX TEST: {task}")
    print(f"{'='*80}\n")
    
    try:
        response = requests.post(url, json=payload, timeout=240)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['success']}")
            print(f"📝 Summary: {result['task_summary']}")
            print(f"🎬 Actions: {len(result['actions_taken'])}")
            
            print(f"\n📋 Action History:")
            for i, action in enumerate(result['actions_taken'][:10], 1):  # Show first 10
                if 'action' in action:
                    print(f"  {i}. {action['action']}: {action.get('reasoning', 'N/A')[:60]}")
            
            if len(result['actions_taken']) > 10:
                print(f"  ... and {len(result['actions_taken']) - 10} more actions")
            
            return result
        else:
            print(f"❌ Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

if __name__ == "__main__":
    # Test: Multi-step task
    test_complex_task(
        "Go to example.com, then navigate to example.org, "
        "and tell me what both pages say"
    )
