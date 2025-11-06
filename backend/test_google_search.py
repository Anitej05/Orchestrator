import requests
import json

def test_google_search():
    url = "http://localhost:8070/browse"
    payload = {
        "task": "Go to google.com and search for 'Python programming'",
        "thread_id": "test-google",
        "max_steps": 15
    }
    
    print("Testing Google Search with improved DOM extraction...")
    print("="*80)
    
    try:
        response = requests.post(url, json=payload, timeout=180)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Success: {result['success']}")
            print(f"📝 Summary: {result['task_summary']}")
            print(f"🎬 Total Actions: {len(result['actions_taken'])}")
            
            print(f"\n📋 Action History:")
            for i, action in enumerate(result['actions_taken'], 1):
                if 'action' in action:
                    print(f"  {i}. {action['action']}: {action.get('reasoning', 'N/A')[:80]}")
            
            return result
        else:
            print(f"❌ Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

if __name__ == "__main__":
    test_google_search()
