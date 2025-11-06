import requests
import json

def test_wikipedia():
    url = "http://localhost:8070/browse"
    payload = {
        "task": "Go to wikipedia.org, search for 'Python', and tell me the first paragraph",
        "thread_id": "test-wiki",
        "max_steps": 10
    }
    
    print("Testing Wikipedia Search...")
    print("="*80)
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Success: {result['success']}")
            print(f"📝 Summary: {result['task_summary'][:200]}")
            print(f"🎬 Total Actions: {len(result['actions_taken'])}")
            
            print(f"\n📋 Action History:")
            for i, action in enumerate(result['actions_taken'], 1):
                if 'action' in action:
                    print(f"  {i}. {action['action']}: {action.get('reasoning', 'N/A')[:70]}")
            
            return result
        else:
            print(f"❌ Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

if __name__ == "__main__":
    test_wikipedia()
