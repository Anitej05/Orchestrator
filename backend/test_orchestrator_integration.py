"""
Integration test simulating how the orchestrator calls the browser agent.
This mimics the exact flow from your orchestrator logs.
"""
import requests
import json

AGENT_URL = "http://localhost:8070"

def test_orchestrator_flow():
    """
    Simulates the exact scenario from your logs:
    'Open a browser and take a screenshot of today's doodle'
    """
    print("="*70)
    print("ORCHESTRATOR INTEGRATION TEST")
    print("="*70)
    print("\nSimulating orchestrator calling browser agent...")
    print("Task: 'Open a browser and take a screenshot of today's doodle'\n")
    
    # Step 1: Get agent definition (orchestrator does this during agent search)
    print("Step 1: Fetching agent definition...")
    try:
        response = requests.get(f"{AGENT_URL}/", timeout=5)
        if response.status_code == 200:
            agent_def = response.json()
            print(f"✅ Agent found: {agent_def['name']}")
            print(f"   Capabilities: {', '.join(agent_def['capabilities'][:3])}...")
        else:
            print(f"❌ Failed to get agent definition: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Step 2: Call the browse endpoint (orchestrator does this during execution)
    print("\nStep 2: Executing browser task...")
    payload = {
        "task": "Navigate to https://www.google.com and take a screenshot of the page, describe what you see",
        "extract_data": False
    }
    
    print(f"Request payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        print("\n⏳ Sending request to browser agent...")
        response = requests.post(
            f"{AGENT_URL}/browse",
            json=payload,
            timeout=90  # Browser tasks can take time
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Response received!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Success: {result.get('success')}")
            print(f"   Task Summary: {result.get('task_summary', '')[:120]}...")
            print(f"   Actions Taken: {len(result.get('actions_taken', []))}")
            
            # Show action details
            if result.get('actions_taken'):
                print(f"\n   Action Details:")
                for i, action in enumerate(result.get('actions_taken', [])[:5]):
                    desc = action.get('description', '')[:100]
                    print(f"     {i+1}. {desc}")
            
            if result.get('error'):
                print(f"\n❌ Error in response: {result.get('error')}")
                return False
            
            if result.get('success'):
                print(f"\n🎉 SUCCESS! Browser agent completed the task.")
                print(f"\nThis is exactly what the orchestrator would receive.")
                return True
            else:
                print(f"\n❌ Task reported as failed")
                return False
                
        else:
            print(f"\n❌ Request failed with status: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"\n❌ Request timed out (browser tasks can take time)")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_orchestrator_flow()
    
    print("\n" + "="*70)
    if success:
        print("✅ INTEGRATION TEST PASSED")
        print("\nThe browser agent is working correctly with the orchestrator!")
        print("The 'ChatOpenAI' provider error has been fixed.")
    else:
        print("❌ INTEGRATION TEST FAILED")
        print("\nPlease check the error messages above.")
    print("="*70)
