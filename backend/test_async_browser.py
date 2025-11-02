"""
Test the async browser agent pattern.
"""
import requests
import json
import time

AGENT_URL = "http://localhost:8070"

print("="*80)
print("TESTING ASYNC BROWSER AGENT PATTERN")
print("="*80)

# Step 1: Submit task
print("\nStep 1: Submitting task...")
payload = {
    "task": "Go to https://github.com/browser-use/browser-use and tell me the star count",
    "extract_data": False
}

try:
    response = requests.post(f"{AGENT_URL}/browse/async", json=payload, timeout=10)
    
    if response.status_code == 200:
        submit_data = response.json()
        task_id = submit_data.get('task_id')
        
        print(f"✅ Task submitted successfully!")
        print(f"   Task ID: {task_id}")
        print(f"   Status: {submit_data.get('status')}")
        print(f"   Message: {submit_data.get('message')}")
        
        # Step 2: Poll for completion
        print(f"\nStep 2: Polling for completion...")
        print("   (This will wait as long as needed - no timeout!)")
        
        poll_count = 0
        while True:
            time.sleep(2)  # Poll every 2 seconds
            poll_count += 1
            
            status_response = requests.get(f"{AGENT_URL}/browse/status/{task_id}", timeout=10)
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get('status')
                progress = status_data.get('progress', '')
                
                print(f"   Poll #{poll_count}: Status = {status} | {progress}")
                
                if status == 'completed':
                    result = status_data.get('result')
                    print(f"\n✅ Task completed successfully!")
                    print(f"\n{'='*80}")
                    print("RESULT")
                    print('='*80)
                    print(f"Success: {result.get('success')}")
                    print(f"Task Summary: {result.get('task_summary')}")
                    print(f"Actions Taken: {len(result.get('actions_taken', []))}")
                    print('='*80)
                    break
                
                elif status == 'failed':
                    result = status_data.get('result')
                    print(f"\n❌ Task failed!")
                    print(f"Error: {result.get('error')}")
                    break
                
                # Continue polling if pending or processing
            else:
                print(f"   ❌ Status check failed: {status_response.status_code}")
                break
    else:
        print(f"❌ Task submission failed: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*80)
print("TEST COMPLETED")
print("="*80)
