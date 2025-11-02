"""
Real endpoint testing for Browser Automation Agent.
This script sends actual HTTP requests to test all endpoints.
"""
import requests
import json
import time

AGENT_URL = "http://localhost:8070"

def print_test(name):
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print('='*70)

def print_result(success, message):
    icon = "✅" if success else "❌"
    print(f"{icon} {message}")

# Test 1: Agent Definition
print_test("GET / - Agent Definition")
try:
    response = requests.get(f"{AGENT_URL}/", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print_result(True, f"Status: {response.status_code}")
        print(f"   Agent ID: {data.get('id')}")
        print(f"   Agent Name: {data.get('name')}")
        print(f"   Capabilities: {len(data.get('capabilities', []))}")
        print(f"   Endpoints: {len(data.get('endpoints', []))}")
    else:
        print_result(False, f"Unexpected status: {response.status_code}")
except Exception as e:
    print_result(False, f"Error: {e}")

# Test 2: Simple Navigation Task
print_test("POST /browse - Simple Navigation")
try:
    payload = {
        "task": "Go to https://example.com and tell me the page title",
        "extract_data": False
    }
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{AGENT_URL}/browse", json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        print_result(result.get('success', False), f"Status: {response.status_code}")
        print(f"   Success: {result.get('success')}")
        print(f"   Task Summary: {result.get('task_summary', '')[:100]}")
        print(f"   Actions Taken: {len(result.get('actions_taken', []))}")
        if result.get('error'):
            print(f"   Error: {result.get('error')}")
    else:
        print_result(False, f"Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
except Exception as e:
    print_result(False, f"Error: {e}")

# Test 3: Data Extraction Task
print_test("POST /browse - Data Extraction")
try:
    payload = {
        "task": "Navigate to https://example.com and extract the main heading text",
        "extract_data": True
    }
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{AGENT_URL}/browse", json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        print_result(result.get('success', False), f"Status: {response.status_code}")
        print(f"   Success: {result.get('success')}")
        print(f"   Task Summary: {result.get('task_summary', '')[:100]}")
        print(f"   Actions Taken: {len(result.get('actions_taken', []))}")
        if result.get('extracted_data'):
            print(f"   Extracted Data: {json.dumps(result.get('extracted_data'), indent=2)[:200]}")
        if result.get('error'):
            print(f"   Error: {result.get('error')}")
    else:
        print_result(False, f"Status: {response.status_code}")
except Exception as e:
    print_result(False, f"Error: {e}")

# Test 4: Screenshot Task
print_test("POST /browse - Screenshot Task")
try:
    payload = {
        "task": "Take a screenshot of https://example.com and describe what you see",
        "extract_data": False
    }
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{AGENT_URL}/browse", json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        print_result(result.get('success', False), f"Status: {response.status_code}")
        print(f"   Success: {result.get('success')}")
        print(f"   Task Summary: {result.get('task_summary', '')[:150]}")
        print(f"   Actions Taken: {len(result.get('actions_taken', []))}")
        if result.get('error'):
            print(f"   Error: {result.get('error')}")
    else:
        print_result(False, f"Status: {response.status_code}")
except Exception as e:
    print_result(False, f"Error: {e}")

# Test 5: Invalid Request (Missing required field)
print_test("POST /browse - Invalid Request (Missing 'task')")
try:
    payload = {
        "extract_data": False
        # Missing 'task' field
    }
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{AGENT_URL}/browse", json=payload, timeout=5)
    
    if response.status_code == 422:
        print_result(True, f"Correctly rejected with status: {response.status_code}")
        print(f"   Validation error as expected")
    else:
        print_result(False, f"Unexpected status: {response.status_code}")
except Exception as e:
    print_result(False, f"Error: {e}")

# Test 6: Complex Multi-Step Task
print_test("POST /browse - Complex Multi-Step Task")
try:
    payload = {
        "task": "Go to https://www.wikipedia.org, find the search box, and tell me what placeholder text it has",
        "extract_data": False
    }
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{AGENT_URL}/browse", json=payload, timeout=90)
    
    if response.status_code == 200:
        result = response.json()
        print_result(result.get('success', False), f"Status: {response.status_code}")
        print(f"   Success: {result.get('success')}")
        print(f"   Task Summary: {result.get('task_summary', '')[:150]}")
        print(f"   Actions Taken: {len(result.get('actions_taken', []))}")
        
        # Show action details
        for i, action in enumerate(result.get('actions_taken', [])[:3]):
            print(f"   Action {i+1}: {action.get('description', '')[:80]}")
        
        if result.get('error'):
            print(f"   Error: {result.get('error')}")
    else:
        print_result(False, f"Status: {response.status_code}")
except Exception as e:
    print_result(False, f"Error: {e}")

print("\n" + "="*70)
print("ALL ENDPOINT TESTS COMPLETED")
print("="*70)
