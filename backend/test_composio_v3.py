#!/usr/bin/env python3
"""Test Composio v3 API endpoints"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

def test_v3_api():
    api_key = os.getenv("COMPOSIO_API_KEY")
    connection_id = os.getenv("GMAIL_CONNECTION_ID")
    
    print(f"API Key: {api_key[:15]}...")
    print(f"Connection ID: {connection_id}")
    print()
    
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Test 1: List actions
    print("Test 1: List Gmail actions")
    url = "https://backend.composio.dev/api/v3/actions?appNames=gmail"
    try:
        response = httpx.get(url, headers=headers, timeout=10.0)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data.get('items', []))} actions")
            for action in data.get('items', [])[:3]:
                print(f"  - {action.get('name')}")
        else:
            print(f"Error: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    
    # Test 2: Execute action with POST
    print("Test 2: Execute GMAIL_FETCH_EMAILS (POST)")
    url = f"https://backend.composio.dev/api/v3/actions/GMAIL_FETCH_EMAILS/execute"
    payload = {
        "connectedAccountId": connection_id,
        "input": {
            "query": "is:unread",
            "max_results": 2
        }
    }
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:300]}")
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    
    # Test 3: Try different endpoint format
    print("Test 3: Try v2 endpoint")
    url = "https://backend.composio.dev/api/v2/actions/GMAIL_FETCH_EMAILS/execute"
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:300]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_v3_api()
