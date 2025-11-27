#!/usr/bin/env python3
"""Test Composio connections API"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

def test_connections():
    api_key = os.getenv("COMPOSIO_API_KEY")
    
    print(f"API Key: {api_key[:15]}...")
    print()
    
    headers = {
        "x-api-key": api_key
    }
    
    # Test: List connected accounts
    print("Listing connected accounts...")
    url = "https://backend.composio.dev/api/v3/connectedAccounts"
    try:
        response = httpx.get(url, headers=headers, timeout=10.0)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data}")
            print()
            
            if 'items' in data:
                for account in data['items']:
                    print(f"Account:")
                    print(f"  ID: {account.get('id')}")
                    print(f"  App: {account.get('appName')}")
                    print(f"  Status: {account.get('status')}")
                    print()
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_connections()
